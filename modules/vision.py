"""
modules/vision.py  — Phase 5 版本

新增改动（Phase 5 主动播报）：
1. _announce_cooldown 字典：记录每个职业标签上次播报时间，实现防抖
2. _check_and_announce()：对比新旧状态表，发现新职业人员时触发 TTS
3. _recognition_loop 末尾调用 _check_and_announce()
4. 对外暴露 set_auto_announce(enabled) 供前端运行时切换开关

其余逻辑不变。
"""

import base64
import time
import threading
import concurrent.futures
import cv2
from openai import OpenAI
from ultralytics import YOLO
import config

model = YOLO('yolov8n.pt')

client = OpenAI(
    api_key=config.SILICONFLOW_API_KEY,
    base_url=config.SILICONFLOW_BASE_URL
)

_llm_executor = concurrent.futures.ThreadPoolExecutor(max_workers=6)

# ── 识别状态表 ──────────────────────────────────────────────────────────────
_state_table: list = []
_state_lock = threading.Lock()

# 控制信号：外部请求立即触发一轮识别
_immediate_flag = threading.Event()


def get_state_table():
    with _state_lock:
        return [
            {k: v for k, v in e.items() if k != "pending"}
            for e in _state_table
        ]


def request_immediate_recognition():
    _immediate_flag.set()


def _update_one(entry_id: int, occupation: str):
    with _state_lock:
        for e in _state_table:
            if e["id"] == entry_id:
                e["occupation"] = occupation
                e["updated_at"] = time.time()
                e["pending"] = False
                break


def _replace_state(new_entries: list):
    with _state_lock:
        global _state_table
        _state_table = new_entries


# ── 工具函数 ────────────────────────────────────────────────────────────────
def detect_persons(frame):
    results = model(frame, verbose=False)[0]
    boxes = []
    for box in results.boxes:
        if int(box.cls[0]) == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2))
    return boxes


def get_position_label(bbox, frame_width):
    cx = (bbox[0] + bbox[2]) / 2
    ratio = cx / frame_width
    if ratio < 0.33:
        return "左侧"
    elif ratio < 0.66:
        return "中间"
    else:
        return "右侧"


def crop_person(frame, box):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, x1 - 10)
    y1 = max(0, y1 - 10)
    x2 = min(w, x2 + 10)
    y2 = min(h, y2 + 10)
    return frame[y1:y2, x1:x2]


def frame_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buffer).decode('utf-8')


def calc_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a + area_b - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def match_box_to_state(box, state_table, iou_threshold=0.3):
    best_match = None
    best_iou = iou_threshold
    for entry in state_table:
        iou = calc_iou(box, entry["bbox"])
        if iou > best_iou:
            best_iou = iou
            best_match = entry
    return best_match


def is_entry_stale(entry) -> bool:
    age = time.time() - entry.get("updated_at", 0)
    occ = entry.get("occupation", "")
    if occ in ("路人", "未知", "识别超时"):
        return age > getattr(config, "UNCERTAIN_RECHECK_AGE", 4.0)
    return age > getattr(config, "LABEL_REUSE_MAX_AGE", 4.0)


# ── LLM 识别（单人，带超时）────────────────────────────────────────────────
def recognize_occupation(person_img):
    b64 = frame_to_base64(person_img)

    def _call():
        response = client.chat.completions.create(
            model=config.VISION_MODEL,
            max_tokens=30,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                    },
                    {
                        "type": "text",
                        "text": (
                            "请判断图中人物的职业身份。"
                            "只输出职业名称，不超过4个字，不要任何解释。"
                            "如果无法判断则输出'路人'。"
                        )
                    }
                ]
            }]
        )
        return response.choices[0].message.content.strip()

    try:
        future = _llm_executor.submit(_call)
        return future.result(timeout=config.LLM_TIMEOUT)
    except concurrent.futures.TimeoutError:
        print(f"  [LLM] 识别超时（>{config.LLM_TIMEOUT}s），跳过")
        return None
    except Exception as e:
        print(f"  [LLM] 调用失败: {e}")
        return None


# ── 核心：逐人异步识别 ──────────────────────────────────────────────────────
def _recognize_all_async(frame, boxes, entry_ids):
    frame_width = frame.shape[1]

    def _task(entry_id, box):
        crop = crop_person(frame, box)
        occupation = recognize_occupation(crop)

        if occupation is None:
            with _state_lock:
                for e in _state_table:
                    if e["id"] == entry_id:
                        if e["occupation"] in ("识别中...", ""):
                            e["occupation"] = "未知"
                        e["pending"] = False
                        e["updated_at"] = time.time()
                        break
            print(f"  [LLM] 人物{entry_id} 超时，保留旧标签")
        else:
            _update_one(entry_id, occupation)
            position = get_position_label(box, frame_width)
            print(f"  [LLM] 人物{entry_id}（{position}）→ {occupation}")

    futures = []
    for entry_id, box in zip(entry_ids, boxes):
        f = _llm_executor.submit(_task, entry_id, box)
        futures.append(f)
    return futures


# ── Phase 5：主动播报 ────────────────────────────────────────────────────────

# 运行时开关，可通过 set_auto_announce() 动态切换
_auto_announce_enabled: bool = getattr(config, "AUTO_ANNOUNCE", True)
_announce_lock = threading.Lock()

# 防抖字典：occupation_key → 上次播报时间戳
# key 格式："{position}_{occupation}"，按位置+职业去重，避免同一位置同一职业反复播报
_announce_cooldown: dict = {}


# 主动播报的外部回调（用于 SSE 推送到前端）
# 签名：callback(text: str) → None
_on_announce_callback = None


def set_announce_callback(callback):
    """注册主动播报回调，每次播报触发时调用 callback(text)。"""
    global _on_announce_callback
    _on_announce_callback = callback


def set_auto_announce(enabled: bool):
    """运行时切换主动播报开关（供前端 API 调用）。"""
    global _auto_announce_enabled
    with _announce_lock:
        _auto_announce_enabled = enabled
    print(f"[主动播报] {'启用' if enabled else '停用'}")


def _is_valid_occupation(occ: str) -> bool:
    """判断职业标签是否值得播报（过滤无效标签）。"""
    skip = getattr(config, "ANNOUNCE_SKIP_OCCUPATIONS",
                   {"路人", "未知", "识别中...", "识别超时", ""})
    return occ not in skip


def _build_announce_key(entry: dict) -> str:
    """用位置+职业作为防抖 key，同位置同职业 30s 内不重播。"""
    return f"{entry.get('position', '')}_{entry.get('occupation', '')}"


def _check_and_announce(prev_state: list, new_state: list):
    """
    对比前后两轮状态表，发现新出现的有效职业人员时触发 TTS 主动播报。

    判断"新人"的规则：
      - new_state 里的某个条目，在 prev_state 里找不到 IoU > 0.3 的匹配
      - 该条目的职业标签是有效的（非路人/未知/识别中）
      - 该 (position, occupation) 组合在 ANNOUNCE_COOLDOWN 秒内未播报过
    """
    with _announce_lock:
        enabled = _auto_announce_enabled
    if not enabled:
        return

    # 延迟导入，避免循环依赖（tts → asr → vision 可能形成环）
    from modules.tts import speak

    cooldown = getattr(config, "ANNOUNCE_COOLDOWN", 30.0)
    now = time.time()
    new_persons_to_announce = []

    for entry in new_state:
        occ = entry.get("occupation", "")
        if not _is_valid_occupation(occ):
            continue

        # 在旧状态里查找 IoU 匹配
        matched_in_prev = match_box_to_state(entry["bbox"], prev_state)
        if matched_in_prev:
            # 旧状态里有这个人 → 不是新人，跳过
            continue

        # 是新人，检查防抖
        key = _build_announce_key(entry)
        last_time = _announce_cooldown.get(key, 0)
        if now - last_time < cooldown:
            print(f"[主动播报] {key} 冷却中，跳过（距上次 {now - last_time:.0f}s）")
            continue

        # 通过所有检查，加入待播报列表
        new_persons_to_announce.append(entry)
        _announce_cooldown[key] = now

    if not new_persons_to_announce:
        return

    # 构建播报文本
    # 多人同时出现：合并成一句话；单人：直接说
    if len(new_persons_to_announce) == 1:
        e = new_persons_to_announce[0]
        text = f"您{e['position']}出现了一位{e['occupation']}"
    else:
        parts = []
        for e in new_persons_to_announce:
            parts.append(f"{e['position']}一位{e['occupation']}")
        text = "您附近出现了" + "，".join(parts)

    print(f"[主动播报] 触发：{text}")
    speak(text)
    if _on_announce_callback:
        _on_announce_callback(text)


# ── 后台识别线程 ────────────────────────────────────────────────────────────
def _recognition_loop(camera, interval):
    """
    识别循环（Phase 5 bugfix）：

    关键修正：
      prev_state 必须保存「上一轮循环结束时」的状态，
      而不是「本轮开始时」的状态——两者在同一帧内取值完全相同，
      导致新人永远在 prev_state 中能找到匹配，主动播报永不触发。

    正确做法：在循环顶部用 prev_state 保存上一轮留下的结果，
    在循环底部更新 prev_state 供下一轮使用。
    """
    # 上一轮识别完成后的状态，初始为空（程序刚启动时画面里没有已知人员）
    prev_state: list = []

    while True:
        triggered = _immediate_flag.wait(timeout=interval)
        if triggered:
            _immediate_flag.clear()
            print("[识别线程] 收到即时识别请求")

        frame = camera.get_frame()
        if frame is None:
            continue

        frame_width = frame.shape[1]
        boxes = detect_persons(frame)

        if not boxes:
            _replace_state([])
            prev_state = []   # 画面清空，重置快照
            continue

        # old_state：用于标签复用（判断是否需要重新识别），取当前状态表
        old_state = get_state_table()
        new_entries = []
        boxes_to_recognize = []
        ids_to_recognize = []

        for i, box in enumerate(boxes):
            entry_id = i + 1
            position = get_position_label(box, frame_width)
            matched = match_box_to_state(box, old_state)

            if matched and not is_entry_stale(matched) and matched["occupation"] not in ("未知", "识别中...", ""):
                new_entries.append({
                    "id":         entry_id,
                    "position":   position,
                    "occupation": matched["occupation"],
                    "bbox":       box,
                    "updated_at": matched.get("updated_at", time.time()),
                    "pending":    False,
                })
            else:
                old_occ = matched["occupation"] if matched else ""
                placeholder = old_occ if (old_occ and old_occ not in ("未知", "识别中...")) else "识别中..."
                new_entries.append({
                    "id":         entry_id,
                    "position":   position,
                    "occupation": placeholder,
                    "bbox":       box,
                    "updated_at": time.time(),
                    "pending":    True,
                })
                boxes_to_recognize.append(box)
                ids_to_recognize.append(entry_id)

        _replace_state(new_entries)

        if boxes_to_recognize:
            print(f"[识别线程] 共 {len(boxes)} 人，"
                  f"其中 {len(boxes_to_recognize)} 人需要重新识别")
            futures = _recognize_all_async(frame, boxes_to_recognize, ids_to_recognize)

            # 等待本轮所有识别完成，确保播报的是最终标签而非"识别中..."
            # 超时保护：最多等 LLM_TIMEOUT + 1 秒
            wait_deadline = time.time() + config.LLM_TIMEOUT + 1.0
            for f in futures:
                remaining = wait_deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    f.result(timeout=max(0.1, remaining))
                except Exception:
                    pass
        else:
            print(f"[识别线程] 共 {len(boxes)} 人，全部复用旧标签")

        # ── Phase 5：主动播报差异检查 ────────────────────────────────────────
        # final_state 是本轮识别全部落定后的结果
        # prev_state  是上一轮循环留下的结果（真正的"上一帧"快照）
        final_state = get_state_table()
        _check_and_announce(prev_state, final_state)

        # 本轮结束，将 final_state 保存为下一轮的 prev_state
        prev_state = final_state


def start_recognition_thread(camera):
    interval = config.RECOGNITION_INTERVAL
    t = threading.Thread(
        target=_recognition_loop,
        args=(camera, interval),
        daemon=True
    )
    t.start()
    print(f"[识别线程] 已启动，间隔 {interval}s")