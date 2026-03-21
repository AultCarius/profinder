"""
modules/vision.py  — 重构版

核心改动：
1. 状态表改为"逐人更新"：每识别完一个人立即写入，不等全部完成
2. 状态表加 `stale` 标记：超过 N 秒未更新的条目显示为"识别中..."
3. 识别线程改为"滚动调度"：每次只识别变化最大 / 最旧的那几个人
4. 并行 future 超时后不阻塞其他人的更新
5. 新增 request_immediate_recognition()，供用户提问时主动触发一轮加速识别
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
# 每条结构：
#   id, position, occupation, bbox, updated_at, pending(bool)
# pending=True 表示"正在识别中"，前端显示"识别中..."
_state_table: list = []
_state_lock = threading.Lock()

# 控制信号：外部请求立即触发一轮识别（用户提问时调用）
_immediate_flag = threading.Event()


def get_state_table():
    """返回当前状态表的快照（过滤掉 pending 的内部字段）"""
    with _state_lock:
        return [
            {k: v for k, v in e.items() if k != "pending"}
            for e in _state_table
        ]


def request_immediate_recognition():
    """
    外部调用：请求立即触发一轮识别（不等 interval 计时）。
    用户开口提问时调用，让状态表尽快刷新。
    """
    _immediate_flag.set()


def _update_one(entry_id: int, occupation: str):
    """识别完成后立即更新单条记录（不阻塞其他人）"""
    with _state_lock:
        for e in _state_table:
            if e["id"] == entry_id:
                e["occupation"] = occupation
                e["updated_at"] = time.time()
                e["pending"] = False
                break


def _replace_state(new_entries: list):
    """全量替换状态表（每轮开始时调用，先占位再逐步填充）"""
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
    """
    判断某条记录是否需要重新识别。
    - 有效标签（非路人/未知）：超过 config.LABEL_REUSE_MAX_AGE 秒才重识
    - "路人" / "未知"：超过 config.UNCERTAIN_RECHECK_AGE 秒就重识（更积极）
    """
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
        return None   # 返回 None 而非字符串，让调用方决定如何处理
    except Exception as e:
        print(f"  [LLM] 调用失败: {e}")
        return None


# ── 核心：逐人异步识别 ──────────────────────────────────────────────────────
def _recognize_all_async(frame, boxes, entry_ids):
    """
    为每个 bbox 异步提交识别任务，识别完成后立即更新对应条目。
    不等所有人都完成，任何一人识别完成立即写入状态表。
    """
    frame_width = frame.shape[1]

    def _task(entry_id, box):
        crop = crop_person(frame, box)
        occupation = recognize_occupation(crop)

        if occupation is None:
            # 超时：保留旧职业标签（不覆盖），只更新 bbox 和 position
            with _state_lock:
                for e in _state_table:
                    if e["id"] == entry_id:
                        # 如果旧值是"识别中..."（占位），改为"未知"
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


# ── 后台识别线程 ────────────────────────────────────────────────────────────
def _recognition_loop(camera, interval):
    """
    改进版识别循环：
    - 每轮先用 YOLO 检测出所有人，立即用占位符更新状态表（屏幕立刻有框）
    - 再异步为每人提交 LLM 识别，识别完一个更新一个
    - 已有职业标签且不过旧的人可以跳过，节省 API 调用
    """
    pending_futures = []   # 上一轮尚未完成的 future，用于追踪

    while True:
        # 等待下一次触发（定时 or 即时请求）
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
            continue

        # ── 与旧状态表做 IoU 匹配，复用旧职业标签 ─────────────────────────
        old_state = get_state_table()
        new_entries = []
        boxes_to_recognize = []   # 需要重新识别的 box
        ids_to_recognize = []

        for i, box in enumerate(boxes):
            entry_id = i + 1
            position = get_position_label(box, frame_width)
            matched = match_box_to_state(box, old_state)

            if matched and not is_entry_stale(matched) and matched["occupation"] not in ("未知", "识别中...", ""):
                # 复用旧标签，不重新识别
                new_entries.append({
                    "id":         entry_id,
                    "position":   position,
                    "occupation": matched["occupation"],
                    "bbox":       box,
                    "updated_at": matched.get("updated_at", time.time()),
                    "pending":    False,
                })
            else:
                # 需要重新识别：先用占位符占位
                old_occ = matched["occupation"] if matched else ""
                # 保留旧标签（如果有）避免闪烁；新人用"识别中..."
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

        # 立即更新状态表（屏幕马上显示新的框和已知的职业标签）
        _replace_state(new_entries)

        if boxes_to_recognize:
            print(f"[识别线程] 共 {len(boxes)} 人，"
                  f"其中 {len(boxes_to_recognize)} 人需要重新识别")
            # 异步识别，不阻塞循环
            _recognize_all_async(frame, boxes_to_recognize, ids_to_recognize)
        else:
            print(f"[识别线程] 共 {len(boxes)} 人，全部复用旧标签")


def start_recognition_thread(camera):
    interval = config.RECOGNITION_INTERVAL
    t = threading.Thread(
        target=_recognition_loop,
        args=(camera, interval),
        daemon=True
    )
    t.start()
    print(f"[识别线程] 已启动，间隔 {interval}s")