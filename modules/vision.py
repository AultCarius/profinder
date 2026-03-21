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

_llm_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# ── 识别状态表 ──────────────────────────────────────────
_state_table = []
_state_lock  = threading.Lock()

def get_state_table():
    with _state_lock:
        return list(_state_table)

def _update_state(new_state):
    with _state_lock:
        global _state_table
        _state_table = new_state

# ── 工具函数 ────────────────────────────────────────────
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
    best_iou   = iou_threshold
    for entry in state_table:
        iou = calc_iou(box, entry["bbox"])
        if iou > best_iou:
            best_iou   = iou
            best_match = entry
    return best_match

# ── LLM 识别（带超时）──────────────────────────────────
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
        return "识别超时"
    except Exception as e:
        print(f"  [LLM] 调用失败: {e}")
        return "未知"

# ── 后台识别线程 ────────────────────────────────────────
_is_recognizing = False

def _recognition_loop(camera, interval):
    global _is_recognizing
    while True:
        time.sleep(interval)

        if _is_recognizing:
            print(f"[识别线程] 上次识别尚未完成，跳过本次")
            continue

        _is_recognizing = True
        t_start = time.time()
        try:
            frame = camera.get_frame()
            if frame is None:
                continue

            frame_width = frame.shape[1]
            boxes = detect_persons(frame)

            if not boxes:
                _update_state([])
                print("[识别线程] 画面中无人，状态表已清空")
                continue

            # 并行识别所有人物
            crops = [crop_person(frame, box) for box in boxes]
            futures = {
                _llm_executor.submit(recognize_occupation, crop): i
                for i, crop in enumerate(crops)
            }

            results = ["未知"] * len(boxes)
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                try:
                    results[i] = future.result()
                except Exception:
                    results[i] = "未知"

            new_state = []
            for i, (box, occupation) in enumerate(zip(boxes, results)):
                position = get_position_label(box, frame_width)
                new_state.append({
                    "id":         i + 1,
                    "position":   position,
                    "occupation": occupation,
                    "bbox":       box
                })
                print(f"  [LLM] 人物{i+1}（{position}）→ {occupation}")

            _update_state(new_state)
            total_cost = time.time() - t_start
            print(f"[识别线程] 本轮完成，共 {len(boxes)} 人，"
                  f"总耗时 {total_cost:.1f}s，建议间隔 > {total_cost:.1f}s")

        finally:
            _is_recognizing = False

def start_recognition_thread(camera):
    interval = config.RECOGNITION_INTERVAL
    t = threading.Thread(
        target=_recognition_loop,
        args=(camera, interval),
        daemon=True
    )
    t.start()
    print(f"[识别线程] 已启动，间隔 {interval}s")