import queue
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from flask import Flask, Response, render_template
from modules.camera import CameraStream
from modules.vision import get_state_table, start_recognition_thread
from modules.asr import set_asr_callback
import config

app = Flask(__name__)
camera = CameraStream(
    index=config.CAMERA_INDEX,
    width=config.FRAME_WIDTH,
    height=config.FRAME_HEIGHT,
)

# 启动后台识别线程
start_recognition_thread(camera)

FONT = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 24)

# ── ASR 结果队列（SSE 消费）────────────────────────────────────────────────
# 每个 SSE 客户端独占一个队列；用全局列表管理所有已连接客户端
_asr_clients: list[queue.Queue] = []
_asr_clients_lock = __import__('threading').Lock()


def _asr_result_handler(text: str):
    """ASR 回调：把转写文本广播给所有 SSE 客户端。"""
    with _asr_clients_lock:
        for q in _asr_clients:
            q.put(text)


# 在 asr 模块注册回调
set_asr_callback(_asr_result_handler)


# ── 视频帧渲染 ──────────────────────────────────────────────────────────────
def draw_chinese(frame, text, x, y, bg_color=(0, 255, 0), text_color=(0, 0, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    bbox    = draw.textbbox((0, 0), text, font=FONT)
    text_w  = bbox[2] - bbox[0]
    text_h  = bbox[3] - bbox[1]
    draw.rectangle([x, y - text_h - 8, x + text_w + 8, y], fill=bg_color)
    draw.text((x + 4, y - text_h - 4), text, font=FONT, fill=text_color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def generate_frames():
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue

        # 只读状态表，不在此处跑 YOLO（YOLO 在后台识别线程里跑）
        state = get_state_table()

        for entry in state:
            x1, y1, x2, y2 = entry["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{entry['position']} · {entry['occupation']}"
            frame = draw_chinese(frame, label, x1, y1)

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + buffer.tobytes()
            + b'\r\n'
        )


# ── SSE：ASR 转写结果推送 ────────────────────────────────────────────────────
def _sse_stream(client_queue: queue.Queue):
    """生成 SSE 事件流，阻塞等待 ASR 结果，逐条推送。"""
    try:
        while True:
            text = client_queue.get()          # 阻塞直到有新结果
            # SSE 格式：data: ...\n\n
            yield f"data: {text}\n\n"
    except GeneratorExit:
        pass
    finally:
        with _asr_clients_lock:
            _asr_clients.remove(client_queue)


# ── Flask 路由 ───────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )


@app.route('/asr_stream')
def asr_stream():
    """SSE 接口：客户端连接后，每次 ASR 完成即推送转写文本。"""
    q = queue.Queue()
    with _asr_clients_lock:
        _asr_clients.append(q)
    return Response(
        _sse_stream(q),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',   # 兼容 Nginx 反代时关闭缓冲
        },
    )