"""
ui/app.py

Flask 路由：
  /              — 页面
  /video_feed    — MJPEG 视频流
  /asr_stream    — SSE：ASR 转写结果推送
  /chat_stream   — SSE：对话记录推送（问题 + 回答）
"""

import json
import queue
import threading

import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from flask import Flask, Response, render_template

from modules.camera import CameraStream
from modules.vision import detect_persons, get_state_table, start_recognition_thread, match_box_to_state
from modules.asr import set_asr_callback
from modules.llm import answer_question
import config

app = Flask(__name__)

camera = CameraStream(
    index=config.CAMERA_INDEX,
    width=config.FRAME_WIDTH,
    height=config.FRAME_HEIGHT,
)
start_recognition_thread(camera)

FONT = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 24)


# ── ASR 结果广播 ──────────────────────────────────────────────────────────────
_asr_clients: list = []
_asr_lock = threading.Lock()

# ── 对话记录广播 ──────────────────────────────────────────────────────────────
_chat_clients: list = []
_chat_lock = threading.Lock()


def _broadcast(client_list, lock, payload: str):
    with lock:
        for q in client_list:
            q.put(payload)


# ── ASR 回调：转写完 → 广播 ASR → 触发 LLM → 广播对话 ────────────────────────
def _on_asr_result(text: str):
    if not text:
        return

    # 推送 ASR 转写
    _broadcast(_asr_clients, _asr_lock, text)

    # 广播用户问题到对话记录
    _broadcast(_chat_clients, _chat_lock,
               json.dumps({"role": "user", "text": text}, ensure_ascii=False))

    # 异步 LLM（不阻塞 VAD 线程）
    def _llm_task():
        state = get_state_table()
        frame = camera.get_frame()
        reply = answer_question(text, state, frame)
        _broadcast(_chat_clients, _chat_lock,
                   json.dumps({"role": "assistant", "text": reply}, ensure_ascii=False))

    threading.Thread(target=_llm_task, daemon=True).start()


set_asr_callback(_on_asr_result)


# ── 视频帧渲染 ────────────────────────────────────────────────────────────────
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

        boxes = detect_persons(frame)
        state = get_state_table()

        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            matched = match_box_to_state((x1, y1, x2, y2), state)
            label   = f"{matched['position']} · {matched['occupation']}" if matched else "识别中..."
            frame   = draw_chinese(frame, label, x1, y1)

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + buffer.tobytes()
            + b'\r\n'
        )


# ── SSE 工具 ──────────────────────────────────────────────────────────────────
def _sse_gen(client_list, lock):
    q = queue.Queue()
    with lock:
        client_list.append(q)
    try:
        while True:
            yield f"data: {q.get()}\n\n"
    except GeneratorExit:
        pass
    finally:
        with lock:
            if q in client_list:
                client_list.remove(q)


# ── Flask 路由 ────────────────────────────────────────────────────────────────
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
    return Response(
        _sse_gen(_asr_clients, _asr_lock),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@app.route('/chat_stream')
def chat_stream():
    return Response(
        _sse_gen(_chat_clients, _chat_lock),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )