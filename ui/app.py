"""
ui/app.py  — Phase 8 版本

新增改动：
  - 引入 detect_intent() 做意图分流
  - 翻译状态机：_translate_pending 标志，下一句话走翻译流程
  - process() 替换 answer_question()，返回 {reply, mode}
  - chat_stream SSE 新增 mode 字段，前端可根据 mode 渲染不同气泡样式
"""

import json
import queue
import threading
import time

import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from flask import Flask, Response, render_template, request, jsonify

from modules.camera import CameraStream
from modules.vision import (
    detect_persons, get_state_table,
    start_recognition_thread, match_box_to_state,
    request_immediate_recognition,
    set_auto_announce,
    set_announce_callback,
)
from modules.scene import (
    get_scene, start_scene_thread,
    request_scene_update, set_scene_callback,
)
from modules.asr import set_asr_callback
from modules.llm import process, detect_intent
from modules.tts import speak
import config

app = Flask(__name__)

camera = CameraStream(
    index=config.CAMERA_INDEX,
    width=config.FRAME_WIDTH,
    height=config.FRAME_HEIGHT,
)
start_recognition_thread(camera)
start_scene_thread(camera)

FONT = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 24)

# ── SSE 广播 ──────────────────────────────────────────────────────────────────
_asr_clients:  list = []
_chat_clients: list = []
_asr_lock  = threading.Lock()
_chat_lock = threading.Lock()


def _broadcast(client_list, lock, payload: str):
    with lock:
        for q in client_list:
            q.put(payload)


# ── Phase 8：翻译状态机 ───────────────────────────────────────────────────────
# 当用户说了翻译触发词后，下一句语音走翻译流程，而不是问答
_translate_pending = False
_translate_lock    = threading.Lock()


def _set_translate_pending(val: bool):
    global _translate_pending
    with _translate_lock:
        _translate_pending = val


def _get_translate_pending() -> bool:
    with _translate_lock:
        return _translate_pending


# ── ASR 回调 ────────────────────────────────────────────────────────────────
def _on_asr_result(text: str):
    if not text:
        return

    _broadcast(_asr_clients, _asr_lock, text)

    # ── 意图判断 ──────────────────────────────────────────────────────────────
    # 优先检查是否处于翻译等待状态（上一句触发了翻译词）
    if _get_translate_pending():
        # 本句直接翻译，清除等待状态
        _set_translate_pending(False)
        _broadcast(_chat_clients, _chat_lock,
                   json.dumps({"role": "user", "text": text, "mode": "translate"},
                               ensure_ascii=False))

        def _translate_task():
            result = process(text, [], None, scene="", intent="translate")
            _broadcast(_chat_clients, _chat_lock,
                       json.dumps({"role": "assistant", "text": result["reply"],
                                   "mode": "translate"}, ensure_ascii=False))
            speak(result["reply"])

        threading.Thread(target=_translate_task, daemon=True).start()
        return

    # 检测本句意图
    intent = detect_intent(text)

    if intent == "translate":
        # 触发词本身不翻译，提示用户说出要翻译的内容
        _set_translate_pending(True)
        prompt_text = "好的，请说您要翻译的内容。"
        _broadcast(_chat_clients, _chat_lock,
                   json.dumps({"role": "user", "text": text, "mode": "translate"},
                               ensure_ascii=False))
        _broadcast(_chat_clients, _chat_lock,
                   json.dumps({"role": "assistant", "text": prompt_text,
                               "mode": "translate"}, ensure_ascii=False))
        speak(prompt_text)
        return

    # help 或 qa：都需要拿画面和状态表
    _broadcast(_chat_clients, _chat_lock,
               json.dumps({"role": "user", "text": text, "mode": intent},
                           ensure_ascii=False))

    def _llm_task(captured_intent=intent, captured_text=text):
        request_immediate_recognition()
        if captured_intent != "help":
            # 求助模式直接用当前帧，不需要等识别刷新（避免多等 2 秒）
            request_scene_update()

        wait_deadline = time.time() + 2.0
        while time.time() < wait_deadline:
            state = get_state_table()
            if all(not e.get("pending", False) for e in state):
                break
            time.sleep(0.1)

        state  = get_state_table()
        frame  = camera.get_frame()
        scene  = get_scene()
        result = process(captured_text, state, frame,
                         scene=scene, intent=captured_intent)

        _broadcast(_chat_clients, _chat_lock,
                   json.dumps({"role": "assistant", "text": result["reply"],
                               "mode": result["mode"]}, ensure_ascii=False))
        speak(result["reply"])

    threading.Thread(target=_llm_task, daemon=True).start()


set_asr_callback(_on_asr_result)


def _on_announce(text: str):
    _broadcast(_chat_clients, _chat_lock,
               json.dumps({"role": "announce", "text": text, "mode": "announce"},
                           ensure_ascii=False))

set_announce_callback(_on_announce)


def _on_scene_change(label: str):
    _broadcast(_chat_clients, _chat_lock,
               json.dumps({"role": "scene", "text": label}, ensure_ascii=False))

set_scene_callback(_on_scene_change)


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
            time.sleep(0.03)
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
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/asr_stream')
def asr_stream():
    return Response(_sse_gen(_asr_clients, _asr_lock),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

@app.route('/chat_stream')
def chat_stream():
    return Response(_sse_gen(_chat_clients, _chat_lock),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

@app.route('/auto_announce', methods=['POST'])
def toggle_auto_announce():
    data    = request.get_json(silent=True) or {}
    enabled = bool(data.get("enabled", True))
    set_auto_announce(enabled)
    return jsonify({"status": "ok", "auto_announce": enabled})

@app.route('/scene')
def scene_status():
    return jsonify({"scene": get_scene()})

@app.route('/camera/status')
def camera_status():
    return jsonify({"alive": camera.is_alive()})

@app.route('/camera/reconnect', methods=['POST'])
def camera_reconnect():
    ok = camera.reconnect()
    return jsonify({"status": "ok" if ok else "failed", "alive": ok})