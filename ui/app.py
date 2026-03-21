import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from flask import Flask, Response, render_template
from modules.camera import CameraStream
from modules.vision import detect_persons, get_state_table, start_recognition_thread, match_box_to_state
import config

app = Flask(__name__)
camera = CameraStream(
    index=config.CAMERA_INDEX,
    width=config.FRAME_WIDTH,
    height=config.FRAME_HEIGHT
)

# 启动后台识别线程
start_recognition_thread(camera)

FONT = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 24)

def draw_chinese(frame, text, x, y, bg_color=(0, 255, 0), text_color=(0, 0, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    bbox = draw.textbbox((0, 0), text, font=FONT)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    draw.rectangle([x, y - text_h - 8, x + text_w + 8, y], fill=bg_color)
    draw.text((x + 4, y - text_h - 4), text, font=FONT, fill=text_color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def generate_frames():
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue

        # 只用 YOLO 画框（快），职业标签从状态表读（不阻塞）
        boxes = detect_persons(frame)
        state = get_state_table()

        # 用状态表里的职业标签匹配当前检测到的框
        # 简单策略：按顺序对应
        # 改成（IoU 匹配，可靠）
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            matched = match_box_to_state((x1, y1, x2, y2), state)
            if matched:
                label = f"{matched['position']} · {matched['occupation']}"
            else:
                label = "识别中..."
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )