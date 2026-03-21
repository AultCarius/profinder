import time
import numpy as np
from modules.asr import start_mic_stream, start_vad_thread, set_speech_callback

def on_speech(audio: np.ndarray):
    duration = len(audio) / 44100
    print(f"[回调] 收到一段语音，时长 {duration:.2f}s，帧数 {len(audio)}")

set_speech_callback(on_speech)
start_mic_stream()
start_vad_thread()

print("请对着麦克风说几句话，说完停顿一下，Ctrl+C 退出")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("退出")