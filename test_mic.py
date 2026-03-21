import time
import numpy as np
import sounddevice as sd
import config

DURATION = 3  # 录音秒数

print(f"使用设备 [{config.MIC_DEVICE_INDEX}] 录音 {DURATION} 秒，请说话...")
audio = sd.rec(
    int(DURATION * config.MIC_SAMPLE_RATE),
    samplerate=config.MIC_SAMPLE_RATE,
    channels=1,
    dtype='float32',
    device=config.MIC_DEVICE_INDEX
)
sd.wait()

peak = np.abs(audio).max()
print(f"峰值：{peak:.4f}")

if peak < 0.001:
    print("几乎没有声音，设备可能选错了")
else:
    print("有声音，现在播放回放...")
    sd.play(audio, samplerate=config.MIC_SAMPLE_RATE)
    sd.wait()
    print("播放完成")