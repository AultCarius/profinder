import sounddevice as sd

# 设置中文编码兼容（避免打印设备名乱码）
import sys

sys.stdout.reconfigure(encoding='utf-8')

try:
    # 查询所有音频设备
    devices = sd.query_devices()

    # 筛选有输入通道的设备（麦克风/录音设备）
    input_devices = []
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            input_devices.append({
                "index": i,
                "name": d["name"],
                "samplerate": int(d["default_samplerate"])
            })

    # 输出结果
    if input_devices:
        print("==== 可用的输入音频设备（麦克风） ====")
        for dev in input_devices:
            print(f"[{dev['index']}] {dev['name']}  采样率:{dev['samplerate']}")
    else:
        print("⚠️  未检测到任何输入音频设备（麦克风）")
        print("所有音频设备列表：")
        for i, d in enumerate(devices):
            print(f"[{i}] {d['name']}  输入通道:{d['max_input_channels']}  输出通道:{d['max_output_channels']}")

except OSError as e:
    print(f"❌ 设备查询失败：{e}")
    print("解决方案：")
    print("1. Windows：安装 PortAudio → https://www.portaudio.com/")
    print("2. macOS：brew install portaudio")
    print("3. Linux：sudo apt-get install portaudio19-dev")
except Exception as e:
    print(f"❌ 未知错误：{e}")