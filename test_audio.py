import sounddevice as sd

# 获取默认设备
default_input, default_output = sd.default.device

print("默认音频设备：")
print(f"默认输入设备（麦克风）序号：{default_input}")
print(f"默认输出设备（喇叭）序号：{default_output}")

# 打印所有设备详情
print("\n所有设备列表：")
print(sd.query_devices())