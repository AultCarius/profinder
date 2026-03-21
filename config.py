# config.py
SILICONFLOW_API_KEY = "sk-iigejgynsdcivbhwurtuztvinsyttazulmekaxfzhblasqeu"
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"

# 视觉 LLM（多模态，用于职业识别）
# VISION_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
VISION_MODEL = "Qwen/Qwen3-VL-32B-Instruct"

# 问答 LLM（纯文字）
CHAT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# 摄像头编号（通常是 0）
CAMERA_INDEX = 1
FRAME_WIDTH  = 1024
FRAME_HEIGHT = 768

RECOGNITION_INTERVAL = 3.0  # 识别间隔秒数，根据打印的建议值调整
LLM_TIMEOUT          = 8.0  # 单次 LLM 识别超时秒数

MIC_DEVICE_INDEX = 12      # 麦克风 (Realtek(R) Audio)
MIC_SAMPLE_RATE  = 44100

VAD_THRESHOLD = 0.3   # VAD 置信度阈值，小音量麦克风可调低到 0.2

MIC_GAIN = 3.0   # 软件增益倍数，音量够用后可设为 1.0
MIC_BLOCK_SIZE = 1600