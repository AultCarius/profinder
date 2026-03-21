# config.py
SILICONFLOW_API_KEY = "sk-iigejgynsdcivbhwurtuztvinsyttazulmekaxfzhblasqeu"
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"

# 视觉 LLM（多模态，用于职业识别）
# VISION_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
VISION_MODEL = "Qwen/Qwen3-VL-32B-Instruct"

# 问答 LLM（纯文字）
CHAT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# 在线 ASR（SenseVoiceSmall，硅基流动）
ASR_MODEL = "FunAudioLLM/SenseVoiceSmall"
ASR_TIMEOUT = 15  # 单次 ASR 请求超时秒数

# ── LLM 问答模式 ────────────────────────────────────────────────────────────
# "vision_only" : 每次都把当前帧发给 VL 模型（推荐，信息最完整）
# "text_first"  : 文字 LLM 优先，回答含"需要仔细看"时才回退 VL（省钱省时）
# "text_only"   : 永远只用文字 LLM + 状态表（最快，但无法回答外观类问题）
LLM_MODE = "vision_only"

# 摄像头编号（通常是 0）
CAMERA_INDEX = 1
FRAME_WIDTH = 1024
FRAME_HEIGHT = 768

RECOGNITION_INTERVAL = 3.0  # 识别间隔秒数
LLM_TIMEOUT = 8.0  # 单次视觉 LLM 超时秒数

MIC_DEVICE_INDEX = 12  # 麦克风 (Realtek(R) Audio)
MIC_SAMPLE_RATE = 44100

VAD_THRESHOLD = 0.3  # VAD 置信度阈值
MIC_GAIN = 3.0  # 软件增益倍数
MIC_BLOCK_SIZE = 1600
