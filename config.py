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
CAMERA_INDEX = 2
FRAME_WIDTH = 1024
FRAME_HEIGHT = 768

RECOGNITION_INTERVAL = 3.0  # 识别间隔秒数
LLM_TIMEOUT = 8.0  # 单次视觉 LLM 超时秒数

MIC_DEVICE_INDEX = 1  # 麦克风 (Realtek(R) Audio)
MIC_SAMPLE_RATE = 44100

VAD_THRESHOLD = 0.3  # VAD 置信度阈值
MIC_GAIN = 3.0  # 软件增益倍数
MIC_BLOCK_SIZE = 1600

# ── Phase 5：主动播报 ────────────────────────────────────────────────────────
# 是否启用主动播报（新人进入画面时自动 TTS）
AUTO_ANNOUNCE = True

# 同一职业人员多少秒内不重复播报（防抖）
ANNOUNCE_COOLDOWN = 30.0

# 不触发主动播报的职业标签（路人等无效标签）
ANNOUNCE_SKIP_OCCUPATIONS = {"路人", "未知", "识别中...", "识别超时", ""}

# 标签复用的新鲜度阈值（秒），超过此时间的条目视为过时
LABEL_REUSE_MAX_AGE = 4.0

# "路人"/"未知" 条目的重新识别间隔（更积极地重试）
UNCERTAIN_RECHECK_AGE = 4.0

# 等待即时识别完成的最长时间（秒）
WAIT_FOR_RECOGNITION = True

# ── Phase 6：场景识别 ────────────────────────────────────────────────────────
# 场景识别结果缓存有效期（秒）
SCENE_CACHE_TTL = 30.0

# 场景识别后台定时刷新间隔（秒）
SCENE_REFRESH_INTERVAL = 30.0