"""
modules/llm.py

两阶段问答 + 分层回退：
  1. 用状态表 JSON + 用户问题组装 Prompt，调用文字 LLM 获取回答
  2. 若回答含"需要仔细看"，携带当前帧调用视觉 LLM 重试
"""

import base64
import json

import cv2
from openai import OpenAI

import config

_client = OpenAI(
    api_key=config.SILICONFLOW_API_KEY,
    base_url=config.SILICONFLOW_BASE_URL,
)

# ── 系统 Prompt ──────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """你是一个帮助认知障碍用户识别周围职业人员的语音助手。
当前画面中检测到的人员信息已以 JSON 列表提供给你。
请根据这些信息用简短口语化的中文回答用户的问题。
要求：
- 回答不超过 60 字
- 不使用 Markdown 符号、括号、引号
- 不使用数字编号列表
- 若 JSON 列表为空或信息不足以回答，输出"需要仔细看"，不得猜测
"""

_FALLBACK_PROMPT = """你是一个帮助认知障碍用户识别周围职业人员的语音助手。
请观察图片中所有人物，并根据用户问题给出口语化中文回答。
要求：
- 回答不超过 60 字
- 不使用 Markdown 符号、括号、引号
- 不使用数字编号列表
"""


# ── 工具函数 ─────────────────────────────────────────────────────────────────
def _frame_to_base64(frame) -> str:
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode('utf-8')


def _state_to_text(state_table: list) -> str:
    """把状态表转成人类可读的简短描述，同时保留 JSON 原文给 LLM。"""
    if not state_table:
        return "[]"
    clean = [
        {"位置": e["position"], "职业": e["occupation"]}
        for e in state_table
    ]
    return json.dumps(clean, ensure_ascii=False)


# ── 第一阶段：文字 LLM ────────────────────────────────────────────────────────
def ask_text_llm(user_question: str, state_table: list) -> str:
    """
    用状态表 + 用户问题调用文字 LLM。
    返回回答字符串；失败时返回空字符串。
    """
    state_json = _state_to_text(state_table)
    user_content = f"当前画面人员：{state_json}\n用户问题：{user_question}"

    try:
        resp = _client.chat.completions.create(
            model=config.CHAT_MODEL,
            max_tokens=120,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            timeout=config.LLM_TIMEOUT,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[LLM-text] 调用失败：{e}")
        return ""


# ── 第二阶段：视觉 LLM 回退 ───────────────────────────────────────────────────
def ask_vision_llm(user_question: str, frame) -> str:
    """
    携带当前摄像头帧调用视觉 LLM。
    返回回答字符串；失败时返回空字符串。
    """
    b64 = _frame_to_base64(frame)

    try:
        resp = _client.chat.completions.create(
            model=config.VISION_MODEL,
            max_tokens=120,
            messages=[
                {"role": "system", "content": _FALLBACK_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                        {"type": "text", "text": user_question},
                    ],
                },
            ],
            timeout=config.LLM_TIMEOUT,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[LLM-vision] 调用失败：{e}")
        return ""


# ── 主入口：分层问答 ──────────────────────────────────────────────────────────
def answer_question(user_question: str, state_table: list, current_frame) -> str:
    """
    完整问答流程：
      1. 文字 LLM 优先
      2. 若回答含"需要仔细看" → 视觉 LLM 回退
      3. 两次都失败 → 返回兜底提示

    参数
    ----
    user_question  : ASR 转写的用户问题
    state_table    : vision.get_state_table() 返回的当前状态列表
    current_frame  : camera.get_frame() 返回的当前帧（BGR ndarray）
    """
    print(f"[LLM] 收到问题：{user_question}")

    # 第一阶段
    answer = ask_text_llm(user_question, state_table)
    print(f"[LLM-text] 回答：{answer}")

    # 触发回退
    if not answer or "需要仔细看" in answer:
        print("[LLM] 触发视觉回退...")
        answer = ask_vision_llm(user_question, current_frame)
        print(f"[LLM-vision] 回答：{answer}")

    # 最终兜底
    if not answer:
        answer = "抱歉，我现在无法判断，请稍后再问。"

    return answer