"""
modules/llm.py

三种问答模式，由 config.LLM_MODE 控制：

  "vision_only"  每次都把当前帧发给 VL 模型，信息最完整
  "text_first"   文字 LLM 优先，回答含"需要仔细看"时回退 VL
  "text_only"    永远只用文字 LLM + 状态表
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

# ── Prompt ────────────────────────────────────────────────────────────────────

# vision_only / 回退时用
_VISION_SYSTEM = """\
你是帮助认知障碍老人寻求协助的语音助手。你能看到摄像头画面。

回答规则（必须严格遵守）：
1. 第一句直接给出行动建议或问题答案
2. 只在用户问外貌或问"什么样子"时，才描述外观；其他情况不描述外貌
3. 总字数不超过50字
4. 只说中文口语，不用括号、引号、顿号、数字序号、Markdown符号
5. 画面里看不到相关信息时，输出"需要仔细看"，绝对不要猜测
"""

# text_first / text_only 用
_TEXT_SYSTEM = """你是一个帮助认知障碍用户识别周围职业人员的语音助手。
当前画面中检测到的人员信息已以 JSON 列表提供。
请根据这些信息用简短口语化中文回答用户问题。
要求：
- 回答不超过 60 字
- 不使用 Markdown 符号、括号、引号、数字列表
- 若 JSON 列表为空或信息不足以准确回答，输出"需要仔细看"，绝对不要猜测
"""


# ── 工具 ──────────────────────────────────────────────────────────────────────
def _frame_to_base64(frame) -> str:
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode('utf-8')


def _state_to_json(state_table: list) -> str:
    if not state_table:
        return "[]"
    return json.dumps(
        [{"位置": e["position"], "职业": e["occupation"]} for e in state_table],
        ensure_ascii=False,
    )


# ── 核心调用 ──────────────────────────────────────────────────────────────────
def _call_vision(user_question: str, frame, extra_context: str = "") -> str:
    """把当前帧 + 问题发给 VL 模型。"""
    b64 = _frame_to_base64(frame)
    user_content = []

    # 把状态表作为补充文字上下文（VL 模型也能利用）
    if extra_context:
        user_content.append({"type": "text", "text": f"[当前检测到的人员] {extra_context}\n"})

    user_content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
    })
    user_content.append({"type": "text", "text": user_question})

    try:
        resp = _client.chat.completions.create(
            model=config.VISION_MODEL,
            max_tokens=150,
            messages=[
                {"role": "system", "content": _VISION_SYSTEM},
                {"role": "user",   "content": user_content},
            ],
            timeout=config.LLM_TIMEOUT,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[LLM-vision] 调用失败：{e}")
        return ""


def _call_text(user_question: str, state_table: list) -> str:
    """只用状态表回答，不发图片。"""
    state_json   = _state_to_json(state_table)
    user_content = f"当前画面人员：{state_json}\n用户问题：{user_question}"
    try:
        resp = _client.chat.completions.create(
            model=config.CHAT_MODEL,
            max_tokens=120,
            messages=[
                {"role": "system", "content": _TEXT_SYSTEM},
                {"role": "user",   "content": user_content},
            ],
            timeout=config.LLM_TIMEOUT,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[LLM-text] 调用失败：{e}")
        return ""


# ── 主入口 ────────────────────────────────────────────────────────────────────
def answer_question(user_question: str, state_table: list, current_frame) -> str:
    """
    根据 config.LLM_MODE 选择问答策略：

    vision_only  → 直接 VL，帧 + 状态表 JSON 都传入
    text_first   → 文字 LLM 优先，"需要仔细看" 时回退 VL
    text_only    → 纯文字 LLM
    """
    mode = getattr(config, "LLM_MODE", "vision_only")
    print(f"[LLM] 模式={mode}  问题：{user_question}")

    # ── vision_only ────────────────────────────────────────────────────────
    if mode == "vision_only":
        answer = _call_vision(user_question, current_frame,
                              extra_context=_state_to_json(state_table))

    # ── text_only ──────────────────────────────────────────────────────────
    elif mode == "text_only":
        answer = _call_text(user_question, state_table)

    # ── text_first（原两阶段逻辑）─────────────────────────────────────────
    else:
        answer = _call_text(user_question, state_table)
        print(f"[LLM-text] 回答：{answer}")
        if not answer or "需要仔细看" in answer:
            print("[LLM] 触发视觉回退...")
            answer = _call_vision(user_question, current_frame,
                                  extra_context=_state_to_json(state_table))
            print(f"[LLM-vision] 回答：{answer}")

    print(f"[LLM] 最终回答：{answer}")

    if not answer:
        return "抱歉，我现在无法判断，请稍后再问。"
    return answer