"""
modules/llm.py — Phase 8 版本

新增改动：
  - 意图识别：在 process() 入口统一分流三种模式
      translate  → 翻译（Phase 8）
      help       → 求助引导（Phase 8 新增通用求助）
      qa         → 原有问答逻辑
  - translate_text()：调用文字 LLM 翻译任意语言 → 普通话
  - answer_help()：携图 + 场景 + 人员状态，给出具体求助行动建议
  - 对外暴露 process() 作为统一入口，替换原有 answer_question()
  - answer_question() 保留作为别名，向后兼容
"""

import base64
import json
import re

import cv2
from openai import OpenAI

import config

_client = OpenAI(
    api_key=config.SILICONFLOW_API_KEY,
    base_url=config.SILICONFLOW_BASE_URL,
)

# ──────────────────────────────────────────────────────────────────────────────
# 意图触发词
# ──────────────────────────────────────────────────────────────────────────────

# 翻译触发：下一句话作为翻译内容
TRANSLATE_TRIGGERS = [
    "帮我翻译", "翻译一下", "翻译这句", "他说什么", "她说什么",
    "这是什么意思", "什么意思", "听不懂", "翻译",
]

# 求助触发：当前情境下给出行动建议
HELP_TRIGGERS = [
    "我需要帮助", "帮帮我", "我需要帮忙", "我不知道怎么办",
    "我迷路了", "我找不到", "我不知道去哪", "我迷路",
    "帮我", "救我", "求助", "我需要有人帮我",
    "我不知道该怎么做", "我该怎么办", "怎么办",
]


def _match_intent(text: str) -> str:
    """
    返回意图字符串：'translate' / 'help' / 'qa'
    先检查翻译，再检查求助，都不匹配则为普通问答。
    """
    for kw in TRANSLATE_TRIGGERS:
        if kw in text:
            return "translate"
    for kw in HELP_TRIGGERS:
        if kw in text:
            return "help"
    return "qa"


# ──────────────────────────────────────────────────────────────────────────────
# Prompt 定义
# ──────────────────────────────────────────────────────────────────────────────

_VISION_SYSTEM = """\
你是帮助认知障碍老人寻求协助的语音助手。你能看到摄像头画面。

用户的家在哈尔滨南通大街十四号.

回答规则（必须严格遵守）：
1. 第一句直接给出行动建议或问题答案
2. 只在用户问外貌或问"什么样子"时，才描述外观；其他情况不描述外貌
3. 总字数不超过50字
4. 只说中文口语，不用括号、引号、顿号、数字序号、Markdown符号
5. 画面里看不到相关信息时，输出"需要仔细看"，绝对不要猜测
"""

_TEXT_SYSTEM = """\
你是一个帮助认知障碍用户识别周围职业人员的语音助手。
当前画面中检测到的人员信息已以 JSON 列表提供。
请根据这些信息用简短口语化中文回答用户问题。
要求：
- 回答不超过 60 字
- 不使用 Markdown 符号、括号、引号、数字列表
- 若 JSON 列表为空或信息不足以准确回答，输出"需要仔细看"，绝对不要猜测
"""

# 求助专用 prompt：重点是给出「找谁」「说什么」两个明确动作
_HELP_SYSTEM = """\
你是帮助认知障碍老人在户外寻求协助的语音助手。你能看到摄像头画面。

用户的家在哈尔滨南通大街十四号.

用户现在感到迷茫或需要帮助。请根据画面中的人员和场景，给用户提供回复。

回答规则（必须严格遵守）：

1. 优先推荐画面中职业明确的工作人员（如保安、医生、警察、服务员），说清楚他们在哪个方向, 并回答用户问题比如"我的家在哪"
2. 如果没有合适人选，建议用户原地等待或走向人多的地方
3. 给出一句用户可以直接开口说的话，例如「您好，我需要帮助，能帮我打电话给家人吗」
4. 总字数不超过60字，只说中文口语，不用任何符号
5. 语气要镇定温和，不要让用户感到慌张
"""

# 翻译专用 prompt
_TRANSLATE_SYSTEM = """\
你是一个翻译助手。请将用户发送的内容翻译成普通话中文。
只输出译文，不要解释、不要加引号、不要说"翻译结果是"。
如果内容本身就是普通话，原文输出即可。
"""


# ──────────────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────────────

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


def _scene_prefix(scene: str) -> str:
    if not scene:
        return ""
    return f"当前场景：{scene}\n"


def _build_context_text(state_table: list, scene: str) -> str:
    parts = []
    if scene:
        parts.append(f"当前场景：{scene}")
    parts.append(f"当前画面人员：{_state_to_json(state_table)}")
    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# 三种模式的核心调用
# ──────────────────────────────────────────────────────────────────────────────

def _call_vision(user_question: str, frame, extra_context: str = "", scene: str = "") -> str:
    b64 = _frame_to_base64(frame)
    user_content = []

    ctx_parts = []
    if scene:
        ctx_parts.append(_scene_prefix(scene))
    if extra_context:
        ctx_parts.append(f"[当前检测到的人员] {extra_context}")
    if ctx_parts:
        user_content.append({"type": "text", "text": "\n".join(ctx_parts) + "\n"})

    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
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


def _call_text(user_question: str, state_table: list, scene: str = "") -> str:
    user_content = f"{_build_context_text(state_table, scene)}\n用户问题：{user_question}"
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


def translate_text(text_to_translate: str) -> str:
    """
    Phase 8：翻译任意语言到普通话。
    直接调用文字 LLM，不需要画面信息。
    """
    print(f"[LLM-translate] 翻译内容：{text_to_translate}")
    try:
        resp = _client.chat.completions.create(
            model=config.CHAT_MODEL,
            max_tokens=150,
            messages=[
                {"role": "system", "content": _TRANSLATE_SYSTEM},
                {"role": "user",   "content": text_to_translate},
            ],
            timeout=config.LLM_TIMEOUT,
        )
        result = resp.choices[0].message.content.strip()
        print(f"[LLM-translate] 结果：{result}")
        return result
    except Exception as e:
        print(f"[LLM-translate] 调用失败：{e}")
        return ""


def answer_help(state_table: list, frame, scene: str = "") -> str:
    """
    Phase 8：求助模式。携图 + 场景 + 人员，给出具体行动建议。
    固定使用视觉 LLM，确保建议基于实际画面。
    """
    print(f"[LLM-help] 触发求助，场景={scene or '未知'}")
    b64 = _frame_to_base64(frame)

    ctx_parts = []
    if scene:
        ctx_parts.append(f"当前场景：{scene}")
    state_json = _state_to_json(state_table)
    if state_json != "[]":
        ctx_parts.append(f"当前画面人员：{state_json}")

    user_content = []
    if ctx_parts:
        user_content.append({"type": "text", "text": "\n".join(ctx_parts) + "\n"})
    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    user_content.append({"type": "text", "text": "我需要帮助，请告诉我该怎么做。"})

    try:
        resp = _client.chat.completions.create(
            model=config.VISION_MODEL,
            max_tokens=120,
            messages=[
                {"role": "system", "content": _HELP_SYSTEM},
                {"role": "user",   "content": user_content},
            ],
            timeout=config.LLM_TIMEOUT,
        )
        result = resp.choices[0].message.content.strip()
        print(f"[LLM-help] 建议：{result}")
        return result
    except Exception as e:
        print(f"[LLM-help] 调用失败：{e}")
        return "请向附近的工作人员说：您好，我需要帮助，能帮我联系家人吗。"


# ──────────────────────────────────────────────────────────────────────────────
# 统一入口
# ──────────────────────────────────────────────────────────────────────────────

def process(user_text: str, state_table: list, current_frame,
            scene: str = "", intent: str = "qa") -> dict:
    """
    统一处理入口，由 app.py 调用。

    参数：
      user_text     用户说的话（原文）
      state_table   当前人员状态表
      current_frame 当前摄像头帧
      scene         当前场景标签
      intent        意图，'qa' / 'translate' / 'help'
                    由 app.py 在调用前用 detect_intent() 判断

    返回 dict：
      {
        "reply":  str,   # 播报给用户的文字
        "mode":   str,   # 实际执行的模式标签，用于前端气泡样式
      }
    """
    mode = getattr(config, "LLM_MODE", "vision_only")

    if intent == "translate":
        reply = translate_text(user_text)
        if not reply:
            reply = "抱歉，翻译失败，请再说一遍。"
        return {"reply": reply, "mode": "translate"}

    if intent == "help":
        reply = answer_help(state_table, current_frame, scene=scene)
        return {"reply": reply, "mode": "help"}

    # ── 普通问答 ──────────────────────────────────────────────────────────────
    print(f"[LLM] 问答模式={mode}  场景={scene or '未知'}  问题：{user_text}")

    if mode == "vision_only":
        answer = _call_vision(user_text, current_frame,
                              extra_context=_state_to_json(state_table),
                              scene=scene)
    elif mode == "text_only":
        answer = _call_text(user_text, state_table, scene=scene)
    else:  # text_first
        answer = _call_text(user_text, state_table, scene=scene)
        print(f"[LLM-text] 回答：{answer}")
        if not answer or "需要仔细看" in answer:
            print("[LLM] 触发视觉回退...")
            answer = _call_vision(user_text, current_frame,
                                  extra_context=_state_to_json(state_table),
                                  scene=scene)
            print(f"[LLM-vision] 回答：{answer}")

    print(f"[LLM] 最终回答：{answer}")
    if not answer:
        answer = "抱歉，我现在无法判断，请稍后再问。"
    return {"reply": answer, "mode": "qa"}


def detect_intent(text: str) -> str:
    """对外暴露意图识别，供 app.py 使用。"""
    return _match_intent(text)


# 向后兼容别名
def answer_question(user_question: str, state_table: list,
                    current_frame, scene: str = "") -> str:
    result = process(user_question, state_table, current_frame, scene=scene, intent="qa")
    return result["reply"]