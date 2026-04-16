"""
modules/scene.py  — Phase 6 场景识别

功能：
  - 调用 VLM 分析当前帧，识别用户所处的场所类型（医院/商场/街道等）
  - 结果缓存 SCENE_CACHE_TTL 秒，避免频繁调用 VLM
  - get_scene()：外部获取当前场景字符串
  - request_scene_update()：外部主动触发一次场景刷新（用户提问时并行调用）
  - start_scene_thread()：启动定时刷新后台线程
  - set_scene_callback()：注册场景变化回调（供 app.py SSE 推送）
"""

import base64
import threading
import time

import cv2
from openai import OpenAI

import config

_client = OpenAI(
    api_key=config.SILICONFLOW_API_KEY,
    base_url=config.SILICONFLOW_BASE_URL,
)

# ── 场景状态 ────────────────────────────────────────────────────────────────
_scene_state = {
    "label":      "",       # 当前场景文字，如"医院门诊楼外"
    "updated_at": 0.0,      # 上次成功识别的时间戳
    "pending":    False,    # 正在识别中
}
_scene_lock = threading.Lock()

# 外部触发立即识别的事件
_scene_flag = threading.Event()

# 场景变化回调（SSE 推送）
_on_scene_change = None

# 缓存有效期（秒）—— 可在 config 中覆盖
SCENE_CACHE_TTL = getattr(config, "SCENE_CACHE_TTL", 30.0)

# 定时刷新间隔（秒）
SCENE_REFRESH_INTERVAL = getattr(config, "SCENE_REFRESH_INTERVAL", 30.0)


def get_scene() -> str:
    """返回当前场景标签（空字符串表示尚未识别）。"""
    with _scene_lock:
        return _scene_state["label"]


def request_scene_update():
    """外部主动触发一次场景识别（不等定时，立即执行）。"""
    _scene_flag.set()


def set_scene_callback(callback):
    """注册场景变化回调。签名：callback(label: str) → None"""
    global _on_scene_change
    _on_scene_change = callback


# ── VLM 调用 ────────────────────────────────────────────────────────────────
_SCENE_PROMPT = (
    "请观察图片，判断这是什么类型的室外或室内公共场所。"
    "只输出场所名称，不超过8个字，不要任何解释。"
    "例如：医院门诊楼外、商场入口、地铁站出口、街道路边、便利店内、公园广场"
)


def _frame_to_base64(frame) -> str:
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf).decode('utf-8')


def _call_vlm(frame) -> str:
    """调用 VLM 识别场景，返回场所名称字符串；失败返回空字符串。"""
    b64 = _frame_to_base64(frame)
    try:
        resp = _client.chat.completions.create(
            model=config.VISION_MODEL,
            max_tokens=20,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": _SCENE_PROMPT},
                ],
            }],
            timeout=config.LLM_TIMEOUT,
        )
        label = resp.choices[0].message.content.strip()
        # 清理 VLM 可能带的标点或多余字符
        for ch in ('。', '，', '、', '"', '"', '\n'):
            label = label.replace(ch, '')
        return label[:12]   # 最多 12 字，防止过长
    except Exception as e:
        print(f"[场景识别] VLM 调用失败：{e}")
        return ""


# ── 识别执行 ────────────────────────────────────────────────────────────────
def _do_recognize(camera):
    """获取当前帧并执行一次场景识别。"""
    frame = camera.get_frame()
    if frame is None:
        return

    with _scene_lock:
        # 检查缓存是否还新鲜
        if (time.time() - _scene_state["updated_at"]) < SCENE_CACHE_TTL:
            return
        _scene_state["pending"] = True

    print("[场景识别] 开始识别...")
    label = _call_vlm(frame)

    with _scene_lock:
        prev = _scene_state["label"]
        _scene_state["label"]      = label if label else _scene_state["label"]
        _scene_state["updated_at"] = time.time()
        _scene_state["pending"]    = False
        changed = label and label != prev

    if label:
        print(f"[场景识别] 结果：{label}")
    if changed and _on_scene_change:
        _on_scene_change(label)


# ── 后台线程 ────────────────────────────────────────────────────────────────
def _scene_loop(camera, interval):
    """定时 + 按需触发场景识别循环。"""
    while True:
        triggered = _scene_flag.wait(timeout=interval)
        if triggered:
            _scene_flag.clear()
            # 按需触发时强制刷新缓存
            with _scene_lock:
                _scene_state["updated_at"] = 0.0
        _do_recognize(camera)


def start_scene_thread(camera):
    """启动场景识别后台线程，需在摄像头初始化后调用。"""
    t = threading.Thread(
        target=_scene_loop,
        args=(camera, SCENE_REFRESH_INTERVAL),
        daemon=True,
    )
    t.start()
    print(f"[场景识别] 线程已启动，刷新间隔 {SCENE_REFRESH_INTERVAL}s")