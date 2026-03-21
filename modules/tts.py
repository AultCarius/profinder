"""
modules/tts.py

Edge-TTS 语音播报模块：
  - speak(text)：将文字合成语音并播放
  - 播放期间自动暂停 VAD，播完后恢复
"""

import asyncio
import io
import threading

import edge_tts
import numpy as np
import soundfile as sf
import sounddevice as sd

from modules.asr import set_vad_enabled

# TTS 语音配置（中文女声，语速稍快适合辅助场景）
TTS_VOICE = "zh-CN-XiaoxiaoNeural"
TTS_RATE  = "+10%"

# 同一时间只允许一个播报任务，避免重叠
_tts_lock = threading.Lock()


def _run_async(coro):
    """在当前线程内安全地运行一个 asyncio 协程。"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


async def _synthesize(text: str) -> bytes:
    """调用 Edge-TTS，返回 MP3 字节流。"""
    communicate = edge_tts.Communicate(text, voice=TTS_VOICE, rate=TTS_RATE)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    buf.seek(0)
    return buf.read()


def _play_audio(mp3_bytes: bytes):
    """将 MP3 字节流解码后用 sounddevice 播放（阻塞直到播完）。"""
    buf = io.BytesIO(mp3_bytes)
    data, samplerate = sf.read(buf, dtype="float32")
    # 若为单声道确保 shape=(N,)
    if data.ndim > 1:
        data = data.mean(axis=1)
    sd.play(data, samplerate=samplerate)
    sd.wait()


def speak(text: str):
    """
    公开接口：合成并播放文字。
    - 非阻塞（在新线程里运行）
    - 播放期间暂停 VAD，防止 TTS 声音触发自身
    """
    if not text or not text.strip():
        return

    def _task():
        if not _tts_lock.acquire(blocking=False):
            print("[TTS] 上一段播报未完成，跳过本次")
            return
        try:
            print(f"[TTS] 开始播报：{text[:30]}{'...' if len(text) > 30 else ''}")
            set_vad_enabled(False)          # Task 4.2：播报期间暂停 VAD
            mp3 = _run_async(_synthesize(text))
            _play_audio(mp3)
            print("[TTS] 播报完成")
        except Exception as e:
            print(f"[TTS] 播报失败：{e}")
        finally:
            set_vad_enabled(True)           # 播完后恢复 VAD
            _tts_lock.release()

    threading.Thread(target=_task, daemon=True).start()