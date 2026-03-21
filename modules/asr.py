import io
import queue
import threading
import wave

import numpy as np
import requests
import sounddevice as sd
import torch
from silero_vad import load_silero_vad
from scipy.signal import resample_poly
from math import gcd

import config

# ── 全局共享队列（模块级单例）──────────────────────────────────────────────
_audio_queue = queue.Queue()
_stream      = None


def _audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[麦克风] 状态异常: {status}")
    amplified = indata * config.MIC_GAIN
    _audio_queue.put(amplified.copy())


def start_mic_stream():
    global _stream
    _stream = sd.InputStream(
        device=config.MIC_DEVICE_INDEX,
        samplerate=config.MIC_SAMPLE_RATE,
        channels=1,
        dtype='float32',
        blocksize=config.MIC_BLOCK_SIZE,
        callback=_audio_callback,
    )
    _stream.start()
    print(f"[麦克风] 已启动，设备={config.MIC_DEVICE_INDEX}")


def stop_mic_stream():
    global _stream
    if _stream:
        _stream.stop()
        _stream.close()
        _stream = None


# ── VAD ────────────────────────────────────────────────────────────────────
_vad_model   = load_silero_vad()
_vad_enabled = True
_vad_lock    = threading.Lock()


def set_vad_enabled(enabled: bool):
    global _vad_enabled
    with _vad_lock:
        _vad_enabled = enabled
    print(f"[VAD] {'启用' if enabled else '暂停'}")


def _resample_to_16k(chunk: np.ndarray) -> np.ndarray:
    src  = config.MIC_SAMPLE_RATE
    dst  = 16000
    g    = gcd(src, dst)
    up   = dst // g
    down = src // g
    return resample_poly(chunk, up, down).astype(np.float32)


def _get_vad_confidence(chunk: np.ndarray) -> float:
    resampled   = _resample_to_16k(chunk)
    valid_sizes = [256, 512, 768, 1024]
    target_size = min(valid_sizes, key=lambda x: abs(x - len(resampled)))
    if len(resampled) < target_size:
        resampled = np.pad(resampled, (0, target_size - len(resampled)))
    else:
        resampled = resampled[:target_size]
    tensor = torch.from_numpy(resampled).float()
    return _vad_model(tensor, 16000).item()


# ── 在线 ASR（SenseVoiceSmall via 硅基流动）───────────────────────────────
def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """float32 numpy 音频 → 16bit WAV 字节流（内存，不写磁盘）"""
    pcm_int16 = np.clip(audio, -1.0, 1.0)
    pcm_int16 = (pcm_int16 * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())
    return buf.getvalue()


def transcribe_audio(audio: np.ndarray) -> str:
    """
    将 VAD 收集到的 float32 音频上传到硅基流动 SenseVoiceSmall 转写。
    返回识别到的中文文本；失败时返回空字符串。
    """
    wav_bytes = _audio_to_wav_bytes(audio, config.MIC_SAMPLE_RATE)
    try:
        resp = requests.post(
            url=f"{config.SILICONFLOW_BASE_URL}/audio/transcriptions",
            headers={"Authorization": f"Bearer {config.SILICONFLOW_API_KEY}"},
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data={"model": config.ASR_MODEL},
            timeout=config.ASR_TIMEOUT,
        )
        resp.raise_for_status()
        text = resp.json().get("text", "").strip()
        print(f"[ASR] 转写结果：{text}")
        return text
    except requests.exceptions.Timeout:
        print("[ASR] 请求超时")
        return ""
    except requests.exceptions.HTTPError as e:
        print(f"[ASR] HTTP 错误：{e.response.status_code} {e.response.text}")
        return ""
    except Exception as e:
        print(f"[ASR] 调用失败：{e}")
        return ""


# ── ASR 结果回调（供外部注册，例如 Flask SSE 推送）───────────────────────
# 回调签名：callback(text: str) → None
_on_asr_result = None


def set_asr_callback(callback):
    """注册 ASR 结果回调，每次转写完成后调用 callback(text)。"""
    global _on_asr_result
    _on_asr_result = callback


# ── VAD 监听线程 ────────────────────────────────────────────────────────────
def _vad_loop():
    speaking      = False
    speech_chunks = []
    silence_count = 0
    SILENCE_LIMIT = 20   # 连续 20 块静音判为说话结束（约 0.23s @ blocksize=1600）

    print("[VAD] 监听中...")
    while True:
        try:
            chunk = _audio_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        with _vad_lock:
            enabled = _vad_enabled
        if not enabled:
            continue

        chunk_1d = chunk.flatten()
        volume   = np.abs(chunk_1d).max()

        # 音量过低直接当静音处理
        if volume < 0.01:
            if speaking:
                silence_count += 1
                if silence_count >= SILENCE_LIMIT:
                    _finish_speech(speech_chunks)
                    speaking, speech_chunks, silence_count = False, [], 0
            continue

        confidence = _get_vad_confidence(chunk_1d)
        print(f"[VAD DEBUG] 音量={volume:.4f}  置信度={confidence:.3f}  说话中={speaking}")

        is_speech = confidence > config.VAD_THRESHOLD

        if is_speech:
            if not speaking:
                speaking, speech_chunks, silence_count = True, [], 0
                print("[VAD] 检测到说话开始")
            speech_chunks.append(chunk_1d)
            silence_count = 0
        elif speaking:
            speech_chunks.append(chunk_1d)
            silence_count += 1
            if silence_count >= SILENCE_LIMIT:
                _finish_speech(speech_chunks)
                speaking, speech_chunks, silence_count = False, [], 0


def _finish_speech(chunks):
    """说话结束：合并音频 → 异步调用 ASR → 触发回调"""
    print(f"[VAD] 说话结束，收集了 {len(chunks)} 个音频块")
    if not chunks:
        return
    audio = np.concatenate(chunks)

    def _run():
        text = transcribe_audio(audio)
        if text and _on_asr_result:
            _on_asr_result(text)

    threading.Thread(target=_run, daemon=True).start()


def start_vad_thread():
    t = threading.Thread(target=_vad_loop, daemon=True)
    t.start()
    print("[VAD] 线程已启动")