import queue
import threading
import time
import numpy as np
import sounddevice as sd
import torch
from silero_vad import load_silero_vad
import config
from scipy.signal import resample_poly
from math import gcd

# ── 全局共享队列（模块级单例）───────────────────────────
_audio_queue = queue.Queue()
_stream      = None

def _audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[麦克风] 状态异常: {status}")
    # 软件增益，放大 3 倍（可在 config 里配置）
    amplified = indata * config.MIC_GAIN
    _audio_queue.put(amplified.copy())

def start_mic_stream():
    global _stream
    _stream = sd.InputStream(
        device=config.MIC_DEVICE_INDEX,
        samplerate=config.MIC_SAMPLE_RATE,
        channels=1,
        dtype='float32',
        blocksize=config.MIC_BLOCK_SIZE,  # 改这里
        callback=_audio_callback
    )
    _stream.start()
    print(f"[麦克风] 已启动，设备={config.MIC_DEVICE_INDEX}")

def stop_mic_stream():
    global _stream
    if _stream:
        _stream.stop()
        _stream.close()
        _stream = None

# ── VAD ─────────────────────────────────────────────────
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
    resampled = resample_poly(chunk, up, down)
    return resampled.astype(np.float32)

def _get_vad_confidence(chunk: np.ndarray) -> float:
    resampled = _resample_to_16k(chunk)
    tensor    = torch.from_numpy(resampled).float()
    # Silero VAD 支持 256、512、768、1024 帧，取最近的合法长度
    valid_sizes = [256, 512, 768, 1024]
    target_size = min(valid_sizes, key=lambda x: abs(x - len(resampled)))
    if len(resampled) < target_size:
        resampled = np.pad(resampled, (0, target_size - len(resampled)))
    else:
        resampled = resampled[:target_size]
    tensor = torch.from_numpy(resampled).float()
    return _vad_model(tensor, 16000).item()

# ── VAD 监听线程 ─────────────────────────────────────────
_on_speech_end = None

def set_speech_callback(callback):
    global _on_speech_end
    _on_speech_end = callback

def _vad_loop():
    speaking      = False
    speech_chunks = []
    silence_count = 0
    SILENCE_LIMIT = 20        # 连续 20 块静音判定说话结束（约 0.23s）
    VAD_THRESHOLD = config.VAD_THRESHOLD       # 降低阈值，适配小音量麦克风

    print("[VAD] 监听中...")
    while True:
        # 从队列取音频块
        try:
            chunk = _audio_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        with _vad_lock:
            enabled = _vad_enabled
        if not enabled:
            continue

        chunk_1d   = chunk.flatten()
        volume     = np.abs(chunk_1d).max()

        # 音量过低直接跳过，不送 VAD（省资源，也避免噪声误触发）
        if volume < 0.01:
            if speaking:
                silence_count += 1
                if silence_count >= SILENCE_LIMIT:
                    _finish_speech(speech_chunks)
                    speaking = False
                    speech_chunks = []
                    silence_count = 0
            continue

        confidence = _get_vad_confidence(chunk_1d)
        print(f"[VAD DEBUG] 音量={volume:.4f}  置信度={confidence:.3f}  说话中={speaking}")

        is_speech = confidence > VAD_THRESHOLD

        if is_speech:
            if not speaking:
                speaking      = True
                speech_chunks = []
                silence_count = 0
                print("[VAD] 检测到说话开始")
            speech_chunks.append(chunk_1d)
            silence_count = 0

        elif speaking:
            speech_chunks.append(chunk_1d)
            silence_count += 1
            if silence_count >= SILENCE_LIMIT:
                _finish_speech(speech_chunks)
                speaking      = False
                speech_chunks = []
                silence_count = 0

def _finish_speech(chunks):
    print(f"[VAD] 说话结束，收集了 {len(chunks)} 个音频块")
    if _on_speech_end and chunks:
        audio = np.concatenate(chunks)
        threading.Thread(
            target=_on_speech_end,
            args=(audio,),
            daemon=True
        ).start()

def start_vad_thread():
    t = threading.Thread(target=_vad_loop, daemon=True)
    t.start()
    print("[VAD] 线程已启动")