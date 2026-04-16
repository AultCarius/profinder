"""
modules/camera.py — 重连版

新增：
  - is_alive()：检测摄像头线程是否在正常出帧
  - reconnect()：重新打开摄像头，重启读帧线程
  - _reader 内部检测连续读帧失败，超过阈值后自动标记断线
"""

import cv2
import threading
import time


class CameraStream:
    def __init__(self, index=1, width=1280, height=720):
        self._index  = index
        self._width  = width
        self._height = height

        self._frame      = None
        self._lock       = threading.Lock()
        self._stop       = False
        self._last_frame_time = 0.0   # 上次成功出帧的时间戳
        self._fail_count = 0           # 连续读帧失败计数

        self.cap = None
        self._thread = None
        self._open_cap()
        self._start_thread()

    # ── 内部：打开设备 ────────────────────────────────────────────────────────
    def _open_cap(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = cv2.VideoCapture(self._index + cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._fail_count = 0

    def _start_thread(self):
        self._stop  = False
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    # ── 内部：读帧循环 ────────────────────────────────────────────────────────
    def _reader(self):
        MAX_FAILS = 30   # 连续失败 30 次（约 1s）视为断线
        while not self._stop:
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
                self._last_frame_time = time.time()
                self._fail_count = 0
            else:
                self._fail_count += 1
                if self._fail_count >= MAX_FAILS:
                    print(f"[摄像头] 连续 {MAX_FAILS} 帧读取失败，设备可能已断线")
                    # 停止循环，等待外部调用 reconnect()
                    break
                time.sleep(0.03)

    # ── 公开：获取当前帧 ──────────────────────────────────────────────────────
    def get_frame(self):
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    # ── 公开：检测是否存活 ────────────────────────────────────────────────────
    def is_alive(self) -> bool:
        """
        返回摄像头是否在正常出帧。
        条件：读帧线程在运行，且 3 秒内有过成功出帧。
        """
        thread_ok = self._thread is not None and self._thread.is_alive()
        frame_ok  = (time.time() - self._last_frame_time) < 3.0
        return thread_ok and frame_ok

    # ── 公开：重新连接 ────────────────────────────────────────────────────────
    def reconnect(self) -> bool:
        """
        重新打开摄像头并重启读帧线程。
        返回 True 表示重连成功（cap 已打开），False 表示设备仍不可用。
        """
        print("[摄像头] 尝试重连...")
        # 先停掉旧线程
        self._stop = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        # 重新打开设备
        self._open_cap()

        if not self.cap.isOpened():
            print("[摄像头] 重连失败：设备无法打开")
            return False

        # 启动新线程
        self._start_thread()

        # 等最多 2 秒看是否能出帧
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if self._last_frame_time > 0 and self.is_alive():
                print("[摄像头] 重连成功")
                return True
            time.sleep(0.1)

        print("[摄像头] 重连超时：设备打开但无出帧")
        return False

    # ── 公开：释放 ────────────────────────────────────────────────────────────
    def release(self):
        self._stop = True
        if self._thread:
            self._thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()