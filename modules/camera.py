import cv2
import threading

class CameraStream:
    def __init__(self, index=1, width=1280, height=720):
        self.cap = cv2.VideoCapture(index + cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲，降低延迟

        self._frame = None
        self._lock  = threading.Lock()
        self._stop  = False

        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        """后台线程持续读取摄像头最新帧"""
        while not self._stop:
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self._frame = frame

    def get_frame(self):
        """获取当前最新帧（BGR）"""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def release(self):
        self._stop = True
        self._thread.join()
        self.cap.release()