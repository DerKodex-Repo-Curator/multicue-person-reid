from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FramePacket:
    frame: np.ndarray
    index: int
    timestamp: float


class ThreadedCamera:
    """Background OpenCV camera reader.

    Heavy ML inference can take longer than a camera frame interval. Reading the
    camera on the same thread makes the preview feel frozen and can leave macOS
    camera buffers stale. This helper continuously grabs frames and exposes the
    most recent one to the UI/inference loop.
    """

    def __init__(self, camera: int, *, width: int | None = None, height: int | None = None, fps: int | None = None) -> None:
        self.camera = camera
        self.cap = cv2.VideoCapture(camera)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera}")
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        self._lock = threading.Lock()
        self._packet: FramePacket | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._index = 0

    def start(self) -> "ThreadedCamera":
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._loop, name=f"camera-{self.camera}", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            if ok:
                with self._lock:
                    self._packet = FramePacket(frame=frame, index=self._index, timestamp=time.time())
                    self._index += 1
            else:
                time.sleep(0.01)

    def read(self) -> FramePacket | None:
        with self._lock:
            if self._packet is None:
                return None
            # Copy so drawing overlays never mutates the capture buffer.
            return FramePacket(self._packet.frame.copy(), self._packet.index, self._packet.timestamp)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self.cap.release()
