"""
Chitti Camera Module.
Provides thread-safe webcam access, frame capture, device discovery, and clean resource management.
"""

import sys
import threading
import time
from typing import List, Optional, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from src.utils.logging import log_debug, log_warning, log_error


class CameraError(Exception):
    """Base exception for camera hardware and capture failures."""
    pass


class Camera:
    """Manages OpenCV webcam capture with lifecycle guarantees and error resilience."""

    def __init__(
        self,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self._cap: Optional[Any] = None
        self._lock = threading.Lock()
        self._is_opened = False

    @staticmethod
    def list_available_cameras(max_devices_to_test: int = 5) -> List[int]:
        """Probes video capture devices and returns list of available device indices."""
        if cv2 is None:
            return []
        available = []
        for idx in range(max_devices_to_test):
            cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append(idx)
                cap.release()
        return available

    def open(self) -> bool:
        """Opens the camera device and configures stream properties."""
        if cv2 is None:
            raise CameraError("OpenCV is not installed. Install via `pip install opencv-python`.")

        with self._lock:
            if self._is_opened and self._cap is not None and self._cap.isOpened():
                return True

            log_debug(f"Opening camera index {self.device_index}...")
            self._cap = cv2.VideoCapture(self.device_index, cv2.CAP_ANY)

            if not self._cap.isOpened():
                self._cap = None
                self._is_opened = False
                raise CameraError(
                    f"Unable to open camera index {self.device_index}. "
                    f"Check if camera is in use by another application or change CAMERA_INDEX."
                )

            # Set resolution and target FPS
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)

            # Warm-up camera by reading a couple of frames to auto-expose
            for _ in range(3):
                self._cap.read()

            self._is_opened = True
            log_debug(f"Camera index {self.device_index} successfully opened.")
            return True

    def capture_frame(self) -> np.ndarray:
        """Captures a single BGR frame from the camera."""
        with self._lock:
            if not self._is_opened or self._cap is None or not self._cap.isOpened():
                self.open()

            ret, frame = self._cap.read()
            if not ret or frame is None:
                raise CameraError("Failed to read frame from camera stream.")
            return frame

    def release(self):
        """Releases the camera device cleanly."""
        with self._lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception as e:
                    log_warning(f"Error releasing camera: {e}")
                finally:
                    self._cap = None
                    self._is_opened = False
                    log_debug("Camera released.")

    def is_opened(self) -> bool:
        """Returns True if the camera is currently opened and active."""
        with self._lock:
            return self._is_opened and self._cap is not None and self._cap.isOpened()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
