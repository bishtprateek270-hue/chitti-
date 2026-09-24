"""
Chitti Face Detector Module.
Uses OpenCV YuNet deep learning face detector with fallback to Haar Cascade.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from src.vision.models import FaceDetection
from src.utils.logging import log_debug, log_warning, log_chitti


class FaceDetector:
    """Detects human face bounding boxes and facial landmarks in image frames."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        score_threshold: float = 0.45,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ):
        self.model_path = Path(model_path) if model_path else Path("models/vision/face_detection_yunet_2023mar.onnx")
        if not self.model_path.is_absolute():
            root_dir = Path(__file__).resolve().parent.parent.parent
            self.model_path = root_dir / self.model_path

        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self._detector: Optional[Any] = None
        self._haar_cascade: Optional[Any] = None
        self._input_size = (320, 320)
        self._load_detector()

    def _load_detector(self):
        """Initializes the YuNet face detector or falls back to Haar Cascade."""
        if cv2 is None:
            return

        if self.model_path.exists() and hasattr(cv2, "FaceDetectorYN"):
            try:
                log_debug(f"Loading YuNet face detector from {self.model_path}...")
                self._detector = cv2.FaceDetectorYN.create(
                    model=str(self.model_path),
                    config="",
                    input_size=self._input_size,
                    score_threshold=self.score_threshold,
                    nms_threshold=self.nms_threshold,
                    top_k=self.top_k,
                )
                log_debug("YuNet face detector loaded successfully.")
                return
            except Exception as e:
                log_warning(f"Failed to initialize YuNet face detector: {e}. Using Haar fallback.")

        # Fallback to OpenCV default Haar Cascade
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar_cascade = cv2.CascadeClassifier(cascade_path)
            log_debug("Haar Cascade face detector initialized.")
        except Exception as e:
            log_warning(f"Failed to initialize Haar Cascade: {e}")

    def detect_faces(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detects faces in the given BGR image frame.
        Returns a list of FaceDetection objects with bounding boxes and landmarks.
        """
        if frame is None or frame.size == 0 or cv2 is None:
            return []

        h, w = frame.shape[:2]
        detections: List[FaceDetection] = []

        # 1. Primary path: YuNet
        if self._detector is not None:
            try:
                # Update input size dynamically to match frame dimensions
                if self._input_size != (w, h):
                    self._detector.setInputSize((w, h))
                    self._input_size = (w, h)

                _, faces = self._detector.detect(frame)
                if faces is not None:
                    for face in faces:
                        # YuNet format: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rc, y_rc, x_lc, y_lc, score]
                        bbox_x = max(0, int(face[0]))
                        bbox_y = max(0, int(face[1]))
                        bbox_w = max(1, int(face[2]))
                        bbox_h = max(1, int(face[3]))
                        confidence = float(face[-1])

                        landmarks = [
                            (float(face[4]), float(face[5])),   # Right eye
                            (float(face[6]), float(face[7])),   # Left eye
                            (float(face[8]), float(face[9])),   # Nose tip
                            (float(face[10]), float(face[11])), # Right mouth corner
                            (float(face[12]), float(face[13])), # Left mouth corner
                        ]

                        detections.append(
                            FaceDetection(
                                bbox=(bbox_x, bbox_y, bbox_w, bbox_h),
                                confidence=confidence,
                                landmarks=landmarks,
                                raw_detection=face,
                            )
                        )
                return detections
            except Exception as e:
                log_warning(f"YuNet face detection error: {e}. Trying Haar fallback.")

        # 2. Fallback path: Haar Cascade
        if self._haar_cascade is not None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self._haar_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                )
                for (x, y, fw, fh) in faces:
                    detections.append(
                        FaceDetection(
                            bbox=(int(x), int(y), int(fw), int(fh)),
                            confidence=0.85,
                            landmarks=None,
                            raw_detection=None,
                        )
                    )
            except Exception as e:
                log_warning(f"Haar face detection error: {e}")

        return detections
