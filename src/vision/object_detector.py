"""
Chitti Object Detector Module.
Uses Ultralytics YOLOv8 for local object detection with CUDA/CPU execution and structured results.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np

from src.vision.models import DetectedObject
from src.utils.logging import log_debug, log_warning, log_chitti


class ObjectDetector:
    """Detects common household and office objects using a local YOLO model."""

    # Common objects to prioritize
    COMMON_CLASSES = {
        "person", "laptop", "cell phone", "keyboard", "mouse", "bottle",
        "cup", "book", "chair", "backpack", "clock", "tv", "remote",
        "pen", "scissors", "teddy bear", "plant", "vase", "dining table", "bed"
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.35,
        device: str = "auto",
    ):
        self.model_path = Path(model_path) if model_path else Path("models/vision/yolov8n.pt")
        if not self.model_path.is_absolute():
            root_dir = Path(__file__).resolve().parent.parent.parent
            self.model_path = root_dir / self.model_path

        self.confidence_threshold = confidence_threshold
        self.device = self._resolve_device(device)
        self._model = None
        self._load_model()

    def _resolve_device(self, requested_device: str) -> str:
        req = requested_device.strip().lower()
        if req == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda:0"
            except Exception:
                pass
            return "cpu"
        elif req == "cuda":
            return "cuda:0"
        return "cpu"

    def _load_model(self):
        """Loads the YOLO model into memory."""
        try:
            from ultralytics import YOLO
            log_debug(f"Loading YOLO object detector ({self.model_path}) on {self.device.upper()}...")
            # If local file exists, load directly; otherwise load yolov8n.pt
            target = str(self.model_path) if self.model_path.exists() else "yolov8n.pt"
            self._model = YOLO(target)
            log_debug("YOLO object detector loaded successfully.")
        except Exception as e:
            log_warning(f"Failed to load YOLO object detector: {e}")
            self._model = None

    def detect_objects(
        self,
        frame: np.ndarray,
        filter_common_only: bool = False,
    ) -> List[DetectedObject]:
        """
        Runs object detection on a BGR frame and returns structured DetectedObject items.
        """
        if frame is None or frame.size == 0 or self._model is None:
            return []

        try:
            # Run inference quietly without printing per-frame output
            results = self._model.predict(
                source=frame,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False,
            )

            detected_objects: List[DetectedObject] = []
            if not results:
                return []

            result = results[0]
            boxes = result.boxes

            for box in boxes:
                cls_id = int(box.cls[0].item())
                class_name = result.names.get(cls_id, f"class_{cls_id}").lower()
                conf = float(box.conf[0].item())

                if filter_common_only and class_name not in self.COMMON_CLASSES:
                    continue

                # Bounding box [x1, y1, x2, y2] -> (x, y, w, h)
                coords = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                w = max(1, x2 - x1)
                h = max(1, y2 - y1)

                detected_objects.append(
                    DetectedObject(
                        class_name=class_name,
                        confidence=conf,
                        bbox=(x1, y1, w, h),
                    )
                )

            return detected_objects
        except Exception as e:
            log_warning(f"Object detection inference error: {e}")
            return []
