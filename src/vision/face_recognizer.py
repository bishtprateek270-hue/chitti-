"""
Chitti Face Recognizer Module.
Extracts 128-dimensional facial feature embeddings using OpenCV SFace with face alignment and cosine similarity matching.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from src.vision.models import FaceDetection
from src.utils.logging import log_debug, log_warning


def cosine_similarity(emb_a: Union[List[float], np.ndarray], emb_b: Union[List[float], np.ndarray]) -> float:
    """Computes cosine similarity between two face feature vectors."""
    a = np.array(emb_a, dtype=np.float32).flatten()
    b = np.array(emb_b, dtype=np.float32).flatten()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class FaceRecognizer:
    """Generates face embeddings and matches identities against registered face vectors."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        cosine_threshold: float = 0.60,
    ):
        self.model_path = Path(model_path) if model_path else Path("models/vision/face_recognition_sface_2021dec.onnx")
        if not self.model_path.is_absolute():
            root_dir = Path(__file__).resolve().parent.parent.parent
            self.model_path = root_dir / self.model_path

        self.cosine_threshold = cosine_threshold
        self._recognizer: Optional[Any] = None
        self._load_recognizer()

    def _load_recognizer(self):
        """Initializes the SFace deep learning feature extractor."""
        if cv2 is None:
            return

        if self.model_path.exists() and hasattr(cv2, "FaceRecognizerSF"):
            try:
                log_debug(f"Loading SFace face recognizer from {self.model_path}...")
                self._recognizer = cv2.FaceRecognizerSF.create(
                    model=str(self.model_path),
                    config="",
                )
                log_debug("SFace face recognizer loaded successfully.")
                return
            except Exception as e:
                log_warning(f"Failed to initialize SFace recognizer: {e}")

    def extract_embedding(self, frame: np.ndarray, face: FaceDetection) -> Optional[List[float]]:
        """
        Extracts a normalized 128-dimensional embedding vector for a detected face.
        Aligns the face based on facial landmarks when using SFace.
        """
        if frame is None or cv2 is None or face is None:
            return None

        # 1. Primary path: SFace with face alignment
        if self._recognizer is not None and face.raw_detection is not None:
            try:
                aligned_face = self._recognizer.alignCrop(frame, face.raw_detection)
                feature = self._recognizer.feature(aligned_face)
                feature_vec = np.array(feature, dtype=np.float32).flatten()
                # Normalize embedding vector
                norm = np.linalg.norm(feature_vec)
                if norm > 0:
                    feature_vec = feature_vec / norm
                return feature_vec.tolist()
            except Exception as e:
                log_warning(f"SFace feature extraction error: {e}")

        # 2. Fallback path: Cropped face standard resize & color histogram / gradient feature
        try:
            x, y, w, h = face.bbox
            img_h, img_w = frame.shape[:2]
            crop_x = max(0, min(x, img_w - 1))
            crop_y = max(0, min(y, img_h - 1))
            crop_w = max(1, min(w, img_w - crop_x))
            crop_h = max(1, min(h, img_h - crop_y))

            crop = frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
            if crop.size == 0:
                return None

            resized = cv2.resize(crop, (64, 64))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            # Create a 128-dimensional representation
            hist = cv2.calcHist([gray], [0], None, [128], [0, 256]).flatten()
            norm = np.linalg.norm(hist)
            if norm > 0:
                hist = hist / norm
            return hist.tolist()
        except Exception as e:
            log_warning(f"Fallback feature extraction error: {e}")
            return None

    def match_identity(
        self,
        query_embedding: List[float],
        registered_faces: List[Tuple[int, str, List[float]]],
    ) -> Tuple[str, float, bool, Optional[int]]:
        """
        Compares query embedding against registered face database.
        Returns: (name, similarity_score, is_known, person_id)
        """
        if not query_embedding or not registered_faces:
            return ("Unknown", 0.0, False, None)

        best_sim = -1.0
        best_name = "Unknown"
        best_id = None

        for person_id, name, ref_emb in registered_faces:
            sim = cosine_similarity(query_embedding, ref_emb)
            if sim > best_sim:
                best_sim = sim
                best_name = name
                best_id = person_id

        if best_sim >= self.cosine_threshold:
            return (best_name, best_sim, True, best_id)
        else:
            return ("Unknown", max(0.0, best_sim), False, None)
