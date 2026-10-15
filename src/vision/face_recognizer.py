"""
Chitti Face Recognizer Module.
Extracts 128-dimensional facial feature embeddings using OpenCV SFace with face alignment,
quality filtering, multi-exemplar gallery matching, dual metrics (Cosine + L2), and temporal smoothing.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import collections
import threading
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


def l2_distance(emb_a: Union[List[float], np.ndarray], emb_b: Union[List[float], np.ndarray]) -> float:
    """Computes Euclidean (L2) distance between two normalized face feature vectors."""
    a = np.array(emb_a, dtype=np.float32).flatten()
    b = np.array(emb_b, dtype=np.float32).flatten()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a > 0.0:
        a = a / norm_a
    if norm_b > 0.0:
        b = b / norm_b
    return float(np.linalg.norm(a - b))


class TemporalIdentityTracker:
    """
    Maintains a rolling temporal window of face recognition results across consecutive frames.
    Requires consistent consensus before confirming an identity, eliminating single-frame false positives.
    """

    def __init__(self, window_size: int = 4, min_consensus: int = 2, min_avg_score: float = 0.65):
        self.window_size = window_size
        self.min_consensus = min_consensus
        self.min_avg_score = min_avg_score
        self._history: collections.deque = collections.deque(maxlen=window_size)
        self._lock = threading.Lock()

    def update(self, detected_name: str, similarity: float, is_known: bool) -> Tuple[str, float, bool]:
        """
        Pushes a new frame detection into history and returns the temporally smoothed decision.
        """
        with self._lock:
            self._history.append((detected_name, similarity, is_known))
            if not self._history:
                return ("Unknown", 0.0, False)

            name_scores: Dict[str, List[float]] = {}
            for name, score, known in self._history:
                if known and name.lower() not in ("unknown", "none"):
                    name_scores.setdefault(name, []).append(score)

            if not name_scores:
                last_score = self._history[-1][1]
                return ("Unknown", last_score, False)

            # Find candidate with highest occurrences
            best_candidate = max(name_scores.keys(), key=lambda n: len(name_scores[n]))
            scores = name_scores[best_candidate]
            count = len(scores)
            avg_score = float(np.mean(scores))

            if count >= self.min_consensus and avg_score >= self.min_avg_score:
                return (best_candidate, avg_score, True)
            else:
                return ("Unknown", avg_score, False)

    def reset(self):
        with self._lock:
            self._history.clear()


class FaceRecognizer:
    """Generates face embeddings and matches identities against registered face vectors."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        cosine_threshold: float = 0.68,
        l2_threshold: float = 0.82,
        min_face_size: int = 50,
        min_sharpness: float = 25.0,
    ):
        self.model_path = Path(model_path) if model_path else Path("models/vision/face_recognition_sface_2021dec.onnx")
        if not self.model_path.is_absolute():
            root_dir = Path(__file__).resolve().parent.parent.parent
            self.model_path = root_dir / self.model_path

        self.cosine_threshold = cosine_threshold
        self.l2_threshold = l2_threshold
        self.min_face_size = min_face_size
        self.min_sharpness = min_sharpness
        self._recognizer: Optional[Any] = None
        self._temporal_tracker = TemporalIdentityTracker()
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

    def check_face_quality(self, frame: np.ndarray, face: FaceDetection) -> Tuple[bool, str]:
        """
        Validates whether a detected face meets quality criteria for reliable recognition.
        Rejects tiny faces, motion blur, and detections missing landmarks.
        """
        if frame is None or face is None:
            return False, "Empty frame or detection"

        x, y, w, h = face.bbox
        if w < self.min_face_size or h < self.min_face_size:
            return False, f"Face too small ({w}x{h} < {self.min_face_size}x{self.min_face_size})"

        if face.raw_detection is None and face.landmarks is None:
            return False, "Face missing facial alignment landmarks"

        if cv2 is not None and frame.size > 0:
            try:
                img_h, img_w = frame.shape[:2]
                crop_x = max(0, min(x, img_w - 1))
                crop_y = max(0, min(y, img_h - 1))
                crop_w = max(1, min(w, img_w - crop_x))
                crop_h = max(1, min(h, img_h - crop_y))
                crop = frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
                if crop.size > 0:
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    if lap_var < self.min_sharpness:
                        return False, f"Face is motion-blurred (sharpness {lap_var:.1f} < {self.min_sharpness})"
            except Exception as e:
                log_debug(f"Face sharpness check non-fatal notice: {e}")

        return True, "Quality OK"

    def extract_embedding(
        self,
        frame: np.ndarray,
        face: FaceDetection,
        enforce_quality: bool = True,
    ) -> Optional[List[float]]:
        """
        Extracts a normalized 128-dimensional embedding vector for a detected face.
        Aligns the face based on facial landmarks using SFace.
        Returns None if face fails quality criteria or deep feature extraction fails.
        """
        if frame is None or cv2 is None or face is None:
            return None

        if enforce_quality:
            is_ok, reason = self.check_face_quality(frame, face)
            if not is_ok:
                log_debug(f"[VISION] Face quality check rejected: {reason}")
                return None

        # Primary path: SFace with face alignment
        if self._recognizer is not None and face.raw_detection is not None:
            try:
                aligned_face = self._recognizer.alignCrop(frame, face.raw_detection)
                feature = self._recognizer.feature(aligned_face)
                feature_vec = np.array(feature, dtype=np.float32).flatten()
                norm = np.linalg.norm(feature_vec)
                if norm > 0:
                    feature_vec = feature_vec / norm
                return feature_vec.tolist()
            except Exception as e:
                log_warning(f"SFace feature extraction error: {e}")
                return None

        return None

    def _compute_gallery_similarity(self, q_vec: np.ndarray, ref_emb: Any) -> float:
        """
        Computes robust similarity against a single embedding vector, a multi-exemplar dictionary,
        or a list of exemplar embeddings.
        """
        # Case A: Dictionary containing {"mean": [...], "exemplars": [[...], ...]}
        if isinstance(ref_emb, dict):
            exemplars = ref_emb.get("exemplars", [])
            mean_vec = ref_emb.get("mean")
            if exemplars and isinstance(exemplars, list):
                sims = [cosine_similarity(q_vec, ex) for ex in exemplars if ex]
                if sims:
                    max_sim = max(sims)
                    top2_mean = float(np.mean(sorted(sims, reverse=True)[:min(2, len(sims))]))
                    centroid_sim = cosine_similarity(q_vec, mean_vec) if mean_vec else top2_mean
                    return float(0.50 * max_sim + 0.30 * top2_mean + 0.20 * centroid_sim)
            if mean_vec:
                return float(cosine_similarity(q_vec, mean_vec))

        # Case B: List of exemplar vectors [[...], [...], ...]
        if isinstance(ref_emb, list) and len(ref_emb) > 0 and isinstance(ref_emb[0], (list, np.ndarray)):
            sims = [cosine_similarity(q_vec, ex) for ex in ref_emb if ex]
            if sims:
                max_sim = max(sims)
                top2_mean = float(np.mean(sorted(sims, reverse=True)[:min(2, len(sims))]))
                return float(0.60 * max_sim + 0.40 * top2_mean)

        # Case C: Single 1D vector (List[float] or np.ndarray)
        return float(cosine_similarity(q_vec, ref_emb))

    def match_identity(
        self,
        query_embedding: List[float],
        registered_faces: List[Tuple[int, str, Any]],
        strict_margin: bool = True,
    ) -> Tuple[str, float, bool, Optional[int]]:
        """
        Compares query embedding against registered face database.
        Supports single-vector embeddings, multi-exemplar galleries, and centroid matching.
        Enforces dual metric (Cosine >= threshold AND L2 distance <= l2_threshold)
        and margin separation between competing identities.
        Returns: (name, similarity_score, is_known, person_id)
        """
        if not query_embedding or not registered_faces:
            return ("Unknown", 0.0, False, None)

        q_vec = np.array(query_embedding, dtype=np.float32).flatten()
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        scored_candidates: List[Tuple[float, str, int]] = []

        for person_id, name, ref_emb in registered_faces:
            sim = self._compute_gallery_similarity(q_vec, ref_emb)
            scored_candidates.append((sim, name, person_id))

        if not scored_candidates:
            return ("Unknown", 0.0, False, None)

        # Sort descending by similarity score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        best_sim, best_name, best_id = scored_candidates[0]

        # Calculate equivalent L2 distance for unit vectors
        l2_dist = float(np.sqrt(max(0.0, 2.0 - 2.0 * best_sim)))

        # Ambiguity / Margin separation check if multiple distinct people are registered
        if strict_margin and len(scored_candidates) > 1:
            second_sim, second_name, _ = scored_candidates[1]
            if second_name.lower() != best_name.lower():
                if (best_sim - second_sim) < 0.05 and best_sim < 0.78:
                    log_debug(
                        f"[VISION] Ambiguous face match: '{best_name}' ({best_sim:.2f}) vs '{second_name}' ({second_sim:.2f}). Treating as Unknown."
                    )
                    return ("Unknown", max(0.0, best_sim), False, None)

        # Verification check: Must satisfy cosine threshold AND L2 distance threshold
        if best_sim >= self.cosine_threshold and l2_dist <= self.l2_threshold:
            return (best_name, best_sim, True, best_id)
        else:
            return ("Unknown", max(0.0, best_sim), False, None)

    def match_identity_smoothed(
        self,
        query_embedding: List[float],
        registered_faces: List[Tuple[int, str, Any]],
    ) -> Tuple[str, float, bool, Optional[int]]:
        """
        Runs identity matching with temporal smoothing across consecutive frames.
        """
        raw_name, raw_sim, raw_known, raw_id = self.match_identity(query_embedding, registered_faces)
        smooth_name, smooth_sim, smooth_known = self._temporal_tracker.update(raw_name, raw_sim, raw_known)
        final_id = raw_id if smooth_known else None
        return (smooth_name, smooth_sim, smooth_known, final_id)

    def reset_tracker(self):
        """Resets the temporal smoothing buffer."""
        self._temporal_tracker.reset()
