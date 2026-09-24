"""
Chitti Vision Manager Module.
Coordinates camera streaming, face detection, face recognition, object detection, registration, and LLM context formatting.
"""

import collections
import re
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from src.vision.models import (
    FaceDetection,
    RecognizedPerson,
    DetectedObject,
    VisionAnalysisResult,
)
from src.vision.camera import Camera, CameraError
from src.vision.face_detector import FaceDetector
from src.vision.face_recognizer import FaceRecognizer
from src.vision.face_database import FaceDatabase
from src.vision.object_detector import ObjectDetector
from src.config import get_config
from src.utils.logging import log_debug, log_warning, log_chitti, log_state


VISION_QUERY_PATTERNS = [
    r"(?i)\b(?:what(?:\s+can|\s+do)?\s+you\s+see)\b",
    r"(?i)\b(?:who\s+is\s+(?:in\s+front\s+of\s+you|there|sitting|standing|this|that))\b",
    r"(?i)\b(?:is\s+anyone\s+there|anyone\s+in\s+front\s+of\s+you)\b",
    r"(?i)\b(?:who\s+am\s+i|do\s+you\s+recognize\s+me|do\s+you\s+know\s+who\s+i\s+am)\b",
    r"(?i)\b(?:what\s+objects\s+(?:can\s+you\s+see|are\s+there|are\s+on\s+the\s+desk))\b",
    r"(?i)\b(?:is\s+there\s+a\s+\w+)\b",
    r"(?i)\b(?:look\s+at\s+(?:me|this|the\s+camera))\b",
    r"(?i)\b(?:describe\s+(?:what\s+you\s+see|the\s+scene|the\s+room))\b",
    r"(?i)\b(?:how\s+many\s+people\s+(?:are\s+there|do\s+you\s+see))\b",
]


class VisionManager:
    """Central manager for Chitti's visual perception capabilities."""

    def __init__(
        self,
        camera: Optional[Camera] = None,
        face_detector: Optional[FaceDetector] = None,
        face_recognizer: Optional[FaceRecognizer] = None,
        face_db: Optional[FaceDatabase] = None,
        object_detector: Optional[ObjectDetector] = None,
    ):
        cfg = get_config()

        self.camera = camera or Camera(
            device_index=cfg.camera.device_index,
            width=cfg.camera.width,
            height=cfg.camera.height,
            fps=cfg.camera.fps,
        )

        self.face_detector = face_detector or FaceDetector(
            model_path=cfg.vision.face_detection_model,
        )

        self.face_recognizer = face_recognizer or FaceRecognizer(
            model_path=cfg.vision.face_recognition_model,
            cosine_threshold=cfg.vision.face_recognition_threshold,
        )

        self.face_db = face_db or FaceDatabase(db_path=cfg.vision.faces_db_path)

        self.object_detector = object_detector or ObjectDetector(
            model_path=cfg.vision.object_detection_model,
            confidence_threshold=cfg.vision.object_confidence_threshold,
            device=cfg.vision.device,
        )

        self.pending_face_deletion: Optional[str] = None

    @staticmethod
    def is_vision_query(user_text: str) -> bool:
        """Determines if the user prompt is asking about the visual scene or recognized people."""
        if not user_text:
            return False
        for pattern in VISION_QUERY_PATTERNS:
            if re.search(pattern, user_text):
                return True
        return False

    def analyze_frame(self, frame: Optional[np.ndarray] = None) -> VisionAnalysisResult:
        """
        Runs complete visual analysis pipeline (faces + recognition + objects) on a single frame.
        """
        now_iso = datetime.now(timezone.utc).isoformat()

        # Capture frame if not provided
        if frame is None:
            try:
                frame = self.camera.capture_frame()
            except Exception as e:
                log_warning(f"Camera frame capture failed: {e}")
                return VisionAnalysisResult(
                    timestamp=now_iso,
                    summary_text="Camera is currently unavailable.",
                )

        recognized_people: List[RecognizedPerson] = []
        registered_faces = self.face_db.get_all_registered_faces()

        # 1. Face Detection & Recognition
        try:
            detections = self.face_detector.detect_faces(frame)
            for det in detections:
                emb = self.face_recognizer.extract_embedding(frame, det)
                if emb is not None:
                    name, sim, is_known, person_id = self.face_recognizer.match_identity(emb, registered_faces)
                    recognized_people.append(
                        RecognizedPerson(
                            name=name,
                            confidence=det.confidence,
                            is_known=is_known,
                            bbox=det.bbox,
                            person_id=person_id,
                            similarity=sim,
                        )
                    )
                else:
                    recognized_people.append(
                        RecognizedPerson(
                            name="Unknown",
                            confidence=det.confidence,
                            is_known=False,
                            bbox=det.bbox,
                        )
                    )
        except Exception as e:
            log_warning(f"Face processing error: {e}")

        # 2. Object Detection
        detected_objects: List[DetectedObject] = []
        try:
            detected_objects = self.object_detector.detect_objects(frame)
        except Exception as e:
            log_warning(f"Object detection error: {e}")

        # 3. Aggregate Object Counts
        counts = dict(collections.Counter(obj.class_name for obj in detected_objects))

        # 4. Create natural summary
        summary = self._generate_summary(recognized_people, detected_objects, counts)

        return VisionAnalysisResult(
            timestamp=now_iso,
            faces=recognized_people,
            objects=detected_objects,
            object_counts=counts,
            summary_text=summary,
            frame=frame,
        )

    def _generate_summary(
        self,
        faces: List[RecognizedPerson],
        objects: List[DetectedObject],
        counts: Dict[str, int],
    ) -> str:
        """Generates a concise textual description of the scene."""
        parts = []

        if faces:
            known = [f.name for f in faces if f.is_known]
            unknown_cnt = sum(1 for f in faces if not f.is_known)

            face_desc = []
            if known:
                face_desc.append(f"{', '.join(known)}")
            if unknown_cnt > 0:
                face_desc.append(f"{unknown_cnt} unrecognized person" if unknown_cnt == 1 else f"{unknown_cnt} unrecognized people")

            parts.append(f"People detected: {', '.join(face_desc)}.")
        else:
            parts.append("No people detected in view.")

        non_person_objs = {k: v for k, v in counts.items() if k != "person"}
        if non_person_objs:
            obj_desc = [f"{count} {name}" if count == 1 else f"{count} {name}s" for name, count in non_person_objs.items()]
            parts.append(f"Objects in view: {', '.join(obj_desc)}.")
        else:
            parts.append("No common objects identified.")

        return " ".join(parts)

    def format_vision_context_for_llm(self, result: VisionAnalysisResult) -> str:
        """
        Formats visual perception into structured LLM context.
        """
        lines = ["\n\nCURRENT VISUAL PERCEPTION (WHAT CHITTI SEES):"]

        if result.faces:
            lines.append("People:")
            for f in result.faces:
                if f.is_known:
                    lines.append(f"- {f.name} (Registered identity, recognition similarity: {f.similarity:.2f})")
                else:
                    lines.append("- Unknown person (Not recognized)")
        else:
            lines.append("People:\n- None visible")

        if result.object_counts:
            lines.append("\nObjects Detected:")
            for obj_name, count in result.object_counts.items():
                lines.append(f"- {obj_name}: {count}")
        else:
            lines.append("\nObjects Detected:\n- None")

        lines.append(f"\nVisual Summary: {result.summary_text}")
        lines.append("Use this visual information to answer the user's question accurately.")
        return "\n".join(lines)

    def register_person(
        self,
        name: str,
        num_samples: int = 3,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[bool, str]:
        """
        Captures multiple face samples from the camera, generates average face embeddings,
        and saves the identity into the SQLite face database.
        """
        clean_name = name.strip()
        if not clean_name:
            return False, "Invalid name provided for face registration."

        print(f"\n[VISION] Starting face registration for '{clean_name}' ({num_samples} samples required)...", flush=True)
        print("  Looking at the camera... Capturing face samples now.", flush=True)
        collected_embeddings: List[List[float]] = []

        start_time = time.time()
        timeout = 15.0  # 15 seconds timeout to collect samples
        last_hint_time = 0.0

        while len(collected_embeddings) < num_samples and (time.time() - start_time) < timeout:
            try:
                frame = self.camera.capture_frame()
                detections = self.face_detector.detect_faces(frame)

                if len(detections) >= 1:
                    # Select the most prominent face
                    det = max(detections, key=lambda d: d.confidence)
                    emb = self.face_recognizer.extract_embedding(frame, det)
                    if emb is not None:
                        collected_embeddings.append(emb)
                        if progress_callback:
                            progress_callback(len(collected_embeddings), num_samples)
                        print(f"  -> [Sample {len(collected_embeddings)}/{num_samples}] Face captured! (confidence: {det.confidence:.2f})", flush=True)
                        time.sleep(0.25)
                else:
                    if time.time() - last_hint_time > 2.0:
                        print("  [Hint] Looking for your face... Please look at the camera.", flush=True)
                        last_hint_time = time.time()
                time.sleep(0.05)
            except Exception as e:
                log_warning(f"Error during face sample capture: {e}")
                time.sleep(0.1)

        if len(collected_embeddings) == 0:
            return (
                False,
                f"Registration timed out. No face detected. "
                f"Please ensure good lighting and face the camera directly.",
            )

        # Average normalized embedding vectors
        arr = np.array(collected_embeddings, dtype=np.float32)
        mean_vec = np.mean(arr, axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm
        final_embedding = mean_vec.tolist()

        person_id = self.face_db.register_or_update_face(
            name=clean_name,
            embedding=final_embedding,
            sample_count=len(collected_embeddings),
            metadata={"registered_at": datetime.now(timezone.utc).isoformat()},
        )

        print(f"\n[VISION] Successfully registered '{clean_name}' (ID: {person_id}) with {len(collected_embeddings)} samples.", flush=True)
        return True, f"I have successfully registered {clean_name} in my face database."

    def delete_person(self, name: str) -> Tuple[bool, str]:
        """Deletes a registered person from the face database."""
        clean_name = name.strip()
        success = self.face_db.delete_face(clean_name)
        if success:
            log_chitti(f"[VISION] Removed '{clean_name}' from registered faces.")
            return True, f"I have removed {clean_name} from my registered faces."
        return False, f"I could not find anyone named {clean_name} in my face database."

    def list_registered_people(self) -> List[str]:
        """Returns a list of all registered person names."""
        return self.face_db.list_all_names()

    def handle_face_commands(self, user_text: str) -> Optional[Tuple[str, str]]:
        """
        Parses explicit face management commands (register person, delete person, list people).
        """
        text = user_text.strip()
        lower = text.lower()

        # Handle pending confirmation
        if self.pending_face_deletion:
            target_name = self.pending_face_deletion
            if any(w in lower for w in ["yes", "confirm", "proceed", "sure", "delete"]):
                self.pending_face_deletion = None
                success, msg = self.delete_person(target_name)
                return ("face_deleted", msg)
            elif any(w in lower for w in ["no", "cancel", "stop", "abort", "don't"]):
                self.pending_face_deletion = None
                return ("cancelled", f"Deletion of {target_name} cancelled.")
            else:
                return ("confirm_required", f"Please answer Yes to remove {target_name} from registered faces, or No to cancel.")

        # List registered faces
        if re.search(r"(?i)\b(?:who\s+is\s+registered|list\s+registered\s+(?:people|faces)|show\s+registered\s+faces)\b", text):
            people = self.list_registered_people()
            if people:
                return ("faces_listed", f"The following people are registered in my face database: {', '.join(people)}.")
            return ("faces_listed", "There are currently no people registered in my face database.")

        # Delete / Remove person
        match_del = re.search(r"(?i)\b(?:remove|delete|forget)\s+([a-zA-Z0-9_\s]+)\s+from\s+(?:face\s+recognition|registered\s+faces|faces)\b", text)
        if not match_del:
            match_del = re.search(r"(?i)\b(?:forget\s+face\s+of|delete\s+face\s+of)\s+([a-zA-Z0-9_\s]+)\b", text)

        if match_del:
            target = match_del.group(1).strip().rstrip(".!? \t\n")
            if target:
                self.pending_face_deletion = target
                return (
                    "confirm_required",
                    f"Are you sure you want to remove {target} from Chitti's registered faces? Please say Yes or No.",
                )

        # Register person voice command
        match_reg = re.search(r"(?i)\b(?:register|remember\s+face\s+as|my\s+name\s+is|register\s+person)\s+([a-zA-Z0-9_]+)\s*(?:as\s+(?:a\s+)?person)?\b", text)
        if match_reg and any(w in lower for w in ["register", "face"]):
            target_name = match_reg.group(1).strip().rstrip(".!? \t\n")
            if target_name and target_name.lower() not in ["a", "the", "person", "someone"]:
                success, msg = self.register_person(target_name, num_samples=3)
                return ("person_registered", msg)

        return None
