"""
Chitti Vision Data Models.
Defines structured classes for face detections, recognized people, detected objects, and vision analysis snapshots.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class FaceDetection:
    """Represents a detected face location in an image."""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    landmarks: Optional[List[Tuple[float, float]]] = None
    raw_detection: Optional[Any] = None


@dataclass
class RecognizedPerson:
    """Represents an identified or unknown person."""
    name: str
    confidence: float
    is_known: bool
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    person_id: Optional[int] = None
    similarity: float = 0.0


@dataclass
class DetectedObject:
    """Represents a detected object in a frame."""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)


@dataclass
class VisionAnalysisResult:
    """Comprehensive snapshot of visual analysis."""
    timestamp: str
    faces: List[RecognizedPerson] = field(default_factory=list)
    objects: List[DetectedObject] = field(default_factory=list)
    object_counts: Dict[str, int] = field(default_factory=dict)
    summary_text: str = ""
    frame: Optional[np.ndarray] = None

    @property
    def has_faces(self) -> bool:
        return len(self.faces) > 0

    @property
    def has_objects(self) -> bool:
        return len(self.objects) > 0

    @property
    def known_people_names(self) -> List[str]:
        return [f.name for f in self.faces if f.is_known]

    @property
    def unknown_people_count(self) -> int:
        return sum(1 for f in self.faces if not f.is_known)
