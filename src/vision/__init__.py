"""
Chitti Computer Vision Package (Phase 3).
Provides camera capture, face detection, face recognition, object detection, and multimodal scene analysis.
"""

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
from src.vision.vision_manager import VisionManager

__all__ = [
    "FaceDetection",
    "RecognizedPerson",
    "DetectedObject",
    "VisionAnalysisResult",
    "Camera",
    "CameraError",
    "FaceDetector",
    "FaceRecognizer",
    "FaceDatabase",
    "ObjectDetector",
    "VisionManager",
]
