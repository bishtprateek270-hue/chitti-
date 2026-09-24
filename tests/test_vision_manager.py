"""Tests for Chitti Vision Manager."""

from unittest.mock import patch, MagicMock
import numpy as np
import pytest
from src.vision.vision_manager import VisionManager
from src.vision.models import (
    FaceDetection,
    RecognizedPerson,
    DetectedObject,
    VisionAnalysisResult,
)


def test_is_vision_query_detection():
    # Visual queries
    assert VisionManager.is_vision_query("What do you see?") is True
    assert VisionManager.is_vision_query("What can you see right now?") is True
    assert VisionManager.is_vision_query("Who is in front of you?") is True
    assert VisionManager.is_vision_query("Is anyone there?") is True
    assert VisionManager.is_vision_query("Do you recognize me?") is True
    assert VisionManager.is_vision_query("What objects are on the desk?") is True

    # Non-visual queries
    assert VisionManager.is_vision_query("What is machine learning?") is False
    assert VisionManager.is_vision_query("How is the weather today?") is False
    assert VisionManager.is_vision_query("Remember that I use Python.") is False


def test_analyze_frame_mock(tmp_path):
    db_file = tmp_path / "faces.db"

    # Setup mocks
    mock_cam = MagicMock()
    mock_cam.capture_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_detector = MagicMock()
    mock_det = FaceDetection(bbox=(10, 20, 100, 100), confidence=0.95, raw_detection=np.zeros(15))
    mock_detector.detect_faces.return_value = [mock_det]

    mock_recognizer = MagicMock()
    mock_recognizer.extract_embedding.return_value = [0.1] * 128
    mock_recognizer.match_identity.return_value = ("Prateek", 0.92, True, 1)

    mock_obj_detector = MagicMock()
    mock_obj = DetectedObject(class_name="laptop", confidence=0.89, bbox=(120, 100, 200, 150))
    mock_obj_detector.detect_objects.return_value = [mock_obj]

    manager = VisionManager(
        camera=mock_cam,
        face_detector=mock_detector,
        face_recognizer=mock_recognizer,
        object_detector=mock_obj_detector,
    )

    result = manager.analyze_frame()

    assert result.has_faces is True
    assert len(result.faces) == 1
    assert result.faces[0].name == "Prateek"
    assert result.faces[0].is_known is True

    assert result.has_objects is True
    assert result.object_counts == {"laptop": 1}
    assert "Prateek" in result.summary_text
    assert "laptop" in result.summary_text


def test_format_vision_context_for_llm():
    result = VisionAnalysisResult(
        timestamp="2026-09-24T12:00:00Z",
        faces=[
            RecognizedPerson(name="Prateek", confidence=0.95, is_known=True, bbox=(10, 20, 100, 100), similarity=0.91),
            RecognizedPerson(name="Unknown", confidence=0.90, is_known=False, bbox=(150, 20, 100, 100)),
        ],
        objects=[
            DetectedObject(class_name="laptop", confidence=0.92, bbox=(20, 200, 150, 100)),
            DetectedObject(class_name="bottle", confidence=0.85, bbox=(200, 200, 50, 100)),
        ],
        object_counts={"laptop": 1, "bottle": 1},
        summary_text="People detected: Prateek, 1 unrecognized person. Objects in view: 1 laptop, 1 bottle.",
    )

    manager = VisionManager(camera=MagicMock(), face_detector=MagicMock(), face_recognizer=MagicMock(), object_detector=MagicMock())
    context = manager.format_vision_context_for_llm(result)

    assert "CURRENT VISUAL PERCEPTION" in context
    assert "Prateek" in context
    assert "Unknown person" in context
    assert "laptop: 1" in context
    assert "bottle: 1" in context
