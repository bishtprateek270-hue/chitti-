"""Tests for Chitti Object Detector."""

from unittest.mock import patch, MagicMock
import numpy as np
import pytest
from src.vision.object_detector import ObjectDetector
from src.vision.models import DetectedObject


def test_object_detector_empty_frame():
    detector = ObjectDetector()
    assert detector.detect_objects(None) == []
    empty_frame = np.zeros(0, dtype=np.uint8)
    assert detector.detect_objects(empty_frame) == []


def test_object_detector_mock_inference():
    detector = ObjectDetector(confidence_threshold=0.30)

    # Mock Ultralytics YOLO output
    mock_box1 = MagicMock()
    mock_box1.cls = [MagicMock(item=lambda: 0)]
    mock_box1.conf = [MagicMock(item=lambda: 0.92)]
    mock_box1.xyxy = [MagicMock(cpu=lambda: MagicMock(numpy=lambda: np.array([10, 20, 110, 220])))]

    mock_box2 = MagicMock()
    mock_box2.cls = [MagicMock(item=lambda: 63)]
    mock_box2.conf = [MagicMock(item=lambda: 0.88)]
    mock_box2.xyxy = [MagicMock(cpu=lambda: MagicMock(numpy=lambda: np.array([150, 200, 350, 400])))]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box1, mock_box2]
    mock_result.names = {0: "person", 63: "laptop"}

    mock_yolo = MagicMock()
    mock_yolo.predict.return_value = [mock_result]
    detector._model = mock_yolo

    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    objects = detector.detect_objects(test_frame)

    assert len(objects) == 2
    assert objects[0].class_name == "person"
    assert objects[0].confidence == 0.92
    assert objects[0].bbox == (10, 20, 100, 200)

    assert objects[1].class_name == "laptop"
    assert objects[1].confidence == 0.88
    assert objects[1].bbox == (150, 200, 200, 200)
