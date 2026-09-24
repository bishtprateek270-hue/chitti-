"""Tests for Chitti Face Detector."""

from unittest.mock import patch, MagicMock
import numpy as np
from src.vision.face_detector import FaceDetector


def test_face_detector_detect_faces_empty_frame():
    detector = FaceDetector()
    assert detector.detect_faces(None) == []
    empty_frame = np.zeros(0, dtype=np.uint8)
    assert detector.detect_faces(empty_frame) == []


def test_face_detector_detect_faces_mock_yunet():
    detector = FaceDetector()

    # Create dummy face output: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rc, y_rc, x_lc, y_lc, score]
    mock_face = np.array([50, 60, 100, 120, 70, 80, 110, 80, 90, 100, 75, 130, 105, 130, 0.95], dtype=np.float32)
    mock_detector = MagicMock()
    mock_detector.detect.return_value = (1, np.array([mock_face]))
    detector._detector = mock_detector

    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect_faces(test_frame)

    assert len(detections) == 1
    det = detections[0]
    assert det.bbox == (50, 60, 100, 120)
    assert pytest_approx_equal(det.confidence, 0.95)
    assert len(det.landmarks) == 5


def pytest_approx_equal(a, b, tol=1e-3):
    return abs(a - b) < tol
