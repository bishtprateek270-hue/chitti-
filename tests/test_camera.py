"""Tests for Chitti Camera Module."""

from unittest.mock import patch, MagicMock
import numpy as np
import pytest
from src.vision.camera import Camera, CameraError


def test_camera_init():
    cam = Camera(device_index=0, width=640, height=480, fps=30)
    assert cam.device_index == 0
    assert cam.width == 640
    assert cam.height == 480
    assert cam.fps == 30
    assert not cam.is_opened()


def test_camera_list_available_devices():
    with patch("src.vision.camera.cv2") as mock_cv2:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_DSHOW = 700

        cams = Camera.list_available_cameras(max_devices_to_test=2)
        assert len(cams) == 2
        assert cams == [0, 1]


def test_camera_open_and_capture_mock():
    cam = Camera(device_index=0)
    mock_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128

    with patch("src.vision.camera.cv2") as mock_cv2:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, mock_frame)
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_DSHOW = 700

        cam.open()
        assert cam.is_opened()

        frame = cam.capture_frame()
        assert frame is not None
        assert frame.shape == (480, 640, 3)

        cam.release()
        assert not cam.is_opened()


def test_camera_open_failure_raises_camera_error():
    cam = Camera(device_index=99)
    with patch("src.vision.camera.cv2") as mock_cv2:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_DSHOW = 700

        with pytest.raises(CameraError):
            cam.open()
