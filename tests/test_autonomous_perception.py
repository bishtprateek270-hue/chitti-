"""
Unit & Integration Tests for Chitti Autonomous Visual Perception Engine.
Tests motion gating, face recognition triggering, cooldown management, and proactive greetings.
"""

import time
import unittest
from unittest.mock import MagicMock
import numpy as np

from src.vision.autonomous_perception import AutonomousPerceptionEngine
from src.vision.models import FaceDetection
from src.vision.vision_manager import VisionManager


class TestAutonomousPerceptionEngine(unittest.TestCase):
    """Test suite for AutonomousPerceptionEngine."""

    def setUp(self):
        self.mock_vision = MagicMock(spec=VisionManager)
        self.mock_camera = MagicMock()
        self.mock_face_detector = MagicMock()
        self.mock_face_recognizer = MagicMock()
        self.mock_face_db = MagicMock()

        self.mock_vision.camera = self.mock_camera
        self.mock_vision.face_detector = self.mock_face_detector
        self.mock_vision.face_recognizer = self.mock_face_recognizer
        self.mock_vision.face_db = self.mock_face_db

    def test_motion_gating_static_scene(self):
        """Identical consecutive frames should be detected as static (no motion)."""
        engine = AutonomousPerceptionEngine(
            vision_manager=self.mock_vision,
            motion_threshold=2.0,
        )

        frame1 = np.ones((480, 640, 3), dtype=np.uint8) * 100
        frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 100

        # First frame initializes baseline
        has_motion_1 = engine._has_motion(frame1)
        self.assertTrue(has_motion_1)

        # Identical second frame should report no motion
        has_motion_2 = engine._has_motion(frame2)
        self.assertFalse(has_motion_2)

    def test_motion_gating_moving_scene(self):
        """Altered frame should trigger motion detection."""
        engine = AutonomousPerceptionEngine(
            vision_manager=self.mock_vision,
            motion_threshold=2.0,
        )

        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 200

        engine._has_motion(frame1)
        has_motion = engine._has_motion(frame2)
        self.assertTrue(has_motion)

    def test_greeting_generation_known_and_unknown(self):
        """Contextual greeting generation for known vs unknown visitors."""
        engine = AutonomousPerceptionEngine(vision_manager=self.mock_vision)

        greeting_known = engine._generate_greeting("Prateek Singh Bisht", is_known=True)
        self.assertIn("Prateek", greeting_known)

        greeting_unknown = engine._generate_greeting("Unknown", is_known=False)
        self.assertIn("register face", greeting_unknown)

    def test_proactive_greeting_dispatch_with_cooldown(self):
        """Engine should dispatch greeting on face match and respect cooldown."""
        greeted_events = []

        def mock_callback(name, is_known, greeting):
            greeted_events.append((name, is_known, greeting))

        # Setup mock detections
        dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        self.mock_camera.capture_frame.return_value = dummy_frame

        det = FaceDetection(bbox=(50, 50, 100, 100), confidence=0.92)
        self.mock_face_detector.detect_faces.return_value = [det]
        self.mock_face_recognizer.extract_embedding.return_value = np.zeros(128, dtype=np.float32)
        self.mock_face_recognizer.match_identity.return_value = ("Prateek", 0.88, True, 1)
        self.mock_face_db.get_all_registered_faces.return_value = [(1, "Prateek", [0.0]*128)]

        engine = AutonomousPerceptionEngine(
            vision_manager=self.mock_vision,
            on_person_detected=mock_callback,
            poll_interval=0.05,
            greet_cooldown=5.0,  # 5 seconds cooldown
        )

        engine.start()
        # Allow background thread to process at least one frame
        time.sleep(0.3)

        self.assertGreaterEqual(len(greeted_events), 1)
        name, is_known, greeting = greeted_events[0]
        self.assertEqual(name, "Prateek")
        self.assertTrue(is_known)
        self.assertIn("Prateek", greeting)

        count_before = len(greeted_events)
        # Immediate second check should be suppressed by cooldown
        time.sleep(0.2)
        self.assertEqual(len(greeted_events), count_before)

        engine.stop()
        self.assertFalse(engine._is_running)


if __name__ == "__main__":
    unittest.main()
