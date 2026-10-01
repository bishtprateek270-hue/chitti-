"""
Comprehensive Unit & Integration Test Suite for Phase 5:
Ambient Wake-Word & Low-Latency Hands-Free Voice Engine.
"""

import time
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.audio.ambient_listener import AmbientState, AmbientVoiceListener
from src.audio.interrupter import BargeInDetector
from src.audio.streaming_stt import StreamingSTTEngine
from src.audio.tts import TTSEngine
from src.audio.wake_word import WakeWordDetector, WakeWordEvent
from src.ui.hud_overlay import HUDMode


class TestVoiceEnginePhase5(unittest.TestCase):
    """Tests all Phase 5 modules: wake-word spotting, streaming STT, barge-in, and ambient loop."""

    def setUp(self):
        self.sample_rate = 16000

    def test_wake_word_detector_silence_no_trigger(self):
        """Pure silence or low noise should never trigger a wake word."""
        detector = WakeWordDetector(sample_rate=self.sample_rate, sensitivity=0.6)
        silence_chunk = np.zeros(1280, dtype=np.float32)
        event = detector.process_audio_chunk(silence_chunk)
        self.assertIsNone(event)

    def test_wake_word_detector_ambient_noise_no_trigger(self):
        """Continuous fan / background static noise should never trigger a wake word."""
        detector = WakeWordDetector(sample_rate=self.sample_rate, sensitivity=0.7)
        # 1.5 seconds of static/fan noise
        noise = np.random.uniform(-0.035, 0.035, int(self.sample_rate * 1.5)).astype(np.float32)
        for i in range(0, len(noise), 1280):
            chunk = noise[i : i + 1280]
            if len(chunk) == 1280:
                event = detector.process_audio_chunk(chunk)
                self.assertIsNone(event)

    def test_wake_word_detector_acoustic_chitti_trigger(self):
        """Synthesized acoustic pattern matching 'Chitti' should fire a WakeWordEvent."""
        detector = WakeWordDetector(sample_rate=self.sample_rate, sensitivity=0.85, cooldown_seconds=0.1)

        # Build acoustic pattern: noise + bursts
        pattern = np.zeros(int(self.sample_rate * 0.8), dtype=np.float32)
        seg_len = len(pattern) // 4

        # Seg 0: high-frequency fricative noise
        pattern[0:seg_len] = np.random.uniform(-0.15, 0.15, seg_len).astype(np.float32)
        # Seg 1: vowel harmonic (220 Hz pitch)
        t = np.linspace(0, 0.2, seg_len)
        pattern[seg_len : 2 * seg_len] = (0.25 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        # Seg 2: stop closure
        pattern[2 * seg_len : 3 * seg_len] = 0.005
        # Seg 3: vowel harmonic (250 Hz pitch)
        pattern[3 * seg_len :] = (0.25 * np.sin(2 * np.pi * 250 * t)).astype(np.float32)

        event = detector.process_audio_chunk(pattern)
        self.assertIsNotNone(event)
        self.assertIn("Chitti", event.keyword)
        self.assertGreaterEqual(event.confidence, 0.5)

    def test_wake_word_cooldown(self):
        """Wake-word detector should respect cooldown window between triggers."""
        detector = WakeWordDetector(sample_rate=self.sample_rate, sensitivity=0.9, cooldown_seconds=2.0)
        # Artificially set recent trigger
        detector._last_trigger_time = time.time()
        chunk = np.ones(1280, dtype=np.float32) * 0.2
        event = detector.process_audio_chunk(chunk)
        self.assertIsNone(event)

    def test_barge_in_detector_energy_trigger(self):
        """BargeInDetector should fire interrupt callback when loud speech arrives."""
        interrupted = False

        def on_interrupt():
            nonlocal interrupted
            interrupted = True

        detector = BargeInDetector(sample_rate=self.sample_rate, energy_threshold=0.03, consecutive_frames=2)
        loud_chunk = np.ones(800, dtype=np.float32) * 0.1

        # Frame 1
        res1 = detector.process_frame(loud_chunk)
        self.assertFalse(res1)
        # Frame 2
        res2 = detector.process_frame(loud_chunk)
        self.assertTrue(res2)

    def test_streaming_stt_transcription_with_mock(self):
        """StreamingSTTEngine should process audio and invoke STTEngine."""
        mock_stt = MagicMock()
        mock_stt.transcribe.return_value = "open calculator"

        engine = StreamingSTTEngine(stt_engine=mock_stt)
        dummy_audio = np.ones(16000, dtype=np.float32) * 0.05

        with patch.object(engine, "record_until_silence", return_value=dummy_audio):
            text = engine.transcribe_streaming()
            self.assertEqual(text, "open calculator")
            mock_stt.transcribe.assert_called_once()

    def test_ambient_voice_listener_lifecycle(self):
        """AmbientVoiceListener should execute the full state machine on wake word."""
        handled_commands = []

        def mock_handler(cmd):
            handled_commands.append(cmd)

        mock_tts = MagicMock(spec=TTSEngine)
        mock_hud = MagicMock()

        mock_stt_engine = MagicMock(spec=StreamingSTTEngine)
        mock_stt_engine.transcribe_streaming.return_value = "what is my next task"

        mock_wake = MagicMock(spec=WakeWordDetector)

        listener = AmbientVoiceListener(
            command_handler=mock_handler,
            tts_engine=mock_tts,
            streaming_stt=mock_stt_engine,
            wake_detector=mock_wake,
            hud=mock_hud,
        )

        listener.start()
        self.assertEqual(listener.state, AmbientState.STANDBY)
        mock_wake.start.assert_called_once()
        mock_hud.set_mode.assert_called_with(HUDMode.IDLE)

        # Simulate wake word spotted
        event = WakeWordEvent(keyword="Hey Chitti", confidence=0.95, timestamp=time.time())
        listener._on_wake_word_triggered(event)

        # Allow worker thread to run
        time.sleep(0.3)

        self.assertIn("what is my next task", handled_commands)
        self.assertEqual(listener.state, AmbientState.STANDBY)

        # Test speak with barge in
        listener.speak_with_barge_in("Your next task is completing Phase 5.")
        mock_tts.speak.assert_called_with("Your next task is completing Phase 5.", wait=True)

        listener.stop()
        mock_wake.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
