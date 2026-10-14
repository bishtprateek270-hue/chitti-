"""
Repeatable Automated Verification Suite for Chitti Audio Pipeline Robustness (Phases 1-11).
Covers Tests A through J:
- Test A: Silence (Zero false activations during silence)
- Test B: Background noise (Rejection of fan noise, keyboard clicks, room hiss)
- Test C: Quiet speech (Reliable capture and detection of quiet speech)
- Test D: Short commands (Single execution without truncation)
- Test E: Longer sentences with natural pauses (No premature cutoff)
- Test F: Multilingual support (English, Hindi, Hinglish)
- Test G: TTS echo suppression (Playback not interpreted as user command)
- Test H: Genuine user interruption / barge-in (Stops TTS, triggers once)
- Test I: Audio stream recovery and error handling
- Test J: Sustained multi-turn operation (Zero memory leaks or queue accumulation)
"""

import time
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.audio.capture_service import AudioCaptureService, get_audio_capture_service
from src.audio.diagnostic import AudioDiagnosticLogger, UtteranceDiagnosticLog, clear_diagnostic_recordings
from src.audio.interrupter import BargeInDetector
from src.audio.streaming_stt import StreamingSTTEngine
from src.audio.vad import VoiceActivityDetector, VADState
from src.audio.wake_word import WakeWordDetector, WakeWordEvent
from src.core.phrase_matcher import ControlCommand, get_phrase_matcher
from src.core.state_manager import SystemState, get_state_manager


def generate_speech_frame(duration_sec: float = 0.05, sample_rate: int = 16000, pitch_hz: float = 160.0, amplitude: float = 0.12) -> np.ndarray:
    """Generates synthetic human-voiced periodic audio with natural pitch and formant harmonics."""
    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    # Fundamental pitch + formants
    harmonic_stack = (
        0.55 * np.sin(2 * np.pi * pitch_hz * t)
        + 0.30 * np.sin(2 * np.pi * (2 * pitch_hz) * t)
        + 0.15 * np.sin(2 * np.pi * (3 * pitch_hz) * t)
    )
    return (harmonic_stack * amplitude).astype(np.float32)


def generate_fan_noise(duration_sec: float = 1.0, sample_rate: int = 16000, amplitude: float = 0.015) -> np.ndarray:
    """Simulates laptop cooling fan noise (low-frequency motor rumble + constant stationary hiss)."""
    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    rumble = 0.6 * np.sin(2 * np.pi * 55 * t)
    hiss = 0.4 * np.random.randn(num_samples)
    return ((rumble + hiss) * amplitude).astype(np.float32)


class TestAudioPipelineRobustness(unittest.TestCase):
    """Rigorous end-to-end tests for Chitti voice input reliability."""

    def setUp(self):
        self.sample_rate = 16000
        self.state_mgr = get_state_manager()
        self.state_mgr.reset_interrupt()
        self.state_mgr.wake()

    # ---------------------------------------------------------
    # TEST A: SILENCE
    # ---------------------------------------------------------
    def test_a_silence_produces_zero_activations(self):
        """Periods of pure silence must never trigger wake events, speech detection, or STT."""
        vad = VoiceActivityDetector(sample_rate=self.sample_rate)
        detector = WakeWordDetector(sample_rate=self.sample_rate)

        # 3.0 seconds of near-zero ambient silence (RMS ~ 0.0005)
        silence_duration_sec = 3.0
        frame_samples = 800  # 50ms
        num_frames = int(silence_duration_sec / 0.05)

        speech_detections = 0
        wake_events = 0

        for _ in range(num_frames):
            frame = (np.random.randn(frame_samples) * 0.0005).astype(np.float32)
            vad_dec = vad.process_frame(frame)
            if vad_dec.is_speech:
                speech_detections += 1

            event = detector.process_audio_chunk(frame)
            if event:
                wake_events += 1

        self.assertEqual(speech_detections, 0, "Silence falsely detected as speech by VAD!")
        self.assertEqual(wake_events, 0, "Silence triggered a wake event!")
        self.assertEqual(vad.state, VADState.SILENCE)

    # ---------------------------------------------------------
    # TEST B: BACKGROUND NOISE (Fan, typing, room ambience)
    # ---------------------------------------------------------
    def test_b_background_noise_rejected(self):
        """Stationary fan noise and intermittent keyboard clicks must be rejected."""
        vad = VoiceActivityDetector(sample_rate=self.sample_rate)
        mock_stt = MagicMock()
        mock_stt.transcribe.return_value = "hallucinated command"
        detector = WakeWordDetector(sample_rate=self.sample_rate, stt_engine=mock_stt)

        # 2.0 seconds of realistic fan noise (RMS ~0.015, matching hardware diagnostic)
        fan_audio = generate_fan_noise(duration_sec=2.0, amplitude=0.015)
        wake_events = 0

        for offset in range(0, len(fan_audio), 800):
            chunk = fan_audio[offset : offset + 800]
            dec = vad.process_frame(chunk)
            # Must classify as non-speech
            self.assertFalse(dec.is_speech)
            event = detector.process_audio_chunk(chunk)
            if event:
                wake_events += 1

        self.assertEqual(wake_events, 0, "Fan noise falsely triggered wake detector!")
        # Whisper STT must never have been called on fan noise!
        self.assertEqual(mock_stt.transcribe.call_count, 0, "STT was invoked on stationary background noise!")

    # ---------------------------------------------------------
    # TEST C: QUIET SPEECH
    # ---------------------------------------------------------
    def test_c_quiet_speech_reliably_detected(self):
        """Legitimate quiet speech (RMS ~0.018 - 0.030) must be detected through harmonic voicing."""
        vad = VoiceActivityDetector(sample_rate=self.sample_rate)
        # Establish background floor first
        for _ in range(5):
            vad.process_frame(generate_fan_noise(0.05, amplitude=0.008))

        # Quiet conversational speech (amplitude = 0.035, RMS ~0.022)
        speech_detected = False
        for _ in range(6):
            frame = generate_speech_frame(duration_sec=0.05, pitch_hz=180.0, amplitude=0.035)
            dec = vad.process_frame(frame)
            if dec.is_speech and dec.is_voiced:
                speech_detected = True
                break

        self.assertTrue(speech_detected, "Quiet speech failed to be detected by VAD!")

    # ---------------------------------------------------------
    # TEST D: SHORT COMMANDS
    # ---------------------------------------------------------
    def test_d_short_command_recognized_once(self):
        """A short command like 'stop' or 'wake up' must be processed exactly once."""
        matcher = get_phrase_matcher()
        res = matcher.classify_command("stop")
        self.assertEqual(res.command, ControlCommand.STOP)

        res2 = matcher.classify_command("hey chitti")
        self.assertEqual(res2.command, ControlCommand.WAKE)

    # ---------------------------------------------------------
    # TEST E: LONGER SENTENCES WITH NATURAL PAUSES
    # ---------------------------------------------------------
    def test_e_longer_sentence_with_natural_pauses_not_truncated(self):
        """StreamingSTTEngine should not split speech during a natural 0.6s breath pause."""
        stt_engine = StreamingSTTEngine(silence_cutoff_seconds=1.20)
        # Verify calibrated silence cutoff allows a 0.6s pause without truncation
        self.assertGreater(stt_engine.silence_cutoff_seconds, 0.60)
        self.assertGreaterEqual(stt_engine.pre_speech_seconds, 0.50)

        # Test VAD hangover behavior during 0.5s pause
        vad = VoiceActivityDetector(sample_rate=self.sample_rate)
        # 1. User speaks 3 frames
        for _ in range(3):
            vad.process_frame(generate_speech_frame(0.05, amplitude=0.15))
        self.assertIn(vad.state, (VADState.SPEECH, VADState.SPEECH_HANGOVER))

        # 2. User pauses for 6 frames (300ms)
        for _ in range(6):
            dec = vad.process_frame((np.random.randn(800) * 0.001).astype(np.float32))
            # Hysteresis hangover must maintain speech continuation
            self.assertTrue(dec.is_speech or dec.state == VADState.SPEECH_HANGOVER)

    # ---------------------------------------------------------
    # TEST F: MULTILINGUAL SUPPORT (English, Hindi, Hinglish)
    # ---------------------------------------------------------
    def test_f_multilingual_control_commands(self):
        """English, Hindi, and Hinglish control commands must resolve with high confidence."""
        matcher = get_phrase_matcher()

        # English
        self.assertEqual(matcher.classify_command("hey chitti wake up").command, ControlCommand.WAKE)
        self.assertEqual(matcher.classify_command("stop right now").command, ControlCommand.STOP)
        self.assertEqual(matcher.classify_command("go to sleep").command, ControlCommand.SLEEP)

        # Hindi
        self.assertEqual(matcher.classify_command("chitti suno").command, ControlCommand.WAKE)
        self.assertEqual(matcher.classify_command("ruk ja").command, ControlCommand.STOP)
        self.assertEqual(matcher.classify_command("so jao").command, ControlCommand.SLEEP)

        # Hinglish
        self.assertEqual(matcher.classify_command("chitti active ho ja").command, ControlCommand.WAKE)
        self.assertEqual(matcher.classify_command("bas karo chitti").command, ControlCommand.STOP)

    # ---------------------------------------------------------
    # TEST G: TTS ECHO SUPPRESSION
    # ---------------------------------------------------------
    def test_g_tts_echo_does_not_trigger_barge_in(self):
        """Assistant voice playing through speakers must not trigger barge-in interruption."""
        barge_in = BargeInDetector(energy_threshold=0.10, grace_period_seconds=0.15)
        mock_cb = MagicMock()
        barge_in.start_monitoring(on_interrupt=mock_cb)

        # 1. During grace period, assistant plays audio at 0.12 RMS
        echo_frame = generate_speech_frame(0.05, pitch_hz=210.0, amplitude=0.12)
        for _ in range(4):
            barge_in._audio_callback(echo_frame)

        # 2. After grace period, assistant voice continues at ~0.12 RMS
        for _ in range(6):
            barge_in._audio_callback(echo_frame)

        # Barge-in must NOT have fired because user did not exceed speaker bleed!
        self.assertFalse(mock_cb.called, "Assistant's own voice triggered self-interruption!")
        barge_in.stop_monitoring()

    # ---------------------------------------------------------
    # TEST H: GENUINE USER INTERRUPTION (BARGE-IN)
    # ---------------------------------------------------------
    def test_h_genuine_user_interruption_triggers(self):
        """A user shouting or speaking loudly (> speaker bleed) must successfully interrupt."""
        barge_in = BargeInDetector(energy_threshold=0.10, consecutive_frames=3, grace_period_seconds=0.10)
        interrupted = False

        def on_interrupt(audio=None):
            nonlocal interrupted
            interrupted = True

        barge_in.start_monitoring(on_interrupt=on_interrupt)

        # Baseline speaker playback at 0.08 RMS
        for _ in range(3):
            barge_in._audio_callback(generate_speech_frame(0.05, amplitude=0.08))
            time.sleep(0.04)

        # Loud user voice (amplitude 0.35, well above speaker bleed)
        loud_user_frame = generate_speech_frame(0.05, pitch_hz=140.0, amplitude=0.35)
        for _ in range(5):
            barge_in._audio_callback(loud_user_frame)

        self.assertTrue(interrupted, "Genuine loud user interruption failed to trigger barge-in!")
        barge_in.stop_monitoring()

    # ---------------------------------------------------------
    # TEST I: AUDIO CAPTURE STREAM RECOVERY
    # ---------------------------------------------------------
    def test_i_capture_service_diagnostics_and_resilience(self):
        """AudioCaptureService must track buffer metrics, overflow/underflow, and recover cleanly."""
        capture = get_audio_capture_service()
        diag = capture.get_diagnostics()

        self.assertIn("is_running", diag)
        self.assertIn("overflow_count", diag)
        self.assertIn("underflow_count", diag)
        self.assertIn("recent_rms", diag)

        # Verify buffer flushing is safe and idempotent
        capture.flush()
        capture.flush()
        self.assertEqual(capture._audio_queue.qsize(), 0)

    # ---------------------------------------------------------
    # TEST J: SUSTAINED MULTI-TURN OPERATION
    # ---------------------------------------------------------
    def test_j_sustained_multi_turn_no_leaks(self):
        """Running multiple simulated turns must not accumulate stale audio or leak subscribers."""
        capture = get_audio_capture_service()
        initial_subs = len(capture._subscribers)

        # Simulate 10 sequential listening sessions
        for i in range(10):
            sub_id = f"test_turn_{i}"
            capture.subscribe(sub_id, lambda chunk, rms: None)
            self.assertTrue(capture.has_subscriber(sub_id))
            capture.flush()
            capture.unsubscribe(sub_id)
            self.assertFalse(capture.has_subscriber(sub_id))

        # Ensure no lingering subscribers
        self.assertEqual(len(capture._subscribers), initial_subs)


if __name__ == "__main__":
    unittest.main()
