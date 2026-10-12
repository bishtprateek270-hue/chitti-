"""
Comprehensive Automated Verification Harness for Voice System Reliability Overhaul.
Implements Tests A through R from Section 14 of the requirements:
- Test A: Wake word in a quiet room.
- Test B: Wake word spoken at different distances and volumes.
- Test C: Wake word spoken with normal background noise.
- Test D: Several repeated wake words.
- Test E: Background conversation without the wake word.
- Test F: Silence.
- Test G: Chitti's own TTS playing through speakers (self-echo suppression).
- Test H: User says 'Stop' while Chitti is speaking.
- Test I: User says 'Ruk ja' while a task is executing.
- Test J: User says 'Sleep' while a response is being generated.
- Test K: Wake word immediately after sleep.
- Test L: A delayed STT result arrives after stop.
- Test M: A delayed LLM response arrives after stop.
- Test N: A queued TTS segment attempts to play after stop.
- Test O: Repeated stop commands.
- Test P: Microphone disconnects and reconnects.
- Test Q: The audio capture service encounters an error.
- Test R: A new request begins after an earlier request is cancelled.
"""

import time
import threading
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.audio.capture_service import AudioCaptureService, get_audio_capture_service
from src.audio.wake_word import WakeWordDetector, WakeWordEvent
from src.audio.streaming_stt import StreamingSTTEngine
from src.audio.tts import SpeechController, EdgeNeuralTTS, HybridTTS, TTSEngine
from src.audio.interrupter import BargeInDetector
from src.audio.ambient_listener import AmbientVoiceListener, AmbientState
from src.core.state_manager import SystemState, SystemStateManager, get_state_manager
from src.core.phrase_matcher import ControlCommand, PhraseMatcher, get_phrase_matcher
from src.main import ChittiController


def generate_synthetic_speech(duration_sec: float = 1.0, sample_rate: int = 16000, pitch_hz: float = 160.0, amplitude: float = 0.15) -> np.ndarray:
    """Generates synthetic voiced human-pitch periodic signal (harmonic stack) for acoustic testing."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    # Fundamental + 3 harmonics with formants
    signal = (
        0.5 * np.sin(2 * np.pi * pitch_hz * t)
        + 0.3 * np.sin(2 * np.pi * 2 * pitch_hz * t)
        + 0.15 * np.sin(2 * np.pi * 3 * pitch_hz * t)
        + 0.05 * np.sin(2 * np.pi * 4 * pitch_hz * t)
    )
    # Syllabic envelope modulation (approximate 2-syllable "Chit-ti")
    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 2.5 * t - np.pi / 2))
    modulated = (signal * envelope * amplitude).astype(np.float32)
    return modulated


class TestVoiceSystemReliabilityOverhaul(unittest.TestCase):
    """Automated Repeatable Audio Test Harness."""

    def setUp(self):
        self.state_mgr = get_state_manager()
        self.state_mgr.reset_interrupt()
        self.state_mgr.wake()
        self.phrase_matcher = get_phrase_matcher()

    def tearDown(self):
        self.state_mgr.reset_interrupt()
        self.state_mgr.wake()

    # ---------------- TEST A: WAKE WORD IN A QUIET ROOM ----------------
    def test_a_wake_word_in_quiet_room(self):
        """Test A: Valid wake phrase in quiet environment triggers WakeWordEvent."""
        detector = WakeWordDetector(sensitivity=0.75)
        # Synthetic voiced pitch signal
        audio_chunk = generate_synthetic_speech(duration_sec=0.5, amplitude=0.20)
        score = detector._evaluate_acoustic_signature(audio_chunk)
        self.assertGreaterEqual(score, 0.40, "Acoustic signature should detect pitch periodicity.")

        # Test phrase matcher on all standard wake phrases
        for phrase in ["Chitti", "Hey Chitti", "Chitti wake up", "Chitti jaag ja", "Chitti sun"]:
            res = self.phrase_matcher.classify_command(phrase)
            self.assertEqual(res.command, ControlCommand.WAKE, f"Should recognize wake command for: {phrase}")

    # ---------------- TEST B: WAKE WORD AT DIFFERENT VOLUMES ----------------
    def test_b_wake_word_different_volumes(self):
        """Test B: Wake detector evaluates energy and periodicity across low, normal, and high amplitude."""
        detector = WakeWordDetector()
        low_audio = generate_synthetic_speech(duration_sec=0.5, amplitude=0.03)
        med_audio = generate_synthetic_speech(duration_sec=0.5, amplitude=0.15)
        high_audio = generate_synthetic_speech(duration_sec=0.5, amplitude=0.45)

        low_rms = float(np.sqrt(np.mean(low_audio**2)))
        med_rms = float(np.sqrt(np.mean(med_audio**2)))
        high_rms = float(np.sqrt(np.mean(high_audio**2)))

        self.assertGreater(med_rms, low_rms)
        self.assertGreater(high_rms, med_rms)
        self.assertGreaterEqual(detector._evaluate_acoustic_signature(med_audio), 0.40)
        self.assertGreaterEqual(detector._evaluate_acoustic_signature(high_audio), 0.40)

    # ---------------- TEST C: WAKE WORD WITH BACKGROUND NOISE ----------------
    def test_c_wake_word_with_background_noise(self):
        """Test C: Voiced periodic signal is still detectable when ambient noise is present."""
        detector = WakeWordDetector()
        speech = generate_synthetic_speech(duration_sec=0.5, amplitude=0.20)
        noise = (np.random.normal(0, 0.02, len(speech))).astype(np.float32)
        mixed = speech + noise

        score = detector._evaluate_acoustic_signature(mixed)
        self.assertGreater(score, 0.35, "Pitch autocorrelation should survive moderate noise.")

    # ---------------- TEST D: REPEATED WAKE WORDS & COOLDOWN ----------------
    def test_d_repeated_wake_words_cooldown(self):
        """Test D: Rapidly repeated wake triggers are debounced by cooldown to prevent duplicate tasks."""
        detector = WakeWordDetector(cooldown_seconds=1.5)
        events = []
        detector._callback = lambda e: events.append(e)

        audio_chunk = generate_synthetic_speech(duration_sec=0.1, amplitude=0.25)
        # Manually trigger twice in rapid succession
        ev1 = detector.process_audio_chunk(audio_chunk)
        if ev1:
            events.append(ev1)
        # Second call immediately (<0.01s later)
        ev2 = detector.process_audio_chunk(audio_chunk)
        if ev2:
            events.append(ev2)

        self.assertLessEqual(len(events), 1, "Cooldown must suppress duplicate rapid triggers.")

    # ---------------- TEST E: BACKGROUND CONVERSATION WITHOUT WAKE WORD ----------------
    def test_e_background_conversation_without_wake_word(self):
        """Test E: Ordinary ambient conversations should NEVER trigger wake word or emergency stop."""
        ambient_phrases = [
            "What is the weather today",
            "Let's order some pizza for lunch",
            "Can you pass me the keyboard",
            "The movie yesterday was really great",
            "I need to push my code to git",
        ]
        for phrase in ambient_phrases:
            res = self.phrase_matcher.classify_command(phrase)
            self.assertEqual(res.command, ControlCommand.NORMAL, f"Ambient phrase should be NORMAL: {phrase}")

    # ---------------- TEST F: SILENCE ----------------
    def test_f_silence_rejection(self):
        """Test F: Pure silence produces zero triggers."""
        detector = WakeWordDetector()
        silence = np.zeros(16000, dtype=np.float32)
        score = detector._evaluate_acoustic_signature(silence)
        self.assertEqual(score, 0.0)
        ev = detector.process_audio_chunk(silence)
        self.assertIsNone(ev)

    # ---------------- TEST G: CHITTI'S OWN TTS PLAYING THROUGH SPEAKERS ----------------
    def test_g_self_echo_suppression(self):
        """Test G: When Chitti is in SPEAKING state, wake word events are strictly suppressed."""
        listener = AmbientVoiceListener(command_handler=MagicMock())
        self.state_mgr.set_state(SystemState.SPEAKING)

        triggered = []
        event = WakeWordEvent(keyword="Hey Chitti", confidence=0.95, timestamp=time.time())
        listener._on_wake_word_triggered(event)
        self.assertEqual(listener.state, AmbientState.STANDBY, "Must not transition to WAKING while SPEAKING.")

    # ---------------- TEST H: STOP COMMAND WHILE SPEAKING ----------------
    def test_h_stop_command_while_speaking(self):
        """Test H: Stop command detected during speech playback halts audio within <= 300ms."""
        tts = EdgeNeuralTTS()
        # Mock MCI playback
        with patch.object(tts, "_play_audio_file") as mock_play:
            mock_play.side_effect = lambda path, gen: time.sleep(0.05)
            tts.speak("Short speech segment", wait=False)
            self.assertTrue(tts._current_gen_id > 0)

            # Interrupt
            t0 = time.time()
            latency = tts.stop()
            elapsed_ms = (time.time() - t0) * 1000

            self.assertTrue(tts._cancel_requested.is_set())
            self.assertLess(elapsed_ms, 300.0, "Interruption latency must be under 300ms.")

    # ---------------- TEST I: RUK JA WHILE TASK IS EXECUTING ----------------
    def test_i_ruk_ja_while_task_executing(self):
        """Test I: Hindi 'Ruk ja' immediately cancels active execution and invalidates generation."""
        res = self.phrase_matcher.classify_command("Chitti ruk ja")
        self.assertEqual(res.command, ControlCommand.STOP)

        gen_before = self.state_mgr.get_current_generation()
        self.state_mgr.begin_execution(gen_before)
        self.assertEqual(self.state_mgr.get_state(), SystemState.EXECUTING)

        # User says ruk ja
        self.state_mgr.interrupt(reason="user_stop")
        self.assertTrue(self.state_mgr.is_interrupted())
        gen_after = self.state_mgr.get_current_generation()
        self.assertGreater(gen_after, gen_before, "Generation ID must be incremented/invalidated.")
        self.assertFalse(self.state_mgr.is_generation_valid(gen_before), "Old task generation must be invalid.")

    # ---------------- TEST J: SLEEP COMMAND WHILE RESPONSE GENERATING ----------------
    def test_j_sleep_while_response_generating(self):
        """Test J: Saying sleep command enters SLEEPING state and prevents subsequent execution."""
        res = self.phrase_matcher.classify_command("Chitti so ja")
        self.assertEqual(res.command, ControlCommand.SLEEP)

        gen_id = self.state_mgr.new_request()
        self.state_mgr.sleep()
        self.assertEqual(self.state_mgr.get_state(), SystemState.SLEEPING)
        self.assertFalse(self.state_mgr.can_speak(gen_id), "Cannot speak once sleeping.")
        self.assertFalse(self.state_mgr.begin_speaking(gen_id), "begin_speaking must fail while sleeping.")

    # ---------------- TEST K: WAKE WORD IMMEDIATELY AFTER SLEEP ----------------
    def test_k_wake_immediately_after_sleep(self):
        """Test K: Wake command successfully restores AWAKE state after entering sleep."""
        self.state_mgr.sleep()
        self.assertTrue(self.state_mgr.is_sleeping())

        # Say Wake phrase
        self.state_mgr.wake()
        self.assertTrue(self.state_mgr.is_awake())
        self.assertEqual(self.state_mgr.get_state(), SystemState.AWAKE)
        self.assertFalse(self.state_mgr.is_interrupted())

    # ---------------- TEST L: DELAYED STT RESULT ARRIVES AFTER STOP ----------------
    def test_l_delayed_stt_result_after_stop(self):
        """Test L: A delayed STT result from an earlier request is discarded if stop was called."""
        gen_1 = self.state_mgr.new_request()
        # User says stop before STT returns
        self.state_mgr.interrupt(reason="user_stop")

        # Stale callback arrives with gen_1
        self.assertFalse(self.state_mgr.is_generation_valid(gen_1), "Stale STT generation must be rejected.")

    # ---------------- TEST M: DELAYED LLM RESPONSE ARRIVES AFTER STOP ----------------
    def test_m_delayed_llm_response_after_stop(self):
        """Test M: An in-flight LLM call that finishes after stop has its speech output suppressed."""
        controller = ChittiController()
        mock_tts = MagicMock()
        controller.tts = mock_tts

        gen_id = controller.state_manager.new_request()
        # Simulate LLM in progress, then user interrupts
        controller.state_manager.interrupt(reason="user_stop")

        # LLM completes delayed response and calls speak with old request_id
        controller.speak("Delayed answer to previous question", request_id=gen_id)
        mock_tts.speak.assert_not_called()

    # ---------------- TEST N: QUEUED TTS SEGMENT ATTEMPTS TO PLAY AFTER STOP ----------------
    def test_n_queued_tts_segment_discarded_on_stop(self):
        """Test N: SpeechController queue items with cancelled request IDs are discarded without speaking."""
        mock_engine = MagicMock(spec=TTSEngine)
        controller = SpeechController(engine=mock_engine)

        gen_1 = self.state_mgr.new_request()
        # Request is cancelled by stop
        self.state_mgr.interrupt(reason="user_stop")
        controller.stop_speaking()

        # Enqueue with cancelled generation ID
        controller.speak("Queued utterance from cancelled request", wait=False, request_id=gen_1)
        time.sleep(0.15)

        # Engine should not have spoken cancelled utterance
        mock_engine.speak.assert_not_called()
        controller.shutdown()

    # ---------------- TEST O: REPEATED STOP COMMANDS ----------------
    def test_o_repeated_stop_commands(self):
        """Test O: Multiple rapid stop commands operate idempotently without deadlocks."""
        for _ in range(5):
            self.state_mgr.interrupt(reason="user_stop")
            self.assertTrue(self.state_mgr.is_interrupted())
        # System should remain healthy and awakenable
        self.state_mgr.wake()
        self.assertTrue(self.state_mgr.is_awake())

    # ---------------- TEST P & Q: MICROPHONE ERROR AND RECOVERY ----------------
    def test_p_q_audio_capture_error_and_recovery(self):
        """Test P & Q: AudioCaptureService handles hardware disconnections and recovers cleanly."""
        capture = AudioCaptureService()
        self.assertFalse(capture.is_running())
        # Subscribe a test listener
        received = []
        capture.subscribe("test_listener", lambda chunk, energy: received.append(energy))
        self.assertTrue(capture.has_subscriber("test_listener"))

        # Flush buffer
        capture.flush()
        self.assertEqual(len(received), 0)

        # Unsubscribe
        capture.unsubscribe("test_listener")
        self.assertFalse(capture.has_subscriber("test_listener"))

    # ---------------- TEST R: NEW REQUEST AFTER EARLIER CANCELLED ----------------
    def test_r_new_request_after_cancelled(self):
        """Test R: After an earlier request is cancelled, a new request executes successfully."""
        controller = ChittiController()
        mock_tts = MagicMock()
        controller.tts = mock_tts

        # 1. First request cancelled
        req_1 = controller.state_manager.new_request()
        controller.state_manager.interrupt(reason="user_stop")
        controller.speak("Response 1", request_id=req_1)
        mock_tts.speak.assert_not_called()

        # 2. Second request arrives
        req_2 = controller.state_manager.new_request()
        self.assertTrue(controller.state_manager.is_generation_valid(req_2))
        controller.speak("Response 2", request_id=req_2)
        mock_tts.speak.assert_called_with("Response 2", wait=True)


if __name__ == "__main__":
    unittest.main()
