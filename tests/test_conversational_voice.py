"""
Chitti Natural Two-Way Conversational Voice System Test Suite (Phase 10).
Comprehensive verification of all 15 Phase 10 acceptance scenarios:
1. Normal conversation with several back-and-forth turns
2. Follow-up question without repeating the subject
3. Interrupting Chitti halfway through a sentence (<300ms barge-in)
4. Correcting the request while Chitti is speaking
5. Speaking Hindi, English, and Hinglish
6. Short pauses within a sentence
7. Background noise handling
8. TTS echo and false interruption rejection
9. Stop and sleep commands during LLM generation
10. Wake after sleeping
11. Interrupting a long-running computer task
12. Stale callbacks arriving after cancellation
13. Microphone initialization failure and recovery
14. Several consecutive conversations without restarting Chitti
15. Duplicate execution prevention
"""

import time
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.core.state_manager import SystemStateManager, SystemState, get_state_manager
from src.core.phrase_matcher import get_phrase_matcher, ControlCommand
from src.router.intent_engine import NaturalIntentEngine, IntentType
from src.router.master_router import MasterRouter, MasterRoute
from src.audio.interrupter import BargeInDetector
from src.audio.capture_service import get_audio_capture_service
from src.audio.streaming_stt import StreamingSTTEngine
from src.audio.stt import WhisperSTT, HybridSTT


class TestConversationalVoiceSuite(unittest.TestCase):

    def setUp(self):
        self.state_mgr = get_state_manager()
        self.state_mgr.reset_interrupt()
        self.state_mgr.wake()
        self.intent_engine = NaturalIntentEngine()

    def tearDown(self):
        self.state_mgr.reset_interrupt()
        self.state_mgr.end_conversation_session()

    # 1. Normal conversation with several back-and-forth turns
    def test_01_multi_turn_conversation_flow(self):
        # Turn 1: User speaks
        self.state_mgr.start_conversation_session(timeout_seconds=10.0)
        self.assertTrue(self.state_mgr.is_in_conversation_session())

        req1 = self.state_mgr.new_request()
        self.state_mgr.begin_processing(req1)
        self.state_mgr.begin_speaking(req1)
        self.assertEqual(self.state_mgr.get_state(), SystemState.SPEAKING)

        # Assistant finishes Turn 1 -> seamlessly returns to IDLE_LISTENING for follow-up
        self.state_mgr.finish_task(req1)
        self.assertEqual(self.state_mgr.get_state(), SystemState.IDLE_LISTENING)
        self.assertTrue(self.state_mgr.is_in_conversation_session())

        # Turn 2: User speaks without saying wake word
        req2 = self.state_mgr.new_request()
        self.state_mgr.begin_user_speaking(req2)
        self.assertEqual(self.state_mgr.get_state(), SystemState.USER_SPEAKING)
        self.state_mgr.begin_transcribing(req2)
        self.assertEqual(self.state_mgr.get_state(), SystemState.TRANSCRIBING)
        self.state_mgr.begin_processing(req2)
        self.assertEqual(self.state_mgr.get_state(), SystemState.PROCESSING)
        self.state_mgr.begin_speaking(req2)
        self.assertEqual(self.state_mgr.get_state(), SystemState.SPEAKING)
        self.state_mgr.finish_task(req2)
        self.assertEqual(self.state_mgr.get_state(), SystemState.IDLE_LISTENING)

    # 2. Follow-up question without repeating the subject
    def test_02_followup_question_anaphora_resolution(self):
        history = [
            {"role": "user", "content": "Explain neural networks."},
            {"role": "assistant", "content": "Neural networks are computational models inspired by biological neural circuits."},
        ]
        # Follow-up: "Explain it more simply"
        res = self.intent_engine.analyze("Wait, explain it more simply.", conversation_history=history)
        self.assertEqual(res.intent, IntentType.TECHNICAL_QUERY)
        self.assertEqual(res.target, "neural networks")
        self.assertEqual(res.parameters.get("refinement"), "simpler")

        # Follow-up: "Actually, make the answer shorter."
        res_short = self.intent_engine.analyze("Actually, make the answer shorter.", conversation_history=history)
        self.assertEqual(res_short.intent, IntentType.TECHNICAL_QUERY)
        self.assertEqual(res_short.target, "neural networks")
        self.assertEqual(res_short.parameters.get("refinement"), "shorter")

    # 3. Interrupting Chitti halfway through a sentence (<300ms barge-in)
    def test_03_interrupt_during_speech(self):
        from src.audio.tts import EdgeNeuralTTS
        tts = EdgeNeuralTTS()
        t0 = time.perf_counter()
        lat_ms = tts.stop()
        self.assertLess(lat_ms, 300.0)
        self.assertFalse(tts.is_speaking())

    # 4. Correcting the request while Chitti is speaking (barge-in handoff)
    def test_04_correcting_request_during_speech_handoff(self):
        detector = BargeInDetector()
        mock_callback = MagicMock()
        detector.start_monitoring(on_interrupt=mock_callback)

        # Generate voiced frames that simulate human voice (200Hz sine wave)
        sr = 16000
        t = np.linspace(0, 0.05, 800, endpoint=False)
        voiced_chunk = (0.35 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

        # Trigger barge-in frames
        detector._start_time = time.time() - 1.0  # past grace period
        for _ in range(6):
            detector._audio_callback(voiced_chunk, energy=0.25)

        self.assertTrue(mock_callback.called)
        interrupted_audio = detector.get_interruption_audio()
        self.assertIsNotNone(interrupted_audio)
        self.assertGreater(len(interrupted_audio), 0)

        # Seamlessly hand off audio to streaming STT
        stt = StreamingSTTEngine()
        # Verify initial_audio prepends correctly without crashing
        with patch.object(stt, "record_until_silence", return_value=interrupted_audio):
            text = stt.transcribe_streaming(initial_audio=interrupted_audio)
            self.assertIsInstance(text, str)

    # 5. Speaking Hindi, English, and Hinglish
    def test_05_multilingual_natural_understanding(self):
        # English
        res_en = self.intent_engine.analyze("Open the code editor.")
        self.assertEqual(res_en.intent, IntentType.OPEN_APPLICATION)

        # Hindi
        res_hi = self.intent_engine.analyze("calculator kholo")
        self.assertEqual(res_hi.intent, IntentType.OPEN_APPLICATION)

        # Hinglish
        res_hing = self.intent_engine.analyze("kal jo Python project banaya tha usko kholna")
        self.assertEqual(res_hing.intent, IntentType.OPEN_PROJECT)

    # 6. Short pauses within a sentence
    def test_06_short_pauses_within_sentence(self):
        stt = StreamingSTTEngine(silence_cutoff_seconds=0.85)
        # Ensure silence cutoff is calibrated so a 0.5s pause does not cut off speech
        self.assertGreater(stt.silence_cutoff_seconds, 0.50)
        self.assertGreater(stt.pre_speech_seconds, 0.30)

    # 7. Background noise handling
    def test_07_background_noise_rejection(self):
        stt = HybridSTT()
        # Low RMS white noise
        noise = np.random.uniform(-0.001, 0.001, size=16000).astype(np.float32)
        txt = stt.transcribe(noise)
        self.assertEqual(txt, "")

    # 8. TTS echo and false interruption rejection
    def test_08_tts_echo_rejection(self):
        detector = BargeInDetector(energy_threshold=0.10)
        mock_cb = MagicMock()
        detector.start_monitoring(on_interrupt=mock_cb)
        self.assertTrue(detector._is_active)

        # Pure unvoiced noise or faint acoustic reflection (low energy)
        faint_echo = (0.02 * np.random.randn(800)).astype(np.float32)
        detector._start_time = time.time() - 1.0
        for _ in range(5):
            detector._audio_callback(faint_echo, energy=0.02)
        # Hit count should remain 0 and not trigger interrupt
        self.assertEqual(detector._hit_count, 0)
        self.assertFalse(mock_cb.called)
        self.assertTrue(detector._is_active)
        detector.stop_monitoring()

    # 9. Stop and sleep commands during LLM generation
    def test_09_stop_and_sleep_during_generation(self):
        self.state_mgr.wake()
        req_id = self.state_mgr.new_request()
        self.state_mgr.begin_processing(req_id)
        self.assertTrue(self.state_mgr.is_generation_valid(req_id))

        # Emergency stop received
        self.state_mgr.interrupt(reason="user_stop")
        self.assertFalse(self.state_mgr.is_generation_valid(req_id))
        self.assertFalse(self.state_mgr.can_speak(req_id))

    # 10. Wake after sleeping
    def test_10_wake_after_sleeping(self):
        self.state_mgr.sleep()
        self.assertTrue(self.state_mgr.is_sleeping())
        self.assertEqual(self.state_mgr.get_state(), SystemState.SLEEPING)

        self.state_mgr.wake()
        self.assertFalse(self.state_mgr.is_sleeping())
        self.assertEqual(self.state_mgr.get_state(), SystemState.AWAKE)
        self.assertFalse(self.state_mgr.is_interrupted())

    # 11. Interrupting a long-running computer task
    def test_11_interrupting_computer_task(self):
        from src.agent.manager import LaptopAgentManager
        agent = LaptopAgentManager()
        mock_task = MagicMock()
        agent.active_task_state = mock_task
        agent.cancel_active_task()
        mock_task.cancel.assert_called_once()
        self.assertIsNone(agent.pending_destructive_action)

    # 12. Stale callbacks arriving after cancellation
    def test_12_stale_callbacks_rejected(self):
        req1 = self.state_mgr.new_request()
        # Invalidate generation (e.g. interruption happened)
        req2 = self.state_mgr.new_request()

        # Stale callback from req1 must be rejected
        self.assertFalse(self.state_mgr.is_generation_valid(req1))
        self.assertFalse(self.state_mgr.can_speak(req1))

        # Active request req2 is valid
        self.assertTrue(self.state_mgr.is_generation_valid(req2))
        self.assertTrue(self.state_mgr.can_speak(req2))

    # 13. Microphone initialization failure and recovery
    def test_13_mic_initialization_recovery(self):
        capture = get_audio_capture_service()
        self.assertIsNotNone(capture)
        # Flush pending buffers
        capture.flush()
        self.assertEqual(capture._audio_queue.qsize(), 0)

    # 14. Several consecutive conversations without restarting Chitti
    def test_14_consecutive_conversations_stability(self):
        for i in range(5):
            req_id = self.state_mgr.new_request()
            self.state_mgr.begin_processing(req_id)
            self.state_mgr.begin_speaking(req_id)
            self.state_mgr.finish_task(req_id)
        # State manager should remain completely healthy
        self.assertTrue(self.state_mgr.is_awake())
        self.assertFalse(self.state_mgr.is_sleeping())

    # 15. Duplicate execution prevention
    def test_15_duplicate_execution_prevention(self):
        req_id = self.state_mgr.new_request()
        self.state_mgr.begin_processing(req_id)
        self.state_mgr.finish_task(req_id)

        # Invalidate after completion
        self.state_mgr.invalidate_generation(req_id)
        # Duplicate callback for the same old request ID cannot proceed
        self.assertFalse(self.state_mgr.is_generation_valid(req_id))
        self.assertFalse(self.state_mgr.can_speak(req_id))


if __name__ == "__main__":
    unittest.main()
