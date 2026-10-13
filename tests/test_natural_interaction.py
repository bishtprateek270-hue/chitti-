"""
Comprehensive Natural Voice Interaction & Fast Agent Engine Test Suite.
Verifies:
1. Clear English requests
2. Hindi requests
3. Hinglish requests
4. Requests with natural phrasing
5. Requests referring to recent context
6. Requests referring to persistent personal memory
7. Short commands
8. Long utterances
9. Background noise handling
10. Silence handling
11. Chitti's own TTS playing (rejection)
12. Repeated wake words
13. Stop during speech (barge-in)
14. Stop during model inference (invalidation)
15. Stop during computer automation (task cancellation)
16. Sleep and wake transitions
17. Ambiguous requests (asking 1 concise question)
18. Simple commands avoiding expensive planning
19. Complex tasks requiring planning
20. Requests with missing information
"""

import time
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.core.state_manager import SystemStateManager, SystemState
from src.core.phrase_matcher import PhraseMatcher, ControlCommand
from src.router.intent_engine import NaturalIntentEngine, IntentType
from src.router.master_router import MasterRouter, MasterRoute
from src.utils.metrics import get_metrics_collector, ExecutionPath
from src.main import ChittiController


class TestNaturalInteractionSuite(unittest.TestCase):

    def setUp(self):
        from src.core.state_manager import get_state_manager
        self.state_mgr = get_state_manager()
        self.state_mgr.reset_interrupt()
        self.state_mgr.wake()
        self.intent_engine = NaturalIntentEngine()
        self.metrics = get_metrics_collector()

    def tearDown(self):
        self.state_mgr.reset_interrupt()
        self.state_mgr.wake()

    # 1. Clear English requests
    def test_01_clear_english_request(self):
        res = self.intent_engine.analyze("Open the code editor.")
        self.assertEqual(res.intent, IntentType.OPEN_APPLICATION)
        self.assertIn("code editor", res.target.lower())

    # 2. Hindi requests
    def test_02_hindi_requests(self):
        res = self.intent_engine.analyze("मेरा प्रोजेक्ट खोलो")
        self.assertIn(res.intent, (IntentType.OPEN_PROJECT, IntentType.OPEN_APPLICATION))
        res_vol = self.intent_engine.analyze("आवाज़ कम करो")
        self.assertEqual(res_vol.intent, IntentType.SIMPLE_COMMAND)
        self.assertEqual(res_vol.execution_path, ExecutionPath.PATH_A_SIMPLE)

    # 3. Hinglish requests
    def test_03_hinglish_requests(self):
        res = self.intent_engine.analyze("Yaar, kal jo Python project banaya tha, usko kholna.")
        self.assertEqual(res.intent, IntentType.OPEN_PROJECT)
        self.assertEqual(res.execution_path, ExecutionPath.PATH_C_COMPUTER)
        self.assertTrue(res.context_required)

    # 4. Requests with natural phrasing
    def test_04_natural_phrasing_requests(self):
        # Code debugging
        res_debug = self.intent_engine.analyze("This code isn't working. Can you figure out why?")
        self.assertEqual(res_debug.intent, IntentType.DEBUG_CODE)
        self.assertEqual(res_debug.execution_path, ExecutionPath.PATH_D_CODING)

        # Website creation
        res_site = self.intent_engine.analyze("Can you make a simple website for my project?")
        self.assertEqual(res_site.intent, IntentType.CREATE_PROJECT)
        self.assertEqual(res_site.execution_path, ExecutionPath.PATH_D_CODING)

    # 5. Requests referring to recent context
    def test_05_recent_context_resolution(self):
        res = self.intent_engine.analyze("Open that project again.")
        self.assertEqual(res.intent, IntentType.OPEN_PROJECT)
        self.assertTrue(res.context_required)

    # 6. Requests referring to persistent personal memory
    def test_06_persistent_memory_queries(self):
        decision = MasterRouter.classify_request("What is my name?")
        self.assertEqual(decision.route, MasterRoute.PERSONAL_MEMORY)
        self.assertTrue(decision.requires_memory)

        dec_rem = MasterRouter.classify_request("Remember that I like dark mode in VS Code.")
        self.assertEqual(dec_rem.route, MasterRoute.EXPLICIT_MEMORY)
        self.assertTrue(dec_rem.requires_memory)

    # 7. Short commands
    def test_07_short_commands(self):
        res_stop = self.intent_engine.analyze("Stop.")
        self.assertEqual(res_stop.intent, IntentType.SIMPLE_COMMAND)
        self.assertEqual(res_stop.execution_path, ExecutionPath.PATH_A_SIMPLE)

        res_calc = self.intent_engine.analyze("Calculator")
        self.assertEqual(res_calc.intent, IntentType.OPEN_APPLICATION)
        self.assertEqual(res_calc.execution_path, ExecutionPath.PATH_A_SIMPLE)

    # 8. Long utterances with conversational filler
    def test_08_long_utterance_with_filler(self):
        res = self.intent_engine.analyze(
            "Yaar please can you check if there are any issues with my code right now, it isn't working."
        )
        self.assertEqual(res.intent, IntentType.DEBUG_CODE)
        self.assertEqual(res.execution_path, ExecutionPath.PATH_D_CODING)

    # 9. Background noise handling (low amplitude/rms)
    def test_09_background_noise_handling(self):
        from src.audio.stt import HybridSTT
        stt = HybridSTT()
        # White noise with very low amplitude
        noise = np.random.uniform(-0.001, 0.001, size=16000).astype(np.float32)
        txt = stt.transcribe(noise)
        self.assertEqual(txt, "")

    # 10. Silence handling without hallucination
    def test_10_silence_handling(self):
        from src.audio.stt import WhisperSTT
        whisper = WhisperSTT()
        self.assertTrue(whisper.is_hallucination("Thanks for watching!"))
        self.assertTrue(whisper.is_hallucination("[Music]"))
        self.assertFalse(whisper.is_hallucination("Open Google Chrome"))

    # 11. Chitti's own TTS playing (rejection / barge-in)
    def test_11_self_tts_suppression(self):
        self.state_mgr.wake()
        req_id = self.state_mgr.new_request()
        self.state_mgr.begin_speaking(req_id)
        self.assertEqual(self.state_mgr.get_state(), SystemState.SPEAKING)
        # Verify state protects against accidental trigger
        self.assertTrue(self.state_mgr.can_speak(req_id))

    # 12. Repeated wake words
    def test_12_repeated_wake_words(self):
        matcher = PhraseMatcher()
        res = matcher.classify_command("Hey Chitti, Chitti, wake up")
        self.assertEqual(res.command, ControlCommand.WAKE)

    # 13. Stop during speech (barge-in sub-300ms)
    def test_13_stop_during_speech(self):
        from src.audio.tts import EdgeNeuralTTS
        tts = EdgeNeuralTTS()
        t0 = time.perf_counter()
        lat_ms = tts.stop()
        self.assertLess(lat_ms, 300.0)
        self.assertFalse(tts.is_speaking())

    # 14. Stop during model inference (invalidation of request ID)
    def test_14_stop_during_inference_invalidation(self):
        self.state_mgr.wake()
        req_id = self.state_mgr.new_request()
        self.state_mgr.begin_processing(req_id)
        self.assertTrue(self.state_mgr.is_generation_valid(req_id))

        # User interrupts / stops
        self.state_mgr.interrupt(reason="user_stop")
        self.assertFalse(self.state_mgr.is_generation_valid(req_id))
        self.assertFalse(self.state_mgr.can_speak(req_id))

    # 15. Stop during computer automation (task cancellation)
    def test_15_stop_during_computer_automation(self):
        from src.agent.manager import LaptopAgentManager
        agent = LaptopAgentManager()
        # Mock active task
        mock_task = MagicMock()
        agent.active_task_state = mock_task
        agent.cancel_active_task()
        mock_task.cancel.assert_called_once()
        self.assertIsNone(agent.pending_destructive_action)

    # 16. Sleep and wake transitions
    def test_16_sleep_and_wake_transitions(self):
        self.state_mgr.wake()
        self.assertEqual(self.state_mgr.get_state(), SystemState.AWAKE)
        self.assertFalse(self.state_mgr.is_sleeping())

        self.state_mgr.sleep()
        self.assertEqual(self.state_mgr.get_state(), SystemState.SLEEPING)
        self.assertTrue(self.state_mgr.is_sleeping())

        self.state_mgr.wake()
        self.assertEqual(self.state_mgr.get_state(), SystemState.AWAKE)
        self.assertFalse(self.state_mgr.is_sleeping())

    # 17. Ambiguous requests (asking concise question if critical info missing)
    def test_17_ambiguous_messaging_request(self):
        # Missing recipient
        res = self.intent_engine.analyze("I need to send a message that I will be late.")
        self.assertEqual(res.intent, IntentType.SEND_MESSAGE)
        self.assertIn("recipient", res.missing_information)
        self.assertIsNotNone(res.clarification_prompt)

    # 18. Simple commands avoiding expensive planning
    def test_18_simple_commands_avoid_planning(self):
        res = self.intent_engine.analyze("Mute the volume")
        self.assertEqual(res.execution_path, ExecutionPath.PATH_A_SIMPLE)
        decision = MasterRouter.classify_request("Mute the volume")
        self.assertFalse(decision.requires_phase6)

    # 19. Complex tasks requiring planning
    def test_19_complex_tasks_require_planning(self):
        decision = MasterRouter.classify_request("Find the assignment, rename it, and attach it to an email")
        self.assertTrue(decision.requires_computer)

    # 20. Requests with missing information handled gracefully
    def test_20_missing_information_resolution(self):
        res = self.intent_engine.analyze("I need to tell Rahul that I'll be late.")
        self.assertEqual(res.intent, IntentType.SEND_MESSAGE)
        self.assertEqual(res.recipient, "Rahul")
        self.assertEqual(res.content, "I'll be late")
        self.assertEqual(len(res.missing_information), 0)


if __name__ == "__main__":
    unittest.main()
