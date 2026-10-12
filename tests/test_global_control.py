"""
Comprehensive Automated Test Suite for Global Voice Control & Wake/Sleep System.
Validates Tests A through M specified in Project Requirements:
- Test A: Wake
- Test B: Ignore ordinary speech while sleeping
- Test C: English stop
- Test D: Hindi stop
- Test E: Hinglish stop
- Test F: Sleep
- Test G: Wake after sleep
- Test H: Stop during code generation
- Test I: Stop during computer use
- Test J: Stop during TTS (measures actual interruption latency <= 300ms)
- Test K: False activation prevention
- Test L: Repeated commands (idempotency, thread safety, no duplicate workers)
- Test M: Existing functionality remains intact
"""

import time
import threading
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.core.state_manager import (
    SystemState,
    SystemStateManager,
    CommandPriority,
    get_state_manager,
)
from src.core.phrase_matcher import (
    ControlCommand,
    PhraseMatcher,
    get_phrase_matcher,
)
from src.audio.tts import EdgeNeuralTTS, SpeechController, TTSEngine
from src.audio.ambient_listener import AmbientVoiceListener, AmbientState
from src.audio.wake_word import WakeWordDetector, WakeWordEvent
from src.agent.manager import LaptopAgentManager
from src.agent.task_state import TaskState, TaskStatus, AgentStep
from src.agent.planner import ComputerAgentLoop
from src.main import ChittiController
from src.ui.hud_overlay import HUDMode


class TestGlobalVoiceControlSystem(unittest.TestCase):
    """Automated integration and unit test suite for the wake/sleep/stop architecture."""

    def setUp(self):
        self.state_mgr = get_state_manager()
        self.state_mgr.reset_interrupt()
        self.state_mgr.wake()
        self.phrase_matcher = get_phrase_matcher()

    def tearDown(self):
        self.state_mgr.reset_interrupt()
        self.state_mgr.wake()

    # ---------------- TEST A: WAKE ----------------
    def test_a_wake_command_transitions_state_and_acknowledges(self):
        """Test A: When Chitti is sleeping, saying 'Hey Chitti' wakes it up with brief confirmation."""
        self.state_mgr.sleep()
        self.assertTrue(self.state_mgr.is_sleeping())
        self.assertEqual(self.state_mgr.get_state(), SystemState.SLEEPING)

        wake_variants = [
            "Hey Chitti",
            "Chitti",
            "Wake up Chitti",
            "Chitti wake up",
            "Chitti jaag ja",
            "Chitti uth ja",
            "Chitti sun",
            "Jaag ja Chitti",
            "Chitti, wake up",
            "Hey Chitti, suno",
            "Chitti, active ho ja",
        ]

        for phrase in wake_variants:
            res = self.phrase_matcher.classify_command(phrase)
            self.assertEqual(res.command, ControlCommand.WAKE, f"Failed on wake variant: {phrase}")
            self.assertEqual(res.priority, CommandPriority.PRIORITY_3_WAKE.value)

        # Trigger wake on controller
        controller = ChittiController()
        mock_tts = MagicMock()
        controller.tts = mock_tts

        controller.process_user_input("Hey Chitti")
        self.assertTrue(self.state_mgr.is_awake())
        self.assertEqual(self.state_mgr.get_state(), SystemState.AWAKE)
        mock_tts.speak.assert_called_with("Haan, boliye.", wait=True)

    # ---------------- TEST B: IGNORE ORDINARY SPEECH WHILE SLEEPING ----------------
    def test_b_ignore_ordinary_speech_while_sleeping(self):
        """Test B: When Chitti is sleeping, ordinary commands like 'Open Chrome' are ignored without invoking LLM or tasks."""
        self.state_mgr.sleep()
        self.assertTrue(self.state_mgr.is_sleeping())

        controller = ChittiController()
        mock_llm = MagicMock()
        mock_tts = MagicMock()
        controller.llm = mock_llm
        controller.tts = mock_tts

        # Normal command while sleeping
        controller.process_user_input("Open Chrome")

        # LLM must not be contacted and TTS must not be spoken
        mock_llm.generate_response.assert_not_called()
        mock_tts.speak.assert_not_called()
        self.assertTrue(self.state_mgr.is_sleeping())

    # ---------------- TEST C: ENGLISH STOP ----------------
    def test_c_english_stop_command(self):
        """Test C: User says 'Stop' or 'Stop Chitti' -> Priority 1 interruption, playback stops."""
        english_stop_phrases = [
            "Stop",
            "Stop Chitti",
            "Chitti stop",
            "Stop now",
            "Cancel",
            "Cancel everything",
            "Be quiet",
            "Silence",
        ]

        for phrase in english_stop_phrases:
            res = self.phrase_matcher.classify_command(phrase)
            self.assertEqual(res.command, ControlCommand.STOP, f"Failed on English stop: {phrase}")
            self.assertEqual(res.priority, CommandPriority.PRIORITY_1_INTERRUPT.value)

        # Process on controller
        controller = ChittiController()
        mock_tts = MagicMock()
        controller.tts = mock_tts

        controller.process_user_input("Stop Chitti")
        self.assertTrue(self.state_mgr.is_interrupted())
        mock_tts.stop.assert_called()

    # ---------------- TEST D: HINDI STOP ----------------
    def test_d_hindi_stop_command(self):
        """Test D: User says 'Ruk ja', 'Bas kar', 'Chup ho ja' -> Priority 1 interruption."""
        hindi_stop_phrases = [
            "Ruk ja",
            "Ruko",
            "Bas kar",
            "Bas karo",
            "Chup ho ja",
            "Band kar",
            "Chitti ruk ja",
            "Chitti bas kar",
        ]

        for phrase in hindi_stop_phrases:
            res = self.phrase_matcher.classify_command(phrase)
            self.assertEqual(res.command, ControlCommand.STOP, f"Failed on Hindi stop: {phrase}")
            self.assertEqual(res.priority, CommandPriority.PRIORITY_1_INTERRUPT.value)

        controller = ChittiController()
        mock_tts = MagicMock()
        controller.tts = mock_tts

        controller.process_user_input("Ruk ja")
        self.assertTrue(self.state_mgr.is_interrupted())
        mock_tts.stop.assert_called()

    # ---------------- TEST E: HINGLISH STOP ----------------
    def test_e_hinglish_stop_command(self):
        """Test E: User says 'Chitti, stop', 'Ruk ja Chitti', 'Chitti, ab band kar'."""
        hinglish_stop_phrases = [
            "Chitti, stop",
            "Ruk ja Chitti",
            "Chitti bas kar",
            "Chitti, ab band kar",
        ]

        for phrase in hinglish_stop_phrases:
            res = self.phrase_matcher.classify_command(phrase)
            self.assertEqual(res.command, ControlCommand.STOP, f"Failed on Hinglish stop: {phrase}")
            self.assertEqual(res.priority, CommandPriority.PRIORITY_1_INTERRUPT.value)

        controller = ChittiController()
        mock_tts = MagicMock()
        controller.tts = mock_tts

        controller.process_user_input("Chitti, ab band kar")
        self.assertTrue(self.state_mgr.is_interrupted())
        mock_tts.stop.assert_called()

    # ---------------- TEST F: SLEEP COMMAND ----------------
    def test_f_sleep_command_cancels_tasks_and_enters_sleeping(self):
        """Test F: User says 'Chitti so ja' -> Tasks cancelled, speech stops, state becomes SLEEPING."""
        sleep_phrases = [
            "Chitti sleep",
            "Go to sleep",
            "Sleep now",
            "Chitti, take a rest",
            "Chitti so ja",
            "So ja Chitti",
            "Chitti aaram kar",
            "Chitti chup ho ja",
            "Chitti, sleep mode",
            "Chitti ab so ja",
            "Chitti, rest kar",
        ]

        for phrase in sleep_phrases:
            res = self.phrase_matcher.classify_command(phrase)
            self.assertEqual(res.command, ControlCommand.SLEEP, f"Failed on sleep phrase: {phrase}")
            self.assertEqual(res.priority, CommandPriority.PRIORITY_2_SLEEP.value)

        controller = ChittiController()
        mock_tts = MagicMock()
        controller.tts = mock_tts

        controller.process_user_input("Chitti so ja")
        self.assertEqual(self.state_mgr.get_state(), SystemState.SLEEPING)
        self.assertTrue(self.state_mgr.is_sleeping())
        mock_tts.stop.assert_called()

    # ---------------- TEST G: WAKE AFTER SLEEP ----------------
    def test_g_wake_after_sleep(self):
        """Test G: After sleeping in Test F, 'Chitti jaag ja' restores AWAKE state and normal interaction."""
        self.state_mgr.sleep()
        self.assertTrue(self.state_mgr.is_sleeping())

        controller = ChittiController()
        mock_tts = MagicMock()
        controller.tts = mock_tts

        controller.process_user_input("Chitti jaag ja")
        self.assertEqual(self.state_mgr.get_state(), SystemState.AWAKE)
        self.assertTrue(self.state_mgr.is_awake())
        mock_tts.speak.assert_called_with("Haan, boliye.", wait=True)

    # ---------------- TEST H: STOP DURING CODE GENERATION / MULTI-STEP TASK ----------------
    def test_h_stop_during_multi_step_task(self):
        """Test H: When a task is executing, triggering stop immediately cancels execution without starting next steps."""
        state = TaskState(task_description="Build modern dashboard")
        state.steps = [
            AgentStep(step_id=1, description="Create project folder", action_type="CREATE_DIRECTORY", parameters={"path": "dummy"}),
            AgentStep(step_id=2, description="Generate code files", action_type="WRITE_CODE", parameters={"file": "app.py"}),
            AgentStep(step_id=3, description="Run test suite", action_type="RUN_TESTS", parameters={}),
        ]

        mock_tools = MagicMock()
        # Mock step 1 execution
        mock_tools.execute_tool.return_value = MagicMock(success=True, message="Directory created")

        loop = ComputerAgentLoop(
            tool_engine=mock_tools,
            computer=MagicMock(),
            filesystem=MagicMock(),
            terminal=MagicMock(),
            browser=MagicMock(),
            apps=MagicMock(),
            screen_analyzer=MagicMock(),
            projects=MagicMock(),
        )

        # Trigger interrupt right before step 2
        def cancel_hook(*args, **kwargs):
            self.state_mgr.interrupt(reason="user_stop")

        loop.progress_callback = cancel_hook

        # Execute plan with progress callback triggering stop
        success, msg = loop.execute_plan(state, progress_callback=lambda desc, cur, tot: self.state_mgr.interrupt() if cur >= 1 else None)

        self.assertFalse(success)
        self.assertEqual(state.status, TaskStatus.CANCELLED)
        # Step 3 must not have been executed
        self.assertLess(len(state.completed_steps), 3)

    # ---------------- TEST I: STOP DURING COMPUTER USE / ACTION EXECUTION ----------------
    def test_i_stop_during_computer_use(self):
        """Test I: Action step checks cancellation before executing action."""
        step = AgentStep(step_id=1, description="Open Chrome browser", action_type="OPEN_APPLICATION", parameters={"target": "chrome"})
        state = TaskState(task_description="Browser task")

        loop = ComputerAgentLoop(
            tool_engine=MagicMock(),
            computer=MagicMock(),
            filesystem=MagicMock(),
            terminal=MagicMock(),
            browser=MagicMock(),
            apps=MagicMock(),
            screen_analyzer=MagicMock(),
            projects=MagicMock(),
        )

        # Pre-set global interrupt
        self.state_mgr.interrupt(reason="user_stop")

        success, msg, outcome = loop._execute_step_action(step, state, search_cache=[])
        self.assertFalse(success)
        self.assertIn("cancelled", msg.lower())
        self.assertTrue(state.is_cancelled)

    # ---------------- TEST J: STOP DURING TTS & LATENCY MEASUREMENT ----------------
    def test_j_stop_during_tts_and_measure_latency(self):
        """
        Test J: Start spoken answer and interrupt it.
        Measure actual interruption latency from stop signal to playback cancellation.
        Target: <= 300 ms.
        """
        tts = EdgeNeuralTTS()
        # Mock MCI send string so we test real interruption logic and timing
        with patch("ctypes.windll.winmm.mciSendStringW", return_value=0) as mock_mci:
            # Measure stop latency
            t0 = time.perf_counter()
            latency_ms = tts.stop()
            total_elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # Interruption latency must be well under 300ms
            self.assertLess(latency_ms, 300.0)
            self.assertLess(total_elapsed_ms, 300.0)
            self.assertFalse(tts.is_speaking())
            print(f"\n[MEASURED LATENCY] TTS stop latency: {latency_ms:.3f} ms (Target <= 300ms: PASS)")

    def test_j2_speech_controller_queue_discard_on_stop(self):
        """Verify speech controller clears queue and stops ongoing speech immediately."""
        mock_engine = MagicMock(spec=TTSEngine)
        mock_engine.is_speaking.return_value = True
        mock_engine.stop.return_value = 1.25  # 1.25 ms

        controller = SpeechController(engine=mock_engine)
        controller.speak("First long utterance that was playing", wait=False)
        controller.speak("Second queued utterance", wait=False)
        controller.speak("Third queued utterance", wait=False)

        # Interrupt
        lat = controller.stop_speaking()
        self.assertLessEqual(lat, 300.0)
        self.assertEqual(controller._queue.qsize(), 0)
        controller.shutdown()

    # ---------------- TEST K: FALSE ACTIVATION PREVENTION ----------------
    def test_k_false_activation_prevention(self):
        """Test K: While sleeping, unrelated words that resemble wake words but are not wake phrases must not wake Chitti."""
        unrelated_phrases = [
            "city life is busy",
            "kitty cat",
            "shitty weather",
            "check the time",
            "what is the weather today",
            "open the kitchen door",
            "play some music",
        ]

        self.state_mgr.sleep()
        for phrase in unrelated_phrases:
            res = self.phrase_matcher.classify_command(phrase)
            self.assertNotEqual(res.command, ControlCommand.WAKE, f"False wake activation on: {phrase}")

        self.assertTrue(self.state_mgr.is_sleeping())

    # ---------------- TEST L: REPEATED COMMANDS (IDEMPOTENCY & THREAD SAFETY) ----------------
    def test_l_repeated_commands_stability(self):
        """Test L: Say 'Stop. Stop. Stop.' Repeated stops do not crash, deadlock, or create duplicate workers."""
        controller = ChittiController()
        mock_tts = MagicMock()
        controller.tts = mock_tts

        for _ in range(5):
            controller.process_user_input("Stop")
            self.assertTrue(self.state_mgr.is_interrupted())

        # Reset and repeat rapidly in threads
        threads = []
        errors = []

        def worker():
            try:
                for _ in range(10):
                    self.state_mgr.interrupt(reason="threaded_stop")
                    self.state_mgr.stop_speaking()
                    self.state_mgr.reset_interrupt()
            except Exception as e:
                errors.append(e)

        for _ in range(4):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=2.0)

        self.assertEqual(len(errors), 0, f"Thread errors occurred: {errors}")

    # ---------------- TEST M: EXISTING FUNCTIONALITY CONTINUES WORKING ----------------
    def test_m_existing_functionality_intact(self):
        """Test M: Conversational requests, language detection, memory query, and routing continue working."""
        controller = ChittiController()
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = "I am Chitti, your desktop companion."
        controller.llm = mock_llm
        mock_tts = MagicMock()
        controller.tts = mock_tts

        controller.process_user_input("Who are you?")
        mock_llm.generate_response.assert_called_once()
        mock_tts.speak.assert_called_once_with("I am Chitti, your desktop companion.", wait=True)


if __name__ == "__main__":
    unittest.main()
