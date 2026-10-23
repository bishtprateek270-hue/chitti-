import time
import unittest
from unittest.mock import MagicMock

from src.audio.ambient_listener import AmbientVoiceListener, AmbientState
from src.audio.wake_word import WakeWordDetector, WakeWordEvent
from src.core.phrase_matcher import get_phrase_matcher, ControlCommand
from src.core.state_manager import get_state_manager, SystemState


class TestAmbientWakeFiltering(unittest.TestCase):
    def setUp(self):
        self.state_mgr = get_state_manager()
        self.state_mgr.reset_interrupt()
        self.state_mgr.end_conversation_session()
        self.state_mgr.wake()
        self.matcher = get_phrase_matcher()

    def tearDown(self):
        self.state_mgr.reset_interrupt()
        self.state_mgr.end_conversation_session()
        self.state_mgr.wake()

    def test_phrase_matcher_wake_words_and_city_rejection(self):
        """Phrases containing 'city', 'kitty', or 'chatting' must NOT trigger wake words."""
        city_res = self.matcher.classify_command("I want to visit the city")
        self.assertEqual(city_res.command, ControlCommand.NORMAL)

        chat_res = self.matcher.classify_command("we are just chatting")
        self.assertEqual(chat_res.command, ControlCommand.NORMAL)

        # Explicit wake phrases must trigger wake words
        hey_res = self.matcher.classify_command("hey chitti open vs code")
        self.assertEqual(hey_res.command, ControlCommand.WAKE)
        self.assertEqual(hey_res.extracted_tail, "open vs code")

        chitti_res = self.matcher.classify_command("chitti open yt")
        self.assertEqual(chitti_res.command, ControlCommand.WAKE)
        self.assertEqual(chitti_res.extracted_tail, "open yt")

    def test_unaddressed_commands_are_ignored(self):
        """Commands without 'Hey Chitti' or 'Chitti' prefix must be dropped when idling in standby."""
        executed = []

        listener = AmbientVoiceListener(
            command_handler=lambda cmd, **kw: executed.append(cmd),
            wake_detector=MagicMock(spec=WakeWordDetector),
            hud=MagicMock(),
        )

        # 1. Someone in the room says "open vs ocde"
        listener._handle_incoming_transcript("open vs ocde")
        self.assertEqual(executed, [], "Must not execute unaddressed command 'open vs ocde'")

        # 2. Someone says "open yt"
        listener._handle_incoming_transcript("open yt")
        self.assertEqual(executed, [], "Must not execute unaddressed command 'open yt'")

        # 3. Background noise: "Turn up the volume"
        listener._handle_incoming_transcript("turn up the volume")
        self.assertEqual(executed, [], "Must not execute ambient speech without Chitti prefix")

    def test_addressed_single_breath_commands_are_executed(self):
        """Commands starting with 'Hey Chitti' or 'Chitti' must execute the cleaned tail."""
        executed = []

        listener = AmbientVoiceListener(
            command_handler=lambda cmd, **kw: executed.append(cmd),
            wake_detector=MagicMock(spec=WakeWordDetector),
            hud=MagicMock(),
        )

        # 1. User says "hey chitti open vs code"
        listener._handle_incoming_transcript("hey chitti open vs code")
        self.assertIn("open vs code", executed)

        # 2. User says "chitti open yt"
        listener._handle_incoming_transcript("chitti open yt")
        self.assertIn("open yt", executed)

        # 3. User says "chitti, open browser"
        listener._handle_incoming_transcript("chitti, open browser")
        self.assertIn("open browser", executed)

    def test_standalone_wake_then_command(self):
        """When user says 'Hey Chitti', Chitti wakes and listens, then user command executes."""
        executed = []

        listener = AmbientVoiceListener(
            command_handler=lambda cmd, **kw: executed.append(cmd),
            wake_detector=MagicMock(spec=WakeWordDetector),
            hud=MagicMock(),
        )

        # Standalone wake trigger
        event = WakeWordEvent(keyword="Hey Chitti", confidence=0.95, timestamp=time.time())
        listener._on_wake_word_triggered(event)

        # In that prompted window, user speaks their command
        listener._handle_incoming_transcript("open vs code")
        self.assertIn("open vs code", executed)


if __name__ == "__main__":
    unittest.main()
