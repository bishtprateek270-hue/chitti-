"""
Tests for WhatsApp and Messaging Contact Emoji Handling & Normalization.
Ensures contacts saved with emojis or requested with emojis can be cleanly searched,
matched, selected, and messaged without failure.
"""
import pytest
from src.utils.text import strip_emojis, clean_contact_query, matches_contact_name
from src.agent.intent import ActionIntentAnalyzer, ActionIntentType
from src.computer_use.planner import ComputerUseEngine
from src.computer_use.actions import ComputerActionType, ComputerActionStep
from src.computer_use.verifier import ComputerActionVerifier, VerificationStatus
from src.computer_use.controller import ComputerController
from src.computer_use.screen import ScreenObserver


def test_strip_emojis():
    assert strip_emojis("Rahul ❤️") == "Rahul"
    assert strip_emojis("🔥 Ayush 🔥") == "Ayush"
    assert strip_emojis("Mummy 🥰") == "Mummy"
    assert strip_emojis("Papa 💼") == "Papa"
    assert strip_emojis("👑 Boss 👑") == "Boss"
    assert strip_emojis("Bhai 🚀") == "Bhai"
    assert strip_emojis("NoEmojiHere") == "NoEmojiHere"
    assert strip_emojis("") == ""


def test_clean_contact_query():
    assert clean_contact_query("Rahul ❤️") == "Rahul"
    assert clean_contact_query("  🔥 Ayush 🔥  ") == "Ayush"
    assert clean_contact_query("Mummy 🥰 (Home)") == "Mummy (Home)"
    assert clean_contact_query("\"Rahul 🔥\"") == "Rahul"
    assert clean_contact_query("Ayush") == "Ayush"


def test_matches_contact_name():
    # Stored contact has emojis, search query is plain
    assert matches_contact_name("Rahul", "Rahul ❤️") is True
    assert matches_contact_name("Ayush", "🔥 Ayush 🔥") is True
    assert matches_contact_name("Mummy", "Mummy 🥰") is True
    # Stored contact is plain, search query has emojis
    assert matches_contact_name("Rahul ❤️", "Rahul") is True
    # Mismatched contacts
    assert matches_contact_name("Rahul", "Suresh ❤️") is False


def test_intent_extraction_with_emoji_contact():
    # 1. Plain query to contact
    intent1 = ActionIntentAnalyzer.extract_intent("open whatsapp web and send hi to rahul")
    assert intent1.intent == ActionIntentType.SEND_MESSAGE
    assert intent1.parameters.get("search_contact").lower() == "rahul"

    # 2. Query mentioning contact with emoji
    intent2 = ActionIntentAnalyzer.extract_intent("open whatsapp web and send hi to rahul ❤️")
    assert intent2.intent == ActionIntentType.SEND_MESSAGE
    assert "rahul" in intent2.target.lower()
    assert intent2.parameters.get("search_contact").lower() == "rahul"

    # 3. Message first, contact with emoji
    intent3 = ActionIntentAnalyzer.extract_intent("send 'I will reach soon' to 🔥 Ayush 🔥 on whatsapp")
    assert intent3.intent == ActionIntentType.SEND_MESSAGE
    assert intent3.parameters.get("search_contact").lower() == "ayush"
    assert intent3.content == "I will reach soon"


def test_computer_use_plan_emoji_contact():
    engine = ComputerUseEngine()
    state = engine.plan_task("open whatsapp web and send 'meeting at 5' to Rahul ❤️")

    assert len(state._internal_steps) == 7
    search_step = state._internal_steps[2]
    assert search_step.action_type == ComputerActionType.SEARCH_CONTACT
    # Verify search query is cleaned of emojis for typing into WhatsApp search box
    assert search_step.parameters["search_contact"] == "Rahul"
    assert "Rahul" in search_step.parameters["contact"]


def test_computer_use_search_contact_execution(monkeypatch):
    engine = ComputerUseEngine()
    state = engine.plan_task("open whatsapp web and send hi to 🔥 Ayush 🔥")

    # Mock controller actions to capture typed text
    typed_history = []
    hotkey_history = []
    pressed_keys = []

    monkeypatch.setattr(engine.controller, "focus_window", lambda title: True)
    monkeypatch.setattr(engine.controller, "hotkey", lambda *keys: hotkey_history.append(keys) or True)
    monkeypatch.setattr(engine.controller, "type_text", lambda text: typed_history.append(text) or True)
    monkeypatch.setattr(engine.controller, "press_key", lambda k: pressed_keys.append(k) or True)
    monkeypatch.setattr(engine.controller, "wait", lambda s: None)

    search_step = state._internal_steps[2]
    result = engine._execute_step_action(search_step)

    assert result.success is True
    # The search query typed into the WhatsApp search box MUST be the clean name "Ayush"
    assert "Ayush" in typed_history
    assert ("ctrl", "alt", "/") in hotkey_history
    assert "enter" in pressed_keys


def test_verifier_with_emoji_contact():
    controller = ComputerController()
    observer = ScreenObserver(controller=controller)
    verifier = ComputerActionVerifier(controller=controller, screen_observer=observer)

    # Verifying search / select for contact with emoji
    res_search = verifier.verify_action(ComputerActionType.SEARCH_CONTACT, target="Rahul ❤️", parameters={"contact": "Rahul ❤️"})
    assert res_search.status == VerificationStatus.SUCCESS
    assert "Rahul" in res_search.evidence

    # Verifying sent message to contact with emoji
    res_msg = verifier.verify_action(ComputerActionType.SEND_MESSAGE, target="🔥 Ayush 🔥", parameters={"contact": "🔥 Ayush 🔥", "text": "Hello"})
    assert res_msg.status == VerificationStatus.SUCCESS
    assert "Ayush" in res_msg.evidence
