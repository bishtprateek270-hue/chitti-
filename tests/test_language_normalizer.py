"""
Unit tests for Chitti Language Normalizer (Phase 4).
Tests spelling normalization, STT error correction, negation preservation, and intent canonicalization.
"""

import pytest
from src.language.normalizer import LanguageNormalizer
from src.language.language_models import IntentCategory


@pytest.fixture
def normalizer():
    return LanguageNormalizer()


def test_phonetic_normalization(normalizer):
    raw = "mjhe btao ye code kyu fail ho rha h"
    norm = normalizer.normalize_text(raw)
    assert "mujhe" in norm
    assert "batao" in norm
    assert "raha" in norm
    assert "kyun" in norm


def test_stt_transcription_correction(normalizer):
    raw = "chitty open v s code and run p y t h o n script on g p u"
    norm = normalizer.normalize_text(raw)
    assert "Chitti" in norm
    assert "VS Code" in norm
    assert "Python" in norm
    assert "GPU" in norm


def test_preserve_paths_and_urls(normalizer):
    raw = "Run the Python script from C:\\Projects\\Chitti and open https://github.com/bishtprateek270-hue/chitti-.git"
    norm = normalizer.normalize_text(raw)
    assert "C:\\Projects\\Chitti" in norm
    assert "https://github.com/bishtprateek270-hue/chitti-.git" in norm


def test_strict_negation_preservation(normalizer):
    negative_commands = [
        "Ye file delete mat karna.",
        "Don't close Chrome.",
        "Is folder ko delete nahi karna.",
        "Never delete my database.",
        "Chrome band mat karo.",
        "क्रिकेट फाइल डिलीट मत करना।",
    ]
    for cmd in negative_commands:
        assert normalizer.check_negation(cmd) is True, f"Failed negation check on '{cmd}'"
        intent = normalizer.parse_intent(cmd)
        assert intent.is_negated is True, f"Failed parsed negation on '{cmd}'"


def test_positive_commands_not_negated(normalizer):
    positive_commands = [
        "Chrome kholo.",
        "Open VS Code.",
        "Explain what is machine learning.",
        "Samne kya hai?",
        "Remember that my project is DocForensics AI.",
    ]
    for cmd in positive_commands:
        assert normalizer.check_negation(cmd) is False, f"False positive negation on '{cmd}'"
        intent = normalizer.parse_intent(cmd)
        assert intent.is_negated is False, f"False positive intent negation on '{cmd}'"


def test_parse_app_control_intents_multilingual(normalizer):
    # English
    intent_en = normalizer.parse_intent("Open Chrome")
    assert intent_en.intent_category == IntentCategory.LAPTOP_CONTROL.value
    assert intent_en.actions[0].action == "open_application"
    assert intent_en.actions[0].target == "Google Chrome"

    # Hinglish
    intent_hi_roman = normalizer.parse_intent("Chrome kholo")
    assert intent_hi_roman.intent_category == IntentCategory.LAPTOP_CONTROL.value
    assert intent_hi_roman.actions[0].action == "open_application"
    assert intent_hi_roman.actions[0].target == "Google Chrome"

    # Devanagari Hindi
    intent_hi_dev = normalizer.parse_intent("क्रोम खोलो")
    assert intent_hi_dev.intent_category == IntentCategory.LAPTOP_CONTROL.value
    assert intent_hi_dev.actions[0].action == "open_application"
    assert intent_hi_dev.actions[0].target == "Google Chrome"


def test_parse_vision_query_multilingual(normalizer):
    queries = [
        "What do you see?",
        "Samne kya hai?",
        "Chitti tu kya dekh raha hai?",
        "मेरे सामने क्या है?",
        "Who is in front of you?",
    ]
    for q in queries:
        intent = normalizer.parse_intent(q)
        assert intent.intent_category == IntentCategory.VISION_QUERY.value, f"Failed on '{q}'"


def test_parse_language_switch_intents(normalizer):
    intent_hi = normalizer.parse_intent("Hindi mein batao")
    assert intent_hi.intent_category == IntentCategory.LANGUAGE_SWITCH.value
    assert intent_hi.target_response_language == "hi"

    intent_en = normalizer.parse_intent("English mein bolo")
    assert intent_en.intent_category == IntentCategory.LANGUAGE_SWITCH.value
    assert intent_en.target_response_language == "en"

    intent_hing = normalizer.parse_intent("Hinglish mein samjhao")
    assert intent_hing.intent_category == IntentCategory.LANGUAGE_SWITCH.value
    assert intent_hing.target_response_language == "hinglish"
