"""
Unit tests for Chitti Language Detector (Phase 4).
Tests English, Devanagari Hindi, Roman Hindi, and Hinglish detection.
"""

import pytest
from src.language.detector import LanguageDetector
from src.language.language_models import LanguageCode


@pytest.fixture
def detector():
    return LanguageDetector()


def test_detect_pure_english(detector):
    samples = [
        "Open Chrome and check my notifications.",
        "What is machine learning?",
        "Can you help me debug this Python function?",
        "Show me my CPU and RAM usage.",
        "Where is my project file located?",
    ]
    for text in samples:
        res = detector.detect(text)
        assert res.language == LanguageCode.ENGLISH.value, f"Failed on '{text}'"
        assert res.confidence >= 0.70
        assert res.script == "latin"


def test_detect_devanagari_hindi(detector):
    samples = [
        "क्रोम खोलो।",
        "मशीन लर्निंग क्या है?",
        "मेरा सीपीयू कितना इस्तेमाल हो रहा है?",
        "आज मौसम कैसा है?",
        "मेरे सामने क्या है?",
        "मेरी फाइल कहाँ है?",
    ]
    for text in samples:
        res = detector.detect(text)
        assert res.language == LanguageCode.HINDI.value, f"Failed on '{text}'"
        assert res.script == "devanagari"
        assert res.confidence >= 0.70


def test_detect_roman_hindi(detector):
    samples = [
        "chrome kholo",
        "machine learning kya hai",
        "mera cpu kitna use ho raha hai",
        "kal mujhe college jana hai",
        "ye error kyu aa raha hai",
        "samne kya hai",
        "bhai is code ko check kar",
    ]
    for text in samples:
        res = detector.detect(text)
        assert res.language == LanguageCode.HINGLISH.value, f"Failed on '{text}'"
        assert res.is_roman_hindi is True
        assert res.confidence >= 0.65


def test_detect_hinglish_mixed(detector):
    samples = [
        "Chrome kholo aur YouTube open karo.",
        "Mujhe explain karo ki CNN kaise work karta hai.",
        "Can you bata sakte ho mera GPU kitna use ho raha hai?",
        "Open my project aur check karo backend kyu nahi chal raha.",
        "Ye file delete mat karna please.",
        "bhai can you mera Chrome open kar sakte ho",
    ]
    for text in samples:
        res = detector.detect(text)
        assert res.language == LanguageCode.HINGLISH.value, f"Failed on '{text}'"
        assert len(res.detected_markers) > 0


def test_detect_phonetic_variations(detector):
    samples = [
        "mjhe btao ye code kyu fail ho rha h",
        "kal assignment submit krna h",
        "bhai apka model kese train ho rha h",
    ]
    for text in samples:
        res = detector.detect(text)
        assert res.language == LanguageCode.HINGLISH.value, f"Failed on '{text}'"


def test_detect_empty_or_whitespace(detector):
    res = detector.detect("")
    assert res.language == LanguageCode.UNKNOWN.value

    res = detector.detect("   \n\t  ")
    assert res.language == LanguageCode.UNKNOWN.value
