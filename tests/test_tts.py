"""Tests for Chitti Text-to-Speech module."""

from unittest.mock import patch, MagicMock
from src.audio.tts import sanitize_text_for_speech, Pyttsx3TTS, TTSError
from src.config import TTSConfig


def test_sanitize_text_markdown_stripping():
    sample = "### Heading\nHere is **bold** text, *italic* word, and `inline code`."
    cleaned = sanitize_text_for_speech(sample)
    assert "Heading" in cleaned
    assert "bold" in cleaned
    assert "italic" in cleaned
    assert "inline code" in cleaned
    assert "#" not in cleaned
    assert "*" not in cleaned
    assert "`" not in cleaned


def test_sanitize_text_links_and_lists():
    sample = "- First item\n- Second item with [OpenAI](https://openai.com)"
    cleaned = sanitize_text_for_speech(sample)
    assert "First item" in cleaned
    assert "Second item with OpenAI" in cleaned
    assert "https://" not in cleaned
    assert "-" not in cleaned


def test_pyttsx3_speak_mock():
    tts = Pyttsx3TTS(TTSConfig(rate=180, volume=0.9))

    mock_engine = MagicMock()
    with patch.object(tts, "_init_engine", return_value=mock_engine):
        tts.speak("Hello from Chitti", wait=True)
        mock_engine.say.assert_called_with("Hello from Chitti")
        mock_engine.runAndWait.assert_called_once()
