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


def test_edge_neural_tts_fallback_delegation():
    from src.audio.tts import EdgeNeuralTTS

    mock_fallback = MagicMock()
    tts = EdgeNeuralTTS(TTSConfig(), fallback_engine=mock_fallback)

    # Simulate synthesis failure (e.g. 403 or network issue)
    with patch.object(tts, "_synthesize_async", side_effect=Exception("403 Forbidden")):
        tts.speak("Testing fallback speech", wait=True)
        mock_fallback.speak.assert_called_once()
        assert "Testing fallback speech" in mock_fallback.speak.call_args[0][0]


def test_edge_neural_tts_male_voice_defaults():
    from src.audio.tts import EdgeNeuralTTS
    tts = EdgeNeuralTTS(TTSConfig(voice_gender="male", voice_id=None))
    assert tts.DEFAULT_HINDI_VOICE == "hi-IN-MadhurNeural"
    assert tts.DEFAULT_HINGLISH_VOICE == "en-IN-PrabhatNeural"
    assert tts._detect_voice_for_text("नमस्ते, आप कैसे हैं?") == "hi-IN-MadhurNeural"
    assert tts._detect_voice_for_text("Hello, how are you?") == "en-IN-PrabhatNeural"


def test_edge_neural_tts_gender_switch():
    from src.audio.tts import EdgeNeuralTTS
    tts = EdgeNeuralTTS(TTSConfig(voice_gender="female", voice_id=None))
    assert tts._detect_voice_for_text("Hello!") == "en-IN-NeerjaNeural"
    tts.set_gender("male")
    assert tts.config.voice_gender == "male"
    assert tts.voice_id == "hi-IN-MadhurNeural"
    assert tts._detect_voice_for_text("Hello!") == "en-IN-PrabhatNeural"


def test_pyttsx3_male_voice_auto_selection():
    tts = Pyttsx3TTS(TTSConfig(voice_gender="male", voice_id=None))
    mock_voice_david = MagicMock(id="DAVID_VOICE_ID", name="Microsoft David Desktop", gender="Male")
    mock_voice_zira = MagicMock(id="ZIRA_VOICE_ID", name="Microsoft Zira Desktop", gender="Female")
    mock_engine = MagicMock()
    mock_engine.getProperty.return_value = [mock_voice_david, mock_voice_zira]

    with patch("src.audio.tts.pyttsx3.init", return_value=mock_engine):
        engine = tts._init_engine()
        mock_engine.setProperty.assert_any_call("voice", "DAVID_VOICE_ID")

