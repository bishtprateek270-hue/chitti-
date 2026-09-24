"""Tests for Chitti Speech-to-Text module."""

from unittest.mock import patch, MagicMock
import numpy as np

from src.audio.stt import WhisperSTT, STTError
from src.config import STTConfig


def test_transcribe_empty_audio_returns_empty_string():
    stt = WhisperSTT()
    empty_audio = np.zeros(0, dtype=np.float32)
    assert stt.transcribe(empty_audio) == ""

    silent_audio = np.zeros(16000, dtype=np.float32)
    assert stt.transcribe(silent_audio) == ""


def test_transcribe_whisper_mock():
    stt = WhisperSTT(STTConfig(model="base", requested_device="cpu"))
    dummy_audio = np.random.uniform(-0.5, 0.5, 16000).astype(np.float32)

    mock_whisper_model = MagicMock()
    mock_whisper_model.transcribe.return_value = {"text": "Hello Chitti"}
    stt._model = mock_whisper_model

    result = stt.transcribe(dummy_audio)
    assert result == "Hello Chitti"
    mock_whisper_model.transcribe.assert_called_once()


def test_stt_dtype_normalization():
    stt = WhisperSTT(STTConfig(model="base", requested_device="cpu"))
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": "Testing int16"}
    stt._model = mock_model

    # Pass int16 array
    int16_audio = np.array([1000, -2000, 3000], dtype=np.int16)
    result = stt.transcribe(int16_audio)
    assert result == "Testing int16"
