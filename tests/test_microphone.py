"""Tests for Chitti Microphone Manager."""

from unittest.mock import patch, MagicMock
import numpy as np
from src.audio.microphone import MicrophoneManager, AudioCaptureError
from src.config import AudioConfig


def test_list_microphones_structure():
    mock_devices = [
        {"name": "Realtek Audio", "max_input_channels": 2, "default_samplerate": 48000.0},
        {"name": "Speakers", "max_input_channels": 0, "default_samplerate": 48000.0},
    ]
    with patch("src.audio.microphone.sd") as mock_sd:
        mock_sd.query_devices.return_value = mock_devices
        mock_sd.default.device = [0, 1]

        mics = MicrophoneManager.list_microphones()
        assert len(mics) == 1
        assert mics[0]["name"] == "Realtek Audio"
        assert mics[0]["channels"] == 2
        assert mics[0]["is_default"] is True


def test_get_selected_device():
    # Explicit configured index
    cfg = AudioConfig(microphone_index=3)
    mgr = MicrophoneManager(cfg)
    assert mgr.get_selected_device() == 3

    # Default fallback
    cfg_auto = AudioConfig(microphone_index=None)
    mgr_auto = MicrophoneManager(cfg_auto)
    with patch("src.audio.microphone.sd") as mock_sd:
        mock_sd.default.device = [1, 2]
        assert mgr_auto.get_selected_device() == 1


def test_record_speech_error_handling():
    with patch("src.audio.microphone.sd", None):
        mgr = MicrophoneManager()
        try:
            mgr.record_speech()
            assert False, "Should have raised AudioCaptureError"
        except AudioCaptureError as e:
            assert "sounddevice" in str(e)
