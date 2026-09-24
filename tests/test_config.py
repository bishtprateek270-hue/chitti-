"""Tests for Chitti Configuration module."""

import os
from src.config import AppConfig, LLMConfig, STTConfig, TTSConfig, AudioConfig, _detect_device


def test_default_config_instantiation():
    """Verifies default config fields and types."""
    cfg = AppConfig()
    assert isinstance(cfg.llm, LLMConfig)
    assert isinstance(cfg.stt, STTConfig)
    assert isinstance(cfg.tts, TTSConfig)
    assert isinstance(cfg.audio, AudioConfig)
    assert cfg.log_level in ("DEBUG", "INFO", "WARNING", "ERROR")


def test_stt_device_resolution():
    """Verifies that 'auto' resolves to 'cuda' or 'cpu' dynamically."""
    device = _detect_device("auto")
    assert device in ("cuda", "cpu")

    explicit_cpu = _detect_device("cpu")
    assert explicit_cpu == "cpu"

    explicit_cuda = _detect_device("cuda")
    assert explicit_cuda == "cuda"


def test_llm_config_values():
    """Verifies LLM configuration formatting."""
    llm_cfg = LLMConfig(base_url="http://localhost:11434/", model="llama3.2", timeout_seconds=45)
    assert llm_cfg.base_url == "http://localhost:11434"
    assert llm_cfg.model == "llama3.2"
    assert llm_cfg.timeout_seconds == 45


def test_audio_config_defaults():
    """Verifies audio sample rate and silence parameters."""
    audio_cfg = AudioConfig()
    assert audio_cfg.sample_rate == 16000
    assert audio_cfg.channels == 1
    assert audio_cfg.silence_threshold > 0
    assert audio_cfg.silence_duration > 0
