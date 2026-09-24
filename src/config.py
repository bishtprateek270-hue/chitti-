"""
Chitti Configuration Module.
Loads environment variables, validates settings, and provides runtime configuration.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load .env file from project root
ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)


def _detect_device(requested_device: str) -> str:
    """Detect if CUDA is available when 'auto' is requested."""
    req = requested_device.strip().lower()
    if req == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"
    return req


@dataclass
class LLMConfig:
    base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    )
    model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2")
    )
    timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60"))
    )

    def __post_init__(self):
        if self.base_url:
            self.base_url = self.base_url.rstrip("/")


@dataclass
class STTConfig:
    model: str = field(
        default_factory=lambda: os.getenv("STT_MODEL", "base")
    )
    requested_device: str = field(
        default_factory=lambda: os.getenv("STT_DEVICE", "auto")
    )
    language: Optional[str] = field(
        default_factory=lambda: os.getenv("STT_LANGUAGE", "en") or None
    )

    @property
    def device(self) -> str:
        """Resolved device ('cuda' or 'cpu')."""
        return _detect_device(self.requested_device)


@dataclass
class TTSConfig:
    engine: str = field(
        default_factory=lambda: os.getenv("TTS_ENGINE", "pyttsx3")
    )
    rate: int = field(
        default_factory=lambda: int(os.getenv("TTS_RATE", "175"))
    )
    volume: float = field(
        default_factory=lambda: float(os.getenv("TTS_VOLUME", "1.0"))
    )
    voice_id: Optional[str] = field(
        default_factory=lambda: os.getenv("TTS_VOICE_ID", "").strip() or None
    )


@dataclass
class AudioConfig:
    microphone_index: Optional[int] = field(
        default_factory=lambda: (
            int(os.getenv("MICROPHONE_DEVICE_INDEX"))
            if os.getenv("MICROPHONE_DEVICE_INDEX", "").strip()
            else None
        )
    )
    sample_rate: int = field(
        default_factory=lambda: int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
    )
    channels: int = field(
        default_factory=lambda: int(os.getenv("AUDIO_CHANNELS", "1"))
    )
    silence_threshold: float = field(
        default_factory=lambda: float(os.getenv("AUDIO_SILENCE_THRESHOLD", "0.015"))
    )
    silence_duration: float = field(
        default_factory=lambda: float(os.getenv("AUDIO_SILENCE_DURATION", "1.5"))
    )
    max_record_seconds: int = field(
        default_factory=lambda: int(os.getenv("AUDIO_MAX_RECORD_SECONDS", "30"))
    )


@dataclass
class MemoryConfig:
    enabled: bool = field(
        default_factory=lambda: os.getenv("MEMORY_ENABLED", "true").lower() in ("true", "1", "yes")
    )
    db_path: str = field(
        default_factory=lambda: os.getenv("MEMORY_DB_PATH", "data/memory/chitti_memory.db")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    top_k: int = field(
        default_factory=lambda: int(os.getenv("MEMORY_TOP_K", "5"))
    )
    similarity_threshold: float = field(
        default_factory=lambda: float(os.getenv("MEMORY_SIMILARITY_THRESHOLD", "0.35"))
    )
    importance_threshold: int = field(
        default_factory=lambda: int(os.getenv("MEMORY_IMPORTANCE_THRESHOLD", "1"))
    )


@dataclass
class AppConfig:
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper()
    )
    llm: LLMConfig = field(default_factory=LLMConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)


# Global singleton instance
config = AppConfig()


def get_config() -> AppConfig:
    """Returns the application configuration instance."""
    return config


def reload_config() -> AppConfig:
    """Reloads configuration from .env and environment."""
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    global config
    config = AppConfig()
    return config
