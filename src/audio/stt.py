"""
Chitti Speech-to-Text (STT) Module.
Wraps OpenAI Whisper with GPU/CPU acceleration, eager preloading,
robust error handling, audio normalization, and hallucination filtering.
"""

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union, Tuple
import numpy as np

from src.config import STTConfig, get_config
from src.utils.logging import log_debug, log_warning, log_chitti


class STTError(Exception):
    """Base exception for STT failures."""
    pass


class STTEngine(ABC):
    """Abstract interface for Speech-to-Text engines."""

    @abstractmethod
    def transcribe(self, audio: Union[np.ndarray, str, Path]) -> str:
        """Transcribes given audio array (float32 at 16kHz) or audio file to text."""
        pass


# Common Whisper subtitle hallucination artifacts on silence / noise
WHISPER_HALLUCINATION_PATTERNS = [
    r"(?i)\b(?:thanks\s+for\s+watching|thank\s+you\s+for\s+watching|subscribe\s+to\s+my\s+channel)\b",
    r"(?i)\b(?:subtitles\s+by|translated\s+by|captioned\s+by)\b",
    r"(?i)\b(?:humans\s+are\s+so\s+hard|oh\s+julian)\b",
    r"(?i)^\[.*\]$",  # e.g. [Music], [Applause], [Silence]
]


class WhisperSTT(STTEngine):
    """OpenAI Whisper STT Engine with CUDA/CPU execution and hallucination rejection."""

    def __init__(self, config: Optional[STTConfig] = None):
        self.config = config or get_config().stt
        self.model_name = self.config.model
        self.device = self.config.device
        self.language = self.config.language
        self._model = None

    def preload(self):
        """Eagerly loads the Whisper model into memory at startup."""
        self._load_model()

    def _load_model(self):
        """Loads the Whisper model into GPU or CPU memory."""
        if self._model is not None:
            return

        try:
            import whisper
        except ImportError as e:
            raise STTError(
                "openai-whisper package is not installed. Install it via `pip install openai-whisper`."
            ) from e

        try:
            log_chitti(f"Loading speech recognition model ({self.model_name}) on {self.device.upper()}...")
            self._model = whisper.load_model(self.model_name, device=self.device)
            log_chitti("Speech recognition model loaded successfully.")
        except Exception as e:
            # If CUDA failed, try fallback to CPU
            if self.device == "cuda":
                log_warning(f"CUDA loading failed for Whisper: {e}. Falling back to CPU...")
                try:
                    self.device = "cpu"
                    self._model = whisper.load_model(self.model_name, device="cpu")
                    log_chitti("Whisper successfully loaded on CPU fallback.")
                    return
                except Exception as fallback_err:
                    raise STTError(f"Failed to load Whisper on both CUDA and CPU: {fallback_err}") from fallback_err
            raise STTError(f"Failed to load Whisper model '{self.model_name}': {e}") from e

    @staticmethod
    def is_hallucination(text: str) -> bool:
        """Returns True if the transcribed text matches known Whisper phantom hallucinations."""
        clean = text.strip()
        if not clean:
            return True

        for pat in WHISPER_HALLUCINATION_PATTERNS:
            if re.search(pat, clean):
                return True

        # Check for extreme word repetition (e.g. "you you you you you")
        words = clean.split()
        if len(words) >= 4 and len(set(words)) == 1:
            return True

        return False

    def transcribe(self, audio: Union[np.ndarray, str, Path]) -> str:
        """
        Transcribes audio to text with pre-filtering and post-processing.
        Accepts:
            - np.ndarray: 1D float32 array sampled at 16kHz
            - str / Path: path to audio file
        Returns:
            - Transcribed string (or empty string if nothing detected / silence)
        """
        self._load_model()

        # Handle empty or trivial array
        if isinstance(audio, np.ndarray):
            if audio.size == 0 or np.max(np.abs(audio)) < 1e-4:
                return ""
            # Ensure float32 dtype
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            # Rescale if needed (if in int16 range)
            max_val = np.max(np.abs(audio))
            if max_val > 1.0:
                audio = audio / 32768.0

        if isinstance(audio, Path):
            audio = str(audio)

        fp16 = (self.device == "cuda")

        options = {
            "fp16": fp16,
            "verbose": False,
        }
        if self.language:
            options["language"] = self.language

        try:
            result = self._model.transcribe(audio, **options)
            text = result.get("text", "").strip()

            if self.is_hallucination(text):
                log_debug(f"Rejected Whisper hallucination: '{text}'")
                return ""

            log_debug(f"Whisper transcription: '{text}'")
            return text
        except Exception as e:
            log_warning(f"Whisper transcription encountered an issue: {e}")
            return ""


def get_stt(config: Optional[STTConfig] = None) -> STTEngine:
    """Factory function returning the configured STT engine."""
    return WhisperSTT(config)
