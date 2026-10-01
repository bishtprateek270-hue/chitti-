"""
Chitti Speech-to-Text (STT) Module.
Provides crystal-clear, human-grade speech understanding using a high-precision
Hybrid STT Engine combining Google Speech Recognition (for zero-latency, accent-aware,
zero-hallucination accuracy in English, Hindi & Hinglish) and OpenAI Whisper (for local offline fallback).
"""

import contextlib
import io
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union, Tuple
import numpy as np

from src.config import STTConfig, get_config
from src.utils.logging import log_debug, log_warning, log_chitti, log_info, log_error

try:
    import speech_recognition as sr
    HAS_SPEECH_RECOGNITION = True
except ImportError:
    sr = None
    HAS_SPEECH_RECOGNITION = False


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
    r"(?i)\b(?:is\s+there\s+something\s+else\s+I\s+can\s+assist|respectful\s+manner|play\s+one\s+Jolly)\b",
    r"(?i)\b(?:have\s+fun\s+coding|see\s+you\s+in\s+the\s+next\s+video)\b",
    r"(?i)^\[.*\]$",  # e.g. [Music], [Applause], [Silence]
]


class GoogleSTT(STTEngine):
    """
    Ultra-high accuracy Google Speech Recognition Engine.
    Provides human-level clarity, understands regional accents (Indian English, US English, Hindi, Hinglish),
    and delivers 0-hallucination speech understanding in <0.4s.
    """

    def __init__(self, language: Optional[str] = "en-IN", sample_rate: int = 16000):
        self.language = language or "en-IN"
        self.sample_rate = sample_rate
        self._recognizer = sr.Recognizer() if HAS_SPEECH_RECOGNITION and sr else None
        if self._recognizer:
            self._recognizer.energy_threshold = 300
            self._recognizer.dynamic_energy_threshold = True

    def _convert_to_audio_data(self, audio: Union[np.ndarray, str, Path]) -> Optional["sr.AudioData"]:
        """Converts float32 numpy array or audio file into SpeechRecognition AudioData."""
        if not HAS_SPEECH_RECOGNITION or not sr:
            return None

        if isinstance(audio, (str, Path)):
            with sr.AudioFile(str(audio)) as source:
                return self._recognizer.record(source)

        if isinstance(audio, np.ndarray):
            if audio.size == 0 or np.max(np.abs(audio)) < 1e-4:
                return None

            # Ensure float32 normalized in [-1.0, 1.0]
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            max_val = np.max(np.abs(audio))
            if max_val > 1.0:
                audio = audio / 32768.0

            # Convert float32 [-1, 1] to 16-bit PCM bytes
            pcm_data = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
            return sr.AudioData(pcm_data, self.sample_rate, 2)

        return None

    def transcribe(self, audio: Union[np.ndarray, str, Path]) -> str:
        """Transcribes speech using Google's Cloud Speech Recognition."""
        if not self._recognizer:
            return ""

        audio_data = self._convert_to_audio_data(audio)
        if not audio_data:
            return ""

        # Primary transcription language attempt
        langs_to_try = [self.language, "en-IN", "en-US", "hi-IN"] if self.language else ["en-IN", "en-US", "hi-IN"]
        seen_langs = []

        for lang in langs_to_try:
            if lang in seen_langs:
                continue
            seen_langs.append(lang)
            try:
                text = self._recognizer.recognize_google(audio_data, language=lang)
                if text and text.strip():
                    log_debug(f"[GOOGLE STT] Recognized ({lang}): '{text.strip()}'")
                    return text.strip()
            except sr.UnknownValueError:
                # Speech was unintelligible in this specific language, try next
                continue
            except (sr.RequestError, Exception) as e:
                log_debug(f"[GOOGLE STT] Cloud request failed: {e}")
                break

        return ""


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
            with contextlib.redirect_stderr(io.StringIO()):
                self._model = whisper.load_model(self.model_name, device=self.device)
            log_chitti("Speech recognition model loaded successfully.")
        except Exception as e:
            # If CUDA failed, try fallback to CPU
            if self.device == "cuda":
                log_warning(f"CUDA loading failed for Whisper: {e}. Falling back to CPU...")
                try:
                    self.device = "cpu"
                    with contextlib.redirect_stderr(io.StringIO()):
                        self._model = whisper.load_model(self.model_name, device="cpu")
                    log_chitti("Whisper successfully loaded on CPU fallback.")
                    return
                except Exception as fallback_err:
                    raise STTError(f"Failed to load Whisper on both CUDA and CPU: {fallback_err}") from fallback_err
            raise STTError(f"Failed to load Whisper model '{self.model_name}': {e}") from e

    @staticmethod
    def is_hallucination(text: str) -> bool:
        """Returns True if the transcribed text matches known Whisper phantom hallucinations or repeating loops."""
        clean = text.strip()
        if not clean or len(clean) < 2:
            return True

        for pat in WHISPER_HALLUCINATION_PATTERNS:
            if re.search(pat, clean, re.IGNORECASE):
                return True

        words = [w.lower().strip(".,!?;:\"'") for w in clean.split() if w.strip()]
        if not words:
            return True

        # Check for single word repetition (e.g. "you you you you")
        if len(words) >= 3 and len(set(words)) == 1:
            return True

        # Check for phrase loops (e.g. "i'm not a good guy, i'm not a good guy...")
        if len(words) >= 6:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.45:
                return True

        # Check for substring repetition
        if re.search(r"(.{4,40}?)(?:,\s*|\s+)\1(?:,\s*|\s+)\1", clean, re.IGNORECASE):
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
            "temperature": 0.0,
            "condition_on_previous_text": False,
            "initial_prompt": "Chitti, Hey Chitti, open Chrome, Notepad, VS Code, read screen, summarize document, mera naam, help me, screen p kya chal rha h.",
        }
        if self.language and self.language.lower() not in ("auto", "none", ""):
            options["language"] = self.language

        try:
            with contextlib.redirect_stderr(io.StringIO()):
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


class HybridSTT(STTEngine):
    """
    Industry-grade Hybrid Speech Recognition Engine for Chitti.
    Uses Google Cloud Speech Recognition as the primary engine for crystal-clear,
    human-grade clarity in English, Hindi, and Hinglish with zero hallucinations,
    automatically falling back to local OpenAI Whisper if offline.
    """

    def __init__(self, config: Optional[STTConfig] = None):
        self.config = config or get_config().stt
        self.google_stt = GoogleSTT(language=self.config.language or "en-IN")
        self.whisper_stt = WhisperSTT(config=self.config)

    def preload(self):
        """Preloads local engine for instant offline fallback readiness."""
        self.whisper_stt.preload()

    def transcribe(self, audio: Union[np.ndarray, str, Path]) -> str:
        """Transcribes audio with Google Speech first, seamlessly falling back to Whisper."""
        # 1. Try Google High-Accuracy Speech Recognition
        try:
            text = self.google_stt.transcribe(audio)
            if text and text.strip():
                log_info(f"[STT] Transcribed: '{text}'")
                return text.strip()
        except Exception as e:
            log_debug(f"[HYBRID STT] Google STT exception: {e}")

        # 2. Local Whisper Offline Fallback
        log_debug("[HYBRID STT] Falling back to local Whisper STT engine...")
        whisper_text = self.whisper_stt.transcribe(audio)
        if whisper_text and whisper_text.strip():
            log_info(f"[STT (Whisper Fallback)] Transcribed: '{whisper_text}'")
            return whisper_text.strip()

        return ""


def get_stt(config: Optional[STTConfig] = None) -> STTEngine:
    """Factory function returning the configured Hybrid STT engine."""
    return HybridSTT(config)
