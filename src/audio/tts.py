"""
Chitti Text-to-Speech (TTS) Module.
Wraps local TTS engines (such as pyttsx3) with text sanitization, rate/volume control, and voice selection.
"""

import re
import threading
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

from src.config import TTSConfig, get_config
from src.utils.logging import log_debug, log_warning, log_chitti


class TTSError(Exception):
    """Base exception for TTS failures."""
    pass


class TTSEngine(ABC):
    """Abstract interface for Text-to-Speech engines."""

    @abstractmethod
    def speak(self, text: str, wait: bool = True) -> None:
        """Synthesizes and speaks text."""
        pass

    @abstractmethod
    def list_voices(self) -> List[Dict[str, Any]]:
        """Lists available voice profiles."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stops ongoing speech."""
        pass


def sanitize_text_for_speech(text: str) -> str:
    """
    Cleans markdown formatting, emojis, bullet markers, and code syntax
    so that the TTS output sounds natural and fluent.
    """
    if not text:
        return ""

    # Remove markdown links [text](url) -> text
    cleaned = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove code blocks ```...``` and inline code `...`
    cleaned = re.sub(r'```[\s\S]*?```', '', cleaned)
    cleaned = re.sub(r'`([^`]+)`', r'\1', cleaned)
    # Remove markdown headers (#, ##, etc.)
    cleaned = re.sub(r'^#{1,6}\s*', '', cleaned, flags=re.MULTILINE)
    # Remove asterisks and underscores (bold/italic)
    cleaned = re.sub(r'[\*_]{1,3}', '', cleaned)
    # Remove bullet markers (- , * , 1. )
    cleaned = re.sub(r'^\s*[-*+]\s+', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'^\s*\d+\.\s+', '', cleaned, flags=re.MULTILINE)
    # Collapse multiple whitespace/newlines
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


class Pyttsx3TTS(TTSEngine):
    """Local pyttsx3 Text-to-Speech implementation using SAPI5 on Windows."""

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or get_config().tts
        self.rate = self.config.rate
        self.volume = self.config.volume
        self.voice_id = self.config.voice_id
        self._lock = threading.Lock()

    def _init_engine(self):
        """Creates a fresh pyttsx3 engine instance safely."""
        if pyttsx3 is None:
            raise TTSError("pyttsx3 is not installed. Install via `pip install pyttsx3`.")
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
            if self.voice_id:
                engine.setProperty("voice", self.voice_id)
            return engine
        except Exception as e:
            raise TTSError(f"Failed to initialize pyttsx3 engine: {e}") from e

    def list_voices(self) -> List[Dict[str, Any]]:
        """Returns list of installed system voices."""
        if pyttsx3 is None:
            return []
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty("voices")
            voice_list = []
            for v in voices:
                voice_list.append({
                    "id": getattr(v, "id", ""),
                    "name": getattr(v, "name", ""),
                    "languages": getattr(v, "languages", []),
                    "gender": getattr(v, "gender", ""),
                    "age": getattr(v, "age", "")
                })
            engine.stop()
            return voice_list
        except Exception as e:
            log_warning(f"Could not query TTS voices: {e}")
            return []

    def speak(self, text: str, wait: bool = True) -> None:
        """
        Speaks the given text using the local TTS engine.
        Cleans markdown before speaking.
        """
        clean_text = sanitize_text_for_speech(text)
        if not clean_text:
            return

        def _run_speak():
            with self._lock:
                try:
                    engine = self._init_engine()
                    engine.say(clean_text)
                    engine.runAndWait()
                    engine.stop()
                except Exception as e:
                    log_warning(f"TTS playback encountered an error: {e}")

        if wait:
            _run_speak()
        else:
            thread = threading.Thread(target=_run_speak, daemon=True)
            thread.start()

    def stop(self) -> None:
        """Stops any active speech synthesis."""
        try:
            if pyttsx3 is not None:
                engine = pyttsx3.init()
                engine.stop()
        except Exception:
            pass


def get_tts(config: Optional[TTSConfig] = None) -> TTSEngine:
    """Factory function returning the configured TTS engine."""
    return Pyttsx3TTS(config)
