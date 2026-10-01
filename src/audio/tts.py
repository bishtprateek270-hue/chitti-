"""
Chitti Text-to-Speech (TTS) Module.
Provides high-fidelity, human-grade speech synthesis in English, Hindi, and Hinglish
using Microsoft Edge Neural TTS with automatic language adaptation and local SAPI5 fallback.
"""

import asyncio
import ctypes
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

try:
    import edge_tts
    HAS_EDGE_TTS = True
except ImportError:
    edge_tts = None
    HAS_EDGE_TTS = False

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

from src.config import TTSConfig, get_config
from src.utils.logging import log_debug, log_warning, log_chitti, log_info


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
    so that the TTS output sounds natural, fluent, and human.
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
    # Remove terminal/system log tags like [CHITTI], [THINKING]
    cleaned = re.sub(r'\[(?:CHITTI|THINKING|VISION|AGENT|INFO|STATE|WARNING|ERROR|GOAL|TARGET)\]', '', cleaned, flags=re.IGNORECASE)
    # Collapse multiple whitespace/newlines
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


class EdgeNeuralTTS(TTSEngine):
    """
    State-of-the-Art Microsoft Edge Neural TTS Engine.
    Speaks fluent, crystal-clear, human-grade Hindi, Hinglish, and English.
    """

    # High-quality neural voices
    DEFAULT_HINDI_VOICE = "hi-IN-SwaraNeural"       # Female, natural, fluent Hindi/Hinglish
    DEFAULT_HINGLISH_VOICE = "en-IN-NeerjaNeural"   # Female, clear Indian English & Hinglish
    DEFAULT_MALE_VOICE = "hi-IN-MadhurNeural"       # Male, natural Hindi/Hinglish

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or get_config().tts
        self.voice_id = self.config.voice_id or self.DEFAULT_HINDI_VOICE
        self.volume = self.config.volume
        self.rate = self.config.rate
        self._is_playing = False
        self._lock = threading.Lock()
        self._alias = f"chitti_tts_{os.getpid()}"

    def _detect_voice_for_text(self, text: str) -> str:
        """Selects the best natural human neural voice based on text language."""
        if self.config.voice_id and self.config.voice_id.strip():
            return self.config.voice_id.strip()

        # Check for Devanagari Hindi characters (U+0900 to U+097F)
        has_devanagari = bool(re.search(r'[\u0900-\u097F]', text))
        # Check for romanized Hinglish keywords
        has_hinglish = bool(re.search(r'(?i)\b(?:kya|hai|hoon|karo|batao|kaise|mera|meri|mere|aap|tum|chalao|kholo|banao)\b', text))

        if has_devanagari or has_hinglish:
            return self.DEFAULT_HINDI_VOICE
        return self.DEFAULT_HINGLISH_VOICE

    def _play_audio_file(self, file_path: str):
        """Plays audio using Windows native Multimedia API with instant cancellation support."""
        if not os.path.exists(file_path):
            return

        winmm = ctypes.windll.winmm
        alias = self._alias

        # Close any previous instance
        winmm.mciSendStringW(f'close {alias}', None, 0, 0)

        # Open and play
        open_cmd = f'open "{file_path}" type mpegvideo alias {alias}'
        res = winmm.mciSendStringW(open_cmd, None, 0, 0)
        if res != 0:
            log_warning(f"[TTS] MCI open failed (code: {res})")
            return

        self._is_playing = True
        play_cmd = f'play {alias} wait'
        winmm.mciSendStringW(play_cmd, None, 0, 0)

        # Cleanup
        winmm.mciSendStringW(f'close {alias}', None, 0, 0)
        self._is_playing = False

    async def _synthesize_async(self, text: str, voice: str, output_path: str):
        """Asynchronously synthesizes speech to an MP3 file."""
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate="+0%",
            volume="+0%",
        )
        await communicate.save(output_path)

    def speak(self, text: str, wait: bool = True) -> None:
        """
        Synthesizes text into high-fidelity neural speech and plays it.
        """
        if not HAS_EDGE_TTS or edge_tts is None:
            raise TTSError("edge-tts package is not installed.")

        clean_text = sanitize_text_for_speech(text)
        if not clean_text:
            return

        voice = self._detect_voice_for_text(clean_text)

        def _run_speak():
            with self._lock:
                temp_file = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
                        temp_file = tf.name

                    # Run async synthesis in event loop
                    asyncio.run(self._synthesize_async(clean_text, voice, temp_file))
                    self._play_audio_file(temp_file)
                except Exception as e:
                    log_warning(f"[TTS] Neural speech synthesis failed: {e}")
                    raise TTSError(f"Edge Neural TTS failed: {e}") from e
                finally:
                    if temp_file and os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                        except Exception:
                            pass

        if wait:
            _run_speak()
        else:
            t = threading.Thread(target=_run_speak, daemon=True, name="EdgeTTSWorker")
            t.start()

    def list_voices(self) -> List[Dict[str, Any]]:
        """Lists recommended human neural voices."""
        return [
            {"id": "hi-IN-SwaraNeural", "name": "Swara (Hindi/Hinglish - Female, Natural)", "languages": ["hi-IN", "en-IN"]},
            {"id": "hi-IN-MadhurNeural", "name": "Madhur (Hindi/Hinglish - Male, Natural)", "languages": ["hi-IN", "en-IN"]},
            {"id": "en-IN-NeerjaNeural", "name": "Neerja (Indian English / Hinglish - Female)", "languages": ["en-IN"]},
            {"id": "en-IN-PrabhatNeural", "name": "Prabhat (Indian English / Hinglish - Male)", "languages": ["en-IN"]},
            {"id": "en-US-JennyNeural", "name": "Jenny (US English - Female)", "languages": ["en-US"]},
        ]

    def stop(self) -> None:
        """Immediately stops ongoing speech playback."""
        if ctypes and hasattr(ctypes, "windll"):
            ctypes.windll.winmm.mciSendStringW(f'stop {self._alias}', None, 0, 0)
            ctypes.windll.winmm.mciSendStringW(f'close {self._alias}', None, 0, 0)
        self._is_playing = False


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
        """Speaks the given text using local SAPI5."""
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
        """Stops pyttsx3 playback."""
        pass


class HybridTTS(TTSEngine):
    """
    Hybrid Text-to-Speech Engine for Chitti.
    Prioritizes ultra-natural Microsoft Edge Neural TTS (for human-grade Hindi, Hinglish,
    and English), seamlessly falling back to local SAPI5 if offline.
    """

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or get_config().tts
        self.edge_tts = EdgeNeuralTTS(config=self.config) if HAS_EDGE_TTS else None
        self.fallback_tts = Pyttsx3TTS(config=self.config)

    def speak(self, text: str, wait: bool = True) -> None:
        """Speaks using Neural TTS first, falling back to local SAPI5 if offline."""
        if self.edge_tts:
            try:
                self.edge_tts.speak(text, wait=wait)
                return
            except Exception as e:
                log_debug(f"[HYBRID TTS] Edge Neural TTS exception, falling back: {e}")

        # Fallback to local SAPI5
        self.fallback_tts.speak(text, wait=wait)

    def list_voices(self) -> List[Dict[str, Any]]:
        """Returns list of voices available."""
        if self.edge_tts:
            return self.edge_tts.list_voices()
        return self.fallback_tts.list_voices()

    def stop(self) -> None:
        """Stops ongoing speech playback."""
        if self.edge_tts:
            try:
                self.edge_tts.stop()
            except Exception:
                pass
        self.fallback_tts.stop()


def get_tts(config: Optional[TTSConfig] = None) -> TTSEngine:
    """Factory function returning the configured TTS engine."""
    cfg = config or get_config().tts
    if cfg.engine.lower() in ("edge-tts", "neural", "edge", "hybrid"):
        return HybridTTS(cfg)
    if cfg.engine.lower() == "pyttsx3":
        return Pyttsx3TTS(cfg)
    return HybridTTS(cfg)
