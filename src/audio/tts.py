"""
Chitti Text-to-Speech (TTS) Module & Centralized Speech Controller.
Provides high-fidelity, human-grade speech synthesis in English, Hindi, and Hinglish
using Microsoft Edge Neural TTS with automatic language adaptation, local SAPI5 fallback,
and sub-300ms instant playback interruption and cancellation token synchronization.
"""

import asyncio
import ctypes
import os
from pathlib import Path
import queue
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
from src.core.state_manager import get_state_manager, SystemState
from src.utils.logging import log_debug, log_warning, log_chitti, log_info


class TTSError(Exception):
    """Base exception for TTS failures."""
    pass


class TTSEngine(ABC):
    """Abstract interface for Text-to-Speech engines."""

    @abstractmethod
    def speak(self, text: str, wait: bool = True, request_id: Optional[int] = None) -> None:
        """Synthesizes and speaks text."""
        pass

    @abstractmethod
    def list_voices(self) -> List[Dict[str, Any]]:
        """Lists available voice profiles."""
        pass

    @abstractmethod
    def stop(self) -> float:
        """Stops ongoing speech immediately. Returns interruption latency in milliseconds."""
        pass

    @abstractmethod
    def is_speaking(self) -> bool:
        """Returns True if speech playback or synthesis is currently active."""
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
    Supports sub-300ms playback interruption and cancellation token checking.
    """

    # High-quality neural voices
    MALE_HINDI_VOICE = "hi-IN-MadhurNeural"       # Male, natural Hindi/Hinglish
    MALE_ENGLISH_VOICE = "en-IN-PrabhatNeural"    # Male, clear Indian English & Hinglish
    MALE_US_VOICE = "en-US-ChristopherNeural"     # Male, US English

    FEMALE_HINDI_VOICE = "hi-IN-SwaraNeural"      # Female, natural Hindi/Hinglish
    FEMALE_ENGLISH_VOICE = "en-IN-NeerjaNeural"   # Female, clear Indian English & Hinglish
    FEMALE_US_VOICE = "en-US-JennyNeural"         # Female, US English

    DEFAULT_HINDI_VOICE = MALE_HINDI_VOICE
    DEFAULT_HINGLISH_VOICE = MALE_ENGLISH_VOICE
    DEFAULT_MALE_VOICE = MALE_HINDI_VOICE
    DEFAULT_FEMALE_VOICE = FEMALE_HINDI_VOICE

    def __init__(self, config: Optional[TTSConfig] = None, fallback_engine: Optional[TTSEngine] = None):
        self.config = config or get_config().tts
        self.fallback_engine = fallback_engine
        self.voice_id = self.config.voice_id or self.DEFAULT_HINDI_VOICE
        self.volume = self.config.volume
        self.rate = self.config.rate
        self._is_playing = False
        self._lock = threading.Lock()
        self._alias = f"chitti_tts_{os.getpid()}"

        # Interruption tracking
        self._current_gen_id = 0
        self._cancel_requested = threading.Event()
        self._last_interrupt_latency_ms = 0.0

    def set_voice(self, voice_id: str):
        """Switches the active neural voice on the fly."""
        self.voice_id = voice_id.strip()
        self.config.voice_id = self.voice_id
        log_chitti(f"[TTS] Voice switched to: {self.voice_id}")

    def set_gender(self, gender: str):
        """Switches voice gender between 'male' and 'female'."""
        gender = gender.strip().lower()
        if gender == "male":
            self.voice_id = self.MALE_HINDI_VOICE
            self.config.voice_id = None
            self.config.voice_gender = "male"
        else:
            self.voice_id = self.FEMALE_HINDI_VOICE
            self.config.voice_id = None
            self.config.voice_gender = "female"
        log_chitti(f"[TTS] Voice gender switched to: {gender} ({self.voice_id})")

    def is_speaking(self) -> bool:
        return self._is_playing

    def _detect_voice_for_text(self, text: str) -> str:
        """Selects the best natural human neural voice based on text language and configured voice/gender."""
        is_male = getattr(self.config, "voice_gender", "male").lower() != "female"
        cfg_voice = (self.config.voice_id or "").strip()

        if cfg_voice.lower() == "male":
            is_male = True
            cfg_voice = ""
        elif cfg_voice.lower() == "female":
            is_male = False
            cfg_voice = ""
        elif cfg_voice.lower() in ("auto", "default", "none"):
            cfg_voice = ""

        # If a specific voice ID is configured
        if cfg_voice:
            return cfg_voice

        # Check for Devanagari Hindi characters (U+0900 to U+097F)
        has_devanagari = bool(re.search(r'[\u0900-\u097F]', text))
        # Check for romanized Hinglish keywords
        has_hinglish = bool(re.search(r'(?i)\b(?:kya|hai|hoon|karo|batao|kaise|mera|meri|mere|aap|tum|chalao|kholo|banao|shukriya|namaste|theek|suno|bolna|awaz)\b', text))

        if has_devanagari or has_hinglish:
            return self.MALE_HINDI_VOICE if is_male else self.FEMALE_HINDI_VOICE
        return self.MALE_ENGLISH_VOICE if is_male else self.FEMALE_ENGLISH_VOICE

    def _play_audio_file(self, file_path: str, gen_id: int):
        """Plays audio using Windows native Multimedia API with instant cancellation support."""
        if not os.path.exists(file_path):
            return

        if self._cancel_requested.is_set() or gen_id != self._current_gen_id:
            return

        winmm = getattr(ctypes, "windll", None)
        if not winmm or not hasattr(winmm, "winmm"):
            return
        winmm_api = winmm.winmm
        alias = self._alias

        # Close any previous instance
        winmm_api.mciSendStringW(f'close {alias}', None, 0, 0)

        # Open
        open_cmd = f'open "{file_path}" type mpegvideo alias {alias}'
        res = winmm_api.mciSendStringW(open_cmd, None, 0, 0)
        if res != 0:
            log_warning(f"[TTS] MCI open failed (code: {res})")
            return

        if self._cancel_requested.is_set() or gen_id != self._current_gen_id:
            winmm_api.mciSendStringW(f'close {alias}', None, 0, 0)
            return

        self._is_playing = True
        state_mgr = get_state_manager()
        state_mgr.set_state(SystemState.SPEAKING)

        # Start non-blocking play
        play_cmd = f'play {alias}'
        winmm_api.mciSendStringW(play_cmd, None, 0, 0)

        # Monitor playback in tight polling loop with cancellation checking
        status_buf = ctypes.create_unicode_buffer(128)
        try:
            while self._is_playing and not self._cancel_requested.is_set() and gen_id == self._current_gen_id:
                winmm_api.mciSendStringW(f'status {alias} mode', status_buf, 128, 0)
                status_val = status_buf.value.lower().strip()
                if status_val not in ("playing", "seeking"):
                    break
                time.sleep(0.015)  # 15ms poll allows instant response to stop
        finally:
            winmm_api.mciSendStringW(f'stop {alias}', None, 0, 0)
            winmm_api.mciSendStringW(f'close {alias}', None, 0, 0)
            self._is_playing = False
            if state_mgr.get_state() == SystemState.SPEAKING:
                state_mgr.set_state(SystemState.AWAKE)

    async def _synthesize_async(self, text: str, voice: str, output_path: str):
        """Asynchronously synthesizes speech to an MP3 file."""
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate="+0%",
            volume="+0%",
        )
        await communicate.save(output_path)

    def speak(self, text: str, wait: bool = True, request_id: Optional[int] = None) -> None:
        """Synthesizes text into high-fidelity neural speech and plays it."""
        if not HAS_EDGE_TTS or edge_tts is None:
            raise TTSError("edge-tts package is not installed.")

        clean_text = sanitize_text_for_speech(text)
        if not clean_text:
            return

        # Check state manager
        state_mgr = get_state_manager()
        if not state_mgr.can_speak(request_id):
            log_debug(f"[TTS] Suppressed speech output: request #{request_id} is invalid, sleeping, or interrupted.")
            return

        # Prepare generation cycle
        with self._lock:
            self._current_gen_id += 1
            gen_id = self._current_gen_id
            self._cancel_requested.clear()

        voice = self._detect_voice_for_text(clean_text)

        def _run_speak():
            temp_file = None
            try:
                if self._cancel_requested.is_set() or gen_id != self._current_gen_id or not state_mgr.can_speak(request_id):
                    return

                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
                    temp_file = tf.name

                # Run async synthesis
                asyncio.run(self._synthesize_async(clean_text, voice, temp_file))

                # Check cancellation immediately after download/synthesis
                if self._cancel_requested.is_set() or gen_id != self._current_gen_id or not state_mgr.can_speak(request_id):
                    log_debug(f"[TTS] Speech discarded after synthesis: Stop was requested for request #{request_id}.")
                    return

                self._play_audio_file(temp_file, gen_id)

            except Exception as e:
                if not self._cancel_requested.is_set() and gen_id == self._current_gen_id:
                    log_warning(f"[TTS] Neural speech synthesis failed: {e}. Attempting fallback to local TTS.")
                    if self.fallback_engine:
                        try:
                            self.fallback_engine.speak(clean_text, wait=wait, request_id=request_id)
                            return
                        except Exception as fb_err:
                            log_warning(f"[TTS] Fallback TTS playback also failed: {fb_err}")
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
            t = threading.Thread(target=_run_speak, daemon=True, name=f"EdgeTTSWorker-{gen_id}")
            t.start()

    def list_voices(self) -> List[Dict[str, Any]]:
        """Lists recommended human neural voices."""
        return [
            {"id": "hi-IN-MadhurNeural", "name": "Madhur (Hindi/Hinglish/English - Male, Natural)", "languages": ["hi-IN", "en-IN"], "gender": "male"},
            {"id": "en-IN-PrabhatNeural", "name": "Prabhat (Indian English - Male, Natural)", "languages": ["en-IN"], "gender": "male"},
            {"id": "en-US-ChristopherNeural", "name": "Christopher (US English - Male)", "languages": ["en-US"], "gender": "male"},
            {"id": "hi-IN-SwaraNeural", "name": "Swara (Hindi/Hinglish - Female, Natural)", "languages": ["hi-IN", "en-IN"], "gender": "female"},
            {"id": "en-IN-NeerjaNeural", "name": "Neerja (Indian English / Hinglish - Female)", "languages": ["en-IN"], "gender": "female"},
            {"id": "en-US-JennyNeural", "name": "Jenny (US English - Female)", "languages": ["en-US"], "gender": "female"},
        ]

    def stop(self) -> float:
        """Immediately stops ongoing speech playback and cancels pending synthesis."""
        t0 = time.perf_counter()
        with self._lock:
            self._current_gen_id += 1  # Invalidate in-flight synthesis
            self._cancel_requested.set()

        winmm = getattr(ctypes, "windll", None)
        if winmm and hasattr(winmm, "winmm"):
            try:
                winmm.winmm.mciSendStringW(f'stop {self._alias}', None, 0, 0)
                winmm.winmm.mciSendStringW(f'close {self._alias}', None, 0, 0)
            except Exception:
                pass

        self._is_playing = False
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._last_interrupt_latency_ms = elapsed_ms
        log_info(f"[CHITTI][TTS] Playback stopped in {elapsed_ms:.2f}ms.")
        return elapsed_ms


class Pyttsx3TTS(TTSEngine):
    """Local pyttsx3 Text-to-Speech implementation using SAPI5 on Windows."""

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or get_config().tts
        self.rate = self.config.rate
        self.volume = self.config.volume
        self.voice_id = self.config.voice_id
        self._lock = threading.Lock()
        self._is_playing = False
        self._current_engine = None
        self._cancel_requested = threading.Event()

    def is_speaking(self) -> bool:
        return self._is_playing

    def set_voice(self, voice_id: str):
        self.voice_id = voice_id.strip()
        self.config.voice_id = self.voice_id

    def set_gender(self, gender: str):
        gender = gender.strip().lower()
        self.config.voice_gender = gender
        try:
            voices = self.list_voices()
            for v in voices:
                name_lower = v.get("name", "").lower()
                gender_lower = v.get("gender", "").lower()
                if gender == "male" and ("david" in name_lower or gender_lower == "male"):
                    self.voice_id = v["id"]
                    self.config.voice_id = self.voice_id
                    break
                elif gender == "female" and ("zira" in name_lower or gender_lower == "female"):
                    self.voice_id = v["id"]
                    self.config.voice_id = self.voice_id
                    break
        except Exception:
            pass

    def _init_engine(self):
        """Creates a fresh pyttsx3 engine instance safely."""
        if pyttsx3 is None:
            raise TTSError("pyttsx3 is not installed. Install via `pip install pyttsx3`.")
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
            target_voice = self.voice_id
            if not target_voice:
                is_male = getattr(self.config, "voice_gender", "male").lower() != "female"
                try:
                    voices = engine.getProperty("voices")
                    for v in voices:
                        name_lower = getattr(v, "name", "").lower()
                        gender_lower = getattr(v, "gender", "").lower()
                        if is_male and ("david" in name_lower or gender_lower == "male"):
                            target_voice = v.id
                            break
                        elif not is_male and ("zira" in name_lower or gender_lower == "female"):
                            target_voice = v.id
                            break
                except Exception:
                    pass
            if target_voice:
                engine.setProperty("voice", target_voice)
            return engine
        except Exception as e:
            raise TTSError(f"Failed to initialize pyttsx3 engine: {e}") from e

    def list_voices(self) -> List[Dict[str, Any]]:
        """Returns list of installed system voices."""
        if pyttsx3 is None:
            return []
        try:
            engine = self._init_engine()
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

    def speak(self, text: str, wait: bool = True, request_id: Optional[int] = None) -> None:
        """Speaks the given text using local SAPI5."""
        clean_text = sanitize_text_for_speech(text)
        if not clean_text:
            return

        state_mgr = get_state_manager()
        if not state_mgr.can_speak(request_id):
            return

        self._cancel_requested.clear()

        def _run_speak():
            with self._lock:
                if self._cancel_requested.is_set():
                    return
                try:
                    self._is_playing = True
                    state_mgr.set_state(SystemState.SPEAKING)
                    engine = self._init_engine()
                    self._current_engine = engine
                    engine.say(clean_text)
                    engine.runAndWait()
                except Exception as e:
                    if not self._cancel_requested.is_set():
                        log_warning(f"TTS playback encountered an error: {e}")
                finally:
                    self._is_playing = False
                    self._current_engine = None
                    if state_mgr.get_state() == SystemState.SPEAKING:
                        state_mgr.set_state(SystemState.AWAKE)

        if wait:
            _run_speak()
        else:
            thread = threading.Thread(target=_run_speak, daemon=True, name="Pyttsx3Worker")
            thread.start()

    def stop(self) -> float:
        """Stops pyttsx3 playback immediately."""
        t0 = time.perf_counter()
        self._cancel_requested.set()
        if self._current_engine:
            try:
                self._current_engine.stop()
            except Exception:
                pass
        self._is_playing = False
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        log_info(f"[CHITTI][TTS] Local SAPI5 stopped in {elapsed_ms:.2f}ms.")
        return elapsed_ms


class HybridTTS(TTSEngine):
    """
    Hybrid Text-to-Speech Engine for Chitti.
    Prioritizes ultra-natural Microsoft Edge Neural TTS, seamlessly falling back to local SAPI5 if offline.
    """

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or get_config().tts
        self.fallback_tts = Pyttsx3TTS(config=self.config)
        self.edge_tts = EdgeNeuralTTS(config=self.config, fallback_engine=self.fallback_tts) if HAS_EDGE_TTS else None

    def is_speaking(self) -> bool:
        if self.edge_tts and self.edge_tts.is_speaking():
            return True
        return self.fallback_tts.is_speaking()

    def speak(self, text: str, wait: bool = True, request_id: Optional[int] = None) -> None:
        """Speaks using Neural TTS first, falling back to local SAPI5 if offline."""
        if self.edge_tts:
            try:
                self.edge_tts.speak(text, wait=wait, request_id=request_id)
                return
            except Exception as e:
                log_debug(f"[HYBRID TTS] Edge Neural TTS exception, falling back: {e}")

        # Fallback to local SAPI5
        self.fallback_tts.speak(text, wait=wait, request_id=request_id)

    def set_voice(self, voice_id: str):
        """Sets voice on Edge Neural and fallback engines."""
        if self.edge_tts and hasattr(self.edge_tts, "set_voice"):
            self.edge_tts.set_voice(voice_id)
        if self.fallback_tts and hasattr(self.fallback_tts, "set_voice"):
            self.fallback_tts.set_voice(voice_id)

    def set_gender(self, gender: str):
        """Sets gender on Edge Neural and fallback engines."""
        if self.edge_tts and hasattr(self.edge_tts, "set_gender"):
            self.edge_tts.set_gender(gender)
        if self.fallback_tts and hasattr(self.fallback_tts, "set_gender"):
            self.fallback_tts.set_gender(gender)

    def list_voices(self) -> List[Dict[str, Any]]:
        """Returns list of voices available."""
        if self.edge_tts:
            return self.edge_tts.list_voices()
        return self.fallback_tts.list_voices()

    def stop(self) -> float:
        """Stops ongoing speech playback."""
        latencies = []
        if self.edge_tts:
            try:
                latencies.append(self.edge_tts.stop())
            except Exception:
                pass
        try:
            latencies.append(self.fallback_tts.stop())
        except Exception:
            pass
        return max(latencies) if latencies else 0.0


class SpeechController:
    """
    Centralized, thread-safe Speech Controller.
    Suggested interface:
        - speak(text, wait=False)
        - stop_speaking()
        - clear_speech_queue()
        - is_speaking()
        - shutdown()
    Guarantees:
        - Instant interruption of ongoing playback.
        - Discarding pending speech tasks in the queue.
        - Prevents multiple workers speaking simultaneously.
        - Prevents cancelled speech from resuming unexpectedly.
    """

    def __init__(self, engine: Optional[TTSEngine] = None, config: Optional[TTSConfig] = None):
        self.engine = engine or get_tts(config)
        self._queue: queue.Queue = queue.Queue()
        self._is_running = True
        self._lock = threading.Lock()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="ChittiSpeechController")
        self._worker_thread.start()

    def set_voice(self, voice_id: str):
        """Changes voice dynamically on active speech engine."""
        if hasattr(self.engine, "set_voice"):
            self.engine.set_voice(voice_id)

    def set_gender(self, gender: str):
        """Changes voice gender dynamically on active speech engine."""
        if hasattr(self.engine, "set_gender"):
            self.engine.set_gender(gender)

    def speak(self, text: str, wait: bool = False, priority: bool = False, request_id: Optional[int] = None) -> None:
        """Queues or synchronously executes speech."""
        if not text or not text.strip():
            return

        state_mgr = get_state_manager()
        if not state_mgr.can_speak(request_id):
            log_debug(f"[SPEECH CONTROLLER] Ignored speech request for invalid generation #{request_id} (state={state_mgr.get_state().value}).")
            return

        if wait:
            # Synchronous direct speak
            try:
                self.engine.speak(text, wait=True, request_id=request_id)
            except TypeError:
                self.engine.speak(text, wait=True)
        else:
            if priority:
                # Clear queue and put priority speech
                self.clear_speech_queue()
            self._queue.put((text, request_id))

    def stop_speaking(self) -> float:
        """Immediately halts ongoing speech and clears all pending speech."""
        # 1. Clear speech queue first so no more utterances start
        self.clear_speech_queue()
        # 2. Stop active engine playback
        res = self.engine.stop()
        latency_ms = float(res) if isinstance(res, (int, float)) else 0.0
        log_info(f"[CHITTI][TTS] Speech playback stopped immediately (latency: {latency_ms:.2f}ms).")
        return latency_ms

    def clear_speech_queue(self) -> int:
        """Clears all pending speech items from the queue. Returns count cleared."""
        cleared = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                cleared += 1
            except (queue.Empty, ValueError):
                break
        if cleared > 0:
            log_info(f"[CHITTI][TTS] Discarded {cleared} pending utterances from speech queue.")
        return cleared

    def is_speaking(self) -> bool:
        """Returns True if the engine is speaking or queue is non-empty."""
        return self.engine.is_speaking() or not self._queue.empty()

    def shutdown(self):
        """Shuts down the speech controller worker."""
        self._is_running = False
        self.clear_speech_queue()
        self.stop_speaking()

    def _worker_loop(self):
        """Sequential single-worker loop to prevent multiple speech voices overlapping."""
        while self._is_running:
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                if isinstance(item, tuple):
                    text, req_id = item
                else:
                    text, req_id = item, None

                state_mgr = get_state_manager()
                if state_mgr.can_speak(req_id):
                    try:
                        self.engine.speak(text, wait=True, request_id=req_id)
                    except TypeError:
                        self.engine.speak(text, wait=True)
                else:
                    log_debug(f"[SPEECH CONTROLLER] Discarded queued utterance for invalid generation #{req_id}: '{text}'")
            except Exception as e:
                log_debug(f"[SPEECH CONTROLLER] Synthesis notice: {e}")
            finally:
                self._queue.task_done()


def get_tts(config: Optional[TTSConfig] = None) -> TTSEngine:
    """Factory function returning the configured TTS engine."""
    cfg = config or get_config().tts
    if cfg.engine.lower() in ("edge-tts", "neural", "edge", "hybrid"):
        return HybridTTS(cfg)
    if cfg.engine.lower() == "pyttsx3":
        return Pyttsx3TTS(cfg)
    return HybridTTS(cfg)


# Global Speech Controller singleton
_SPEECH_CONTROLLER: Optional[SpeechController] = None
_SPEECH_LOCK = threading.Lock()


def get_speech_controller(config: Optional[TTSConfig] = None) -> SpeechController:
    """Returns the singleton centralized SpeechController."""
    global _SPEECH_CONTROLLER
    if _SPEECH_CONTROLLER is None:
        with _SPEECH_LOCK:
            if _SPEECH_CONTROLLER is None:
                _SPEECH_CONTROLLER = SpeechController(config=config)
    return _SPEECH_CONTROLLER
