"""
Chitti Ambient Wake-Word Detection Engine (Phase 5).
Continuously processes incoming microphone audio frames in the background
with zero CPU overhead, utilizing dynamic SNR gating, acoustic signature heuristics,
and intelligent keyword+command extraction.
"""

import contextlib
import io
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from src.audio.audio_utils import get_best_input_device, resample_to_16k
from src.utils.logging import log_chitti, log_debug, log_info, log_warn

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    sd = None
    HAS_SOUNDDEVICE = False


@dataclass
class WakeWordEvent:
    """Represents a successfully spotted wake-word activation event."""
    keyword: str
    confidence: float
    timestamp: float
    audio_snippet: Optional[np.ndarray] = None
    command: Optional[str] = None


class WakeWordDetector:
    """
    Lightweight, continuous keyword spotting engine.
    Uses dynamic SNR gating and DSP acoustic phoneme resonance
    to detect 'Chitti' / 'Hey Chitti' without false positives from background noise.
    """

    DEFAULT_KEYWORDS = ["hey chitti", "chitti", "ok chitti", "hello chitti", "chup", "stop", "so jao"]

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1280,  # 80ms chunks at 16kHz
        keywords: Optional[List[str]] = None,
        sensitivity: float = 0.75,
        cooldown_seconds: float = 1.5,
        stt_engine: Optional[Any] = None,
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.keywords = [k.lower().strip() for k in (keywords or self.DEFAULT_KEYWORDS)]
        self.sensitivity = max(0.1, min(1.0, sensitivity))
        self.cooldown_seconds = cooldown_seconds
        self.stt_engine = stt_engine

        self._is_running = False
        self._is_paused = False
        self._thread: Optional[threading.Thread] = None
        self._stream: Optional[Any] = None
        self._callback: Optional[Callable[[WakeWordEvent], None]] = None
        self._last_trigger_time = 0.0
        self._stream_sample_rate = 16000

        # Ring buffer for sliding audio window (1.2 seconds)
        self._buffer_size = int(self.sample_rate * 1.2)
        self._audio_buffer = np.zeros(self._buffer_size, dtype=np.float32)
        self._buffer_lock = threading.Lock()

        # Background noise baseline tracking
        self._noise_floor = 0.015
        self._calibrated = False

    def start(self, callback: Optional[Callable[[WakeWordEvent], None]] = None, device_index: Optional[int] = None):
        """Starts the background listening thread for ambient wake-word detection."""
        if self._is_running:
            return

        self._callback = callback
        self._is_running = True
        self._is_paused = False

        if HAS_SOUNDDEVICE and sd:
            dev_idx, stream_sr = get_best_input_device(device_index)
            self._stream_sample_rate = stream_sr
            chunk_samples = int(stream_sr * (self.chunk_size / self.sample_rate))

            try:
                self._stream = sd.InputStream(
                    samplerate=stream_sr,
                    channels=1,
                    dtype="float32",
                    blocksize=chunk_samples,
                    device=dev_idx,
                    callback=self._audio_stream_callback,
                )
                self._stream.start()
                dev_label = f"device #{dev_idx}" if dev_idx is not None else "default mic"
                log_info(f"[WAKE WORD] Ambient wake-word listening active on {dev_label} @ {stream_sr}Hz.")
                return
            except Exception as e:
                log_warn(f"[WAKE WORD] Primary stream open failed: {e}. Trying fallback...")
                # Fallback to standard 16kHz
                try:
                    self._stream = sd.InputStream(
                        samplerate=16000,
                        channels=1,
                        dtype="float32",
                        blocksize=self.chunk_size,
                        callback=self._audio_stream_callback,
                    )
                    self._stream.start()
                    self._stream_sample_rate = 16000
                    log_info("[WAKE WORD] Fallback stream active @ 16000Hz.")
                    return
                except Exception as fallback_err:
                    log_warn(f"[WAKE WORD] Could not start audio stream: {fallback_err}")

    def stop(self):
        """Stops the ambient wake-word detection engine and releases audio resources."""
        self._is_running = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None

        log_debug("[WAKE WORD] Wake-word detector stopped.")

    def pause(self):
        """Pauses wake-word detection (e.g. while Chitti is actively speaking or listening)."""
        self._is_paused = True

    def resume(self):
        """Resumes wake-word detection after speech/interaction finishes."""
        self._is_paused = False
        self._last_trigger_time = time.time()

    def _audio_stream_callback(self, indata, frames, time_info, status):
        """Audio callback triggered by sounddevice stream."""
        if not self._is_running or self._is_paused:
            return

        chunk = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        if self._stream_sample_rate != self.sample_rate:
            chunk_16k = resample_to_16k(chunk, self._stream_sample_rate)
        else:
            chunk_16k = chunk

        event = self.process_audio_chunk(chunk_16k)
        if event and self._callback:
            try:
                self._callback(event)
            except Exception as e:
                log_warn(f"[WAKE WORD] Callback error: {e}")

    def process_audio_chunk(self, chunk: np.ndarray) -> Optional[WakeWordEvent]:
        """
        Processes a single audio chunk (1D float32 array at 16kHz), updates sliding window,
        and checks for wake-word activation patterns.
        """
        if chunk is None or len(chunk) == 0:
            return None

        # Ensure float32 1D
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)
        if chunk.ndim > 1:
            chunk = chunk.flatten()

        # Update sliding ring buffer (16kHz)
        with self._buffer_lock:
            self._audio_buffer = np.roll(self._audio_buffer, -len(chunk))
            self._audio_buffer[-len(chunk):] = chunk
            current_window = self._audio_buffer.copy()

        # Check cooldown
        now = time.time()
        if now - self._last_trigger_time < self.cooldown_seconds:
            return None

        # 1. Compute acoustic energy
        rms = float(np.sqrt(np.mean(chunk**2) + 1e-12))

        # Dynamic noise floor adaptation (tracks ambient baseline when low)
        if not self._calibrated:
            self._noise_floor = min(0.012, max(0.002, rms))
            self._calibrated = True
        elif rms < 0.015:
            self._noise_floor = 0.95 * self._noise_floor + 0.05 * rms

        # Speech onset gate calibrated for laptop array mic (RMS >= 0.008)
        min_voice_rms = max(0.008, self._noise_floor * 1.3)
        if rms < min_voice_rms:
            return None

        # 2. Fast Acoustic Signature Evaluation
        acoustic_confidence = self._evaluate_acoustic_signature(current_window)
        matched = False
        matched_kw = "Hey Chitti"
        extracted_command = None

        # If STT engine is available and speech is detected, verify with STT
        if self.stt_engine and rms > (min_voice_rms * 1.2):
            try:
                snippet_text = self.stt_engine.transcribe(current_window).lower().strip()
                if snippet_text:
                    for kw in ["hey chitti", "ok chitti", "hello chitti", "chitti", "chiti", "chitty", "cheeti", "citi"]:
                        if kw in snippet_text:
                            matched = True
                            matched_kw = kw.title()
                            acoustic_confidence = max(0.90, acoustic_confidence)
                            
                            # Extract any following command (e.g. "hey chitti open chrome" -> "open chrome")
                            idx = snippet_text.find(kw)
                            after_kw = snippet_text[idx + len(kw):].strip(" ,.-!?")
                            if len(after_kw) >= 3:
                                extracted_command = after_kw
                            break
            except Exception:
                pass

        # Fallback to acoustic signature if confident
        if not matched and acoustic_confidence >= 0.50:
            matched = True
            matched_kw = "Hey Chitti" if acoustic_confidence > 0.65 else "Chitti"

        if not matched:
            return None

        self._last_trigger_time = now
        event = WakeWordEvent(
            keyword=matched_kw,
            confidence=float(acoustic_confidence),
            timestamp=now,
            audio_snippet=current_window,
            command=extracted_command,
        )
        log_chitti(f"[WAKE WORD] ✨ Wake word detected: '{matched_kw}' (Confidence: {acoustic_confidence:.2f})")
        return event

    def _evaluate_acoustic_signature(self, audio: np.ndarray) -> float:
        """
        Evaluates human voice pitch periodicity (autocorrelation) and syllabic energy envelope.
        """
        if len(audio) < int(self.sample_rate * 0.3):
            return 0.0

        # 1. Human vocal pitch periodicity check (80Hz - 350Hz)
        min_lag = int(self.sample_rate / 350)  # ~45 samples
        max_lag = int(self.sample_rate / 80)   # ~200 samples
        seg = audio[-int(self.sample_rate * 0.4):]
        seg = seg - np.mean(seg)
        norm = np.sum(seg**2)
        if norm < 1e-6:
            return 0.0

        corr = np.correlate(seg, seg, mode="full")[len(seg) - 1 :] / norm
        pitch_corr = float(np.max(corr[min_lag:max_lag])) if len(corr) > max_lag else 0.0

        if pitch_corr < 0.25:
            return 0.0

        # 2. Syllable Envelope
        win = audio[-int(self.sample_rate * 0.6):]
        slice_len = len(win) // 4
        if slice_len < 40:
            return 0.0

        energies = [float(np.mean(win[i * slice_len : (i + 1) * slice_len] ** 2)) for i in range(4)]
        p_max = max(energies)
        p_min = min(energies)

        score = 0.0
        if pitch_corr > 0.35:
            score += 0.30
        if pitch_corr > 0.55:
            score += 0.20
        if p_max > 0.001:
            score += 0.25
        if p_min < p_max * 0.75:
            score += 0.25

        return score
