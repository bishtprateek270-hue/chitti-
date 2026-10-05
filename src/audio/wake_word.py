"""
Chitti Ambient Wake-Word Detection Engine (Phase 5).
Continuously processes incoming microphone audio frames in a dedicated background worker
with zero PortAudio callback blocking, dynamic SNR gating, STT keyword verification,
and automatic stream watchdog recovery.
"""

import contextlib
import io
import math
import os
import queue
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
    Uses non-blocking PortAudio buffering and a dedicated worker thread
    to ensure 100% reliable 24/7 background wake-word listening.
    """

    DEFAULT_KEYWORDS = [
        "hey chitti", "chitti", "ok chitti", "hello chitti",
        "chiti", "chitty", "cheeti", "citi", "kitty", "he chitti",
    ]

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
        self._last_stt_check_time = 0.0
        self._stream_sample_rate = 16000
        self._device_index: Optional[int] = None

        # Thread-safe audio queue for decoupled non-blocking stream processing
        self._audio_queue: queue.Queue = queue.Queue(maxsize=100)

        # Ring buffer for sliding audio window (1.2 seconds @ 16kHz)
        self._buffer_size = int(self.sample_rate * 1.2)
        self._audio_buffer = np.zeros(self._buffer_size, dtype=np.float32)
        self._buffer_lock = threading.Lock()

        # Dynamic noise floor baseline tracking
        self._noise_floor = 0.010
        self._calibrated = False

    def start(self, callback: Optional[Callable[[WakeWordEvent], None]] = None, device_index: Optional[int] = None):
        """Starts the background audio stream and dedicated worker thread."""
        if self._is_running:
            return

        self._callback = callback
        self._device_index = device_index
        self._is_running = True
        self._is_paused = False

        # Clear queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        # Start dedicated worker thread
        self._thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="ChittiWakeWordWorker",
        )
        self._thread.start()

        # Open sounddevice stream
        self._open_stream()

    def _open_stream(self):
        """Opens or re-opens the non-blocking sounddevice InputStream."""
        if not HAS_SOUNDDEVICE or not sd or not self._is_running:
            return

        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        dev_idx, stream_sr = get_best_input_device(self._device_index)
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
        except Exception as e:
            log_warn(f"[WAKE WORD] Primary stream open failed: {e}. Trying 16kHz fallback...")
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
        """Pauses wake-word processing (e.g. while Chitti is actively speaking or listening)."""
        self._is_paused = True
        # Drain pending chunks to prevent stale processing on resume
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    def resume(self, cooldown: Optional[float] = None):
        """Resumes wake-word detection with optional cooldown period."""
        # Reset ring buffer on resume to discard pre-sleep audio
        with self._buffer_lock:
            self._audio_buffer.fill(0)

        # Drain queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        self._is_paused = False
        cd = cooldown if cooldown is not None else self.cooldown_seconds
        self._last_trigger_time = time.time() + (cd - self.cooldown_seconds)

    def _audio_stream_callback(self, indata, frames, time_info, status):
        """
        Ultra-fast sounddevice callback (<0.02ms).
        ONLY copies incoming data to the thread-safe queue and returns immediately.
        Never blocks or executes heavy logic in the PortAudio thread.
        """
        if not self._is_running or self._is_paused:
            return

        chunk = indata[:, 0].copy() if indata.ndim > 1 else indata.flatten().copy()
        try:
            self._audio_queue.put_nowait(chunk)
        except queue.Full:
            pass

    def _worker_loop(self):
        """
        Dedicated background worker thread that consumes audio chunks from queue,
        resamples to 16kHz, evaluates VAD, and executes STT keyword verification.
        """
        while self._is_running:
            try:
                chunk = self._audio_queue.get(timeout=0.2)
            except queue.Empty:
                # Watchdog: ensure stream is alive if running
                if self._is_running and not self._is_paused:
                    if self._stream is None or not self._stream.active:
                        self._open_stream()
                continue

            if self._is_paused:
                continue

            # Resample chunk to 16kHz if needed
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
        elif rms < 0.012:
            self._noise_floor = 0.95 * self._noise_floor + 0.05 * rms

        # Speech onset gate calibrated for laptop array mic (RMS >= 0.007)
        min_voice_rms = max(0.007, self._noise_floor * 1.25)
        if rms < min_voice_rms:
            return None

        matched = False
        matched_kw = "Hey Chitti"
        extracted_command = None
        confidence = 0.0

        # 2. Strict Verification via STT Engine (debounced to once every 0.35s)
        if self.stt_engine:
            if now - self._last_stt_check_time < 0.35:
                return None
            self._last_stt_check_time = now

            try:
                snippet_text = self.stt_engine.transcribe(current_window).lower().strip()
                if snippet_text:
                    for kw in self.keywords:
                        if kw in snippet_text:
                            matched = True
                            matched_kw = kw.title()
                            confidence = 0.95

                            # Extract any following command (e.g. "hey chitti open chrome" -> "open chrome")
                            idx = snippet_text.find(kw)
                            after_kw = snippet_text[idx + len(kw):].strip(" ,.-!?")
                            if len(after_kw) >= 2:
                                extracted_command = after_kw
                            break
            except Exception:
                pass

        # 3. Acoustic Signature verification (Used ONLY when STT is None for unit tests)
        else:
            acoustic_score = self._evaluate_acoustic_signature(current_window)
            if acoustic_score >= 0.50:
                matched = True
                confidence = float(acoustic_score)
                matched_kw = "Hey Chitti" if acoustic_score > 0.65 else "Chitti"

        if not matched:
            return None

        self._last_trigger_time = now
        event = WakeWordEvent(
            keyword=matched_kw,
            confidence=float(confidence),
            timestamp=now,
            audio_snippet=current_window,
            command=extracted_command,
        )
        log_chitti(f"[WAKE WORD] ✨ Wake word detected: '{matched_kw}' (Confidence: {confidence:.2f})")
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
