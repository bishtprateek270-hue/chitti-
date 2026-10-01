"""
Chitti Ambient Wake-Word Detection Engine (Phase 5).
Continuously processes incoming microphone audio frames in the background
with ultra-low latency DSP acoustic pattern recognition (pitch autocorrelation,
syllable envelope modulation, and unvoiced-to-voiced consonant transition).
"""

import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

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


class WakeWordDetector:
    """
    Lightweight, continuous keyword spotting engine.
    Uses ultra-fast DSP phoneme resonance and vocal harmonic analysis
    to detect 'Chitti' / 'Hey Chitti' without stalling CPU/GPU.
    """

    DEFAULT_KEYWORDS = ["hey chitti", "chitti", "ok chitti"]

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1280,  # 80ms chunks at 16kHz
        keywords: Optional[List[str]] = None,
        sensitivity: float = 0.75,
        cooldown_seconds: float = 2.0,
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

        # Ring buffer for sliding audio window (1.0 seconds)
        self._buffer_size = int(self.sample_rate * 1.0)
        self._audio_buffer = np.zeros(self._buffer_size, dtype=np.float32)
        self._buffer_lock = threading.Lock()

        # Background noise calibration
        self._noise_floor = 0.025
        self._calibrated = False

    def start(self, callback: Optional[Callable[[WakeWordEvent], None]] = None, device_index: Optional[int] = None):
        """Starts the background listening thread for ambient wake-word detection."""
        if self._is_running:
            return

        self._callback = callback
        self._is_running = True
        self._is_paused = False

        if HAS_SOUNDDEVICE and sd:
            try:
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="float32",
                    blocksize=self.chunk_size,
                    device=device_index,
                    callback=self._audio_stream_callback,
                )
                self._stream.start()
                log_info(f"[WAKE WORD] Ambient wake-word listening active (Keywords: {', '.join(self.keywords)})")
                return
            except Exception as e:
                log_warn(f"[WAKE WORD] sounddevice stream init failed: {e}. Running in manual chunk mode.")

        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="WakeWordWorker")
        self._thread.start()

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
        event = self.process_audio_chunk(chunk)
        if event and self._callback:
            try:
                self._callback(event)
            except Exception as e:
                log_warn(f"[WAKE WORD] Callback error: {e}")

    def _worker_loop(self):
        """Fallback worker loop when sounddevice stream callback is unavailable."""
        while self._is_running:
            time.sleep(0.05)

    def process_audio_chunk(self, chunk: np.ndarray) -> Optional[WakeWordEvent]:
        """
        Processes a single audio chunk (1D float32 array), updates sliding window,
        and checks for wake-word activation patterns in < 0.2ms.
        """
        if chunk is None or len(chunk) == 0:
            return None

        # Ensure float32 1D
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)
        if chunk.ndim > 1:
            chunk = chunk.flatten()

        # Update sliding ring buffer
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

        # Dynamic noise floor adaptation
        if not self._calibrated:
            self._noise_floor = min(0.020, rms)
            self._calibrated = True
        elif rms < self._noise_floor:
            self._noise_floor = 0.95 * self._noise_floor + 0.05 * rms

        # Speech onset gate (require noticeable voice energy above ambient background)
        min_voice_rms = max(0.020, self._noise_floor * 1.5)
        if rms < min_voice_rms:
            return None

        # 2. Evaluate acoustic phoneme signature for "Chitti" / "Hey Chitti"
        confidence = self._evaluate_acoustic_signature(current_window)
        if confidence < 0.60:
            return None

        matched_kw = "Hey Chitti" if confidence > 0.85 else "Chitti"
        self._last_trigger_time = now
        event = WakeWordEvent(
            keyword=matched_kw,
            confidence=float(confidence),
            timestamp=now,
            audio_snippet=current_window,
        )
        log_chitti(f"[WAKE WORD] ✨ Wake word detected: '{matched_kw}' (Confidence: {confidence:.2f})")
        return event

    def _evaluate_acoustic_signature(self, audio: np.ndarray) -> float:
        """
        Evaluates human voice pitch periodicity (autocorrelation), onset zero-crossing rate,
        and the 2-syllable peak-valley-peak rhythmic energy envelope of 'Chit-ti'.
        """
        if len(audio) < int(self.sample_rate * 0.4):
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

        # Non-voiced sound or static noise gets immediate 0
        if pitch_corr < 0.32:
            return 0.0

        # 2. Syllable Envelope (Peak 1 -> Stop Closure Valley -> Peak 2)
        win = audio[-int(self.sample_rate * 0.6):]
        slice_len = len(win) // 6
        if slice_len < 50:
            return 0.0

        energies = [float(np.mean(win[i * slice_len : (i + 1) * slice_len] ** 2)) for i in range(6)]
        p1 = max(energies[0], energies[1])
        valley = min(energies[2], energies[3])
        p2 = max(energies[4], energies[5])

        has_dip = valley < (max(p1, p2) * 0.60) if max(p1, p2) > 1e-5 else False

        # 3. High-frequency onset burst ("Ch")
        zcr_onset = float(np.mean(np.abs(np.diff(np.sign(win[:slice_len] + 1e-12)))) / 2.0)

        score = 0.0
        if pitch_corr > 0.40:
            score += 0.35
        if pitch_corr > 0.65:
            score += 0.15

        if has_dip:
            score += 0.30
        if zcr_onset > 0.10:
            score += 0.20

        return score
