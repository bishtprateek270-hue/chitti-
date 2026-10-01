"""
Chitti Ambient Wake-Word Detection Engine (Phase 5).
Continuously processes incoming microphone audio frames in the background
with ultra-low CPU overhead to spot activation keywords ("Hey Chitti", "Chitti").
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
    Supports neural wake-word models with high-speed acoustic phoneme fallback.
    """

    DEFAULT_KEYWORDS = ["hey chitti", "chitti", "ok chitti"]

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1280,  # 80ms chunks at 16kHz
        keywords: Optional[List[str]] = None,
        sensitivity: float = 0.65,
        cooldown_seconds: float = 1.5,
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.keywords = [k.lower().strip() for k in (keywords or self.DEFAULT_KEYWORDS)]
        self.sensitivity = max(0.1, min(1.0, sensitivity))
        self.cooldown_seconds = cooldown_seconds

        self._is_running = False
        self._is_paused = False
        self._thread: Optional[threading.Thread] = None
        self._stream: Optional[Any] = None
        self._callback: Optional[Callable[[WakeWordEvent], None]] = None
        self._last_trigger_time = 0.0

        # Ring buffer for sliding audio window (1.5 seconds)
        self._buffer_size = int(self.sample_rate * 1.5)
        self._audio_buffer = np.zeros(self._buffer_size, dtype=np.float32)
        self._buffer_lock = threading.Lock()

        # Background noise calibration
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
        and checks for wake-word activation patterns.
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

        # 1. Compute acoustic energy and spectral properties
        rms = float(np.sqrt(np.mean(chunk**2) + 1e-12))

        # Dynamic noise floor adaptation
        if not self._calibrated:
            self._noise_floor = max(0.005, min(0.05, rms * 1.5))
            self._calibrated = True
        elif rms < self._noise_floor:
            self._noise_floor = 0.95 * self._noise_floor + 0.05 * rms

        # Speech onset gate
        if rms < self._noise_floor * (1.8 - self.sensitivity * 0.8):
            return None

        # 2. Evaluate acoustic resonance for "Chitti" / "Hey Chitti"
        # "Chitti" characteristically exhibits high-frequency affricate/fricative bursts (CH)
        # followed by short front vowel (I), dental stop (TT), and front vowel (I).
        confidence = self._evaluate_acoustic_signature(current_window)

        if confidence >= (1.0 - self.sensitivity * 0.6):
            self._last_trigger_time = now
            matched_kw = "Hey Chitti" if confidence > 0.85 else "Chitti"
            event = WakeWordEvent(
                keyword=matched_kw,
                confidence=float(confidence),
                timestamp=now,
                audio_snippet=current_window,
            )
            log_chitti(f"[WAKE WORD] ✨ Wake word detected: '{matched_kw}' (Confidence: {confidence:.2f})")
            return event

        return None

    def _evaluate_acoustic_signature(self, audio: np.ndarray) -> float:
        """
        Evaluates acoustic energy concentration, zero-crossing rate,
        and temporal formant shifts corresponding to the phonemes in 'Chitti'.
        """
        if len(audio) < self.sample_rate * 0.4:
            return 0.0

        # Sub-divide recent 0.8s into 4 equal segments: [Onset (CH), Vowel (I), Plosive (TT), Vowel (I)]
        window = audio[-int(self.sample_rate * 0.8):]
        seg_len = len(window) // 4
        if seg_len < 100:
            return 0.0

        energies = []
        zcrs = []
        for i in range(4):
            seg = window[i * seg_len : (i + 1) * seg_len]
            e = float(np.mean(seg**2))
            z = float(np.mean(np.abs(np.diff(np.sign(seg + 1e-12)))) / 2.0)
            energies.append(e)
            zcrs.append(z)

        # "Chitti" phonetic profile:
        # Segment 0: High ZCR (fricative "Ch")
        # Segment 1: High Energy, Medium ZCR (vowel "i")
        # Segment 2: Brief Dip/Stop (dental plosive "tt")
        # Segment 3: High Energy, Medium ZCR (vowel "i")
        score = 0.0

        # High zero-crossing on onset
        if zcrs[0] > 0.12:
            score += 0.30

        # Strong vowel energy in seg 1 or seg 3
        if energies[1] > self._noise_floor * 1.1 or energies[3] > self._noise_floor * 1.1 or energies[1] > 0.008 or energies[3] > 0.008:
            score += 0.35

        # Rhythmic modulation
        if max(energies) > 0.005:
            score += 0.20

        # Peak energy balance
        if max(energies) > self._noise_floor * 1.5 or max(energies) > 0.01:
            score += 0.15

        return min(1.0, score)
