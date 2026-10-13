"""
Chitti Real-Time Audio Interrupter & Barge-In Detection Engine (Phase 5).
Listens for deliberate user speech during TTS playback to immediately pause or cancel
current speech output when the user speaks up loudly and clearly, ignoring background noise and speaker echo.
"""

import threading
import time
from typing import Any, Callable, Optional
import numpy as np

from src.audio.capture_service import get_audio_capture_service
from src.utils.logging import log_chitti, log_debug, log_info

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    sd = None
    HAS_SOUNDDEVICE = False


class BargeInDetector:
    """
    Monitors audio input during assistant playback. Triggers interruption ONLY
    when intentional human vocal speech exceeds background acoustic echo and noise thresholds.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 800,  # 50ms chunks
        energy_threshold: float = 0.10,
        consecutive_frames: int = 5,
        grace_period_seconds: float = 0.2,
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.energy_threshold = energy_threshold
        self.consecutive_frames = consecutive_frames
        self.grace_period_seconds = grace_period_seconds

        self._is_active = False
        self._stream: Optional[Any] = None
        self._on_interrupt: Optional[Callable[..., None]] = None
        self._hit_count = 0
        self._hit_chunks: list = []
        self._interruption_audio: Optional[np.ndarray] = None
        self._start_time = 0.0
        self._lock = threading.Lock()

    def start_monitoring(
        self,
        on_interrupt: Callable[..., None],
        device_index: Optional[int] = None,
    ):
        """Starts real-time microphone monitoring during TTS playback."""
        if self._is_active:
            return

        self._on_interrupt = on_interrupt
        self._hit_count = 0
        self._hit_chunks.clear()
        self._interruption_audio = None
        self._start_time = time.time()
        self._is_active = True

        if HAS_SOUNDDEVICE:
            capture_service = get_audio_capture_service()
            if not capture_service.is_running():
                capture_service.start(device_index=device_index)
            capture_service.subscribe("barge_in", self._audio_callback)
            log_debug("[BARGE-IN] Active barge-in monitoring enabled via AudioCaptureService.")

    def stop_monitoring(self):
        """Stops microphone monitoring after TTS playback concludes."""
        self._is_active = False
        if HAS_SOUNDDEVICE:
            capture_service = get_audio_capture_service()
            capture_service.unsubscribe("barge_in")
        log_debug("[BARGE-IN] Monitoring stopped.")

    def get_interruption_audio(self) -> Optional[np.ndarray]:
        """Returns the pre-buffered audio captured at the moment of user interruption."""
        with self._lock:
            return self._interruption_audio

    def _is_voiced(self, chunk: np.ndarray) -> bool:
        """Verifies fundamental human voice pitch periodicity (80-350Hz)."""
        if len(chunk) < int(self.sample_rate * 0.03):
            return False
        min_lag = max(1, int(self.sample_rate / 350))
        max_lag = min(len(chunk) - 1, int(self.sample_rate / 80))
        if max_lag <= min_lag:
            return True

        seg = chunk - np.mean(chunk)
        norm = np.sum(seg**2)
        if norm < 1e-6:
            return False

        corr = np.correlate(seg, seg, mode="full")[len(seg) - 1 :] / norm
        if len(corr) <= max_lag:
            return False
        pitch_corr = float(np.max(corr[min_lag:max_lag]))
        return pitch_corr >= 0.30

    def _audio_callback(self, indata, energy: float = 0.0, *args):
        """Processes audio frame to detect intentional speech interruption."""
        if not self._is_active:
            return

        # Ignore initial speaker onset echo during grace period
        if (time.time() - self._start_time) < self.grace_period_seconds:
            return

        chunk = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        rms = float(np.sqrt(np.mean(chunk**2) + 1e-12))

        with self._lock:
            if rms >= self.energy_threshold and self._is_voiced(chunk):
                self._hit_count += 1
                self._hit_chunks.append(chunk.copy())
                if self._hit_count >= self.consecutive_frames:
                    log_chitti("[BARGE-IN] 🛑 Deliberate user interruption detected! Stopping speech playback.")
                    self._is_active = False
                    self._interruption_audio = np.concatenate(self._hit_chunks) if self._hit_chunks else None
                    cb = self._on_interrupt
                    inter_audio = self._interruption_audio

                    if cb:
                        try:
                            cb(inter_audio)
                        except TypeError:
                            try:
                                cb()
                            except Exception as e:
                                log_debug(f"[BARGE-IN] Interrupt callback notice: {e}")
                        except Exception as e:
                            log_debug(f"[BARGE-IN] Interrupt callback notice: {e}")
            else:
                self._hit_count = max(0, self._hit_count - 1)
                if self._hit_count == 0 and len(self._hit_chunks) > 10:
                    self._hit_chunks.clear()

    def process_frame(self, chunk: np.ndarray) -> bool:
        """
        Processes a single frame manually (used in testing or custom streams).
        Returns True if barge-in was triggered.
        """
        if chunk is None or len(chunk) == 0:
            return False

        rms = float(np.sqrt(np.mean(chunk**2) + 1e-12))
        if rms >= self.energy_threshold:
            self._hit_count += 1
            if self._hit_count >= self.consecutive_frames:
                return True
        else:
            self._hit_count = max(0, self._hit_count - 1)
        return False
