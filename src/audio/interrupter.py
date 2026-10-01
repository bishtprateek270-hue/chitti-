"""
Chitti Real-Time Audio Interrupter & Barge-In Detection Engine (Phase 5).
Listens for user speech during TTS playback to immediately pause or cancel
current speech output when the user speaks up.
"""

import threading
import time
from typing import Any, Callable, Optional
import numpy as np

from src.utils.logging import log_chitti, log_debug, log_info

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    sd = None
    HAS_SOUNDDEVICE = False


class BargeInDetector:
    """
    Monitors audio input during assistant playback. Triggers immediate interruption
    if user voice energy exceeds background acoustic echo thresholds.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 800,  # 50ms chunks
        energy_threshold: float = 0.045,
        consecutive_frames: int = 3,
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.energy_threshold = energy_threshold
        self.consecutive_frames = consecutive_frames

        self._is_active = False
        self._stream: Optional[Any] = None
        self._on_interrupt: Optional[Callable[[], None]] = None
        self._hit_count = 0
        self._lock = threading.Lock()

    def start_monitoring(
        self,
        on_interrupt: Callable[[], None],
        device_index: Optional[int] = None,
    ):
        """Starts real-time microphone monitoring during TTS playback."""
        if self._is_active:
            return

        self._on_interrupt = on_interrupt
        self._hit_count = 0
        self._is_active = True

        if HAS_SOUNDDEVICE and sd:
            try:
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="float32",
                    blocksize=self.chunk_size,
                    device=device_index,
                    callback=self._audio_callback,
                )
                self._stream.start()
                log_debug("[BARGE-IN] Active barge-in monitoring enabled.")
            except Exception as e:
                log_debug(f"[BARGE-IN] Monitoring stream failed: {e}")

    def stop_monitoring(self):
        """Stops microphone monitoring after TTS playback concludes."""
        self._is_active = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        log_debug("[BARGE-IN] Monitoring stopped.")

    def _audio_callback(self, indata, frames, time_info, status):
        """Processes audio frame to detect speech interruption."""
        if not self._is_active:
            return

        chunk = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        rms = float(np.sqrt(np.mean(chunk**2) + 1e-12))

        with self._lock:
            if rms >= self.energy_threshold:
                self._hit_count += 1
                if self._hit_count >= self.consecutive_frames:
                    log_chitti("[BARGE-IN] 🛑 User interruption detected! Stopping speech playback.")
                    self._is_active = False
                    if self._on_interrupt:
                        try:
                            self._on_interrupt()
                        except Exception as e:
                            log_debug(f"[BARGE-IN] Interrupt callback notice: {e}")
            else:
                self._hit_count = max(0, self._hit_count - 1)

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
