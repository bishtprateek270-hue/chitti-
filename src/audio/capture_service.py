"""
Chitti Authoritative Audio Capture Service.
Acts as the SINGLE authoritative owner of the hardware microphone device across Chitti.
Eliminates competing PortAudio streams, device busy errors, and dropped buffer frames
by providing a unified, crash-resilient 16kHz mono audio pipeline with pub/sub tap capabilities.
"""

import math
import queue
import threading
import time
from typing import Callable, Dict, Optional, Tuple
import numpy as np

from src.audio.audio_utils import get_best_input_device, resample_to_16k
from src.config import AudioConfig, get_config
from src.utils.logging import log_chitti, log_debug, log_info, log_warn

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    sd = None
    HAS_SOUNDDEVICE = False


class AudioCaptureService:
    """
    Single Authoritative Microphone Capture Service for Chitti.
    Opens exactly ONE hardware InputStream and multiplexes audio to registered subscribers.
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or get_config().audio
        self.target_sample_rate = 16000
        self.chunk_duration = 0.05  # 50ms chunks (800 samples @ 16kHz)
        self.chunk_samples_16k = int(self.target_sample_rate * self.chunk_duration)

        self._stream: Optional[Any] = None
        self._is_running = False
        self._stream_sample_rate = 16000
        self._device_index: Optional[int] = self.config.microphone_index

        # Thread synchronization & subscriber registry
        self._subscribers: Dict[str, Callable[[np.ndarray, float], None]] = {}
        self._subscribers_lock = threading.RLock()

        # Bounded queue (max ~2 seconds of audio) to prevent memory leak/unbounded growth
        self._audio_queue: queue.Queue = queue.Queue(maxsize=40)
        self._dispatcher_thread: Optional[threading.Thread] = None

        # Diagnostic metrics
        self._last_energy = 0.0
        self._error_count = 0
        self._frames_captured = 0
        self._overflow_count = 0
        self._underflow_count = 0
        self._last_frame_time = time.time()
        self._last_restart_attempt = 0.0

    def start(self, device_index: Optional[int] = None):
        """Starts the single authoritative microphone capture service and dispatcher."""
        if self._is_running:
            return

        if not HAS_SOUNDDEVICE or sd is None:
            log_warn("[VOICE] sounddevice is unavailable. Audio capture disabled.")
            return

        if device_index is not None:
            self._device_index = device_index

        self._is_running = True
        self.flush()

        # Start background dispatcher thread
        self._dispatcher_thread = threading.Thread(
            target=self._dispatcher_loop,
            daemon=True,
            name="ChittiAudioDispatcher",
        )
        self._dispatcher_thread.start()

        # Initialize hardware stream
        self._open_stream()

    def _open_stream(self) -> bool:
        """Opens the hardware audio stream on the selected device."""
        if not HAS_SOUNDDEVICE or sd is None:
            return False

        dev_idx, stream_sr = get_best_input_device(self._device_index)
        self._stream_sample_rate = stream_sr
        chunk_samples = max(256, int(stream_sr * self.chunk_duration))

        try:
            self._stream = sd.InputStream(
                samplerate=stream_sr,
                channels=1,
                dtype="float32",
                blocksize=chunk_samples,
                device=dev_idx,
                callback=self._audio_callback,
            )
            self._stream.start()
            dev_label = f"device #{dev_idx}" if dev_idx is not None else "default mic"
            log_info(f"[VOICE] Microphone ready on {dev_label} @ {stream_sr}Hz.")
            log_info("[VOICE] Capture loop started.")
            return True
        except Exception as e:
            log_warn(f"[VOICE] Primary microphone open failed ({e}). Trying fallback @ 16kHz...")
            try:
                self._stream = sd.InputStream(
                    samplerate=16000,
                    channels=1,
                    dtype="float32",
                    blocksize=self.chunk_samples_16k,
                    callback=self._audio_callback,
                )
                self._stream.start()
                self._stream_sample_rate = 16000
                log_info("[VOICE] Microphone ready (fallback 16kHz stream active).")
                log_info("[VOICE] Capture loop started.")
                return True
            except Exception as fallback_err:
                log_warn(f"[VOICE] Audio input error: could not open microphone: {fallback_err}")
                return False

    def stop(self):
        """Stops the audio capture service and releases the hardware microphone."""
        self._is_running = False
        log_info("[VOICE] Capture loop stopped.")

        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if self._dispatcher_thread and self._dispatcher_thread.is_alive():
            self._dispatcher_thread.join(timeout=1.0)
            self._dispatcher_thread = None

        self.flush()

    def is_running(self) -> bool:
        """Returns True if the capture service is active."""
        return self._is_running

    def subscribe(self, subscriber_id: str, callback: Callable[[np.ndarray, float], None]):
        """
        Registers an audio subscriber (e.g. 'wake_word', 'streaming_stt', 'barge_in').
        The callback receives (chunk_16k: np.ndarray, energy_rms: float).
        """
        with self._subscribers_lock:
            self._subscribers[subscriber_id] = callback
            log_debug(f"[VOICE] Subscriber registered: '{subscriber_id}' (Total: {len(self._subscribers)})")

    def unsubscribe(self, subscriber_id: str):
        """Unregisters an audio subscriber."""
        with self._subscribers_lock:
            if subscriber_id in self._subscribers:
                del self._subscribers[subscriber_id]
                log_debug(f"[VOICE] Subscriber unregistered: '{subscriber_id}' (Total: {len(self._subscribers)})")

    def has_subscriber(self, subscriber_id: str) -> bool:
        """Checks if a subscriber is active."""
        with self._subscribers_lock:
            return subscriber_id in self._subscribers

    def flush(self):
        """Discards all pending audio frames from the buffer to prevent stale processing."""
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    def get_recent_energy(self) -> float:
        """Returns the RMS energy of the most recent audio chunk."""
        return self._last_energy

    def get_diagnostics(self) -> Dict[str, Any]:
        """Returns runtime diagnostic metrics for microphone capture."""
        return {
            "is_running": self._is_running,
            "device_index": self._device_index,
            "sample_rate": self._stream_sample_rate,
            "frames_captured": self._frames_captured,
            "recent_rms": self._last_energy,
            "overflow_count": self._overflow_count,
            "underflow_count": self._underflow_count,
            "subscriber_count": len(self._subscribers),
        }

    def _audio_callback(self, indata, frames, time_info, status):
        """
        Fastest possible PortAudio callback.
        Copies data to non-blocking bounded queue to prevent audio dropouts.
        """
        if not self._is_running:
            return

        if status:
            if getattr(status, "input_overflow", False):
                self._overflow_count += 1
            if getattr(status, "input_underflow", False):
                self._underflow_count += 1

        if not np.all(np.isfinite(indata)):
            return

        self._last_frame_time = time.time()
        chunk = indata[:, 0].copy() if indata.ndim > 1 else indata.flatten().copy()
        try:
            self._audio_queue.put_nowait(chunk)
        except queue.Full:
            # Drop oldest frame to make room and keep stream live without lag
            try:
                self._audio_queue.get_nowait()
                self._audio_queue.put_nowait(chunk)
            except Exception:
                pass

    def _restart_stream(self):
        """Restarts the hardware InputStream if stalled or disconnected."""
        try:
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
            self.flush()
            # Force hardware rediscovery to catch Windows audio endpoints initialized after logon
            from src.audio.audio_utils import get_best_input_device
            get_best_input_device(preferred_index=self._device_index, force_refresh=True)
            success = self._open_stream()
            if success:
                self._last_frame_time = time.time()
                log_info("[VOICE] Microphone stream successfully revived by watchdog.")
        except Exception as e:
            log_debug(f"[VOICE] Watchdog restart notice: {e}")

    def _dispatcher_loop(self):
        """
        Background worker thread: consumes raw chunks, resamples to 16kHz mono,
        calculates RMS energy, and dispatches to registered subscribers.
        Includes an automatic watchdog to revive stalled hardware streams.
        """
        while self._is_running:
            try:
                raw_chunk = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                now = time.time()
                # Watchdog: if running and have subscribers, check if hardware stream is dead or stalled
                with self._subscribers_lock:
                    has_active_subs = len(self._subscribers) > 0
                if has_active_subs and (self._stream is None or (now - self._last_frame_time > 3.0)):
                    if now - self._last_restart_attempt >= 2.5:
                        self._last_restart_attempt = now
                        log_warn("[VOICE] Hardware audio stream not active or stalled. Reviving microphone...")
                        self._restart_stream()
                continue

            self._frames_captured += 1

            # Resample to 16kHz if needed
            if self._stream_sample_rate != self.target_sample_rate:
                chunk_16k = resample_to_16k(raw_chunk, self._stream_sample_rate)
            else:
                chunk_16k = raw_chunk

            if chunk_16k.dtype != np.float32:
                chunk_16k = chunk_16k.astype(np.float32)

            # Compute RMS energy
            rms = float(np.sqrt(np.mean(chunk_16k**2) + 1e-12))
            self._last_energy = rms

            # Dispatch to subscribers outside the global lock
            with self._subscribers_lock:
                current_subs = list(self._subscribers.items())

            for sub_id, cb in current_subs:
                try:
                    cb(chunk_16k, rms)
                except Exception as e:
                    log_debug(f"[VOICE] Audio dispatch notice for '{sub_id}': {e}")


# Global Singleton Instance
_CAPTURE_SERVICE: Optional[AudioCaptureService] = None
_CAPTURE_LOCK = threading.Lock()


def get_audio_capture_service(config: Optional[AudioConfig] = None) -> AudioCaptureService:
    """Returns the application-wide authoritative AudioCaptureService instance."""
    global _CAPTURE_SERVICE
    if _CAPTURE_SERVICE is None:
        with _CAPTURE_LOCK:
            if _CAPTURE_SERVICE is None:
                _CAPTURE_SERVICE = AudioCaptureService(config=config)
    return _CAPTURE_SERVICE
