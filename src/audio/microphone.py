"""
Chitti Microphone & Audio Capture Module.
Handles device detection, stream capture, and Voice Activity Detection (VAD).
"""

import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

from src.config import AudioConfig, get_config
from src.utils.logging import log_debug, log_warning


class AudioCaptureError(Exception):
    """Raised when audio capture or device access fails."""
    pass


class MicrophoneManager:
    """Manages audio input devices and captures voice with automatic silence detection."""

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or get_config().audio
        self.sample_rate = self.config.sample_rate
        self.channels = self.config.channels
        self.device_index = self.config.microphone_index
        self.silence_threshold = self.config.silence_threshold
        self.silence_duration = self.config.silence_duration
        self.max_duration = self.config.max_record_seconds

    @staticmethod
    def list_microphones() -> List[Dict[str, Any]]:
        """Returns a list of all available input audio devices."""
        if sd is None:
            return []
        try:
            devices = sd.query_devices()
            input_devices = []
            for idx, dev in enumerate(devices):
                if dev.get("max_input_channels", 0) > 0:
                    input_devices.append({
                        "index": idx,
                        "name": dev.get("name", "Unknown"),
                        "channels": dev.get("max_input_channels", 1),
                        "default_samplerate": dev.get("default_samplerate", 16000),
                        "is_default": idx == sd.default.device[0]
                    })
            return input_devices
        except Exception as e:
            log_warning(f"Failed to query audio devices: {e}")
            return []

    def get_selected_device(self) -> Optional[int]:
        """Returns the configured or default microphone index."""
        if self.device_index is not None:
            return self.device_index
        if sd is None:
            return None
        try:
            default_dev = sd.default.device[0]
            if default_dev >= 0:
                return default_dev
            # Fallback to first input device
            mics = self.list_microphones()
            return mics[0]["index"] if mics else None
        except Exception:
            return None

    def record_speech(
        self,
        prompt_message: str = "Listening (speak now)...",
        silence_threshold: Optional[float] = None,
        silence_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
    ) -> np.ndarray:
        """
        Records audio from the microphone until speech is finished (silence detected).
        Returns a 1D float32 numpy array normalized to [-1.0, 1.0] at configured sample rate.
        """
        if sd is None:
            raise AudioCaptureError("sounddevice library is not available.")

        threshold = silence_threshold or self.silence_threshold
        silence_dur = silence_duration or self.silence_duration
        max_dur = max_duration or self.max_duration

        device = self.get_selected_device()
        chunk_duration = 0.1  # 100ms per block
        chunk_samples = int(self.sample_rate * chunk_duration)

        recorded_chunks: List[np.ndarray] = []
        speech_detected = False
        silence_start_time: Optional[float] = None
        start_time = time.time()

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                device=device,
                blocksize=chunk_samples,
            ) as stream:
                log_debug(f"Audio stream opened on device: {device}")

                while True:
                    elapsed = time.time() - start_time
                    if elapsed >= max_dur:
                        log_debug("Max recording duration reached.")
                        break

                    data, overflowed = stream.read(chunk_samples)
                    if overflowed:
                        log_debug("Audio buffer overflowed during recording.")

                    chunk = data.flatten()
                    recorded_chunks.append(chunk)

                    # Compute RMS energy of chunk
                    rms = np.sqrt(np.mean(chunk**2)) if len(chunk) > 0 else 0.0

                    if rms > threshold:
                        speech_detected = True
                        silence_start_time = None  # Reset silence timer
                    elif speech_detected:
                        # User was speaking and now went silent
                        if silence_start_time is None:
                            silence_start_time = time.time()
                        elif time.time() - silence_start_time >= silence_dur:
                            log_debug("End of speech detected via silence.")
                            break
                    else:
                        # Waiting for user to start speaking
                        # If silent for more than 5 seconds without initiating speech, break
                        if elapsed >= 5.0 and not speech_detected:
                            log_debug("No speech detected within initial timeout.")
                            break

        except Exception as e:
            raise AudioCaptureError(f"Microphone recording failed: {e}") from e

        if not recorded_chunks:
            return np.zeros(0, dtype=np.float32)

        audio_data = np.concatenate(recorded_chunks, axis=0)
        return audio_data
