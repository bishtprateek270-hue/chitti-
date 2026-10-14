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

        from src.audio.capture_service import get_audio_capture_service
        from src.audio.vad import VoiceActivityDetector, VADState

        capture_service = get_audio_capture_service()
        if not capture_service.is_running():
            capture_service.start(device_index=device)

        vad = VoiceActivityDetector(sample_rate=self.sample_rate)
        recorded_chunks: List[np.ndarray] = []
        pre_chunks: List[np.ndarray] = []
        speech_detected = False
        silence_start_time: Optional[float] = None
        start_time = time.time()
        is_done = False

        def mic_callback(chunk_16k: np.ndarray, rms: float):
            nonlocal speech_detected, silence_start_time, is_done
            if is_done:
                return

            vad_decision = vad.process_frame(chunk_16k)
            is_active_speech = (
                vad_decision.is_speech
                or vad_decision.state in (VADState.SPEECH, VADState.SPEECH_HANGOVER)
            )

            if is_active_speech:
                speech_detected = True
                silence_start_time = None
                recorded_chunks.append(chunk_16k)
            elif speech_detected:
                recorded_chunks.append(chunk_16k)
                if silence_start_time is None:
                    silence_start_time = time.time()
                elif time.time() - silence_start_time >= silence_dur:
                    is_done = True
            else:
                pre_chunks.append(chunk_16k)
                if len(pre_chunks) > 6:  # ~300ms pre-speech buffer
                    pre_chunks.pop(0)
                if (time.time() - start_time) >= 3.5:
                    is_done = True

        capture_service.subscribe("microphone_manager", mic_callback)
        try:
            while not is_done:
                time.sleep(0.02)
                if (time.time() - start_time) >= max_dur:
                    break
        except Exception as e:
            raise AudioCaptureError(f"Microphone recording failed: {e}") from e
        finally:
            is_done = True
            capture_service.unsubscribe("microphone_manager")

        if not speech_detected or not recorded_chunks:
            return np.zeros(0, dtype=np.float32)

        all_audio = pre_chunks + recorded_chunks
        audio_data = np.concatenate(all_audio, axis=0)
        return audio_data
