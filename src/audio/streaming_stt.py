"""
Chitti Real-Time Streaming Speech-to-Text (STT) Engine (Phase 5).
Captures audio with a pre-speech ring-buffer, dynamic Voice Activity Detection (VAD),
fast auto-silence cutoff, and low-latency Whisper transcription.
"""

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from src.audio.stt import STTEngine, WhisperSTT
from src.config import AudioConfig, get_config
from src.utils.logging import log_chitti, log_debug, log_info, log_warn

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    sd = None
    HAS_SOUNDDEVICE = False


@dataclass
class StreamingAudioState:
    """Tracks the live state of an active audio streaming session."""
    is_speaking: bool = False
    speech_frames: int = 0
    silence_frames: int = 0
    total_samples: int = 0
    start_time: float = 0.0


class StreamingSTTEngine:
    """
    Manages low-latency real-time voice streaming, automatic speech endpointing,
    and background transcription.
    """

    def __init__(
        self,
        stt_engine: Optional[STTEngine] = None,
        config: Optional[AudioConfig] = None,
        pre_speech_seconds: float = 0.35,
        silence_cutoff_seconds: float = 0.75,
        min_speech_seconds: float = 0.40,
        max_record_seconds: float = 15.0,
    ):
        self.config = config or get_config().audio
        self.sample_rate = self.config.sample_rate
        self.stt_engine = stt_engine or WhisperSTT()

        self.pre_speech_seconds = pre_speech_seconds
        self.silence_cutoff_seconds = 1.2
        self.min_speech_seconds = min_speech_seconds
        self.max_record_seconds = max_record_seconds

        self.chunk_size = int(self.sample_rate * 0.05)  # 50ms chunk (800 samples at 16kHz)
        self.pre_buffer_size = int(self.sample_rate * self.pre_speech_seconds)

        self._pre_buffer = np.zeros(self.pre_buffer_size, dtype=np.float32)
        self._recorded_chunks: List[np.ndarray] = []
        self._is_recording = False
        self._lock = threading.Lock()

        # Dynamic VAD Energy Thresholds
        self.silence_threshold = self.config.silence_threshold
        self._adaptive_floor = 0.015

    def record_until_silence(
        self,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_chunk: Optional[Callable[[np.ndarray, float], None]] = None,
        timeout: Optional[float] = None,
        device_index: Optional[int] = None,
    ) -> np.ndarray:
        """
        Synchronously streams and records microphone input until the user stops speaking.
        Automatically includes pre-speech buffer so leading consonants are never truncated.
        """
        if not HAS_SOUNDDEVICE or not sd:
            raise RuntimeError("sounddevice is required for microphone audio streaming.")

        max_sec = timeout or self.max_record_seconds
        silence_cutoff_frames = int(self.silence_cutoff_seconds / 0.05)
        min_speech_frames = int(self.min_speech_seconds / 0.05)
        max_total_frames = int(max_sec / 0.05)

        state = StreamingAudioState()
        self._recorded_chunks.clear()
        self._pre_buffer.fill(0)
        self._is_recording = True
        speech_started = False

        def stream_callback(indata, frames, time_info, status):
            nonlocal speech_started
            if not self._is_recording:
                return

            chunk = indata[:, 0].copy() if indata.ndim > 1 else indata.flatten().copy()
            energy = float(np.sqrt(np.mean(chunk**2) + 1e-12))

            # Adapt noise floor
            if not state.is_speaking and energy < self._adaptive_floor:
                self._adaptive_floor = 0.95 * self._adaptive_floor + 0.05 * energy

            threshold = max(self.silence_threshold, self._adaptive_floor * 1.6)

            with self._lock:
                if energy >= threshold:
                    state.silence_frames = 0
                    state.speech_frames += 1

                    if not state.is_speaking and state.speech_frames >= 2:
                        state.is_speaking = True
                        if not speech_started:
                            speech_started = True
                            if on_speech_start:
                                try:
                                    on_speech_start()
                                except Exception:
                                    pass

                    if state.is_speaking:
                        self._recorded_chunks.append(chunk)
                else:
                    if state.is_speaking:
                        state.silence_frames += 1
                        self._recorded_chunks.append(chunk)
                    else:
                        # Roll pre-buffer
                        self._pre_buffer = np.roll(self._pre_buffer, -len(chunk))
                        self._pre_buffer[-len(chunk):] = chunk

            if on_chunk:
                try:
                    on_chunk(chunk, energy)
                except Exception:
                    pass

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.chunk_size,
                device=device_index,
                callback=stream_callback,
            ):
                start_time = time.time()
                while self._is_recording:
                    time.sleep(0.03)
                    elapsed = time.time() - start_time

                    # Check exit conditions
                    with self._lock:
                        if state.is_speaking and state.silence_frames >= silence_cutoff_frames:
                            log_debug(f"[STREAMING STT] Auto-silence cutoff triggered ({elapsed:.2f}s).")
                            break
                        if elapsed >= max_sec:
                            log_debug(f"[STREAMING STT] Max recording timeout reached ({max_sec}s).")
                            break
                        # Timeout if user never spoke after 5s
                        if not state.is_speaking and elapsed >= 6.0:
                            log_debug("[STREAMING STT] No speech detected within initial window.")
                            break

        finally:
            self._is_recording = False

        # Assemble full audio: pre-speech buffer + recorded speech chunks
        with self._lock:
            if not self._recorded_chunks:
                return np.array([], dtype=np.float32)

            recorded_body = np.concatenate(self._recorded_chunks)
            # Prepend the pre-speech buffer to preserve initial syllables
            full_audio = np.concatenate([self._pre_buffer, recorded_body])
            return full_audio

    def transcribe_streaming(
        self,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_chunk: Optional[Callable[[np.ndarray, float], None]] = None,
        device_index: Optional[int] = None,
    ) -> str:
        """
        Records the user's voice with real-time streaming VAD and immediately transcribes.
        Returns the clean recognized text string.
        """
        audio = self.record_until_silence(
            on_speech_start=on_speech_start,
            on_chunk=on_chunk,
            device_index=device_index,
        )

        if len(audio) < int(self.sample_rate * 0.3):
            return ""

        log_info(f"[STREAMING STT] Transcribing {len(audio)/self.sample_rate:.2f}s audio stream...")
        text = self.stt_engine.transcribe(audio)
        return text.strip()
