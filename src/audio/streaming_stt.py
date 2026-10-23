"""
Chitti Real-Time Streaming Speech-to-Text (STT) Engine (Phase 5).
Captures audio with a pre-speech ring-buffer, dynamic Voice Activity Detection (VAD),
fast auto-silence cutoff, and low-latency transcription.
"""

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from src.audio.audio_utils import get_best_input_device, resample_to_16k
from src.audio.capture_service import get_audio_capture_service
from src.audio.stt import STTEngine, get_stt
from src.audio.vad import VoiceActivityDetector, VADState
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
        pre_speech_seconds: float = 0.50,
        silence_cutoff_seconds: float = 0.85,
        min_speech_seconds: float = 0.15,
        max_record_seconds: float = 7.0,
    ):
        self.config = config or get_config().audio
        self.sample_rate = 16000
        self.stt_engine = stt_engine or get_stt()

        self.pre_speech_seconds = pre_speech_seconds
        self.silence_cutoff_seconds = silence_cutoff_seconds
        self.min_speech_seconds = min_speech_seconds
        self.max_record_seconds = max_record_seconds

        self.chunk_size = int(self.sample_rate * 0.05)  # 50ms chunk (800 samples at 16kHz)
        self.pre_buffer_size = int(self.sample_rate * self.pre_speech_seconds)

        self._pre_buffer = np.zeros(self.pre_buffer_size, dtype=np.float32)
        self._recorded_chunks: List[np.ndarray] = []
        self._is_recording = False
        self._lock = threading.Lock()

        # Multi-feature Voice Activity Detector (Phase 3)
        self.vad = VoiceActivityDetector(
            sample_rate=self.sample_rate,
            frame_duration=0.05,
            snr_onset_ratio=1.35,
            snr_hangover_ratio=1.15,
            min_voice_rms=0.0015,
            onset_consecutive_frames=2,
            hangover_consecutive_frames=int(silence_cutoff_seconds / 0.05),
        )

        # Legacy backward-compatible attributes
        self.silence_threshold = 0.005
        self._adaptive_floor = 0.002

    def stop_recording(self):
        """Immediately halts active recording loop."""
        self._is_recording = False

    def record_until_silence(
        self,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_chunk: Optional[Callable[[np.ndarray, float], None]] = None,
        timeout: Optional[float] = None,
        device_index: Optional[int] = None,
        initial_audio: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Synchronously streams and records microphone input until the user stops speaking.
        Automatically includes pre-speech buffer so leading consonants are never truncated.
        Returns 16000Hz float32 audio array.
        """
        if not HAS_SOUNDDEVICE or not sd:
            raise RuntimeError("sounddevice is required for microphone audio streaming.")

        max_sec = timeout or self.max_record_seconds
        silence_cutoff_frames = int(self.silence_cutoff_seconds / 0.05)
        min_speech_frames = int(self.min_speech_seconds / 0.05)

        state = StreamingAudioState()
        self._recorded_chunks.clear()
        self._pre_buffer.fill(0)
        self._is_recording = True
        speech_started = False
        voiced_frames_count = 0

        self.vad.reset()

        if initial_audio is not None and len(initial_audio) > 0:
            self._recorded_chunks.append(initial_audio.astype(np.float32))
            state.is_speaking = True
            state.speech_frames = max(2, len(initial_audio) // self.chunk_size)
            speech_started = True
            voiced_frames_count = 2

        capture_service = get_audio_capture_service()
        if not capture_service.is_running():
            capture_service.start(device_index=device_index)

        def subscriber_callback(chunk_16k: np.ndarray, energy: float):
            nonlocal speech_started, voiced_frames_count
            if not self._is_recording:
                return

            vad_decision = self.vad.process_frame(chunk_16k)
            if vad_decision.is_voiced:
                voiced_frames_count += 1

            is_active_speech = vad_decision.is_speech
            is_in_speech_state = (
                vad_decision.is_speech
                or vad_decision.state in (VADState.SPEECH, VADState.SPEECH_HANGOVER)
            )

            with self._lock:
                if is_active_speech:
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
                        self._recorded_chunks.append(chunk_16k)
                elif is_in_speech_state:
                    # Brief inter-word pause during speech hangover:
                    # append audio to bridge syllables, but count towards silence if speech paused
                    if state.is_speaking:
                        state.silence_frames += 1
                        self._recorded_chunks.append(chunk_16k)
                else:
                    if state.is_speaking:
                        state.silence_frames += 1
                        # Append at most 4 padding frames (200ms) after speech ends to prevent recording seconds of hiss
                        if state.silence_frames <= 4:
                            self._recorded_chunks.append(chunk_16k)
                    else:
                        # Roll pre-buffer
                        self._pre_buffer = np.roll(self._pre_buffer, -len(chunk_16k))
                        self._pre_buffer[-len(chunk_16k):] = chunk_16k

            if on_chunk:
                try:
                    on_chunk(chunk_16k, energy)
                except Exception:
                    pass

        capture_service.subscribe("streaming_stt", subscriber_callback)
        start_time = time.time()
        try:
            while self._is_recording:
                time.sleep(0.025)
                elapsed = time.time() - start_time

                # Check exit conditions
                with self._lock:
                    if state.is_speaking and state.silence_frames >= silence_cutoff_frames:
                        log_debug(f"[STREAMING STT] Auto-silence cutoff triggered ({elapsed:.2f}s).")
                        break
                    if elapsed >= max_sec:
                        log_debug(f"[STREAMING STT] Max recording timeout reached ({max_sec}s).")
                        break
                    # Timeout if user never spoke after 3.5s
                    if not state.is_speaking and elapsed >= 3.5:
                        log_debug("[STREAMING STT] No speech detected within initial window.")
                        break
        except Exception as e:
            log_warn(f"[STREAMING STT] Audio capture notice: {e}")
        finally:
            self._is_recording = False
            capture_service.unsubscribe("streaming_stt")

        # Assemble full audio: pre-speech buffer + recorded speech chunks (trimmed)
        with self._lock:
            if not self._recorded_chunks or state.speech_frames < min_speech_frames:
                return np.array([], dtype=np.float32)

            # Trim trailing silence frames beyond 4 frames (200ms padding) to prevent STT degradation
            excess_silence = max(0, state.silence_frames - 4)
            if excess_silence > 0 and len(self._recorded_chunks) > excess_silence:
                chunks_to_use = self._recorded_chunks[:-excess_silence]
            else:
                chunks_to_use = self._recorded_chunks

            if not chunks_to_use:
                return np.array([], dtype=np.float32)

            recorded_body = np.concatenate(chunks_to_use)
            full_audio = np.concatenate([self._pre_buffer, recorded_body])
            if full_audio.ndim > 1:
                full_audio = full_audio[:, 0].flatten() if full_audio.shape[1] > 1 else full_audio.flatten()
            return full_audio.astype(np.float32)

    def transcribe_streaming(
        self,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_chunk: Optional[Callable[[np.ndarray, float], None]] = None,
        device_index: Optional[int] = None,
        initial_audio: Optional[np.ndarray] = None,
    ) -> str:
        """
        Records the user's voice with real-time streaming VAD and immediately transcribes.
        Returns the clean recognized text string.
        """
        audio = self.record_until_silence(
            on_speech_start=on_speech_start,
            on_chunk=on_chunk,
            device_index=device_index,
            initial_audio=initial_audio,
        )

        if len(audio) < int(self.sample_rate * 0.15):
            return ""

        peak_amp = float(np.max(np.abs(audio)))
        rms_val = float(np.sqrt(np.mean(audio**2) + 1e-12))
        if peak_amp < 0.0015 or rms_val < 0.0006:
            log_debug(f"[STREAMING STT] Audio energy too faint (peak={peak_amp:.4f}, rms={rms_val:.4f}). Discarded as silence.")
            return ""

        # Pre-normalize audio to standard vocal level (0.75 peak) to ensure crystal-clear transcription
        if peak_amp >= 0.0015 and peak_amp < 0.65:
            gain = min(20.0, 0.75 / peak_amp)
            audio = audio * gain

        log_info(f"[STREAMING STT] Transcribing {len(audio)/self.sample_rate:.2f}s audio stream...")
        text = self.stt_engine.transcribe(audio)
        return text.strip()

