"""
Chitti Audio Subsystem Package.
Provides microphone streaming, whisper speech-to-text, pyttsx3 text-to-speech,
ambient wake-word spotting, dynamic VAD endpointing, and barge-in interruption.
"""

from src.audio.microphone import MicrophoneManager, AudioCaptureError
from src.audio.stt import STTEngine, WhisperSTT, STTError
from src.audio.tts import TTSEngine, Pyttsx3TTS
from src.audio.wake_word import WakeWordDetector, WakeWordEvent
from src.audio.streaming_stt import StreamingSTTEngine
from src.audio.interrupter import BargeInDetector
from src.audio.ambient_listener import AmbientVoiceListener, AmbientState

__all__ = [
    "MicrophoneManager",
    "AudioCaptureError",
    "STTEngine",
    "WhisperSTT",
    "STTError",
    "TTSEngine",
    "Pyttsx3TTS",
    "WakeWordDetector",
    "WakeWordEvent",
    "StreamingSTTEngine",
    "BargeInDetector",
    "AmbientVoiceListener",
    "AmbientState",
]
