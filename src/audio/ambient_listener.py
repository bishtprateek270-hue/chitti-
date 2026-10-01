"""
Chitti Ambient Hands-Free Voice Listener (Phase 5).
Orchestrates continuous background wake-word listening, streaming speech recording,
real-time Dynamic Island HUD state updates, and barge-in speech interruption.
"""

import enum
import threading
import time
from typing import Any, Callable, Dict, Optional
import numpy as np

from src.audio.interrupter import BargeInDetector
from src.audio.streaming_stt import StreamingSTTEngine
from src.audio.tts import TTSEngine
from src.audio.wake_word import WakeWordDetector, WakeWordEvent
from src.ui.hud_overlay import FloatingHUD, HUDMode
from src.utils.logging import log_chitti, log_debug, log_info, log_warn


class AmbientState(str, enum.Enum):
    """Lifecycle states of the Ambient Hands-Free Voice Engine."""
    STANDBY = "STANDBY"        # Passively listening for wake word ("Hey Chitti")
    WAKING = "WAKING"          # Wake word triggered, preparing speech capture
    LISTENING = "LISTENING"    # Actively recording user voice with streaming VAD
    THINKING = "THINKING"      # Brain / Agent processing query
    SPEAKING = "SPEAKING"      # Playing TTS response with barge-in enabled
    PAUSED = "PAUSED"          # Temporarily paused


class AmbientVoiceListener:
    """
    Coordinates 100% hands-free conversation with Chitti.
    Runs silently in the background, wakes up on 'Hey Chitti', records speech,
    hands off command to main pipeline, and speaks response with barge-in support.
    """

    def __init__(
        self,
        command_handler: Callable[[str], None],
        tts_engine: Optional[TTSEngine] = None,
        stt_engine: Optional[Any] = None,
        streaming_stt: Optional[StreamingSTTEngine] = None,
        wake_detector: Optional[WakeWordDetector] = None,
        hud: Optional[FloatingHUD] = None,
    ):
        self.command_handler = command_handler
        self.tts = tts_engine
        self.stt = stt_engine
        self.hud = hud

        self.wake_detector = wake_detector or WakeWordDetector(stt_engine=stt_engine)
        self.streaming_stt = streaming_stt or StreamingSTTEngine(stt_engine=stt_engine)
        self.barge_in = BargeInDetector()

        self.state = AmbientState.STANDBY
        self._is_running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self):
        """Starts the ambient hands-free background loop."""
        if self._is_running:
            return

        self._is_running = True
        self.state = AmbientState.STANDBY
        self.wake_detector.start(callback=self._on_wake_word_triggered)

        if self.hud:
            self.hud.set_mode(HUDMode.IDLE)

        log_chitti("[AMBIENT] 🎙️ Ambient hands-free voice mode active. Say 'Hey Chitti' to interact.")

    def stop(self):
        """Stops the ambient hands-free voice listener."""
        self._is_running = False
        self.wake_detector.stop()
        self.barge_in.stop_monitoring()
        self.state = AmbientState.STANDBY
        log_debug("[AMBIENT] Ambient voice listener stopped.")

    def _on_wake_word_triggered(self, event: WakeWordEvent):
        """Triggered automatically when keyword ('Hey Chitti') is recognized."""
        with self._lock:
            if self.state != AmbientState.STANDBY:
                return
            self.state = AmbientState.WAKING

        log_chitti(f"[AMBIENT] ⚡ Wake word spotted: '{event.keyword}'. Transitioning to LISTENING...")

        # Update HUD state to listening with active pulse
        if self.hud:
            self.hud.set_mode(HUDMode.LISTENING)
            self.hud.set_progress("Listening (speak now)...")

        # Start recording in background worker
        worker = threading.Thread(target=self._record_and_process_speech, daemon=True, name="AmbientSpeechWorker")
        worker.start()

    def _record_and_process_speech(self):
        """Records speech until silence, transcribes, and executes command."""
        try:
            with self._lock:
                self.state = AmbientState.LISTENING

            # Pause wake word detector during active recording
            self.wake_detector.pause()

            # Record and transcribe with streaming VAD
            transcribed_text = self.streaming_stt.transcribe_streaming(
                on_speech_start=lambda: self.hud.set_progress("Hearing you...") if self.hud else None,
                on_chunk=self._on_audio_chunk,
            )

            if not transcribed_text or not transcribed_text.strip():
                log_debug("[AMBIENT] No recognizable speech received.")
                self._return_to_standby()
                return

            log_chitti(f"[AMBIENT] 🗣️ User said: '{transcribed_text}'")

            with self._lock:
                self.state = AmbientState.THINKING

            if self.hud:
                self.hud.set_mode(HUDMode.THINKING)
                self.hud.set_progress(f"Processing: '{transcribed_text}'")

            # Dispatch command to Chitti's main pipeline
            self.command_handler(transcribed_text)

        except Exception as e:
            log_warn(f"[AMBIENT] Speech processing notice: {e}")
        finally:
            self._return_to_standby()

    def _on_audio_chunk(self, chunk: np.ndarray, energy: float):
        """Feeds audio energy into HUD visualizer if available."""
        if self.hud and hasattr(self.hud, "update_waveform"):
            try:
                self.hud.update_waveform(energy)
            except Exception:
                pass

    def speak_with_barge_in(self, text: str):
        """
        Plays speech through TTS with active barge-in monitoring so user can interrupt.
        """
        if not self.tts or not text:
            return

        with self._lock:
            self.state = AmbientState.SPEAKING

        if self.hud:
            self.hud.set_mode(HUDMode.SPEAKING)

        def interrupt_callback():
            log_chitti("[AMBIENT] ✋ Speech interrupted by user barge-in!")
            if hasattr(self.tts, "stop"):
                self.tts.stop()

        self.barge_in.start_monitoring(on_interrupt=interrupt_callback)
        try:
            self.tts.speak(text, wait=True)
        finally:
            self.barge_in.stop_monitoring()
            self._return_to_standby()

    def _return_to_standby(self):
        """Resets the state machine back to passive standby."""
        with self._lock:
            self.state = AmbientState.STANDBY

        self.wake_detector.resume()
        if self.hud:
            self.hud.set_mode(HUDMode.IDLE)
            self.hud.set_progress("")
