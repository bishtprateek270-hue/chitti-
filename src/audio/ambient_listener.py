"""
Chitti Ambient Hands-Free Voice Listener & System State Coordinator.
Orchestrates continuous background wake-word listening, streaming speech recording,
real-time Dynamic Island HUD state updates, barge-in speech interruption,
and authoritative SystemStateManager coordination for Mode A (Wake-Word) & Mode B (Continuous).
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
from src.config import get_config
from src.core.phrase_matcher import get_phrase_matcher, ControlCommand
from src.core.state_manager import get_state_manager, SystemState
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
    Runs silently in the background, wakes up on configured wake phrases,
    enforces sleep/wake state, and routes emergency stops via high-priority fast path.
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

        self.state_mgr = get_state_manager()
        self.phrase_matcher = get_phrase_matcher()

        self.state = AmbientState.STANDBY
        self._is_running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Wire with authoritative state manager
        self.state_mgr.register_on_sleep(self._on_system_sleep)
        self.state_mgr.register_on_wake(self._on_system_wake)
        self.state_mgr.register_on_interrupt(self._on_system_interrupt)

    def start(self):
        """Starts the ambient hands-free background loop."""
        if self._is_running:
            return

        self._is_running = True
        self.state = AmbientState.STANDBY
        self.wake_detector.start(callback=self._on_wake_word_triggered)

        if self.hud:
            self.hud.set_mode(HUDMode.IDLE)

        mode_desc = "Wake-Word Mode (Say 'Hey Chitti' to interact)" if self.state_mgr.operating_mode == "wake_word" else "Continuous Assistant Mode"
        log_chitti(f"[AMBIENT] 🎙️ Ambient hands-free voice mode active: {mode_desc}.")

    def stop(self):
        """Stops the ambient hands-free voice listener."""
        self._is_running = False
        self.wake_detector.stop()
        self.barge_in.stop_monitoring()
        self.state = AmbientState.STANDBY
        log_debug("[AMBIENT] Ambient voice listener stopped.")

    def _on_system_sleep(self):
        """Callback when SystemStateManager enters SLEEPING state."""
        with self._lock:
            self.state = AmbientState.STANDBY
        if hasattr(self.tts, "stop"):
            self.tts.stop()
        self.barge_in.stop_monitoring()
        self.wake_detector.resume(cooldown=4.0)
        if self.hud:
            self.hud.set_mode(HUDMode.IDLE)
            self.hud.set_progress("")
            self.hud.hide()

    def _on_system_wake(self):
        """Callback when SystemStateManager wakes up."""
        if self.hud:
            self.hud.set_mode(HUDMode.IDLE)
            self.hud.set_progress("Awake (Ready)")

    def _on_system_interrupt(self):
        """Callback when SystemStateManager triggers global interrupt."""
        if hasattr(self.tts, "stop"):
            self.tts.stop()
        self.barge_in.stop_monitoring()
        with self._lock:
            self.state = AmbientState.STANDBY
        if self.hud:
            self.hud.set_mode(HUDMode.IDLE)
            self.hud.set_progress("")
            self.hud.hide()
        self.wake_detector.resume(cooldown=3.0)

    def _on_wake_word_triggered(self, event: WakeWordEvent):
        """Triggered automatically when a wake phrase is recognized by the detector."""
        if self.state_mgr.get_state() in (SystemState.SPEAKING, SystemState.INTERRUPTING):
            log_debug("[AMBIENT] Suppressed wake event during speech output.")
            return

        with self._lock:
            if self.state != AmbientState.STANDBY:
                return
            self.state = AmbientState.WAKING

        log_chitti(f"[AMBIENT] ⚡ Wake word spotted: '{event.keyword}'. Transitioning to LISTENING...")

        # Transition authoritative state to AWAKE and LISTENING
        self.state_mgr.wake()
        self.state_mgr.begin_listening()

        if self.hud:
            self.hud.show()
            self.hud.set_mode(HUDMode.LISTENING)
            self.hud.set_progress("Listening (speak now)...")

        # If a full command was already captured with the wake word in one breath:
        if event.command and len(event.command.strip()) >= 3:
            worker = threading.Thread(
                target=self._process_direct_command,
                args=(event.command.strip(),),
                daemon=True,
                name="AmbientDirectWorker",
            )
            worker.start()
        else:
            # Start streaming recording in background worker
            worker = threading.Thread(
                target=self._record_and_process_speech,
                daemon=True,
                name="AmbientSpeechWorker",
            )
            worker.start()

    def _process_direct_command(self, command_text: str):
        """Directly processes a command captured alongside the wake word."""
        try:
            self.wake_detector.pause()
            self._handle_incoming_transcript(command_text)
        except Exception as e:
            log_warn(f"[AMBIENT] Direct command processing notice: {e}")
        finally:
            self._return_to_standby()

    def _record_and_process_speech(self):
        """Records speech until silence, transcribes, and routes through priority handler."""
        try:
            with self._lock:
                self.state = AmbientState.LISTENING

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

            self._handle_incoming_transcript(transcribed_text)

        except Exception as e:
            log_warn(f"[AMBIENT] Speech processing notice: {e}")
        finally:
            self._return_to_standby()

    def _handle_incoming_transcript(self, transcript: str):
        """
        HIGH-PRIORITY FAST PATH ROUTER:
        Evaluates commands against Priority Hierarchy (Stop > Sleep > Wake > Normal)
        BEFORE sending any audio or text to the LLM.
        """
        raw = transcript.strip()
        log_chitti(f"[AMBIENT] 🗣️ User said: '{raw}'")

        # Fast-Path Phrase Evaluation
        match_res = self.phrase_matcher.classify_command(raw)

        # ---------------- PRIORITY 1: GLOBAL STOP / INTERRUPT ----------------
        if match_res.command == ControlCommand.STOP:
            log_chitti(f"[CHITTI][CONTROL] 🛑 Immediate STOP command recognized: '{match_res.matched_phrase}'")
            if self.tts and hasattr(self.tts, "stop"):
                self.tts.stop()
            self.barge_in.stop_monitoring()
            self.state_mgr.interrupt(reason="user_stop")
            self.state_mgr.sleep()
            with self._lock:
                self.state = AmbientState.STANDBY
            if self.hud:
                self.hud.set_mode(HUDMode.IDLE)
                self.hud.set_progress("")
                self.hud.hide()
            self.wake_detector.resume(cooldown=4.0)
            return

        # ---------------- PRIORITY 2: SLEEP COMMAND ----------------
        if match_res.command == ControlCommand.SLEEP:
            log_chitti(f"[CHITTI][CONTROL] 😴 SLEEP command recognized: '{match_res.matched_phrase}'")
            if self.tts and hasattr(self.tts, "stop"):
                self.tts.stop()
            self.barge_in.stop_monitoring()
            self.state_mgr.sleep()
            with self._lock:
                self.state = AmbientState.STANDBY
            if self.hud:
                self.hud.set_mode(HUDMode.IDLE)
                self.hud.set_progress("")
                self.hud.hide()
            self.wake_detector.resume(cooldown=4.0)
            return

        # ---------------- PRIORITY 3: WAKE COMMAND ----------------
        if match_res.command == ControlCommand.WAKE:
            log_chitti(f"[CHITTI][CONTROL] ⚡ WAKE command recognized: '{match_res.matched_phrase}'")
            self.state_mgr.wake()
            if self.hud:
                self.hud.set_mode(HUDMode.IDLE)
                self.hud.set_progress("Awake (Haan, boliye)")
                self.hud.set_response("Haan, boliye.")

            if match_res.extracted_tail:
                # User said wake phrase + command together (e.g. "Chitti open Chrome")
                self._handle_incoming_transcript(match_res.extracted_tail)
            else:
                if self.tts:
                    try:
                        self.tts.speak("Haan, boliye.", wait=False)
                    except Exception:
                        pass
            return

        # ---------------- PRIORITY 4: NORMAL USER COMMAND ----------------
        # Guard: If Chitti is SLEEPING, ignore ordinary speech entirely!
        if self.state_mgr.is_sleeping():
            log_debug(f"[CHITTI][SLEEP] Ignored ordinary conversation while SLEEPING: '{raw}'")
            return

        request_id = self.state_mgr.new_request()
        self.state_mgr.begin_processing(request_id)

        with self._lock:
            self.state = AmbientState.THINKING

        if self.hud:
            self.hud.set_mode(HUDMode.THINKING)
            self.hud.set_progress(f"Processing: '{raw}'")

        # Dispatch command to Chitti's main pipeline
        try:
            self.command_handler(raw, request_id=request_id)
        except TypeError:
            self.command_handler(raw)

    def _on_audio_chunk(self, chunk: np.ndarray, energy: float):
        """Feeds audio energy into HUD visualizer if available."""
        if self.hud and hasattr(self.hud, "update_waveform"):
            try:
                self.hud.update_waveform(energy)
            except Exception:
                pass

    def speak_with_barge_in(self, text: str, request_id: Optional[int] = None):
        """
        Plays speech through TTS with active barge-in monitoring so user can interrupt.
        Prevents Chitti from hearing its own speech as commands while allowing user interruptions.
        """
        if not self.tts or not text:
            return

        state_mgr = get_state_manager()
        if not state_mgr.can_speak(request_id):
            log_debug(f"[AMBIENT] Suppressed speech output: request #{request_id} is invalid or sleeping.")
            return

        state_mgr.begin_speaking(request_id)
        self.wake_detector.pause()

        with self._lock:
            self.state = AmbientState.SPEAKING

        if self.hud:
            self.hud.set_mode(HUDMode.SPEAKING)

        def interrupt_callback():
            log_chitti("[AMBIENT] ✋ Speech interrupted by user barge-in!")
            self.state_mgr.interrupt(reason="barge_in")
            if hasattr(self.tts, "stop"):
                self.tts.stop()

        self.barge_in.start_monitoring(on_interrupt=interrupt_callback)
        try:
            from src.audio.tts import EdgeNeuralTTS, HybridTTS, SpeechController
            if isinstance(self.tts, (EdgeNeuralTTS, HybridTTS, SpeechController)) and request_id is not None:
                self.tts.speak(text, wait=True, request_id=request_id)
            else:
                self.tts.speak(text, wait=True)
        finally:
            self.barge_in.stop_monitoring()
            time.sleep(0.4)  # 400ms acoustic dampening to absorb room echo
            state_mgr.finish_task(request_id)
            self._return_to_standby()

    def go_to_sleep(self):
        """Immediately stops ongoing speech and puts ambient listener in quiet standby."""
        log_chitti("[AMBIENT] 😴 Entering quiet standby mode. Say 'Hey Chitti' or press Alt+Space to wake.")
        self.state_mgr.sleep()
        if self.tts and hasattr(self.tts, "stop"):
            self.tts.stop()
        self.barge_in.stop_monitoring()
        with self._lock:
            self.state = AmbientState.STANDBY
        if self.hud:
            self.hud.set_mode(HUDMode.IDLE)
            self.hud.set_progress("")
            self.hud.hide()
        self.wake_detector.resume(cooldown=4.0)

    def _return_to_standby(self):
        """Resets the state machine back to passive standby."""
        with self._lock:
            self.state = AmbientState.STANDBY

        self.wake_detector.resume(cooldown=1.5)
        if self.hud:
            self.hud.set_mode(HUDMode.IDLE)
            self.hud.set_progress("")
            if self.state_mgr.is_sleeping():
                self.hud.hide()
