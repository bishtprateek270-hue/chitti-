"""
Chitti - Personal Multimodal AI Desktop Companion Robot
PHASE 1: Voice AI Brain  +  PHASE 2: Long-Term Memory  +  PHASE 3: Computer Vision
"""

import sys
import os
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import get_config
from src.utils.logging import (
    setup_logger,
    log_chitti,
    log_error,
    log_warning,
    log_state,
    log_debug
)
from src.brain.personality import ConversationHistory, CHITTI_SYSTEM_PROMPT
from src.brain.llm import get_llm, LLMError, LLMConnectionError, LLMTimeoutError, LLMModelNotFoundError
from src.audio.microphone import MicrophoneManager, AudioCaptureError
from src.audio.stt import get_stt, STTError
from src.audio.tts import get_tts, TTSError
from src.memory.manager import MemoryManager
from src.vision.vision_manager import VisionManager


BANNER = r"""
+------------------------------------------------------------------+
|                                                                  |
|                          C H I T T I                             |
|             Personal AI Desktop Companion Robot                  |
|    Phase 1: Voice  +  Phase 2: Memory  +  Phase 3: Vision        |
|                                                                  |
+------------------------------------------------------------------+
"""


class ChittiController:
    """Main application controller for Chitti (Phase 1, Phase 2, and Phase 3)."""

    def __init__(self):
        self.config = get_config()
        setup_logger(level=self.config.log_level)
        self.history = ConversationHistory(system_prompt=CHITTI_SYSTEM_PROMPT)
        self.mic = None
        self.stt = None
        self.tts = None
        self.llm = None
        self.memory = None
        self.vision = None

    def initialize(self):
        """Initializes all hardware and AI subsystems."""
        print(BANNER)
        log_chitti("Starting Chitti Voice Brain, Memory & Vision Systems...")

        # 1. Initialize Audio/Microphone
        log_chitti("Checking audio devices...")
        try:
            self.mic = MicrophoneManager(self.config.audio)
            devices = self.mic.list_microphones()
            if devices:
                default_dev = next((d["name"] for d in devices if d.get("is_default")), devices[0]["name"])
                log_chitti(f"Audio input device ready: {default_dev}")
            else:
                log_warning("No microphone detected. Text input fallback will be enabled.")
        except Exception as e:
            log_warning(f"Audio initialization notice: {e}")

        # 2. Initialize TTS
        log_chitti("Initializing speech synthesis (TTS)...")
        try:
            self.tts = get_tts(self.config.tts)
            log_chitti(f"TTS engine ready ({self.config.tts.engine}).")
        except Exception as e:
            log_error(f"TTS initialization failed: {e}")
            self.tts = None

        # 3. Initialize STT (Whisper)
        log_chitti(f"Loading speech recognition model ({self.config.stt.model}) on {self.config.stt.device.upper()}...")
        try:
            self.stt = get_stt(self.config.stt)
            log_chitti("STT module initialized.")
        except Exception as e:
            log_error(f"STT initialization failed: {e}")
            self.stt = None

        # 4. Initialize Long-Term Memory (Phase 2)
        if self.config.memory.enabled:
            log_chitti("Initializing persistent long-term memory...")
            try:
                self.memory = MemoryManager(
                    db_path=self.config.memory.db_path,
                    similarity_threshold=self.config.memory.similarity_threshold,
                    importance_threshold=self.config.memory.importance_threshold,
                    embedding_model=self.config.memory.embedding_model,
                )
                stored_count = self.memory.db.count()
                log_chitti(f"Persistent memory loaded ({stored_count} memories stored).")
            except Exception as e:
                log_error(f"Failed to initialize memory manager: {e}")
                self.memory = None
        else:
            log_chitti("Long-term memory is disabled in configuration.")

        # 5. Initialize Computer Vision (Phase 3)
        if self.config.vision.enabled:
            log_chitti("Initializing computer vision system...")
            try:
                self.vision = VisionManager()
                registered_faces_count = self.vision.face_db.count()
                log_chitti(f"Vision system ready ({registered_faces_count} registered faces in database).")
            except Exception as e:
                log_warning(f"Vision initialization notice: {e}. Running in non-vision mode.")
                self.vision = None
        else:
            log_chitti("Computer vision is disabled in configuration.")

        # 6. Initialize LLM (Ollama)
        log_chitti(f"Connecting to AI brain at {self.config.llm.base_url} (model: {self.config.llm.model})...")
        self.llm = get_llm(self.config.llm)
        if self.llm.check_connection():
            available = self.llm.list_available_models()
            log_chitti(f"Ollama connected successfully. Available models: {available if available else 'None'}")
            if self.config.llm.model not in available and not any(self.config.llm.model in m for m in available):
                log_warning(
                    f"Configured model '{self.config.llm.model}' was not found in Ollama. "
                    f"If needed, run `ollama pull {self.config.llm.model}` in another terminal."
                )
        else:
            log_warning(
                f"Ollama server is not responding at {self.config.llm.base_url}. "
                f"Ensure Ollama is started with `ollama serve` or the desktop app."
            )

        log_chitti("System initialization complete. Ready for interaction!\n")

    def speak(self, text: str):
        """Speaks the response using TTS if available."""
        if self.tts and text:
            try:
                self.tts.speak(text, wait=True)
            except Exception as e:
                log_warning(f"Speech playback failed: {e}")

    def process_user_input(self, user_text: str):
        """Processes user input through Vision/Memory/LLM, records history, and speaks output."""
        if not user_text or not user_text.strip():
            log_chitti("I didn't catch that. Could you repeat?")
            self.speak("I didn't catch that. Could you repeat?")
            return

        print(f"\nYou: {user_text}")

        # 1. Check for Face Management Commands (e.g. "Who is registered?", "Remove Prateek from face recognition")
        if self.vision is not None:
            try:
                face_cmd_result = self.vision.handle_face_commands(user_text)
                if face_cmd_result is not None:
                    action_tag, response_text = face_cmd_result
                    self.history.add_user_message(user_text)
                    self.history.add_assistant_message(response_text)

                    log_state("CHITTI")
                    print(response_text)
                    self.speak(response_text)
                    return
            except Exception as e:
                log_warning(f"Face command processing failed: {e}")

        # 2. Check for Explicit Memory Command (e.g. "Remember that...", "Forget that...")
        if self.memory is not None:
            try:
                mem_result = self.memory.handle_interaction(user_text)
                if mem_result is not None:
                    action_tag, response_text = mem_result
                    self.history.add_user_message(user_text)
                    self.history.add_assistant_message(response_text)

                    log_state("CHITTI")
                    print(response_text)
                    self.speak(response_text)
                    return
            except Exception as e:
                log_warning(f"Memory command processing failed: {e}")

        # Add to short-term session history
        self.history.add_user_message(user_text)

        # 3. Vision Perception Trigger (if user asks what Chitti sees or who is there)
        vision_context = None
        recognized_names_seen = []

        if self.vision is not None and self.vision.is_vision_query(user_text):
            try:
                log_state("VISION", "Analyzing camera feed...")
                vision_result = self.vision.analyze_frame()
                log_chitti(f"[VISION] {vision_result.summary_text}")
                vision_context = self.vision.format_vision_context_for_llm(vision_result)
                recognized_names_seen = vision_result.known_people_names
            except Exception as e:
                log_warning(f"Vision analysis failed: {e}")
                vision_context = "\n[VISION NOTICE: Camera is currently unavailable.]"

        self.history.set_vision_context(vision_context)

        # 4. Retrieve relevant long-term memories
        relevant_memories = []
        if self.memory is not None:
            try:
                # Query memories for the user's prompt
                relevant_memories = self.memory.recall(user_text, top_k=self.config.memory.top_k)

                # If a recognized person was detected in vision, also fetch memories about them
                for person_name in recognized_names_seen:
                    person_mems = self.memory.recall(person_name, top_k=2)
                    for pm in person_mems:
                        if pm.id not in [m.id for m in relevant_memories]:
                            relevant_memories.append(pm)

                if relevant_memories:
                    log_chitti(f"[MEMORY] Retrieved {len(relevant_memories)} relevant memories for context.")
            except Exception as e:
                log_warning(f"Memory retrieval failed: {e}")
                relevant_memories = []

        self.history.set_relevant_memories(relevant_memories)

        # 5. Generate LLM response
        log_state("THINKING")
        try:
            messages = self.history.get_messages_for_llm()
            response_text = self.llm.generate_response(messages)
        except LLMConnectionError as e:
            log_error(f"Ollama is unavailable: {e}")
            response_text = "I can't reach my local AI brain right now. Please check if Ollama is running."
        except LLMModelNotFoundError as e:
            log_error(f"Model not found: {e}")
            response_text = f"My configured AI model ({self.config.llm.model}) is not installed in Ollama yet."
        except LLMTimeoutError as e:
            log_error(f"LLM request timed out: {e}")
            response_text = "Thinking about that took too long and timed out. Let's try again."
        except LLMError as e:
            log_error(f"Brain processing error: {e}")
            response_text = "I encountered an issue processing your request."
        except Exception as e:
            log_error(f"Unexpected error: {e}")
            response_text = "Something went wrong while generating my response."

        # Display and speak Chitti's response
        log_state("CHITTI")
        print(response_text)
        self.history.add_assistant_message(response_text)
        self.speak(response_text)

    def trigger_vision_snapshot(self):
        """Performs an instant camera snapshot and reports what Chitti sees."""
        if not self.vision:
            print("\n[VISION] Camera/Vision system is not available.", flush=True)
            return

        print("\n[VISION] Capturing camera frame & analyzing visual scene...", flush=True)
        result = self.vision.analyze_frame()
        print(f"\nVisual Perception Summary:\n{result.summary_text}", flush=True)
        if result.faces:
            print("Detected Faces:", flush=True)
            for f in result.faces:
                status = f"Registered ({f.name}, similarity: {f.similarity:.2f})" if f.is_known else "Unknown Person"
                print(f"  - {status} at bbox {f.bbox}", flush=True)
        if result.object_counts:
            print("Detected Objects:", flush=True)
            for obj, cnt in result.object_counts.items():
                print(f"  - {obj}: {cnt}", flush=True)

        self.speak(result.summary_text)

    def trigger_face_registration_interactive(self):
        """Interactively registers a person's face using the camera."""
        if not self.vision:
            print("\n[VISION] Camera/Vision system is not available.", flush=True)
            return

        name = input("\nEnter the person's name to register: ").strip()
        if not name:
            print("Registration cancelled: Name cannot be empty.", flush=True)
            return

        print(f"\nPlease look directly at the camera. Collecting face samples for '{name}'...", flush=True)
        success, msg = self.vision.register_person(name, num_samples=3)
        print(f"\n{msg}", flush=True)
        self.speak(msg)

    def trigger_list_registered_faces(self):
        """Lists all registered face identities in the database."""
        if not self.vision:
            print("\n[VISION] Face database is not available.", flush=True)
            return
        names = self.vision.list_registered_people()
        print("\n==========================================", flush=True)
        print(" REGISTERED PEOPLE IN FACE DATABASE", flush=True)
        print("==========================================", flush=True)
        if names:
            for i, n in enumerate(names, 1):
                print(f"  {i}. {n}", flush=True)
        else:
            print("  No registered faces found in database.", flush=True)
        print("==========================================\n", flush=True)

    def listen_and_transcribe(self) -> str:
        """Captures voice from microphone and transcribes via Whisper STT."""
        if not self.mic:
            log_error("Microphone is not initialized.")
            return ""

        if not self.stt:
            log_error("Speech-to-text engine is not initialized.")
            return ""

        try:
            log_state("LISTENING", "Speak into your microphone now...")
            audio_data = self.mic.record_speech()

            if audio_data.size == 0:
                log_debug("No audio recorded.")
                return ""

            log_state("TRANSCRIBING", "Processing speech...")
            transcribed_text = self.stt.transcribe(audio_data)
            return transcribed_text
        except AudioCaptureError as e:
            log_error(f"Microphone capture error: {e}")
            return ""
        except STTError as e:
            log_error(f"Speech recognition error: {e}")
            return ""
        except Exception as e:
            log_error(f"Voice processing failed: {e}")
            return ""

    def run(self):
        """Main interactive loop for Chitti."""
        self.initialize()

        print("------------------------------------------------------------------")
        print(" Controls:")
        print("   [ENTER]      : Speak to Chitti via Microphone")
        print("   [T] + ENTER  : Type a text message (Keyboard fallback)")
        print("   [V] + ENTER  : Instant Camera Vision Snapshot")
        print("   [R] + ENTER  : Register a new Person's Face")
        print("   [L] + ENTER  : List all Registered People in Face Database")
        print("   [M] + ENTER  : View all stored persistent long-term memories")
        print("   [C] + ENTER  : Clear short-term session conversation history")
        print("   [Q] + ENTER  : Quit Chitti")
        print("------------------------------------------------------------------\n")

        while True:
            try:
                user_choice = input("\n[Ready] Press ENTER to talk (or T=type, V=vision, R=register, L=list faces, M=memory, C=clear, Q=quit): ").strip().lower()

                if user_choice in ("q", "quit", "exit"):
                    log_chitti("Shutting down Chitti. Goodbye!")
                    self.speak("Goodbye!")
                    if self.vision and self.vision.camera:
                        self.vision.camera.release()
                    break

                if user_choice in ("v", "vision", "see"):
                    self.trigger_vision_snapshot()
                    continue

                if user_choice in ("r", "register", "register face"):
                    self.trigger_face_registration_interactive()
                    continue

                if user_choice in ("l", "list", "faces", "people"):
                    self.trigger_list_registered_faces()
                    continue

                if user_choice in ("c", "clear"):
                    self.history.clear()
                    log_chitti("Session conversation history cleared.")
                    print("Short-term conversation history reset.")
                    continue

                if user_choice in ("m", "memory", "memories"):
                    if self.memory:
                        mems = self.memory.list_memories()
                        if mems:
                            print("\n--- Stored Long-Term Memories ---")
                            for m in mems:
                                print(f"[{m.memory_type.upper()}] (Importance: {m.importance}/5) {m.content}")
                            print(f"Total: {len(mems)} memories.")
                        else:
                            print("\nNo long-term memories stored yet.")
                    else:
                        print("\nMemory system is not active.")
                    continue

                if user_choice in ("t", "type"):
                    typed_text = input("\nYou (typed): ").strip()
                    if typed_text:
                        self.process_user_input(typed_text)
                    continue

                # Default: Push-to-talk microphone capture
                voice_text = self.listen_and_transcribe()
                if voice_text:
                    self.process_user_input(voice_text)
                else:
                    log_chitti("No speech was detected or understood.")
                    print("Hint: Check microphone volume or press 'T' to type.")

            except KeyboardInterrupt:
                print("\n")
                log_chitti("Session interrupted by user. Exiting cleanly...")
                if self.vision and self.vision.camera:
                    self.vision.camera.release()
                break
            except Exception as e:
                log_error(f"Unhandled error in main loop: {e}")


def main():
    """Application entry point."""
    controller = ChittiController()
    controller.run()


if __name__ == "__main__":
    main()
