"""
Chitti - Personal Multimodal AI Desktop Companion Robot
PHASE 1: Voice AI Brain  +  PHASE 2: Long-Term Memory  +  PHASE 3: Computer Vision
PHASE 4 & 4A: Multilingual Understanding, Memory Reliability & Response Quality
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
from src.brain.validator import ResponseValidator
from src.audio.microphone import MicrophoneManager, AudioCaptureError
from src.audio.stt import get_stt, STTError
from src.audio.tts import get_tts, TTSError
from src.memory.manager import MemoryManager
from src.vision.vision_manager import VisionManager
from src.language.detector import LanguageDetector
from src.language.normalizer import LanguageNormalizer
from src.language.translator import Translator
from src.language.language_models import IntentCategory


BANNER = r"""
+------------------------------------------------------------------+
|                                                                  |
|                          C H I T T I                             |
|             Personal AI Desktop Companion Robot                  |
|    Phase 1: Voice  +  Phase 2: Memory  +  Phase 3: Vision        |
|    Phase 4: Multilingual  +  Phase 4A: Brain Reliability         |
|                                                                  |
+------------------------------------------------------------------+
"""


class ChittiController:
    """Main application controller for Chitti."""

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
        self.language_detector = LanguageDetector(confidence_threshold=self.config.language.confidence_threshold)
        self.language_normalizer = LanguageNormalizer(self.language_detector)
        self.translator = None
        self.session_response_language = None

    def initialize(self):
        """Initializes all hardware and AI subsystems."""
        print(BANNER)
        log_chitti("Starting Chitti Multimodal & Multilingual Systems...")

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

        # 3. Initialize STT (Whisper) with eager preloading
        log_chitti(f"Loading speech recognition model ({self.config.stt.model}) on {self.config.stt.device.upper()}...")
        try:
            self.stt = get_stt(self.config.stt)
            if hasattr(self.stt, "preload"):
                self.stt.preload()
            log_chitti("STT module initialized and preloaded in memory.")
        except Exception as e:
            log_error(f"STT initialization failed: {e}")
            self.stt = None

        # 4. Initialize Long-Term Memory (Phase 2 & 4A)
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

        # 7. Initialize Translator (Phase 4)
        self.translator = Translator(llm=self.llm)
        log_chitti("Multilingual intelligence system ready (English, Hindi, Hinglish).")

        log_chitti("System initialization complete. Ready for interaction!\n")

    def speak(self, text: str):
        """Speaks the response using TTS if available."""
        if self.tts and text:
            try:
                self.tts.speak(text, wait=True)
            except Exception as e:
                log_warning(f"Speech playback failed: {e}")

    def process_user_input(self, user_text: str):
        """Processes user input through Multilingual/Vision/Memory/LLM, records history, and speaks output."""
        if not user_text or not user_text.strip():
            log_chitti("I didn't catch that. Could you repeat?")
            self.speak("I didn't catch that. Could you repeat?")
            return

        print(f"\nYou: {user_text}")

        # 0. Multilingual Understanding & Intent Normalization (Phase 4)
        parsed_intent = self.language_normalizer.parse_intent(user_text)
        detected_lang = parsed_intent.detected_language
        log_debug(
            f"[LANGUAGE] Detected: {detected_lang}, Intent: {parsed_intent.intent_category}, "
            f"Normalized: '{parsed_intent.normalized_text}', Negated: {parsed_intent.is_negated}"
        )

        active_lang = self.session_response_language or detected_lang or "en"

        # 1. Check for Explicit Language Switch Command
        if parsed_intent.intent_category == IntentCategory.LANGUAGE_SWITCH.value:
            self.session_response_language = parsed_intent.target_response_language
            lang_confirmations = {
                "en": "Understood. I will respond in English from now on.",
                "hi": "ज़रूर, अब से मैं हिंदी में उत्तर दूँगा।",
                "hinglish": "Bilkul, ab se main Hinglish mein answer karunga.",
            }
            resp = lang_confirmations.get(self.session_response_language, f"Switched response language to {self.session_response_language}.")
            self.history.add_user_message(user_text)
            self.history.add_assistant_message(resp)
            if self.memory is not None:
                self.memory.auto_capture_interaction(user_text, resp, lang=active_lang)
            log_state("CHITTI")
            print(resp)
            self.speak(resp)
            return

        # 2. Check for Dedicated Translation Request
        if parsed_intent.intent_category == IntentCategory.TRANSLATION.value:
            target_lang = parsed_intent.parameters.get("target_language", "en")
            text_to_translate = parsed_intent.parameters.get("text", "")
            if not text_to_translate:
                text_to_translate = user_text

            log_state("TRANSLATING", f"Translating to {target_lang}...")
            trans_res = self.translator.translate(text_to_translate, target_lang, source_language=detected_lang)
            self.history.add_user_message(user_text)
            self.history.add_assistant_message(trans_res.translated_text)
            if self.memory is not None:
                self.memory.auto_capture_interaction(user_text, trans_res.translated_text, lang=active_lang)
            log_state("CHITTI")
            print(trans_res.translated_text)
            self.speak(trans_res.translated_text)
            return

        # 3. Check for Face Management Commands (e.g. "Who is registered?", "Remove Prateek from face recognition")
        if self.vision is not None:
            try:
                face_cmd_result = self.vision.handle_face_commands(user_text)
                if face_cmd_result is None and parsed_intent.normalized_text != user_text:
                    face_cmd_result = self.vision.handle_face_commands(parsed_intent.normalized_text)

                if face_cmd_result is not None:
                    action_tag, response_text = face_cmd_result
                    self.history.add_user_message(user_text)
                    self.history.add_assistant_message(response_text)
                    if self.memory is not None:
                        self.memory.auto_capture_interaction(user_text, response_text, lang=active_lang)

                    log_state("CHITTI")
                    print(response_text)
                    self.speak(response_text)
                    return
            except Exception as e:
                log_warning(f"Face command processing failed: {e}")

        # 4. Check for Direct Identity / Creator / Occupation Query (Phase 4A)
        if self.memory is not None:
            try:
                direct_id_resp = self.memory.resolve_identity_query(user_text, lang=active_lang)
                if (direct_id_resp is None or not isinstance(direct_id_resp, str)) and parsed_intent.normalized_text != user_text:
                    direct_id_resp = self.memory.resolve_identity_query(parsed_intent.normalized_text, lang=active_lang)

                if isinstance(direct_id_resp, str) and direct_id_resp.strip():
                    self.history.add_user_message(user_text)
                    self.history.add_assistant_message(direct_id_resp)
                    self.memory.auto_capture_interaction(user_text, direct_id_resp, lang=active_lang)
                    log_state("CHITTI")
                    print(direct_id_resp)
                    self.speak(direct_id_resp)
                    return
            except Exception as e:
                log_warning(f"Identity resolution failed: {e}")

        # 5. Check for Explicit Memory Command & Fact Statement (multilingual: "yaad rakhna ki...", "remember that...", "my name is...", "forget...")
        if self.memory is not None:
            try:
                mem_result = self.memory.handle_interaction(user_text, lang=active_lang)
                if mem_result is None and parsed_intent.normalized_text != user_text:
                    mem_result = self.memory.handle_interaction(parsed_intent.normalized_text, lang=active_lang)

                if mem_result is not None and isinstance(mem_result, tuple):
                    action_tag, response_text = mem_result
                    if isinstance(response_text, str) and response_text.strip():
                        self.history.add_user_message(user_text)
                        self.history.add_assistant_message(response_text)
                        self.memory.auto_capture_interaction(user_text, response_text, lang=active_lang)

                        log_state("CHITTI")
                        print(response_text)
                        self.speak(response_text)
                        return
            except Exception as e:
                log_warning(f"Memory command processing failed: {e}")

        # Add to short-term session history
        self.history.add_user_message(user_text)

        # 6. Multilingual Vision Perception Trigger
        vision_context = None
        recognized_names_seen = []

        is_vis = (
            parsed_intent.intent_category == IntentCategory.VISION_QUERY.value
            or (self.vision is not None and self.vision.is_vision_query(user_text))
            or (self.vision is not None and self.vision.is_vision_query(parsed_intent.normalized_text))
        )

        if self.vision is not None and is_vis:
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

        # 7. Retrieve relevant long-term memories (cross-lingual semantic search)
        relevant_memories = []
        if self.memory is not None:
            try:
                # Query memories using both raw text and normalized English text
                relevant_memories = self.memory.recall(user_text, top_k=self.config.memory.top_k)
                if parsed_intent.normalized_text and parsed_intent.normalized_text != user_text:
                    norm_mems = self.memory.recall(parsed_intent.normalized_text, top_k=self.config.memory.top_k)
                    for nm in norm_mems:
                        if nm.id not in [m.id for m in relevant_memories]:
                            relevant_memories.append(nm)

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

        # 8. Configure Response Language Guidance for LLM
        lang_name_map = {
            "en": "English",
            "hi": "Hindi (Devanagari script)",
            "hinglish": "natural, conversational Hinglish (Roman Hindi)",
        }
        lang_target_name = lang_name_map.get(active_lang, "English")

        lang_guidance = (
            f"\n\n[RESPONSE LANGUAGE DIRECTIVE: Respond naturally in {lang_target_name}. "
            f"Keep programming code, technical terms (e.g. Python, GPU, RAM, Docker, VS Code), "
            f"file paths, and proper names in English/Roman text. Never translate technical terms awkwardly. "
            f"Never output placeholders like [Creator's Name]. If you don't know something, admit it directly.]"
        )
        self.history.set_language_instruction(lang_guidance)

        # 9. Generate LLM response
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

        # 10. Post-processing Response Validation & Placeholder Sanitization (Phase 4A)
        user_name = self.memory.get_user_name() if self.memory else None
        creator_name = self.memory.get_creator_name() if self.memory else None
        sanitized_response = ResponseValidator.sanitize(
            response_text,
            user_name=user_name,
            creator_name=creator_name,
            target_lang=active_lang,
        )

        # Display and speak Chitti's sanitized response
        log_state("CHITTI")
        print(sanitized_response)
        self.history.add_assistant_message(sanitized_response)
        if self.memory is not None:
            self.memory.auto_capture_interaction(user_text, sanitized_response, lang=active_lang)
        self.speak(sanitized_response)

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
                print(f"  - {f.identity} (confidence: {f.confidence:.2f}, bbox: {f.bbox})", flush=True)
        if result.objects:
            print("Detected Objects:", flush=True)
            for o in result.objects:
                print(f"  - {o.label} ({o.confidence*100:.1f}%)", flush=True)

    def trigger_face_registration_interactive(self):
        """Interactive CLI wizard for registering a new person's face."""
        if not self.vision:
            print("\n[VISION] Camera/Vision system is not available.", flush=True)
            return

        print("\n==========================================", flush=True)
        print(" INTERACTIVE FACE REGISTRATION WIZARD", flush=True)
        print("==========================================", flush=True)
        name = input("Enter the person's full name: ").strip()
        if not name:
            print("[ERROR] Name cannot be empty.", flush=True)
            return

        print(f"\nStarting face registration for '{name}'...", flush=True)
        print("Please look directly into the camera.", flush=True)
        success, message = self.vision.register_face_interactive(name, num_samples=5)
        print(f"\nResult: {message}\n", flush=True)
        if success:
            self.speak(f"Face registration complete for {name}.")

    def trigger_list_registered_faces(self):
        """Prints all registered people in the face database."""
        if not self.vision or not self.vision.face_db:
            print("\n[VISION] Face database is not available.", flush=True)
            return

        records = self.vision.face_db.list_all_faces()
        print("\n==========================================", flush=True)
        print(" REGISTERED PEOPLE IN FACE DATABASE", flush=True)
        print("==========================================", flush=True)
        if records:
            for idx, r in enumerate(records, 1):
                print(f"  {idx}. {r.name} (samples: {r.sample_count}, registered: {r.created_at[:10]})", flush=True)
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
