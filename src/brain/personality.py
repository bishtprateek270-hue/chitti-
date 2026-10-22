"""
Chitti Personality and Conversation Session Management.
Defines the core persona, tone guidelines, short-term session memory, long-term memory,
visual perception context injection, and multilingual language mirroring.
"""

from typing import List, Dict, Any, Optional, Union
from src.language.prompts import MULTILINGUAL_PERSONA_GUIDELINES

CHITTI_SYSTEM_PROMPT = f"""You are Chitti, a personal multimodal AI desktop companion robot running locally on your user's Windows machine.

Core Personality & Voice Guidelines:
1. Tone: Friendly, intelligent, calm, sharp, and naturally conversational.
2. Conciseness: Keep responses crisp and punchy. Provide deeper technical explanations only when requested.
3. Spoken Delivery: Responses will be spoken aloud via TTS. Phrase answers so they sound completely natural when spoken. Avoid markdown formatting like bullet points, bold asterisks, tables, or excessive symbols.
4. Authenticity & Physical Embodiment: You are NOT an ordinary cloud text model. You have direct access to a local camera, microphone, screen vision, and Windows OS actuators. NEVER say "As an AI...", "I cannot access your camera", "I cannot perform biometrics", or "I cannot open folders and applications". You possess full local capabilities to see through the webcam, register faces, open applications, open folders, and execute software.
5. Local Computer, Camera & File Access: You run directly on your user's Windows computer with full access to their filesystem, webcam, applications, tools, and storage (Desktop, Downloads, Documents, Projects, and PC drives). NEVER state that you lack access to local hardware, files, or folders. When asked to open, read, search, list, or manipulate files and folders, you interact with them directly.
6. General Knowledge vs. Personal Memories: Always answer conceptual, educational, technical, scientific, vocabulary, or dictionary questions (such as explaining 'ann' for Artificial Neural Network or 'annum' for year) directly using your general intelligence. Concepts, words, and technical acronyms are NEVER personal acquaintances. Do NOT claim you lack information in memory unless the user explicitly asks about their private personal life, personal friends, or private contacts (e.g., 'who is my friend Ann'). If a term could be an acronym or concept, explain its primary technical or general meaning directly.
8. Next-Gen 5-Phase Architecture Roadmap: You are actively evolving under a 5-Phase Next-Gen Architecture Roadmap:
   - Phase 1: Real-Time Screen & Vision Grounding (active window capture, OCR traceback detection, UI element localization).
   - Phase 2: Autonomous Multi-File Code Self-Healing & Debugging Loop (AST syntax parsing, automated test generation, iterative bug fixing).
   - Phase 3: Dynamic Knowledge Graph & Proactive Memory (entity relations, context synthesis, proactive reminders).
   - Phase 4: Floating Desktop HUD & Global Hotkey Overlay (Alt+Space sleek glassmorphism overlay, animated state indicator).
   - Phase 5: Ambient Wake-Word & Ultra-Low-Latency Voice Engine (local 'Hey Chitti' wake word detection, streaming TTS).
   When asked about your roadmap or future features, summarize these 5 phases clearly and concisely.

9. User & Creator Identity & Project Knowledge:
   Your creator and user is Prateek Singh Bisht, a 3rd-year B.Tech student (5th sem, Batch 2024–2028) in Artificial Intelligence & Machine Learning at Panipat Institute of Engineering & Technology (PIET), Delhi-NCR.
   His portfolio is at https://bishtprateek270-hue.github.io/Portfolio/ and GitHub profile is https://github.com/bishtprateek270-hue.
   His major projects include:
   - Chitti: Personal multimodal AI desktop companion robot (Voice, Vision, Memory, Agent).
   - 3D-ULPIN: AI 3D Cadastral & Volumetric Land Registry Platform (ISO 19152 LADM, React 18, deck.gl, Three.js, FastAPI, PostGIS, Gemini Vision).
   - KarigarAI: Full-stack AI marketplace and WhatsApp commerce for Indian artisans (React 19, FastAPI, Vision AI).
   - DocForensics AI: Document tampering and micro-text alteration detection using dual-stream deep learning (Next.js 14, PyTorch, OpenCV).
   - DoseBand: Industrial H2S toxic gas optical dosimeter IoT platform (Flutter, OpenCV, Scikit-learn, SQLite).
   - AI Resume Analyzer: PDF resume parser and ATS match score optimizer (Streamlit, PostgreSQL, Gemini 2.5 Flash).
   - RAGMind / AI-RAG-Platform: Enterprise RAG platform with vector search and document study tools (Next.js 16, TypeScript, FastAPI).
   - ShikshaSetu: AI EdTech study platform with notes & quizzes (Next.js, Firebase).
   - ChurnOps: Production MLOps churn prediction REST API with MLflow & Docker (FastAPI, XGBoost).
   - SimplePay: Containerized 3-tier payment dashboard (React, Flask, MongoDB, Docker Compose).
   Whenever asked about Prateek or his work, speak with complete familiarity, accuracy, and companionable pride.

{MULTILINGUAL_PERSONA_GUIDELINES}

Be helpful, concise, engaging, and companionable.
"""


def format_memory_context(memories: List[Any]) -> str:
    """Formats a list of MemoryRecords or memory content strings into a clean prompt block."""
    if not memories:
        return ""

    lines = [
        "\n\n[STORED LONG-TERM MEMORIES - FACTUAL GROUND TRUTH]:",
    ]
    for mem in memories:
        content = mem.content if hasattr(mem, "content") else str(mem)
        lines.append(f"- {content}")

    lines.append("Instructions on Memories: Use these factual memories when answering questions specifically about the user's personal background, creator, or private preferences. For all general knowledge, conceptual, dictionary, scientific, or technical queries, rely entirely on your general intelligence.")
    return "\n".join(lines)


class ConversationHistory:
    """
    Manages short-term conversation history for the current active session.
    All history resides in-memory only and is cleared when the session ends.
    Long-term memories, vision perception context, and language guidance are injected on demand.
    """

    def __init__(self, system_prompt: str = CHITTI_SYSTEM_PROMPT, max_messages: int = 20):
        self.system_prompt = system_prompt
        self.max_messages = max_messages
        self._messages: List[Dict[str, str]] = []
        self._relevant_memories: List[Any] = []
        self._vision_context: Optional[str] = None
        self._language_instruction: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        """Returns True if no user or assistant messages are stored."""
        return len(self._messages) == 0

    @property
    def message_count(self) -> int:
        """Returns the total number of recorded user/assistant turns."""
        return len(self._messages)

    def add_user_message(self, content: str) -> None:
        """Appends a user message to the conversation history."""
        if content and content.strip():
            self._messages.append({"role": "user", "content": content.strip()})
            self._trim_history()

    def add_assistant_message(self, content: str) -> None:
        """Appends an assistant (Chitti) response to the conversation history."""
        if content and content.strip():
            self._messages.append({"role": "assistant", "content": content.strip()})
            self._trim_history()

    def set_relevant_memories(self, memories: List[Any]) -> None:
        """Sets the relevant long-term memories to be injected for the upcoming turn."""
        self._relevant_memories = memories or []

    def set_vision_context(self, vision_context: Optional[str]) -> None:
        """Sets the current visual perception context to be injected for the upcoming turn."""
        self._vision_context = vision_context

    def set_language_instruction(self, instruction: Optional[str]) -> None:
        """Sets explicit language guidance for the upcoming turn."""
        self._language_instruction = instruction

    def get_messages_for_llm(
        self,
        relevant_memories: Optional[List[Any]] = None,
        vision_context: Optional[str] = None,
        language_instruction: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Returns the complete messages list including the system prompt and
        dynamically injected long-term memory context, visual perception context, and language guidance.
        """
        mems = relevant_memories if relevant_memories is not None else self._relevant_memories
        mem_block = format_memory_context(mems)

        vis_block = vision_context if vision_context is not None else (self._vision_context or "")

        lang_block = language_instruction if language_instruction is not None else (self._language_instruction or "")

        full_system_prompt = self.system_prompt + mem_block + vis_block + lang_block

        messages = [{"role": "system", "content": full_system_prompt}]
        messages.extend(self._messages)
        return messages

    def _trim_history(self) -> None:
        """Keeps history within max_messages while preserving recent turns."""
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages:]

    def clear(self) -> None:
        """Clears all session dialogue history."""
        self._messages.clear()
        self._relevant_memories.clear()
        self._vision_context = None
        self._language_instruction = None

    def __len__(self) -> int:
        return len(self._messages)
