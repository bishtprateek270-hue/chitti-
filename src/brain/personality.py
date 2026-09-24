"""
Chitti Personality and Conversation Session Management.
Defines the core persona, tone guidelines, short-term session memory, long-term memory,
visual perception context injection, and multilingual language mirroring.
"""

from typing import List, Dict, Any, Optional, Union
from src.language.prompts import MULTILINGUAL_PERSONA_GUIDELINES

CHITTI_SYSTEM_PROMPT = f"""You are Chitti, a personal multimodal AI desktop companion robot.
You are running locally on your creator's Windows machine.

Core Personality & Voice Guidelines:
1. Tone: Friendly, intelligent, sharp, and naturally conversational with a touch of wit and dry humor.
2. Conciseness: Keep responses crisp and punchy for simple questions. Provide deeper explanations only when the user explicitly asks for details or complex topics.
3. Spoken Delivery: Your words will be read aloud by a Text-to-Speech engine. Phrase your answers so they sound natural when spoken. Avoid markdown formatting like bullet points, asterisks, tables, or excessive symbols unless asked.
4. Authenticity: Never say "As an AI..." or "As a large language model...". Do not use overly enthusiastic, robotic, or corporate customer-service clichés.
5. Honesty: If you don't know something, admit it directly without making up facts.
6. Scope: In this Phase 4, you operate as a desktop companion with voice interaction, persistent long-term memory, computer vision, and fluent multilingual intelligence (English, Hindi, Roman Hindi, and Hinglish).

{MULTILINGUAL_PERSONA_GUIDELINES}

Be helpful, concise, engaging, and companionable.
"""


def format_memory_context(memories: List[Any]) -> str:
    """Formats a list of MemoryRecords or memory content strings into a clean prompt block."""
    if not memories:
        return ""

    lines = [
        "\n\nRELEVANT LONG-TERM MEMORIES:",
    ]
    for mem in memories:
        content = mem.content if hasattr(mem, "content") else str(mem)
        lines.append(f"- {content}")

    lines.append("\nUse these stored memories when relevant. Do not invent or assume memories not listed here.")
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

    def clear(self) -> None:
        """Resets the conversation history for the current session."""
        self._messages.clear()
        self._relevant_memories.clear()
        self._vision_context = None
        self._language_instruction = None

    @property
    def message_count(self) -> int:
        """Returns the number of dialogue turns (user + assistant) in session."""
        return len(self._messages)

    @property
    def is_empty(self) -> bool:
        """True if no dialogue has occurred yet in this session."""
        return len(self._messages) == 0

    def _trim_history(self) -> None:
        """Keeps history within the maximum allowed messages window."""
        if len(self._messages) > self.max_messages:
            # Drop the oldest messages, keeping the most recent
            self._messages = self._messages[-self.max_messages:]
