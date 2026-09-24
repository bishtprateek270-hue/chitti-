"""
Chitti Memory Extractor Module.
Extracts facts, detects explicit natural language memory commands, filters sensitive data, and infers memory types.
"""

import re
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

from src.utils.logging import log_debug, log_warning


@dataclass
class ExtractedMemoryCommand:
    """Represents a recognized memory command or extracted memory fact."""
    action: str  # 'remember', 'recall', 'forget', 'forget_all', 'none'
    content: Optional[str] = None
    memory_type: str = "fact"  # fact, preference, project, instruction, personal, context
    importance: int = 3
    target_category: Optional[str] = None
    is_sensitive: bool = False
    rejection_reason: Optional[str] = None


# Patterns for explicit commands
REMEMBER_PREFIXES = [
    r"^(?:chitti,?\s*)?(?:please\s+)?remember\s+that\s+(.*)$",
    r"^(?:chitti,?\s*)?(?:please\s+)?remember\s+my\s+(.*)$",
    r"^(?:chitti,?\s*)?(?:please\s+)?remember\s+(.*)$",
    r"^(?:chitti,?\s*)?(?:please\s+)?don'?t\s+forget\s+(?:that\s+)?(.*)$",
    r"^(?:chitti,?\s*)?(?:please\s+)?keep\s+in\s+mind\s+(?:that\s+)?(.*)$",
]

FORGET_PREFIXES = [
    r"^(?:chitti,?\s*)?(?:please\s+)?forget\s+that\s+(.*)$",
    r"^(?:chitti,?\s*)?(?:please\s+)?forget\s+about\s+(.*)$",
    r"^(?:chitti,?\s*)?(?:please\s+)?forget\s+my\s+(.*)$",
    r"^(?:chitti,?\s*)?(?:please\s+)?forget\s+the\s+memory\s+(?:about\s+)?(.*)$",
    r"^(?:chitti,?\s*)?(?:please\s+)?delete\s+(?:the\s+)?memory\s+(?:about\s+)?(.*)$",
    r"^(?:chitti,?\s*)?(?:please\s+)?forget\s+(.*)$",
]

FORGET_ALL_PATTERNS = [
    r"^(?:chitti,?\s*)?(?:please\s+)?forget\s+everything(?:\s+about\s+me)?$",
    r"^(?:chitti,?\s*)?(?:please\s+)?delete\s+all\s+memories$",
    r"^(?:chitti,?\s*)?(?:please\s+)?clear\s+all\s+memories$",
    r"^(?:chitti,?\s*)?(?:please\s+)?wipe\s+(?:all\s+)?(?:my\s+)?memories$",
]

RECALL_INSPECT_PATTERNS = [
    r"^(?:chitti,?\s*)?(?:show|tell)\s+me\s+what\s+you\s+remember(?:\s+about\s+(.*))?$",
    r"^(?:chitti,?\s*)?what\s+do\s+you\s+remember(?:\s+about\s+(.*))?$",
    r"^(?:chitti,?\s*)?list\s+(?:all\s+)?(?:my\s+)?memories(?:\s+about\s+(.*))?$",
    r"^(?:chitti,?\s*)?what\s+are\s+my\s+memories(?:\s+about\s+(.*))?$",
]

# Sensitive information patterns
SENSITIVE_PATTERNS = [
    r"(?i)\b(?:password|passwd|pwd)\s*(?:is|=|:)\s*\S+",
    r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token)\s*(?:is|=|:)\s*\S+",
    r"(?i)\b(?:sk-[a-zA-Z0-9_-]{20,})\b",
    r"(?i)\b(?:bearer\s+[a-zA-Z0-9_\-\.]{20,})\b",
    r"(?i)\b(?:private[_-]?key)\b",
]


class MemoryExtractor:
    """Analyzes text for memory instructions, classifications, and safety checks."""

    @staticmethod
    def is_sensitive(text: str) -> bool:
        """Returns True if the text appears to contain passwords, API keys, or secrets."""
        for pattern in SENSITIVE_PATTERNS:
            if re.search(pattern, text):
                return True
        return False

    @staticmethod
    def infer_memory_type(content: str) -> str:
        """Infers category based on keywords in extracted content."""
        lower = content.lower()
        if any(k in lower for k in ["project", "docforensics", "building", "working on", "app", "codebase", "repo"]):
            return "project"
        if any(k in lower for k in ["prefer", "favorite", "favourite", "like", "love", "dislike", "hate", "rather than", "instead of"]):
            return "preference"
        if any(k in lower for k in ["always", "never", "rule", "instruction", "must", "format"]):
            return "instruction"
        if any(k in lower for k in ["name is", "live in", "living in", "work as", "role", "age", "birthday", "email"]):
            return "personal"
        return "fact"

    @staticmethod
    def clean_remembered_content(raw_text: str) -> str:
        """Normalizes extracted statement for clean storage."""
        cleaned = raw_text.strip().rstrip(".!? \t\n")
        return cleaned

    def extract_command(self, user_text: str) -> ExtractedMemoryCommand:
        """
        Parses user text to determine if it is an explicit memory command.
        """
        raw = user_text.strip()
        cleaned = self.clean_remembered_content(raw)
        lower = cleaned.lower()

        # Check for sensitive data first
        if self.is_sensitive(raw):
            return ExtractedMemoryCommand(
                action="sensitive_rejected",
                is_sensitive=True,
                rejection_reason="For security reasons, Chitti does not store passwords, API keys, or private tokens in persistent memory."
            )

        # 1. Check for FORGET ALL / CLEAR ALL
        for pattern in FORGET_ALL_PATTERNS:
            if re.match(pattern, lower):
                return ExtractedMemoryCommand(action="forget_all")

        # 2. Check for RECALL / INSPECT
        for pattern in RECALL_INSPECT_PATTERNS:
            match = re.match(pattern, lower)
            if match:
                cat = match.group(1).strip() if match.group(1) else None
                return ExtractedMemoryCommand(action="recall", target_category=cat)

        # 3. Check for FORGET specific
        for pattern in FORGET_PREFIXES:
            match = re.match(pattern, cleaned, re.IGNORECASE)
            if match:
                target = match.group(1).strip().rstrip(".!? \t\n")
                if target:
                    return ExtractedMemoryCommand(action="forget", content=target)

        # 4. Check for REMEMBER explicit
        for pattern in REMEMBER_PREFIXES:
            match = re.match(pattern, cleaned, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip().rstrip(".!? \t\n")
                if extracted:
                    mem_type = self.infer_memory_type(extracted)
                    # Explicit remember commands have high importance
                    importance = 4
                    if any(k in extracted.lower() for k in ["critical", "important", "main", "primary"]):
                        importance = 5
                    return ExtractedMemoryCommand(
                        action="remember",
                        content=extracted,
                        memory_type=mem_type,
                        importance=importance,
                    )

        # No explicit memory command found
        return ExtractedMemoryCommand(action="none")
