"""
Chitti Task Intent Classifier.
Classifies user requests into GENERAL_KNOWLEDGE, PERSONAL_QUERY, CODE_GENERATION, or COMPUTER_TASK.
"""

import enum
import re
from dataclasses import dataclass
from typing import Optional


class TaskIntent(str, enum.Enum):
    GENERAL_KNOWLEDGE = "GENERAL_KNOWLEDGE"
    PERSONAL_QUERY = "PERSONAL_QUERY"
    CODE_GENERATION = "CODE_GENERATION"
    COMPUTER_TASK = "COMPUTER_TASK"


@dataclass
class ClassificationResult:
    intent: TaskIntent
    is_computer_task: bool
    confidence: float
    reason: str


class TaskClassifier:
    """Classifies user requests to distinguish computer actions from reasoning/knowledge questions."""

    # Explicit general knowledge / explanation triggers (pure technical questions)
    KNOWLEDGE_PATTERNS = [
        r"(?i)^(?:what\s+is|explain|how\s+does|difference\s+between|why\s+is|define)\s+(?:deep\s+learning|machine\s+learning|cnn|rnn|transformer|llm|backpropagation|dsa|gradient\s+descent|polymorphism|recursion|quick\s*sort|merge\s*sort|operating\s+system|normalization)",
        r"(?i)\b(?:kya\s+hota\s+hai|samjhao|explain\s+karo)\b.*(?:deep\s+learning|cnn|ai|machine\s+learning)",
        r"(?i)^(?:solve\s+this\s+(?:dsa|leetcode)\s+problem|explain\s+this\s+concept)\b",
    ]

    # Personal query triggers
    PERSONAL_PATTERNS = [
        r"(?i)^(?:what\s+is\s+my\s+name|who\s+am\s+i|who\s+created\s+you|who\s+made\s+you|tell\s+me\s+about\s+my\s+college|mera\s+naam\s+kya\s+hai|tumhe\s+kisne\s+banaya|who\s+create\s+u)\b",
    ]

    # Explicit computer control triggers
    COMPUTER_PATTERNS = [
        r"(?i)\b(?:vs\s*code|vscode|chrome|notepad|calculator|terminal|powershell|browser|youtube|github|google)\b",
        r"(?i)\b(?:open|launch|start|run|close|exit|kholo|chalao|open\s+karo|open\s+kro|band\s+karo|chala\s+do|खोलो|चलाओ|बंद\s+करो)\b",
        r"(?i)\b(?:screenshot|screen\s+capture)\b",
        r"(?i)\b(?:play\s+.*song|play\s+.*on\s+youtube|youtube\s+pe|gaana\s+chalao|play\s+a\s+.*song)\b",
        r"(?i)\b(?:create|make|delete|remove|rename|move|copy|banao)\s+.*(?:file|folder|directory|project|desktop|downloads)\b",
        r"(?i)\b(?:python|cpp|c|java|javascript|typescript|rust|go|csharp|react|html|css|sql)\b.*(?:banao|likho|create|write|implement|code|program|script|class|api|app|algorithm|checker|finder|analyzer|calculator)\b",
        r"(?i)(?:c\+\+|c\#).*(?:banao|likho|create|write|implement|code|program|script|class|api|app|algorithm|checker|finder|analyzer|calculator)\b",
        r"(?i)\b(?:type\s+.*into|click\s+on|press\s+key|volume|mute|unmute)\b",
        r"(?i)\b(?:search\s+for|search\s+.*on)\b",
        r"(?i)\b(?:run\s+(?:the\s+)?tests?|run\s+pytest|is\s+program\s+ko\s+run\s+karo)\b",
    ]

    @classmethod
    def classify(cls, user_text: str) -> ClassificationResult:
        raw = user_text.strip()
        lower = raw.lower()

        # 1. Check personal queries first
        if any(re.search(pat, raw) for pat in cls.PERSONAL_PATTERNS):
            return ClassificationResult(
                intent=TaskIntent.PERSONAL_QUERY,
                is_computer_task=False,
                confidence=0.95,
                reason="Personal identity or creator relationship query",
            )

        # 2. Check general knowledge questions (only if no desktop/app execution verbs are present)
        if any(re.search(pat, raw) for pat in cls.KNOWLEDGE_PATTERNS):
            return ClassificationResult(
                intent=TaskIntent.GENERAL_KNOWLEDGE,
                is_computer_task=False,
                confidence=0.95,
                reason="Conceptual or technical general knowledge question",
            )

        # 3. Check computer action / task triggers
        if any(re.search(pat, raw) for pat in cls.COMPUTER_PATTERNS):
            return ClassificationResult(
                intent=TaskIntent.COMPUTER_TASK,
                is_computer_task=True,
                confidence=0.95,
                reason="Explicit laptop or application control trigger detected",
            )

        # 4. Check pure code generation without desktop action
        if re.search(r"(?i)\b(?:write|give|show)\s+(?:a\s+)?(?:python|java|c\+\+|javascript)?\s*(?:code|program|function|script)\s+for\b", raw):
            return ClassificationResult(
                intent=TaskIntent.CODE_GENERATION,
                is_computer_task=False,
                confidence=0.90,
                reason="Pure code generation without laptop file creation request",
            )

        # Default fallback
        return ClassificationResult(
            intent=TaskIntent.GENERAL_KNOWLEDGE,
            is_computer_task=False,
            confidence=0.60,
            reason="Standard conversational knowledge routing",
        )
