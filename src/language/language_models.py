"""
Chitti Language Models and Data Structures.
Defines enums, dataclasses, and representations for language detection,
semantic normalization, intent representation, and translation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class LanguageCode(str, Enum):
    """Supported language codes."""
    ENGLISH = "en"
    HINDI = "hi"
    HINGLISH = "hinglish"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class IntentCategory(str, Enum):
    """Categorized user intent classes."""
    CONVERSATION = "conversation"
    LAPTOP_CONTROL = "laptop_control"
    VISION_QUERY = "vision_query"
    MEMORY_COMMAND = "memory_command"
    TRANSLATION = "translation"
    LANGUAGE_SWITCH = "language_switch"
    CLARIFICATION = "clarification"


@dataclass
class LanguageDetectionResult:
    """Result of language identification analysis."""
    language: str  # "en", "hi", "hinglish", "mixed", "unknown"
    confidence: float
    script: str = "latin"  # "latin", "devanagari", "mixed"
    is_roman_hindi: bool = False
    detected_markers: List[str] = field(default_factory=list)
    english_ratio: float = 0.0
    hindi_ratio: float = 0.0


@dataclass
class NormalizedAction:
    """A discrete executable action item within a user request."""
    action: str  # e.g., "open_application", "open_website", "search_web", "delete_file"
    target: Optional[str] = None  # e.g., "Chrome", "YouTube", "VS Code"
    parameters: Dict[str, Any] = field(default_factory=dict)
    is_negated: bool = False  # e.g., "delete mat karna" -> is_negated=True


@dataclass
class NormalizedIntent:
    """
    Internal canonical semantic representation of a user query across all languages.
    Translates user intention into structured meaning before reasoning or tool execution.
    """
    raw_text: str
    detected_language: str
    normalized_text: str  # Canonical semantic English representation for internal reasoning
    intent_category: str  # IntentCategory value
    actions: List[NormalizedAction] = field(default_factory=list)
    is_negated: bool = False  # Crucial: prevents destructive actions on "mat / nahi / don't"
    confidence: float = 1.0
    requires_confirmation: bool = False
    parameters: Dict[str, Any] = field(default_factory=dict)
    target_response_language: Optional[str] = None  # Desired language for the assistant reply
    clarification_prompt: Optional[str] = None


@dataclass
class TranslationRequest:
    """Request specification for translating text."""
    source_text: str
    target_language: str  # "en", "hi", "hinglish"
    source_language: Optional[str] = None
    preserved_terms: List[str] = field(default_factory=list)


@dataclass
class TranslationResult:
    """Result of a translation operation."""
    translated_text: str
    source_language: str
    target_language: str
    preserved_terms: List[str] = field(default_factory=list)
    confidence: float = 1.0
