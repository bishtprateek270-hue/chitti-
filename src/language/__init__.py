"""
Chitti Language Module.
Provides language detection, phonetic and STT normalization, semantic intent canonicalization,
and technical term-preserving translation.
"""

from src.language.language_models import (
    LanguageCode,
    IntentCategory,
    LanguageDetectionResult,
    NormalizedAction,
    NormalizedIntent,
    TranslationRequest,
    TranslationResult,
)
from src.language.detector import LanguageDetector
from src.language.normalizer import LanguageNormalizer
from src.language.translator import Translator
from src.language.prompts import (
    MULTILINGUAL_PERSONA_GUIDELINES,
    TRANSLATION_SYSTEM_PROMPT,
)

__all__ = [
    "LanguageCode",
    "IntentCategory",
    "LanguageDetectionResult",
    "NormalizedAction",
    "NormalizedIntent",
    "TranslationRequest",
    "TranslationResult",
    "LanguageDetector",
    "LanguageNormalizer",
    "Translator",
    "MULTILINGUAL_PERSONA_GUIDELINES",
    "TRANSLATION_SYSTEM_PROMPT",
]
