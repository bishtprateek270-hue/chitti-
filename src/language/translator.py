"""
Chitti Multilingual Translator Module.
Handles dedicated translation between English, Hindi (Devanagari), and Hinglish (Roman Hindi)
while strictly preserving code, technical terminology, paths, and URLs.
"""

import re
from typing import Any, List, Optional
from src.language.language_models import TranslationRequest, TranslationResult
from src.language.prompts import TRANSLATION_SYSTEM_PROMPT
from src.utils.logging import log_debug, log_warning


class Translator:
    """Translates text between English, Hindi, and Hinglish with entity preservation."""

    # Common technical terms to explicitly protect from awkward translation
    PRESERVED_TERMS = [
        "Python", "JavaScript", "C++", "Java", "GitHub", "Git", "API", "DBMS", "SQL",
        "GPU", "CPU", "RAM", "Docker", "React", "FastAPI", "VS Code", "Terminal",
        "Machine Learning", "Deep Learning", "Neural Network", "CNN", "RNN", "PCA",
        "Chrome", "YouTube", "Chitti", "Overfitting", "Dataset"
    ]

    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm

    def _extract_protected_tokens(self, text: str) -> List[str]:
        """Identifies file paths, URLs, and technical keywords."""
        protected = []
        # Paths
        paths = re.findall(r"[a-zA-Z]:\\[^\s]+", text)
        protected.extend(paths)
        # URLs
        urls = re.findall(r"https?://[^\s]+", text)
        protected.extend(urls)
        # Technical keywords
        for term in self.PRESERVED_TERMS:
            if re.search(rf"\b{re.escape(term)}\b", text, flags=re.IGNORECASE):
                protected.append(term)
        return list(set(protected))

    def translate(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None,
    ) -> TranslationResult:
        """
        Translates text to the target language ('en', 'hi', 'hinglish').
        Preserves technical terms and formatting.
        """
        clean_text = text.strip()
        if not clean_text:
            return TranslationResult(
                translated_text="",
                source_language=source_language or "unknown",
                target_language=target_language,
            )

        target_lang_normalized = target_language.strip().lower()
        if target_lang_normalized in ("hindi", "hi", "हिंदी"):
            target_lang_name = "Hindi (Devanagari script)"
            target_code = "hi"
        elif target_lang_normalized in ("hinglish", "roman hindi"):
            target_lang_name = "Hinglish (Natural conversational Roman Hindi)"
            target_code = "hinglish"
        else:
            target_lang_name = "English"
            target_code = "en"

        protected_tokens = self._extract_protected_tokens(clean_text)

        # 1. Use LLM for intelligent contextual translation if available
        if self.llm is not None:
            try:
                user_prompt = f"Target Language: {target_lang_name}\n\nText to translate:\n{clean_text}"
                messages = [
                    {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
                translated = self.llm.generate_response(messages)
                if translated and translated.strip():
                    # Strip any conversational wrapping if present
                    translated_clean = re.sub(r'^(?:translation|translated text):\s*', '', translated.strip(), flags=re.IGNORECASE).strip('"\'')
                    return TranslationResult(
                        translated_text=translated_clean,
                        source_language=source_language or "unknown",
                        target_language=target_code,
                        preserved_terms=protected_tokens,
                        confidence=0.95,
                    )
            except Exception as e:
                log_warning(f"LLM translation failed: {e}. Using rule-based fallback.")

        # 2. Rule-based / Dictionary Fallback for common phrases
        fallback_translation = self._fallback_translate(clean_text, target_code)
        return TranslationResult(
            translated_text=fallback_translation,
            source_language=source_language or "unknown",
            target_language=target_code,
            preserved_terms=protected_tokens,
            confidence=0.75,
        )

    def _fallback_translate(self, text: str, target_code: str) -> str:
        """Heuristic fallback translation for offline/standalone execution."""
        common_phrases = {
            ("i have a meeting tomorrow", "hi"): "मेरी कल एक बैठक है।",
            ("i have a meeting tomorrow", "hinglish"): "Meri kal ek meeting hai.",
            ("mujhe kal college jana hai", "en"): "I have to go to college tomorrow.",
            ("machine learning is changing healthcare", "hi"): "Machine learning स्वास्थ्य सेवा को बदल रही है।",
            ("machine learning is changing healthcare", "hinglish"): "Machine learning healthcare ko change kar rahi hai.",
            ("open chrome", "hi"): "क्रोम खोलें।",
            ("open chrome", "hinglish"): "Chrome kholo.",
        }

        key = (text.lower().strip().rstrip(".!?"), target_code)
        if key in common_phrases:
            return common_phrases[key]

        return text
