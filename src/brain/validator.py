"""
Chitti Response Validator and Sanitizer Module.
Ensures zero placeholder leakage, fixes common LLM grammatical glitches in Hindi/Hinglish,
and guards against personal fact hallucinations.
"""

import re
from typing import Optional, List, Any


KNOWN_GLITCHES = [
    (r"(?i)\bmain\s+chitti\s+bana\s+hai\b", "Main Chitti hoon"),
    (r"(?i)\bmain\s+chitti\s+bani\s+hai\b", "Main Chitti hoon"),
    (r"(?i)\baapko\s+kuchh?\s+aur\s+baat\s+karne\s+do\b", "Aur kuch poochna hai?"),
    (r"(?i)\bmain\s+aapki\s+ek\s+cuppa\s+coffee\s+karengi\b", "Main aapki kya madad kar sakta hoon?"),
]

PLACEHOLDER_PATTERNS = [
    r"\[Creator(?:'s)?\s*Name\]",
    r"\[User(?:'s)?\s*Name\]",
    r"\[Name\]",
    r"\[Your\s*Name\]",
    r"<TODO>",
    r"<NAME>",
]


class ResponseValidator:
    """Validates and sanitizes LLM responses before speech or display."""

    @staticmethod
    def sanitize(
        response_text: str,
        user_name: Optional[str] = None,
        creator_name: Optional[str] = None,
        target_lang: str = "en",
    ) -> str:
        """
        Sanitizes the generated response:
        1. Strips leaked bracket placeholders.
        2. Fixes known grammatical glitches in Hinglish.
        3. Cleans stray markdown formatting.
        """
        if not response_text:
            return ""

        text = response_text.strip()

        # 1. Replace or eliminate placeholders
        for pattern in PLACEHOLDER_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                if "creator" in pattern.lower():
                    if creator_name:
                        text = re.sub(pattern, creator_name, text, flags=re.IGNORECASE)
                    else:
                        if target_lang == "hi":
                            text = "मुझे मेरे क्रिएटर की जानकारी अभी मेमोरी में नहीं मिली।"
                        elif target_lang in ("hinglish", "mixed"):
                            text = "Mujhe abhi mere creator ki information memory mein nahi mili."
                        else:
                            text = "I don't have my creator's name in memory yet."
                elif "user" in pattern.lower() or "name" in pattern.lower():
                    if user_name:
                        text = re.sub(pattern, user_name, text, flags=re.IGNORECASE)
                    else:
                        if target_lang == "hi":
                            text = "मुझे आपका नाम अभी याद नहीं है।"
                        elif target_lang in ("hinglish", "mixed"):
                            text = "Mujhe tumhara naam abhi yaad nahi hai."
                        else:
                            text = "I don't have your name in memory yet."
                else:
                    text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

        # 2. Fix known grammatical glitches
        for glitch_pat, replacement in KNOWN_GLITCHES:
            text = re.sub(glitch_pat, replacement, text)

        # 3. Clean trailing repetitive punctuation
        text = re.sub(r"[\s\.,;!]+$", ".", text)
        if not text.endswith((".", "?", "!")):
            text += "."

        return text
