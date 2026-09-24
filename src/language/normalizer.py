"""
Chitti Language Normalizer Module.
Performs spelling normalization, STT error correction, negation preservation,
and semantic intent canonicalization across English, Hindi, and Hinglish.
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Set
from src.language.language_models import (
    LanguageCode,
    IntentCategory,
    NormalizedAction,
    NormalizedIntent,
)
from src.language.detector import LanguageDetector


# Phonetic and colloquial spelling mappings for Roman Hindi
PHONETIC_NORMALIZATION_MAP: Dict[str, str] = {
    # Verb stems & auxiliaries
    "krna": "karna",
    "krnaa": "karna",
    "kro": "karo",
    "krr": "kar",
    "kr": "kar",
    "krta": "karta",
    "krti": "karti",
    "krte": "karte",
    "rha": "raha",
    "rhi": "rahi",
    "rhe": "rahe",
    "btao": "batao",
    "btae": "bataiye",
    "smjhao": "samjhao",
    "smjh": "samajh",
    "dilaa": "dilao",
    "dila": "dilao",
    "kholnaa": "kholna",

    # Pronouns
    "mjhe": "mujhe",
    "mujko": "mujhe",
    "mre": "mere",
    "apko": "aapko",
    "apka": "aapka",
    "apki": "aapki",
    "apke": "aapke",
    "tmhe": "tumhe",
    "tmhara": "tumhara",
    "hmara": "humara",
    "unko": "unhe",

    # Question words
    "kyu": "kyun",
    "kyn": "kyun",
    "kyon": "kyun",
    "kese": "kaise",
    "kha": "kahan",
    "kon": "kaun",

    # Negations & Particles
    "nhi": "nahi",
    "nahin": "nahi",
    "kabhinahi": "kabhi nahi",
    "vse": "waise",
    "wese": "waise",
    "thik": "theek",
    "achha": "acha",
    "fir": "phir",
    "upr": "upar",
    "wla": "wala",
    "wli": "wali",
    "wle": "wale",
    "yar": "yaar",
}

# Common speech-to-text misrecognitions to canonical technical & entity tokens
STT_CORRECTION_MAP: Dict[str, str] = {
    "chitty": "Chitti",
    "chiti": "Chitti",
    "chitti robot": "Chitti",
    "v s code": "VS Code",
    "vs code": "VS Code",
    "vscode": "VS Code",
    "visual studio code": "VS Code",
    "you tube": "YouTube",
    "youtube": "YouTube",
    "g p u": "GPU",
    "gpu": "GPU",
    "c p u": "CPU",
    "cpu": "CPU",
    "r a m": "RAM",
    "ram": "RAM",
    "d b m s": "DBMS",
    "dbms": "DBMS",
    "p y t h o n": "Python",
    "python": "Python",
    "g i t h u b": "GitHub",
    "github": "GitHub",
    "c n n": "CNN",
    "r n n": "RNN",
    "p c a": "PCA",
    "a i": "AI",
    "m l": "ML",
    "d l": "DL",
}

# Negation words across English, Hindi, and Hinglish
NEGATION_TOKENS: Set[str] = {
    "mat", "nahi", "nahin", "nhi", "na", "kabhi nahi", "kabhinahi",
    "don't", "dont", "do not", "never", "cannot", "cant", "can't",
    "not", "no", "stop"
}

# Application & Website aliases (English, Roman Hindi, Devanagari)
APP_NAME_MAP: Dict[str, str] = {
    "chrome": "Google Chrome",
    "google chrome": "Google Chrome",
    "browser": "Browser",
    "vs code": "VS Code",
    "vscode": "VS Code",
    "code editor": "VS Code",
    "terminal": "Terminal",
    "powershell": "PowerShell",
    "cmd": "Command Prompt",
    "youtube": "YouTube",
    "github": "GitHub",
    "spotify": "Spotify",
    "notepad": "Notepad",
    "calculator": "Calculator",
    "क्रोम": "Google Chrome",
    "गूगल क्रोम": "Google Chrome",
    "वीएस कोड": "VS Code",
    "टर्मिनल": "Terminal",
    "कैल्कुलेटर": "Calculator",
    "कैलकुलेटर": "Calculator",
    "file manager": "File Explorer",
    "file explorer": "File Explorer",
    "explorer": "File Explorer",
    "downloads": "File Explorer",
}


class LanguageNormalizer:
    """Normalizes spelling, corrects STT mistakes, extracts semantic intents, and preserves negations."""

    def __init__(self, detector: Optional[LanguageDetector] = None):
        self.detector = detector or LanguageDetector()

    def normalize_text(self, text: str) -> str:
        """
        Applies phonetic spelling normalization and STT error corrections
        without modifying technical tokens or file paths.
        """
        if not text:
            return ""

        # Preserve Windows file paths, URLs, and code blocks
        path_pattern = r"(?:[a-zA-Z]:\\[^\s]+)"
        paths = re.findall(path_pattern, text)
        for i, p in enumerate(paths):
            text = text.replace(p, f"__PATH_{i}__")

        url_pattern = r"https?://[^\s]+"
        urls = re.findall(url_pattern, text)
        for i, u in enumerate(urls):
            text = text.replace(u, f"__URL_{i}__")

        # 1. Apply STT corrections (multi-word phrases first)
        for mistake, correct in sorted(STT_CORRECTION_MAP.items(), key=lambda x: len(x[0]), reverse=True):
            pattern = rf"\b{re.escape(mistake)}\b"
            text = re.sub(pattern, correct, text, flags=re.IGNORECASE)

        # 2. Apply phonetic corrections word by word
        tokens = text.split()
        normalized_tokens = []
        for token in tokens:
            clean = re.sub(r"[^\w]", "", token.lower())
            if clean in PHONETIC_NORMALIZATION_MAP:
                fixed = PHONETIC_NORMALIZATION_MAP[clean]
                if token.endswith((".", ",", "?", "!")):
                    fixed += token[-1]
                normalized_tokens.append(fixed)
            else:
                normalized_tokens.append(token)

        result = " ".join(normalized_tokens)

        # Restore preserved paths and URLs
        for i, p in enumerate(paths):
            result = result.replace(f"__PATH_{i}__", p)
        for i, u in enumerate(urls):
            result = result.replace(f"__URL_{i}__", u)

        return result

    def check_negation(self, text: str) -> bool:
        """
        Detects if user expression contains negative intent.
        CRITICAL: Never miss negations (e.g. 'mat', 'nahi', 'don't', 'never', 'मत', 'नहीं').
        """
        lower = text.lower()
        tokens = set(re.findall(r"\b\w+(?:'\w+)?\b", lower))

        # If it's an inquiry about why something isn't working, it's not a negated command
        if any(qp in lower for qp in ["kyun nahi", "kyu nahi", "why not", "why is", "why my"]):
            return False

        # Check multi-word negations
        if any(mw in lower for mw in ["do not", "kabhi nahi", "don't", "can't", "cannot", "band mat", "delete mat", "run mat", "delete nahi"]):
            return True

        for t in tokens:
            if t in NEGATION_TOKENS:
                return True

        # Check standalone Devanagari negations
        devanagari_clean = re.sub(r"[।,\.\?!]", " ", text)
        words = set(devanagari_clean.split())
        if "मत" in words or "नहीं" in words or "नही" in words:
            return True

        return False

    def parse_intent(self, raw_text: str) -> NormalizedIntent:
        """
        Parses raw user input in English, Hindi, Roman Hindi, or Hinglish into a
        structured NormalizedIntent with canonical actions and parameters.
        """
        normalized_text = self.normalize_text(raw_text)
        det_result = self.detector.detect(normalized_text)
        is_negated = self.check_negation(normalized_text)
        lower = normalized_text.lower()

        # 1. Check for Explicit Language Switch Command
        lang_switch_patterns = [
            (r"(?i)\b(?:(?:from\s+now\s+on\s+)?english\s+mein\s+(?:bolo|baat\s+karo|samjhao|answer\s+(?:karo|karna|kar)|jawab\s+do)|switch\s+(?:back\s+)?to\s+english|speak\s+in\s+english)\b", "en"),
            (r"(?i)\b(?:(?:from\s+now\s+on\s+)?hindi\s+mein\s+(?:batao|bolo|samjhao|answer\s+(?:karo|karna|kar)|jawab\s+do)|switch\s+(?:back\s+)?to\s+hindi|speak\s+in\s+hindi)\b", "hi"),
            (r"(?i)\b(?:(?:from\s+now\s+on\s+)?hinglish\s+mein\s+(?:batao|samjhao|bolo|answer\s+(?:karo|karna|kar)|jawab\s+do)|switch\s+(?:back\s+)?to\s+hinglish|thoda\s+easy\s+hinglish\s+mein)\b", "hinglish"),
        ]
        for pattern, target_lang in lang_switch_patterns:
            if re.search(pattern, lower):
                return NormalizedIntent(
                    raw_text=raw_text,
                    detected_language=det_result.language,
                    normalized_text=f"Switch assistant response language to {target_lang}.",
                    intent_category=IntentCategory.LANGUAGE_SWITCH.value,
                    target_response_language=target_lang,
                    parameters={"target_language": target_lang},
                    confidence=0.98,
                )

        # Devanagari language switches
        if "अंग्रेजी में बात" in raw_text or "इंग्लिश में बोलो" in raw_text:
            return NormalizedIntent(
                raw_text=raw_text,
                detected_language="hi",
                normalized_text="Switch assistant response language to en.",
                intent_category=IntentCategory.LANGUAGE_SWITCH.value,
                target_response_language="en",
                parameters={"target_language": "en"},
                confidence=0.98,
            )
        if "हिंदी में बताओ" in raw_text or "हिंदी में समझाओ" in raw_text:
            return NormalizedIntent(
                raw_text=raw_text,
                detected_language="hi",
                normalized_text="Switch assistant response language to hi.",
                intent_category=IntentCategory.LANGUAGE_SWITCH.value,
                target_response_language="hi",
                parameters={"target_language": "hi"},
                confidence=0.98,
            )

        # 2. Check for Explicit Translation Command
        trans_patterns = [
            r"(?i)translate\s+(?:this\s+)?(?:sentence\s+)?(?:in|to|into)\s+(\w+)\s*:\s*(.+)",
            r"(?i)isko\s+(\w+)\s+mein\s+translate\s+karo\s*:\s*(.+)",
            r"(?i)translate\s+to\s+(\w+)\s*:\s*(.+)",
        ]
        for tp in trans_patterns:
            trans_match = re.search(tp, raw_text)
            if trans_match:
                target_lang_str = trans_match.group(1).lower()
                text_to_translate = trans_match.group(2).strip()

                target_code = "en"
                if "hindi" in target_lang_str or "हिंदी" in target_lang_str:
                    target_code = "hi"
                elif "hinglish" in target_lang_str:
                    target_code = "hinglish"

                return NormalizedIntent(
                    raw_text=raw_text,
                    detected_language=det_result.language,
                    normalized_text=f"Translate text to {target_code}: {text_to_translate}",
                    intent_category=IntentCategory.TRANSLATION.value,
                    parameters={"target_language": target_code, "text": text_to_translate},
                    confidence=0.95,
                )

        # 3. Check for Vision Perception Query across all languages
        vision_patterns = [
            r"(?i)\b(?:what(?:\s+can|\s+do)?\s+you\s+see|who\s+is\s+(?:in\s+front\s+of\s+you|there|this|that|standing)|is\s+anyone\s+there)\b",
            r"(?i)\b(?:samne\s+kya\s+hai|kya\s+dekh\s+rahe\s+ho|tu\s+kya\s+dekh\s+raha\s+hai|mere\s+samne\s+kaun\s+hai|kaun\s+hai\s+samne|kon\s+hai|kya\s+koi\s+samne)\b",
            r"(?i)\b(?:kya\s+koi\s+hai|mujhe\s+dekh\s+sakte\s+ho|do\s+you\s+recognize\s+me|who\s+am\s+i|face\s+recogni\w*|face\s+detect\w*)\b",
            r"(?i)\b(?:objects?\s+are\s+on\s+my\s+desk|is\s+there\s+a\s+bottle|do\s+you\s+see\s+a\s+keyboard|describe\s+the\s+room\s+scene)\b",
            r"(?i)\b(?:samne\s+kaun\s+sa\s+object|kya\s+table\s+pe\s+bottle|kya\s+samne\s+laptop\s+hai|camera\s+se\s+dekh|look\s+through\s+camera)\b"
        ]
        is_vision = any(re.search(vp, lower) for vp in vision_patterns)
        
        # Devanagari vision checks
        if any(term in raw_text for term in ["सामने क्या है", "सामने कौन", "क्या कोई सामने", "तुम क्या देख रहे"]):
            is_vision = True

        if is_vision:
            return NormalizedIntent(
                raw_text=raw_text,
                detected_language=det_result.language,
                normalized_text="Describe what you see in front of the camera.",
                intent_category=IntentCategory.VISION_QUERY.value,
                confidence=0.96,
            )

        # 4. Check for Memory Query / Command across languages
        if re.search(r"(?i)\b(?:yaad\s+rakhna|yaad\s+rakho|remember\s+that|save\s+this|don't\s+forget\s+that|memory\s+delete)\b", lower) or "याद रखना" in raw_text:
            return NormalizedIntent(
                raw_text=raw_text,
                detected_language=det_result.language,
                normalized_text=normalized_text,
                intent_category=IntentCategory.MEMORY_COMMAND.value,
                actions=[NormalizedAction(action="remember", parameters={"raw": raw_text})],
                is_negated=is_negated,
                confidence=0.95,
            )

        if re.search(r"(?i)\b(?:bhool\s+jao|forget\s+that|forget\s+all|forget\s+everything|purani\s+memory)\b", lower) or "भूल जाओ" in raw_text or "यादें भूल" in raw_text:
            return NormalizedIntent(
                raw_text=raw_text,
                detected_language=det_result.language,
                normalized_text=normalized_text,
                intent_category=IntentCategory.MEMORY_COMMAND.value,
                actions=[NormalizedAction(action="forget", parameters={"raw": raw_text})],
                is_negated=is_negated,
                confidence=0.95,
            )

        # 5. Check for Laptop / App Control Intent across English, Hindi, Hinglish
        app_actions: List[NormalizedAction] = []
        open_keywords = ["open", "start", "launch", "kholo", "khol", "chalao", "chala", "खोलो", "खोल", "run", "load"]
        close_keywords = ["close", "band karo", "band", "exit", "quit", "बंद करो", "बंद", "stop"]

        # Check for open app / website / tool
        for app_alias, formal_name in APP_NAME_MAP.items():
            pattern_open = rf"(?i)\b(?:(?:{'|'.join(open_keywords)})\s+{re.escape(app_alias)}|{re.escape(app_alias)}\s+(?:{'|'.join(open_keywords)}))\b"
            is_dev_open = (app_alias in raw_text and any(k in raw_text for k in ["खोलो", "खोल", "चलाओ"]))
            if re.search(pattern_open, lower) or is_dev_open:
                app_actions.append(
                    NormalizedAction(
                        action="open_application",
                        target=formal_name,
                        is_negated=is_negated,
                    )
                )

            pattern_close = rf"(?i)\b(?:(?:{'|'.join(close_keywords)})\s+{re.escape(app_alias)}|{re.escape(app_alias)}\s+(?:{'|'.join(close_keywords)}))\b"
            is_dev_close = (app_alias in raw_text and any(k in raw_text for k in ["बंद करो", "बंद", "मत"]))
            if re.search(pattern_close, lower) or is_dev_close:
                app_actions.append(
                    NormalizedAction(
                        action="close_application",
                        target=formal_name,
                        is_negated=is_negated,
                    )
                )

        if app_actions:
            canonical_desc = []
            for act in app_actions:
                if act.is_negated:
                    canonical_desc.append(f"Do not {act.action.replace('_', ' ')} {act.target}")
                else:
                    canonical_desc.append(f"{act.action.replace('_', ' ').capitalize()} {act.target}")

            return NormalizedIntent(
                raw_text=raw_text,
                detected_language=det_result.language,
                normalized_text=". ".join(canonical_desc) + ".",
                intent_category=IntentCategory.LAPTOP_CONTROL.value,
                actions=app_actions,
                is_negated=is_negated,
                confidence=0.93,
            )

        # 6. Check for Task / Reminder Intent (e.g. "kal mujhe DBMS wala assignment yaad dila dena")
        reminder_match = re.search(r"(?i)\b(?:(?:kal|tomorrow|aaj|today|sham\s+ko|morning)\s+.*(?:yaad\s+dila|remind\s+me|reminder|assignment)|remind\s+me\s+to\s+.*)\b", lower)
        if reminder_match:
            return NormalizedIntent(
                raw_text=raw_text,
                detected_language=det_result.language,
                normalized_text=f"Set a reminder for the user based on: {normalized_text}",
                intent_category="reminder",
                actions=[NormalizedAction(action="create_reminder", parameters={"text": raw_text})],
                is_negated=is_negated,
                confidence=0.90,
            )

        # 7. General Conversational / Technical Explanation
        return NormalizedIntent(
            raw_text=raw_text,
            detected_language=det_result.language,
            normalized_text=normalized_text,
            intent_category=IntentCategory.CONVERSATION.value,
            is_negated=is_negated,
            confidence=det_result.confidence,
        )
