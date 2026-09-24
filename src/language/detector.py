"""
Chitti Language Detector Module.
Provides fast, zero-dependency, ultra-low-latency detection for English,
Devanagari Hindi, Roman Hindi, and Hinglish.
"""

import re
from typing import List, Set, Tuple
from src.language.language_models import LanguageCode, LanguageDetectionResult


# Unambiguous Roman Hindi & Hinglish keywords, particles, verbs, pronouns, and phonetic variants
UNAMBIGUOUS_ROMAN_HINDI_MARKERS: Set[str] = {
    # Pronouns & Possessives
    "mai", "mujhe", "mujko", "mjhe", "mera", "meri", "mere", "mre",
    "tum", "tumhe", "tumhara", "tumhari", "tumhare", "tmhe", "tmhara",
    "aap", "aapko", "aapka", "aapki", "aapke", "apko", "apka", "apki", "apke",
    "hum", "hume", "humko", "humara", "humari", "humare", "hmara",
    "wo", "woh", "usko", "uska", "uski", "uske", "unhe", "unka", "unki", "unke",
    "ye", "yeh", "isko", "iska", "iski", "iske", "inhe", "inka", "inke",
    "koi", "kisi", "kisko", "kisne", "kuch", "kuchh", "kuchbhi", "sabko",

    # Question & Relative words
    "kya", "kaise", "kese", "kyu", "kyun", "kyn", "kyon", "kab", "kaha", "kahan", "kha",
    "kaun", "kon", "kitna", "kitni", "kitne", "kitnaa", "kaisa", "kaisi",
    "jab", "jaha", "jahan", "jaise", "jiska", "jiski", "jiske", "jisne",

    # Verbs & Action stems
    "kholo", "khol", "kholna", "kholnaa", "karna", "karo", "kar", "krna", "kro",
    "karega", "karegi", "karenge", "karta", "karti", "karte", "krta", "krti",
    "dekh", "dekho", "dekhna", "dekhnaa", "dekhta", "dekhti", "dekhte", "dekhoge",
    "batao", "bata", "btao", "bataiye", "batana", "samjhao", "samjh", "samajh", "smjhao",
    "chala", "chalao", "chalo", "chal", "chalna", "chalega", "chalegi", "chalte",
    "sunao", "suno", "sun", "sunna", "sunoge", "rakho", "rakh", "hatao", "hata",
    "dila", "dilao", "dilaa", "aana", "jaana", "jao", "aao",
    "likho", "likh", "padho", "padh", "roko", "ruko",
    "bolo", "bolna", "bologi", "bologe", "kehna", "pucho",
    "dhundho", "dhundh", "khojo", "lao", "dena", "dijiye",

    # Auxiliaries & Tenses
    "hai", "hain", "tha", "thi", "hoga", "hogi", "honge", "hu", "hoon",
    "raha", "rahi", "rahe", "rha", "rhi", "rhe", "sakta", "sakti", "sakte", "skta", "skti",
    "chahiye", "chahie", "padega", "padegi", "padenge",

    # Conjunctions, Particles & Postpositions
    "aur", "lekin", "magar", "bhi", "toh",
    "mein", "pe", "wala", "wali", "wale", "wla", "wli",
    "saath", "bina", "andar", "bahar", "upar", "niche", "aage", "piche", "samne",
    "pehle", "phir", "fir", "kabhi", "hamesha", "zyada",

    # Negations
    "mat", "nahi", "nahin", "nhi", "kabhinahi",

    # Common Conversational / Fillers
    "bhai", "yaar", "yar", "namaste", "arre", "arey", "acha", "achha", "theek", "thik",
    "shukriya", "dhanyawad", "zara", "thoda", "thodi", "thode",
    "waise", "wese", "vse"
}

# Ambiguous tokens that exist in both languages (e.g. English 'is', 'me', 'to', 'do', 'or', 'the', 'main')
AMBIGUOUS_TOKENS: Set[str] = {
    "is", "me", "to", "do", "or", "he", "us", "so", "par", "se", "ko", "ka", "ki", "ke",
    "na", "h", "the", "main"
}

# Common English stopwords and functional tokens
COMMON_ENGLISH_WORDS: Set[str] = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
    "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
    "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
    "is", "are", "was", "were", "been", "being", "am", "please", "open", "close",
    "show", "tell", "explain", "why", "where", "search", "run", "start", "stop",
    "delete", "check", "code", "file", "folder", "laptop", "screen", "camera",
    "standing", "front", "room", "remember", "forget", "memories", "switch", "translate",
    "help", "debug", "function", "scene", "view", "between", "difference", "script",
    "load", "workspace", "according", "identities", "desk", "prefer", "mode", "feed",
    "table", "keyboard", "describe", "fail", "failing", "error", "throwing"
}

# Technical keywords that can appear in any language without biasing towards English
TECHNICAL_TERMS: Set[str] = {
    "python", "javascript", "c++", "java", "github", "git", "api", "database", "dbms",
    "sql", "nosql", "gpu", "cpu", "ram", "docker", "react", "fastapi", "flask",
    "vs code", "vscode", "terminal", "powershell", "bash", "linux", "windows",
    "machine learning", "deep learning", "neural network", "cnn", "rnn", "transformer",
    "overfitting", "underfitting", "pca", "kmeans", "regression", "classification",
    "chrome", "youtube", "browser", "google", "playlist", "wifi", "bluetooth",
    "chitti", "code", "bug", "error", "exception", "server", "backend", "frontend",
    "spotify", "notepad", "calculator", "explorer"
}


class LanguageDetector:
    """Fast and accurate language detector for English, Hindi (Devanagari), Roman Hindi, and Hinglish."""

    def __init__(self, confidence_threshold: float = 0.65):
        self.confidence_threshold = confidence_threshold

    @staticmethod
    def _is_devanagari_char(char: str) -> bool:
        """Returns True if character belongs to Devanagari Unicode block."""
        return "\u0900" <= char <= "\u097F"

    def detect(self, text: str) -> LanguageDetectionResult:
        """
        Analyzes the text and determines if it is English, Hindi (Devanagari),
        Roman Hindi, Hinglish, or mixed.
        """
        if not text or not text.strip():
            return LanguageDetectionResult(
                language=LanguageCode.UNKNOWN.value,
                confidence=0.0,
                script="none"
            )

        clean_text = text.strip()
        tokens = re.findall(r"\b\w+\b", clean_text.lower())
        if not tokens:
            return LanguageDetectionResult(
                language=LanguageCode.UNKNOWN.value,
                confidence=0.0,
                script="none"
            )

        # 1. Script Analysis for Devanagari
        total_chars = sum(len(t) for t in tokens)
        devanagari_chars = sum(sum(1 for c in t if self._is_devanagari_char(c)) for t in tokens)
        devanagari_ratio = devanagari_chars / max(1, total_chars)

        if devanagari_ratio > 0.25:
            # Contains substantial Devanagari characters -> Hindi
            confidence = min(0.99, 0.75 + (devanagari_ratio * 0.25))
            return LanguageDetectionResult(
                language=LanguageCode.HINDI.value,
                confidence=confidence,
                script="devanagari",
                hindi_ratio=devanagari_ratio,
            )

        # 2. Token Analysis for Latin Script
        hindi_markers_found: List[str] = []
        english_words_found: List[str] = []
        tech_words_found: List[str] = []

        for idx, token in enumerate(tokens):
            if token in UNAMBIGUOUS_ROMAN_HINDI_MARKERS:
                hindi_markers_found.append(token)
            elif token in AMBIGUOUS_TOKENS:
                # Disambiguate based on neighboring tokens
                prev_token = tokens[idx - 1] if idx > 0 else ""
                next_token = tokens[idx + 1] if idx + 1 < len(tokens) else ""
                
                # Check 'the'
                if token == "the":
                    if prev_token in {"wo", "log", "hum", "wahan"} or next_token in {"aur", "toh"}:
                        hindi_markers_found.append(token)
                    else:
                        english_words_found.append(token)
                # Check 'main' (I in Hindi vs Main in English)
                elif token == "main":
                    if next_token in {"generally", "code", "karta", "hu", "hoon", "bhi", "kuch", "apne", "nahi"} or prev_token in {"ki", "aur", "lekin"}:
                        hindi_markers_found.append(token)
                    else:
                        english_words_found.append(token)
                # Check 'do'
                elif token == "do" and prev_token in {"khol", "kar", "bata", "chala", "de", "dekh", "sun", "mat", "kr"}:
                    hindi_markers_found.append(token)
                elif token in {"se", "ko", "ka", "ki", "ke", "mein"} and (prev_token in UNAMBIGUOUS_ROMAN_HINDI_MARKERS or next_token in UNAMBIGUOUS_ROMAN_HINDI_MARKERS):
                    hindi_markers_found.append(token)
                elif token == "me" and prev_token not in {"tell", "help", "show", "with", "for", "about", "remind", "see", "hear", "ask", "let"}:
                    if next_token in UNAMBIGUOUS_ROMAN_HINDI_MARKERS or prev_token in UNAMBIGUOUS_ROMAN_HINDI_MARKERS:
                        hindi_markers_found.append(token)
                    else:
                        english_words_found.append(token)
                elif token == "is" and (next_token in {"code", "file", "folder", "project", "baat"} and (len(tokens) > idx+2 and tokens[idx+2] in {"ko", "me", "mein", "par", "mat", "nahi"})):
                    hindi_markers_found.append(token)
                elif token in {"h", "na"} and (prev_token in UNAMBIGUOUS_ROMAN_HINDI_MARKERS or next_token in UNAMBIGUOUS_ROMAN_HINDI_MARKERS):
                    hindi_markers_found.append(token)
                else:
                    english_words_found.append(token)
            elif token in COMMON_ENGLISH_WORDS:
                english_words_found.append(token)
            elif token in TECHNICAL_TERMS:
                tech_words_found.append(token)

        total_meaningful_tokens = len(tokens)
        hindi_count = len(hindi_markers_found)
        english_count = len(english_words_found)

        hindi_ratio = hindi_count / max(1, total_meaningful_tokens)
        english_ratio = english_count / max(1, total_meaningful_tokens)

        # 3. Decision Logic
        if hindi_count == 0:
            # Pure English (even if technical tokens exist)
            confidence = min(0.98, 0.75 + (english_ratio * 0.25))
            return LanguageDetectionResult(
                language=LanguageCode.ENGLISH.value,
                confidence=confidence,
                script="latin",
                detected_markers=[],
                english_ratio=english_ratio,
                hindi_ratio=0.0,
            )

        if hindi_count > 0 and english_count == 0:
            # Pure Roman Hindi
            confidence = min(0.98, 0.70 + (hindi_ratio * 0.28))
            return LanguageDetectionResult(
                language=LanguageCode.HINGLISH.value,
                confidence=confidence,
                script="latin",
                is_roman_hindi=True,
                detected_markers=hindi_markers_found,
                english_ratio=0.0,
                hindi_ratio=hindi_ratio,
            )

        if hindi_count > 0 and english_count > 0:
            # Check for English instructional requests wrapping Hindi phrases
            # e.g. "Translate this into Hindi: ..." or "Translate this sentence into English: mujhe ghar jana hai."
            if clean_text.lower().startswith("translate this") or clean_text.lower().startswith("switch to") or clean_text.lower().startswith("translate to"):
                return LanguageDetectionResult(
                    language=LanguageCode.ENGLISH.value,
                    confidence=0.92,
                    script="latin",
                    english_ratio=english_ratio,
                    hindi_ratio=hindi_ratio,
                )

            confidence = min(0.96, 0.65 + ((hindi_ratio + english_ratio) * 0.20))
            return LanguageDetectionResult(
                language=LanguageCode.HINGLISH.value,
                confidence=confidence,
                script="latin",
                is_roman_hindi=True,
                detected_markers=hindi_markers_found,
                english_ratio=english_ratio,
                hindi_ratio=hindi_ratio,
            )

        return LanguageDetectionResult(
            language=LanguageCode.ENGLISH.value,
            confidence=0.70,
            script="latin",
        )
