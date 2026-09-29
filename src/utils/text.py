"""
Text normalization, emoji stripping, and string matching utilities for Chitti.
"""
import re
import unicodedata
from typing import Optional

# Comprehensive pattern matching emojis, pictographs, dingbats, and variation selectors
EMOJI_PATTERN = re.compile(
    "["
    "\U00010000-\U0010FFFF"  # Supplementary Multilingual Plane (all emojis, symbols, pictographs)
    "\u2600-\u27BF"          # Miscellaneous Symbols & Dingbats (stars, arrows, scissors, checkmarks)
    "\u2300-\u23FF"          # Miscellaneous Technical
    "\u2B50-\u2B55"          # Stars, circles, geometric shapes
    "\uFE00-\uFE0F"          # Variation Selectors (e.g. \ufe0f emoji style presentation)
    "\u200D"                 # Zero-Width Joiner (ZWJ sequences)
    "\u200B-\u200F"          # Zero-Width spaces and directional marks
    "\u2E80-\u2FD5"          # CJK Radicals / supplementary symbols
    "]+",
    flags=re.UNICODE
)


def strip_emojis(text: str) -> str:
    """
    Strips all emojis, pictographs, dingbats, variation selectors, and zero-width symbols.
    """
    if not text:
        return ""
    # Strip emojis by Unicode block regex
    cleaned = EMOJI_PATTERN.sub("", text)
    # Strip any additional characters with Unicode category 'So' (Symbol, other) or 'Sk' (Symbol, modifier)
    cleaned = "".join(
        ch for ch in cleaned
        if unicodedata.category(ch) not in ("So", "Sk")
    )
    # Collapse whitespace and strip
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def clean_contact_query(name: str) -> str:
    """
    Cleans a contact name by stripping emojis, leading/trailing punctuation,
    and returns a clean search query suitable for WhatsApp / contact searching.
    Example:
        'Rahul ❤️' -> 'Rahul'
        '🔥 Ayush 🔥' -> 'Ayush'
        'Mummy 🥰 (Home)' -> 'Mummy (Home)'
    """
    if not name:
        return ""
    cleaned = strip_emojis(name)
    # Remove leading/trailing quotes, colons, hyphens, stars, spaces
    cleaned = re.sub(r"^[\s\"':;,\-_*#@!~]+|[\s\"':;,\-_*#@!~]+$", "", cleaned).strip()
    return cleaned if cleaned else name.strip()


def matches_contact_name(query: str, target: str) -> bool:
    """
    Checks if a contact query matches a target contact name ignoring emojis, case, and decoration.
    Example:
        matches_contact_name('Rahul', 'Rahul ❤️') -> True
        matches_contact_name('Ayush', '🔥 Ayush 🔥') -> True
    """
    if not query or not target:
        return False
    clean_q = clean_contact_query(query).lower()
    clean_t = clean_contact_query(target).lower()
    if not clean_q or not clean_t:
        return False
    return clean_q in clean_t or clean_t in clean_q


def normalize_typos(text: str) -> str:
    """
    Corrects common typographical and speech recognition errors in action verbs,
    application names, and programming keywords.
    """
    if not text:
        return ""
    normalized = text
    # Common verb typos
    normalized = re.sub(r"(?i)\b(?:opem|opne|oppen|oepn)\b", "open", normalized)
    normalized = re.sub(r"(?i)\b(?:lauch|luanch|lanuch)\b", "launch", normalized)
    normalized = re.sub(r"(?i)\b(?:messag|mesage|mesg|msg)\b", "message", normalized)
    normalized = re.sub(r"(?i)\b(?:serach|sreach)\b", "search", normalized)
    normalized = re.sub(r"(?i)\b(?:bulid|buid|biuld)\b", "build", normalized)
    normalized = re.sub(r"(?i)\b(?:craete|creat|crate)\b", "create", normalized)
    normalized = re.sub(r"(?i)\b(?:fuctional|funtional|functioanl)\b", "functional", normalized)
    # Common app name & keyword typos
    normalized = re.sub(r"(?i)\b(?:watsapp|whatapp|whatspp|whatsap|watsap|wtsp)\b", "whatsapp", normalized)
    normalized = re.sub(r"(?i)\b(?:vscdoe|vscde)\b", "vscode", normalized)
    normalized = re.sub(r"(?i)\b(?:youtub|yotube|utube)\b", "youtube", normalized)
    normalized = re.sub(r"(?i)\b(?:chrone|chorme|crm)\b", "chrome", normalized)
    normalized = re.sub(r"(?i)\b(?:notepd|notepadd)\b", "notepad", normalized)
    normalized = re.sub(r"(?i)\b(?:calcultor|calclator|caculator|calcualtor|calcutor)\b", "calculator", normalized)
    return normalized

