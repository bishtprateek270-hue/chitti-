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
    Cleans a contact name by stripping emojis, leading prepositions ('to', 'for'),
    leading/trailing punctuation, and returns a clean search query.
    Example:
        'Rahul ❤️' -> 'Rahul'
        '🔥 Ayush 🔥' -> 'Ayush'
        'to Ayush' -> 'Ayush'
        'Mummy 🥰 (Home)' -> 'Mummy (Home)'
    """
    if not name:
        return ""
    cleaned = strip_emojis(name)
    cleaned = re.sub(r"(?i)^(?:to\s+|for\s+|ko\s+)", "", cleaned).strip()
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
    application names, web services, and programming keywords.
    """
    if not text:
        return ""
    normalized = text
    # Common verb typos
    normalized = re.sub(r"(?i)\b(?:opem|opne|oppen|oepn|oprn|opn|opeb|openkaro|khol)\b", "open", normalized)
    normalized = re.sub(r"(?i)\b(?:lauch|luanch|lanuch|lanch|launchh)\b", "launch", normalized)
    normalized = re.sub(r"(?i)\b(?:strat|stert|strt|stat)\b", "start", normalized)
    normalized = re.sub(r"(?i)\b(?:messag|mesage|mesg|msg|massge|massage)\b", "message", normalized)
    normalized = re.sub(r"(?i)\b(?:snd|snde|bhej|bhejo)\b", "send", normalized)
    normalized = re.sub(r"(?i)\b(?:serach|sreach|serch|sarch)\b", "search", normalized)
    normalized = re.sub(r"(?i)\b(?:bulid|buid|biuld|bld)\b", "build", normalized)
    normalized = re.sub(r"(?i)\b(?:craete|creat|crate|cratee)\b", "create", normalized)
    normalized = re.sub(r"(?i)\b(?:fuctional|funtional|functioanl)\b", "functional", normalized)
    
    # Common app name & web service typos
    normalized = re.sub(r"(?i)\b(?:gemmini|gemnii|gemni|gimini|gemini\s+ai)\b", "gemini", normalized)
    normalized = re.sub(r"(?i)\b(?:youtub|yotube|utube|you\s+tube|u\s*tube|ytube)\b", "youtube", normalized)
    normalized = re.sub(r"(?i)\byt\b", "youtube", normalized)
    normalized = re.sub(r"(?i)\b(?:watsapp|whatapp|whatspp|whatsap|watsap|wtsp|whats\s+app|whatsaap)\b", "whatsapp", normalized)
    normalized = re.sub(r"(?i)\b(?:vscdoe|vscde|vs\s+code|vsc)\b", "vscode", normalized)
    normalized = re.sub(r"(?i)\b(?:chrone|chorme|crm|gchrome|goggle\s+chrome)\b", "chrome", normalized)
    normalized = re.sub(r"(?i)\b(?:notepd|notepadd|notpad)\b", "notepad", normalized)
    normalized = re.sub(r"(?i)\b(?:calcultor|calclator|caculator|calcualtor|calcutor|calc)\b", "calculator", normalized)
    normalized = re.sub(r"(?i)\b(?:gmaill|gmai|gmil|g\s*mail)\b", "gmail", normalized)
    normalized = re.sub(r"(?i)\b(?:spotfiy|spoty|spotfy|spotifi)\b", "spotify", normalized)
    normalized = re.sub(r"(?i)\b(?:telegrram|telegrm|tg)\b", "telegram", normalized)
    normalized = re.sub(r"(?i)\b(?:disocrd|disord|dc)\b", "discord", normalized)
    normalized = re.sub(r"(?i)\b(?:chatgbt|chat\s+gpt|cgtp|chatgpd)\b", "chatgpt", normalized)
    
    return normalized

