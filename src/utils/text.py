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


def polish_message_text(text: str) -> str:
    """
    Polishes and auto-corrects basic spelling, typographical, and grammatical mistakes
    in user message bodies (e.g. 'ho how are u' -> 'Hi, how are you?').
    """
    if not text:
        return ""
    
    msg = text.strip()

    # 1. Fix common greeting typos
    msg = re.sub(r"(?i)\b(?:ho|hlo|hlw|helllo|helo|hiii+|hii)\b", "Hi", msg)
    msg = re.sub(r"(?i)\bhye\b", "Hey", msg)
    msg = re.sub(r"(?i)\bgood\s*mrng\b", "Good morning", msg)
    msg = re.sub(r"(?i)\bgood\s*evng\b", "Good evening", msg)
    msg = re.sub(r"(?i)\bgood\s*nyt|gud\s*nyt\b", "Good night", msg)

    # 2. Expand SMS shorthand & contractions
    msg = re.sub(r"\b[uU]\b", "you", msg)
    msg = re.sub(r"(?i)\bur\b", "your", msg)
    msg = re.sub(r"\b[rR]\b", "are", msg)
    msg = re.sub(r"(?i)\b(?:plz|pls)\b", "please", msg)
    msg = re.sub(r"(?i)\b(?:thx|thnx|tq|ty)\b", "thanks", msg)
    msg = re.sub(r"(?i)\bbtw\b", "by the way", msg)
    msg = re.sub(r"(?i)\basap\b", "as soon as possible", msg)
    msg = re.sub(r"(?i)\bbcz|cuz|coz\b", "because", msg)
    msg = re.sub(r"(?i)\bwont\b", "won't", msg)
    msg = re.sub(r"(?i)\bdont\b", "don't", msg)
    msg = re.sub(r"(?i)\bcant\b", "can't", msg)
    msg = re.sub(r"(?i)\bdidnt\b", "didn't", msg)
    msg = re.sub(r"(?i)\bisnt\b", "isn't", msg)
    msg = re.sub(r"(?i)\barent\b", "aren't", msg)
    msg = re.sub(r"(?i)\bim\b", "I'm", msg)
    msg = re.sub(r"\bi\b", "I", msg)

    # 3. Fix common spelling errors
    msg = re.sub(r"(?i)\bcomming\b", "coming", msg)
    msg = re.sub(r"(?i)\brecieved\b", "received", msg)
    msg = re.sub(r"(?i)\btommorow|tomorow\b", "tomorrow", msg)
    msg = re.sub(r"(?i)\byesterdayy\b", "yesterday", msg)
    msg = re.sub(r"(?i)\bdefinately|definitly\b", "definitely", msg)
    msg = re.sub(r"(?i)\bseperate\b", "separate", msg)
    msg = re.sub(r"(?i)\buntill\b", "until", msg)
    msg = re.sub(r"(?i)\bwich\b", "which", msg)
    msg = re.sub(r"(?i)\bther\b", "there", msg)
    msg = re.sub(r"(?i)\bthier\b", "their", msg)
    msg = re.sub(r"(?i)\bavailble|avialable\b", "available", msg)
    msg = re.sub(r"(?i)\bintrested|intrest\b", "interested", msg)

    # 4. Normalize greeting structure: e.g. "Hi how are you" -> "Hi, how are you"
    msg = re.sub(r"(?i)^(Hi|Hello|Hey)\s+(how\s+are\s+you)", r"\1, \2", msg)
    
    # 5. Fix Question mark for common question openers if punctuation missing
    if re.search(r"(?i)^(?:how\s+are\s+you|what\s+is|when\s+is|where\s+is|can\s+you|could\s+you|are\s+you)\b", msg) and not msg.endswith(("?", ".", "!")):
        msg = f"{msg}?"
    elif re.search(r"(?i)\bhow\s+are\s+you$", msg) and not msg.endswith(("?", ".", "!")):
        msg = f"{msg}?"

    # 6. Capitalize first character
    if msg and msg[0].islower():
        msg = msg[0].upper() + msg[1:]

    # Clean multiple spaces
    msg = re.sub(r"\s+", " ", msg).strip()
    return msg

