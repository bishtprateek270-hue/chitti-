"""
Chitti Memory Extractor Module.
Extracts structured personal facts, detects explicit natural language memory commands,
filters sensitive data, and handles multi-fact compound statements across English, Hindi, and Hinglish.
"""

import re
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any

from src.utils.logging import log_debug, log_warning


@dataclass
class ExtractedFact:
    """Represents a single atomic extracted fact."""
    content: str
    memory_type: str = "fact"  # identity, relationship, professional_identity, preference, project, instruction, fact
    key: Optional[str] = None  # user_name, creator, occupation, favorite_*, etc.
    value: Optional[str] = None
    importance: int = 4


@dataclass
class ExtractedMemoryCommand:
    """Represents a recognized memory command or extracted memory facts."""
    action: str  # 'remember', 'recall', 'forget', 'forget_all', 'none', 'sensitive_rejected'
    content: Optional[str] = None
    memory_type: str = "fact"
    importance: int = 4
    target_category: Optional[str] = None
    facts: List[ExtractedFact] = field(default_factory=list)
    is_sensitive: bool = False
    rejection_reason: Optional[str] = None


# Sensitive information patterns
SENSITIVE_PATTERNS = [
    r"(?i)\b(?:password|passwd|pwd)\s*(?:is|=|:)\s*\S+",
    r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token)\s*(?:is|=|:)\s*\S+",
    r"(?i)\b(?:sk-[a-zA-Z0-9_-]{20,})\b",
    r"(?i)\b(?:bearer\s+[a-zA-Z0-9_\-\.]{20,})\b",
    r"(?i)\b(?:private[_-]?key)\b",
]

# Explicit memory command prefixes across English, Hindi, Hinglish (with typo tolerance)
REMEMBER_PREFIX_PATTERNS = [
    r"^(?:chitti,?\s*)?(?:please\s+)?(?:r[e]?m[e]?mb[e]?r|rmbr)\s+that\s+(.*)$",
    r"^(?:chitti,?\s*)?(?:please\s+)?(?:r[e]?m[e]?mb[e]?r|rmbr)\s+my\s+(.*)$",
    r"^(?:chitti,?\s*)?(?:please\s+)?(?:r[e]?m[e]?mb[e]?r|rmbr)\s+(.*)$",
    r"^(?:chitti,?\s*)?(?:please\s+)?don'?t\s+forget\s+(?:that\s+)?(.*)$",
    r"^(?:chitti,?\s*)?(?:please\s+)?keep\s+in\s+mind\s+(?:that\s+)?(.*)$",
    r"^(?:chitti,?\s*)?(?:please\s+)?note\s+(?:down\s+)?(?:that\s+)?(.*)$",
    r"^(?:chitti,?\s*)?(?:ye\s+baat\s+)?yaad\s+r[a]?khna\s+(?:ki\s+)?(.*)$",
    r"^(?:chitti,?\s*)?yaad\s+r[a]?kho\s+(?:ki\s+)?(.*)$",
    r"^(?:chitti,?\s*)?याद\s+रखना\s+(?:कि\s+)?(.*)$",
    r"^(?:chitti,?\s*)?याद\s+रखो\s+(?:कि\s+)?(.*)$",
]

REMEMBER_SUFFIX_PATTERNS = [
    r"^(.*)[,\.\s]+(?:please\s+)?(?:r[e]?m[e]?mb[e]?r|rmbr)\s+this[!\.]?$",
    r"^(.*)[,\.\s]+(?:please\s+)?(?:r[e]?m[e]?mb[e]?r|rmbr)\s+that[!\.]?$",
    r"^(.*)[,\.\s]+(?:please\s+)?(?:r[e]?m[e]?mb[e]?r|rmbr)\s+it[!\.]?$",
    r"^(.*)[,\.\s]+(?:please\s+)?(?:r[e]?m[e]?mb[e]?r|rmbr)[!\.]?$",
    r"^(.*)[,\.\s]+(?:ye\s+)?yaad\s+r[a]?khna[!\.]?$",
    r"^(.*)[,\.\s]+yaad\s+r[a]?kh[!\.]?$",
    r"^(.*)[,\.\s]+yaad\s+r[a]?kho[!\.]?$",
    r"^(.*)[,\.\s]+bhoolna\s+mat[!\.]?$",
    r"^(.*)[,\.\s]+don'?t\s+forget[!\.]?$",
    r"^(.*)[,\.\s]+याद\s+रखना[!\.]?$",
]

FORGET_PATTERNS = [
    r"^(?:chitti,?\s*)?(?:please\s+)?forget\s+that\s+(.*)$",
    r"^(?:chitti,?\s*)?(?:please\s+)?forget\s+about\s+(.*)$",
    r"^(?:chitti,?\s*)?(?:please\s+)?forget\s+my\s+(.*)$",
    r"^(?:chitti,?\s*)?(?:please\s+)?forget\s+the\s+memory\s+(?:about\s+)?(.*)$",
    r"^(?:chitti,?\s*)?(?:please\s+)?delete\s+(?:the\s+)?memory\s+(?:about\s+)?(.*)$",
    r"^(?:chitti,?\s*)?(?:please\s+)?forget\s+(.*)$",
    r"^(?:chitti,?\s*)?(?:ye\s+)?bhool\s+jao\s+(?:ki\s+)?(.*)$",
    r"^(?:chitti,?\s*)?(?:meri\s+)?memory\s+(?:delete\s+karo|hatao)\s+(?:ki\s+)?(.*)$",
    r"^(?:chitti,?\s*)?भूल\s+जाओ\s+(?:कि\s+)?(.*)$",
]

FORGET_ALL_PATTERNS = [
    r"^(?:chitti,?\s*)?(?:please\s+)?forget\s+everything(?:\s+about\s+me)?[\.!\?]?$",
    r"^(?:chitti,?\s*)?(?:please\s+)?delete\s+all\s+memories[\.!\?]?$",
    r"^(?:chitti,?\s*)?(?:please\s+)?clear\s+all\s+memories[\.!\?]?$",
    r"^(?:chitti,?\s*)?(?:please\s+)?wipe\s+(?:all\s+)?(?:my\s+)?memories[\.!\?]?$",
    r"^(?:chitti,?\s*)?(?:sab|sabkuch)\s+bhool\s+jao[\.!\?]?$",
    r"^(?:chitti,?\s*)?meri\s+saari\s+memories\s+(?:delete|clear)\s+kar\s+do[\.!\?]?$",
    r"^(?:chitti,?\s*)?सब\s+भूल\s+जाओ[\.!\?]?$",
    r"^(?:chitti,?\s*)?मेरी\s+सारी\s+यादें\s+(?:मिटा|डिलीट)\s+कर\s+दो[\.!\?]?$",
]

RECALL_INSPECT_PATTERNS = [
    r"^(?:chitti,?\s*)?(?:show|tell)\s+me\s+what\s+you\s+remember(?:\s+about\s+(.*))?[\.!\?]?$",
    r"^(?:chitti,?\s*)?what\s+do\s+you\s+remember(?:\s+about\s+(.*))?[\.!\?]?$",
    r"^(?:chitti,?\s*)?list\s+(?:all\s+)?(?:my\s+)?memories(?:\s+about\s+(.*))?[\.!\?]?$",
    r"^(?:chitti,?\s*)?what\s+are\s+my\s+memories(?:\s+about\s+(.*))?[\.!\?]?$",
    r"^(?:chitti,?\s*)?(?:mujhe\s+)?batao\s+(?:tujhe\s+)?kya\s+yaad\s+hai(?:\s+mere\s+bare\s+mein)?[\.!\?]?$",
    r"^(?:chitti,?\s*)?meri\s+memories\s+(?:dikhao|batao)[\.!\?]?$",
    r"^(?:chitti,?\s*)?तुम्हें\s+मेरे\s+बारे\s+में\s+क्या\s+याद\s+है\??$",
]


class MemoryExtractor:
    """Analyzes text for explicit memory commands, discrete personal facts, and safety checks."""

    @staticmethod
    def is_sensitive(text: str) -> bool:
        """Returns True if the text appears to contain passwords, API keys, or secrets."""
        for pattern in SENSITIVE_PATTERNS:
            if re.search(pattern, text):
                return True
        return False

    @staticmethod
    def clean_remembered_content(raw_text: str) -> str:
        """Normalizes statement for clean extraction by stripping surrounding command markers."""
        cleaned = raw_text.strip().rstrip(".!? \t\n")

        # Strip Chitti name prefixes
        cleaned = re.sub(r"^(?:chitti,?\s*|bhai,?\s*|hey chitti,?\s*)", "", cleaned, flags=re.IGNORECASE).strip()

        # Check suffixes first
        for pat in REMEMBER_SUFFIX_PATTERNS:
            m = re.match(pat, cleaned, re.IGNORECASE)
            if m:
                cleaned = m.group(1).strip().rstrip(",; \t\n")
                break

        # Check prefixes
        for pat in REMEMBER_PREFIX_PATTERNS:
            m = re.match(pat, cleaned, re.IGNORECASE)
            if m:
                cleaned = m.group(1).strip()
                break

        return cleaned

    @staticmethod
    def split_into_clauses(text: str) -> List[str]:
        """Splits a compound sentence into distinct fact clauses."""
        # Normalize connectors
        # Split on commas, semicolons, 'and', 'also', 'aur', 'tatha', 'plus'
        parts = re.split(r"(?:,\s*(?:and|also|aur|bhi)?\s*|;\s*|\s+(?:and|also|aur|tatha|plus)\s+)", text, flags=re.IGNORECASE)
        clauses = [p.strip().rstrip(".!? \t\n") for p in parts if p.strip()]
        return clauses if clauses else [text]

    @classmethod
    def extract_discrete_facts(cls, raw_text: str) -> List[ExtractedFact]:
        """
        Parses text and extracts discrete structured facts:
        - user_name
        - creator
        - occupation / profession
        - preferences
        - projects
        - general facts
        """
        cleaned_core = cls.clean_remembered_content(raw_text)
        clauses = cls.split_into_clauses(cleaned_core)
        facts: List[ExtractedFact] = []

        user_name_found: Optional[str] = None
        is_creator_found = False

        # First pass to discover user name if present
        for clause in clauses:
            # 1. User Name Patterns (English / Hindi / Hinglish / Devanagari)
            name_patterns = [
                r"(?i)\b(?:my\s+name\s+is|i\s+am|i'm)\s+([A-Za-z\s]+?)(?:\s+(?:and|who|i|mera|maine|also|and\s+i|and\s+i'm|and\s+i\s+have)|$)",
                r"(?i)\b(?:mera\s+na+m\s+(?:hai\s+)?|main\s+)([A-Za-z\s]+?)(?:\s+(?:hoon|hu|h|hai)|$)",
                r"(?:मेरा\s+नाम\s+|मैं\s+)([\u0900-\u097F\s]+?)(?:\s+हूँ|\s+हूं|\s+है|$)",
            ]
            for pat in name_patterns:
                m = re.search(pat, clause)
                if m:
                    extracted_name = m.group(1).strip()
                    # Filter out common filler/pronoun miscaptures
                    if extracted_name.lower() not in {"an", "a", "your", "tumhara", "the", "an aiml", "aiml", "engineer", "creator"}:
                        user_name_found = extracted_name
                        break

        # Second pass to extract all discrete facts
        for clause in clauses:
            clause_clean = clause.strip()
            if not clause_clean:
                continue

            lower = clause_clean.lower()

            # 1. Creator Relationship Extraction (Highest Priority)
            creator_patterns = [
                r"(?i)\b(?:i\s+(?:have\s+)?(?:created|create|made|make|built|build|developed|develop)\s+you|i\s+am\s+your\s+creator|i'm\s+your\s+creator|you\s+(?:were|are|have\s+been)\s+(?:created|made|built|developed)\s+by\s+me)\b",
                r"(?i)\b(?:maine\s+(?:hi\s+)?(?:tumhe|tujhe|chitti\s+ko)\s+banaya\s*(?:hai|h)?|main\s+(?:hi\s+)?(?:tumhara|tera)\s+creator\s+(?:hoon|hu|h)|maine\s+(?:hi\s+)?create\s+kiya\s+(?:hai|h))\b",
                r"(?:मैंने\s+(?:ही\s+)?तुम्हें\s+बनाया\s+है|मैं\s+तुम्हारा\s+क्रिएटर\s+हूँ)",
            ]
            if any(re.search(pat, clause_clean) for pat in creator_patterns):
                creator_name = user_name_found if user_name_found else "User"
                facts.append(ExtractedFact(
                    content=f"{creator_name} is my creator (User created Chitti).",
                    memory_type="relationship",
                    key="creator",
                    value=creator_name,
                    importance=5,
                ))
                is_creator_found = True
                continue

            # 2. Occupation / Profession Extraction (Higher priority than general 'I am')
            is_occ = False
            occ_val = None
            if any(k in lower for k in ["aiml engineer", "ai engineer", "ml engineer", "software engineer", "developer", "data scientist", "researcher", "student", "doctor", "designer"]):
                if "software engineer" in lower:
                    occ_val = "Software Engineer"
                elif "data scientist" in lower:
                    occ_val = "Data Scientist"
                elif "aiml" in lower or "ai/ml" in lower or "ai engineer" in lower:
                    occ_val = "AIML engineer"
                elif "developer" in lower:
                    occ_val = "Developer"
                elif "student" in lower:
                    occ_val = "Student"
                elif "doctor" in lower:
                    occ_val = "Doctor"
                elif "designer" in lower:
                    occ_val = "Designer"
                else:
                    occ_val = "Engineer"

            occ_patterns = [
                r"(?i)\b(?:i\s+am\s+an?|i'm\s+an?|i\s+work\s+as\s+an?)\s+([A-Za-z0-9_\-\s]+?)(?:\s+engineer|\s+developer|\s+scientist|\s+student|\s+designer|\s+researcher|\s+doctor|$)",
                r"(?i)\b(?:main\s+ek\s+|main\s+)([A-Za-z0-9_\-\s]+?)(?:\s+hoon|\s+hu|\s+h)\b",
                r"(?:मैं\s+एक\s+|मैं\s+)([\u0900-\u097F\s]+?)(?:\s+हूँ|\s+हूं)",
            ]
            if not occ_val:
                for pat in occ_patterns:
                    m = re.search(pat, clause_clean)
                    if m:
                        captured = m.group(1).strip()
                        if captured.lower() not in {"user", "creator", "prateek", "human"}:
                            occ_val = captured.title()
                            break

            if occ_val:
                facts.append(ExtractedFact(
                    content=f"User is an {occ_val}.",
                    memory_type="professional_identity",
                    key="occupation",
                    value=occ_val,
                    importance=4,
                ))
                continue

            # 3. User Name Extraction
            name_val = None
            name_patterns = [
                r"(?i)\b(?:my\s+name\s+is)\s+([A-Za-z\s]+)",
                r"(?i)\b(?:i\s+am|i'm)\s+(?!an?\s+|the\s+|your\s+|a\s+)([A-Za-z\s]+)",
                r"(?i)\b(?:mera\s+na+m\s+)([A-Za-z\s]+?)(?:\s+hai|\s+h|$)",
                r"(?:मेरा\s+नाम\s+)([\u0900-\u097F\s]+?)(?:\s+है|$)",
            ]
            for pat in name_patterns:
                m = re.search(pat, clause_clean)
                if m:
                    candidate = m.group(1).strip()
                    if candidate.lower() not in {"an", "a", "your", "tumhara", "the", "creator", "engineer", "developer", "aiml"}:
                        name_val = candidate.title()
                        break

            if name_val:
                facts.append(ExtractedFact(
                    content=f"User's name is {name_val}.",
                    memory_type="identity",
                    key="user_name",
                    value=name_val,
                    importance=5,
                ))
                user_name_found = name_val
                continue

            # 4. Preference Extraction
            if any(k in lower for k in ["favorite", "favourite", "prefer", "like", "love", "pasand"]):
                pref_content = clause_clean
                if not pref_content.lower().startswith("user"):
                    pref_content = f"User preference: {clause_clean}"
                facts.append(ExtractedFact(
                    content=pref_content,
                    memory_type="preference",
                    key="preference",
                    value=clause_clean,
                    importance=3,
                ))
                continue

            # 5. Project / Work Extraction
            if any(k in lower for k in ["project", "docforensics", "building", "app", "codebase", "churnops"]):
                facts.append(ExtractedFact(
                    content=clause_clean if clause_clean.lower().startswith("user") else f"User project: {clause_clean}",
                    memory_type="project",
                    key="project",
                    value=clause_clean,
                    importance=4,
                ))
                continue

            # 6. Fallback General Fact
            if len(clause_clean) > 3:
                facts.append(ExtractedFact(
                    content=clause_clean,
                    memory_type="fact",
                    importance=3,
                ))

        # If user_name was found and creator was also stated, ensure creator fact reflects the name
        if user_name_found:
            for f in facts:
                if f.key == "creator" and (f.value == "User" or not f.value):
                    f.value = user_name_found
                    f.content = f"{user_name_found} is my creator (User created Chitti)."

        return facts

    def extract_command(self, user_text: str) -> ExtractedMemoryCommand:
        """
        Parses user text to determine if it is an explicit memory command or fact statement.
        """
        raw = user_text.strip()
        lower = raw.lower()

        # Check for sensitive data first
        if self.is_sensitive(raw):
            return ExtractedMemoryCommand(
                action="sensitive_rejected",
                is_sensitive=True,
                rejection_reason="For security reasons, Chitti does not store passwords, API keys, or private tokens in persistent memory."
            )

        # 1. Check for FORGET ALL / CLEAR ALL
        for pattern in FORGET_ALL_PATTERNS:
            if re.match(pattern, lower):
                return ExtractedMemoryCommand(action="forget_all")

        # 2. Check for RECALL / INSPECT
        for pattern in RECALL_INSPECT_PATTERNS:
            match = re.match(pattern, lower)
            if match:
                cat = match.group(1).strip() if match.group(1) else None
                return ExtractedMemoryCommand(action="recall", target_category=cat)

        # 3. Check for FORGET specific
        for pattern in FORGET_PATTERNS:
            match = re.match(pattern, raw, re.IGNORECASE)
            if match:
                target = match.group(1).strip().rstrip(".!? \t\n")
                if target:
                    return ExtractedMemoryCommand(action="forget", content=target)

        # 4. Check for explicit REMEMBER commands (Prefixes, Suffixes, or Infixes)
        is_explicit_remember = False
        for pat in REMEMBER_PREFIX_PATTERNS + REMEMBER_SUFFIX_PATTERNS:
            if re.match(pat, raw, re.IGNORECASE):
                is_explicit_remember = True
                break

        # Check for remember / yaad rakhna keywords anywhere in text
        if not is_explicit_remember:
            is_explicit_remember = bool(re.search(
                r"(?i)\b(?:r[e]?m[e]?mb[e]?r|rmbr|yaad\s+r[a]?khna|yaad\s+r[a]?kho|yaad\s+r[a]?kh|bhoolna\s+mat|don'?t\s+forget)\b",
                raw
            ))

        # Also detect direct self-identification or creator statements as facts to remember
        is_self_identity = bool(re.search(
            r"(?i)\b(?:my\s+name\s+is|i\s+am\s+[A-Za-z]|i'm\s+[A-Za-z]|mera\s+na+m\s+|"
            r"i\s+(?:have\s+)?(?:created|create|made|make|built|build|developed)\s+you|maine\s+tumhe\s+banaya|main\s+tumhara\s+creator|main\s+tera\s+creator|"
            r"main\s+(?:ek\s+)?[A-Za-z0-9_\-\s]+?(?:engineer|developer|scientist|student|designer|doctor|hoon|hu|h)|"
            r"i\s+work\s+as|मेरा\s+नाम\s+|मैंने\s+तुम्हें\s+बनाया|मैं\s+.*हूँ)\b",
            raw
        ))

        if is_explicit_remember or is_self_identity:
            facts = self.extract_discrete_facts(raw)
            if facts:
                primary_content = facts[0].content
                primary_type = facts[0].memory_type
                primary_imp = facts[0].importance
                return ExtractedMemoryCommand(
                    action="remember",
                    content=primary_content,
                    memory_type=primary_type,
                    importance=primary_imp,
                    facts=facts,
                )

        # No memory command found
        return ExtractedMemoryCommand(action="none")
