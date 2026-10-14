"""
Chitti Global Voice Control & Phrase Matcher Module.
Provides high-priority, zero-latency detection of Wake, Sleep, and Emergency Stop phrases
across English, Hindi, and Hinglish with configurable phrase registries and priority enforcement.
"""

import enum
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from src.core.state_manager import CommandPriority
from src.utils.logging import log_chitti, log_debug, log_info


class ControlCommand(str, enum.Enum):
    """Core Voice Control Command Actions."""
    STOP = "STOP"
    SLEEP = "SLEEP"
    WAKE = "WAKE"
    NORMAL = "NORMAL"


@dataclass
class MatchResult:
    """Represents the outcome of phrase evaluation."""
    command: ControlCommand
    priority: int
    matched_phrase: str
    confidence: float
    original_text: str
    cleaned_text: str
    extracted_tail: Optional[str] = None  # Any command spoken directly after wake word


class PhraseMatcher:
    """
    Configurable, high-precision phrase classifier.
    Ensures emergency stop and interrupt phrases receive instant Priority 1 processing.
    """

    # 1. Wake phrases (as specified in Project requirements)
    WAKE_PHRASES_EN = [
        "chitti wake up",
        "wake up chitti",
        "hey chitti",
        "chitti",
        "chitti are you there",
        "are you there chitti",
    ]
    WAKE_PHRASES_HI = [
        "chitti jaag ja",
        "chitti jaago",
        "chitti uth ja",
        "chitti utho",
        "chitti sun",
        "chitti suno",
        "jaag ja chitti",
        "suno chitti",
        "चिट्टी जाग जा",
        "चिट्टी उठ जा",
        "चिट्टी सुन",
        "जाग जा चिट्टी",
    ]
    WAKE_PHRASES_HINGLISH = [
        "chitti wake up",
        "hey chitti suno",
        "chitti active ho ja",
        "chitti active ho jao",
        "active ho ja chitti",
    ]

    # 2. Sleep phrases
    SLEEP_PHRASES_EN = [
        "chitti sleep",
        "go to sleep",
        "sleep now",
        "chitti take a rest",
        "take a rest",
        "sleep mode",
        "goodbye chitti",
        "standby mode",
        "go to standby",
    ]
    SLEEP_PHRASES_HI = [
        "so jao",
        "so ja",
        "chitti so ja",
        "so ja chitti",
        "chitti so jao",
        "so jao chitti",
        "chitti aaram kar",
        "chitti aaram karo",
        "chitti chup ho ja",
        "चिट्टी सो जा",
        "सो जा चिट्टी",
        "चिट्टी आराम कर",
        "चिट्टी चुप हो जा",
    ]
    SLEEP_PHRASES_HINGLISH = [
        "chitti sleep mode",
        "chitti ab so ja",
        "chitti ab so jao",
        "chitti rest kar",
        "chitti rest karo",
        "sleep mode me jao",
        "standby pe jao",
    ]

    # 3. Emergency Stop phrases
    STOP_PHRASES_EN = [
        "stop",
        "stop chitti",
        "chitti stop",
        "stop now",
        "cancel",
        "cancel everything",
        "be quiet",
        "silence",
        "abort",
        "shut up",
        "halt",
        "mute",
    ]
    STOP_PHRASES_HI = [
        "ruk ja",
        "ruko",
        "ruk jao",
        "bas kar",
        "bas karo",
        "chup ho ja",
        "chup ho jao",
        "chup raho",
        "chup",
        "band kar",
        "band karo",
        "chitti ruk ja",
        "chitti ruko",
        "chitti bas kar",
        "चिट्टी रुक जा",
        "रुक जा",
        "रुक जाओ",
        "बस कर",
        "बस करो",
        "चुप हो जा",
        "बंद कर",
        "बंद करो",
    ]
    STOP_PHRASES_HINGLISH = [
        "chitti stop",
        "ruk ja chitti",
        "chitti bas kar",
        "chitti ab band kar",
        "chitti cancel",
        "cancel kar do",
        "stop ho jao",
        "chitti shant ho jao",
    ]

    def __init__(
        self,
        wake_phrases: Optional[List[str]] = None,
        sleep_phrases: Optional[List[str]] = None,
        stop_phrases: Optional[List[str]] = None,
    ):
        self.wake_phrases: Set[str] = set(
            w.lower().strip()
            for w in (wake_phrases or (self.WAKE_PHRASES_EN + self.WAKE_PHRASES_HI + self.WAKE_PHRASES_HINGLISH))
        )
        self.sleep_phrases: Set[str] = set(
            s.lower().strip()
            for s in (sleep_phrases or (self.SLEEP_PHRASES_EN + self.SLEEP_PHRASES_HI + self.SLEEP_PHRASES_HINGLISH))
        )
        self.stop_phrases: Set[str] = set(
            p.lower().strip()
            for p in (stop_phrases or (self.STOP_PHRASES_EN + self.STOP_PHRASES_HI + self.STOP_PHRASES_HINGLISH))
        )

        # Precompile regexes for fast matching
        self._build_matchers()

    def _build_matchers(self):
        """Constructs optimized regexes sorted by phrase length descending."""
        # Clean punctuation and normalize spacing helper
        def make_pattern(phrases: Set[str]) -> str:
            # Sort longest first so "chitti stop" matches before "stop"
            sorted_phrases = sorted(phrases, key=lambda x: len(x), reverse=True)
            escaped = [re.escape(p) for p in sorted_phrases]
            return r"(?i)\b(?:" + "|".join(escaped) + r")\b"

        self._stop_regex = re.compile(make_pattern(self.stop_phrases), re.IGNORECASE)
        self._sleep_regex = re.compile(make_pattern(self.sleep_phrases), re.IGNORECASE)
        self._wake_regex = re.compile(make_pattern(self.wake_phrases), re.IGNORECASE)

    def clean_text(self, text: str) -> str:
        """Normalizes text for evaluation: strips punctuation, lowercase, collapse spaces."""
        if not text:
            return ""
        # Remove common speech-recognition punctuation (commas, periods, exclamation, quotes)
        cleaned = re.sub(r"[,\.!\?\"'\-–—:;]", " ", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
        return cleaned

    def classify_command(self, raw_text: str) -> MatchResult:
        """
        Evaluates input text against the Priority Hierarchy:
        Priority 1: Emergency Stop Phrases
        Priority 2: Sleep Phrases
        Priority 3: Wake Phrases
        Priority 4: Normal User Command
        """
        cleaned = self.clean_text(raw_text)
        if not cleaned:
            return MatchResult(
                command=ControlCommand.NORMAL,
                priority=CommandPriority.PRIORITY_4_USER_COMMAND.value,
                matched_phrase="",
                confidence=0.0,
                original_text=raw_text,
                cleaned_text="",
            )

        # ---------------- PRIORITY 1: EMERGENCY STOP ----------------
        stop_match = self._stop_regex.search(cleaned)
        sleep_match = self._sleep_regex.search(cleaned)

        # ---------------- SPECIFICITY & PRIORITY ----------------
        # If a sleep phrase is strictly more specific (longer) than the stop match
        # (e.g. "chitti chup ho ja" [sleep] vs "chup ho ja" [stop]), honor the explicit sleep command
        if sleep_match and (not stop_match or len(sleep_match.group(0)) > len(stop_match.group(0))):
            phrase = sleep_match.group(0).lower()
            conf = 1.0 if cleaned == phrase else 0.90
            log_debug(f"[CONTROL] Match SLEEP: '{phrase}' in '{cleaned}' (Priority 2)")
            return MatchResult(
                command=ControlCommand.SLEEP,
                priority=CommandPriority.PRIORITY_2_SLEEP.value,
                matched_phrase=phrase,
                confidence=conf,
                original_text=raw_text,
                cleaned_text=cleaned,
            )

        # ---------------- PRIORITY 1: EMERGENCY STOP ----------------
        if stop_match:
            phrase = stop_match.group(0).lower()
            # Verify it's not a negated query like "don't stop"
            if not re.search(r"\b(?:don'?t|do\s+not|never|mat)\s+" + re.escape(phrase), cleaned):
                # High confidence if exact match or standalone utterance
                conf = 1.0 if cleaned == phrase else 0.95
                log_debug(f"[CONTROL] Match STOP: '{phrase}' in '{cleaned}' (Priority 1)")
                return MatchResult(
                    command=ControlCommand.STOP,
                    priority=CommandPriority.PRIORITY_1_INTERRUPT.value,
                    matched_phrase=phrase,
                    confidence=conf,
                    original_text=raw_text,
                    cleaned_text=cleaned,
                )

        # ---------------- PRIORITY 2: SLEEP ----------------
        if sleep_match:
            phrase = sleep_match.group(0).lower()
            conf = 1.0 if cleaned == phrase else 0.90
            log_debug(f"[CONTROL] Match SLEEP: '{phrase}' in '{cleaned}' (Priority 2)")
            return MatchResult(
                command=ControlCommand.SLEEP,
                priority=CommandPriority.PRIORITY_2_SLEEP.value,
                matched_phrase=phrase,
                confidence=conf,
                original_text=raw_text,
                cleaned_text=cleaned,
            )

        # ---------------- PRIORITY 3: WAKE ----------------
        wake_match = self._wake_regex.search(cleaned)
        if wake_match:
            phrase = wake_match.group(0).lower()
            idx = cleaned.find(phrase)
            prefix = cleaned[:idx].strip()
            # If preceded by words other than filler ('hey', 'arre', 'oye'), treat as conversational greeting
            if prefix in ("", "hey", "arre", "oye"):
                tail = cleaned[idx + len(phrase):].strip()
                conf = 1.0 if not tail else 0.95
                log_debug(f"[CONTROL] Match WAKE: '{phrase}', tail='{tail}' (Priority 3)")
                return MatchResult(
                    command=ControlCommand.WAKE,
                    priority=CommandPriority.PRIORITY_3_WAKE.value,
                    matched_phrase=phrase,
                    confidence=conf,
                    original_text=raw_text,
                    cleaned_text=cleaned,
                    extracted_tail=tail if len(tail) >= 2 else None,
                )

        # ---------------- PRIORITY 4: NORMAL USER COMMAND ----------------
        return MatchResult(
            command=ControlCommand.NORMAL,
            priority=CommandPriority.PRIORITY_4_USER_COMMAND.value,
            matched_phrase="",
            confidence=0.0,
            original_text=raw_text,
            cleaned_text=cleaned,
        )


# Global Singleton Instance
_PHRASE_MATCHER: Optional[PhraseMatcher] = None


def get_phrase_matcher() -> PhraseMatcher:
    """Returns the application-wide PhraseMatcher instance."""
    global _PHRASE_MATCHER
    if _PHRASE_MATCHER is None:
        _PHRASE_MATCHER = PhraseMatcher()
    return _PHRASE_MATCHER
