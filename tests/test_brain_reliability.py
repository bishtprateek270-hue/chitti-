"""
Comprehensive tests for Phase 4A: Brain Reliability, Memory Integration, Identity & Response Quality.
Tests exact user scenarios:
- Learning compound identities (Name + Creator + Profession)
- Identity & Creator question answering in English, Hindi, and Hinglish
- Zero placeholder leakage
- Anti-hallucination on unknown facts
- Grammar and response sanitization
"""

import pytest
from unittest.mock import MagicMock

from src.main import ChittiController
from src.memory.manager import MemoryManager
from src.memory.extractor import MemoryExtractor
from src.brain.validator import ResponseValidator
from src.audio.stt import WhisperSTT


@pytest.fixture
def test_controller(tmp_path):
    """Creates a mock-backed ChittiController with an isolated temporary memory database."""
    controller = ChittiController()
    db_file = tmp_path / "test_brain_memory.db"
    controller.memory = MemoryManager(db_path=str(db_file), similarity_threshold=0.25)
    controller.tts = MagicMock()
    controller.speak = MagicMock()
    controller.llm = MagicMock()
    controller.llm.generate_response.return_value = "This is a generic LLM response."
    return controller


def test_learn_compound_identity_statement(test_controller):
    """
    Test user scenario:
    'my name is Prateek Singh Bisht and I created you, also I'm an AIML engineer. Remember this.'
    """
    input_text = "my name is Prateek Singh Bisht and I created you, also I'm an AIML engineer. Remember this."
    test_controller.process_user_input(input_text)

    # Verify that discrete facts are stored in memory
    assert test_controller.memory.db.count() == 3
    assert test_controller.memory.get_user_name() == "Prateek Singh Bisht"
    assert test_controller.memory.get_creator_name() == "Prateek Singh Bisht"
    assert test_controller.memory.get_user_occupation() == "AIML engineer"


def test_answer_identity_and_creator_queries(test_controller):
    """
    Tests answering questions about identity, creator, and profession after learning.
    """
    # 1. Teach Chitti
    test_controller.process_user_input("my name is Prateek Singh Bisht and I created you, also I'm an AIML engineer. Remember this.")

    # 2. Ask "mera nam kya h" (Hinglish)
    resp_name_hi = test_controller.memory.resolve_identity_query("mera nam kya h", lang="hinglish")
    assert resp_name_hi == "Tumhara naam Prateek Singh Bisht hai."

    # 3. Ask "who create u" / "who created you" (English)
    resp_creator_en = test_controller.memory.resolve_identity_query("who create u", lang="en")
    assert "Prateek Singh Bisht" in resp_creator_en
    assert "[Creator's Name]" not in resp_creator_en

    # 4. Ask "tujhe kisne bnaya h" (Hinglish)
    resp_creator_hi = test_controller.memory.resolve_identity_query("tujhe kisne bnaya h", lang="hinglish")
    assert resp_creator_hi == "Mujhe Prateek Singh Bisht ne banaya hai."
    assert "[Creator's Name]" not in resp_creator_hi

    # 5. Ask "main kya karta hoon" (Hinglish)
    resp_occ_hi = test_controller.memory.resolve_identity_query("main kya karta hoon", lang="hinglish")
    assert resp_occ_hi == "Tum AIML engineer ho."


def test_unknown_identity_when_not_remembered(test_controller):
    """
    If no identity or creator is stored, Chitti must NEVER hallucinate or output [Creator's Name].
    """
    resp_creator = test_controller.memory.resolve_identity_query("who created you", lang="en")
    assert "[Creator's Name]" not in resp_creator
    assert "don't have" in resp_creator.lower() or "not" in resp_creator.lower()

    resp_name = test_controller.memory.resolve_identity_query("mera naam kya hai", lang="hinglish")
    assert "[User Name]" not in resp_name
    assert "nahi mila" in resp_name


def test_response_validator_placeholder_elimination():
    """
    Tests that ResponseValidator strips any accidental placeholders from LLM outputs.
    """
    raw_llm_out = "Hello [Creator's Name], I am ready to help you."
    sanitized = ResponseValidator.sanitize(raw_llm_out, creator_name="Prateek Singh Bisht", target_lang="en")
    assert sanitized == "Hello Prateek Singh Bisht, I am ready to help you."
    assert "[Creator's Name]" not in sanitized

    raw_llm_out_unknown = "My creator is [Creator's Name]."
    sanitized_unknown = ResponseValidator.sanitize(raw_llm_out_unknown, creator_name=None, target_lang="hinglish")
    assert "[Creator's Name]" not in sanitized_unknown
    assert "Mujhe abhi mere creator ki information memory mein nahi mili." in sanitized_unknown


def test_response_validator_grammar_glitch_repair():
    """
    Tests that known grammatical glitches are repaired automatically.
    """
    glitched = "Main Chitti bana hai, aapki desktop companion robot hoon."
    sanitized = ResponseValidator.sanitize(glitched, target_lang="hinglish")
    assert "Main Chitti hoon" in sanitized
    assert "Main Chitti bana hai" not in sanitized


def test_stt_hallucination_filter():
    """
    Tests that Whisper hallucination artifacts are caught and filtered.
    """
    assert WhisperSTT.is_hallucination("Humans are so hard. Oh Julian") is True
    assert WhisperSTT.is_hallucination("Thank you for watching!") is True
    assert WhisperSTT.is_hallucination("[Music]") is True
    assert WhisperSTT.is_hallucination("who created you") is False
    assert WhisperSTT.is_hallucination("mera naam kya hai") is False
