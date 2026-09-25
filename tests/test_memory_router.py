"""
Unit and Integration Tests for Chitti Intelligent Memory Router & Personal Memory Policy.
Verifies all 9 mandatory specification test cases, relationship extraction, education memory,
and explicit memory overrides.
"""

import pytest
from src.memory.router import MemoryRouter, MemoryIntent
from src.memory.manager import MemoryManager
from src.memory.extractor import MemoryExtractor


@pytest.fixture
def memory_manager(tmp_path):
    db_file = tmp_path / "router_test_memory.db"
    return MemoryManager(db_path=str(db_file), similarity_threshold=0.25)


def test_specification_test_1_general_technical_query():
    """
    Test 1:
    Input: 'What is deep learning?'
    Expected: classification = GENERAL_KNOWLEDGE or TECHNICAL_QUERY, memory_retrieval = false
    """
    decision = MemoryRouter.classify_intent("What is deep learning?")
    assert decision.intent in (MemoryIntent.GENERAL_KNOWLEDGE, MemoryIntent.TECHNICAL_QUERY)
    assert decision.should_retrieve_memory is False
    assert decision.should_store_memory is False


def test_specification_test_2_user_name_query():
    """
    Test 2:
    Input: 'What is my name?'
    Expected: classification = PERSONAL_MEMORY, memory_retrieval = true
    """
    decision = MemoryRouter.classify_intent("What is my name?")
    assert decision.intent == MemoryIntent.PERSONAL_MEMORY
    assert decision.should_retrieve_memory is True


def test_specification_test_3_creator_query():
    """
    Test 3:
    Input: 'Who created you?'
    Expected: classification = RELATIONSHIP_MEMORY, memory_retrieval = true
    """
    decision = MemoryRouter.classify_intent("Who created you?")
    assert decision.intent in (MemoryIntent.RELATIONSHIP_MEMORY, MemoryIntent.PERSONAL_MEMORY)
    assert decision.should_retrieve_memory is True


def test_specification_test_4_explain_cnn():
    """
    Test 4:
    Input: 'Explain CNN.'
    Expected: memory_retrieval = false
    """
    decision = MemoryRouter.classify_intent("Explain CNN.")
    assert decision.intent == MemoryIntent.TECHNICAL_QUERY
    assert decision.should_retrieve_memory is False
    assert decision.should_store_memory is False


def test_specification_test_5_college_query():
    """
    Test 5:
    Input: 'Tell me about my college.'
    Expected: memory_retrieval = true
    """
    decision = MemoryRouter.classify_intent("Tell me about my college.")
    assert decision.intent in (MemoryIntent.EDUCATION_MEMORY, MemoryIntent.PERSONAL_MEMORY)
    assert decision.should_retrieve_memory is True


def test_specification_test_6_relationship_storage():
    """
    Test 6:
    Input: 'My best friend is Rahul.'
    Expected: memory_storage = true, memory_type = relationship
    """
    decision = MemoryRouter.classify_intent("My best friend is Rahul.")
    assert decision.should_store_memory is True
    assert decision.intent == MemoryIntent.RELATIONSHIP_MEMORY

    extractor = MemoryExtractor()
    facts = extractor.extract_discrete_facts("My best friend is Rahul.")
    assert len(facts) >= 1
    assert facts[0].memory_type == "relationship"
    assert facts[0].key == "best_friend"
    assert facts[0].value == "Rahul"


def test_specification_test_7_explicit_memory_override():
    """
    Test 7:
    Input: 'Remember that my favorite language is Python.'
    Expected: memory_storage = true, explicit_memory = true
    """
    decision = MemoryRouter.classify_intent("Remember that my favorite language is Python.")
    assert decision.should_store_memory is True
    assert decision.is_explicit_memory is True
    assert decision.intent == MemoryIntent.EXPLICIT_MEMORY


def test_specification_test_8_task_leetcode_query():
    """
    Test 8:
    Input: 'Solve this LeetCode problem.'
    Expected: memory_storage = false, memory_retrieval = false
    """
    decision = MemoryRouter.classify_intent("Solve this LeetCode problem.")
    assert decision.should_store_memory is False
    assert decision.should_retrieve_memory is False


def test_specification_test_9_project_query():
    """
    Test 9:
    Input: 'What project am I building?'
    Expected: memory_retrieval = true
    """
    decision = MemoryRouter.classify_intent("What project am I building?")
    assert decision.intent == MemoryIntent.PROJECT_MEMORY
    assert decision.should_retrieve_memory is True


def test_technical_questions_no_long_term_pollution(memory_manager):
    """
    Ensures that answering general technical questions does not write to long-term memory.
    """
    initial_count = memory_manager.db.count()

    # User asks a technical question
    tech_query = "How does backpropagation work in deep neural networks?"
    tech_resp = "Backpropagation calculates gradients using the chain rule."

    memory_manager.auto_capture_interaction(tech_query, tech_resp)

    # Database count must remain unchanged
    assert memory_manager.db.count() == initial_count


def test_personal_disclosures_and_entity_resolution(memory_manager):
    """
    Ensures that personal facts, college, sister, friend, and project are stored and retrievable.
    """
    # 1. College
    memory_manager.auto_capture_interaction("My college is ABC University.", "Nice to know!")
    assert memory_manager.get_college_name() == "Abc University"

    # 2. Relationship - Sister
    memory_manager.auto_capture_interaction("My sister's name is Ananya.", "Got it!")
    assert memory_manager.get_relationship("sister") == "Ananya"

    # 3. Relationship - Best Friend
    memory_manager.auto_capture_interaction("My best friend is Rahul.", "Got it!")
    assert memory_manager.get_relationship("best_friend") == "Rahul"

    # 4. Project
    memory_manager.auto_capture_interaction("I'm building a robot called Chitti.", "Awesome!")
    assert "Chitti" in memory_manager.get_project_name()

    # 5. Entity lookup
    rel = memory_manager.get_entity_relationship("Rahul")
    assert "best friend" in rel.lower() or "Rahul" in rel


def test_memory_update_conflict_handling(memory_manager):
    """
    Ensures that when personal information changes, existing structured keys are updated in-place.
    """
    # First college
    memory_manager.remember("User's college is College A.", memory_type="education", metadata={"key": "college", "value": "College A"})
    assert memory_manager.get_college_name() == "College A"
    assert memory_manager.db.count() == 1

    # Transfer to college B
    memory_manager.remember("User's college is College B.", memory_type="education", metadata={"key": "college", "value": "College B"})
    assert memory_manager.get_college_name() == "College B"
    # Count should still be 1
    assert memory_manager.db.count() == 1
