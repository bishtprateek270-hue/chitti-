"""Tests for Memory Manager End-to-End Orchestration & Identity Resolution (Phase 4A)."""

import pytest
from src.memory.manager import MemoryManager
from src.memory.extractor import ExtractedFact


@pytest.fixture
def manager(tmp_path):
    db_file = tmp_path / "test_manager.db"
    return MemoryManager(
        db_path=str(db_file),
        similarity_threshold=0.25,
    )


def test_remember_and_recall_flow(manager):
    success, msg, rec = manager.remember("My main AI project is DocForensics AI.", memory_type="project")
    assert success is True
    assert "remember" in msg.lower()

    results = manager.recall("What is my main AI project?")
    assert len(results) > 0
    assert "DocForensics AI" in results[0].content


def test_remember_multi_facts_and_identity_queries(manager):
    # Store multiple facts
    facts = [
        ExtractedFact(content="User's name is Prateek Singh Bisht.", memory_type="identity", key="user_name", value="Prateek Singh Bisht"),
        ExtractedFact(content="Prateek Singh Bisht is my creator (User created Chitti).", memory_type="relationship", key="creator", value="Prateek Singh Bisht"),
        ExtractedFact(content="User is an AIML engineer.", memory_type="professional_identity", key="occupation", value="AIML engineer"),
    ]
    success, msg, records = manager.remember_facts(facts)
    assert success is True
    assert len(records) == 3
    assert manager.db.count() == 3

    # Test user name retrieval
    assert manager.get_user_name() == "Prateek Singh Bisht"
    assert manager.get_creator_name() == "Prateek Singh Bisht"
    assert manager.get_user_occupation() == "AIML engineer"

    # Test identity queries in English
    name_resp_en = manager.resolve_identity_query("what is my name", lang="en")
    assert "Prateek Singh Bisht" in name_resp_en

    creator_resp_en = manager.resolve_identity_query("who created you?", lang="en")
    assert "Prateek Singh Bisht" in creator_resp_en

    occ_resp_en = manager.resolve_identity_query("what do I do?", lang="en")
    assert "AIML engineer" in occ_resp_en

    # Test identity queries in Hinglish / Hindi
    name_resp_hi = manager.resolve_identity_query("mera nam kya h", lang="hinglish")
    assert "Prateek Singh Bisht" in name_resp_hi
    assert "Tumhara naam" in name_resp_hi

    creator_resp_hi = manager.resolve_identity_query("tujhe kisne bnaya h", lang="hinglish")
    assert "Prateek Singh Bisht" in creator_resp_hi
    assert "Mujhe Prateek Singh Bisht ne banaya hai." in creator_resp_hi

    occ_resp_hi = manager.resolve_identity_query("main kya karta hoon", lang="hinglish")
    assert "AIML engineer" in occ_resp_hi


def test_identity_queries_when_unknown(manager):
    # When no identity has been registered yet
    name_resp = manager.resolve_identity_query("mera naam kya hai", lang="hinglish")
    assert "nahi mila" in name_resp or "yaad nahi" in name_resp
    assert "[Creator's Name]" not in name_resp

    creator_resp = manager.resolve_identity_query("who created you", lang="en")
    assert "don't have" in creator_resp.lower() or "not" in creator_resp.lower()
    assert "[Creator's Name]" not in creator_resp


def test_deduplication_updates_memory(manager):
    success1, msg1, rec1 = manager.remember("My favorite language is Python.")
    assert manager.db.count() == 1

    # Insert slight variant
    success2, msg2, rec2 = manager.remember("My favorite language is Python programming.")
    assert success2 is True
    assert "updated" in msg2.lower()
    # Count should still be 1 (updated, not duplicated)
    assert manager.db.count() == 1


def test_forget_by_query_flow(manager):
    manager.remember("I like drinking black coffee.")
    assert manager.db.count() == 1

    success, msg = manager.forget_by_query("black coffee")
    assert success is True
    assert "forgotten" in msg.lower()
    assert manager.db.count() == 0


def test_forget_all_confirmation_cycle(manager):
    manager.remember("User lives in London.", memory_type="personal")
    manager.remember("User is building a desktop companion robot named Chitti.", memory_type="project")
    assert manager.db.count() == 2

    # User asks to forget everything
    action, msg = manager.handle_interaction("Forget everything about me.")
    assert action == "confirm_required"
    assert "confirm" in msg.lower() or "yes or no" in msg.lower()
    assert manager.db.count() == 2

    # User confirms
    action_confirm, msg_confirm = manager.handle_interaction("Yes, please delete all.")
    assert action_confirm == "forget_all_confirmed"
    assert manager.db.count() == 0


def test_forget_all_cancellation(manager):
    manager.remember("Safe memory")
    assert manager.db.count() == 1

    manager.handle_interaction("Forget everything")
    action_cancel, msg_cancel = manager.handle_interaction("No, cancel that.")
    assert action_cancel == "cancelled"
    assert manager.db.count() == 1


def test_persistence_across_reloads(tmp_path):
    db_file = tmp_path / "persistent_test.db"

    # Session 1: write memory
    mgr1 = MemoryManager(db_path=str(db_file))
    mgr1.remember("DocForensics AI is a computer vision and deep learning project.")
    del mgr1

    # Session 2: reload and retrieve
    mgr2 = MemoryManager(db_path=str(db_file))
    recalled = mgr2.recall("DocForensics")
    assert len(recalled) > 0
    assert "DocForensics" in recalled[0].content


def test_auto_capture_interaction(manager):
    user_msg = "I really love drinking iced matcha lattes with oat milk."
    bot_resp = "Iced matcha latte with oat milk sounds delicious and refreshing!"

    facts, _ = manager.auto_capture_interaction(user_msg, bot_resp)
    assert len(facts) >= 1
    assert any("matcha" in f.content.lower() for f in facts)

    # Verify retrieval can recall what user likes
    recalled_pref = manager.recall("What do I like to drink?")
    assert len(recalled_pref) > 0
    assert any("matcha" in r.content.lower() for r in recalled_pref)


