"""Tests for Memory Manager End-to-End Orchestration."""

import pytest
from src.memory.manager import MemoryManager


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
    assert manager.db.count() == 2  # Not deleted yet!

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
    assert manager.db.count() == 1  # Intact!


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
