"""Tests for Chitti SQLite Memory Database."""

import pytest
from pathlib import Path
from src.memory.database import MemoryDatabase, MemoryRecord


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_memory.db"
    return MemoryDatabase(str(db_file))


def test_insert_and_get_memory(temp_db):
    record = MemoryRecord(
        content="User's favorite language is Python.",
        memory_type="preference",
        importance=4,
        embedding=[0.1, 0.2, 0.3],
        metadata={"source": "test"}
    )
    mem_id = temp_db.insert_memory(record)
    assert mem_id > 0

    retrieved = temp_db.get_memory(mem_id)
    assert retrieved is not None
    assert retrieved.id == mem_id
    assert retrieved.content == "User's favorite language is Python."
    assert retrieved.memory_type == "preference"
    assert retrieved.importance == 4
    assert retrieved.embedding == [0.1, 0.2, 0.3]
    assert retrieved.metadata == {"source": "test"}


def test_update_memory(temp_db):
    record = MemoryRecord(content="Initial project: DocForensics", memory_type="project")
    mem_id = temp_db.insert_memory(record)

    success = temp_db.update_memory(
        mem_id,
        content="Updated project: DocForensics AI",
        importance=5
    )
    assert success is True

    updated = temp_db.get_memory(mem_id)
    assert updated.content == "Updated project: DocForensics AI"
    assert updated.importance == 5


def test_delete_memory(temp_db):
    record = MemoryRecord(content="Temporary fact", memory_type="fact")
    mem_id = temp_db.insert_memory(record)

    assert temp_db.count() == 1
    deleted = temp_db.delete_memory(mem_id)
    assert deleted is True
    assert temp_db.count() == 0
    assert temp_db.get_memory(mem_id) is None


def test_delete_by_category(temp_db):
    temp_db.insert_memory(MemoryRecord(content="Project 1", memory_type="project"))
    temp_db.insert_memory(MemoryRecord(content="Project 2", memory_type="project"))
    temp_db.insert_memory(MemoryRecord(content="Fact 1", memory_type="fact"))

    deleted_count = temp_db.delete_by_category("project")
    assert deleted_count == 2
    assert temp_db.count() == 1

    remaining = temp_db.list_all_memories()
    assert remaining[0].memory_type == "fact"


def test_clear_all_memories(temp_db):
    temp_db.insert_memory(MemoryRecord(content="Fact 1"))
    temp_db.insert_memory(MemoryRecord(content="Fact 2"))
    assert temp_db.count() == 2

    cleared = temp_db.clear_all_memories()
    assert cleared == 2
    assert temp_db.count() == 0
