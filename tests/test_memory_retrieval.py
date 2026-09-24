"""Tests for Semantic Memory Retrieval and Deduplication."""

import pytest
import numpy as np
from src.memory.database import MemoryDatabase, MemoryRecord
from src.memory.retriever import (
    MemoryRetriever,
    FallbackFeatureEmbedding,
    cosine_similarity,
)


@pytest.fixture
def retriever_with_db(tmp_path):
    db_file = tmp_path / "test_retrieval.db"
    db = MemoryDatabase(str(db_file))
    embedding_engine = FallbackFeatureEmbedding(dim=128)
    retriever = MemoryRetriever(
        db=db,
        embedding_engine=embedding_engine,
        similarity_threshold=0.20,
    )
    return db, retriever


def test_cosine_similarity_identical_vectors():
    vec_a = [1.0, 2.0, 3.0]
    vec_b = [1.0, 2.0, 3.0]
    sim = cosine_similarity(vec_a, vec_b)
    assert pytest.approx(sim, 0.001) == 1.0


def test_cosine_similarity_orthogonal_vectors():
    vec_a = [1.0, 0.0]
    vec_b = [0.0, 1.0]
    sim = cosine_similarity(vec_a, vec_b)
    assert pytest.approx(sim, 0.001) == 0.0


def test_semantic_retrieval_ranking(retriever_with_db):
    db, retriever = retriever_with_db

    # Insert test memories
    rec1 = MemoryRecord(
        content="My primary AI project is DocForensics AI.",
        memory_type="project",
        importance=5
    )
    rec2 = MemoryRecord(
        content="I prefer tea over coffee in the morning.",
        memory_type="preference",
        importance=2
    )
    rec3 = MemoryRecord(
        content="DocForensics AI uses PyTorch and deep learning.",
        memory_type="project",
        importance=4
    )

    db.insert_memory(rec1)
    db.insert_memory(rec2)
    db.insert_memory(rec3)

    results = retriever.retrieve("What AI project am I building?", top_k=2)
    assert len(results) > 0
    top_memory = results[0][0]
    assert "DocForensics" in top_memory.content


def test_duplicate_detection(retriever_with_db):
    db, retriever = retriever_with_db

    rec = MemoryRecord(content="My favorite programming language is Python.", memory_type="preference")
    db.insert_memory(rec)

    # Search duplicate with exact and similar wording
    dup_exact = retriever.find_duplicate("My favorite programming language is Python.", threshold=0.8)
    assert dup_exact is not None
    assert dup_exact[0].id == rec.id

    dup_similar = retriever.find_duplicate("My favourite programming language is Python", threshold=0.7)
    assert dup_similar is not None
    assert dup_similar[0].id == rec.id

    dup_different = retriever.find_duplicate("I love swimming in summer", threshold=0.7)
    assert dup_different is None
