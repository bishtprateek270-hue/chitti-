"""Tests for SQLite Face Database."""

import pytest
from src.vision.face_database import FaceDatabase


@pytest.fixture
def face_db(tmp_path):
    db_file = tmp_path / "test_faces.db"
    return FaceDatabase(str(db_file))


def test_register_and_get_face(face_db):
    emb = [0.1] * 128
    pid = face_db.register_or_update_face("Prateek", emb, sample_count=3)
    assert pid > 0

    record = face_db.get_face_by_name("Prateek")
    assert record is not None
    assert record["name"] == "Prateek"
    assert record["sample_count"] == 3
    assert len(record["embedding"]) == 128


def test_update_existing_face(face_db):
    emb1 = [0.1] * 128
    pid1 = face_db.register_or_update_face("Prateek", emb1, sample_count=1)

    emb2 = [0.2] * 128
    pid2 = face_db.register_or_update_face("Prateek", emb2, sample_count=5)

    assert pid1 == pid2  # Same ID updated
    assert face_db.count() == 1

    record = face_db.get_face_by_name("Prateek")
    assert record["sample_count"] == 5
    assert record["embedding"] == emb2


def test_list_and_delete_face(face_db):
    face_db.register_or_update_face("Prateek", [0.1] * 128)
    face_db.register_or_update_face("Ayush", [0.2] * 128)

    names = face_db.list_all_names()
    assert "Ayush" in names
    assert "Prateek" in names

    deleted = face_db.delete_face("Ayush")
    assert deleted is True
    assert face_db.count() == 1
    assert "Ayush" not in face_db.list_all_names()


def test_clear_all_faces(face_db):
    face_db.register_or_update_face("Person1", [0.1] * 128)
    face_db.register_or_update_face("Person2", [0.2] * 128)
    assert face_db.count() == 2

    cleared = face_db.clear_all_faces()
    assert cleared == 2
    assert face_db.count() == 0
