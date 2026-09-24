"""Tests for Chitti Face Recognizer."""

import numpy as np
import pytest
from src.vision.face_recognizer import FaceRecognizer, cosine_similarity
from src.vision.models import FaceDetection


def test_cosine_similarity_calculation():
    vec_a = [1.0, 0.0, 0.0]
    vec_b = [1.0, 0.0, 0.0]
    assert cosine_similarity(vec_a, vec_b) == 1.0

    vec_c = [0.0, 1.0, 0.0]
    assert cosine_similarity(vec_a, vec_c) == 0.0


def test_match_identity_registered():
    recognizer = FaceRecognizer(cosine_threshold=0.60)

    # 3 registered people with 3-dimensional normalized embeddings
    db = [
        (1, "Prateek", [1.0, 0.0, 0.0]),
        (2, "Ayush", [0.0, 1.0, 0.0]),
    ]

    # Query embedding close to Prateek
    query_prateek = [0.95, 0.05, 0.0]
    name, sim, is_known, pid = recognizer.match_identity(query_prateek, db)
    assert is_known is True
    assert name == "Prateek"
    assert pid == 1
    assert sim > 0.90


def test_match_identity_unknown_when_below_threshold():
    recognizer = FaceRecognizer(cosine_threshold=0.60)

    db = [
        (1, "Prateek", [1.0, 0.0, 0.0]),
        (2, "Ayush", [0.0, 1.0, 0.0]),
    ]

    # Query embedding orthogonal / distinct
    query_unknown = [0.0, 0.0, 1.0]
    name, sim, is_known, pid = recognizer.match_identity(query_unknown, db)
    assert is_known is False
    assert name == "Unknown"
    assert pid is None
    assert sim < 0.60
