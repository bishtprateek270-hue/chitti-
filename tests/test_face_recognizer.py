"""Tests for Chitti Face Recognizer."""

import numpy as np
import pytest
from src.vision.face_recognizer import (
    FaceRecognizer,
    TemporalIdentityTracker,
    cosine_similarity,
    l2_distance,
)
from src.vision.models import FaceDetection


def test_cosine_similarity_calculation():
    vec_a = [1.0, 0.0, 0.0]
    vec_b = [1.0, 0.0, 0.0]
    assert cosine_similarity(vec_a, vec_b) == 1.0

    vec_c = [0.0, 1.0, 0.0]
    assert cosine_similarity(vec_a, vec_c) == 0.0


def test_l2_distance_calculation():
    vec_a = [1.0, 0.0, 0.0]
    vec_b = [1.0, 0.0, 0.0]
    assert l2_distance(vec_a, vec_b) == 0.0

    vec_c = [0.0, 1.0, 0.0]
    # L2 distance between orthogonal unit vectors is sqrt(2) approx 1.414
    assert abs(l2_distance(vec_a, vec_c) - 1.4142) < 0.01


def test_match_identity_registered():
    recognizer = FaceRecognizer(cosine_threshold=0.60)

    # 2 registered people with 3-dimensional normalized embeddings
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


def test_multi_exemplar_gallery_matching():
    recognizer = FaceRecognizer(cosine_threshold=0.68)

    # Gallery contains multi-angle exemplars for Prateek
    gallery_payload = {
        "mean": [0.90, 0.10, 0.0],
        "exemplars": [
            [0.98, 0.02, 0.0],  # Frontal
            [0.85, 0.15, 0.0],  # Slight turn
            [0.92, 0.08, 0.0],  # Tilt
        ],
    }
    db = [(1, "Prateek", gallery_payload)]

    # Query matches the turned pose
    query_turned = [0.86, 0.14, 0.0]
    name, sim, is_known, pid = recognizer.match_identity(query_turned, db)
    assert is_known is True
    assert name == "Prateek"
    assert pid == 1
    assert sim >= 0.70


def test_face_quality_filtering():
    recognizer = FaceRecognizer(min_face_size=50)
    dummy_frame = np.zeros((320, 320, 3), dtype=np.uint8)

    # 1. Tiny face rejection (< 50x50)
    tiny_face = FaceDetection(bbox=(10, 10, 30, 30), confidence=0.90, landmarks=[(0, 0)] * 5)
    is_ok, reason = recognizer.check_face_quality(dummy_frame, tiny_face)
    assert is_ok is False
    assert "small" in reason.lower()

    # 2. Missing landmarks rejection
    no_landmark_face = FaceDetection(bbox=(10, 10, 80, 80), confidence=0.90, landmarks=None, raw_detection=None)
    is_ok, reason = recognizer.check_face_quality(dummy_frame, no_landmark_face)
    assert is_ok is False
    assert "landmarks" in reason.lower()


def test_ambiguity_margin_rejection():
    recognizer = FaceRecognizer(cosine_threshold=0.68)

    # Two people with competing vectors
    db = [
        (1, "PersonA", [0.72, 0.50, 0.48]),
        (2, "PersonB", [0.70, 0.52, 0.48]),
    ]

    # Query has similarity ~0.72 to PersonA and ~0.70 to PersonB (margin < 0.05 and score < 0.78)
    query = [1.0, 0.0, 0.0]
    name, sim, is_known, pid = recognizer.match_identity(query, db, strict_margin=True)
    # Ambiguous between PersonA and PersonB -> safely marks as Unknown
    assert is_known is False
    assert name == "Unknown"


def test_temporal_identity_tracker():
    tracker = TemporalIdentityTracker(window_size=4, min_consensus=2, min_avg_score=0.65)

    # Frame 1: Single trigger should NOT immediately declare known identity (anti-flicker)
    name, score, is_known = tracker.update("Prateek", 0.75, True)
    assert is_known is False
    assert name == "Unknown"

    # Frame 2: Second consistent detection confirms identity
    name, score, is_known = tracker.update("Prateek", 0.77, True)
    assert is_known is True
    assert name == "Prateek"
    assert score >= 0.75

    # Strangers appearing for 1 frame don't override consensus
    tracker.reset()
    name, score, is_known = tracker.update("Stranger", 0.72, True)
    assert is_known is False
