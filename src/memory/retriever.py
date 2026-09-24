"""
Chitti Semantic Memory Retrieval Module.
Provides embedding generation, cosine vector similarity calculation, multi-factor ranking, and deduplication.
"""

import math
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

from src.memory.database import MemoryRecord, MemoryDatabase
from src.utils.logging import log_debug, log_warning


class BaseEmbeddingEngine(ABC):
    """Abstract Base Class for text embedding engines."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generates a dense vector embedding for the given text."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a batch of texts."""
        pass


class SentenceTransformerEmbedding(BaseEmbeddingEngine):
    """Local SentenceTransformer embedding model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            log_debug(f"Loading SentenceTransformer embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        except Exception as e:
            log_warning(f"Failed to load SentenceTransformer ({e}). Using lightweight local fallback.")
            self._model = None

    def embed_text(self, text: str) -> List[float]:
        self._load()
        if self._model is not None:
            try:
                emb = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
                return emb.tolist()
            except Exception as e:
                log_warning(f"SentenceTransformer encoding failed: {e}")
        # Fallback to local deterministic feature embedding
        return FallbackFeatureEmbedding().embed_text(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        self._load()
        if self._model is not None:
            try:
                embs = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
                return embs.tolist()
            except Exception:
                pass
        fallback = FallbackFeatureEmbedding()
        return [fallback.embed_text(t) for t in texts]


class FallbackFeatureEmbedding(BaseEmbeddingEngine):
    """
    Lightweight, deterministic hashing-based n-gram embedding fallback.
    Guarantees 100% offline functionality without external downloads.
    """

    def __init__(self, dim: int = 128):
        self.dim = dim

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            return [0.0] * self.dim

        tokens = text.lower().strip().split()
        vec = np.zeros(self.dim, dtype=np.float32)

        for token in tokens:
            # Word unigram
            idx = abs(hash(token)) % self.dim
            vec[idx] += 1.0
            # Character trigrams for morphological similarity
            if len(token) >= 3:
                for i in range(len(token) - 2):
                    trigram = token[i:i+3]
                    t_idx = abs(hash(trigram)) % self.dim
                    vec[t_idx] += 0.5

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]


def get_embedding_engine(model_name: str = "all-MiniLM-L6-v2") -> BaseEmbeddingEngine:
    """Factory function for embedding engines."""
    return SentenceTransformerEmbedding(model_name)


def cosine_similarity(vec_a: Union[List[float], np.ndarray], vec_b: Union[List[float], np.ndarray]) -> float:
    """Computes cosine similarity between two numeric vectors."""
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class MemoryRetriever:
    """Retrieves and ranks relevant memories using semantic similarity and metadata scoring."""

    def __init__(
        self,
        db: MemoryDatabase,
        embedding_engine: Optional[BaseEmbeddingEngine] = None,
        similarity_threshold: float = 0.35,
        importance_threshold: int = 1,
    ):
        self.db = db
        self.embedding_engine = embedding_engine or get_embedding_engine()
        self.similarity_threshold = similarity_threshold
        self.importance_threshold = importance_threshold

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        memory_type: Optional[str] = None,
        min_similarity: Optional[float] = None,
    ) -> List[Tuple[MemoryRecord, float]]:
        """
        Searches memories semantically and returns the top_k ranked records with their combined scores.
        """
        threshold = min_similarity if min_similarity is not None else self.similarity_threshold
        if not query or not query.strip():
            return []

        all_memories = self.db.list_all_memories(memory_type=memory_type)
        if not all_memories:
            return []

        query_emb = self.embedding_engine.embed_text(query)
        scored_results: List[Tuple[MemoryRecord, float, float]] = []

        now = datetime.now(timezone.utc)

        for memory in all_memories:
            if memory.importance < self.importance_threshold:
                continue

            # Check if embedding exists or generate it on the fly
            if memory.embedding is None or len(memory.embedding) == 0:
                memory.embedding = self.embedding_engine.embed_text(memory.content)
                self.db.update_memory(memory.id, embedding=memory.embedding)

            sim = cosine_similarity(query_emb, memory.embedding)
            if sim < threshold:
                continue

            # Calculate Recency Score (Exponential decay over 30 days)
            recency_score = 0.5
            try:
                created = datetime.fromisoformat(memory.created_at)
                days_old = max(0.0, (now - created).total_seconds() / 86400.0)
                recency_score = math.exp(-days_old / 30.0)
            except Exception:
                pass

            # Importance normalized to 0.0 - 1.0
            norm_importance = float(memory.importance) / 5.0

            # Combined Priority: Similarity (70%) + Importance (20%) + Recency (10%)
            combined_score = (sim * 0.70) + (norm_importance * 0.20) + (recency_score * 0.10)
            scored_results.append((memory, combined_score, sim))

        # Sort by combined score descending
        scored_results.sort(key=lambda x: x[1], reverse=True)

        return [(item[0], item[1]) for item in scored_results[:top_k]]

    def find_duplicate(
        self,
        content: str,
        threshold: float = 0.90
    ) -> Optional[Tuple[MemoryRecord, float]]:
        """
        Checks if a semantically equivalent memory already exists in the database.
        Returns (existing_record, similarity) if duplicate found, otherwise None.
        """
        if not content or not content.strip():
            return None

        all_memories = self.db.list_all_memories()
        if not all_memories:
            return None

        # Exact match fast path
        clean_target = content.strip().lower()
        for m in all_memories:
            if m.content.strip().lower() == clean_target:
                return (m, 1.0)

        # Semantic similarity check
        target_emb = self.embedding_engine.embed_text(content)
        best_match = None
        best_sim = 0.0

        for memory in all_memories:
            if memory.embedding is None:
                memory.embedding = self.embedding_engine.embed_text(memory.content)
                self.db.update_memory(memory.id, embedding=memory.embedding)

            sim = cosine_similarity(target_emb, memory.embedding)
            if sim > best_sim:
                best_sim = sim
                best_match = memory

        if best_match and best_sim >= threshold:
            return (best_match, best_sim)
        return None
