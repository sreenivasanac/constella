"""Interfaces and utility providers for text embeddings."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Sequence


class EmbeddingProvider(ABC):
    """Interface for embedding providers used by the workflow."""

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """Return embeddings for the supplied texts."""


class InMemoryEmbeddingProvider(EmbeddingProvider):
    """Simple embedding provider returning supplied vectors, useful for tests."""

    def __init__(self, vectors: Iterable[Sequence[float]]):
        self._vectors = [list(vector) for vector in vectors]

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if len(texts) != len(self._vectors):
            raise ValueError("Provided vectors length must match texts length.")
        return [list(vector) for vector in self._vectors]
