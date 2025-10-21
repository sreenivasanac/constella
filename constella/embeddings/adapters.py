"""Embedding providers implemented via LiteLLM."""

from __future__ import annotations

import logging
import os
from typing import List, Sequence

try:  # pragma: no cover - optional dependency
    from litellm import embedding
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "LiteLLM is required for LiteLLMEmbeddingProvider. Install constella with the 'llm' extra."
    ) from exc

from constella.embeddings.base import EmbeddingProvider


LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = "text-embedding-3-small"


class LiteLLMEmbeddingProvider(EmbeddingProvider):
    """Generates embeddings through LiteLLM using OpenAI-compatible endpoints."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY must be set for LiteLLM embedding provider.")

        # TODO update this to have an intelligent way to calculate batch size
        # such that max of 30,000 tokens per embedding call.
        batch_size = max(1, (len(texts) + 9) // 13)
        all_embeddings: List[List[float]] = []

        try:
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start : start + batch_size])
                response = embedding(model=self.model, input=batch)
                all_embeddings.extend(list(item["embedding"]) for item in response["data"])
            return all_embeddings
        except Exception as exc:  # pragma: no cover - upstream errors vary
            LOGGER.error("Embedding request failed: %s", exc)
            raise
