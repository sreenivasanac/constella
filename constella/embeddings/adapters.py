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

        try:
            response = embedding(model=self.model, input=list(texts))
            return [list(item["embedding"]) for item in response["data"]]
        except Exception as exc:  # pragma: no cover - upstream errors vary
            LOGGER.error("Embedding request failed: %s", exc)
            raise
