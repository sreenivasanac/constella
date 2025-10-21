"""Embedding providers implemented via LiteLLM."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import List, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    from litellm import embedding
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "LiteLLM is required for LiteLLMEmbeddingProvider. Install constella with the 'llm' extra."
    ) from exc

try:  # pragma: no cover - optional dependency
    from count_tokens import count_tokens_in_string
    from count_tokens.count import CHARACTERS_PER_TOKEN, TOKENS_PER_WORD
except Exception:  # pragma: no cover - importing may fail in unsupported envs
    count_tokens_in_string = None
    TOKENS_PER_WORD = 4.0 / 3.0
    CHARACTERS_PER_TOKEN = 4.0

from constella.embeddings.base import EmbeddingProvider


LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = "text-embedding-3-small"

# LiteLLM / OpenAI Embeddings endpoint maximum tokens per call value
MAX_TOKENS_PER_BATCH = 290_000
DEFAULT_CONCURRENCY = max(1, min(8, (os.cpu_count() or 4)))


class LiteLLMEmbeddingProvider(EmbeddingProvider):
    """Generates embeddings through LiteLLM using OpenAI-compatible endpoints."""

    def __init__(self, model: str = DEFAULT_MODEL, max_concurrency: int | None = None):
        self.model = model
        self._max_concurrency = max(1, max_concurrency or DEFAULT_CONCURRENCY)

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY must be set for LiteLLM embedding provider.")

        tokenized_batches = self._build_batches(texts)
        if len(tokenized_batches) == 1:
            batch_embeddings = self._embed_batch(tokenized_batches[0][1])
            return batch_embeddings

        async def _run_batches() -> List[List[List[float]]]:
            max_workers = min(self._max_concurrency, len(tokenized_batches)) or 1
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                tasks = [
                    loop.run_in_executor(executor, self._embed_batch, batch_texts)
                    for _, batch_texts in tokenized_batches
                ]
                return await asyncio.gather(*tasks)

        try:
            try:
                batch_results = asyncio.run(_run_batches())
            except RuntimeError as exc:
                if "running event loop" in str(exc).lower():
                    loop = asyncio.new_event_loop()
                    try:
                        asyncio.set_event_loop(loop)
                        batch_results = loop.run_until_complete(_run_batches())
                    finally:
                        loop.run_until_complete(loop.shutdown_asyncgens())
                        asyncio.set_event_loop(None)
                        loop.close()
                else:  # pragma: no cover - unexpected runtime errors
                    raise
        except Exception as exc:  # pragma: no cover - upstream errors vary
            LOGGER.error("Embedding request failed: %s", exc)
            raise

        ordered_embeddings: List[List[float]] = []
        for (_, batch_texts), batch_embeddings in zip(tokenized_batches, batch_results):
            if len(batch_texts) != len(batch_embeddings):  # pragma: no cover - upstream mismatch
                raise RuntimeError("Embedding batch size mismatch from provider response.")
            ordered_embeddings.extend(batch_embeddings)
        return ordered_embeddings

    def _embed_batch(self, batch: Sequence[str]) -> List[List[float]]:
        response = embedding(model=self.model, input=list(batch))
        return [list(item["embedding"]) for item in response["data"]]

    def _build_batches(self, texts: Sequence[str]) -> List[Tuple[int, List[str]]]:
        batches: List[Tuple[int, List[str]]] = []
        current_batch: List[str] = []
        current_tokens = 0

        for idx, text in enumerate(texts):
            tokens = self._count_tokens(text)

            if tokens > MAX_TOKENS_PER_BATCH:
                if current_batch:
                    batches.append((current_tokens, current_batch))
                    current_batch = []
                    current_tokens = 0
                LOGGER.warning(
                    "Text at index %s exceeds max tokens per batch (%s); sending alone.",
                    idx,
                    MAX_TOKENS_PER_BATCH,
                )
                batches.append((tokens, [text]))
                continue

            if current_batch and current_tokens + tokens > MAX_TOKENS_PER_BATCH:
                batches.append((current_tokens, current_batch))
                current_batch = []
                current_tokens = 0

            current_batch.append(text)
            current_tokens += tokens

        if current_batch:
            batches.append((current_tokens, current_batch))

        if not batches and texts:
            total_tokens = sum(self._count_tokens(text) for text in texts)
            batches.append((total_tokens, list(texts)))

        return batches

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if count_tokens_in_string and sys.version_info[:2] < (3, 14):
            try:
                return max(1, count_tokens_in_string(text))
            except Exception:  # pragma: no cover - fall back to heuristic
                LOGGER.debug("Exact token counting failed; falling back to heuristic.")
        words = text.split()
        if words:
            return max(1, math.ceil(len(words) * TOKENS_PER_WORD))
        return max(1, math.ceil(len(text) / CHARACTERS_PER_TOKEN))
