"""Embedding providers implemented via LiteLLM."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import List, Sequence, Tuple

import litellm

from count_tokens import count_tokens_in_string
from count_tokens.count import CHARACTERS_PER_TOKEN, TOKENS_PER_WORD

from constella.embeddings.base import EmbeddingProvider


LOGGER = logging.getLogger(__name__)

DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
DEFAULT_FIREWORKS_MODEL = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_FIREWORKS_API_BASE = "https://api.fireworks.ai/inference/v1"

DEFAULT_MAX_TOKENS_PER_BATCH = 290_000
DEFAULT_CONCURRENCY = max(1, min(8, (os.cpu_count() or 4)))


class LiteLLMEmbeddingBaseProvider(EmbeddingProvider):
    """Generates embeddings through LiteLLM using OpenAI-compatible endpoints."""

    def __init__(
        self,
        model: str,
        api_base_url: str | None = None
    ):
        self.model = model
        self.api_base_url = api_base_url
        self._max_concurrency = max(1, DEFAULT_CONCURRENCY)
        self.max_tokens_per_batch = DEFAULT_MAX_TOKENS_PER_BATCH

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []

        self._ensure_credentials()

        tokenized_batches = self._build_batches(texts)
        if len(tokenized_batches) == 1:
            return self._embed_batch(tokenized_batches[0][1])

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

    def _ensure_credentials(self) -> None:
        raise NotImplementedError

    def _embedding_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {}
        if self.api_base_url:
            kwargs["api_base"] = self.api_base_url
        return kwargs

    def _embed_batch(self, batch: Sequence[str]) -> List[List[float]]:
        response = litellm.embedding(
            model=self.model,
            input=list(batch),
            **self._embedding_kwargs(),
        )
        return [list(item["embedding"]) for item in response["data"]]

    def _build_batches(self, texts: Sequence[str]) -> List[Tuple[int, List[str]]]:
        batches: List[Tuple[int, List[str]]] = []
        current_batch: List[str] = []
        current_tokens = 0

        for idx, text in enumerate(texts):
            tokens = self._count_tokens(text)

            if tokens > self.max_tokens_per_batch:
                if current_batch:
                    batches.append((current_tokens, current_batch))
                    current_batch = []
                    current_tokens = 0
                LOGGER.warning(
                    "Text at index %s exceeds max tokens per batch (%s); sending alone.",
                    idx,
                    self.max_tokens_per_batch,
                )
                batches.append((tokens, [text]))
                continue

            if current_batch and current_tokens + tokens > self.max_tokens_per_batch:
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
            return max(1, count_tokens_in_string(text))

        approx_by_chars = math.ceil(len(text) / CHARACTERS_PER_TOKEN) if CHARACTERS_PER_TOKEN else 0
        approx_by_words = math.ceil(len(text.split()) * TOKENS_PER_WORD)
        approximation = max(approx_by_chars, approx_by_words)
        return max(1, approximation)


class LiteLLMEmbeddingOpenAIProvider(LiteLLMEmbeddingBaseProvider):
    def __init__(
        self,
        model: str = DEFAULT_OPENAI_MODEL,
        api_base_url: str | None = None
    ):
        super().__init__(
            model=model,
            api_base_url=api_base_url
        )

    def _ensure_credentials(self) -> None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY must be set for LiteLLM OpenAI embedding provider.")

    def _embedding_kwargs(self) -> dict[str, object]:
        kwargs = super()._embedding_kwargs()
        kwargs["api_key"] = os.environ["OPENAI_API_KEY"]
        return kwargs


class LiteLLMEmbeddingFireworksProvider(LiteLLMEmbeddingBaseProvider):
    def __init__(
        self,
        model: str = DEFAULT_FIREWORKS_MODEL,
        api_base_url: str | None = DEFAULT_FIREWORKS_API_BASE
    ):
        super().__init__(
            model=model,
            api_base_url=api_base_url
        )

    def _ensure_credentials(self) -> None:
        if not (os.environ.get("FIREWORKS_AI_API_KEY") or os.environ.get("FIREWORKS_API_KEY")):
            raise RuntimeError("Set FIREWORKS_AI_API_KEY or FIREWORKS_API_KEY before running this provider.")

    def _embedding_kwargs(self) -> dict[str, object]:
        kwargs = super()._embedding_kwargs()
        api_key = os.environ.get("FIREWORKS_AI_API_KEY") or os.environ.get("FIREWORKS_API_KEY")
        kwargs["api_key"] = api_key
        kwargs["custom_llm_provider"] = "openai"
        kwargs["api_base"] = self.api_base_url
        return kwargs

    def _build_batches(self, texts: Sequence[str]) -> List[Tuple[int, List[str]]]:
        batches: List[Tuple[int, List[str]]] = []
        batch_size = 256

        for start in range(0, len(texts), batch_size):
            batch = list(texts[start:start + batch_size])
            token_count = sum(self._count_tokens(text) for text in batch)
            batches.append((token_count, batch))
        return batches


__all__ = [
    "LiteLLMEmbeddingBaseProvider",
    "LiteLLMEmbeddingOpenAIProvider",
    "LiteLLMEmbeddingFireworksProvider",
]
