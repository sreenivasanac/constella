"""LLM-backed auto labeling utilities."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import litellm
from pydantic import BaseModel, Field, ValidationError

from constella.labeling.selection import select_representatives
from constella.config.schemas import (
    LabelingConfig,
    RepresentativeSelectionConfig,
)
from constella.data.models import ContentUnitCollection
from constella.data.results import LabelResult, RepresentativeSample


LOGGER = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = (
    "You are an analyst categorizing large collections of user content units. "
    "Each cluster groups semantically similar items, and you will only see a "
    "handful of representative samples. Infer a concise, human-readable label "
    "that captures the shared topic."
)



class LabelPayload(BaseModel):
    label: str
    explanation: str
    confidence: float
    keywords: List[str] = Field(default_factory=list)

    def to_label_result(
        self,
        *,
        cluster_id: int,
        raw_response: Optional[str],
        usage: Optional[Dict[str, Any]],
    ) -> LabelResult:
        bounded_confidence = float(min(1.0, max(0.0, self.confidence)))
        keywords = [item.strip() for item in self.keywords if item and item.strip()]
        return LabelResult(
            cluster_id=cluster_id,
            label=self.label.strip() or f"Cluster {cluster_id}",
            explanation=self.explanation.strip() or "Automatic labeling failed",
            confidence=bounded_confidence,
            keywords=keywords,
            raw_response=raw_response,
            usage_metadata=usage,
        )


def auto_label_clusters(
    collection: ContentUnitCollection,
    config: Optional[LabelingConfig] = None,
    selection_config: Optional[RepresentativeSelectionConfig] = None,
) -> ContentUnitCollection:
    """Populate label results on the collection using the provided configuration."""

    if len(collection) == 0:
        collection.label_results = {}
        return collection

    resolved_config = config or LabelingConfig()
    resolved_selection = selection_config or RepresentativeSelectionConfig()

    representatives = select_representatives(collection, resolved_selection)
    if not representatives:
        LOGGER.warning("No representatives available for labeling; skipping LLM calls.")
        collection.label_results = {}
        return collection

    cluster_sizes = collection.cluster_size_lookup
    provider_kwargs = _provider_kwargs(resolved_config.llm_provider)

    jobs: List[Tuple[int, List[RepresentativeSample]]] = []
    for cluster_id, samples in representatives.items():
        if not samples:
            continue
        limited_samples = samples[: resolved_selection.n_representatives]
        jobs.append((cluster_id, limited_samples))

    if not jobs:
        LOGGER.warning("Representative sampling yielded no data for labeling.")
        collection.label_results = {}
        return collection

    if resolved_config.async_mode:
        results = _execute_async_labeling(
            jobs,
            collection,
            resolved_config,
            provider_kwargs,
            cluster_sizes,
        )
    else:
        results = _execute_sync_labeling(
            jobs,
            collection,
            resolved_config,
            provider_kwargs,
            cluster_sizes,
        )

    collection.label_results = results
    return collection


def _execute_sync_labeling(
    jobs: Iterable[Tuple[int, List[RepresentativeSample]]],
    collection: ContentUnitCollection,
    config: LabelingConfig,
    provider_kwargs: Dict[str, Any],
    cluster_sizes: Dict[int, int],
) -> Dict[int, LabelResult]:
    results: Dict[int, LabelResult] = {}

    for cluster_id, samples in jobs:
        messages = _build_messages(collection, cluster_id, samples, cluster_sizes.get(cluster_id, len(samples)), config)
        result = _label_cluster_sync(cluster_id, messages, config, provider_kwargs)
        results[cluster_id] = result

    return results


def _execute_async_labeling(
    jobs: Iterable[Tuple[int, List[RepresentativeSample]]],
    collection: ContentUnitCollection,
    config: LabelingConfig,
    provider_kwargs: Dict[str, Any],
    cluster_sizes: Dict[int, int],
) -> Dict[int, LabelResult]:
    jobs = list(jobs)

    async def _runner() -> Dict[int, LabelResult]:
        semaphore = asyncio.Semaphore(max(1, config.max_concurrency))
        tasks = []

        for cluster_id, samples in jobs:
            messages = _build_messages(
                collection,
                cluster_id,
                samples,
                cluster_sizes.get(cluster_id, len(samples)),
                config,
            )

            tasks.append(
                asyncio.create_task(
                    _label_cluster_async(
                        cluster_id,
                        messages,
                        config,
                        provider_kwargs,
                        semaphore,
                    )
                )
            )

        results: Dict[int, LabelResult] = {}
        for task in asyncio.as_completed(tasks):
            cluster_id, label_result = await task
            results[cluster_id] = label_result
        return results

    try:
        return asyncio.run(_runner())
    except RuntimeError:
        LOGGER.warning(
            "Async labeling requested while an event loop is active; falling back to synchronous execution."
        )
        return _execute_sync_labeling(jobs, collection, config, provider_kwargs, cluster_sizes)


def _label_cluster_sync(
    cluster_id: int,
    messages: List[Dict[str, Any]],
    config: LabelingConfig,
    provider_kwargs: Dict[str, Any],
) -> LabelResult:
    delay = max(0.0, config.retry_backoff_seconds[0])

    for attempt in range(1, config.max_retries + 1):
        try:
            response = litellm.completion(
                model=config.model,
                messages=messages,
                temperature=config.temperature,
                max_tokens=config.max_output_tokens,
                response_format=LabelPayload,
                **provider_kwargs,
            )
            response_dict = _coerce_response_to_dict(response)
            return _parse_label_response(cluster_id, response_dict)
        except Exception as exc:  # pragma: no cover - upstream errors vary
            LOGGER.warning(
                "Labeling attempt %s for cluster %s failed: %s",
                attempt,
                cluster_id,
                exc,
            )
            if attempt == config.max_retries:
                break
            time.sleep(delay)
            delay = min(delay * 2, config.retry_backoff_seconds[1])

    LOGGER.error("All labeling attempts failed for cluster %s; returning fallback label.", cluster_id)
    return _fallback_label(cluster_id)


async def _label_cluster_async(
    cluster_id: int,
    messages: List[Dict[str, Any]],
    config: LabelingConfig,
    provider_kwargs: Dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> Tuple[int, LabelResult]:
    delay = max(0.0, config.retry_backoff_seconds[0])

    async with semaphore:
        for attempt in range(1, config.max_retries + 1):
            try:
                response = await litellm.acompletion(
                    model=config.model,
                    messages=messages,
                    temperature=config.temperature,
                    max_tokens=config.max_output_tokens,
                    response_format=LabelPayload,
                    **provider_kwargs,
                )
                response_dict = _coerce_response_to_dict(response)
                return cluster_id, _parse_label_response(cluster_id, response_dict)
            except Exception as exc:  # pragma: no cover - upstream errors vary
                LOGGER.warning(
                    "Async labeling attempt %s for cluster %s failed: %s",
                    attempt,
                    cluster_id,
                    exc,
                )
                if attempt == config.max_retries:
                    break
                await asyncio.sleep(delay)
                delay = min(delay * 2, config.retry_backoff_seconds[1])

    LOGGER.error("Async labeling exhausted retries for cluster %s; returning fallback.", cluster_id)
    return cluster_id, _fallback_label(cluster_id)


def _build_messages(
    collection: ContentUnitCollection,
    cluster_id: int,
    samples: List[RepresentativeSample],
    cluster_size: int,
    config: LabelingConfig,
) -> List[Dict[str, Any]]:
    system_prompt = config.system_prompt_override or DEFAULT_SYSTEM_PROMPT
    avg_similarity = sum(sample.similarity for sample in samples) / max(1, len(samples))

    lines: List[str] = [
        f"Cluster ID: {cluster_id}",
        f"Total content units: {cluster_size}",
        f"Average representative similarity: {avg_similarity:.3f}",
        "Representative content units:",
    ]

    for idx, sample in enumerate(samples, start=1):
        unit = collection[sample.unit_index]
        content_view = unit.get_content(max_char_to_truncate=config.max_chars_per_rep)
        lines.append(
            f"{idx}. Core sample: {str(sample.is_core).lower()} | "
            f"Similarity: {sample.similarity:.3f} | Unit Index: {sample.unit_index}"
        )
        lines.append(content_view)

    lines.append("\nInstructions:")
    lines.extend(
        [
            "- Provide a concise label (<= 5 words) describing the common topic.",
            "- Provide a 1-2 sentence explanation referencing recurring ideas.",
            "- Output a confidence between 0 and 1.",
            "- List up to five keywords or short phrases capturing the theme.",
            "Respond using the structured format with fields: label, explanation, confidence, keywords.",
        ]
    )

    user_prompt = "\n".join(lines)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _provider_kwargs(provider: str) -> Dict[str, Any]:
    provider_normalized = (provider or "").lower()
    if provider_normalized == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OpenAI provider selected but OPENAI_API_KEY is not set."
            )
        return {
            "api_key": api_key,
            "custom_llm_provider": "openai",
        }
    if provider_normalized == "fireworks":
        api_key = os.environ.get("FIREWORKS_AI_API_KEY") or os.environ.get("FIREWORKS_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Fireworks provider selected but FIREWORKS_AI_API_KEY or FIREWORKS_API_KEY is not set."
            )
        api_base = os.environ.get("FIREWORKS_AI_API_BASE", "https://api.fireworks.ai/inference/v1")
        return {
            "api_key": api_key,
            "api_base": api_base,
            "custom_llm_provider": "openai",
        }
    return {}


def _parse_label_response(cluster_id: int, response: Dict[str, Any]) -> LabelResult:
    choice = _first_choice(response)
    if not choice:
        LOGGER.error("No completion choices returned for cluster %s", cluster_id)
        return _fallback_label(cluster_id)

    message = choice.get("message", {})
    parsed_payload = message.get("parsed")
    raw_content = message.get("content")

    payload: Optional[LabelPayload]
    if isinstance(parsed_payload, LabelPayload):
        payload = parsed_payload
    elif parsed_payload is not None:
        try:
            payload = LabelPayload.model_validate(parsed_payload)
        except ValidationError as exc:
            LOGGER.warning(
                "Structured output validation failed for cluster %s: %s",
                cluster_id,
                exc,
            )
            return _fallback_label(cluster_id, raw_response=raw_content)
    else:
        try:
            payload = LabelPayload.model_validate_json(raw_content or "{}")
        except (ValidationError, TypeError, ValueError) as exc:
            LOGGER.warning(
                "Structured response missing for cluster %s; using fallback (%s)",
                cluster_id,
                exc,
            )
            return _fallback_label(cluster_id, raw_response=raw_content)

    serialized = raw_content
    if serialized is None:
        try:
            serialized = payload.model_dump_json()
        except Exception:  # pragma: no cover - best effort serialization
            serialized = None

    return payload.to_label_result(
        cluster_id=cluster_id,
        raw_response=serialized,
        usage=response.get("usage"),
    )


def _fallback_label(cluster_id: int, raw_response: Optional[str] = None) -> LabelResult:
    return LabelResult(
        cluster_id=cluster_id,
        label=f"Cluster {cluster_id}",
        explanation="Automatic labeling failed",
        confidence=0.0,
        keywords=[],
        raw_response=raw_response,
        usage_metadata=None,
    )


def _first_choice(response: Dict[str, Any]) -> Dict[str, Any]:
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            return choice
    return {}

def _coerce_response_to_dict(response: Any) -> Dict[str, Any]:
    if isinstance(response, dict):
        return response
    for extractor in (getattr(response, "model_dump", None), getattr(response, "dict", None)):
        if callable(extractor):
            try:
                data = extractor()
                if isinstance(data, dict):
                    return data
            except Exception:  # pragma: no cover - defensive
                continue
    raw_dict = getattr(response, "__dict__", None)
    if isinstance(raw_dict, dict):
        return raw_dict
    return {"raw_response": repr(response)}

__all__ = [
    "auto_label_clusters",
]
