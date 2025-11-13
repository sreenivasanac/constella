"""Automatic cluster labeling via LLM providers."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import litellm

from constella.config.schemas import (
    LabelingConfig,
    RepresentativeSelectionConfig,
)
from constella.data.models import ContentUnitCollection
from constella.data.results import LabelResult, RepresentativeSample
from constella.labeling.providers import get_provider_config
from constella.labeling.prompts import build_labeling_messages
from constella.labeling.response_parsing import (
    LabelPayload,
    create_fallback_label,
    parse_label_response,
)
from constella.labeling.selection import select_representatives


LOGGER = logging.getLogger(__name__)


def auto_label_clusters(
    collection: ContentUnitCollection,
    config: Optional[LabelingConfig] = None,
    selection_config: Optional[RepresentativeSelectionConfig] = None,
) -> ContentUnitCollection:
    """Automatically generate labels for clusters using LLM.
    
    Selects representative samples from each cluster and uses an LLM to generate
    concise, human-readable labels with explanations and keywords. Results are
    attached to the collection's label_results attribute.
    
    Args:
        collection: Content units with cluster assignments. Must have embeddings
            and cluster_id populated for each unit.
        config: LLM provider and generation settings. Defaults to OpenAI gpt-4o-mini
            if not provided.
        selection_config: Controls representative sampling strategy (core ratio,
            diversity, etc.). Uses defaults if not provided.
    
    Returns:
        The same collection with label_results populated as a dict mapping
        cluster_id to LabelResult objects.
    
    Raises:
        RuntimeError: If the configured provider's API key is not set in environment.
    
    Examples:
        >>> from constella.data.models import ContentUnitCollection, ContentUnit
        >>> from constella.config.schemas import LabelingConfig
        >>> 
        >>> collection = ContentUnitCollection([
        ...     ContentUnit(id="1", text="Python tutorial", cluster_id=0),
        ...     ContentUnit(id="2", text="JavaScript guide", cluster_id=1),
        ... ])
        >>> # Assume embeddings and clustering already done
        >>> 
        >>> config = LabelingConfig(model="gpt-4o-mini", async_mode=True)
        >>> labeled = auto_label_clusters(collection, config)
        >>> print(labeled.label_results[0].label)
    """
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
    provider_kwargs = get_provider_config(resolved_config.llm_provider)

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
    """Execute labeling requests synchronously for all clusters.
    
    Processes each cluster sequentially, making blocking LLM API calls. Suitable
    for small datasets or when async execution is not desired.
    
    Args:
        jobs: Iterable of (cluster_id, samples) tuples to process.
        collection: Content unit collection for accessing unit data.
        config: Labeling configuration with model and retry settings.
        provider_kwargs: Provider-specific API configuration.
        cluster_sizes: Mapping of cluster_id to total cluster size.
    
    Returns:
        Dictionary mapping cluster_id to LabelResult.
    """
    results: Dict[int, LabelResult] = {}

    for cluster_id, samples in jobs:
        messages = build_labeling_messages(
            collection, cluster_id, samples, cluster_sizes.get(cluster_id, len(samples)), config
        )
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
    """Execute labeling requests asynchronously with concurrency control.
    
    Processes multiple clusters concurrently using asyncio, with a semaphore
    limiting concurrent requests. Falls back to synchronous execution if an
    event loop is already active.
    
    Args:
        jobs: Iterable of (cluster_id, samples) tuples to process.
        collection: Content unit collection for accessing unit data.
        config: Labeling configuration with concurrency and retry settings.
        provider_kwargs: Provider-specific API configuration.
        cluster_sizes: Mapping of cluster_id to total cluster size.
    
    Returns:
        Dictionary mapping cluster_id to LabelResult.
    """
    jobs = list(jobs)

    async def _runner() -> Dict[int, LabelResult]:
        semaphore = asyncio.Semaphore(max(1, config.max_concurrency))
        tasks = []

        for cluster_id, samples in jobs:
            messages = build_labeling_messages(
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
    """Generate label for a single cluster using synchronous LLM call.
    
    Makes a blocking call to the LLM API with retry logic and exponential backoff.
    Returns a fallback label if all retry attempts fail.
    
    Args:
        cluster_id: Identifier for the cluster being labeled.
        messages: Chat messages containing system prompt and user prompt.
        config: Configuration with model, temperature, and retry settings.
        provider_kwargs: Provider-specific API parameters.
    
    Returns:
        LabelResult with generated label or fallback values on failure.
    """
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
            return parse_label_response(cluster_id, response_dict)
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
    return create_fallback_label(cluster_id)


async def _label_cluster_async(
    cluster_id: int,
    messages: List[Dict[str, Any]],
    config: LabelingConfig,
    provider_kwargs: Dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> Tuple[int, LabelResult]:
    """Generate label for a single cluster using asynchronous LLM call.
    
    Makes a non-blocking async call to the LLM API with retry logic and exponential
    backoff. Uses a semaphore to limit concurrent requests. Returns a fallback
    label if all retry attempts fail.
    
    Args:
        cluster_id: Identifier for the cluster being labeled.
        messages: Chat messages containing system prompt and user prompt.
        config: Configuration with model, temperature, and retry settings.
        provider_kwargs: Provider-specific API parameters.
        semaphore: Asyncio semaphore controlling concurrency.
    
    Returns:
        Tuple of (cluster_id, LabelResult) for result aggregation.
    """
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
                return cluster_id, parse_label_response(cluster_id, response_dict)
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
    return cluster_id, create_fallback_label(cluster_id)


def _coerce_response_to_dict(response: Any) -> Dict[str, Any]:
    """Convert various response types to dictionary format.
    
    Handles multiple response formats from LiteLLM including dicts, objects with
    model_dump() or dict() methods, and objects with __dict__ attributes.
    
    Args:
        response: Response object from LiteLLM API call.
    
    Returns:
        Dictionary representation of the response.
    """
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
