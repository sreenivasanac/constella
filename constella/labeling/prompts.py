"""Prompt construction utilities for LLM-based cluster labeling."""

from __future__ import annotations

from typing import Any, Dict, List

from constella.config.schemas import LabelingConfig
from constella.data.models import ContentUnitCollection
from constella.data.results import RepresentativeSample


DEFAULT_SYSTEM_PROMPT = (
    "You are an analyst categorizing large collections of user content units. "
    "Each cluster groups semantically similar items, and you will only see a "
    "handful of representative samples. Infer a concise, human-readable label "
    "that captures the shared topic."
)


def build_labeling_messages(
    collection: ContentUnitCollection,
    cluster_id: int,
    samples: List[RepresentativeSample],
    cluster_size: int,
    config: LabelingConfig,
) -> List[Dict[str, Any]]:
    """Construct messages for LLM labeling request.
    
    Builds a structured prompt containing cluster metadata, representative samples,
    and instructions for generating labels. The messages follow the chat format
    expected by LiteLLM (system + user messages).
    
    Args:
        collection: Content unit collection containing the full dataset.
        cluster_id: Identifier for the cluster being labeled.
        samples: Representative samples selected from the cluster.
        cluster_size: Total number of content units in the cluster.
        config: Configuration with prompt overrides and content truncation settings.
    
    Returns:
        List of message dictionaries with "role" and "content" keys, ready for
        LiteLLM completion API.
    
    Examples:
        >>> messages = build_labeling_messages(
        ...     collection=my_collection,
        ...     cluster_id=0,
        ...     samples=representative_samples,
        ...     cluster_size=150,
        ...     config=LabelingConfig()
        ... )
        >>> # Returns: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    """
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


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "build_labeling_messages",
]
