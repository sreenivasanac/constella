"""Entry points for LLM-backed cluster labeling."""

from __future__ import annotations

from constella.config.schemas import LabelingConfig
from constella.data.models import ContentUnitCollection


def auto_label_clusters(
    collection: ContentUnitCollection,
    llm_provider: str,
    config: LabelingConfig | None,
) -> ContentUnitCollection:
    """High-level placeholder for automatic cluster labeling via LLMs."""

    raise NotImplementedError("Auto-labeling is not implemented yet.")
