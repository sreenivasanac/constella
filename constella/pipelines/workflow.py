"""End-to-end orchestration for the minimal clustering workflow."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from constella.config.schemas import ClusteringConfig, VisualizationConfig
from constella.data.models import ClusterAssignment, ContentUnit
from constella.embeddings.base import EmbeddingProvider
from constella.embeddings.adapters import LiteLLMEmbeddingProvider
from constella.visualization.umap import project_embeddings, save_umap_plot
from constella.clustering.kmeans import run_kmeans


LOGGER = logging.getLogger(__name__)


def _extract_texts(units_or_texts: Iterable[str | ContentUnit]) -> Tuple[List[ContentUnit], List[str]]:
    normalized_units: List[ContentUnit] = []
    texts: List[str] = []
    for idx, item in enumerate(units_or_texts):
        if isinstance(item, ContentUnit):
            normalized_units.append(item)
            texts.append(item.text)
        else:
            identifier = f"item_{idx}"
            normalized_units.append(ContentUnit(identifier=identifier, text=str(item)))
            texts.append(str(item))
    return normalized_units, texts


def cluster_texts(
    texts_or_units: Sequence[str | ContentUnit],
    clustering_config: Optional[ClusteringConfig] = None,
    visualization_config: Optional[VisualizationConfig] = None,
    embedding_provider: Optional[EmbeddingProvider] = None,
) -> Tuple[ClusterAssignment, Optional[List[Path]]]:
    """Run the end-to-end embedding, clustering, and optional visualization workflow."""

    if not texts_or_units:
        raise ValueError("No texts provided for clustering.")

    units, texts = _extract_texts(texts_or_units)
    config = clustering_config or ClusteringConfig()
    del embedding_provider # for now custom embedding_provider is not supported
    # if embedding_provider is None:
    if LiteLLMEmbeddingProvider is None:
        # only LiteLLM is supported
        raise RuntimeError(
            "LiteLLM is unavailable. Install constella with the 'llm' extra or provide a custom embedding provider."
        )
    provider = LiteLLMEmbeddingProvider()
    # else:
    #     provider = embedding_provider

    LOGGER.info("Generating embeddings for %s texts", len(texts))
    embeddings = provider.embed_texts(texts)
    assignment = run_kmeans(embeddings, config)

    artifacts: Optional[List[Path]] = None
    if visualization_config:
        projection = project_embeddings(embeddings, visualization_config)
        path = save_umap_plot(projection, assignment.assignments, visualization_config)
        artifacts = [path]

    return assignment, artifacts


__all__ = ["cluster_texts"]
