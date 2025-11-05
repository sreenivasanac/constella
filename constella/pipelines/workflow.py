"""Modular orchestration utilities for the Constella clustering workflow."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from constella.clustering.kmeans import run_kmeans
from constella.config.schemas import ClusteringConfig, LabelingConfig, VisualizationConfig
from constella.data.models import ContentUnitCollection
from constella.embeddings.adapters import (
    LiteLLMEmbeddingFireworksProvider,
    LiteLLMEmbeddingOpenAIProvider,
)
from constella.embeddings.base import EmbeddingProvider
from constella.labeling.llm import auto_label_clusters
from constella.visualization.umap import (
    create_umap_plot_html,
    project_embeddings,
    save_umap_plot,
)


LOGGER = logging.getLogger(__name__)


def embed_texts(
    collection: ContentUnitCollection,
    provider: Optional[EmbeddingProvider] = None,
) -> ContentUnitCollection:
    """Generate embeddings for the supplied collection and attach them in-place."""

    if len(collection) == 0:
        return collection

    resolved_embedding_provider = provider or LiteLLMEmbeddingFireworksProvider()
    LOGGER.info(
        "Generating embeddings for %s units using %s",
        len(collection),
        type(resolved_embedding_provider).__name__,
    )
    vectors = resolved_embedding_provider.embed_texts(collection.all_texts())
    collection.attach_embeddings(vectors)
    return collection


def cluster_embeddings(
    collection: ContentUnitCollection,
    config: Optional[ClusteringConfig] = None,
) -> Tuple[ContentUnitCollection, Dict[str, object]]:
    """Run K-Means clustering on embedded units and attach cluster IDs."""

    if len(collection) == 0:
        raise ValueError("Cannot cluster an empty collection of units.")

    embeddings = collection.embedding_matrix()
    clustering_config = config or ClusteringConfig()
    labels, metrics = run_kmeans(embeddings, clustering_config)
    collection.attach_cluster_ids(labels)
    return collection, metrics


def generate_visualization(
    collection: ContentUnitCollection,
    config: VisualizationConfig,
    *,
    title: Optional[str] = None,
) -> List[Path]:
    """Project embeddings and persist both static and interactive visualizations."""

    embeddings = collection.embedding_matrix()
    labels = collection.cluster_ids()
    projection = project_embeddings(embeddings, config)

    artifacts: List[Path] = []
    static_path = save_umap_plot(projection, labels, config, title=title)
    artifacts.append(static_path)

    html_path = create_umap_plot_html(
        projection,
        labels,
        config,
        texts_or_units=collection.units(),
        title=title,
        output_path=Path(config.output_path).with_suffix(".html"),
    )
    artifacts.append(html_path)
    return artifacts


def run_pipeline(
    collection: ContentUnitCollection,
    *,
    steps: Optional[Sequence[str]] = None,
    configs: Optional[Dict[str, object]] = None,
) -> Tuple[ContentUnitCollection, List[Path], Dict[str, object]]:
    """Execute the requested workflow steps in order."""

    ordered_steps = list(steps or ("embed", "cluster"))
    if not ordered_steps:
        raise ValueError("Pipeline requires at least one step.")

    step_configs = configs or {}
    artifacts: List[Path] = []
    metrics: Dict[str, object] = {}

    for step in ordered_steps:
        if step == "embed":
            provider = _resolve_embedding_provider(step_configs.get("embed"))
            embed_texts(collection, provider=provider)
        elif step == "cluster":
            cluster_cfg = _resolve_clustering_config(step_configs.get("cluster"))
            _, metrics = cluster_embeddings(collection, cluster_cfg)
        elif step == "label":
            label_cfg, provider_name = _resolve_labeling_config(step_configs.get("label"))
            auto_label_clusters(collection, provider_name, label_cfg)
        elif step == "visualize":
            viz_cfg = _resolve_visualization_config(step_configs.get("visualize"))
            artifacts.extend(generate_visualization(collection, viz_cfg))
        else:
            raise ValueError(f"Unknown pipeline step: {step}")

    return collection, artifacts, metrics


def cluster_texts(
    collection: ContentUnitCollection,
    *,
    clustering_config: Optional[ClusteringConfig] = None,
    visualization_config: Optional[VisualizationConfig] = None,
    embedding_provider: Optional[EmbeddingProvider] = None,
) -> Tuple[ContentUnitCollection, Optional[List[Path]], Dict[str, object]]:
    """Run embedding, clustering, and optional visualization for the collection."""

    if len(collection) == 0:
        raise ValueError("No content units provided for clustering.")

    provider = embedding_provider or LiteLLMEmbeddingFireworksProvider()
    embed_texts(collection, provider=provider)
    _, metrics = cluster_embeddings(collection, clustering_config)

    artifacts: Optional[List[Path]] = None
    if visualization_config is not None:
        artifacts = generate_visualization(collection, visualization_config)

    return collection, artifacts, metrics


def _resolve_embedding_provider(config: object) -> Optional[EmbeddingProvider]:
    if config is None:
        return None
    if isinstance(config, EmbeddingProvider):
        return config
    if isinstance(config, dict):
        provider = config.get("provider")
        if provider is not None and not isinstance(provider, EmbeddingProvider):
            raise TypeError("embed configuration 'provider' must be an EmbeddingProvider instance")
        if provider is not None:
            return provider
        if config.get("preset") == "openai":
            return LiteLLMEmbeddingOpenAIProvider()
        if config.get("preset") == "fireworks":
            return LiteLLMEmbeddingFireworksProvider()
    raise TypeError("Unsupported embed configuration")


def _resolve_clustering_config(config: object) -> Optional[ClusteringConfig]:
    if config is None:
        return None
    if isinstance(config, ClusteringConfig):
        return config
    if isinstance(config, dict):
        return ClusteringConfig(**config)
    raise TypeError("cluster configuration must be a ClusteringConfig or dict")


def _resolve_visualization_config(config: object) -> VisualizationConfig:
    if isinstance(config, VisualizationConfig):
        return config
    if isinstance(config, dict):
        return VisualizationConfig(**config)
    raise TypeError("visualize configuration must be a VisualizationConfig or dict")


def _resolve_labeling_config(config: object) -> Tuple[LabelingConfig | None, str]:
    if config is None:
        raise ValueError("Labeling step requires configuration")
    if not isinstance(config, dict):
        raise TypeError("label configuration must be a dict containing provider information")

    provider = config.get("provider") or config.get("llm_provider")
    if provider is None:
        raise ValueError("Labeling configuration requires 'provider' or 'llm_provider'")

    raw_cfg = config.get("config")
    if raw_cfg is None:
        label_cfg = None
    elif isinstance(raw_cfg, LabelingConfig):
        label_cfg = raw_cfg
    elif isinstance(raw_cfg, dict):
        label_cfg = LabelingConfig(**raw_cfg)
    else:
        raise TypeError("Labeling configuration 'config' must be a LabelingConfig or dict")

    return label_cfg, str(provider)


__all__ = [
    "embed_texts",
    "cluster_embeddings",
    "generate_visualization",
    "run_pipeline",
    "cluster_texts",
]
