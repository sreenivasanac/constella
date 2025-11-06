"""Modular orchestration utilities for the Constella clustering workflow."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, cast

from constella.clustering.kmeans import run_kmeans
from constella.config.schemas import (
    ClusteringConfig,
    LabelingConfig,
    VisualizationConfig,
    VisualizationArtifacts,
)
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



def generate_visualization(
    collection: ContentUnitCollection,
    config: VisualizationConfig,
    *,
    title: Optional[str] = None,
) -> VisualizationArtifacts:
    """Project embeddings and persist both static and interactive visualizations."""

    embeddings = collection.embedding_matrix()
    labels = collection.cluster_ids()
    projection = project_embeddings(embeddings, config)

    static_path = save_umap_plot(projection, labels, config, title=title)

    html_path = create_umap_plot_html(
        projection,
        labels,
        config,
        units=collection.units(),
        title=title,
        output_path=Path(config.output_path).with_suffix(".html"),
    )
    return VisualizationArtifacts(
        static_plot=static_path,
        html_plot=html_path,
    )


def run_pipeline(
    collection: ContentUnitCollection,
    *,
    steps: Optional[Sequence[str]] = None,
    configs: Optional[Dict[str, object]] = None,
) -> ContentUnitCollection:
    """Execute the requested workflow steps in order."""

    ordered_steps = list(steps or ("embed", "cluster"))
    if not ordered_steps:
        raise ValueError("Pipeline requires at least one step.")

    step_configs = configs or {}
    generated_artifacts: Optional[VisualizationArtifacts] = None

    def _run_embed() -> None:
        provider = _resolve_embedding_provider(step_configs.get("embed"))
        embed_texts(collection, provider=provider)

    def _run_cluster() -> None:
        cluster_cfg = cast(Optional[ClusteringConfig], step_configs.get("cluster"))
        run_kmeans(collection, cluster_cfg)

    def _run_label() -> None:
        label_params = step_configs.get("label")
        if label_params is None:
            raise ValueError("Labeling step requires configuration")
        label_cfg, provider_name = cast(Tuple[Optional[LabelingConfig], str], label_params)
        auto_label_clusters(collection, provider_name, label_cfg)

    def _run_visualize() -> None:
        viz_cfg = step_configs.get("visualize")
        if viz_cfg is None:
            raise ValueError("Visualization step requires configuration")
        generated_artifacts = generate_visualization(
            collection,
            cast(VisualizationConfig, viz_cfg),
        )
        if generated_artifacts.generated:
            LOGGER.info(
                "Generated visualization artifacts: static=%s, html=%s",
                generated_artifacts.static_plot,
                generated_artifacts.html_plot,
            )
            collection.artifacts = generated_artifacts
        else:
            LOGGER.info("Visualization step completed but produced no persisted artifacts.")
            collection.artifacts = None

    dispatch = {
        "embed": _run_embed,
        "cluster": _run_cluster,
        "label": _run_label,
        "visualize": _run_visualize,
    }

    for step in ordered_steps:
        try:
            dispatch[step]()
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Unknown pipeline step: {step}") from exc


    return collection


def cluster_texts(
    collection: ContentUnitCollection,
    *,
    clustering_config: Optional[ClusteringConfig] = None,
    visualization_config: Optional[VisualizationConfig] = None,
    embedding_provider: Optional[EmbeddingProvider] = None,
) -> ContentUnitCollection:
    """Run embedding, clustering, and optional visualization for the collection."""

    if len(collection) == 0:
        raise ValueError("No content units provided for clustering.")

    provider = embedding_provider or LiteLLMEmbeddingFireworksProvider()
    embed_texts(collection, provider=provider)
    run_kmeans(collection, clustering_config)

    if visualization_config is not None:
        artifacts = generate_visualization(collection, visualization_config)
        collection.artifacts = artifacts if artifacts.generated else None
    else:
        collection.artifacts = None

    return collection


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


__all__ = [
    "embed_texts",
    "generate_visualization",
    "run_pipeline",
    "cluster_texts",
]
