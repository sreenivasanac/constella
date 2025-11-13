"""Modular orchestration utilities for the Constella clustering workflow."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, cast
from uuid import uuid4

from constella.clustering.kmeans import run_kmeans
from constella.config.schemas import (
    ClusteringConfig,
    LabelingConfig,
    RepresentativeSelectionConfig,
    VisualizationConfig,
)
from constella.labeling.selection import select_representatives
from constella.data.models import ContentUnitCollection
from constella.data.results import VisualizationArtifacts
from constella.embeddings.adapters import (
    LiteLLMEmbeddingFireworksProvider,
    LiteLLMEmbeddingOpenAIProvider,
)
from constella.embeddings.base import EmbeddingProvider
from constella.labeling.auto_label import auto_label_clusters
from constella.visualization.artifacts import write_labels_artifact
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
    artifact_dir: Path,
) -> VisualizationArtifacts:
    """Project embeddings and persist both static and interactive visualizations."""

    embeddings = collection.embedding_matrix()
    projection = project_embeddings(embeddings, config)

    static_path = save_umap_plot(
        projection,
        collection,
        config,
        title=title,
        artifact_dir=artifact_dir,
    )

    html_path = create_umap_plot_html(
        projection,
        collection,
        config,
        title=title,
        artifact_dir=artifact_dir,
    )
    return VisualizationArtifacts(
        static_plot=static_path,
        html_plot=html_path,
        labels_json=None,
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
    artifact_dir: Optional[Path] = None

    def _ensure_artifact_dir(preferred_config: Optional[VisualizationConfig] | object = None) -> Path:
        nonlocal artifact_dir
        if artifact_dir is not None:
            return artifact_dir

        base_config: VisualizationConfig
        if isinstance(preferred_config, VisualizationConfig):
            base_config = preferred_config
        else:
            candidate = step_configs.get("visualize")
            if isinstance(candidate, VisualizationConfig):
                base_config = candidate
            else:
                base_config = VisualizationConfig()

        artifact_dir = _create_artifact_run_directory(base_config.output_path)
        return artifact_dir

    def _run_embed() -> None:
        provider = _resolve_embedding_provider(step_configs.get("embed"))
        embed_texts(collection, provider=provider)

    def _run_cluster() -> None:
        cluster_cfg = cast(Optional[ClusteringConfig], step_configs.get("cluster"))
        run_kmeans(collection, cluster_cfg)

    def _run_label() -> None:
        label_config = step_configs.get("label")
        selection_override = step_configs.get("representative_selection")

        if label_config and not isinstance(label_config, LabelingConfig):
            raise TypeError("Labeling step configuration must be a LabelingConfig instance")

        if selection_override and not isinstance(selection_override, RepresentativeSelectionConfig):
            raise TypeError("Representative selection configuration must be a RepresentativeSelectionConfig instance")

        resolved_label_config = cast(LabelingConfig, label_config) if label_config else LabelingConfig()
        resolved_selection_config = cast(Optional[RepresentativeSelectionConfig], selection_override)

        auto_label_clusters(collection, resolved_label_config, resolved_selection_config)

        if collection.label_results:
            directory = _ensure_artifact_dir(step_configs.get("visualize"))
            label_path = write_labels_artifact(collection.label_results, directory)
            if collection.artifacts:
                collection.artifacts.labels_json = label_path
            else:
                collection.artifacts = VisualizationArtifacts(labels_json=label_path)

    def _run_visualize() -> None:
        viz_cfg = step_configs.get("visualize")
        if viz_cfg is None:
            raise ValueError("Visualization step requires configuration")
        visualization_config = cast(VisualizationConfig, viz_cfg)
        directory = _ensure_artifact_dir(visualization_config)
        existing_artifacts = collection.artifacts
        generated_artifacts = generate_visualization(
            collection,
            visualization_config,
            artifact_dir=directory,
        )
        if existing_artifacts and existing_artifacts.labels_json:
            generated_artifacts.labels_json = existing_artifacts.labels_json

        if generated_artifacts.generated or generated_artifacts.labels_json:
            collection.artifacts = generated_artifacts
        else:
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
    selection_config: Optional[RepresentativeSelectionConfig] = None,
    labeling_config: Optional[LabelingConfig] = None,
) -> ContentUnitCollection:
    """Run embedding, clustering, and optional visualization for the collection."""

    if len(collection) == 0:
        raise ValueError("No content units provided for clustering.")

    provider = embedding_provider or LiteLLMEmbeddingFireworksProvider()
    embed_texts(collection, provider=provider)
    run_kmeans(collection, clustering_config)

    artifact_dir: Optional[Path] = None

    def _ensure_artifact_dir(config: Optional[VisualizationConfig]) -> Path:
        nonlocal artifact_dir
        if artifact_dir is not None:
            return artifact_dir
        base_config = config or VisualizationConfig()
        artifact_dir = _create_artifact_run_directory(base_config.output_path)
        return artifact_dir

    if labeling_config is not None:
        auto_label_clusters(collection, labeling_config, selection_config)
        if collection.label_results:
            directory = _ensure_artifact_dir(visualization_config)
            label_path = write_labels_artifact(collection.label_results, directory)
            if collection.artifacts:
                collection.artifacts.labels_json = label_path
            else:
                collection.artifacts = VisualizationArtifacts(labels_json=label_path)
    elif selection_config is not None:
        select_representatives(collection, selection_config)

    if visualization_config is not None:
        existing_artifacts = collection.artifacts
        directory = _ensure_artifact_dir(visualization_config)
        artifacts = generate_visualization(
            collection,
            visualization_config,
            artifact_dir=directory,
        )
        if existing_artifacts and existing_artifacts.labels_json:
            artifacts.labels_json = existing_artifacts.labels_json
        collection.artifacts = artifacts if artifacts.generated or artifacts.labels_json else None
    else:
        if not (collection.artifacts and collection.artifacts.labels_json):
            collection.artifacts = None

    return collection

def _create_artifact_run_directory(base_dir: Path) -> Path:
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    for _ in range(3):
        timestamp = datetime.now(timezone.utc).strftime("%m%d%Y_%H%M%S")
        suffix = uuid4().hex[:6]
        candidate = base_dir / f"artifacts_{timestamp}_{suffix}"
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate
        except FileExistsError:
            continue

    raise RuntimeError("Unable to create a unique artifact directory")
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
