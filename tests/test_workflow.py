"""Integration tests for the clustering workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from constella.config.schemas import ClusteringConfig, VisualizationConfig
from constella.data.models import ContentUnit, ContentUnitCollection
from constella.embeddings.base import InMemoryEmbeddingProvider
from constella.pipelines.workflow import cluster_texts


def test_cluster_texts_with_mock_embeddings(tmp_path: Path):
    collection = ContentUnitCollection([
        ContentUnit(identifier=f"item_{idx}", text=text)
        for idx, text in enumerate(["alpha", "beta", "gamma", "delta"])
    ])
    embeddings = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [1.0, 1.0],
        [1.1, 1.0],
    ])
    provider = InMemoryEmbeddingProvider(embeddings)
    config = ClusteringConfig(
        seed=5,
        fallback_n_clusters=2,
        enable_silhouette_selection=False,
        enable_elbow_selection=False,
        enable_davies_bouldin_selection=False,
    )
    viz_config = VisualizationConfig(output_path=tmp_path, random_state=5)

    result_collection = cluster_texts(
        collection,
        clustering_config=config,
        visualization_config=viz_config,
        embedding_provider=provider,
    )

    artifacts = result_collection.artifacts
    assert artifacts is not None
    assert artifacts.generated is True
    artifact_parent = Path(artifacts.static_plot).parent
    assert artifact_parent.parent == tmp_path
    assert artifact_parent.name.startswith("artifacts_")

    for artifact in (artifacts.static_plot, artifacts.html_plot):
        assert artifact is not None
        assert Path(artifact).exists()
        assert Path(artifact).parent == artifact_parent
    assert [unit.embedding for unit in result_collection] == embeddings.tolist()
    assert len(set(unit.cluster_id for unit in result_collection)) == 2
    assert result_collection.metrics is not None


def test_cluster_texts_accepts_content_units():
    collection = ContentUnitCollection([
        ContentUnit(identifier="a", text="foo"),
        ContentUnit(identifier="b", text="bar"),
    ])
    embeddings = np.array([[0.0, 0.0], [1.0, 1.0]])
    provider = InMemoryEmbeddingProvider(embeddings)
    config = ClusteringConfig(seed=2, fallback_n_clusters=2, enable_silhouette_selection=False)

    result_collection = cluster_texts(
        collection,
        clustering_config=config,
        embedding_provider=provider,
    )

    assert set(unit.cluster_id for unit in result_collection) == {0, 1}
    assert result_collection.artifacts is None
    assert [unit.embedding for unit in result_collection] == embeddings.tolist()
