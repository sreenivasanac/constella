"""Integration tests for the clustering workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from constella.config.schemas import ClusteringConfig, VisualizationConfig
from constella.data.models import ContentUnit
from constella.embeddings.base import InMemoryEmbeddingProvider
from constella.pipelines.workflow import cluster_texts


def test_cluster_texts_with_mock_embeddings(tmp_path: Path):
    texts = ["alpha", "beta", "gamma", "delta"]
    embeddings = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [1.0, 1.0],
        [1.1, 1.0],
    ])
    provider = InMemoryEmbeddingProvider(embeddings)
    config = ClusteringConfig(seed=5, fallback_n_clusters=2)
    viz_config = VisualizationConfig(output_path=tmp_path / "plot.png", random_state=5)

    assignment, artifacts = cluster_texts(
        texts,
        clustering_config=config,
        visualization_config=viz_config,
        embedding_provider=provider,
    )

    assert assignment.silhouette_score is not None
    assert artifacts is not None
    assert Path(artifacts[0]).exists()


def test_cluster_texts_accepts_content_units():
    units = [
        ContentUnit(identifier="a", text="foo"),
        ContentUnit(identifier="b", text="bar"),
    ]
    embeddings = np.array([[0.0, 0.0], [1.0, 1.0]])
    provider = InMemoryEmbeddingProvider(embeddings)
    config = ClusteringConfig(seed=2, fallback_n_clusters=2, enable_silhouette_selection=False)

    assignment, artifacts = cluster_texts(
        units,
        clustering_config=config,
        embedding_provider=provider,
    )

    assert set(assignment.assignments) == {0, 1}
    assert artifacts is None
