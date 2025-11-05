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
    config = ClusteringConfig(seed=5, fallback_n_clusters=2)
    viz_config = VisualizationConfig(output_path=tmp_path / "plot.png", random_state=5)

    result_collection, artifacts, metrics = cluster_texts(
        collection,
        clustering_config=config,
        visualization_config=viz_config,
        embedding_provider=provider,
    )

    assert metrics["silhouette_score"] is not None
    assert artifacts is not None and len(artifacts) == 2
    for artifact in artifacts:
        assert Path(artifact).exists()
    assert [unit.embedding for unit in result_collection] == embeddings.tolist()


def test_cluster_texts_accepts_content_units():
    collection = ContentUnitCollection([
        ContentUnit(identifier="a", text="foo"),
        ContentUnit(identifier="b", text="bar"),
    ])
    embeddings = np.array([[0.0, 0.0], [1.0, 1.0]])
    provider = InMemoryEmbeddingProvider(embeddings)
    config = ClusteringConfig(seed=2, fallback_n_clusters=2, enable_silhouette_selection=False)

    result_collection, artifacts, metrics = cluster_texts(
        collection,
        clustering_config=config,
        embedding_provider=provider,
    )

    assert set(unit.cluster_id for unit in result_collection) == {0, 1}
    assert artifacts is None
    assert metrics["n_clusters"] == 2
    assert [unit.embedding for unit in result_collection] == embeddings.tolist()
