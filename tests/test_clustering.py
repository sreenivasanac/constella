"""Tests for the K-Means clustering helper."""

from __future__ import annotations

import numpy as np

from constella.clustering.kmeans import run_kmeans
from constella.config.schemas import ClusteringConfig


def test_run_kmeans_deterministic_assignments():
    rng = np.random.default_rng(123)
    data = np.vstack([
        rng.normal(loc=0.0, scale=0.2, size=(10, 2)),
        rng.normal(loc=5.0, scale=0.2, size=(10, 2)),
    ])

    config = ClusteringConfig(seed=7, fallback_n_clusters=2, max_candidate_clusters=4)
    assignment = run_kmeans(data, config)

    assert assignment.config_snapshot == config
    assert set(assignment.assignments) == {0, 1}
    assert assignment.silhouette_score is not None


def test_run_kmeans_respects_fallback_when_insufficient_points():
    data = np.array([[0.0, 0.0], [1.0, 1.0]])
    config = ClusteringConfig(seed=1, fallback_n_clusters=2, enable_silhouette_selection=True)
    assignment = run_kmeans(data, config)

    assert len(set(assignment.assignments)) == 2
    assert assignment.silhouette_score is None


def test_run_kmeans_handles_single_cluster_silhouette_error():
    data = np.array([[0.0, 0.0], [0.0, 0.0]])
    config = ClusteringConfig(seed=3, fallback_n_clusters=2, enable_silhouette_selection=True)
    assignment = run_kmeans(data, config)

    assert len(assignment.assignments) == 2
    assert assignment.silhouette_score is None
