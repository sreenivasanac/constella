"""Tests for the K-Means clustering helper."""

from __future__ import annotations

import numpy as np

from constella.clustering.kmeans import (
    _resolve_silhouette_sample_size,
    _select_elbow_k,
    _select_cluster_count,
    run_kmeans,
)
from constella.config.schemas import ClusteringConfig


def test_run_kmeans_deterministic_assignments():
    rng = np.random.default_rng(123)
    data = np.vstack([
        rng.normal(loc=0.0, scale=0.2, size=(10, 2)),
        rng.normal(loc=5.0, scale=0.2, size=(10, 2)),
    ])

    config = ClusteringConfig(seed=7, fallback_n_clusters=2, max_candidate_clusters=4)
    labels, metrics = run_kmeans(data, config)

    assert metrics["config_snapshot"] == config
    expected_k = _select_cluster_count(data, config)
    assert len(set(labels)) == expected_k
    assert metrics["silhouette_score"] is not None


def test_run_kmeans_respects_fallback_when_insufficient_points():
    data = np.array([[0.0, 0.0], [1.0, 1.0]])
    config = ClusteringConfig(seed=1, fallback_n_clusters=2, enable_silhouette_selection=True)
    labels, metrics = run_kmeans(data, config)

    assert len(set(labels)) == 2
    assert metrics["silhouette_score"] is None


def test_run_kmeans_handles_single_cluster_silhouette_error():
    data = np.array([[0.0, 0.0], [0.0, 0.0]])
    config = ClusteringConfig(seed=3, fallback_n_clusters=2, enable_silhouette_selection=True)
    labels, metrics = run_kmeans(data, config)

    assert len(labels) == 2
    assert metrics["silhouette_score"] is None


def test_silhouette_selection_finds_optimal_cluster_count():
    rng = np.random.default_rng(1234)
    cluster_a = rng.normal(loc=(-5.0, 0.0), scale=0.2, size=(30, 2))
    cluster_b = rng.normal(loc=(0.0, 5.0), scale=0.2, size=(30, 2))
    cluster_c = rng.normal(loc=(5.0, 0.0), scale=0.2, size=(30, 2))
    data = np.vstack([cluster_a, cluster_b, cluster_c])

    config = ClusteringConfig(
        seed=11,
        fallback_n_clusters=5,
        max_candidate_clusters=6,
        enable_elbow_selection=False,
        enable_davies_bouldin_selection=False,
    )

    labels, metrics = run_kmeans(data, config)

    assert len(set(labels)) == 3
    assert metrics["silhouette_score"] is not None
    assert metrics["silhouette_score"] > 0.5


def test_resolve_silhouette_sample_size_adjusts_within_bounds():
    assert _resolve_silhouette_sample_size(None, 10, 3) is None
    # Requested below minimum should elevate to labels + 1 when possible.
    assert _resolve_silhouette_sample_size(2, 20, 4) == 5
    # Requested above range should clamp to n_samples - 1 while staying > labels.
    assert _resolve_silhouette_sample_size(50, 20, 3) == 19
    # When data cannot support sampling beyond labels, fall back to full dataset.
    assert _resolve_silhouette_sample_size(3, 5, 4) is None


def test_select_elbow_k_prefers_highest_deviation():
    k_values = [2, 3, 4, 5]
    inertias = [500.0, 300.0, 200.0, 180.0]
    assert _select_elbow_k(k_values, inertias) == 3


def test_elbow_selection_used_when_silhouette_disabled():
    rng = np.random.default_rng(99)
    cluster_a = rng.normal(loc=(-4.0, 0.0), scale=0.25, size=(25, 2))
    cluster_b = rng.normal(loc=(0.0, 4.0), scale=0.25, size=(25, 2))
    cluster_c = rng.normal(loc=(4.0, 0.0), scale=0.25, size=(25, 2))
    data = np.vstack([cluster_a, cluster_b, cluster_c])

    config = ClusteringConfig(
        seed=21,
        fallback_n_clusters=2,
        max_candidate_clusters=6,
        enable_silhouette_selection=False,
        enable_elbow_selection=True,
        enable_davies_bouldin_selection=False,
    )

    labels, _ = run_kmeans(data, config)

    assert len(set(labels)) == 3


def test_combined_selection_prefers_max(monkeypatch):
    rng = np.random.default_rng(7)
    cluster_a = rng.normal(loc=(-4.0, 0.0), scale=0.25, size=(30, 2))
    cluster_b = rng.normal(loc=(0.0, 4.0), scale=0.25, size=(30, 2))
    cluster_c = rng.normal(loc=(4.0, 0.0), scale=0.25, size=(30, 2))
    data = np.vstack([cluster_a, cluster_b, cluster_c])

    def fake_elbow(candidates, inertias):
        return max(candidates)

    def fake_silhouette(vectors, labels, sample_size, rng_obj, context):
        if "for final clustering" in context:
            return 0.5
        if "k=3" in context:
            return 0.4
        if "k=4" in context:
            return 0.6
        return 0.2

    def fake_davies(vectors, labels, context):
        if "k=6" in context:
            return 0.1
        if "k=5" in context:
            return 0.2
        return 0.5

    monkeypatch.setattr("constella.clustering.kmeans._select_elbow_k", fake_elbow)
    monkeypatch.setattr("constella.clustering.kmeans._safe_silhouette_score", fake_silhouette)
    monkeypatch.setattr("constella.clustering.kmeans._safe_davies_bouldin_score", fake_davies)

    config = ClusteringConfig(
        seed=15,
        fallback_n_clusters=2,
        max_candidate_clusters=6,
        enable_silhouette_selection=True,
        enable_elbow_selection=True,
        enable_davies_bouldin_selection=True,
    )

    labels, _ = run_kmeans(data, config)

    assert len(set(labels)) == 6


def test_davies_bouldin_selection_used_when_others_disabled():
    rng = np.random.default_rng(2024)
    cluster_a = rng.normal(loc=(-3.0, 0.0), scale=0.2, size=(35, 2))
    cluster_b = rng.normal(loc=(0.0, 3.0), scale=0.2, size=(35, 2))
    cluster_c = rng.normal(loc=(3.0, 0.0), scale=0.2, size=(35, 2))
    data = np.vstack([cluster_a, cluster_b, cluster_c])

    config = ClusteringConfig(
        seed=33,
        fallback_n_clusters=2,
        max_candidate_clusters=6,
        enable_silhouette_selection=False,
        enable_elbow_selection=False,
        enable_davies_bouldin_selection=True,
    )

    labels, _ = run_kmeans(data, config)

    assert len(set(labels)) == 3
