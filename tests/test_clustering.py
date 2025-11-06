"""Tests for the K-Means clustering helper."""

from __future__ import annotations

import numpy as np
import pytest

from constella.clustering.kmeans import (
    _safe_silhouette_score,
    _select_elbow_k,
    _select_cluster_count,
    run_kmeans,
)
from constella.config.schemas import ClusteringConfig
from constella.data.models import ContentUnit, ContentUnitCollection


def _collection_from_vectors(vectors: np.ndarray) -> ContentUnitCollection:
    units = [
        ContentUnit(identifier=f"unit_{idx}", text=f"text_{idx}", embedding=list(vector))
        for idx, vector in enumerate(vectors)
    ]
    return ContentUnitCollection(units)


def test_run_kmeans_deterministic_assignments():
    rng = np.random.default_rng(123)
    data = np.vstack([
        rng.normal(loc=0.0, scale=0.2, size=(10, 2)),
        rng.normal(loc=5.0, scale=0.2, size=(10, 2)),
    ])

    config = ClusteringConfig(seed=7, fallback_n_clusters=2, max_candidate_clusters=4)
    collection = run_kmeans(_collection_from_vectors(data), config)

    expected_k = _select_cluster_count(data, config)
    assigned = set(collection.cluster_ids())
    assert len(assigned) == expected_k
    assert len(collection.cluster_ids()) == len(data)
    assert collection.metrics is not None
    assert collection.metrics.n_clusters == expected_k


def test_run_kmeans_respects_fallback_when_insufficient_points():
    data = np.array([[0.0, 0.0], [1.0, 1.0]])
    config = ClusteringConfig(seed=1, fallback_n_clusters=2, enable_silhouette_selection=True)
    collection = run_kmeans(_collection_from_vectors(data), config)

    assert len(set(collection.cluster_ids())) == 2


@pytest.mark.filterwarnings("ignore:Number of distinct clusters.*:sklearn.exceptions.ConvergenceWarning")
def test_run_kmeans_handles_single_cluster_silhouette_error():
    data = np.array([[0.0, 0.0], [0.0, 0.0]])
    config = ClusteringConfig(seed=3, fallback_n_clusters=2, enable_silhouette_selection=True)
    collection = run_kmeans(_collection_from_vectors(data), config)

    assert len(collection.cluster_ids()) == 2


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

    collection = run_kmeans(_collection_from_vectors(data), config)

    assert len(set(collection.cluster_ids())) == 3


def test_silhouette_sampling_adjusts_within_bounds(monkeypatch):
    vectors = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [1.0, 1.0],
        [1.1, 1.0],
        [2.0, 2.0],
        [2.1, 2.0],
    ])
    labels = np.array([0, 0, 1, 1, 2, 2])

    captured_sizes = []

    def fake_silhouette_score(_, __, *, sample_size=None, random_state=None):
        captured_sizes.append(sample_size)
        return 0.5

    monkeypatch.setattr("constella.clustering.kmeans.silhouette_score", fake_silhouette_score)

    rng = np.random.RandomState(0)

    _safe_silhouette_score(vectors, labels, None, rng, "context")
    _safe_silhouette_score(vectors, labels, 2, rng, "context")
    _safe_silhouette_score(vectors, labels, 50, rng, "context")

    assert captured_sizes[0] is None
    assert captured_sizes[1] == 4  # labels + 1
    assert captured_sizes[2] == 5  # n_samples - 1


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

    collection = run_kmeans(_collection_from_vectors(data), config)

    assert len(set(collection.cluster_ids())) == 3


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

    collection = run_kmeans(_collection_from_vectors(data), config)

    assert len(set(collection.cluster_ids())) == 6


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

    collection = run_kmeans(_collection_from_vectors(data), config)

    assert len(set(collection.cluster_ids())) == 3


def test_run_kmeans_accepts_dict_configuration():
    data = np.array([[0.0, 0.0], [1.0, 1.0], [1.1, 1.1], [0.1, 0.2]])

    config = ClusteringConfig(
        fallback_n_clusters=2,
        seed=99,
        enable_silhouette_selection=False,
        enable_elbow_selection=False,
        enable_davies_bouldin_selection=False,
    )
    collection = run_kmeans(_collection_from_vectors(data), config)

    assert len(set(collection.cluster_ids())) == 2


def test_run_kmeans_accepts_kwarg_overrides():
    data = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

    config = ClusteringConfig(fallback_n_clusters=3, seed=123)
    collection = run_kmeans(_collection_from_vectors(data), config)

    assert len(collection.cluster_ids()) == len(data)
