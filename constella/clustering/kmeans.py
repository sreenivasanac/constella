"""K-Means clustering utilities used by the constella pipeline."""

from __future__ import annotations

import logging
import warnings
from typing import Iterable, Sequence

import numpy as np
from numpy.random import RandomState
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from constella.config.schemas import ClusteringConfig
from constella.data.models import ClusterAssignment


LOGGER = logging.getLogger(__name__)


def _candidate_clusters(config: ClusteringConfig, sample_size: int) -> Sequence[int]:
    max_candidates = min(config.max_candidate_clusters, sample_size - 1)
    min_candidates = max(config.min_cluster_count, 2)
    if max_candidates < min_candidates:
        max_candidates = min_candidates
    base_range = list(range(min_candidates, max_candidates + 1))
    if config.fallback_n_clusters not in base_range:
        base_range.append(min(config.fallback_n_clusters, sample_size))
    unique = sorted({k for k in base_range if 1 < k <= sample_size})
    return unique or [min(sample_size, max(2, config.fallback_n_clusters))]


def _select_cluster_count(vectors: np.ndarray, config: ClusteringConfig) -> int:
    candidates = _candidate_clusters(config, len(vectors))
    if not config.enable_silhouette_selection or len(candidates) == 1:
        return config.fallback_n_clusters

    rs = RandomState(config.seed)
    sample_size = config.silhouette_sample_size
    best_score = -1.0
    best_k = config.fallback_n_clusters

    # Check for potential numerical issues
    data_range = np.max(vectors) - np.min(vectors)
    init_method = 'random' if data_range > 1e3 else 'k-means++'

    for k in candidates:
        if k >= len(vectors):
            continue
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')
                km = KMeans(n_clusters=k, init=init_method, random_state=config.seed, n_init=10)
                labels = km.fit_predict(vectors)
        except (RuntimeWarning, FloatingPointError, Exception) as e:
            LOGGER.error(f"Numerical error in K-Means fitting for k={k}: {e}")
            continue

        if len(set(labels)) == 1:
            continue
        try:
            score = silhouette_score(
                vectors,
                labels,
                sample_size=sample_size,
                random_state=rs,
            )
        except (ValueError, RuntimeWarning, FloatingPointError, Exception) as e:
            LOGGER.error(f"Error in silhouette score calculation for k={k}: {e}")
            continue
        LOGGER.debug("Silhouette score for k=%s: %s", k, score)
        if score > best_score:
            best_score = score
            best_k = k

    LOGGER.info("Selected cluster count: %s", best_k)
    return best_k


def run_kmeans(vectors: Iterable[Sequence[float]], config: ClusteringConfig) -> ClusterAssignment:
    """Execute deterministic K-Means and return assignment metadata."""

    array = np.array(list(vectors), dtype=np.float64)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("Input vectors must form a non-empty 2D array.")

    means = np.mean(array, axis=0, keepdims=True)
    array = array - means

    stds = np.std(array, axis=0, keepdims=True)
    stds[stds == 0] = 1.0
    array = array / stds

    n_clusters = _select_cluster_count(array, config)
    n_clusters = min(n_clusters, len(array))
    
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')
            kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=config.seed)
            labels = kmeans.fit_predict(array)
    except (RuntimeWarning, FloatingPointError, Exception) as e:
        LOGGER.error(f"Critical numerical error in main K-Means fitting: {e}")
        raise

    silhouette = None
    if len(set(labels)) > 1:
        try:
            silhouette = silhouette_score(array, labels)
        except (ValueError, RuntimeWarning, FloatingPointError, Exception) as e:
            LOGGER.error(f"Error calculating final silhouette score: {e}")
            silhouette = None

    LOGGER.info("Clustering completed with inertia=%s", kmeans.inertia_)

    return ClusterAssignment(
        assignments=labels.tolist(),
        centers=kmeans.cluster_centers_.tolist(),
        inertia=float(kmeans.inertia_),
        silhouette_score=float(silhouette) if silhouette is not None else None,
        config_snapshot=config,
    )
