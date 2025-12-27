"""K-Means clustering utilities used by the constella pipeline."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.random import RandomState
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

from constella.config.schemas import ClusteringConfig
from constella.data.results import ClusteringMetrics

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from constella.data.models import ContentUnitCollection


LOGGER = logging.getLogger(__name__)


# https://www.geeksforgeeks.org/machine-learning/ml-determine-the-optimal-value-of-k-in-k-means-clustering/

def _candidate_clusters(config: ClusteringConfig, sample_size: int) -> Sequence[int]:
    """Derive feasible cluster counts constrained by config limits and sample size.
    
    Generates a sequence of valid cluster counts based on configuration constraints
    and the available sample size. Ensures all candidates fall within valid bounds
    and includes a fallback option.
    
    Args:
        config: Clustering configuration containing min/max cluster constraints.
        sample_size: Number of data points available for clustering.
        
    Returns:
        Sorted sequence of valid cluster counts to evaluate. Falls back to a
        single-element list with the fallback cluster count if no valid range exists.
    """

    if sample_size <= 2:
        return [min(sample_size, max(1, config.fallback_n_clusters))]

    lower = max(2, min(config.min_cluster_count, sample_size - 1))
    upper = min(config.max_candidate_clusters, sample_size - 1)
    if upper < lower:
        upper = lower

    fallback = min(sample_size - 1, max(2, config.fallback_n_clusters))
    candidates = {fallback, *range(lower, upper + 1)}
    filtered = sorted(k for k in candidates if 1 < k < sample_size)
    return filtered or [fallback]


def _resolve_kmeans_init(vectors: np.ndarray) -> str:
    """Determine the appropriate initialization method for K-Means based on data range.
    
    Uses 'random' initialization for data with large value ranges (>1000) to avoid
    numerical instability, otherwise uses 'k-means++' for better convergence.
    
    Args:
        vectors: Input data array to analyze.
        
    Returns:
        Initialization method string: either 'random' or 'k-means++'.
    """
    data_range = np.max(vectors) - np.min(vectors)
    return 'random' if data_range > 1e3 else 'k-means++'


def _fit_kmeans(
    vectors: np.ndarray,
    n_clusters: int,
    seed: int,
    init: str,
) -> Tuple[KMeans, np.ndarray]:
    """Fit a K-Means model to the input vectors with suppressed warnings.
    
    Performs K-Means clustering with the specified parameters, suppressing
    sklearn RuntimeWarnings that may occur during convergence.
    
    Args:
        vectors: Input data array of shape (n_samples, n_features).
        n_clusters: Number of clusters to form.
        seed: Random state seed for reproducibility.
        init: Initialization method ('k-means++' or 'random').
        
    Returns:
        Tuple containing:
            - Fitted KMeans model instance.
            - Array of cluster labels for each input vector.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')
        model = KMeans(n_clusters=n_clusters, init=init, n_init=10, random_state=seed)
        labels = model.fit_predict(vectors)
    return model, labels


def _safe_silhouette_score(
    vectors: np.ndarray,
    labels: np.ndarray,
    sample_size: Optional[int],
    rng: RandomState,
    context: str,
) -> Optional[float]:
    """Safely compute silhouette score with error handling and sampling.
    
    Calculates the silhouette coefficient for clustering quality assessment,
    with robust error handling and optional sampling for large datasets.
    Returns None if computation fails or input is invalid.
    
    Args:
        vectors: Input data array of shape (n_samples, n_features).
        labels: Cluster labels for each sample.
        sample_size: Optional sample size for computation efficiency. If None,
            uses all samples.
        rng: Random number generator for sampling reproducibility.
        context: Descriptive string for error logging context.
        
    Returns:
        Silhouette score as float, or None if computation fails or is invalid.
    """
    unique_labels = np.unique(labels)
    if unique_labels.size <= 1 or unique_labels.size >= len(vectors):
        return None

    effective_sample = None
    if sample_size is not None and len(vectors) > 2:
        max_usable = len(vectors) - 1
        min_required = unique_labels.size + 1
        if max_usable > unique_labels.size and min_required < len(vectors):
            bounded = min(max(sample_size, min_required), max_usable)
            if unique_labels.size < bounded < len(vectors):
                effective_sample = int(bounded)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')
            score = silhouette_score(
                vectors,
                labels,
                sample_size=effective_sample,
                random_state=rng if effective_sample is not None else None,
            )
            return float(score)
    except (ValueError, FloatingPointError) as exc:
        LOGGER.error("Error calculating silhouette score %s: %s", context, exc)
    except Exception:  # pragma: no cover - unexpected numerical issues
        LOGGER.exception("Unexpected error calculating silhouette score %s", context)
    return None

# https://www.geeksforgeeks.org/machine-learning/davies-bouldin-index/
def _safe_davies_bouldin_score(
    vectors: np.ndarray,
    labels: np.ndarray,
    context: str,
) -> Optional[float]:
    """Safely compute Davies-Bouldin index with error handling.
    
    Calculates the Davies-Bouldin score to evaluate clustering quality.
    Lower scores indicate better clustering. Returns None if computation
    fails or input is invalid.
    
    Args:
        vectors: Input data array of shape (n_samples, n_features).
        labels: Cluster labels for each sample.
        context: Descriptive string for error logging context.
        
    Returns:
        Davies-Bouldin score as float, or None if computation fails or is invalid.
    """
    unique_labels = np.unique(labels)
    if unique_labels.size <= 1:
        return None

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')
            return float(davies_bouldin_score(vectors, labels))
    except (ValueError, FloatingPointError) as exc:
        LOGGER.error("Error calculating Davies-Bouldin score %s: %s", context, exc)
    except Exception:  # pragma: no cover - unexpected numerical issues
        LOGGER.exception("Unexpected error calculating Davies-Bouldin score %s", context)
    return None

# https://www.perplexity.ai/search/elbow-method-kmeans-scikit-lea-jHTIQx3cSFKklK4Dq1tWWw#0
def _select_elbow_k(candidates: Sequence[int], inertias: Sequence[float]) -> Optional[int]:
    """Select optimal cluster count using the elbow method.
    
    Applies the elbow method by finding the point of maximum perpendicular
    distance from the line connecting the first and last inertia values.
    This identifies where adding more clusters yields diminishing returns.
    
    Args:
        candidates: Sequence of cluster counts that were evaluated.
        inertias: Corresponding inertia (within-cluster sum of squares) values.
        
    Returns:
        Optimal cluster count as integer, or None if determination fails.
    """
    if len(candidates) < 2 or len(candidates) != len(inertias):
        return None

    k_values = np.asarray(candidates, dtype=float)
    wss = np.asarray(inertias, dtype=float)

    if np.isclose(k_values[0], k_values[-1]):
        return None

    slope = (wss[0] - wss[-1]) / (k_values[0] - k_values[-1])
    intercept = wss[0] - slope * k_values[0]
    baseline = slope * k_values + intercept
    deviations = baseline - wss

    idx = int(np.argmax(deviations))
    if deviations[idx] <= 0:
        return None
    return int(candidates[idx])


def _evaluate_candidate_metrics(
    vectors: np.ndarray,
    config: ClusteringConfig,
    candidates: Sequence[int],
) -> Dict[int, Dict[str, float]]:
    """Evaluate clustering quality metrics for all candidate cluster counts.
    
    Fits K-Means models for each candidate cluster count and computes
    configured quality metrics (inertia, silhouette score, Davies-Bouldin score).
    Skips candidates that cause numerical errors during fitting.
    
    Args:
        vectors: Input data array of shape (n_samples, n_features).
        config: Clustering configuration specifying which metrics to compute.
        candidates: Sequence of cluster counts to evaluate.
        
    Returns:
        Dictionary mapping cluster count to metric dictionary. Each metric dict
        contains 'inertia' and optionally 'silhouette' and 'davies_bouldin' scores.
    """
    sample_size = config.silhouette_sample_size if config.enable_silhouette_selection else None
    rng = RandomState(config.seed)
    init_method = _resolve_kmeans_init(vectors)

    metric_callbacks: List[Tuple[str, Callable[[np.ndarray, int], Optional[float]]]] = []

    if config.enable_silhouette_selection:
        def _silhouette(labels: np.ndarray, k: int) -> Optional[float]:
            score = _safe_silhouette_score(vectors, labels, sample_size, rng, f"for k={k}")
            if score is not None:
                LOGGER.debug("Silhouette score for k=%s: %s", k, score)
            return score

        metric_callbacks.append(("silhouette", _silhouette))

    if config.enable_davies_bouldin_selection:
        def _davies(labels: np.ndarray, k: int) -> Optional[float]:
            score = _safe_davies_bouldin_score(vectors, labels, f"for k={k}")
            if score is not None:
                LOGGER.debug("Davies-Bouldin score for k=%s: %s", k, score)
            return score

        metric_callbacks.append(("davies_bouldin", _davies))

    metrics: Dict[int, Dict[str, float]] = {}

    for k in candidates:
        if k >= len(vectors):
            continue
        try:
            km, labels = _fit_kmeans(vectors, k, config.seed, init_method)
        except (RuntimeWarning, FloatingPointError, Exception) as exc:
            LOGGER.error("Numerical error in K-Means fitting for k=%s: %s", k, exc)
            continue

        entry: Dict[str, float] = {"inertia": float(km.inertia_)}

        for name, callback in metric_callbacks:
            value = callback(labels, k)
            if value is not None:
                entry[name] = value

        metrics[k] = entry

    return metrics


def _best_metric_choice(
    metrics: Dict[int, Dict[str, float]],
    metric_name: str,
    prefer_max: bool,
) -> Optional[int]:
    """Select the cluster count with the best value for a specific metric.
    
    Searches through computed metrics to find the cluster count that optimizes
    the specified metric. For ties, prefers smaller cluster counts.
    
    Args:
        metrics: Dictionary mapping cluster count to metric values.
        metric_name: Name of the metric to optimize (e.g., 'silhouette', 'davies_bouldin').
        prefer_max: If True, select maximum metric value; if False, select minimum.
        
    Returns:
        Cluster count with optimal metric value, or None if metric not found.
    """
    best_value: Optional[float] = None
    best_k: Optional[int] = None

    for k, values in metrics.items():
        value = values.get(metric_name)
        if value is None:
            continue

        better = (
            best_value is None
            or (prefer_max and value > best_value)
            or (not prefer_max and value < best_value)
            or (value == best_value and best_k is not None and k < best_k)
        )

        if better:
            best_value = value
            best_k = k

    return best_k


def _select_cluster_count(vectors: np.ndarray, config: ClusteringConfig) -> int:
    """Determine optimal cluster count using multiple selection methods.
    
    Evaluates candidate cluster counts using configured methods (silhouette,
    Davies-Bouldin, elbow) and selects the maximum recommendation. Falls back
    to configured fallback value if no method produces a valid result.
    
    Args:
        vectors: Input data array of shape (n_samples, n_features).
        config: Clustering configuration specifying selection methods and constraints.
        
    Returns:
        Selected cluster count as integer.
    """
    candidates = _candidate_clusters(config, len(vectors))
    if not candidates:
        return min(len(vectors), max(2, config.fallback_n_clusters))

    metrics = _evaluate_candidate_metrics(vectors, config, candidates)
    if not metrics:
        LOGGER.warning("No viable cluster candidates succeeded; using fallback=%s", config.fallback_n_clusters)
        return min(len(vectors), max(2, config.fallback_n_clusters))

    silhouette_choice = (
        _best_metric_choice(metrics, "silhouette", prefer_max=True)
        if config.enable_silhouette_selection
        else None
    )
    davies_choice = (
        _best_metric_choice(metrics, "davies_bouldin", prefer_max=False)
        if config.enable_davies_bouldin_selection
        else None
    )

    if config.enable_silhouette_selection and silhouette_choice is None:
        LOGGER.warning("Silhouette selection produced no valid candidates; considering alternatives")
    if config.enable_davies_bouldin_selection and davies_choice is None:
        LOGGER.warning("Davies-Bouldin selection produced no valid candidates; considering alternatives")

    elbow_choice: Optional[int] = None
    if config.enable_elbow_selection:
        ordered = sorted(metrics.items())
        ks = [k for k, _ in ordered]
        inertias = [values["inertia"] for _, values in ordered if "inertia" in values]
        if len(ks) >= 2 and len(inertias) == len(ks):
            elbow_choice = _select_elbow_k(ks, inertias)
            if elbow_choice is not None:
                LOGGER.debug("Evaluated elbow inertias: %s", list(zip(ks, inertias)))
            else:
                LOGGER.warning("Elbow selection could not determine an optimal cluster count")
        else:
            LOGGER.debug("Insufficient candidates to perform elbow selection")

    choices = [value for value in (silhouette_choice, elbow_choice, davies_choice) if value is not None]
    if choices:
        final_k = max(choices)
    else:
        final_k = config.fallback_n_clusters

    LOGGER.info(
        "Selected cluster count: %s (silhouette=%s, elbow=%s, davies_bouldin=%s)",
        final_k,
        silhouette_choice,
        elbow_choice,
        davies_choice,
    )
    return final_k


def run_kmeans(
    collection: "ContentUnitCollection",
    config: Optional[ClusteringConfig] = None,
) -> "ContentUnitCollection":
    """Execute K-Means clustering on a content unit collection with automatic cluster selection.
    
    Performs complete K-Means clustering workflow: determines optimal cluster count
    using configured selection methods, fits the model, computes quality metrics,
    and attaches results to the collection.
    
    Args:
        collection: ContentUnitCollection containing embeddings to cluster.
            Must be non-empty with valid 2D embedding matrix.
        config: Optional clustering configuration. If None, uses default ClusteringConfig.
        
    Returns:
        The same collection with cluster IDs attached and metrics populated.
        
    Raises:
        ValueError: If collection is empty or embeddings are invalid.
    """

    if len(collection) == 0:
        raise ValueError("Input collection must not be empty.")

    embeddings = collection.embedding_matrix()
    array = np.array(list(embeddings), dtype=np.float64)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("Embeddings must form a non-empty 2D array.")

    resolved_config = config or ClusteringConfig()

    n_clusters = _select_cluster_count(array, resolved_config)
    n_clusters = min(n_clusters, len(array))

    init_method = _resolve_kmeans_init(array)

    try:
        km, labels = _fit_kmeans(array, n_clusters, resolved_config.seed, init_method)
    except (RuntimeWarning, FloatingPointError, Exception) as exc:  # pragma: no cover - sklearn edge cases
        LOGGER.error("Critical numerical error in main K-Means fitting: %s", exc)
        raise

    silhouette = _safe_silhouette_score(
        array,
        labels,
        resolved_config.silhouette_sample_size,
        RandomState(resolved_config.seed),
        "for final clustering",
    )

    LOGGER.info("Clustering completed with inertia=%s", km.inertia_)
    if silhouette is not None:
        LOGGER.info("Silhouette score=%.4f", silhouette)

    collection.attach_cluster_ids(labels.tolist())
    collection.metrics = ClusteringMetrics(
        n_clusters=int(n_clusters),
        inertia=float(km.inertia_),
        silhouette_score=float(silhouette) if silhouette is not None else None,
        config_snapshot=resolved_config,
    )

    return collection
