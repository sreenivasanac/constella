"""Configuration schemas for clustering and visualization workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ClusteringConfig:
    """Settings controlling the deterministic K-Means workflow."""

    seed: int = 42
    fallback_n_clusters: int = 5
    min_cluster_count: int = 2
    max_candidate_clusters: int = 10
    enable_silhouette_selection: bool = False
    silhouette_sample_size: Optional[int] = None


@dataclass(frozen=True)
class VisualizationConfig:
    """Configuration for optional UMAP visualization generation."""

    output_path: Path
    show_plot: bool = False
    n_neighbors: int = 15
    min_dist: float = 0.1
    random_state: Optional[int] = None
