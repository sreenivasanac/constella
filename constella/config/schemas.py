"""Configuration schemas for Constella workflows."""

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
    enable_silhouette_selection: bool = True
    silhouette_sample_size: Optional[int] = None
    enable_elbow_selection: bool = True
    enable_davies_bouldin_selection: bool = True

@dataclass(frozen=True)
class VisualizationConfig:
    """Configuration for optional UMAP visualization generation."""

    output_path: Path
    show_plot: bool = False
    n_neighbors: int = 15
    min_dist: float = 0.1
    random_state: Optional[int] = None


@dataclass(frozen=True)
class LabelingConfig:
    """Placeholder configuration for future automatic cluster labeling."""

    test_labeling_config_parameter = None
