"""Configuration schemas for Constella workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
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

@dataclass
class VisualizationConfig:
    """Configuration for optional UMAP visualization generation."""

    output_path: Path
    show_plot: bool = False
    n_neighbors: int = 15
    min_dist: float = 0.1
    random_state: Optional[int] = None


@dataclass
class LabelingConfig:
    """Minimal configuration for future automatic cluster labeling."""

    provider: str = "fireworks"
    max_representatives: int = 5
    prompt_template: Optional[str] = None


@dataclass(frozen=True)
class ClusteringMetrics:
    """Snapshot of clustering diagnostics captured during K-Means execution."""

    n_clusters: int
    inertia: Optional[float] = None
    silhouette_score: Optional[float] = None
    centers: Optional[List[List[float]]] = None
    config_snapshot: Optional[ClusteringConfig] = None


@dataclass
class VisualizationArtifacts:
    """References to visualization artifacts generated during the workflow."""

    static_plot: Optional[Path] = None
    html_plot: Optional[Path] = None

    @property
    def generated(self) -> bool:
        return any((self.static_plot, self.html_plot))

    def __bool__(self) -> bool:  # pragma: no cover - truthiness convenience
        return self.generated
