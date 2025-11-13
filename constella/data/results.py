"""Result data structures produced by workflow operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from constella.config.schemas import ClusteringConfig


@dataclass
class RepresentativeSample:
    """Representative sample metadata for a cluster."""

    cluster_id: int
    unit_index: int
    distance: float
    similarity: float
    is_core: bool


@dataclass
class LabelResult:
    """Structured label output returned from the LLM."""

    cluster_id: int
    label: str
    explanation: str
    confidence: float
    keywords: List[str] = field(default_factory=list)
    raw_response: Optional[str] = None
    usage_metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class ClusteringMetrics:
    """Snapshot of clustering diagnostics captured during K-Means execution."""

    n_clusters: int
    inertia: Optional[float] = None
    silhouette_score: Optional[float] = None
    config_snapshot: Optional[ClusteringConfig] = None


@dataclass
class VisualizationArtifacts:
    """References to visualization artifacts generated during the workflow."""

    static_plot: Optional[Path] = None
    html_plot: Optional[Path] = None
    labels_json: Optional[Path] = None

    @property
    def generated(self) -> bool:
        return any((self.static_plot, self.html_plot, self.labels_json))

    def __bool__(self) -> bool:  # pragma: no cover - truthiness convenience
        return self.generated


__all__ = [
    "RepresentativeSample",
    "LabelResult",
    "ClusteringMetrics",
    "VisualizationArtifacts",
]
