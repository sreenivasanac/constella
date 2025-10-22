"""Core data containers for the embedding and clustering workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from constella.config.schemas import ClusteringConfig


@dataclass(frozen=True)
class ContentUnit:
    """Represents a text unit subject to embedding and clustering."""

    identifier: str
    text: str


@dataclass(frozen=True)
class EmbeddingVector:
    """Embedding vector aligned with a content unit."""

    unit_id: str
    values: Sequence[float]


@dataclass(frozen=True)
class ClusterAssignment:
    """Output produced by the clustering stage."""

    assignments: Sequence[int]
    centers: Sequence[Sequence[float]]
    inertia: float
    silhouette_score: float | None
    config_snapshot: ClusteringConfig
