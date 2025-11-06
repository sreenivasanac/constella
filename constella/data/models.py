"""Core data containers for the embedding and clustering workflow."""

from __future__ import annotations

from collections.abc import Iterable, MutableSequence, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

from constella.config.schemas import ClusteringMetrics, VisualizationArtifacts


@dataclass
class ContentUnit:
    """Represents a text unit subject to embedding and clustering with additional attributes."""

    identifier: str
    text: str
    name: Optional[str] = None
    title: Optional[str] = None
    path: Optional[str] = None
    embedding: Optional[List[float]] = None
    # Pass a human-readable size description (e.g., "5 MB", "100 characters")
    size: Optional[str] = None
    # Metadata 1 will be used in embedding
    metadata1: Dict[str, Any] = field(default_factory=dict)
    # Metadata 2 will NOT be used in embedding
    metadata2: Dict[str, Any] = field(default_factory=dict)
    cluster_id: Optional[int] = None

    def set_embedding(self, emb: Optional[Sequence[float]]) -> None:
        self.embedding = list(emb) if emb is not None else None

    def set_cluster(self, cid: int | None) -> None:
        self.cluster_id = int(cid) if cid is not None else None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ContentUnitCollection(MutableSequence[ContentUnit]):
    """Container managing ordered collections of ``ContentUnit`` objects."""

    def __init__(self, units: Iterable[ContentUnit]):
        self._units: List[ContentUnit] = list(units or [])
        self._metrics: Optional[ClusteringMetrics] = None
        self._artifacts: Optional[VisualizationArtifacts] = None

    # -- MutableSequence interface -------------------------------------------------
    def __getitem__(self, index: int) -> ContentUnit:
        return self._units[index]

    def __setitem__(self, index: int, value: ContentUnit) -> None:
        self._validate_unit(value)
        self._units[index] = value

    def __delitem__(self, index: int) -> None:
        del self._units[index]

    def __len__(self) -> int:  # pragma: no cover - delegation to list
        return len(self._units)

    def insert(self, index: int, value: ContentUnit) -> None:
        self._validate_unit(value)
        self._units.insert(index, value)

    def __iter__(self) -> Iterator[ContentUnit]:  # pragma: no cover - delegation to list
        return iter(self._units)

    # -- Helper methods ------------------------------------------------------------
    @staticmethod
    def _validate_unit(unit: ContentUnit) -> None:
        if not isinstance(unit, ContentUnit):
            raise TypeError("ContentUnitCollection elements must be ContentUnit instances")


    def units(self) -> List[ContentUnit]:
        return self._units

    @property
    def metrics(self) -> Optional[ClusteringMetrics]:
        return self._metrics

    @metrics.setter
    def metrics(self, value: Optional[ClusteringMetrics]) -> None:
        self._metrics = value

    @property
    def artifacts(self) -> Optional[VisualizationArtifacts]:
        return self._artifacts

    @artifacts.setter
    def artifacts(self, value: Optional[VisualizationArtifacts]) -> None:
        self._artifacts = value

    def texts(self) -> List[str]:
        return self.all_texts()

    def all_texts(self) -> List[str]:
        def _normalize(value: Any) -> Optional[str]:
            if isinstance(value, str):
                stripped = value.strip()
                return stripped or None
            if value is not None:
                return str(value)
            return None

        combined: List[str] = []
        for unit in self._units:
            parts = [
                fragment
                for attr in ("title", "name", "text")
                for fragment in (_normalize(getattr(unit, attr)),)
                if fragment
            ]

            path_fragment = _normalize(unit.path)
            if path_fragment:
                parts.append(path_fragment)

            for metadata in (unit.metadata1, unit.metadata2):
                if metadata:
                    parts.extend(
                        fragment
                        for fragment in (_normalize(val) for val in metadata.values())
                        if fragment
                    )

            if not parts:
                parts.append(unit.identifier)

            combined.append("\n".join(parts))

        return combined

    def identifiers(self) -> List[str]:
        return [unit.identifier for unit in self._units]

    # -- Embedding helpers ---------------------------------------------------------
    def ensure_embeddings(self) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for unit in self._units:
            if unit.embedding is None:
                raise ValueError(f"Unit {unit.identifier} is missing an embedding")
            embeddings.append(list(unit.embedding))
        return embeddings

    def embedding_matrix(self) -> List[List[float]]:
        return self.ensure_embeddings()

    def attach_embeddings(self, vectors: Iterable[Sequence[float]]) -> None:
        self._attach_sequence(vectors, ContentUnit.set_embedding, "embeddings")

    # -- Clustering helpers --------------------------------------------------------
    def ensure_cluster_assignments(self) -> List[int]:
        assignments: List[int] = []
        for unit in self._units:
            if unit.cluster_id is None:
                raise ValueError(f"Unit {unit.identifier} is missing a cluster assignment")
            assignments.append(int(unit.cluster_id))
        return assignments

    def cluster_ids(self) -> List[int]:
        return self.ensure_cluster_assignments()

    def attach_cluster_ids(self, cluster_ids: Iterable[int]) -> None:
        def _assign(unit: ContentUnit, cid: int) -> None:
            unit.set_cluster(int(cid))

        self._attach_sequence(cluster_ids, _assign, "cluster IDs")

    def _attach_sequence(
        self,
        values: Iterable[Any],
        setter: Callable[[ContentUnit, Any], None],
        label: str,
    ) -> None:
        iterator = iter(values)
        for unit in self._units:
            try:
                value = next(iterator)
            except StopIteration as exc:  # pragma: no cover - defensive guard
                raise ValueError(f"Fewer {label} than units were provided") from exc
            setter(unit, value)

        if any(True for _ in iterator):  # pragma: no cover - defensive guard
            raise ValueError(f"More {label} than units were provided")


__all__ = [
    "ContentUnit",
    "ContentUnitCollection",
]
