"""Core data containers for the embedding and clustering workflow."""

from __future__ import annotations

from collections.abc import Iterable, MutableSequence, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

from constella.data.results import (
    ClusteringMetrics,
    LabelResult,
    RepresentativeSample,
    VisualizationArtifacts,
)


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
    # Metadata 1 will be used in embedding and printing
    metadata1: Dict[str, Any] = field(default_factory=dict)
    # Metadata 2 will NOT be used in embedding and printing
    metadata2: Dict[str, Any] = field(default_factory=dict)
    cluster_id: Optional[int] = None

    def set_embedding(self, emb: Optional[Sequence[float]]) -> None:
        self.embedding = list(emb) if emb is not None else None

    def set_cluster(self, cid: int | None) -> None:
        self.cluster_id = int(cid) if cid is not None else None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get_content(self, max_char_to_truncate: Optional[int] = None) -> str:
        """Return a formatted view of the content unit for presentation/prompting.

        Parameters
        ----------
        max_char_to_truncate:
            Optional limit applied only to the text body; other attributes remain
            untouched. When set and the text exceeds the limit, the text body is
            truncated.
        """

        lines: List[str] = []

        def _append(label: str, value: Any) -> None:
            if value is None:
                return
            string_value = str(value).strip()
            if string_value:
                lines.append(f"{label}: {string_value}")

        _append("Id", self.identifier)
        _append("Title", self.title)
        _append("Name", self.name)
        _append("Path", self.path)
        _append("Size", self.size)
        if self.cluster_id is not None:
            _append("Cluster ID", self.cluster_id)

        if self.text:
            text_value = self.text
            if max_char_to_truncate is not None and len(text_value) > max_char_to_truncate:
                text_value = text_value[:max_char_to_truncate].rstrip() + "..."
            _append("Text", text_value)

        for key in sorted(self.metadata1):
            value = self.metadata1[key]
            if value is not None:
                _append(f"Metadata.{key}", value)

        if not lines:
            lines.append(f"Identifier: {self.identifier}")

        return "\n".join(lines)


class ContentUnitCollection(MutableSequence[ContentUnit]):
    """Container managing ordered collections of ``ContentUnit`` objects."""

    def __init__(self, units: Iterable[ContentUnit]):
        self._units: List[ContentUnit] = list(units or [])
        self._metrics: Optional[ClusteringMetrics] = None
        self._artifacts: Optional[VisualizationArtifacts] = None
        self._representatives: Dict[int, List[RepresentativeSample]] = {}
        self._label_results: Dict[int, LabelResult] = {}

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

    def all_texts(
        self,
        *,
        max_char_to_truncate: Optional[int] = None,
    ) -> List[str]:

        return [
            unit.get_content(max_char_to_truncate=max_char_to_truncate)
            for unit in self._units
        ]

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

    @property
    def unique_cluster_ids(self) -> List[int]:
        """Return sorted unique cluster identifiers present among units."""

        unique: set[int] = set()
        for unit in self._units:
            cid = unit.cluster_id
            if cid is None:
                continue
            try:
                unique.add(int(cid))
            except (TypeError, ValueError):
                continue
        return sorted(unique)

    @property
    def unique_cluster_labels(self) -> List[str]:
        """Return visual labels corresponding to the unique cluster identifiers."""

        unique_cluster_ids = self.unique_cluster_ids
        return [self.get_visual_label_for_cluster_id(cid) for cid in unique_cluster_ids]

    @property
    def representatives(self) -> Dict[int, List[RepresentativeSample]]:
        return self._representatives

    @representatives.setter
    def representatives(self, value: Optional[Dict[int, List[RepresentativeSample]]]) -> None:
        self._representatives = value or {}

    @property
    def label_results(self) -> Dict[int, LabelResult]:
        return self._label_results

    @label_results.setter
    def label_results(self, value: Optional[Dict[int, LabelResult]]) -> None:
        self._label_results = value or {}

    def _attach_sequence(
        self,
        values: Iterable[Any],
        setter: Callable[[ContentUnit, Any], None],
        label: str,
    ) -> None:
        """Apply values to units via *setter*, enforcing a one-to-one pairing."""

        iterator = iter(values)
        for unit in self._units:
            try:
                value = next(iterator)
            except StopIteration as exc:  # pragma: no cover - defensive guard
                raise ValueError(f"Fewer {label} than units were provided") from exc
            setter(unit, value)

        if any(True for _ in iterator):  # pragma: no cover - defensive guard
            raise ValueError(f"More {label} than units were provided")

    @property
    def cluster_size_lookup(self) -> Dict[int, int]:
        """Count units per cluster id, ignoring units without valid assignments."""

        sizes: Dict[int, int] = {}
        for unit in self._units:
            cid = unit.cluster_id
            if cid is None:
                continue
            try:
                key = int(cid)
            except (TypeError, ValueError):
                continue
            sizes[key] = sizes.get(key, 0) + 1
        return sizes

    def get_visual_label_for_cluster_id(self, cluster_id: Optional[int]) -> str:
        """Return a user-facing label for the provided cluster identifier."""

        if cluster_id is None:
            return "Unassigned"

        try:
            lookup_id = int(cluster_id)
        except (TypeError, ValueError):
            return "Unassigned"

        label_result = (self.label_results or {}).get(lookup_id)
        if label_result and label_result.label.strip():
            return label_result.label.strip()
        return f"Cluster {lookup_id}"

    def iter_visual_labels(self) -> List[str]:
        """Resolve display labels for every unit in the collection."""

        target_units = list(self._units)
        label_lookup = dict(zip(self.unique_cluster_ids, self.unique_cluster_labels))
        default_label = self.get_visual_label_for_cluster_id(None)
        resolved: List[str] = []
        for unit in target_units:
            cid = unit.cluster_id
            if cid is None:
                resolved.append(default_label)
                continue
            try:
                lookup_id = int(cid)
            except (TypeError, ValueError):
                resolved.append(default_label)
                continue
            resolved.append(
                label_lookup.get(lookup_id, self.get_visual_label_for_cluster_id(lookup_id))
            )
        return resolved

    @property
    def visual_labels(self) -> List[str]:
        """Return display labels aligned with ``units()`` ordering."""

        return self.iter_visual_labels()


__all__ = [
    "ContentUnit",
    "ContentUnitCollection",
]
