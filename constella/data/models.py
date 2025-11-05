"""Core data containers for the embedding and clustering workflow."""

from __future__ import annotations

from collections.abc import Iterable, MutableSequence, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterator, List, Optional


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

    def clone(self, *, deep: bool = False) -> "ContentUnitCollection":
        if deep:
            return ContentUnitCollection([deepcopy(unit) for unit in self._units])
        return ContentUnitCollection(self._units.copy())

    def units(self) -> List[ContentUnit]:
        return self._units

    def texts(self) -> List[str]:
        return self.all_texts()

    def all_texts(self) -> List[str]:
        combined_texts: List[str] = []
        for unit in self._units:
            parts: List[str] = []

            for attr in ("title", "name", "text"):
                value = getattr(unit, attr)
                if isinstance(value, str):
                    stripped_value = value.strip()
                    if stripped_value:
                        parts.append(stripped_value)

            if unit.path:
                parts.append(unit.path)

            for metadata in (unit.metadata1, unit.metadata2):
                if not metadata:
                    continue
                for meta_value in metadata.values():
                    if isinstance(meta_value, str):
                        stripped_meta_value = meta_value.strip()
                        if stripped_meta_value:
                            parts.append(stripped_meta_value)
                    elif meta_value is not None:
                        parts.append(str(meta_value))

            if not parts:
                parts.append(unit.identifier)

            combined_texts.append("\n".join(parts))

        return combined_texts

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
        iterator = iter(vectors)
        for unit in self._units:
            try:
                vector = next(iterator)
            except StopIteration as exc:  # pragma: no cover - defensive guard
                raise ValueError("Fewer embeddings than units were provided") from exc
            unit.set_embedding(vector)

        try:
            next(iterator)
        except StopIteration:
            return
        raise ValueError("More embeddings than units were provided")

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
        iterator = iter(cluster_ids)
        for unit in self._units:
            try:
                cid = next(iterator)
            except StopIteration as exc:  # pragma: no cover - defensive guard
                raise ValueError("Fewer cluster IDs than units were provided") from exc
            unit.set_cluster(int(cid))

        try:
            next(iterator)
        except StopIteration:
            return
        raise ValueError("More cluster IDs than units were provided")


__all__ = [
    "ContentUnit",
    "ContentUnitCollection",
]
