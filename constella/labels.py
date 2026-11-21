"""Labels, the nearest-labels shortlist and the assignment rule. No I/O, no model calls."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from constella.groups import normalise

SHORTLIST = 8  # Jev's best label was among the 8 nearest by embedding for 96.6% of items (v4, measured)
ACCEPT = 0.8  # labels accepted at 0.8 were about 97% correct (v4, measured)
DOUBTFUL_BELOW = 0.9  # accepted answers under this were right 17 times in 20, so they are worth a spot check
PATH_SEP = "/"


@dataclass(frozen=True)
class Label:
    id: str
    path: str  # folder path, e.g. "AI/RAG"
    definition: str  # one sentence; Jev reads it literally, so the definition is the label
    seed: bool = False  # came from the user or an import, not proposed by an agent


@dataclass
class Assignment:
    item_id: object
    primary: tuple[str, float] | None = None  # (label id, p): highest p at or above the cut-off
    secondary: tuple[str, float] | None = None  # the next one
    others: list[tuple[str, float]] = field(default_factory=list)  # every further accepted label
    doubtful: list[str] = field(default_factory=list)  # accepted label ids with p under DOUBTFUL_BELOW

    @property
    def pending(self) -> bool:
        return self.primary is None

    def label_ids(self) -> list[str]:
        return [x[0] for x in (self.primary, self.secondary, *self.others) if x]


def shortlist(item_vectors: np.ndarray, label_vectors: np.ndarray, n: int = SHORTLIST) -> np.ndarray:
    """Indices of the n labels nearest each item (cosine), nearest first. Shape (items, min(n, labels)).

    The host supplies both matrices; a label's vector is the embedding of its definition.
    """
    sims = normalise(item_vectors) @ normalise(label_vectors).T
    return np.argsort(-sims, axis=1, kind="stable")[:, :n]


def assign(
    scores: Mapping[object, Mapping[str, float]],
    accept: float = ACCEPT,
    doubtful_below: float = DOUBTFUL_BELOW,
) -> dict[object, Assignment]:
    """Turn per-item answers {item id: {label id: p}} into primary, secondary and other labels.

    An item with no label at or above ``accept`` is pending. Changing ``accept`` re-asks nothing.
    """
    out = {}
    for item_id, by_label in scores.items():
        passed = sorted(((lid, p) for lid, p in by_label.items() if p >= accept), key=lambda x: (-x[1], x[0]))
        out[item_id] = Assignment(
            item_id=item_id,
            primary=passed[0] if passed else None,
            secondary=passed[1] if len(passed) > 1 else None,
            others=passed[2:],
            doubtful=[lid for lid, p in passed if p < doubtful_below],
        )
    return out
