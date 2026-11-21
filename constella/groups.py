"""Embedding groups: a way to SELECT items, never a label to apply wholesale.

A host groups a set, shows the groups, picks one, and may group that subset
again. Every label is still checked per item afterwards (see ``loop``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import MiniBatchKMeans

MAX_GROUPS = 40  # 40 groups x 3 samples of 100 characters fits an overview of about 4,000 tokens
ITEMS_PER_GROUP = 8  # fewer items than this per group and the groups stop being worth reading
N_CENTRAL = 3
N_SPREAD = 2
DEFAULT_SEED = 0
KMEANS_BATCH = 1024
KMEANS_INITS = 3


@dataclass
class Group:
    id: int  # 0 is the largest group; stable for the same input and seed
    ids: list  # member ids, in input order
    size: int
    centroid: np.ndarray  # unit-length float32
    cohesion: float  # mean cosine similarity of the members to the centroid; 1.0 is as tight as it gets
    central: list  # ids of the members nearest the centre, nearest first
    spread: list  # ids of a few members far from the centre and from each other


def normalise(vectors: np.ndarray) -> np.ndarray:
    """Return the rows as unit-length float32. Rejects anything that is not a 2-D numpy array."""
    if not isinstance(vectors, np.ndarray) or vectors.ndim != 2:
        raise TypeError("vectors must be a 2-D numpy array (float32), not a Python list")
    v = vectors.astype(np.float32, copy=False)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(norms == 0, 1, norms)


def choose_k(n: int, max_groups: int = MAX_GROUPS) -> int:
    """k comes from what the reader can take in (the token budget), not from the data."""
    return max(1, min(max_groups, math.ceil(n / ITEMS_PER_GROUP)))


def _spread(members: np.ndarray, centroid: np.ndarray, skip: set, n: int) -> list[int]:
    """Farthest-point picks: start far from the centre, then keep taking the member
    farthest from everything picked so far. Shows the edges of a group."""
    free = [i for i in range(len(members)) if i not in skip]
    if not free or n <= 0:
        return []
    sims = members[free] @ centroid
    picked = [free[int(np.argmin(sims))]]
    nearest = members[free] @ members[picked[0]]  # highest similarity to anything picked
    while len(picked) < min(n, len(free)):
        nxt = int(np.argmin(nearest))
        if free[nxt] in picked:
            break
        picked.append(free[nxt])
        nearest = np.maximum(nearest, members[free] @ members[free[nxt]])
    return picked


def group(
    vectors: np.ndarray,
    ids: list,
    k: int | None = None,
    max_groups: int = MAX_GROUPS,
    seed: int = DEFAULT_SEED,
    n_central: int = N_CENTRAL,
    n_spread: int = N_SPREAD,
) -> list[Group]:
    """Cut the items into groups with mini-batch K-Means on unit vectors.

    Pass any subset (its rows and its ids) to group a group again. The same
    input and seed always give the same groups with the same ids.
    """
    v = normalise(vectors)
    if len(ids) != len(v):
        raise ValueError("ids and vectors must have the same length")
    if len(v) == 0:
        return []
    k = min(len(v), k or choose_k(len(v), max_groups))
    if k == 1:
        assigned = np.zeros(len(v), dtype=int)
    else:
        km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=KMEANS_INITS, batch_size=KMEANS_BATCH)
        assigned = km.fit_predict(v)

    rows_by_cluster = [np.flatnonzero(assigned == c) for c in range(k)]
    # largest first; ties broken by first member, so ids do not depend on K-Means' own numbering
    rows_by_cluster = sorted((r for r in rows_by_cluster if len(r)), key=lambda r: (-len(r), r[0]))

    groups = []
    for gid, rows in enumerate(rows_by_cluster):
        members = v[rows]
        centroid = members.mean(axis=0)
        centroid /= np.linalg.norm(centroid) or 1
        sims = members @ centroid
        central = np.argsort(-sims, kind="stable")[:n_central]
        spread = _spread(members, centroid, set(central.tolist()), n_spread)
        groups.append(
            Group(
                id=gid,
                ids=[ids[i] for i in rows],
                size=len(rows),
                centroid=centroid.astype(np.float32),
                cohesion=float(sims.mean()),
                central=[ids[rows[i]] for i in central],
                spread=[ids[rows[i]] for i in spread],
            )
        )
    return groups


def nearest_groups(groups: list[Group], n: int = 2) -> dict[int, list[tuple[int, float]]]:
    """For each group, the n closest other groups as (group id, cosine similarity of centroids)."""
    if len(groups) < 2:
        return {g.id: [] for g in groups}
    centroids = np.stack([g.centroid for g in groups])
    sims = centroids @ centroids.T
    np.fill_diagonal(sims, -np.inf)
    order = np.argsort(-sims, axis=1, kind="stable")[:, :n]
    return {g.id: [(groups[j].id, float(sims[i, j])) for j in order[i]] for i, g in enumerate(groups)}
