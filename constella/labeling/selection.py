"""Representative sampling helpers for clusters."""

from __future__ import annotations

import random
from typing import Dict, List, Optional

import numpy as np

from constella.config.schemas import RepresentativeSelectionConfig
from constella.data.models import ContentUnitCollection
from constella.data.results import RepresentativeSample


def select_representatives(
    collection: ContentUnitCollection,
    config: Optional[RepresentativeSelectionConfig] = None,
) -> Dict[int, List[RepresentativeSample]]:
    """Return representative samples for each cluster in the collection.

    The algorithm prioritises the closest points to the centroid ("core" samples)
    and optionally mixes in randomly sampled items from the remaining members to
    preserve diversity.
    """

    if len(collection) == 0:
        return {}

    resolved = config or RepresentativeSelectionConfig()

    embeddings = np.array(collection.embedding_matrix(), dtype=np.float64)
    if embeddings.ndim != 2 or embeddings.size == 0:
        raise ValueError("Embeddings must form a non-empty 2D array for selection.")

    labels = np.array(collection.cluster_ids(), dtype=int)
    unique_clusters = sorted(int(label) for label in set(labels.tolist()))

    representatives: Dict[int, List[RepresentativeSample]] = {}

    for cluster_id in unique_clusters:
        mask = labels == cluster_id
        indices = np.nonzero(mask)[0]
        if indices.size == 0:
            continue

        cluster_vectors = embeddings[indices]
        centroid = cluster_vectors.mean(axis=0)
        deltas = cluster_vectors - centroid
        distances = np.linalg.norm(deltas, axis=1)
        similarities = 1.0 / (1.0 + distances)

        ordered_local_indices = np.argsort(distances)
        n_in_cluster = len(indices)
        if n_in_cluster == 0:
            continue

        n_eff = min(n_in_cluster, max(1, resolved.n_representatives))
        n_core = min(n_eff, max(1, int(round(n_eff * resolved.core_ratio))))
        n_core = max(1, min(n_core, n_eff))
        n_diverse = max(0, n_eff - n_core)

        ordered_pairs = [(int(local_idx), int(indices[local_idx])) for local_idx in ordered_local_indices]
        core_pairs = ordered_pairs[:n_core]
        remaining_pairs = ordered_pairs[n_core:]

        if resolved.diversity_sampling and n_diverse > 0 and remaining_pairs:
            rng = random.Random(resolved.random_seed + int(cluster_id))
            if len(remaining_pairs) > n_diverse:
                selected_pairs = rng.sample(remaining_pairs, n_diverse)
            else:
                selected_pairs = remaining_pairs
        else:
            selected_pairs = remaining_pairs[:n_diverse]

        diverse_pairs = sorted(selected_pairs, key=lambda pair: distances[pair[0]])

        samples: List[RepresentativeSample] = []

        for local_idx, global_idx in core_pairs:
            samples.append(
                RepresentativeSample(
                    cluster_id=cluster_id,
                    unit_index=global_idx,
                    distance=float(distances[local_idx]),
                    similarity=float(similarities[local_idx]),
                    is_core=True,
                )
            )

        for local_idx, global_idx in diverse_pairs:
            samples.append(
                RepresentativeSample(
                    cluster_id=cluster_id,
                    unit_index=global_idx,
                    distance=float(distances[local_idx]),
                    similarity=float(similarities[local_idx]),
                    is_core=False,
                )
            )

        representatives[cluster_id] = samples

    collection.representatives = representatives
    return representatives


__all__ = [
    "select_representatives",
]
