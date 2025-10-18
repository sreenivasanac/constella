"""Utilities to project embeddings and optionally persist UMAP plots."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:  # pragma: no cover - exercised indirectly in environments with umap installed
    import umap  # type: ignore

    _HAS_UMAP = True
except ModuleNotFoundError:  # pragma: no cover - handled in tests
    from sklearn.decomposition import PCA

    umap = None
    _HAS_UMAP = False

from constella.config.schemas import VisualizationConfig


LOGGER = logging.getLogger(__name__)


def project_embeddings(
    vectors: Sequence[Sequence[float]],
    config: VisualizationConfig,
) -> np.ndarray:
    """Project embeddings into 2D using UMAP when available, else PCA fallback."""

    array = np.array(vectors, dtype=float)
    if array.ndim != 2 or array.shape[0] < 2:
        raise ValueError("Need at least two embeddings for UMAP projection.")

    if _HAS_UMAP:
        reducer = umap.UMAP(
            n_neighbors=config.n_neighbors,
            min_dist=config.min_dist,
            random_state=config.random_state,
        )
        LOGGER.info("Generating UMAP projection for %s embeddings", array.shape[0])
        return reducer.fit_transform(array)

    LOGGER.warning("UMAP unavailable; using PCA fallback for projection.")
    pca = PCA(n_components=2)
    return pca.fit_transform(array)


def save_umap_plot(
    projection: np.ndarray,
    labels: Sequence[int],
    config: VisualizationConfig,
    title: Optional[str] = None,
) -> Path:
    """Persist a scatter plot of the UMAP projection."""

    if projection.shape[1] != 2:
        raise ValueError("Projection must have exactly two dimensions for plotting.")

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Saving UMAP plot to %s", output_path)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(projection[:, 0], projection[:, 1], c=labels, cmap="Spectral", s=20)
    plt.colorbar(scatter, label="Cluster")
    if title:
        plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    if config.show_plot:
        plt.show()

    return output_path
