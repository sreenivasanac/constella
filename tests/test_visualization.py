"""Tests for visualization utilities."""

from __future__ import annotations

import numpy as np

from constella.config.schemas import VisualizationConfig
from constella.visualization.umap import project_embeddings, save_umap_plot


def test_project_embeddings_shape(tmp_path):
    vectors = np.array([
        [0.0, 0.0, 0.1],
        [1.0, 0.9, 1.1],
        [0.1, 0.2, 0.0],
    ])
    config = VisualizationConfig(output_path=tmp_path / "plot.png", random_state=7)
    projection = project_embeddings(vectors, config)

    assert projection.shape == (3, 2)


def test_save_umap_plot_creates_file(tmp_path):
    projection = np.array([[0.0, 0.0], [1.0, 1.0]])
    labels = [0, 1]
    config = VisualizationConfig(output_path=tmp_path / "plot.png", random_state=7)
    path = save_umap_plot(projection, labels, config)

    assert path.exists()
