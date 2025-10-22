"""Tests for visualization utilities."""

from __future__ import annotations

import numpy as np
import pytest

from constella.config.schemas import VisualizationConfig
from constella.data.models import ContentUnit
from constella.visualization.umap import create_umap_plot_html, project_embeddings, save_umap_plot


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


def test_create_umap_plot_html_generates_hoverable_artifact(tmp_path):
    projection = np.array([[0.0, 0.0], [1.0, 1.0]])
    labels = [0, 1]
    units = [
        ContentUnit(identifier="unit_a", text="alpha"),
        ContentUnit(identifier="unit_b", text="beta"),
    ]
    config = VisualizationConfig(output_path=tmp_path / "plot.png", random_state=7)

    html_path = create_umap_plot_html(
        projection,
        labels,
        config,
        units,
        title="Sample",
    )

    assert html_path.exists()
    assert html_path.suffix == ".html"
    content = html_path.read_text(encoding="utf-8")
    assert "Sample" in content
    assert "alpha" in content
    assert "unit_a" in content


def test_create_umap_plot_html_validates_lengths(tmp_path):
    projection = np.array([[0.0, 0.0]])
    labels = [0, 1]
    units = [ContentUnit(identifier="x", text="x-text"), ContentUnit(identifier="y", text="y-text")]
    config = VisualizationConfig(output_path=tmp_path / "plot.png", random_state=7)

    with pytest.raises(ValueError):
        create_umap_plot_html(projection, labels, config, units)



