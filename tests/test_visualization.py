"""Tests for visualization utilities."""

from __future__ import annotations

import numpy as np
import pytest

from constella.config.schemas import VisualizationConfig
from constella.data.models import ContentUnit, ContentUnitCollection
from constella.data.results import LabelResult
from constella.visualization.umap import create_umap_plot_html, project_embeddings, save_umap_plot


def test_project_embeddings_shape(tmp_path):
    vectors = np.array([
        [0.0, 0.0, 0.1],
        [1.0, 0.9, 1.1],
        [0.1, 0.2, 0.0],
    ])
    config = VisualizationConfig(output_path=tmp_path, random_state=7)
    projection = project_embeddings(vectors, config)

    assert projection.shape == (3, 2)


def test_save_umap_plot_creates_file(tmp_path):
    projection = np.array([[0.0, 0.0], [1.0, 1.0]])
    collection = ContentUnitCollection(
        [
            ContentUnit(identifier="a", text="alpha", cluster_id=0),
            ContentUnit(identifier="b", text="beta", cluster_id=1),
        ]
    )
    config = VisualizationConfig(output_path=tmp_path, random_state=7)
    artifact_dir = tmp_path / "artifacts_run"
    path = save_umap_plot(projection, collection, config, artifact_dir=artifact_dir)

    assert path == artifact_dir / "umap.png"
    assert path.exists()


def test_create_umap_plot_html_generates_hoverable_artifact(tmp_path):
    projection = np.array([[0.0, 0.0], [1.0, 1.0]])
    collection = ContentUnitCollection(
        [
            ContentUnit(
                identifier="unit_a",
                text="alpha",
                title="Title A",
                name="Name A",
                path="/path/a",
                size="120 chars",
                cluster_id=0,
            ),
            ContentUnit(identifier="unit_b", text="beta", title="Title B", cluster_id=1),
        ]
    )
    collection.label_results = {
        0: LabelResult(cluster_id=0, label="Alpha Topics", explanation="", confidence=0.9),
        1: LabelResult(cluster_id=1, label="Beta Topics", explanation="", confidence=0.8),
    }
    config = VisualizationConfig(output_path=tmp_path, random_state=7)
    artifact_dir = tmp_path / "artifacts_run"

    html_path = create_umap_plot_html(
        projection,
        collection,
        config,
        title="Sample",
        artifact_dir=artifact_dir,
    )

    assert html_path.exists()
    assert html_path.suffix == ".html"
    assert html_path == artifact_dir / "umap.html"
    data_script = artifact_dir / "umap_data.js"
    assert data_script.exists()
    content = html_path.read_text(encoding="utf-8")
    assert "Sample" in content
    assert "Alpha Topics" in content
    assert "alpha" in content
    assert "unit_a" in content
    assert "Title A" in content
    assert "Name A" in content
    assert "/path/a" in content
    assert "120 chars" in content


def test_create_umap_plot_html_validates_lengths(tmp_path):
    projection = np.array([[0.0, 0.0]])
    collection = ContentUnitCollection(
        [
            ContentUnit(identifier="x", text="x-text", cluster_id=0),
            ContentUnit(identifier="y", text="y-text", cluster_id=1),
        ]
    )
    config = VisualizationConfig(output_path=tmp_path, random_state=7)

    with pytest.raises(ValueError):
        create_umap_plot_html(projection, collection, config, artifact_dir=tmp_path)



