"""Utilities to project embeddings and optionally persist UMAP plots."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps, colors

import streamlit as st
from streamlit.components.v1 import html as st_html

import umap  # type: ignore

from constella.config.schemas import VisualizationConfig
from constella.data.models import ContentUnit
from constella.visualization.html_templates import build_umap_hover_html


LOGGER = logging.getLogger(__name__)


def project_embeddings(
    vectors: Sequence[Sequence[float]],
    config: VisualizationConfig,
) -> np.ndarray:
    """Project embeddings into 2D using UMAP."""

    array = np.array(vectors, dtype=float)
    if array.ndim != 2 or array.shape[0] < 2:
        raise ValueError("Need at least two embeddings for UMAP projection.")

    reducer = umap.UMAP(
        n_neighbors=config.n_neighbors,
        min_dist=config.min_dist,
        random_state=config.random_state,
    )
    LOGGER.info("Generating UMAP projection for %s embeddings", array.shape[0])
    return reducer.fit_transform(array)


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


def create_umap_plot_html(
    projection: np.ndarray,
    labels: Sequence[int],
    config: VisualizationConfig,
    texts_or_units: Sequence[str | ContentUnit],
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
    render_streamlit: bool = True,
) -> Path:
    """Persist an interactive HTML scatter plot with hover tooltips.

    When ``render_streamlit`` is True and Streamlit is installed, the HTML snippet
    is embedded immediately via ``st.components.v1.html``.
    """

    if projection.ndim != 2 or projection.shape[1] != 2:
        raise ValueError("Projection must be a 2D array with two columns.")
    if projection.shape[0] == 0:
        raise ValueError("Need at least one embedding to render the HTML plot.")
    if len(labels) != projection.shape[0]:
        raise ValueError("Number of labels must match projection rows.")
    if len(texts_or_units) != projection.shape[0]:
        raise ValueError("Number of texts/content units must match projection rows.")

    target_path = Path(output_path) if output_path else Path(config.output_path)
    if target_path.suffix.lower() != ".html":
        target_path = target_path.with_suffix(".html")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_meta = []
    for idx, item in enumerate(texts_or_units):
        if isinstance(item, ContentUnit):
            normalized_meta.append({"identifier": item.identifier, "text": item.text})
        else:
            normalized_meta.append({"identifier": f"item_{idx}", "text": str(item)})

    unique_labels = sorted({int(label) for label in labels})
    cmap = colormaps.get_cmap("Spectral")
    if unique_labels:
        if len(unique_labels) == 1:
            samples = [0.5]
        else:
            samples = np.linspace(0.0, 1.0, len(unique_labels))
        color_lookup = {
            label_value: colors.to_hex(cmap(sample))
            for label_value, sample in zip(unique_labels, samples)
        }
    else:
        color_lookup = {}

    points = []
    for coords, label, meta in zip(projection, labels, normalized_meta):
        label_value = int(label)
        points.append(
            {
                "x": float(coords[0]),
                "y": float(coords[1]),
                "label": label_value,
                "color": color_lookup.get(label_value, "#1f77b4"),
                "identifier": meta["identifier"],
                "text": meta["text"],
            }
        )

    width = 800
    height = 600
    x_min = float(np.min(projection[:, 0]))
    x_max = float(np.max(projection[:, 0]))
    y_min = float(np.min(projection[:, 1]))
    y_max = float(np.max(projection[:, 1]))

    html_content = build_umap_hover_html(
        data_json=json.dumps(points, ensure_ascii=False),
        title_json=json.dumps(title or "UMAP Projection", ensure_ascii=False),
        width=width,
        height=height,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )

    LOGGER.info("Saving interactive UMAP plot to %s", target_path)
    target_path.write_text(html_content, encoding="utf-8")

    if render_streamlit:
        try:
            if title:
                st.markdown(f"### {title}")
            st_html(html_content, height=height + 120, scrolling=True)
        except Exception as exc:  # pragma: no cover - depends on runtime
            LOGGER.debug("Streamlit rendering skipped: %s", exc, exc_info=True)

    return target_path
