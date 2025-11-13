"""Utilities to project embeddings and optionally persist UMAP plots."""

from __future__ import annotations

import inspect
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
from constella.data.models import ContentUnit, ContentUnitCollection
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

    reducer_kwargs = {
        "n_neighbors": config.n_neighbors,
        "min_dist": config.min_dist,
        "random_state": config.random_state,
    }
    if config.random_state is not None:
        try:
            umap_signature = inspect.signature(umap.UMAP)
        except (TypeError, ValueError):
            umap_signature = None
        if umap_signature and "n_jobs" in umap_signature.parameters:
            reducer_kwargs["n_jobs"] = 1

    reducer = umap.UMAP(**reducer_kwargs)
    LOGGER.info("Generating UMAP projection for %s embeddings", array.shape[0])
    return reducer.fit_transform(array)


def save_umap_plot(
    projection: np.ndarray,
    collection: ContentUnitCollection,
    config: VisualizationConfig,
    *,
    title: Optional[str] = None,
    artifact_dir: Path,
) -> Path:
    """Persist a scatter plot of the UMAP projection."""

    if projection.shape[1] != 2:
        raise ValueError("Projection must have exactly two dimensions for plotting.")

    units = collection.units()
    if projection.shape[0] != len(units):
        raise ValueError("Number of projection rows must match content units.")

    resolved_labels = collection.iter_visual_labels()
    if len(resolved_labels) != projection.shape[0]:
        raise ValueError("Number of labels must match projection rows.")
    ordered_labels = list(dict.fromkeys(resolved_labels))

    if len(ordered_labels) <= 10:
        cmap_name = "tab10"
    elif len(ordered_labels) <= 20:
        cmap_name = "tab20"
    else:
        cmap_name = "gist_ncar"

    base_cmap = colormaps.get_cmap(cmap_name)
    if ordered_labels:
        samples = np.linspace(0.0, 1.0, len(ordered_labels), endpoint=False)
        color_lookup = {
            label_value: colors.to_hex(base_cmap(sample))
            for label_value, sample in zip(ordered_labels, samples)
        }
    else:
        color_lookup = {}
    point_colors = [color_lookup.get(label, "#1f77b4") for label in resolved_labels]

    resolved_dir = Path(artifact_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)
    output_path = resolved_dir / "umap.png"

    LOGGER.info("Saving UMAP plot to %s", output_path)
    plt.figure(figsize=(8, 6))
    plt.scatter(projection[:, 0], projection[:, 1], c=point_colors, s=20)
    if ordered_labels:
        legend_handles = []
        for label_value in ordered_labels:
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markerfacecolor=color_lookup[label_value],
                    markeredgecolor=color_lookup[label_value],
                    label=str(label_value),
                )
            )
        plt.legend(handles=legend_handles, title="Cluster", fontsize="small", loc="best")
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


def _truncate_text_lines(text: str, max_lines: int) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[:max_lines])


def create_umap_plot_html(
    projection: np.ndarray,
    collection: ContentUnitCollection,
    config: VisualizationConfig,
    *,
    title: Optional[str] = None,
    artifact_dir: Path,
) -> Path:
    """Persist an interactive HTML scatter plot with hover tooltips.
    """

    if projection.ndim != 2 or projection.shape[1] != 2:
        raise ValueError("Projection must be a 2D array with two columns.")
    if projection.shape[0] == 0:
        raise ValueError("Need at least one embedding to render the HTML plot.")
    units = collection.units()
    if len(units) != projection.shape[0]:
        raise ValueError("Number of content units must match projection rows.")

    resolved_labels = collection.iter_visual_labels()
    if len(resolved_labels) != projection.shape[0]:
        raise ValueError("Number of labels must match projection rows.")
    ordered_labels = list(dict.fromkeys(resolved_labels))

    resolved_dir = Path(artifact_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)
    target_path = resolved_dir / "umap.html"

    normalized_meta = [
        {"identifier": unit.identifier, "text": unit.get_content()}
        for unit in units
    ]

    cmap = colormaps.get_cmap("Spectral")
    if ordered_labels:
        if len(ordered_labels) == 1:
            samples = [0.5]
        else:
            samples = np.linspace(0.0, 1.0, len(ordered_labels))
        color_lookup = {
            label_value: colors.to_hex(cmap(sample))
            for label_value, sample in zip(ordered_labels, samples)
        }
    else:
        color_lookup = {}

    points = []
    for coords, label, meta, unit in zip(projection, resolved_labels, normalized_meta, units):
        cluster_id = unit.cluster_id if unit.cluster_id is not None else None
        points.append(
            {
                "x": float(coords[0]),
                "y": float(coords[1]),
                "label": str(label),
                "cluster_id": cluster_id,
                "color": color_lookup.get(label, "#1f77b4"),
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

    for point in points:
        point["text"] = _truncate_text_lines(point["text"], 8)

    data_json = json.dumps(points, ensure_ascii=False, indent=2)
    title_json = json.dumps(title or "UMAP Projection", ensure_ascii=False)

    data_script_path = resolved_dir / "umap_data.js"

    preview_points = points[: min(len(points), 5)]
    preview_json = json.dumps(preview_points, ensure_ascii=False).replace("</", "<\\/")

    html_content = build_umap_hover_html(
        title_json=title_json,
        width=width,
        height=height,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        preview_json=preview_json,
    )

    data_script = (
        "window.UMAP_DATA = "
        f"{data_json};"
        "document.dispatchEvent(new Event(\"umap-data-ready\"));\n"
    )

    LOGGER.info("Saving interactive UMAP plot to %s", target_path)
    target_path.write_text(html_content, encoding="utf-8")
    data_script_path.write_text(data_script, encoding="utf-8")

    return target_path
