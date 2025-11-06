"""Streamlit helper for rendering UMAP projections with interactive tooltips."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from constella.data.models import ContentUnit


@dataclass(frozen=True)
class StreamlitUMAPArtifact:
    """Container describing static assets needed for Streamlit rendering."""

    html_path: str
    projection: Sequence[Sequence[float]]
    labels: Sequence[int]
    identifiers: Sequence[str]
    texts: Sequence[str]


def create_umap_plot_html(
    projection: np.ndarray,
    labels: Sequence[int],
    units: Sequence[ContentUnit],
) -> StreamlitUMAPArtifact:
    """Return metadata for a Streamlit-backed interactive UMAP visualization.

    This is a placeholder to orchestrate Streamlit usage without a direct dependency.
    """

    if projection.shape[0] != len(units):
        raise ValueError("Projection rows must match number of content units.")
    if projection.shape[0] != len(labels):
        raise ValueError("Labels must align with projection rows.")

    normalized_ids = [unit.identifier for unit in units]
    normalized_texts = [unit.get_content() for unit in units]

    return StreamlitUMAPArtifact(
        html_path="/tmp/umap_plot.html",
        projection=projection.tolist(),
        labels=list(int(label) for label in labels),
        identifiers=normalized_ids,
        texts=normalized_texts,
    )
