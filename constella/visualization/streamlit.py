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
    texts_or_units: Sequence[str | ContentUnit],
) -> StreamlitUMAPArtifact:
    """Return metadata for a Streamlit-backed interactive UMAP visualization.

    This is a placeholder to orchestrate Streamlit usage without a direct dependency.
    """

    normalized_texts = []
    normalized_ids = []
    for idx, item in enumerate(texts_or_units):
        if isinstance(item, ContentUnit):
            normalized_ids.append(item.identifier)
            normalized_texts.append(item.text)
        else:
            normalized_ids.append(f"item_{idx}")
            normalized_texts.append(str(item))

    return StreamlitUMAPArtifact(
        html_path="/tmp/umap_plot.html",
        projection=projection.tolist(),
        labels=list(int(label) for label in labels),
        identifiers=normalized_ids,
        texts=normalized_texts,
    )
