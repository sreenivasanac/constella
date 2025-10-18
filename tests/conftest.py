"""Test configuration stubs for optional heavy dependencies."""

from __future__ import annotations

import sys
import types


def _ensure_streamlit_stub() -> None:
    if "streamlit" in sys.modules:
        return

    streamlit_mod = types.ModuleType("streamlit")
    components_mod = types.ModuleType("streamlit.components")
    v1_mod = types.ModuleType("streamlit.components.v1")

    def markdown(*args, **kwargs):  # pragma: no cover - passthrough stub
        return None

    def html(*args, **kwargs):  # pragma: no cover - passthrough stub
        return None

    streamlit_mod.markdown = markdown  # type: ignore[attr-defined]
    components_mod.v1 = v1_mod  # type: ignore[attr-defined]
    v1_mod.html = html  # type: ignore[attr-defined]
    streamlit_mod.components = components_mod  # type: ignore[attr-defined]

    sys.modules.setdefault("streamlit", streamlit_mod)
    sys.modules.setdefault("streamlit.components", components_mod)
    sys.modules.setdefault("streamlit.components.v1", v1_mod)


_ensure_streamlit_stub()


def _ensure_umap_stub() -> None:
    if "umap" in sys.modules:
        return

    stub = types.ModuleType("umap")

    class UMAP:  # pragma: no cover - simple deterministic stub
        def __init__(self, n_neighbors: int, min_dist: float, random_state: int | None):
            self.n_neighbors = n_neighbors
            self.min_dist = min_dist
            self.random_state = random_state

        def fit_transform(self, array):
            import numpy as _np

            # Produce a deterministic projection by using the first two dims or PCA fallback
            arr = _np.asarray(array, dtype=float)
            if arr.shape[1] >= 2:
                return arr[:, :2]
            return _np.hstack([arr, _np.zeros((arr.shape[0], 2 - arr.shape[1]))])

    stub.UMAP = UMAP  # type: ignore[attr-defined]
    sys.modules.setdefault("umap", stub)


_ensure_umap_stub()
