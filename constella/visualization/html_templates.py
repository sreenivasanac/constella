"""HTML templates used by visualization helpers."""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from string import Template


_TEMPLATE_NAME = "umap_hover.html"


@lru_cache(maxsize=1)
def _load_umap_hover_template() -> Template:
    template_path = resources.files(__package__) / _TEMPLATE_NAME
    with template_path.open("r", encoding="utf-8") as template_file:
        return Template(template_file.read())


def build_umap_hover_html(
    *,
    title_json: str,
    width: int,
    height: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    preview_json: str,
) -> str:
    """Return an HTML document embedding an interactive hoverable UMAP scatter."""

    context = {
        "title_json": title_json,
        "width": width,
        "height": height,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "preview_json": preview_json,
    }

    return _load_umap_hover_template().safe_substitute(context)
