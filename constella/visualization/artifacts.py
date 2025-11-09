"""Utility helpers for persisting visualization-related artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

from constella.data.results import LabelResult


LOGGER = logging.getLogger(__name__)


def write_labels_artifact(
    label_results: Dict[int, LabelResult],
    artifact_dir: Path,
) -> Path:
    """Persist cluster labels inside the provided artifact directory."""

    if not label_results:
        raise ValueError("No label results available to persist.")

    resolved_dir = Path(artifact_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)
    target_path = resolved_dir / "labels.json"

    payload = {
        str(cluster_id): {
            "label": result.label,
            "explanation": result.explanation,
            "confidence": result.confidence,
            "keywords": result.keywords,
        }
        for cluster_id, result in label_results.items()
    }

    target_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    LOGGER.info("Saved cluster labels to %s", target_path)
    return target_path


__all__ = ["write_labels_artifact"]
