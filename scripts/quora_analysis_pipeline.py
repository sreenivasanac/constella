"""Quora analysis pipeline via Constella.

This script connects to the configured PostgreSQL database, extracts
question/answer content, runs it through the Constella embedding and
clustering workflow, and persists visualization artifacts.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import psycopg

from constella.config.schemas import ClusteringConfig, VisualizationConfig
from constella.data.models import ContentUnit, ContentUnitCollection
from constella.pipelines.workflow import run_pipeline as execute_workflow


LOGGER = logging.getLogger("quora_analysis_pipeline")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Constella pipeline on Quora answers")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"), help="PostgreSQL connection URL")
    parser.add_argument("--limit", type=int, default=None, help="Maximum rows to process (for dry runs)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/quora_analysis"),
        help="Directory for generated artifacts",
    )
    parser.add_argument("--umap-filename", default="umap", help="Base name for UMAP outputs")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args(argv)


def fetch_content_units(conn: psycopg.Connection, limit: Optional[int] = None) -> List[ContentUnit]:
    fields = [
        "id",
        "question_url",
        "answered_question_url",
        "question_text",
        "answer_content",
    ]
    query = f"SELECT {', '.join(fields)} FROM public.quora_answers ORDER BY id"
    if limit is not None:
        query += " LIMIT %s"

    LOGGER.info("Executing query: %s", query)

    units: List[ContentUnit] = []
    with conn.cursor() as cur:
        if limit is None:
            cur.execute(query)
        else:
            cur.execute(query, (limit,))
        rows = cur.fetchall()

    for row in rows:
        identifier = f"answer_{row[0]}"
        question_url = row[1]
        answered_question_url = row[2]
        question_text = row[3]
        answer_content = row[4] or ""

        units.append(
            ContentUnit(
                identifier=identifier,
                text=answer_content,
                title=question_text,
                name=question_url,
                path=answered_question_url,
                size=f"{len(answer_content)} characters" if answer_content else None,
                metadata1={
                    "question_text": question_text,
                    "question_url": question_url,
                    "answered_question_url": answered_question_url,
                },
                metadata2={},
            )
        )

    LOGGER.info("Fetched %d rows", len(units))
    return units


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def serialize_assignments(
    output_dir: Path,
    collection: ContentUnitCollection,
    config: ClusteringConfig,
) -> Path:
    cluster_assignments = {
        unit.identifier: int(unit.cluster_id)
        for unit in collection
        if unit.cluster_id is not None
    }

    metrics = collection.metrics
    snapshot = metrics.config_snapshot if metrics and metrics.config_snapshot is not None else config

    payload = {
        "assignments": cluster_assignments,
        "n_clusters": metrics.n_clusters if metrics else None,
        "silhouette_score": metrics.silhouette_score if metrics else None,
        "inertia": metrics.inertia if metrics else None,
        "centers": metrics.centers if metrics else None,
        "config": asdict(snapshot),
    }

    output_path = output_dir / "clusters.json"
    output_path.write_text(_json_dumps(payload), encoding="utf-8")
    LOGGER.info("Saved cluster assignments to %s", output_path)
    return output_path


def _json_dumps(payload: Dict[str, object]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _configure_logging(level: str) -> None:
    logging.basicConfig(level=level.upper(), format="%(asctime)s %(levelname)s %(message)s")


def _build_configs(args: argparse.Namespace, output_dir: Path) -> Tuple[ClusteringConfig, VisualizationConfig]:
    clustering_config = ClusteringConfig()
    png_path = output_dir / f"{args.umap_filename}.png"
    viz_config = VisualizationConfig(output_path=png_path)
    return clustering_config, viz_config


def run_cli_pipeline(args: argparse.Namespace) -> None:
    _configure_logging(args.log_level)

    if not args.database_url:
        raise SystemExit("DATABASE_URL must be provided via --database-url or environment variable.")

    LOGGER.info("Connecting to database")
    with psycopg.connect(args.database_url) as conn:
        units = fetch_content_units(conn, limit=args.limit)

    if not units:
        raise SystemExit("No data retrieved from quora_answers table.")

    output_dir = ensure_output_dir(args.output_dir)

    LOGGER.info("Running Constella clustering pipeline")
    clustering_config, viz_config = _build_configs(args, output_dir)

    LOGGER.info("Running clustering workflow")
    collection = ContentUnitCollection(units)

    collection = execute_workflow(
        collection,
        steps=("embed", "cluster", "visualize"),
        configs={
            "cluster": clustering_config,
            "visualize": viz_config,
        },
    )

    serialize_assignments(output_dir, collection, clustering_config)

    artifacts = collection.artifacts
    if artifacts:
        for artifact in (artifacts.static_plot, artifacts.html_plot):
            if artifact is not None and Path(artifact).exists():
                LOGGER.info("Generated artifact: %s", artifact)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run_cli_pipeline(args)


if __name__ == "__main__":
    main()
