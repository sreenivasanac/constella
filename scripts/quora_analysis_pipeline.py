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
from typing import List, Optional, Sequence

import psycopg

from constella.config.schemas import ClusteringConfig, VisualizationConfig
from constella.data.models import ContentUnit
from constella.embeddings.adapters import LiteLLMEmbeddingProvider
from constella.embeddings.base import EmbeddingProvider, InMemoryEmbeddingProvider
from constella.pipelines.workflow import cluster_texts
from constella.visualization.umap import create_umap_plot_html, project_embeddings, save_umap_plot


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

    texts: List[ContentUnit] = []
    with conn.cursor() as cur:
        if limit is None:
            cur.execute(query)
        else:
            cur.execute(query, (limit,))
        rows = cur.fetchall()

    for row in rows:
        identifier = f"answer_{row[0]}"
        text_parts = [
            f"Question: {row[3]}" if row[3] else "",
            f"Answered URL: {row[2]}" if row[2] else "",
            f"Answer: {row[4]}" if row[4] else "",
        ]
        text = "\n".join(part for part in text_parts if part)
        texts.append(ContentUnit(identifier=identifier, text=text))

    LOGGER.info("Fetched %d rows", len(texts))
    return texts


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def serialize_assignments(output_dir: Path, assignment, units: Sequence[ContentUnit]) -> Path:
    snapshot = asdict(assignment.config_snapshot)
    payload = {
        "assignments": {
            unit.identifier: cluster for unit, cluster in zip(units, assignment.assignments)
        },
        "metadata": {
            "silhouette_score": assignment.silhouette_score,
            "inertia": assignment.inertia,
            "centers": assignment.centers,
            "config_snapshot": snapshot,
        },
    }

    output_path = output_dir / "clusters.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info("Saved cluster assignments to %s", output_path)
    return output_path


def run_pipeline(args: argparse.Namespace) -> None:
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s")

    if not args.database_url:
        raise SystemExit("DATABASE_URL must be provided via --database-url or environment variable.")

    LOGGER.info("Connecting to database")
    with psycopg.connect(args.database_url) as conn:
        units = fetch_content_units(conn, limit=args.limit)

    if not units:
        raise SystemExit("No data retrieved from quora_answers table.")

    output_dir = ensure_output_dir(args.output_dir)

    LOGGER.info("Running Constella clustering pipeline")
    clustering_config = ClusteringConfig()
    png_path = output_dir / f"{args.umap_filename}.png"
    html_path = output_dir / f"{args.umap_filename}.html"


    LOGGER.info("Running clustering workflow")
    assignment, _, embeddings = cluster_texts(
        units,
        clustering_config=clustering_config,
        visualization_config=None
    )

    serialize_assignments(output_dir, assignment, units)

    visualization_config = VisualizationConfig(output_path=png_path)
    projection = project_embeddings(embeddings, visualization_config)

    save_umap_plot(
        projection,
        assignment.assignments,
        visualization_config,
        title="Quora Answer Clusters",
    )

    create_umap_plot_html(
        projection,
        assignment.assignments,
        visualization_config,
        texts_or_units=units,
        title="Quora Answer Clusters",
        output_path=html_path
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run_pipeline(args)


if __name__ == "__main__":
    main()
