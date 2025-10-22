# Constella

Constella is a token-efficient auto-grouping and optional auto-labelling library for large text collections (emails, support tickets, bookmarks, document snippets, image captions). It couples scalable embeddings with deterministic clustering so that only a handful of representative texts ever reach an LLM.

## Motivation

Processing tens of thousands of content units for auto-grouping and auto-labeling directly with an LLM quickly becomes cost prohibitive and yields poor quality results at long context windows. Constella avoids that trap by: (1) embedding every text with a provider-agnostic interface, (2) grouping items with deterministic K-Means, and (3) sampling only the most informative representatives for optional downstream labelling or analyst review.

## Key Capabilities

- **Token-aware embedding pipeline:** LiteLLM-backed providers for Fireworks (default) and OpenAI automatically batch requests, cap per-batch tokens, and run concurrently to maximize throughput while respecting provider limits.
- **Deterministic clustering:** A dataclass-driven `ClusteringConfig` feeds multi-metric model selection (silhouette, elbow, Davies–Bouldin) before running K-Means with a fixed seed, producing reproducible clusters and inertia diagnostics.
- **Rich diagnostics:** Every `ClusterAssignment` snapshot contains selected `n_clusters`, centroids, inertia, and optional silhouette scores for auditability.
- **Visualization tooling:** UMAP helpers generate publication-ready PNG plots and companion interactive HTML scatter views with hover tooltips, making manual inspection of clusters fast even in headless environments.
- **Composable data models:** Lightweight `ContentUnit`, `EmbeddingVector`, and `ClusterAssignment` dataclasses provide a typed interface that works equally well with raw strings or pre-wrapped metadata objects.

## Architecture at a Glance

- `constella.embeddings.adapters` — LiteLLM providers for Fireworks and OpenAI with concurrency, token-count heuristics, and configurable API bases.
- `constella.clustering.kmeans` — K-Means runner with candidate search, metric scoring, and fallbacks for numerically unstable cases.
- `constella.visualization.umap` — UMAP projection plus static and interactive plotting utilities.
- `constella.pipelines.workflow.cluster_texts` — End-to-end orchestrator that normalizes inputs, generates embeddings, runs clustering, and optionally persists visualizations.
- `constella.config.schemas` / `constella.data.models` — Frozen dataclasses that capture reproducible configuration and output artefacts.

## Workflow

1. Pass a list of raw strings or `ContentUnit` objects into `cluster_texts` with an optional `ClusteringConfig` and `VisualizationConfig`.
2. The Fireworks provider (or a configured alternative) embeds the texts using LiteLLM with CPU-bound batching and token budgeting.
3. Candidate cluster sizes are evaluated with silhouette, elbow, and Davies–Bouldin heuristics before selecting the final `k`.
4. A seeded K-Means run produces assignments, cluster centres, and inertia diagnostics which are returned as a `ClusterAssignment` snapshot.
5. If visualization is enabled, embeddings are projected with UMAP and saved to disk as PNG and/or interactive HTML artefacts for downstream review.

## Advantages

- **Token efficiency:** Only cluster representatives need to be sent to an LLM, compressing label costs by orders of magnitude for large corpora.
- **Scalability:** Batch-friendly embedding requests and optional concurrency keep throughput high even for 10k+ documents.
- **Reproducibility:** Frozen configs and seeded algorithms make it easy to compare runs and debug drift.
- **Model agnosticism:** Any LiteLLM-compatible embedding backend can be registered without altering the pipeline.
- **Analyst-friendly outputs:** Static and interactive plots, plus structured assignment objects, integrate cleanly with dashboards or notebooks.

## Example Use Cases

- Organising bookmark collections into topic folders before syncing back into productivity tools.
- Email or chat triage that groups messages by intent ahead of routing or prioritisation.
- Labelling document repositories, research papers, or knowledge bases by discovered themes.
- Surfacing recurring customer feedback issues by clustering support tickets or product reviews.
- Topic discovery inside news archives or competitive intelligence datasets.

## Getting Started

```bash
pip install -e .

# Choose one of the supported providers
export FIREWORKS_AI_API_KEY="your-fireworks-key"
# or
export OPENAI_API_KEY="sk-your-openai-key"
```

To run the tests locally:

```bash
.venv/bin/python -m pip install -e .[test]
.venv/bin/python -m pytest
```

## Usage

```python
from pathlib import Path

from constella.config.schemas import ClusteringConfig, VisualizationConfig
from constella.pipelines.workflow import cluster_texts

texts = ["First document", "Second document", "Third document"]
config = ClusteringConfig(seed=8, fallback_n_clusters=2)
viz = VisualizationConfig(output_path=Path("/tmp/umap.png"), random_state=8)

assignment, artifacts, embeddings = cluster_texts(texts, config, viz)
```

`assignment` captures cluster memberships, centroids, inertia, and any computed silhouette score, while `artifacts` contains the generated plot locations (PNG/HTML). The raw `embeddings` list can be cached or fed into downstream analytics, persistence layers, or labelling workflows.
