# Constella

Constella is a token-efficient auto-grouping and optional auto-labelling library for large text collections (emails, support tickets, bookmarks, document snippets, image captions). It couples scalable embeddings with deterministic clustering so that only a handful of representative texts ever reach an LLM.

## Motivation

Processing tens of thousands of content units for auto-grouping and auto-labeling directly with an LLM quickly becomes cost prohibitive and yields poor quality results at long context windows. Constella avoids that trap by: (1) embedding every text with a provider-agnostic interface, (2) grouping items with deterministic K-Means, and (3) sampling only the most informative representatives for optional downstream labelling or analyst review.

## Key Capabilities

- **Token-aware embedding pipeline:** LiteLLM-backed providers for Fireworks (default) and OpenAI automatically batch requests, cap per-batch tokens, and run concurrently to maximize throughput while respecting provider limits.
- **Deterministic clustering:** A dataclass-driven `ClusteringConfig` feeds multi-metric model selection (silhouette, elbow, Davies–Bouldin) before running K-Means with a fixed seed, producing reproducible cluster assignments and logging inertia diagnostics.
- **Lightweight outputs:** Cluster IDs and clustering diagnostics live on the returned `ContentUnitCollection`, keeping the API surface compact while retaining provenance information.
- **Visualization tooling:** UMAP helpers generate publication-ready PNG plots and companion D3.js-powered interactive HTML scatter views with hover tooltips, making manual inspection of clusters fast even in headless environments.
- **Composable data models:** Lightweight dataclasses (`ContentUnit`, `ContentUnitCollection`) capture common ContentUnit attributes, and have embedding and cluster assignment values.

## Architecture at a Glance

- `constella.embeddings.adapters` — LiteLLM providers for Fireworks and OpenAI with concurrency, token-count heuristics, and configurable API bases.
- `constella.clustering.kmeans` — K-Means runner with candidate search, metric scoring, and fallbacks for numerically unstable cases.
- `constella.visualization.umap` — UMAP projection plus static and interactive plotting utilities.
- `constella.labeling.llm` — Placeholder entry points for future LLM-backed auto-labeling.
- `constella.pipelines.workflow.cluster_texts` — End-to-end orchestrator that normalizes inputs, generates embeddings, runs clustering, and optionally persists visualizations, returning the enriched collection.
- `constella.config.schemas` / `constella.data.models` — Frozen dataclasses that capture reproducible configuration and output artefacts.
- `scripts.quora_analysis_pipeline` — CLI runner for the Quora dataset that saves cluster assignments and reports generated artifacts.

## Workflow

1. Pass a `ContentUnitCollection` into `cluster_texts` with an optional `ClusteringConfig` (or overrides) and `VisualizationConfig`.
2. The Fireworks provider (or a configured alternative) embeds the texts using LiteLLM with CPU-bound batching and token budgeting.
3. Candidate cluster sizes are evaluated with silhouette, elbow, and Davies–Bouldin heuristics before selecting the final `k`.
4. A seeded K-Means run produces cluster assignments.
5. If visualization is enabled, embeddings are projected with UMAP and saved to disk as PNG and/or D3.js-based interactive HTML artefacts for downstream review.

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
from constella.data.models import ContentUnit, ContentUnitCollection
from constella.pipelines.workflow import cluster_texts

units = ContentUnitCollection([
    ContentUnit(identifier="doc_1", text="First document"),
    ContentUnit(identifier="doc_2", text="Second document"),
    ContentUnit(identifier="doc_3", text="Third document"),
])

collection = cluster_texts(
    units,
    clustering_config=ClusteringConfig(fallback_n_clusters=2, seed=8),
    visualization_config=VisualizationConfig(output_path=Path("/tmp/umap.png"), random_state=8),
)

if collection.metrics:
    print("Clusters:", collection.metrics.n_clusters)
    print("Silhouette score:", collection.metrics.silhouette_score)

if collection.artifacts:
    print("Visualization saved to %s %s", collection.artifacts.static_plot, collection.artifacts.html_plot)
```
`collection` contains cluster assignments on each `ContentUnit`, plus optional metrics and visualization artifact paths for downstream workflows.
