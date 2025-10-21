# Constella v0.1

Constella is a library for auto-grouping for labelling content units like bookmarks, emails, files, images. It uses K-means clustering.

Constella (short for "Constellation") captures the goal of deriving semantic meaning by visualizing fragment points that lie in spatial proximity.

## Problem

Large collections of content — emails, bookmarks, support tickets, document chunks—are costly to auto-organize with direct LLM calls due to token limits and degraded accuracy at long contexts. Constella first groups semantically similar content units through vector embeddings and deterministic clustering, then enables lightweight LLM labeling only on representative samples.

## Approach

1. Generate semantic embeddings for each content unit via a chosen embedding provider.
2. Cluster embeddings with deterministic K-Means, automatically evaluating candidate cluster counts with silhouette, elbow, and Davies-Bouldin scoring.
3. Surface cluster representatives near centroids to summarize groups and support optional downstream LLM labeling or analyst review.
4. Return assignments, centers, metrics, and visualization-ready projections for integration into downstream workflows.

## Example Use Cases

- Auto-organizing large bookmark libraries into topical folders.
- Triage and routing of email or message streams by theme before prioritization.
- Auto-labeling document repositories or knowledge bases by semantic topic.
- Customer feedback analysis that groups similar reviews or support tickets to expose recurring issues.
- Topic discovery across news articles, research abstracts, or product reviews.

## Advantages

- **Token efficiency:** Limits expensive LLM calls to small representative subsets instead of entire corpora.
- **Scalability:** Token-aware batching and parallel embedding requests for scaling to tens of thousands of content units.
- **Determinism:** Fixed-seed K-Means ensures reproducible grouping outcomes.
- **Model agnostic:** Works with any compatible embedding backend, including local or hosted providers.
- **Faster labeling:** Enables rapid category assignment by labeling clusters rather than individual items.

## Features

- Dataclass-based configuration for clustering and optional visualization.
- LiteLLM adapter targeting OpenAI `text-embedding-3-small` embeddings (requires `OPENAI_API_KEY`) with concurrency-aware, token-limited batching.
- Deterministic K-Means clustering with automatic cluster count selection (silhouette, elbow, and Davies-Bouldin metrics).
- UMAP projection helpers that generate static plots and hoverable HTML artifacts without display requirements.
- Workflow entry point `cluster_texts` coordinating embeddings, clustering, and visualization, returning both assignments and raw embeddings for downstream analysis.

## Getting Started

Install the package in editable mode:

To enable embeddings through OpenAI embeddings endpoint using LiteLLM, please provide OpenAI API Key.
```bash
pip install -e .
export OPENAI_API_KEY="sk-your-key"
```

Run the test suite:

```bash
.venv/bin/python -m pip install -e .[test]
.venv/bin/python -m pytest
```

## Usage

```python
from constella.config.schemas import ClusteringConfig, VisualizationConfig
from constella.pipelines.workflow import cluster_texts

texts = ["First document", "Second document", "Third document"]
config = ClusteringConfig(seed=8, fallback_n_clusters=2)
viz = VisualizationConfig(output_path="/tmp/umap.png", random_state=8)

assignment, artifacts, embeddings = cluster_texts(texts, config, viz)
```

The returned `assignment` includes cluster centers, inertia, and an optional silhouette score snapshot for auditing, while `artifacts` contains paths to generated visualizations (static PNGs or interactive HTML). The raw `embeddings` array can be reused for downstream analytics or persistence.
