# Constella

Constella is a token-efficient auto-grouping and optional auto-labelling library for large text collections (emails, support tickets, bookmarks, document snippets, image captions). It couples scalable embeddings with deterministic clustering so that only a handful of representative texts ever reach an LLM.

## Motivation

Processing tens of thousands of content units for auto-grouping and auto-labeling directly with an LLM quickly becomes cost prohibitive and yields poor quality results at long context windows. Constella avoids that trap by: (1) embedding every text with a provider-agnostic interface, (2) grouping items with deterministic K-Means, and (3) sampling only the most informative representatives for optional downstream labelling or analyst review.

## Key Capabilities

- **Token-aware embedding pipeline:** LiteLLM-backed providers for any LLM Embedding API (Fireworks, OpenAI etc). Automatically batch embedding requests, cap per-batch tokens, and run concurrently to maximize throughput while respecting provider limits.
- **Deterministic clustering:** A dataclass-driven `ClusteringConfig` feeds multi-metric model selection (silhouette, elbow, Davies–Bouldin) before running K-Means with a fixed seed, producing reproducible cluster assignments and logging inertia diagnostics.
- **Lightweight outputs:** Assigned Cluster IDs and clustering diagnostics live on the returned `ContentUnitCollection`, keeping the API surface compact while retaining provenance information.
- **LLM-assisted auto-labelling:** Deterministic representative sampling plus LiteLLM OpenAI prompts yield structured labels (name, explanation, confidence, keywords) for each cluster.
- **Visualization tooling:** Visualise the clusters in image format and D3.js-powered interactive HTML scatter views with hover tooltips, making manual inspection of clusters.
- **Composable data models:** Lightweight dataclasses (`ContentUnit`, `ContentUnitCollection`) capture common ContentUnit attributes, and have embedding and cluster assignment values.

## Architecture at a Glance

- `constella.embeddings.adapters` — LiteLLM providers for Fireworks and OpenAI with concurrency, token-count heuristics, and configurable API bases.
- `constella.clustering.kmeans` — K-Means runner with candidate search, metric scoring.
- `constella.labeling.selection` — Representative sampling yielding centroid-aligned core samples plus diversity picks for downstream labelling.
- `constella.visualization.umap` — UMAP projection plus static and interactive plotting utilities.
- `constella.labeling.llm` — Prompt orchestration, LiteLLM OpenAI calls, and JSON parsing for cluster auto-labelling.
- `constella.pipelines.workflow.cluster_texts` — End-to-end orchestrator that normalizes inputs, generates embeddings, runs clustering, and optionally persists visualizations, returning the enriched collection.
- `constella.config.schemas` / `constella.data.models` — dataclasses that capture reproducible configuration and output artefacts.
- `scripts.quora_analysis_pipeline` — CLI runner for the Quora dataset that saves cluster assignments and reports generated artifacts.

## Workflow

1. Pass a `ContentUnitCollection` into `cluster_texts` with an optional `ClusteringConfig` (or overrides) and `VisualizationConfig`.
2. The Fireworks provider (or a configured alternative) embeds the texts using LiteLLM with CPU-bound batching and token budgeting.
3. Candidate cluster sizes are evaluated with silhouette, elbow, and Davies–Bouldin heuristics before selecting the final `k`.
4. A seeded K-Means run produces cluster assignments.
5. Call `get_labels` (or pass a `LabelingConfig` to `cluster_texts`) to obtain structured auto-labels using LiteLLM OpenAI. It uses representative sampling to extract centroid-aligned plus diverse examples per cluster.
6. If visualization is enabled, embeddings are projected with UMAP and saved to disk as PNG and/or D3.js-based interactive HTML artifacts for downstream review.

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

# Fireworks AI API key needed for embedding step
export FIREWORKS_AI_API_KEY="your-fireworks-key"


# Auto labelling step uses LiteLLM to connect with any model.
# The example model for auto labelling uses OpenAI model.
# OpenAI API key is required for example model.
# If any other model you want to use for auto-labeling step, you can change the labeling configuration, and provide related Model API Key. 
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
from constella.embeddings.adapters import LiteLLMEmbeddingFireworksProvider
from constella.pipelines.workflow import cluster_texts

units = ContentUnitCollection([
    ContentUnit(identifier="doc_1", text="First document", metadata1={"source": "faq"}),
    ContentUnit(identifier="doc_2", text="Second document", metadata1={"source": "faq"}),
    ContentUnit(identifier="doc_3", text="Third document", metadata1={"source": "faq"}),
])

# Calling clustering workflow with default settings
collection = cluster_texts(units)

# Calling clustering workflow with explicit parameters
collection = cluster_texts(
    units,
    clustering_config=ClusteringConfig(
        fallback_n_clusters=2,
    ),
    visualization_config=VisualizationConfig(
        output_path=Path("/tmp/constella/umap.png"),
    ),
    embedding_provider=LiteLLMEmbeddingFireworksProvider(),
)

if collection.metrics:
    metrics = collection.metrics
    print("Number of Clusters:", metrics.n_clusters)
    print("Silhouette score:", metrics.silhouette_score)

if collection.artifacts:
    artifacts = collection.artifacts
    print("Static visualization:", artifacts.static_plot)
    print("Interactive visualization:", artifacts.html_plot)
```
`collection` contains cluster assignments on each `ContentUnit`, along with an optional `ClusteringMetrics` snapshot and any generated `VisualizationArtifacts` paths.

To auto-label the resulting clusters, sample representatives and invoke the labelling helper:

```python
from constella.config.schemas import LabelingConfig, RepresentativeSelectionConfig
from constella.labeling.selection import select_representatives
from constella.labeling.llm import get_labels

# sample up to 20 representatives per cluster before labelling
select_config = RepresentativeSelectionConfig(n_representatives=20, core_ratio=0.6)
select_representatives(collection, select_config)

label_config = LabelingConfig(
    model="gpt-4o-mini",
    temperature=0.1,
)

labels = get_labels(collection, label_config)
for cluster_id, result in labels.items():
    print(cluster_id, result.label, result.confidence)
```

`get_labels` defaults to the OpenAI LiteLLM provider; ensure `OPENAI_API_KEY` is set before calling it. Each `LabelResult` supplies the cluster label, explanation, confidence score, and associated keywords.
