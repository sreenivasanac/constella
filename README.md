# Constella v0.1

Constella provides a deterministic pipeline to embed and cluster text corpora.

## Features

- Dataclass-based configuration for clustering and optional visualization.
- LiteLLM adapter targeting OpenAI `text-embedding-3-small` embeddings.
- Deterministic K-Means clustering with optional silhouette-based cluster selection.
- UMAP projection utility (with PCA fallback) that saves scatter plots to disk without display requirements.
- Workflow entry point `cluster_texts` coordinating embeddings, clustering, and visualization.

## Getting Started

Install the package in editable mode:

```bash
pip install -e .
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

assignment, artifacts = cluster_texts(texts, config, viz)
```
