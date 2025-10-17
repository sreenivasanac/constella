# Constella - token-efficient auto-grouping and auto-labelling library for large text-based collection of content

## Problem
Large text-based collection of content (e.g., 10,000 emails, bookmarks, messages, documents, images via image-descriptions) are expensive to intelligently and automatically group, organize and label directly with LLMs because of token costs and degraded accuracy at long context lengths. This project Constella groups semantically similar content-units first -> in a scalable, model‑agnostic way —> second optionally applies an LLM call to a much smaller per‑group representative subset for automated labeling.

## Terminology
- "Content-unit" refers to the individual emails, messages, bookmark content, or text chunks.

## Approach (high level)
1. Compute semantic embeddings for each content-unit (message/email/bookmark/document chunk) of the collection using a parameterized embedding model supplied by the caller.
2. Cluster the embeddings with K‑Means to group content-units by vector proximity (semantic similarity).
3. Select cluster representatives (units closest to the centroid) to summarize the group and, if desired, send only these few representatives to an LLM to generate a human‑readable label.
4. Return cluster assignments, cluster centers, sizes, and representative indices for downstream use.

Cluster counts can be derived dynamically from the distribution of formed clusters or set manually with the aid of visualization tools.

## Example use cases
- Auto‑organizing and auto-labeling large bookmark collections into topical folders (e.g., Productivity, News).
- Email and message triage by theme before routing or prioritization.
- Auto-organizing and auto-labeling document repositories into semantic folders (e.g., Finance, Health).
- Customer feedback analysis that groups similar complaints, reviews, or support tickets.
- Grouping support tickets or product reviews to surface common issues and feature requests.
- Topic discovery in knowledge bases, news articles, or research abstracts.

## Advantages
- Token efficiency: LLM usage is limited to small, representative samples per cluster rather than the full corpus.
- Scalability: Linear complexity for embedding generation (through batch processing) and classical clustering supports scaling to tens of thousands of content-units.
- Deterministic, reproducible grouping via K‑Means (given a fixed seed), reducing subjective variability.
- Model‑agnostic: Works with different embedding backends and can integrate with local or hosted models.
- Faster labeling: LLM labeling applies to clusters instead of individual content-units, speeding categorization.

The goal is a clear, practical pipeline that reduces LLM token usage while preserving high-quality grouping for labeling large collections.

## Minimal workflow
- Input: list of texts plus configuration.
- Embed in batches for throughput and memory efficiency, then run K‑Means to obtain cluster-groups and cluster-centers.
- When clusters contain many content-units, compute optional representatives per cluster by nearest‑to‑centroid.
- Call an external LLM on each cluster’s representatives to propose a label (Optional).
- Output artifacts: cluster assignments, cluster centers, cluster sizes, representative indices, any LLM‑generated labels (if used).

## Deliverables (API sketch, non‑binding)
- fit_embeddings(texts, config) → clusters, centers, sizes
- get_representatives(labels, centers, k) → indices per cluster
- label_clusters(representatives, model_name, model_api_key) → human‑readable names # uses LiteLLM internally
- cluster visualization tools

## Future improvements (as needed)
- Leverage batch embedding and streaming-friendly pipelines to manage throughput and memory usage.
- Support for incremental clustering and online learning scenarios
- GPU acceleration for embedding generation and clustering operations
