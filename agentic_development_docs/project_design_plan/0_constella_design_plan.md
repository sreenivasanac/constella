# Constella Design Plan (v0.1–v0.4)

## Purpose & Alignment
- **Objective**: Deliver a Python library that clusters large text collections efficiently and applies token-aware auto-labeling.
- **Source goals**: Mirrors the outcomes in `initial_plan.md` → embeddings → K-Means clustering → representative sampling → optional LLM-driven labels with minimal token usage.
- **Packaging mandate**: Ship as an installable library (`constella` package) with clear module surfaces; all runnable scripts should live under `constella/` or companion `examples/`.

## Core Architecture Overview

- **Top-level layout**
  ```text
  constella/
    __init__.py
    config/
      schemas.py          # minimal configs for v0.1, expanded validation in later versions
      prompts.yaml        # templated prompts for labeling/refinement (introduced v0.3)
      settings.yaml       # shared logging defaults (added v0.2)
    data/
      models.py           # core data containers (ContentUnit, ClusterAssignment, etc.); batching/label models arrive v0.2+
    embeddings/
      base.py             # embedding provider interface (simple embed_text signature in v0.1)
      adapters.py         # LiteLLM-backed OpenAI text-embedding-3-small connector (v0.1); registry expansion v0.3
    clustering/
      kmeans.py           # primary clustering implementation
      selection.py        # representative sampling strategies
    labeling/
      llm.py              # LLM labeling orchestrator
    pipelines/
      workflow.py         # end-to-end orchestration utilities
    evaluation/
      metrics.py          # internal diagnostics (silhouette, inertia, etc.) introduced v0.4
    visualization/
      umap.py             # 2D projection + plotting utilities
    utils/
      batching.py         # batch helpers for embeddings and sampling (v0.2+)
  tests/
    unit/                # granular tests per module
    integration/
      test_workflow.py
  examples/
    notebook/            # optional notebooks (deferred beyond v0.4 if needed)
  ```
- **Configuration surface**: YAML or dict-driven, validated via `pydantic`. Entry point functions accept typed configs to ensure reproducibility.
- **Configuration surface**: Minimal dataclass-based configs in v0.1 (primarily clustering and visualization toggles), expanded validation via `pydantic` in v0.2.
- **Embedding connectors**: `embeddings/adapters.py` wires LiteLLM to call OpenAI `text-embedding-3-small` for v0.1; broader multi-provider registry (OpenAI, HuggingFace, LiteLLM variants) shifts to v0.3.
- **Data flow**: Ingest list of `ContentUnit` → embed them → run clustering → derive representatives → optionally call labeling pipeline → output `ClusterAssignment` + `LabelResult` collections.

- **Testing hooks**: integration tests use small synthetic corpora and seeded random generators if necessary.

## Shared Design Principles
- Deterministic operations (seeded random states for embeddings where controllable, clustering, and sampling).
<!-- - Dependency injection for external services (embedding models, LLM clients, vector stores) to keep core logic testable. -->
- Token-aware design: Represent clusters via a capped number of representative samples before any LLM invocation.
<!-- - Extensibility: Use strategy classes for clustering and labeling so alternates (e.g., hierarchical clustering, summarization-based labeling) can plug in later. -->
<!-- - Non-destructive visualization: plotting utilities consume copies of data; no mutation of core artifacts. -->

## Versioned Implementation Plan

### v0.1 – Minimal Embedding & Clustering Pipeline
**Goals**
- Establish installable package skeleton with `pyproject.toml`/`setup.cfg` scaffolding.
- Implement embedding ingestion and K-Means clustering with deterministic seeding, attempting silhouette-based auto-selection of `n_clusters` when feasible while retaining a configurable fallback value (references: K-Means – https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html, silhouette score – https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html).
- Generate an initial UMAP-based visualization to inspect cluster separation.
- Expose `constella.pipelines.workflow.cluster_texts(texts, config)` returning cluster IDs, centers, metadata, and any silhouette diagnostics gathered. 
- If data visualisation is requested as a parameter, then generate and save a UMAP projection of the cluster as an image or PDF.

**Modules & Responsibilities**
- `config/schemas.py`: Define lightweight `ClusteringConfig` (fallback `n_clusters`, random seed, silhouette toggles) and optional `VisualizationConfig`.
- `data/models.py`: Define dataclasses for `ContentUnit`, `EmbeddingVector`, `ClusterAssignment`.
- `embeddings/base.py`: Minimal interface `embed_texts(self, texts: List[str]) -> List[List[float]]`. Provide an in-memory mock provider for tests.
- `embeddings/adapters.py`: Implement LiteLLM-backed connector calling OpenAI embedding model `text-embedding-3-small`; expand to multi-provider registry in v0.3.
- `clustering/kmeans.py`: Function `run_kmeans(vectors: np.ndarray, config: ClusteringConfig) -> ClusterAssignment` using scikit-learn KMeans; evaluate candidate cluster counts via silhouette scoring when input size allows and fall back to the configured default, logging inertia either way.
- `visualization/umap.py`: Utility to project embeddings in UI for qualitative inspection of the clusters.
- `pipelines/workflow.py`: Orchestrate embed -> cluster -> (optional) visualize; convert embeddings to numpy arrays. When visualization is requested, save the UMAP plot to disk without forcing display or returning coordinates.
- `config/settings.yaml`: Deferred; use module-level logging configuration in v0.1.

**Data Flow**
1. Client passes `List[ContentUnit]` or `List[str]` with optional lightweight workflow parameters.
2. `cluster_texts` instantiates the LiteLLM-based embedding provider
3. Embedding vectors are generated in-memory, converted to `np.ndarray`, and used to evaluate silhouette scores across candidate cluster counts, to find optimum number of clusters, before running K-Means with the optimum (or fallback) `n_clusters`.
4. UMAP projection is computed from the embedding matrix for visualization. If visualization is requested, persist the plot as an image/PDF (UMAP coordinates need not be returned).
5. Output `ClusterAssignment` object containing cluster assignments per ContentUnit, cluster centers, cluster sizes, and silhouette diagnostics.

**Testing**
- Unit tests for `run_kmeans` (expected cluster counts with seeded inputs), silhouette-based selection paths, `cluster_texts` (ensures pipeline integrates and returns deterministic outputs for synthetic data), and UMAP projection shape validation.
- Integration test verifying packaging import path `import constella` works.


### v0.2 – Representative Selection & Artifact Persistence
**Goals**
- Provide representative sampling near cluster centroids to support labeling.
- Persist clustering artifacts (JSON + optional parquet/arrow) for downstream use.
- Introduce optional disk-backed cache for embeddings to avoid recomputation on reruns.
- Externalize logging defaults via `config/settings.yaml` loader and broaden workflow configuration support.

**Modules & Enhancements**
- `clustering/selection.py`: Implement `select_representatives(assignment, vectors, config)` returning per-cluster indices + similarity scores. Algorithm: compute centroid distances; allow fallback to density-aware sampling (median distance) if clusters > threshold.
- `pipelines/workflow.py`: Extend output to include representative metadata (indices, similarity) and attach raw texts via cross-reference; add configuration for `n_representatives`, `min_cluster_size`, and batching controls introduced at this stage.
- `utils/batching.py`: Provide streaming generator to iterate texts/embeddings; include optional disk caching (e.g., using `joblib` or `.npy` files) wired via config flag.
- `data/models.py`: Add `EmbeddingBatch` (for streamed accumulation) and `RepresentativeSample` dataclasses capturing `cluster_id`, `text`, `vector_index`, `similarity`.
- `pipelines/workflow.py`: Write optional `persist_artifacts(output_dir, assignment, representatives, config)` storing JSON summary plus serialized numpy arrays (respect config toggles).
- `config/settings.yaml`: Introduce shared logging defaults and helper loader hooked into pipeline utilities.
- `config/schemas.py`: Promote full `WorkflowConfig` validation and richer configuration schemas.
- `config/settings.yaml`: Establish shared logging defaults with helper loader used across pipeline utilities.

**Token Efficiency Preparation**
- Representative selection includes configuration to cap tokens: store per-sample word counts and estimated token counts using a configurable heuristic (characters/4 default).

**Testing**
- Unit tests ensuring `select_representatives` picks closest vectors; property tests verifying deterministic outputs.
- Integration test verifying pipeline persistence writes expected files and that reload yields equivalent data.

### Scalability Considerations (v0.3+)
- Support staging large collections (e.g., 10,000 bookmarks/emails) via optional in-memory caches with streaming/batched fallbacks to balance simplicity and real-world throughput.

### v0.3 – LLM-Assisted Labeling with Token Budgets & Adapter Expansion
**Goals**
- Add automated labeling orchestration that utilizes representatives while respecting token budget constraints.
- Provide configurable prompts for initial labeling + optional refinement pass (conceptual inspiration from `theme_manager.py`).
- Introduce asynchronous execution hooks for concurrent LLM calls when supported.
- Reintroduce multi-provider embedding adapter registry with pluggable connectors (OpenAI, HuggingFace, LiteLLM variants, local models) building on LiteLLM foundation.

**Modules & Enhancements**
- `labeling/llm.py`:
  - Define `LabelingConfig` (model name, provider adapter, max_tokens_per_call, max_representatives_per_cluster, budget_strategy).
  - Implement `LabelingOrchestrator` that:
    1. Receives representatives with token estimates.
    2. Applies `budget_strategy` (e.g., greedy accumulate until reaching `max_prompt_tokens`, fallback to summarizing representative subset).
    3. Formats prompts via templates defined in `config/prompts.yaml` (JSON-only responses, similar to legacy theme prompts but refactored for reuse).
    4. Calls injected `LLMClient` interface (async + sync support) that handles retries, exponential backoff, and rate-limit awareness.
    5. Parses JSON safely with guardrails (strip markdown fences, fallback handling, logging on failure).
    6. Optionally runs refinement prompts when `enable_refinement` is true, using additional examples if budget allows.
- `config/prompts.yaml`: Store parameterized prompt templates; include structures to build instructions referencing representative texts, token budgets, and desired output schema.
- `pipelines/workflow.py`: Extend `cluster_texts` (renamed `cluster_and_label`) or add new function that orchestrates labeling stage when `enable_labeling` flag is true. Provide synchronous wrapper that runs asyncio event loop as needed.
- `data/models.py`: Add `LabelResult` (phrase, explanation, confidence, metadata) with optional `subthemes` extension structure.

**Token Control Strategy**
- Estimate prompt tokens per representative using heuristic or optional integration with tokenizer library (exposed via extension point `token_estimator` function in config).
- Allow configs:
  - `max_total_tokens_per_cluster` (cap, default 1500).
  - `min_representatives` (ensure at least 1 sample even when token cap low).
  - `fallback_strategy` (truncate vs. summarize using first representative only).
- Caching: store completed label results keyed by hash of representative texts to avoid duplicate LLM calls (in-memory dictionary with optional disk persistence for reproducibility).

**Testing**
- Mock `LLMClient` returning deterministic JSON to test normal flow, error handling, retry logic, and parsing robustness.
- Ensure token budgeting selects subset matching expectations via targeted unit tests.
- Integration test verifying combined pipeline returns labels within configured token limits.

### v0.4 – Visualization & Advanced Diagnostics
**Goals**
- Introduce UMAP-based dimensionality reduction and plotting utilities capable of reproducing diagnostics similar to `old_code/umap.py` and the associated PNG.
- Provide cluster quality metrics and outlier detection guidance for human analysts.
- Add CLI/`typer` helper (optional) to run pipeline and save visualizations.

**Modules & Enhancements**
- `visualization/umap.py`:
  - Function `project_embeddings(embeddings, config) -> np.ndarray` with configurable `n_neighbors`, `min_dist`, `spread`, `metric`.
  - Function `plot_clusters(projection, assignment, labels, output_path, theme_map)` replicating annotated scatter plots; include optional highlight for outliers (clusters with size < `min_cluster_size` or points beyond percentile threshold).
  - Support headless environments (fallback to Agg backend) and ensure file paths reside within caller-specified output directories.
- `evaluation/metrics.py`: Provide `silhouette_score_safe`, `davies_bouldin_safe`, cluster size distribution summaries, and outlier flags.
- `pipelines/workflow.py`: Allow callers to inject custom embedding providers and related components through configuration for offline/testing scenarios.
- `pipelines/workflow.py`: Add optional call `generate_visual_diagnostics(outputs, viz_config)` generating UMAP projection + metrics JSON; ensure this stage is modular so headless/UMAP optional dependency can be toggled.
- Extend configs to allow specifying visualization dependencies (`umap-learn`, `matplotlib`); guard import errors with clear messaging.

**Testing & Validation**
- Use seeded synthetic data to test that UMAP projection returns expected shape and that plot function writes a file.
- Validate that metrics functions handle small cluster counts gracefully (returning `None` when sample size insufficient).

## Automated Labeling Strategy Details
- **Representative Selection**: Use `select_representatives` outputs sorted by similarity score.
- **Prompt Construction**: Supply at most `max_representatives_per_cluster` texts; each text truncated to `max_chars_per_rep` (configurable) with ellipsis to stay within token limit.
- **LLM Output Schema**: `{"phrase": str, "explanation": str, "confidence": float, "notes": Optional[str]}`. Confidence defaults to heuristic if model does not supply.
- **Refinement**: Optional second-pass using prompt referencing initial phrase + up to `refinement_examples` to disambiguate similar cluster names (inspired by `HierarchicalThemeManager` but simplified to cluster-level only at v0.3).
- **Error Handling**: On JSON parse failure, log warning, retry with fallback prompt that explicitly asks for JSON. After max retries, emit default label `"Cluster <id>"` with explanation `"Automatic labeling failed"` and mark status for downstream review.

## Assumptions, Dependencies & Extension Points
- **Assumptions**
  - Caller supplies raw texts (already cleaned) and can provide embedding/LLM credentials; library does not manage secrets beyond accepting tokens/keys via config/environment variables.
  - Embedding provider returns fixed-length numeric vectors compatible with numpy arrays.
  - Datasets fit in memory for v0.1–v0.2; streaming or chunked persistence considered extension beyond v0.4.
- **Dependencies**
  - Core: `numpy`, `scikit-learn` (KMeans, metrics), `pydantic` or `dataclasses-json` (config validation), `matplotlib`, `umap-learn` (optional, flagged in extras), `aiohttp` or provider SDK for async LLM calls (abstracted via interface).
  - Testing: `pytest`, `pytest-asyncio`, `hypothesis` (optional for property-based tests), `numpy.testing` utilities.
- **Extension Points**
  - Alternative clustering: implement new strategy in `clustering/` (e.g., HDBSCAN) and register via config.
  - Embedding backends: add new adapters in `embeddings/adapters.py` implementing base interface.
  - Token estimators: pluggable callable to calculate token usage using actual tokenizer libraries (e.g., `tiktoken`).
  - Visualization exporters: future integration with Plotly or panel dashboards via `visualization/` plugin system.

## Data & Artifact Management
- Provide structured outputs:
  - `ClusterAssignment`: labels, centers, metrics, config snapshot.
  - `RepresentativeSet`: mapping of cluster_id -> `List[RepresentativeSample]` with token estimations.
  - `LabelCollection`: mapping of cluster_id -> `LabelResult` (with provenance metadata, e.g., LLM model, prompt hash).
  - `DiagnosticReport`: summary metrics, outlier list, optional visualization paths.
- Persist artifacts using JSON (for metadata) and `.npy`/`.pkl` (for numeric arrays) with explicit version tags in filenames (e.g., `clusters_v0.2.json`).


## Testing Strategy Summary
- Maintain separate fixtures for small synthetic corpora with known clustering outcomes.
- Use seeded random numbers to verify deterministic embeddings/clustering results.
- Provide contract tests ensuring any new embedding adapter adheres to interface (raises informative errors when API keys missing).
- Visual regression: simple checksum (hash) of generated PNG to detect major plotting regressions (tolerate small numeric drift by comparing relative coordinate stats instead of pixel-perfect match).

## Documentation & Developer Notes
- Inline docstrings only for complex functions; rely on design doc + code comments sparingly per project guidance.
- Provide `README` updates only upon explicit request; this document should be referenced internally for implementation sequencing.
- Encourage adding `examples/diagnostics.ipynb` (post-v0.4) to demonstrate pipeline usage with visualization outputs.
