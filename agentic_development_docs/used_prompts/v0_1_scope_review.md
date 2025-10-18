## v0.1 Scope Simplification Review

### Required Goals (must remain in v0.1)
- Package skeleton with install metadata.
- End-to-end embedding ingestion and K-Means clustering with deterministic seeding plus silhouette-based auto-selection fallback.
- UMAP projection for qualitative inspection and ability to save the plot when requested.
- `cluster_texts` pipeline returning cluster IDs, centers, metadata, silhouette diagnostics, and UMAP coordinates.

### Candidates to Defer or Simplify
1. **Multi-provider adapter layer**
   - Current plan: `embeddings/adapters.py` registers OpenAI, HuggingFace, LiteLLM backends via a provider registry (Data Flow step 2).
   - Impact: Introduces dependency management and abstraction overhead beyond the minimal embed-and-cluster loop.
   - Recommendation: For v0.1 rely on LiteLLM to reach a single OpenAI model (`text-embedding-3-small`) without exposing a full registry. Move multi-provider expansion to v0.3.

2. **Comprehensive configuration schemas**
   - Current plan: `config/schemas.py` defines `EmbeddingConfig`, `ClusteringConfig`, and `WorkflowConfig` with adapter identifiers, logging levels, etc., plus WorkflowConfig is referenced as required input in Data Flow step 1.
   - Impact: Adds validation surface area even though v0.1 can operate with a minimal clustering configuration (e.g., fallback `n_clusters`, silhouette toggle, visualization flag).
   - Recommendation: In v0.1 expose only lightweight `ClusteringConfig` and `VisualizationConfig`, accepting a raw list of texts/ContentUnits. Defer full `WorkflowConfig` and richer validation to v0.2.

3. **Abstract embedding interface with batching argument**
   - Current plan: `embeddings/base.py` mandates `embed_texts(self, List[str], batch_size)` despite batching being deferred to later versions.
   - Impact: Enforces parameters that v0.1 does not need, complicating the minimal provider stub.
   - Recommendation: Simplify v0.1 interface to `embed_texts(self, texts: List[str])`; reintroduce batch-aware signatures in v0.2 alongside batching utilities.

4. **Centralized `config/settings.yaml` loader**
   - Current plan: establish shared logging defaults via settings file and helper loader in v0.1.
   - Impact: Adds file I/O and configuration plumbing unrelated to the core embed/cluster/visualize loop.
   - Recommendation: Use module-level logging setup for v0.1. Schedule the shared `config/settings.yaml` loader for v0.2.


5. **Mandatory plot display responsibilities in pipeline module**
   - Current plan: `pipelines/workflow.py` orchestrates the pipeline and "Displays" the UMAP plot.
   - Impact: Coupling plotting side-effects into the core pipeline risks complicating minimal usage; requirement only asks to "generate and save" when requested.
   - Recommendation: In v0.1, have the pipeline return UMAP coordinates (and optional image path if requested) without automatic display. Add richer visualization orchestration in v0.2.
