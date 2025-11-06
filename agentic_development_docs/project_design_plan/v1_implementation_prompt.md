## Objective
Implement only the **v0.1 – Minimal Embedding & Clustering Pipeline**  for the `constella` package, producing an installable Python library that clusters text corpora and optionally saves a UMAP visualization.

## Key References
- Core requirements live in `@agentic_development_docs/project_design_plan/0_constella_design_plan.md`. Stay strictly within the v0.1 scope; do **not** begin work on v0.2+ features.
- Follow project guardrails in `@agentic_development_docs/general_rules.md`.
- Adhere to style expectations in `@agentic_development_docs/coding_rules.md`.
- The initial plan of this project is given here agentic_development_docs/project_design_plan/initial_plan.md for an idea of the motivation.
- For inspiration of the earlier implementation, you can read and go through this, but this is older version, don't copy verbatim, also the APIs could have changed:
    - clustering implementation is present in old_code/clustering_manager.py and old_code/clustering_manager2.py
    - Clustering plot as UMAP image is old_code/umap.png
    - the cluster plot UMAP implementation is old_code/umap.py


## Deliverables
1. Installable project skeleton (`pyproject.toml`/`constella/` package) aligned with the architecture defined in the design plan.
2. v0.1 module implementations (these are also given in `@agentic_development_docs/project_design_plan/0_constella_design_plan.md`)
   - `config/schemas.py`: lightweight `ClusteringConfig` (+ optional `VisualizationConfig`) capturing seed, fallback `n_clusters`, silhouette toggles, and visualization output path.
   - `data/models.py`: dataclasses for `ContentUnit`, `EmbeddingVector`, and collections that hold embeddings plus cluster IDs directly on each unit.
   - `embeddings/base.py`: interface `EmbeddingProvider` with `embed_texts`. Include an in-memory/mock provider for tests.
   - `embeddings/adapters.py`: LiteLLM-based adapter targeting OpenAI `text-embedding-3-small`, with graceful handling of missing credentials.
   - `clustering/kmeans.py`: deterministic K-Means runner that optionally searches candidate `n_clusters` via silhouette score before falling back to config default; emits cluster labels.
   - `visualization/umap.py`: utility to project embeddings and (when requested) persist a UMAP scatter plot without forcing display.
   - `pipelines/workflow.py`: `cluster_texts(content_units, clustering_config, visualization_config=None)` orchestrating embedding → clustering → optional visualization, returning the enriched `ContentUnitCollection` plus visualization paths when requested.
3. Module-level logging (basic configuration only; defer YAML settings to later versions).
4. Tests under `tests/` validating deterministic clustering, silhouette branch behavior, workflow integration with mock embeddings, and UMAP projection shape/file creation.
5. Create and update README.md file with the README instructions of v0.1 features of the project.

## Implementation Constraints
- Keep operations deterministic by respecting provided seeds and controlling any random sources.
- Limit dependencies to those required for v0.1 (e.g., `numpy`, `scikit-learn`, `umap-learn`, `matplotlib`, `pydantic` or `dataclasses`).

## Testing & Verification
- Add or update pytest-based suites covering the v0.1 functionality (unit + integration as outlined above).
- Run the full test suite (and any linters already configured) before delivery; address failures.

## Output Expectations
- Provide source code, tests, and any generated artifacts within the repository.
- Summarize implementation results succinctly - writing to markdown file in the path @agentic_development_docs/agent_communication_docs , after completion, highlighting tests executed and their outcomes.
