Develop a detailed design plan for the Constella library using the following guidance:

1. Review @agentic_development_docs/project_design_plan/initial_plan.md to ground the design in the stated goals for token-efficient clustering and labeling.
2. Define four incremental implementation phases—version-0.1 through version-0.4—covering the core system evolution without delving into future roadmap items beyond v0.4.
3. Study the legacy references in @old_code/clustering_manager.py, @old_code/clustering_manager2.py, and @old_code/theme_manager.py strictly for conceptual inspiration; do not reuse their code verbatim.
4. Incorporate visualization considerations informed by @old_code/umap.png and @old_code/umap.py so the eventual system can reproduce similar diagnostic plots.
5. Follow the mandates in @agentic_development_docs/rules1.md, keeping the library packaging requirements front and center.
6. When describing automated labeling, outline how LLM-assisted labeling can leverage representative samples while constraining token usage, drawing on lessons from the referenced theme manager.
7. Document any assumptions, data dependencies, and extension points that implementers should be aware of, noting where additional files in project_design_plan may help (feel free to create them if necessary).

Deliverable: A single design document (or well-linked set of documents inside project_design_plan) that captures detailed, implementation-ready notes for each version milestone (v0.1–v0.4). The output will be handed to an AI coding agent, so be explicit about module structure, configuration surfaces, data flow, and testing hooks. Do not implement any code—focus entirely on clear, actionable design instructions.