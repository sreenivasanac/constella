# AI-First Clustering — Design Plan

Status: **design only, awaiting approval.** No code changes accompany this document.

**Base commit.** Branch `ai-clustering` was created from `a905c64` (`improve_refactor`, tag `v0.1.0`, "chore: prepare for PyPI release v0.1.0"). `main` is at `afa715e` ("Merge pull request #10 from sreenivasanac/dev"). `git rev-list --left-right --count main...a905c64` is `4 2`: the 4 commits only on `main` are merge commits (`25027e1`, `9fad6fd`, `b393693`, `afa715e`); the 2 only on `a905c64` are `3138d6c` (split `llm.py` into modular labeling components) and `a905c64` (PyPI prep). `a905c64` is therefore the right base: it is what `account-clustering` was built on and it has the latest library work, which `main` lacks.

## 1. Goals and non-goals

**Goals**

- Groups are **discovered** by embeddings + clustering and **named** by an LLM. No keyword, domain-table or taxonomy rules decide a group.
- Each item gets a short LLM-written description; the description (not the raw name) is what gets embedded.
- Every model is configurable: local (sentence-transformers, Ollama) or cloud through LiteLLM. The account-cleanup run uses xAI Grok (`xai/` prefix, `XAI_API_KEY`).
- Token-efficient: many items per call, disk cache for everything, only cluster representatives reach the labeler.
- Stays a generic library (bookmarks, emails, tickets, document snippets); account cleanup is the driving use case, not a special case in the algorithm.
- Privacy by construction: only names and domains leave the machine.

**Non-goals**

- No fixed taxonomy and no rules-first mode. No LLM-assigns-a-category-per-item replacement for clustering.
- No account actions (deleting, logging in, changing addresses). Constella only organizes.
- No model training or fine-tuning in v1.

## 2. Pipeline

```text
records (JSON/CSV)
 1. ingest + normalize     column mapping; URL -> registrable domain          [deterministic, normalization]
 2. dedup / merge          identical domain -> one item, sources merged       [deterministic, normalization]
 3. describe (LLM)         name + domain -> short phrase + confidence/unsure  [batched, cached]
 4. fetch fallback         unsure only: GET homepage -> title/meta/snippet -> re-describe
 5. embed                  description -> vector                              [local or cloud, cached]
 6. cluster                improved Constella clustering -> groups (+ parents, + unclustered)
 7. label (LLM)            representatives only -> name, explanation, confidence, keywords
 8. decide (optional)      merge / split / needs-review routing               [decision model, v2]
 9. export                 items, groups, manifest, UMAP HTML
```

Steps 1–2 are the only deterministic logic and they never decide a group: they make two rows for the same site one item. For generic text corpora (tickets, snippets) steps 1–4 are skipped and the existing `ContentUnit.text` is embedded directly, so `cluster_texts` keeps working.

## 3. Description generation

- **Input per item:** display name + registrable domain. Nothing else.
- **Prompt (system, fixed so it can be prompt-cached):** "For each site, write what it is in at most 10 words (e.g. *online bank*, *applicant tracking portal*, *food delivery app*). Describe the kind of service, not the brand. If you do not recognise it or the name is ambiguous, set `unsure` instead of guessing."
- **Batching:** 50 items per call, each with a batch-local integer id.
- **Structured output:** `[{id, description, confidence (0–1), unsure (bool)}]` via LiteLLM `response_format` (the same mechanism `LabelPayload` uses today). Validate that every id comes back exactly once; re-ask missing ids in a smaller batch.
- **Unsure signal:** an item needs the fallback when `unsure` is true **or** `confidence < confidence_threshold` (default 0.7). Self-reported confidence is weakly calibrated, so the explicit flag is the primary signal and the threshold is tunable from the evaluation set.
- **Cache key:** SHA-256 of `{domain or name, model, prompt_version, source: knowledge|homepage}`; one JSON entry per item, so a re-run or a batch-size change costs nothing.
- **Homepage fallback:** direct HTTP GET of `https://<domain>/` only (no path, no query string, no cookies), with a timeout, a size cap and a redirect limit, using the standard library. Extract `<title>`, meta/OG description and about 200 tokens of visible text; never send raw HTML. Re-describe in batches of 20 with that context. Page text is untrusted: the prompt marks it as data, and the output schema is a short phrase.
- **Fetch failures** (dead site, bot wall, local/intranet host): keep the first description, mark the item `needs_review`. Local hosts and IPs are never fetched.
- xAI's server-side Web Search is **not** the default: $5 / 1k calls against $0 for a GET (see §10).

## 4. Embedding choice

Any LiteLLM embedding model stays configurable (existing OpenAI and Fireworks adapters are unchanged). For this use case:

| Option | Status | Cost for ~1,400 descriptions |
| --- | --- | --- |
| **Local sentence-transformers** (small English model, e.g. MiniLM class) — proposed default | New optional extra; descriptions never leave the machine | $0 |
| OpenAI `text-embedding-3-small` — cloud fallback | Existing adapter, `OPENAI_API_KEY` | ~$0.0007 |
| Ollama embedding model | LiteLLM `ollama/` prefix, local | $0 |
| xAI embeddings | Not the default: see below | **TBD** |

xAI's OpenAPI documents `POST /v1/embeddings` and `GET /v1/embedding-models`, but the public pricing page lists no embedding prices, LiteLLM's xAI page does not document embeddings, model ids seen in unofficial writeups are unverified, and at least one third-party report shows an empty embedding-model list for their key. The design offers `xai/<embedding-model>` as an optional path to probe later, never as a requirement. xAI Collections (file RAG) is a different product and is not used.

One code-level detail to fix: today `embed_texts` embeds `ContentUnit.get_content()`, which prefixes `Id:` and metadata lines. The description pipeline embeds the bare description so identifiers do not pull vectors together.

## 5. Clustering algorithm

| | K-Means (current) | HDBSCAN | Agglomerative (hierarchical) |
| --- | --- | --- | --- |
| Choosing scale | `k` search; today capped at `max_candidate_clusters=15` and takes the **max** of silhouette / elbow / Davies–Bouldin picks | `min_cluster_size`; `k` emerges | One tree, cut at any `k` or distance |
| Outliers | None; every item is forced into a cluster | Explicit noise label; on short texts the noise share can be large | None natively; tiny clusters can be treated as unclustered |
| Multi-level groups | Needs a second run per cluster (what the rejected branch did) | Flat in scikit-learn's implementation | Native: two cuts of one tree nest exactly |
| Determinism | Seeded | Deterministic, but usually run after UMAP reduction, which is stochastic | Fully deterministic, no seed |
| Scale | 100k+ | 100k+ | O(n²) memory: fine to roughly 20k items |
| Dependency | scikit-learn | `sklearn.cluster.HDBSCAN` (already satisfied by `scikit-learn>=1.3`) | scikit-learn |

**Recommendation:** add a `ClusteringConfig.algorithm` switch with all three, and make **agglomerative on L2-normalised vectors** (Ward linkage; linkage configurable) the default for the description pipeline:

- ~1,400 short descriptions is small, so O(n²) is irrelevant, and the user wants every service placed.
- "Finance > Banking" falls out of one dendrogram: a fine cut gives groups, a coarser cut of the same tree gives parents, guaranteed nested.
- **Choosing `k`:** build the tree once, then score many cuts cheaply (range scaled to `n`, e.g. 2 … n/10) with cosine silhouette under a minimum-group-size constraint. The parent level is a second, coarser cut picked the same way. An explicit `n_clusters` / `distance_threshold` override stays available.
- **Outliers:** groups below `min_cluster_size` become `Unclustered` (cluster id −1), skipped by representative selection and labeling, surfaced for review rather than forced into a wrong group.

K-Means remains the default for plain `cluster_texts` on large corpora (backward compatible), with its candidate range scaled to `n` instead of the fixed cap of 15. HDBSCAN is offered for callers who prefer explicit noise. The default is a hypothesis to confirm on the gold set (§12), not a claim.

## 6. LLM labeling of clusters

Reuse the existing flow unchanged in shape: `select_representatives` (centroid-near core + diversity picks) → `build_labeling_messages` → LiteLLM structured `LabelPayload` (label, explanation, confidence, keywords). Adaptations:

- Representatives show `name — description` only; 6–8 per cluster is enough for one-line items (default is 20 today).
- `LabelingConfig.model = "xai/grok-4.7"`; `get_provider_config` already returns no overrides for unknown providers, so LiteLLM reads `XAI_API_KEY` itself. Add a `reasoning_effort` pass-through.
- Parent groups are labeled from their children's labels (one short call per parent), not from more items.
- Results go through the disk cache carried over from the rejected branch.

## 7. "Jev" interpretation

**Primary reading — a decision model.** Jev is TypeSafe AI's "System One" decision model: unstructured *state* plus typed *questions* in, typed answers with probabilities out (`choice`, `score`, `noul`); it does not write free text. The Hugging Face post "Decision Models in Hugging Face" (10 Nov 2026) shows the same interface with the open model `autotrust/GEV-26B-Decide` via `pipeline("decision", ...)`, which likely explains the original spelling "Gev". Sources: https://sproutstack.dev/learn/comparisons/jev-vs-jepa-vs-llm, https://huggingface.co/blog/decision.

**Where it fits.** Closed-option decisions around the pipeline, not the pipeline itself:

- After describing: `confident` / `needs_homepage_fetch` / `needs_human_review`.
- After clustering: is this cluster too mixed (split)? should these two merge? which hierarchy cut?

It does **not** replace description writing (generative) or embed + cluster (group discovery).

**Design.** A small decision interface — `decide(state, question) -> {option, probabilities, confidence}` — with interchangeable backends: (a) a Grok structured-output stand-in (portable, uses the planned Grok credits), (b) local GEV through Hugging Face, (c) TypeSafe Jev. No hard dependency on TypeSafe, whose API is in early access. In v1 the first decision is simply the `unsure` field of the description response, so no extra call is needed; merge/split is a later phase.

**Caveats.** Typed output guarantees shape, not correctness. Independent work ("Type-Safe Is Not Error-Free", "JevOut") showed decision models can follow option *names* over definitions and can be flipped by extra context, so low-confidence answers fall back to an LLM or a human. Vendor-reported Jev pricing on Cloudflare Workers AI ($0.042 / 1M input tokens, per the explainer above) may change and is not used in §10.

**Alternate reading — JEPA.** Joint-Embedding Predictive Architecture (LeCun / Meta: I-JEPA, V-JEPA, research LLM-JEPA) predicts missing *representations* rather than tokens. The names collide (there is even a 2026 "Jev-JEPA" preprint), and "strongly consider Gev and embedding" could point here. JEPA is a representation-learning research line, mostly vision/video, not a drop-in text-clustering API, so it is not proposed for v1. See open question 1.

## 8. `account-clustering`: kept, reworked, dropped

| Verdict | Item |
| --- | --- |
| **Keep** | `records/ingest.py` (JSON/CSV column mapping); `records/privacy.py` (`scrub`, `safe_text`); `labeling/cache.py` (generalised to also cache descriptions); `embeddings/fake.py` (offline tests); export of items / groups / manifest; CLI entry point |
| **Rework** | `records/normalize.py`: keep URL cleaning and registrable-domain extraction; the hand-curated brand-alias, product-split and platform-tenant tables become an optional caller-supplied alias map, empty by default. `records/merge.py`: keep grouping by domain and date/flag aggregation. `pipelines/organize.py`: rewritten as the pipeline in §2. `records/models.py`: drop rule-provenance fields |
| **Drop** | `records/rules.py` entirely (domain→category map, keyword rules, source-category priors, public-suffix sector rules); the fixed 20-label taxonomy and `fixed` / `hybrid` modes; `rules` mode; clustering only the "leftovers" |
| **Out of the algorithm** | `records/caller.py` (former employment, mailing regions, shared phones) and `records/actions.py`: optional caller metadata attached **after** clustering, never an input to it; deferred until the core is approved |

## 9. Config surface and CLI

New or extended dataclasses in `constella.config.schemas`:

- `DescriptionConfig`: `model`, `reasoning_effort`, `batch_size=50`, `confidence_threshold=0.7`, `fetch_homepage=True`, `fetch_timeout`, `prompt_version`.
- `EmbeddingConfig`: `backend` (`local` | `openai` | `fireworks` | `litellm` | `fake`), `model`.
- `ClusteringConfig` (extended): `algorithm` (`agglomerative` | `hdbscan` | `kmeans`), `levels` (1 | 2), `min_cluster_size`, `linkage`, existing K-Means fields.
- `LabelingConfig` (existing) plus `reasoning_effort`.
- `DecisionConfig`: `backend` (`off` | `llm` | `hf` | `jev`), `model`, `min_confidence`.
- Shared: `cache_dir`, `privacy=True`.

Models are LiteLLM strings: `xai/<model>` (`XAI_API_KEY`), OpenAI and Fireworks as today (`OPENAI_API_KEY`), `ollama/<model>` for local chat. No key is needed for local embeddings.

```text
constella organize INPUT [INPUT ...] --output-dir DIR
    [--describe-model xai/grok-4.7] [--label-model xai/grok-4.7] [--reasoning-effort low]
    [--embedding local|openai|fireworks|litellm|fake] [--embedding-model NAME]
    [--cluster-algo agglomerative|hdbscan|kmeans] [--levels 1|2] [--min-cluster-size N]
    [--confidence-threshold 0.7] [--fetch on|off] [--decision off|llm|hf|jev]
    [--cache-dir DIR] [--privacy on|off] [--dry-run]
```

`--dry-run` prints item count, planned calls and estimated tokens without calling anything.

## 10. Caching and cost for ~1,400 services

**Caches (all on disk under `cache_dir`):** descriptions per item, homepage extracts per domain, embeddings per `(model, text)`, labels per `(model, prompt_version, messages)`. A second run with unchanged inputs costs about $0; new services cost only their own share.

**Prices used** (xAI pricing page, https://docs.x.ai/developers/pricing, short context, per 1M tokens): `grok-4.7` $2.00 input / $0.50 cached input / $6.00 output; `grok-4.3` $1.25 / $0.20 / $2.50. Batch API: 20% off. Web Search tool: $5 / 1k calls. OpenAI `text-embedding-3-small`: $0.02 / 1M tokens. LiteLLM xAI provider: https://docs.litellm.ai/docs/providers/xai.

**Assumptions:** `xai/grok-4.7`, `reasoning_effort=low` so reasoning tokens are not a dominant extra; no prompt-cache credit; token counts are estimates.

| Stage | Assumptions | Input | Output | Subtotal |
| --- | --- | --- | --- | --- |
| A. Describe (knowledge-first) | 50 sites/call → 28 calls; ~400-token system prompt + ~15 tokens/site ≈ 1,150 in/call; ~40 out/site | 28 × 1,150 = 32,200 → $0.064 | 28 × 2,000 = 56,000 → $0.336 | **~$0.40** |
| B. Homepage fallback | 15% unsure → 210 sites; HTTP GET ($0); 20/call → 11 calls; ~4,700 in/call | 51,700 → $0.103 | 11 × 20 × 40 = 8,800 → $0.053 | **~$0.16** |
| C. Label clusters | ~50 clusters × 6 representatives; ~490 in, ~80 out per cluster | 24,500 → $0.049 | 4,000 → $0.024 | **~$0.07** |
| D. Embeddings | ~1,400 × ~25 tokens ≈ 35,000 tokens | local: $0 · OpenAI: ~$0.0007 · xAI: **TBD** | | **$0** |

- **Total, recommended path:** $0.40 + $0.16 + $0.07 ≈ **$0.63**; allow **$0.60–$0.80** with parse retries. A parent-label level (~10 more calls of size C) adds about $0.015.
- **Same traffic on `grok-4.3`:** $0.18 + $0.09 + $0.04 ≈ **$0.31**.
- **Batch API (20% off, results typically within 24 hours):** `grok-4.7` total ≈ **$0.50**.
- **If the fallback used xAI Web Search:** 210 × $5 / 1,000 = **$1.05** plus tokens, more than the whole recommended run.
- **Reasoning warning:** `grok-4.7` defaults to `reasoning_effort=high` and reasoning tokens are billed as output. Illustration only (not measured): 3,000 extra output tokens per description batch is 28 × 3,000 = 84,000 → **$0.50** extra on stage A alone. Use `low` (or a cheaper model) for these short structured calls.
- An optional Grok-backed decision stage over ~50 clusters is about the size of stage C (~$0.07); not included above.

## 11. Privacy

- **Sent to remote chat models:** service display name and registrable domain; on fallback, the public homepage's title, meta description and text snippet. Titles pass through `scrub` first, because vault titles sometimes embed an email address.
- **Sent to a remote embedding model (only if a cloud backend is chosen):** the generated description.
- **Never sent:** usernames, email addresses, passwords, raw email bodies, URL paths or query strings, dates, card/address flags, caller config.
- Private fields live only on the record objects and are never placed in a `ContentUnit`, so no prompt builder can reach them.
- The homepage GET requests the bare root URL without cookies or credentials; it does tell that site an anonymous client visited. `--fetch off` disables it.
- Logs and the manifest hold counts, model names and cache statistics only. `--privacy off` affects local exports only, never remote payloads.

## 12. Evaluation plan

- **Gold set:** 150–200 services hand-grouped by the user (kept local). Report pairwise precision/recall, purity, and ARI / NMI against it.
- **Purity spot checks:** banks land together; job-application portals together and not inside Shopping; developer tools apart from AI tools unless the tree says otherwise.
- **Internal metrics, used with care:** cosine silhouette and Davies–Bouldin to compare cuts of the *same* algorithm. For HDBSCAN report noise fraction and cluster persistence instead; silhouette penalises non-convex clusters and ignores noise, so it must not rank HDBSCAN against the others.
- **Stability:** agreement (ARI) across subsamples and across embedding models.
- **Description quality:** manual check of ~50 descriptions; unsure rate; how many unsure items the fetch resolves.
- **Ablation:** cluster raw `name + domain` embeddings versus description embeddings on the gold set, to confirm the description step earns its cost.
- **Operational:** share of unclustered items, label confidence, actual tokens versus §10.

## 13. Test plan (offline, no real API calls in CI)

- Fake chat and embedding providers injected everywhere; a test fails if `litellm` is called unmocked.
- Description batching: batch boundaries, id round-trip, missing or duplicate ids, malformed JSON → retry.
- Cache: second run makes zero provider calls; changing model or `prompt_version` invalidates.
- Unsure path: low confidence or `unsure` → mocked HTTP fetch → re-describe; fetch error → `needs_review`; `--fetch off` → no HTTP at all.
- HTML extraction: title / meta / snippet, size cap, non-HTML responses.
- Privacy: a sentinel username, email and query string in the input must not appear in any captured prompt, embedding input, cache file or log.
- Clustering: each algorithm on synthetic blobs; nested two-level cuts; small groups → unclustered; determinism; existing K-Means tests still pass.
- Labeling: representatives only (payload size bounded regardless of cluster size); id −1 skipped.
- Decision interface: fake backend; low confidence → fallback.
- CLI: `--dry-run` and an end-to-end run on a synthetic fixture with fakes.

## 14. Milestones

0. **Design approval** — this document and the open questions below. Nothing is built before this.
1. Core library: clustering backends + `algorithm` switch, local and generic-LiteLLM embedding providers, tests.
2. Description stage: batching, structured output, cache, homepage fallback.
3. Records adapter: port ingest / merge / privacy / export / CLI from `account-clustering`; rules layer not ported.
4. Evaluation on a synthetic fixture and the gold set; then, with explicit go-ahead on the §10 estimate, the real ~1,400-service run.
5. Later: decision-model stage (merge/split), caller metadata, README update.

## 15. Open questions for the user

1. **What did "Jev - decision model" mean?**
   A) TypeSafe Jev / Hugging Face decision models (incl. `GEV-26B-Decide`) for routing and merge/split.
   B) The same decision interface backed by Grok structured output (**recommended default**; A pluggable later).
   C) JEPA-style representation learning (not recommended for v1).
   D) Something else — please name it.
2. **Default embedding backend?** A) local sentence-transformers (**recommended**: $0, private); B) OpenAI `text-embedding-3-small` (~$0.0007); C) probe xAI embeddings first (price and availability TBD); D) Ollama.
3. **Default clustering algorithm for this pipeline?** A) agglomerative (**recommended**: nested levels, deterministic); B) HDBSCAN (explicit noise, flat); C) keep K-Means with a wider `k` range.
4. **Grok chat model?** A) `grok-4.7` with `reasoning_effort=low` (~$0.63, **recommended**); B) `grok-4.3` (~$0.31, try if quality holds on the gold set); C) `grok-4.7` at default high reasoning (cost not bounded by §10). Also: real-time or Batch API (20% off, up to ~24 h)?
5. **Decision-model stage timing?** A) v1 uses only the `unsure` field, merge/split later (**recommended**); B) full decision stage in v1.
6. **Flat or multi-level groups?** A) two levels, e.g. Finance > Banking (**recommended**); B) flat; C) full tree export.
7. **Where do records ingest/merge live?** A) a thin `constella.records` adapter, core stays text-only (**recommended**); B) fully in core; C) outside the package as a script.
8. **Brand aliases** (sibling domains of one company, shared multi-tenant platforms)? A) dedup by identical registrable domain only (**recommended**; the rest is left to clustering); B) optional caller-supplied alias file; C) keep the curated tables from `account-clustering`.
9. **Unclustered items?** A) leave as `Unclustered` for review (**recommended**); B) attach to the nearest group; C) ask the decision model.
10. **Gold set:** are you willing to hand-group ~150–200 services so cluster quality can be measured rather than eyeballed? A) yes; B) smaller set (~50); C) eyeball only.
