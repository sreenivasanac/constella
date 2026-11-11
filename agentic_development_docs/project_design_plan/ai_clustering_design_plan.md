# AI-First Clustering — Design Plan (v2)

Status: **design only, awaiting approval.** No code changes accompany this document. v2 records the answers to the v1 open questions and is backed by experiments on the real data (2026-11-11). Experiment scripts, outputs and group listings are outside the repo in `~/acct_cleanup/experiments/` (they contain personal service lists and are deliberately not copied here). Base commit for the work is unchanged: `a905c64` (`v0.1.0`).

## 1. Decisions (from the v1 questions)

| # | Decision |
| --- | --- |
| 1 | **Jev** is TypeSafe AI's decision model. Grok writes text; Jev answers closed questions. `decide()` stays backend-pluggable (§5). |
| 2 | **Embeddings:** xAI was probed and is not available for this key (§3). Default is OpenAI `text-embedding-3-small`; local sentence-transformers is the zero-egress option. |
| 3 | **Algorithm:** chosen from data (§6): LLM-named top level + decision-model filing + size-balanced Ward second level. |
| 4 | **Chat model:** `grok-4.7`, real-time, `reasoning_effort=low`. `--batch` uses the Batch API (20% off, up to ~24 h). |
| 5 | **Output is a tree** (2 levels) where an item may sit in more than one node; node sizes are balanced (§7). |
| 6 | **No alias tables, no sibling-domain heuristics.** The only normalization is dedup by identical registrable domain. Related brands meet through descriptions + embeddings. |
| 7 | Items that fit nowhere stay **Unclustered** for review. |
| 8 | **No gold set.** Evaluation is LLM-judge spot checks, metrics and a readable tree printout (§9). |
| 9 | Records ingest/merge stays a thin `constella.records` adapter; the core stays text-only. |

Goals are unchanged from v1: groups are discovered and named by models, never by keyword/domain rules; only names and registrable domains leave the machine; every model is configurable; only representatives reach the labeling LLM; the library stays generic.

## 2. Pipeline

```text
records (JSON/CSV)
 1. ingest + dedup       URL -> registrable domain; identical domain -> one item      [deterministic]
 2. describe             name + domain -> 3-10 word description, confidence, unsure   [Grok, 50 items/call, cached]
 3. route                confident / needs homepage fetch / needs review              [decide: choice]
 4. fetch + re-describe  GET https://<domain>/ -> title + meta description            [HTTP, then Grok]
 5. embed                "name — description" -> vector                               [cached]
 6. propose branches     60 micro-cluster summaries -> 8-14 named top branches        [Grok, 1 call]
 7. file items           nearest branch by embedding; decision model when unsure      [decide: choice]
 8. balance top level    split branches > 15% of items, merge branches < 3%           [deterministic]
 9. second level         Ward inside each branch, then split > 40 / merge < 4         [deterministic]
10. unclustered          far-from-centroid items and "no branch fits" -> Unclustered
11. label nodes          6-8 representatives per node, 25 nodes per call              [Grok]
12. multi-membership     centroid-similarity candidates, each confirmed yes/no        [decide: noul]
13. review edits         merge two siblings? split a loose node?                      [decide: noul; Grok names new nodes]
14. export               tree JSON, indented markdown tree, UMAP HTML, manifest
```

For plain text corpora (tickets, snippets) steps 1–4 are skipped; `cluster_texts` keeps K-Means as its default and gains `algorithm=agglomerative|hdbscan`. With `--decision off` and no chat model, steps 6–7 fall back to a balanced Ward top level, so a fully offline path exists.

## 3. Embeddings: xAI probe and chosen backend

Probe run 2026-11-11 with the project's `XAI_API_KEY` (result file: `~/acct_cleanup/experiments/xai_embedding_probe.json`):

| Call | Result |
| --- | --- |
| `GET /v1/embedding-models` | HTTP 200, `{"models": []}` — no embedding model is listed for this team |
| `GET /v1/models` | HTTP 200, 13 models, all chat/image/video (`grok-4.7`, `grok-4.6`, `grok-4.5`, `grok-4.3`, `grok-4.20-*`, `grok-build-0.1`, `grok-imagine-*`); no embedding model |
| `POST /v1/embeddings`, model `v1` | HTTP 404 `not-found`: "The model v1 does not exist or your team … does not have access to it" |
| `POST /v1/embeddings`, model `grok-embedding-small` | HTTP 404, same message |

No vectors, dimensions or embedding prices could be obtained. `/v1/models` does return price fields for chat models: `grok-4.7` has `prompt_text_token_price` 20000, `cached_prompt_text_token_price` 5000, `completion_text_token_price` 60000. Matching these to the published $2.00 / $0.50 / $6.00 per 1M tokens gives the unit: **1 unit = $0.0001 per 1M tokens**.

**Chosen backend:** OpenAI `text-embedding-3-small` (1,536 dims), the backend all experiments ran on: 13,873 tokens for 1,322 items = **$0.0003**. Only the generated "name — description" text is sent. Local `all-MiniLM-L6-v2` (384 dims, $0) also ran and stays available as `--embedding local`, but its trees differ noticeably from the OpenAI ones (adjusted Rand index 0.24–0.44 between backends for the same algorithm) and were not judged, so it is an option, not the default. `xai/<model>` stays a config value to retry if xAI enables embeddings for the account.

## 4. Descriptions (measured)

Prompt and schema as in v1: `{id, description, confidence, unsure}` per item, 50 items per call, cache keyed per item.

- 1,322 entities (2,248 vault logins + 657 mail-inventory rows, deduped by identical registrable domain; 66 have no domain). 27 calls, all ids returned, median description 5 words.
- Latency: median 28.7 s per call, max 39.7 s; about 3.2 minutes of wall time with 4 concurrent calls.
- Tokens: 53,553 input (31,488 billed as cached) and 69,041 output, of which 35,092 were reasoning tokens even at `low`. Cost **$0.474**.
- **Unsure rate 20.1%** (266 items; 271 with `confidence < 0.7`).
- Homepage fallback on a sample of 40 unsure items: 23 pages returned a usable title/meta description, 17 failed or were empty; after one re-describe call ($0.012) **17 of 40 (43%) became confident**. Worth keeping, but more than half of unsure items stay unsure and should go to Unclustered rather than be filed on a guess (see §6 examples).

## 5. Grok + Jev division of labor

Jev ("System One", `jev-1.14.0` / `jev-1.13.0`) takes a *state* plus typed questions (`choice`, `score`, `noul`) and returns answers with probabilities. It costs **$0.042 per 1M input tokens, output free**, so closed questions should go to Jev and only text generation to Grok. Docs: https://docs.typesafe.ai/models, https://docs.typesafe.ai/introduction/quickstart.

| Step | Model | Question |
| --- | --- | --- |
| Describe an item; re-describe from homepage text | **Grok** | generation |
| Route after describing | **Jev** `choice` | confident / needs homepage fetch / needs human review (v1 fallback without Jev: the `unsure` flag) |
| Propose 8–14 top branches (name + one-sentence description) | **Grok** | generation, 1 call |
| File an item into a top branch | embedding first, **Jev** `choice` when the margin is small | options = branch names + "none of these fit" |
| Does this item also belong in node X? | **Jev** `noul` | one question per candidate node, all candidates of an item in one request |
| Merge these two sibling nodes? Split this node? | **Jev** `noul` | state = node labels + a few representatives |
| Label second-level nodes; name nodes created by a split | **Grok** | generation, representatives only |

**Interface.** `decide(state: str, questions: dict[str, Question]) -> dict[str, Answer]`, where `Question = {type: choice|noul|score, instructions, options?}` and `Answer = {choice, probabilities, confidence}` or a probability/score. One state with many questions is one request, which is how Jev bills cheaply (questions are evaluated independently against the same state).

**Backends** (`DecisionConfig.backend`): `typesafe` (`POST https://api.typesafe.ai/v1/systemone`, `TYPESAFE_API_KEY`, `jev-latest`); `openrouter` (`typesafe/jev-1.13` at `/api/alpha/decisions`, `OPENROUTER_API_KEY`, no waitlist, returns `usage.cost`); `cloudflare` (Workers AI `typesafe/jev`); `hf` (local `autotrust/GEV-26B-Decide` via `pipeline("decision")`); `grok` (structured-output stand-in that batches many states per call); `off`. LiteLLM's `/typesafe` pass-through is an alternative transport (https://docs.litellm.ai/docs/pass_through/typesafe).

**Guards.** Typed output guarantees shape, not correctness; "Type-Safe Is Not Error-Free" and "JevOut" report name-over-definition bias and context flips. So: option names are descriptive and the definition is repeated in the instructions; answers below `min_confidence` go to Grok or to review; a `noul` alone never triggers a split (see §7).

**What was tested.** This machine has no TypeSafe or OpenRouter key, so every decision step ran on the Grok stand-in behind the same interface, and Jev cost is estimated from token sizes (§8). Filing all 1,322 items with the stand-in: no ids missing, 30 items answered "no folder fits", agreement with plain embedding assignment was 71% overall and **39% / 63% / 85% / 96%** from the lowest to the highest quartile of embedding margin (best minus second-best branch similarity). That supports the gate: file by embedding when the margin is large, ask the decision model otherwise.

## 6. Clustering experiments and the chosen algorithm

**Setup.** n = 1,322 entities, embeddings of "name — description" (OpenAI), L2-normalised. Ten approaches, each producing a two-level tree. Nodes were labelled by `grok-4.7` (low) from 7 representatives. Judge: `grok-4.7` (low) rated 15 sampled second-level nodes per approach for coherence (1–5) with one planted intruder each, and gave one 1–5 rating to each labelled tree. Size bands used: 8–14 top branches of 3–15% of items each; second-level nodes of 4–40 items.

| Approach | top / 2nd nodes | top min/med/max | 2nd min/med/max | Gini top | % items in top >15% / 2nd >40 | Unclustered % | Silh. 2nd | Intra cos | Judge coherence | Intruder found | Tree 1–5 | Multi % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ward, two cuts of one tree | 14 / 120 | 15/79.5/395 | 3/10/40 | 0.417 | 29.9 / 0 | 0.0 | 0.104 | 0.482 | 3.93 | 1.00 | 3 | 1.6 |
| Ward + size balancing | 12 / 117 | 60/100/181 | 4/9/33 | 0.198 | 0 / 0 | 2.8 | 0.106 | 0.485 | 4.00 | 0.87 | 3 | 2.4 |
| HDBSCAN, raw vectors | 14 / 40 | 7/26.5/82 | 5/8/35 | 0.364 | 0 / 0 | **64.4** | 0.195 | 0.535 | 4.13 | 1.00 | 2 | 0.0 |
| HDBSCAN after UMAP | 14 / 89 | 26/66.5/163 | 5/10/104 | 0.316 | 0 / 7.9 | 16.9 | 0.109 | 0.478 | 3.73 | 0.93 | 3 | 0.6 |
| K-Means (k = n/15 = 88) | 12 / 88 | 32/115/209 | 3/14/51 | 0.270 | 15.8 / 3.9 | 0.0 | 0.076 | 0.450 | 3.60 | 0.87 | 2 | 4.7 |
| BERTopic-style (UMAP + HDBSCAN + c-TF-IDF) | 14 / 38 | 34/67/152 | 11/24/114 | 0.306 | 0 / 27.9 | 13.5 | 0.094 | 0.428 | 4.00 | 1.00 | 2 | 1.4 |
| LLM taxonomy, both levels | 11 / 80 | 76/126/164 | 1/15/49 | 0.131 | 0 / 3.7 | 0.0 | 0.037 | 0.417 | 3.60 | 0.87 | **4** | 11.1 |
| LLM taxonomy, re-centred on members | 11 / 78 | 82/114/164 | 3/15.5/43 | 0.126 | 0 / 6.4 | 3.8 | 0.082 | 0.448 | 3.73 | 0.80 | **4** | 4.2 |
| LLM top level + balanced Ward | 11 / 108 | 74/116/165 | 4/10/35 | 0.119 | 0 / 0 | 3.0 | 0.076 | 0.468 | 3.87 | 0.93 | 3 | 5.9 |
| **LLM top level + decide() filing + balanced Ward (chosen)** | 11 / 99 | 76/106/185 | 4/11/36 | 0.147 | 0 / 0 | 4.7 | 0.085 | 0.466 | **4.13** | 0.93 | 3 | 3.9 |

Silhouette is cosine over second-level nodes with unclustered items excluded, which flatters HDBSCAN. "Multi %" is the share of items with a candidate second node (as close to another node's centroid as that node's median member). Full table, Gini of the second level and the judge's comments: `~/acct_cleanup/experiments/comparison.md`. Listings: `groups_<approach>.md`; the chosen tree: `best_groups.md`.

**What the data says.**

- **Silhouette cannot choose the granularity.** On one Ward tree it rises monotonically from 0.060 at k=30 to 0.104 at k=120, so "best silhouette" always picks the largest k allowed. Size bands, not silhouette, must set the number of nodes; silhouette is only used to compare cuts inside a band.
- **HDBSCAN discards too much.** 64% of items become noise on raw vectors and 17% after UMAP, with a 104-item node left over. The user wants every recognisable service placed.
- **Embedding-only top levels are the weak part.** Plain Ward put 30% of items in one branch. Balancing fixes sizes (Gini 0.42 → 0.20) but not meaning: the judge found junk-drawer branches and a branch themed by region instead of function.
- **An LLM-proposed top level is better balanced and better named** (Gini 0.12–0.15, no branch over 15%) for one Grok call that sees 60 micro-cluster summaries, not the items.
- **Letting the LLM also define the second level looks tidy but fits worst**: lowest silhouette (0.037) and coherence (3.60), oversized leaves, and 11 more Grok calls. Its sub-folder names are written before the contents are known.
- **Ward inside each branch gives the tightest leaves that still respect the bands**, and it is deterministic.

**Qualitative examples** (well-known brands only):

- Embedding-only balanced Ward filed Netflix in a leaf of unrelated video sites under a Shopping branch; the chosen approach files it under Media > Media Streaming Services next to Spotify.
- Balanced Ward put T-Mobile under a region-themed branch and Verizon under a generic "service account portals" leaf; the chosen approach puts both in one "US Telecom Carriers" node.
- The decision step gave LinkedIn the primary home Jobs > Professional Networking and a second branch Social; Uber got Travel > Rideshare plus a second branch for food delivery. That is the multi-membership behaviour asked for.
- Still wrong in the chosen tree: USPS landed in an unrelated government-portal leaf, Slack in a scheduling-tools leaf, and unrecognised items were swept into the Social branch as an "unidentified startups" node instead of going to Unclustered.

**Chosen algorithm: Grok proposes the top level, items are filed by embedding with a decision-model check, and each branch gets a size-balanced Ward second level.** Reasons:

1. *Data:* it ties for the best leaf coherence (4.13), has no node outside the size bands at either level, and a top-level Gini of 0.147 against 0.417 for plain Ward.
2. *README goals:* token-efficient (the taxonomy call reads 240 descriptions, labels read 7 items per node) and reproducible (Ward has no seed, micro-clusters are seeded, model calls are cached).
3. *The user's design:* it is exactly the Grok + Jev split — Grok generates names, the decision model files items — and filing is where the hybrid without `decide()` went wrong.
4. *Research:* BERTopic's lesson is kept (label from representatives, never from all items), and Jev's price ($0.042 per 1M input tokens) makes a per-item filing decision affordable where a Grok call per item is not.

**Limits of this evidence.** The judge sample is 15 nodes per approach, so differences of about 0.3 in coherence are noise; the top five approaches are within that. The tree rating is a single call that sees labels, not contents, and it scored the pure-LLM taxonomy 4 against 3 for the chosen approach. The judge is the same model that wrote the descriptions. The final check is the user reading `best_groups.md`.

## 7. Tree, multi-membership and size balancing

**Data model.** `Node {id, label, parent_id, level, size}`; `Membership {item_id, node_id, kind: primary|secondary, score}`. Every item has exactly one primary membership (or Unclustered) and zero to two secondary ones. Sizes and balancing count primary members only, so multi-membership cannot inflate a node.

**Size bands** (config, defaults from the experiment): 8–14 top branches; a top branch holds 3–15% of items; a second-level node holds 4–40 items. `n` scales the bands, not fixed counts.

**Balancing loop** (deterministic, ran in the experiment with zero out-of-band nodes at the end):

1. Split: any node above its upper band is cut in two with Ward on its own members; repeat until none is too large.
2. Merge: the smallest node below its lower band joins the sibling with the nearest centroid, if the merged node stays within the band; repeat.
3. Second level: per branch, cut a Ward tree at the k with the best silhouette inside `[ceil(m/40), min(12, m/8)]`, then run steps 1–2 with the leaf band.
4. Review (decision model): for the closest sibling pairs ask "merge?", for the loosest nodes ask "split?". In the stand-in test the model said merge for 1 of 10 closest pairs and **split for 10 of 10 loosest nodes** — a yes-bias. So a split needs both a `noul` above threshold and a resulting pair that respects the lower band; merges are applied only inside the band. New nodes are named by Grok.

**Unclustered.** Items whose similarity to their own node centroid is below mean − 2 sd, items the decision model answers "none of these fit" for, and (proposed, see open question 4) items still unsure after the homepage fetch. The chosen run applied the first two rules: 62 items (4.7%).

**Multi-membership.** Two sources, both confirmed by a `noul` before they are stored:

- *Second top branch:* the filing question may name one. The stand-in named one for 326 items (24.7%).
- *Second node by similarity:* an item is a candidate for node B when it is at least as close to B's centroid as B's q-th percentile member. Candidates at q = 25 / 50 / 75: 14.2% / 3.9% / 0.8% of items. Of 40 sampled q=25 candidates the stand-in accepted 15 (38%), which projects to about **5% of items with two homes**.

The two sources differ by a factor of five, so how generous to be is a user choice (open question 2). Listings mark secondary entries, e.g. `LinkedIn (also in: Social)`.

## 8. Cost

Prices: `grok-4.7` $2.00 input / $0.50 cached input / $6.00 output per 1M tokens, reasoning billed as output, Batch API 20% off (https://docs.x.ai/developers/pricing); Jev $0.042 per 1M input tokens, output free; OpenAI `text-embedding-3-small` $0.02 per 1M tokens.

**One full run on 1,322 items, from measured token counts:**

| Stage | Measured / arithmetic | Cost |
| --- | --- | --- |
| Describe (Grok) | 22,065 uncached in × $2 + 31,488 cached × $0.50 + 69,041 out × $6, per 1M | $0.474 |
| Fetch + re-describe (Grok) | 246 unsure items with a domain × 23/40 usable pages ≈ 141 items; measured $0.0119 per 23 items | ≈ $0.073 |
| Embeddings (OpenAI) | 13,873 tokens × $0.02 / 1M | $0.0003 |
| Propose top branches (Grok) | 1 call ≈ 3,300 in + 3,600 out | ≈ $0.027 |
| Label 99 second-level nodes (Grok) | 4 calls × $0.0108 (measured mean per 25-node call) | ≈ $0.043 |
| **Grok + embeddings subtotal** | | **≈ $0.62** |
| Routing (Jev) | 1,322 requests, ≈ 55k tokens | ≈ $0.002 |
| Filing into a branch (Jev) | 1,322 requests × ≈ 515 tokens (item + 11 branch names and descriptions) ≈ 681k tokens | ≈ $0.029 |
| Second-node checks (Jev) | 188 requests, ≈ 11k tokens | ≈ $0.0005 |
| Merge/split review (Jev) | 99 requests, ≈ 19k tokens | ≈ $0.001 |
| **Jev subtotal (estimated, not measured)** | ≈ 766k input tokens × $0.042 / 1M | **≈ $0.03** |
| **Total with Jev** | | **≈ $0.65** |

- **Same decisions on the Grok stand-in:** filing $0.476 (measured, 14 calls, 61k reasoning tokens) + second-node checks 228 pairs ÷ 40 × $0.017 ≈ $0.097 + review 99 ÷ 20 × $0.0154 ≈ $0.076 = **≈ $0.65**, for a run total of **≈ $1.27**. Jev makes the decision layer about 20× cheaper and halves the run. Gating filing by embedding margin (ask only the lower half) halves either figure.
- **Batch API:** Grok subtotal × 0.8 ≈ $0.49, total ≈ $0.52 with Jev.
- **v1 estimate was $0.63** without any decision layer; the describe stage came in at $0.47 instead of $0.40 because half of its output tokens were reasoning even at `low`.
- Re-runs are free: every description, embedding, label and decision is cached on disk.

**Actual experiment spend: $2.13 of the $3.00 budget** (`~/acct_cleanup/experiments/spend_ledger.json`): describe $0.474, fetch test $0.012, embeddings $0.0003, LLM taxonomies $0.285, node labels for ten approaches $0.371, judge $0.477, decision stand-in $0.508.

## 9. Evaluation without a gold set

- **Eyeball first:** an indented tree printout (top branch > node > items as `name (domain) — description`, long nodes truncated) written on every run. This is the primary acceptance check.
- **Size report:** node counts, min/median/max per level, Gini, share of items in out-of-band nodes, unclustered share. Out-of-band share must be 0.
- **LLM-judge spot checks:** coherence 1–5 and intruder detection on sampled nodes (30+ nodes for a release check, since 15 is noisy), plus one whole-tree rating. Use a different judge model from the describer when one is available.
- **Internal metrics, used with care:** cosine silhouette and intra-node cosine compare cuts of the same algorithm only; they do not rank HDBSCAN (noise excluded) and cannot pick the number of nodes.
- **Decision checks:** agreement between embedding filing and the decision model by margin quartile; accept rate of second-membership candidates; Jev versus the Grok stand-in on a 100-item sample once a key exists.
- **Stability:** adjusted Rand index across embedding backends and across re-runs.
- **Operational:** unsure rate, fetch success rate, actual tokens against §8.

## 10. Config, CLI, privacy, tests

- **Config:** `DescriptionConfig` (as v1); `EmbeddingConfig.backend = openai|local|fireworks|litellm|fake`; `ClusteringConfig.algorithm = kmeans|agglomerative|hdbscan`; new `TreeConfig {top_branches=(8,14), top_band=(0.03,0.15), leaf_band=(4,40), multi=off|conservative|liberal, multi_quantile=25}`; `DecisionConfig {backend=off|grok|typesafe|openrouter|cloudflare|hf, model, min_confidence}`; `LabelingConfig` gains `reasoning_effort`.
- **CLI:** `constella organize INPUT... --output-dir DIR [--describe-model xai/grok-4.7] [--reasoning-effort low] [--batch] [--embedding openai|local] [--decision grok|typesafe|openrouter|cloudflare|hf|off] [--multi off|conservative|liberal] [--fetch on|off] [--cache-dir DIR] [--dry-run]`.
- **From `account-clustering`:** keep ingest, privacy scrub, label cache, fake embeddings, export, CLI entry. Drop `rules.py`, the fixed taxonomy, and **all alias / product-split / tenant tables** (decision 6). Caller metadata stays out of the algorithm.
- **Privacy:** chat and decision models receive the display name, the registrable domain, and on fallback the public homepage title and meta description. Titles containing an email address or phone number are replaced by the domain. Embedding models receive only the generated description text. Usernames, addresses, passwords, URL paths and query strings never leave the machine and are never placed in a `ContentUnit`.
- **Tests (offline, fakes for chat, decision and embedding):** v1 list plus: balancing loop keeps every item and ends inside the bands; one primary membership per item; secondary memberships do not change sizes; `decide()` fake backend with low confidence falls back; "none of these fit" lands in Unclustered; filing gate asks the decision model only below the margin threshold.

## 11. Milestones

0. **Approval of this v2** and answers to the questions below. Nothing is built before this.
1. Core: `algorithm` switch (K-Means default kept), Ward tree + balancing loop, `Node` / `Membership` models, tree printout. Offline tests.
2. Description stage: batching, cache, homepage fallback, `--batch`.
3. `decide()` interface with `grok` and `off` backends; filing gate; multi-membership.
4. Jev backends (`typesafe`, `openrouter`; `cloudflare` and `hf` after) and a Jev-versus-stand-in check on 100 items.
5. `constella.records` adapter and `constella organize` CLI.
6. Real run on the full list (≈ $0.65 with Jev, ≈ $1.27 without), user review of the printed tree, README update.

## 12. Open questions for the user

1. **Jev access.** No TypeSafe or OpenRouter key is on this machine, so Jev was never called. Which do you want to provide?
   A) `OPENROUTER_API_KEY` (no waitlist, `typesafe/jev-1.13`) — **recommended to start**;
   B) `TYPESAFE_API_KEY` (direct, `jev-1.14.0`, early-access waitlist);
   C) Cloudflare Workers AI credentials;
   D) none for now: keep the Grok stand-in (about $0.65 more per full run).
2. **How generous should multi-membership be?**
   A) conservative: similarity candidates confirmed by the decision model, about 5% of items get a second home — **recommended**;
   B) liberal: also accept the second branch the filing step names, about 25% of items;
   C) off.
3. **Size bands.** Defaults used: 8–14 top branches, each 3–15% of items; second-level nodes of 4–40 items (the run produced 11 branches and 99 nodes, median 11 items). Keep, or do you want fewer and larger nodes (for example 10–60) or a third level instead?
4. **Unrecognised items.** 20% of descriptions were flagged unsure and the homepage fetch fixed 43% of a sample. For the rest:
   A) send them straight to Unclustered — **recommended** (in the experiment they formed junk nodes such as "unidentified startups");
   B) cluster them anyway on the guessed description.
5. **Review point for the top level.** Should the run pause after Grok proposes the 8–14 branches so you can rename, add or remove branches before items are filed?
   A) optional `--review-taxonomy` flag, off by default — **recommended**;
   B) always pause;
   C) never.
6. **Embedding default.** A) OpenAI `text-embedding-3-small` ($0.0003 per run, the backend the results above were measured on) — **recommended**; B) local MiniLM ($0, descriptions stay on the machine, trees differ and were not judged); C) ask xAI support to enable embeddings for the team and re-probe.
7. **Region-themed branches.** Embedding-only trees grouped services by country; the chosen approach organises by function with regional sub-folders inside. A) by function — **recommended**; B) allow a regional top branch.
