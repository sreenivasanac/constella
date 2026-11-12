# AI-First Clustering — Design Plan (v3.1)

Status: **design only, awaiting approval.** No code changes accompany this document. v3 replaced folders with a **multi-label taxonomy**, size bands with a **data-driven split/merge test**, and the Grok stand-in with **real Jev**. v3.1 adds an **MCP server**, an **iterative LLM ↔ Jev loop** that shrinks the unclustered set, and a test of a **constellation image tool**. All numbers come from runs on the real data (2026-11-11); scripts and outputs are outside the repo in `~/acct_cleanup/experiments/` because they contain personal service lists. The v2 text and its ten-approach comparison are in git at `156c176`. Base commit for the work: `a905c64` (`v0.1.0`).

## 1. Decisions

| # | Decision |
| --- | --- |
| 1 | **Hybrid, not LLM-only** (§2): embeddings locate and sample, an LLM writes descriptions and labels, Jev makes every per-item decision. |
| 2 | **Constella is an MCP server** with seven tools (§4). The `constella organize` CLI drives the same tools with Grok as the built-in agent. |
| 3 | **Labels, not folders.** One primary label and any number of extra labels per item; labels form a DAG. Target: 5–15% of items with more than one label. |
| 4 | **Jev via TypeSafe direct** (`jev-1.14.0`). Every scoring request offers "Uncategorized / Miscellaneous". A label is confident at p ≥ **0.8** (§5). |
| 5 | **Iterative loop:** items below the cutoff go back to the LLM as clusters with counts and samples, the LLM edits the labels, Jev re-scores; stop when an iteration moves under 10% of the pending items (§5). |
| 6 | **No size bands.** A label is split only when its members show sub-structure that beats a null and is stable (§6). |
| 7 | Items still unrecognised after one homepage fetch are **Unclustered** and do not shape the labels. Unclustered is an item status, not a label. |
| 8 | The run **always pauses** after the top-level proposal for user review (§4.3). |
| 9 | **Region is never a topic**, only an optional facet label that is never primary. |
| 10 | Unchanged: `grok-4.7` at `reasoning_effort=low`; OpenAI `text-embedding-3-small`; no alias or sibling-domain tables; no gold set; thin `constella.records` adapter; only names, registrable domains and generated descriptions leave the machine. |

## 2. Hybrid or LLM-only

**Chosen: hybrid.** The evidence:

| Question | Measurement |
| --- | --- |
| Can embeddings decide alone? | No. The nearest top label by embedding equals Jev's best label for 69% of items. |
| Can embeddings shortlist? | Yes. Jev's best top label is among the 6 nearest for 93.5% of items and the 8 nearest for 96.6%; of all labels Jev rates ≥ 0.8, 98.8% are among the 8 nearest. Asking 8 instead of all 19 cuts filing from 2,576 to about 1,050 tokens per item. |
| Can an LLM make the per-item decisions? | It can, at 4× the price: Grok filed 1,322 items for $0.476; Jev for $0.126, with 87.8% identical answers (§7). |
| Can an LLM build every level? | In v2 the LLM-only two-level taxonomy had the lowest coherence (3.60 against 4.13) and the lowest silhouette of ten approaches. |
| Is an LLM proposal reproducible? | No. The same prompt run twice gave 19 and 14 top labels, and the two item partitions agree at an adjusted Rand index of only 0.45. Hence the pause, and a deterministic test for the lower levels. |
| What does the LLM need to read? | An overview of 68 clusters with counts and 2–6 samples each is 3,600 tokens for 1,161 items, and grows with √N. All descriptions would be about 16,000 tokens and grow with N. |

Embeddings cost $0.0003 per run. They are never the judge, only the index: they build the overview, shortlist candidate labels, drive the split test and draw the map.

## 3. Data model and outputs

```text
Item        {id, name, domain, description,
             status: labeled | tentative | unclustered_no_fit | unclustered_unrecognised}
Label       {id, name, description, parents: [label_id], facet: topic | region,
             origin: proposed | split | loop | user, centroid}
Assignment  {item_id, label_id, p, source: jev_noul | jev_choice | embed | user, is_primary}
```

- The label graph is acyclic. Ancestors are implied and never stored as assignments.
- An item has at most one primary assignment, always a topic label. Facet labels are never primary and do not count as multi-label.
- `labeled`: best label p ≥ cutoff. `tentative`: best p between 0.5 and the cutoff; the label is kept, flagged, and the item returns to the loop. `unclustered_no_fit`: Jev chose Miscellaneous or nothing reached 0.5. `unclustered_unrecognised`: description still unsure after the homepage fetch.
- There is no Miscellaneous label in the taxonomy. It exists only as an answer option, and choosing it sets a status.

Files in the run directory: `labels_proposed.yaml` (review artifact), `labels.yaml` (the DAG with counts), `assignments.csv` (item × label × p), `taxonomy.md` (indented tree, counts, 5–8 samples per label, multi-parent labels printed under each parent with `(also under: …)`), `items.csv` (item → status, primary, other labels, facets), `constellation.png` (optional), `manifest.json` (config, models, stage status, tokens and cost).

## 4. MCP server

### 4.1 Tools

All state lives in the run directory; tools take a `run_id` and return counts and samples, never the full item list.

| Tool | Calls | Input (brief) | Output (brief) | Response budget | Cost |
| --- | --- | --- | --- | --- | --- |
| `ingest` | none | `{records_path \| texts[], format}` | `{run_id, items, deduped}` | ≤ 150 tokens | $0 |
| `describe` | Grok, HTTP, OpenAI | `{run_id, scope: new \| unsure, fetch: bool}` | `{described, unsure, unrecognised, cost_usd}` | ≤ 150 tokens | $0.36 per 1,000 items; fetch pass $0.06 per 1,000 |
| `overview` | none | `{run_id, scope: all \| pending \| label:<id>, max_groups, samples}` | `{totals: {labeled, tentative, no_fit, unrecognised}, groups: [{id, name?, parents, n, n_pending, nearest_labels[], samples[], split?: {k, stability}, merge_with?: id}]}` | default ≤ 4,000 tokens, hard cap 8,000 | $0 |
| `render` | none | `{run_id, scope}` | PNG ≤ 1000×800: 2-D map, cluster ids and counts, no item names | about 800 image tokens | $0 |
| `apply_labels` | none | `{run_id, ops: [{op: add \| rename \| redefine \| merge \| delete \| set_parents \| split, …}]}` | `{applied, rejected: [{op, reason}], changed_labels[], needs_approval}` | ≤ 300 tokens | $0 |
| `score` | Jev | `{run_id, scope: changed \| pending \| all, cutoff: 0.8, candidates: 8}` | `{scored, labeled, tentative, no_fit, moved, by_label: [{id, gained}], p_histogram[10], converged, cost_usd}` | ≤ 400 tokens | $0.04–0.11 per 1,000 items |
| `export` | none | `{run_id}` | `{paths, summary}` | ≤ 200 tokens | $0 |

- `overview` is the "tree with counts". Before labels exist its groups are embedding clusters (2·√n of them); afterwards they are labels, and `scope: pending` clusters only the items below the cutoff. Each group carries its size, the nearest existing labels and a few samples (`name — description`). Groups that pass the split test (§6) carry `split`, and label pairs that pass the merge test carry `merge_with`, so the agent sees data-backed suggestions and does not have to guess.
- `apply_labels` is how any agent edits the polyhierarchy. `split` accepts the server's proposed cut of a label; the agent then names the children with `rename` after reading `overview(scope: label:<id>)`. `set_parents` gives a label a second parent.
- `score` asks Jev one request per item: one `noul` per candidate label (the nearest by embedding plus every label changed since the last call) and one `choice` over the same candidates that always includes `uncategorized_miscellaneous`. Answers are cached by question text, so an unchanged (item, label) pair is never asked twice.
- "Get unclustered" is `overview(scope: pending)`; it is not a separate tool.

### 4.2 Agent loop

```text
ingest -> describe(fetch: true)
overview(all) [+ render]            agent proposes top-level labels
apply_labels(add ...)               -> needs_approval: true
-- PAUSE: user reviews labels_proposed.yaml --
score(all)
repeat:
    overview(pending)               counts + samples of what did not fit
    apply_labels(add | redefine | split | merge | set_parents)
    score(changed)                  only new or changed questions are asked
until score.converged
export
```

Per iteration the agent reads at most one overview (≤ 4,000 tokens) and one score summary (≤ 400 tokens) and writes a list of operations. Measured with Grok as the agent: 3,800–4,000 input and 1,400–3,600 output tokens per iteration, $0.014–0.028. Any MCP client (Claude, Grok, another agent) can run the loop alongside its own tools. `constella organize` runs the same loop with `grok-4.7`.

### 4.3 The pause

The first `apply_labels` that creates top-level labels writes `labels_proposed.yaml` and puts the run in `awaiting_approval`; `score` refuses to run until the user approves. Approval comes from the user, not the agent: through an MCP elicitation prompt where the client supports it, otherwise by editing the file and running `constella organize --resume RUN` (or calling `score` again once the file says `approved: true`).

```yaml
n_items: 1161
approved: false
labels:
  - id: banking_and_investing
    name: "Banking and investing"
    description: "Retail banks, credit cards, credit scores, neobanks, and cryptocurrency exchanges."
    facet: topic
    estimated_size: 33   # nearest-label count by embedding, before scoring
    sample_members: ["Bank of America (bankofamerica.com) — ...", ...]
```

The user may rename, edit a description, delete, add, or merge with `merge_into: <id>`. On resume everything already computed is reused (descriptions, pages, embeddings, cached Jev answers); only questions about new or edited labels are asked, and lower levels are rebuilt only for labels whose membership changed. New top-level labels proposed later by the loop are applied but listed for review in the same file (open question 4). In the experiment the proposal was accepted unchanged (approval simulated).

## 5. The LLM ↔ Jev loop

**Scoring rule.** An item is `labeled` when Jev's `choice` is not Miscellaneous and its best `noul` is ≥ the cutoff. Extra labels need p ≥ 0.85 (§8).

**Calibrating the cutoff without a gold set.** Three measurements on the current run, all repeatable on any data:

| Cutoff | Confident (of 1,161 recognised) | Pending | Audit of the band just below |
| --- | --- | --- | --- |
| 0.5 | 1,093 | 5.9% | – |
| 0.7 | 1,004 | 13.5% | p in [0.5, 0.7): 9 of 20 correct |
| **0.8** | **934** | **19.6%** | p in [0.7, 0.8): 12 of 20 correct |
| 0.9 | 766 | 34.0% | p in [0.8, 0.9): 17 of 20 correct |
| 0.95 | 555 | 52.2% | p in [0.9, 1.0]: 20 of 20 correct |

1. *Distribution:* the share of items below each cutoff (columns 2–3).
2. *Audit by band:* 20 items per probability band, judged strictly by `grok-4.7` with the probability hidden (column 4). Weighted by band size, labels accepted at 0.8 are about 97% correct, at 0.5 about 91%.
3. *Re-ask noise:* when the loop re-asked unchanged labels with slightly different wording, 51 items crossed 0.8. Probabilities within about ±0.05 of the cutoff are therefore noise, which is why unchanged questions are served from cache and why `tentative` items keep their label.

**0.8 is the default**: it is the lowest cutoff whose accepted band is judged at least 85% correct, and 0.9 would send a third of the items back for a 3-point gain in precision. The audit is 20 items per band, so each figure carries roughly ±20 points; `score` reports the histogram and the CLI can rerun the audit (`--audit`, about $0.04).

**Stopping rule.** Stop when an iteration moves fewer than 10% of the pending items to `labeled`, when the agent returns no operations, after 5 iterations, or at the budget limit.

**Measured: three iterations on the pending set at cutoff 0.8** (real Jev, Grok as the agent):

| Iteration | LLM operations | Pending before | Newly labeled | Of those, on new or rewritten labels | Pending after | LLM cost | Jev cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 new sub-labels, 1 definition rewritten | 227 | 75 | 26 | 152 | $0.022 | $0.023 |
| 2 | 3 new labels (2 top-level) | 152 | 19 | 17 | 133 | $0.014 | $0.011 |
| 3 | 1 new sub-label, 2 definitions rewritten | 133 | 11 (8%, stop) | 11 | 122 | $0.028 | $0.008 |

- Pending fell from **227 to 122** (19.6% → 10.5% of recognised items); labeled rose from 934 to 1,039. Loop cost: **$0.106** (LLM $0.064, Jev $0.042).
- Honest attribution: 54 of the 105 moves landed on labels the LLM added or rewrote. The other 51 are the re-ask noise described above and would not occur in the designed loop, which never re-asks an unchanged question. The conservative result is 227 → 173 (14.9%).
- New labels were sensible: connected-device clouds, rewards and cashback, non-health insurance, technology product companies, uptime and error monitoring, grocery delivery, course-creator platforms, health information.
- Audit of 20 loop placements: 17 correct, the same rate as the [0.8, 0.9) band.
- **Unrecognised items stay out.** As a test, the 161 unrecognised items were scored on their guessed descriptions: 79 reached 0.8, but only 13 of 20 audited were correct and 5 could not be judged. That is too weak to label them silently (open question 3).

## 6. Data-driven granularity

Two numbers derived from N act as soft priors only: **√N** (a hint in the top-level prompt: an area expected to hold fewer items is usually a sub-label) and **s = ½·log₂N** (a group smaller than s is not worth its own label; a label needs 2·s members before a split is tried). For N = 1,161 they are 34 and 5.

**Top level.** The LLM proposes it with N stated and no count requested. Without any hint Grok returned 41 fine-grained labels; with the √N hint, 19.

**Split test** (per label, on its primary members; reported by `overview`):

1. Cut a Ward tree of the members at every k from 2 to m/s and compute the silhouette.
2. Build a null: the same members with each principal-component column permuted independently (same variance spectrum, no joint structure), 30 replicates, same cuts.
3. One max-statistic test over all k: the best standardised gap (real − null) must exceed that of every null replicate (p ≤ 0.05). This fixes the v2 problem that raw silhouette always prefers the largest k, because the null rises with k too.
4. Choose the smallest k whose gap is within one null standard deviation of the best (the gap-statistic 1-SE rule) and whose partition is stable: mean adjusted Rand index ≥ 0.6 against re-clustering ten 80% subsamples.
5. Children under 3 items are not created; those items stay directly on the parent.
6. A Jev `noul` can veto a split, never cause one. Recurse until no cut passes or a child has fewer than 2·s members.

**Merge test.** Two labels are candidates when separation is below cohesion (mean cross-similarity ≥ the smaller mean within-similarity); a Jev `noul` must agree. A merge across parents yields one label with two parents.

**Measured** (Appendix B): 11 of 19 top labels were split (k from 4 to 19); 8 were left whole, 7 because no cut beat the null and 1 (an 82-item AI-tools label) because no cut was stable. No second-level split passed, so the taxonomy has two levels. 4 of 5,995 label pairs were merge candidates; Jev rejected all 4. Sizes: top labels 20–159 items (median 59, largest 14% of labeled items, Gini 0.29); sub-labels 3–15 items (median 7, Gini 0.23).

## 7. Jev, measured

**Probe.** `GET /v1/models` → 200, lists `jev-latest` and `jev-preview`. `jev-1.14.0` is not listed but accepted. `questions` must be an object keyed by question id (a list returns 422). The response has `answers` and `usage {input_tokens, output_tokens}`; no cost field, no rate-limit headers.

**Cost and latency.** 4,550 requests, 7.67M input tokens, **$0.321** in total. Per request: **p50 0.14–0.15 s, p95 0.23–0.25 s**, max 1.1 s. At 8 concurrent requests 1,161 items take **24 s**. **No 429/529, no retries, no failed requests.** Tokens scale with the number of questions (about 115 each for a 15-word state), so asking only candidate labels is what keeps cost down.

**Agreement with the v2 Grok stand-in** (identical questions; stand-in answers from the v2 cache):

| Question | n | Same answer | Notes |
| --- | --- | --- | --- |
| Branch filing, 11 v2 branches + "none" | 1,322 | **87.8%** (Cohen's κ 0.87) | by embedding-margin quartile 77 / 87 / 92 / 96%; Jev confidence < 0.5 → 49%, ≥ 0.9 → 99% |
| Second-membership yes/no | 40 | 82.5% | r = 0.82 |
| Merge two nodes | 10 | 90% | |
| Split a node | 10 | 100% | both said yes to all 10 |

- Jev abstains more often: "no folder fits" for 86 items against Grok's 30.
- Grok's free-form "second folder" was generous: Jev rates only 40% of Grok's 326 picks at ≥ 0.5. Independent nouls with a threshold give a tunable multi-label rate.
- The split yes-bias holds for both models; in v3 the Jev veto fired 0 times in 11. The data test does the selecting.
- Name bias: the label "Identity and security" attracted sign-in pages of unrelated services (a retailer's sign-in page got it as primary at p = 0.96). See open question 8.

## 8. Multi-label calibration

Every candidate label gets its own `noul` (TypeSafe's "one Noul per option" pattern), so an item can pass several. Extra labels are those with p ≥ τ that are not ancestors of a kept label. Region facets need p ≥ 0.8 and are stored separately.

| τ | Items with 2+ labels | % of labeled items | % of all items | % with labels under two top labels |
| --- | --- | --- | --- | --- |
| 0.50 | 535 | 47.4 | 40.5 | 30.5 |
| 0.60 | 397 | 35.2 | 30.0 | 20.6 |
| 0.70 | 263 | 23.3 | 19.9 | 12.4 |
| 0.75 | 207 | 18.4 | 15.7 | 9.0 |
| 0.80 | 164 | 14.5 | 12.4 | 7.2 |
| **0.85 (used)** | **110** | **9.8** | **8.3** | **5.1** |
| 0.90 | 60 | 5.3 | 4.5 | 3.0 |
| 0.95 | 24 | 2.1 | 1.8 | 1.3 |

The 5–15% target corresponds to τ between 0.80 and 0.90. Examples by probability band: `~/acct_cleanup/experiments/multi_label_examples.md`. Multi-parent labels are rare here: of 330 (sub-label, other top label) questions one reached 0.7 (used-car marketplaces, under Travel and also Shopping).

## 9. Does the image tool help?

**Test.** A 1000×800 PNG of the UMAP map (1,161 points coloured by 68 clusters, `id:count` at each cluster, no item names) was given to `grok-4.7` with vision together with the text overview, and compared with the text overview alone, run twice to measure run-to-run noise.

| Arm | Top labels | Input tokens | Cost | Agreement with text run A (ARI) | Blind judge, sum of three 1–5 scores |
| --- | --- | --- | --- | --- | --- |
| Text only, run A | 19 | 3,603 | $0.038 | – | 12 |
| Text only, run B | 14 | 3,603 | $0.036 | 0.45 | 10–11 |
| Text + image | 21 | 4,495 (802 for the image) | $0.047 | 0.50 | 13–14 |

- The judge preferred the image arm in all four comparisons (both orders, against both text runs).
- The evidence is weak. It is one image run. The two text-only runs differ from each other as much as either differs from the image run. The judge's reasons mostly praise finer separation, and the image arm also had the most labels.
- A second signal points the same way: labels proposed with the image group clusters that sit closer together on the map (spread ratio 0.30 against 0.35 and 0.47).
- The image adds about 800 input tokens ($0.002). Test cost: $0.19.

**Verdict:** keep `render` as an **optional** tool. It is cheap, it did no harm, it may help the proposal slightly, and the same PNG is useful to the person at the pause. The loop does not depend on it and text-only clients lose nothing measurable.

## 10. Results of the v3 run

1,322 entities; descriptions and embeddings reused from v2.

- **Homepage fetch for all 271 unsure items:** 162 pages usable, **110 became confident**, **161 (12.2%) unrecognised**.
- **Taxonomy:** 19 top labels, 110 sub-labels, 1 multi-parent label, 2 optional region facets; the loop added 8 labels (2 top-level). Top labels and counts: Appendix A.
- **At the v3 acceptance rule** (best p ≥ 0.5): 1,128 items labeled (85.3%), **14.7% Unclustered** (12.2% unrecognised + 2.5% no fit). Multi-label rate **9.8%** of labeled items.
- **At cutoff 0.8:** 934 recognised items labeled before the loop and 1,039 after it (89.5% of recognised, 78.6% of all); 122 remain pending, most of them `tentative` with a kept label.
- **Judge spot checks:** 15 labels, mean coherence 3.93 / 5, name fits for 10 of 15; 15 extra-label assignments: primary correct 15, extra correct 14.
- **Known errors:** the sign-in bias (§7); two junk-drawer sub-labels; the 82-item AI-tools label has no sub-labels.

Listing: `~/acct_cleanup/experiments/experiment_labels.md`.

## 11. Cost

Prices: `grok-4.7` $2.00 in / $0.50 cached in / $6.00 out per 1M tokens (reasoning billed as output); Jev $0.042 per 1M input tokens; OpenAI embeddings $0.02 per 1M tokens.

| Stage of one run on 1,322 items (measured) | Cost |
| --- | --- |
| Describe (Grok, 27 calls) | $0.474 |
| Fetch + re-describe 162 pages (Grok, 5 calls) | $0.077 |
| Embeddings | $0.0003 |
| Propose top level (Grok, 1 call) | $0.053 |
| Name 110 sub-labels (Grok, 5 calls) | $0.089 |
| Score against top labels (Jev) | $0.126 |
| Score against sub-labels (Jev) | $0.046 |
| Split veto, merge check, extra parents (Jev) | $0.003 |
| Loop, 3 iterations (Grok $0.064, Jev $0.042) | $0.106 |
| **Total** | **≈ $0.97** |

Describing is half the run and is paid once. Shortlisting 8 top labels by embedding would cut top-label scoring from $0.126 to about $0.05. When another agent drives the MCP tools, the proposal, naming and loop-LLM rows ($0.21) move to that agent's own token bill.

**Actual v3 experiment spend: $0.94 of the $3.00 budget** (`spend_ledger_v3.json`): Grok $0.622 (re-describe $0.077, top-level proposals $0.109, naming $0.089, judge and audits $0.094, loop $0.064, image test $0.190), Jev $0.321 (run $0.175, agreement study $0.103, loop $0.042), embeddings < $0.001.

## 12. Evaluation without a gold set

- **Read the tree:** `taxonomy.md` and `items.csv`; the pause is the first checkpoint.
- **Every run reports:** labels per level, size distribution, the split-test table, merge candidates, the p histogram, pending share per iteration, multi-label rate, unclustered share by reason.
- **Cutoff audit:** 20 items per probability band judged with p hidden (§5), ideally by a model other than the describer.
- **Review list:** `tentative` items and extra labels within 0.05 of τ.
- **Spot checks:** label coherence and name fit, extra-label correctness; 30+ samples for a release.
- **Stability:** rerun the split test with another seed; rerun the top-level proposal and compare (ARI 0.45 between two runs today, which is why the user approves it).
- **Operational:** unsure rate, fetch success, Jev errors and retries, tokens against §11.

## 13. Config, privacy, tests

- **Config:** `DecisionConfig {backend = typesafe|grok|off, model = jev-1.14.0, concurrency = 8, max_retries = 6}`; `TaxonomyConfig {cutoff = 0.8, extra_threshold = 0.85, candidates = 8, split_alpha = 0.05, split_stability = 0.6, null_replicates = 30, parent_threshold = 0.7, loop_min_move = 0.10, loop_max_iterations = 5, facets = []}`; description and embedding config as before.
- **CLI:** `constella organize INPUT... --output-dir RUN [--embedding openai|local] [--decision typesafe|grok|off] [--cutoff 0.8] [--extra-threshold 0.85] [--facets region] [--audit]`, `constella organize --resume RUN`, `constella mcp` (stdio server).
- **Jev client:** plain HTTPS, questions as an object, exponential backoff with jitter on 429/529/5xx, no retry on other 4xx, every response cached on disk, key-shaped strings redacted from stored errors.
- **Privacy:** models and MCP clients receive only display name, registrable domain, generated description, and on fallback the public homepage title and meta description (emails and phone numbers stripped). Tool responses follow the same rule. Usernames, addresses, URL paths and query strings never leave the machine.
- **Tests (offline, fakes for chat, decision and embedding):** split test rejects structureless data and accepts planted clusters; no item lost by split or merge; ancestors never stored; one primary per item, never a facet; Miscellaneous sets a status and never creates a label; `score` refuses while `awaiting_approval`; `score(changed)` asks only new questions; the loop stops under the 10% rule; tool responses respect their token caps; resume re-asks only edited labels; backoff retries 429/529 and stops on 401/422.

## 14. Milestones

0. **Approval of this plan** and answers to §15. Nothing is built before this.
1. Core models, DAG utilities, split and merge tests, `overview`, exports. Offline tests.
2. `describe` with batching, cache and homepage fallback.
3. `decide()` with `typesafe`, `grok` and `off` backends; `score` with the Miscellaneous option and the cutoff.
4. `apply_labels`, the pause, `--resume`.
5. MCP server (`constella mcp`) exposing the seven tools; `constella organize` as the built-in loop.
6. `constella.records` adapter; real run on the full list (≈ $0.97); user review; README update.

## 15. Open questions for the user

1. **Confidence cutoff.**
   A) 0.8: accepted labels about 97% correct in the audit, 19.6% of recognised items pending before the loop and 10.5% after — **recommended**;
   B) 0.9: 34% pending before the loop, for about 3 more points of precision;
   C) 0.7: 13.5% pending, but the [0.7, 0.8) band was only 60% correct.
2. **Items between 0.5 and the cutoff.**
   A) keep their best label as `tentative`, flagged in the outputs — **recommended**;
   B) show them as Unclustered until they pass the cutoff.
3. **Unrecognised items.** Scored on guessed descriptions, 79 of 161 reached 0.8 but only about 65% of an audited sample were right.
   A) keep them Unclustered — **recommended**;
   B) give them `tentative` labels marked "guessed";
   C) add a web-search step for them before the loop (new cost and a new data flow).
4. **New top-level labels proposed inside the loop** (two appeared: technology product companies, non-health insurance).
   A) apply them and list them for review in `labels_proposed.yaml` — **recommended**;
   B) pause again for approval each time;
   C) allow only sub-labels in the loop.
5. **Who proposes labels in MCP mode.**
   A) the calling agent, with `constella organize` using Grok for the same role — **recommended**;
   B) always server-side Grok, so results do not depend on the client model;
   C) server asks the client model through MCP sampling.
6. **Image tool.** The test was mildly positive but not conclusive.
   A) keep `render` as an optional tool — **recommended**;
   B) drop it from the first release;
   C) run a larger test first (about $0.50 for five runs per arm).
7. **Extra-label threshold.**
   A) τ = 0.85, 9.8% of labeled items — **recommended**;
   B) τ = 0.80, 14.5%;
   C) τ = 0.90, 5.3%.
8. **Login and sign-in pages** took "Identity and security" as primary.
   A) tell the describer to describe the service behind the domain, and narrow that label to identity providers and password tools — **recommended**;
   B) keep the primary and rely on the extra label;
   C) add an `access` facet for login portals.
9. **Large labels with no stable sub-structure** (AI tools 82 items, Identity 48, Jobs 47).
   A) leave it to the loop: the agent may add sub-labels there when pending items cluster — **recommended**;
   B) always ask the LLM for sub-labels of any label above √N items that fails the split test;
   C) lower the stability bar from 0.6 to 0.5.
10. **Sub-label granularity** (110 sub-labels, median 7 items).
    A) keep the rule — **recommended**;
    B) coarser: first cut that beats the null, then recurse (fewer, larger sub-labels, a third level, lower stability);
    C) raise the minimum useful size from ½·log₂N (5) to log₂N (10).
11. **Region facet** (India 168 items, United States 176 at p ≥ 0.8).
    A) off by default, `--facets region` — **recommended**;
    B) on by default;
    C) drop facets from the first release.
12. **Multi-parent labels.** One label reached 0.7 for a second parent.
    A) keep 0.7 and let the agent add parents with `set_parents` — **recommended**;
    B) lower to 0.5 (six labels, two of them wrong).

## Appendix A — Top labels of the v3 run

Items carrying the label or any descendant (primary or extra), at the v3 acceptance rule, before the loop.

| Top label | Items | Directly on the label | Sub-labels |
| --- | --- | --- | --- |
| Productivity and business software | 159 | 25 | 18 |
| Education and learning | 96 | 9 | 12 |
| Social and communications | 93 | 18 | 11 |
| Shopping and retail | 91 | 10 | 14 |
| Travel and transportation | 90 | 8 | 12 |
| Media and entertainment | 83 | 10 | 11 |
| AI tools and platforms | 82 | 82 | 0 |
| Banking and investing | 61 | 6 | 9 |
| Government and civic services | 59 | 15 | 5 |
| Developer tools and hosting | 59 | 5 | 9 |
| Health care | 52 | 11 | 7 |
| Identity and security | 48 | 48 | 0 |
| Jobs and careers | 47 | 47 | 0 |
| Payments and transfers | 37 | 37 | 0 |
| Fitness and wellness | 35 | 35 | 0 |
| Dating and matchmaking | 31 | 31 | 0 |
| Food delivery | 26 | 2 | 3 |
| Home and local services | 24 | 24 | 0 |
| Startups and crowdfunding | 20 | 20 | 0 |

Example sub-labels: under Banking and investing — bank account portals, cryptocurrency exchanges, consumer loan lenders, securities brokerages, credit score services; under Travel and transportation — airline ticketing, hotels and rentals, frequent flyer programs, ridesharing and carpools, used car marketplaces (also under: Shopping and retail).

## Appendix B — Split test per top label

m = primary members; z = standardised gap at the chosen k; stability = mean adjusted Rand index over ten 80% subsamples; Jev = veto `noul` (≥ 0.5 allows the split).

| Top label | m | Result | k | z | Stability | Jev |
| --- | --- | --- | --- | --- | --- | --- |
| Productivity and business software | 131 | split | 19 | 13.2 | 0.62 | 0.91 |
| Travel and transportation | 89 | split | 15 | 17.0 | 0.72 | 0.88 |
| Education and learning | 89 | split | 13 | 4.9 | 0.64 | 0.84 |
| Shopping and retail | 81 | split | 15 | 12.6 | 0.62 | 0.82 |
| Media and entertainment | 77 | split | 11 | 7.2 | 0.71 | 0.93 |
| AI tools and platforms | 76 | not split: beats the null (p = 0.03), no stable cut | – | 4.3 max | < 0.6 | – |
| Social and communications | 74 | split | 11 | 5.1 | 0.71 | 0.90 |
| Developer tools and hosting | 67 | split | 10 | 9.0 | 0.60 | 0.91 |
| Banking and investing | 62 | split | 10 | 7.8 | 0.68 | 0.92 |
| Government and civic services | 53 | split | 5 | 5.3 | 0.77 | 0.90 |
| Health care | 52 | split | 10 | 7.0 | 0.62 | 0.88 |
| Jobs and careers | 52 | not split (p = 0.39) | – | 1.4 max | – | – |
| Fitness and wellness | 39 | not split (p = 0.16) | – | 2.0 max | – | – |
| Payments and transfers | 38 | not split (p = 0.23) | – | 1.5 max | – | – |
| Identity and security | 37 | not split (p = 0.36) | – | 1.3 max | – | – |
| Dating and matchmaking | 32 | not split (p = 0.45) | – | 1.0 max | – | – |
| Home and local services | 31 | not split (p = 0.16) | – | 2.2 max | – | – |
| Food delivery | 27 | split | 4 | 2.9 | 0.84 | 0.89 |
| Startups and crowdfunding | 25 | not split (p = 0.94) | – | −0.6 max | – | – |

Clusters under 3 items do not become labels, so the number of sub-labels can be lower than k. All 22 sub-labels large enough to test (≥ 10 members) failed the null test, which is why there is no third level.

## Appendix C — Sources

- Jev: https://docs.typesafe.ai/introduction/quickstart, https://docs.typesafe.ai/models, https://docs.typesafe.ai/api/errors.md, https://api.typesafe.ai/openapi.json; hierarchical classification cookbook: https://docs.typesafe.ai/cookbooks/hierarchical_classification.md (notes gathered 2026-11-11).
- Grok pricing: https://docs.x.ai/developers/pricing.
- Experiment files in `~/acct_cleanup/experiments/`: `experiment_labels.md`, `multi_label_examples.md`, `jev_report.md`, `loop_v3.json`, `image_test_v3.json`, `agreement_v3.json`, `judge_v3.json`, `spend_ledger_v3.json`, `run_v3/` (`labels_proposed.yaml`, `labels_final.json`, `assignments.csv`, `items.csv`, `summary.json`, `build_report.json`, `loop_placements.json`, `constellation.png`), scripts `10_`–`19_`.
