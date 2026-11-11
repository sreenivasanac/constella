# AI-First Clustering — Design Plan (v3)

Status: **design only, awaiting approval.** No code changes accompany this document. v3 turns the v2 open questions into decisions, replaces folders with a **multi-label taxonomy**, replaces size bands with a **data-driven split/merge test**, and reports a rerun on the real data with **real Jev calls** (2026-11-11). Scripts and outputs are outside the repo in `~/acct_cleanup/experiments/` (they contain personal service lists). The v2 text and its ten-approach comparison are in git at `156c176`. Base commit for the work is unchanged: `a905c64` (`v0.1.0`).

## 1. Decisions

| # | Decision |
| --- | --- |
| 1 | **Jev via TypeSafe direct** (`TYPESAFE_API_KEY`, `jev-1.14.0`). Grok writes text; Jev answers every closed question. `decide()` stays backend-pluggable; the Grok stand-in remains as fallback. |
| 2 | **Labels, not folders.** An item carries one primary label and any number of extra labels. Labels form a DAG (a label may have several parents). Target: 5–15% of items with more than one label. |
| 3 | **No size bands.** Granularity comes from the data: a label is split only when its members show sub-structure that beats a null and is stable (§5). |
| 4 | Items still unrecognised after one homepage fetch go **straight to Unclustered**. Unclustered is an item status, not a label. |
| 5 | The run **always pauses** after the top-level proposal; the user edits a YAML file and resumes (§4). |
| 6 | **Embeddings:** OpenAI `text-embedding-3-small` by default; local MiniLM stays as `--embedding local`. |
| 7 | **Region is never a topic.** It exists only as an optional facet label that is never primary. |
| 8 | Unchanged from v2: `grok-4.7` at `reasoning_effort=low`; no alias or sibling-domain tables (dedup by identical registrable domain only); no gold set; records ingest stays a thin `constella.records` adapter; only names, registrable domains and generated descriptions leave the machine. |

## 2. Data model and outputs

```text
Item        {id, name, domain, description, status: labeled | unclustered_unrecognised | unclustered_no_fit}
Label       {id, name, description, parents: [label_id], facet: topic | region, origin: grok | split | user, centroid}
Assignment  {item_id, label_id, p, source: jev_noul | jev_choice | embed | user, is_primary}
```

Rules: the label graph is acyclic; ancestors are implied and never stored as assignments; an item has at most one primary assignment and it is always a topic label; facet labels are never primary and do not count as "multi-label"; an item may sit directly on a parent label when no child fits.

Files written to the run directory:

| File | Content |
| --- | --- |
| `labels_proposed.yaml` | review artifact for the pause (§4) |
| `labels.yaml` | the DAG: id, name, description, parents, facet, origin, counts (direct, primary, incl. descendants) |
| `assignments.csv` | one row per item × label: `item_id, name, domain, label_id, p, source, is_primary` |
| `taxonomy.md` | indented tree with counts and 5–8 sample members per label; a multi-parent label is printed under each parent with `(also under: …)` |
| `items.csv` | item view: description, status, primary label, other labels with p, facets |
| `manifest.json` | config, model ids, stage status, token counts and cost per stage |

## 3. Algorithm

```text
 1. ingest + dedup        URL -> registrable domain; identical domain -> one item           [deterministic]
 2. describe              name + domain -> 3-10 word description, confidence, unsure        [Grok, 50/call]
 3. fetch + re-describe   unsure items only: GET homepage, title + meta description         [HTTP, Grok 40/call]
 4. unclustered #1        still unsure -> status unclustered_unrecognised, leaves the run
 5. embed                 "name — description"                                              [OpenAI]
 6. propose top level     size-stratified sample, N stated, no count asked                  [Grok, 1 call]
 7. PAUSE                 write labels_proposed.yaml, exit                                  [user edits]
 8. resume: file items    one Jev request per item, one noul per top label                  [Jev]
 9. build lower levels    recursive split test per label; merge test across labels          [deterministic + Jev veto]
10. name labels           6-7 representatives per new label, siblings in one call           [Grok]
11. extra parents         noul per (label, nearby top label) -> polyhierarchy               [Jev]
12. assign labels         top-8 candidate sub-labels by embedding, one noul each            [Jev, 1 request/item]
13. primary / extras      thresholds on p (§6); nothing reaches 0.5 -> unclustered_no_fit
14. export                files of §2
```

Step 6 input: items are micro-clustered (k = 2·√N) only to build the sample; Grok sees each micro-cluster's size and 2–6 typical descriptions, never the full list. Plain text corpora skip steps 1–4. With `--decision off` step 8 uses nearest label by embedding (`source: embed`) and steps 11–13 use similarity quantiles, so an offline path remains.

## 4. The pause: review artifact and resume

`constella organize INPUT... --output-dir RUN` runs steps 1–6, writes `RUN/labels_proposed.yaml`, prints the path and exits. There is no CLI flag to skip the pause; the Python API takes the approved label list as an argument, which is what tests use.

```yaml
n_items: 1161
approved: false          # set to true, or just run --resume
labels:
  - id: banking_and_investing
    name: "Banking and investing"
    description: "Retail banks, credit cards, credit scores, neobanks, and cryptocurrency exchanges."
    facet: topic
    estimated_size: 33   # nearest-label count by embedding, before filing
    sample_members: ["Bank of America (bankofamerica.com) — ...", ...]
```

The user may rename a label, edit a description, delete a label, add one (`id`, `name`, `description`), or merge with `merge_into: <id>`. `estimated_size` and `sample_members` are informational and ignored on read.

`constella organize --resume RUN` validates the file (unique ids, non-empty descriptions, merge targets exist) and continues at step 8.

- **Reused from cache:** entities, descriptions, fetched pages, embeddings, and every Jev answer whose question text is unchanged. The cache key is the hash of (model, state, question), and each top label is its own `noul`, so editing or adding one label costs one new question per item, not a re-filing.
- **Recomputed:** nouls for new or edited labels, the lower levels of any label whose membership changed, their names, extra parents, and sub-label assignment.
- `--resume` can be run again after editing `labels.yaml` (the full DAG); only the affected questions are asked again.

In the experiment the proposal was accepted unchanged (approval simulated).

## 5. Data-driven granularity (replaces size bands)

Two numbers are derived from N and used only as soft priors: **√N** (a hint in the top-level prompt: an area expected to hold fewer items is usually a sub-label) and **s = ½·log₂N** (a group smaller than s is not worth its own label; a label needs 2·s members before a split is tried). For N = 1,161 these are 34 and 5.

**Top level.** Grok proposes it with N stated and no count requested. Finding: with no size hint at all Grok returned 41 fine-grained labels (median estimated share 2%); with the √N hint it returned 19. The hint is advisory and the user edits the result at the pause.

**Split test** (per label, on its primary members):

1. Cut a Ward tree of the members at every k from 2 to m/s and compute the silhouette.
2. Build a null: the same members with each principal-component column permuted independently (same variance spectrum, no joint structure), 30 replicates, same cuts.
3. One max-statistic test over all k: the split is considered only if the best standardised gap (real − null) exceeds that of every null replicate (p ≤ 0.05). This corrects the v2 problem that raw silhouette always prefers the largest k, because the null rises with k too.
4. Choose the smallest k whose gap is within one null standard deviation of the best gap (the gap-statistic 1-SE rule) **and** whose partition is stable: mean adjusted Rand index ≥ 0.6 against re-clustering ten 80% subsamples.
5. Children smaller than 3 items are not created; those items stay directly on the parent.
6. A Jev `noul` ("are these sub-groups distinctly different kinds of services?") can veto. It can never cause a split.
7. Recurse into each child. Recursion ends when no cut passes or the child has fewer than 2·s members.

**Merge test.** Two labels are candidates when their separation is below their cohesion: mean cross-similarity ≥ the smaller of the two mean within-similarities. A Jev `noul` must agree. A merge across parents yields one label with two parents.

**Measured** (details in Appendix B): 11 of 19 top labels were split (k from 4 to 19); 8 were left whole, 7 because no cut beat the null and 1 (an 82-item AI-tools label) because cuts beat the null but none was stable. No second-level split passed, so the taxonomy has two levels. Only 4 of 5,995 label pairs were merge candidates and Jev rejected all 4. Resulting sizes: top labels 20–159 items (median 59, largest 14% of labeled items, Gini 0.29); sub-labels 3–15 items (median 7, Gini 0.23); 97 items sit directly on a top label that has children.

## 6. Multi-label method and calibration

Every (item, label) pair that matters gets an independent Jev `noul`: all top labels at filing, then the 8 nearest sub-labels anywhere in the DAG (TypeSafe's "one Noul per option" pattern, since a `choice` always picks exactly one).

- **Primary:** the highest-p label with p ≥ 0.5 after removing implied ancestors. If no noul reaches 0.5, the `choice` answer is used when its confidence is ≥ 0.5 (35 items); otherwise the item is `unclustered_no_fit`.
- **Extra labels:** every other label with p ≥ τ that is not an ancestor of a kept label.
- **Region facets:** p ≥ 0.8, stored separately.

| τ (extra-label threshold) | items with 2+ labels | % of labeled items | % of all items | % with labels under two top labels |
| --- | --- | --- | --- | --- |
| 0.50 | 535 | 47.4 | 40.5 | 30.5 |
| 0.60 | 397 | 35.2 | 30.0 | 20.6 |
| 0.70 | 263 | 23.3 | 19.9 | 12.4 |
| 0.75 | 207 | 18.4 | 15.7 | 9.0 |
| 0.80 | 164 | 14.5 | 12.4 | 7.2 |
| **0.85 (used)** | **110** | **9.8** | **8.3** | **5.1** |
| 0.90 | 60 | 5.3 | 4.5 | 3.0 |
| 0.95 | 24 | 2.1 | 1.8 | 1.3 |

The 5–15% target corresponds to τ between 0.80 and 0.90; the default is **τ = 0.85**. Twenty-five examples by probability band are in `~/acct_cleanup/experiments/multi_label_examples.md`. Typical extras at ≥ 0.85: an AI design tool also under design software; a recruiting system also under HR software; a stock-media marketplace also under creator marketplaces. Below about 0.7 extras are mostly sibling sub-labels of the primary.

**Polyhierarchy at label level** is rare in this data: of 330 (sub-label, other top label) questions one reached 0.7 (used-car marketplaces, under Travel and also under Shopping). Six reach 0.5, but two of those are wrong.

## 7. Grok and Jev: division of labour, measured

| Step | Model | Shape |
| --- | --- | --- |
| Describe, re-describe from homepage | Grok | generation |
| Propose top level; name new labels | Grok | generation, representatives only |
| File into top labels | Jev | one `noul` per top label; `choice` as fallback |
| Assign sub-labels and extra labels | Jev | one `noul` per candidate |
| Split veto, merge confirmation, extra parents | Jev | `noul`, state = label descriptions + 5 representatives |

**Probe.** `GET /v1/models` → 200, lists `jev-latest` and `jev-preview`. `jev-1.14.0` is not listed but accepted. `questions` must be an object keyed by question id (a list returns 422). The response has `answers` and `usage {input_tokens, output_tokens}`; no cost field and no rate-limit headers.

**Cost and latency (real).** 3,838 requests, 6.66M input tokens, **$0.279**, including the agreement study. The two per-item stages of one run: filing 1,161 requests × 22 questions = 2.99M tokens, $0.126; sub-label assignment 1,161 × 8 questions = 1.10M tokens, $0.046. Latency per request: **p50 0.14–0.15 s, p95 0.23–0.25 s**, max 1.1 s. Wall time at 8 concurrent requests: **24 s per 1,161 items** (about 48 requests/s). **No 429/529, no retries, no failed requests.** Tokens scale with the number of questions (about 115 per question for a 15-word state), so the state is in effect billed once per question; asking only candidate labels matters more than shortening the state.

**Agreement with the v2 Grok stand-in** (identical questions; stand-in answers read from the v2 cache):

| Question | n | Same answer | Notes |
| --- | --- | --- | --- |
| Branch filing, 11 v2 branches + "none" | 1,322 | **87.8%** (Cohen's κ 0.87) | by embedding-margin quartile 77 / 87 / 92 / 96%; by Jev confidence: <0.5 → 49%, 0.5–0.7 → 59%, 0.7–0.9 → 82%, ≥0.9 → 99% |
| Second-membership yes/no | 40 | 82.5% | Jev 14 yes, Grok 15 yes; r = 0.82 |
| Merge two nodes | 10 | 90% | Jev 2 yes, Grok 1 yes |
| Split a node | 10 | 100% | both said yes to all 10 |

Consequences for the design:

- Jev can replace the stand-in; disagreement concentrates where Jev's own confidence is below 0.7 (263 items), which is the set to show in a review list.
- Jev abstains more: "no folder fits" for 86 items against Grok's 30.
- Grok's "second folder" was generous: Jev's independent noul for the same folder is ≥ 0.5 for 40% of Grok's 326 picks and ≥ 0.85 for 6%. Per-label nouls with a threshold give a tunable rate; a free-form second pick does not.
- The split yes-bias is real for both models, and in v3 the Jev veto fired 0 times in 11. The data test does the selecting; the veto is kept only because it costs $0.0004 per run.
- Name bias ("Type-Safe Is Not Error-Free"): the label "Identity and security" attracted login and sign-in pages of unrelated services (a retailer's sign-in page got it as primary with p = 0.96). See open question 4.

## 8. Results of the v3 rerun

1,322 entities, descriptions and embeddings reused from v2.

- **Homepage fetch for all 271 unsure items** (20 have no domain): 162 pages returned a title or meta description; **110 became confident**; **161 (12.2%) are Unclustered as unrecognised**.
- **Labeled: 1,128 items (85.3%). Unclustered: 14.7%** = 12.2% unrecognised + 2.5% (33 items) where no label reached 0.5.
- **Taxonomy:** 19 top labels, 110 sub-labels, 1 multi-parent label, 2 region facets (India 168 items, United States 176). Top labels and counts are in Appendix A.
- **Multi-label rate: 9.8% of labeled items** (110 items; 4 have three labels); 5.1% carry labels under two different top labels.
- **Judge spot check** (`grok-4.7` low): 15 labels, mean coherence 3.93 / 5 (v2's chosen tree: 4.13 on 15 nodes; the difference is within noise), name fits the members for 10 of 15; 15 extra-label assignments: primary correct 15 of 15, extra correct 14 of 15.
- **Known errors:** the login-page bias above; two junk-drawer sub-labels (a "mobility extras" label mixing parking, lounges and postal services; an email-marketing label holding unrelated senders); the 82-item AI-tools label has no sub-labels.

Listing: `~/acct_cleanup/experiments/experiment_labels.md`.

## 9. Cost

Prices: `grok-4.7` $2.00 in / $0.50 cached in / $6.00 out per 1M tokens (reasoning billed as output); Jev $0.042 per 1M input tokens, output free; OpenAI embeddings $0.02 per 1M tokens.

**One full run on 1,322 items (all measured):**

| Stage | Cost |
| --- | --- |
| Describe (Grok, 27 calls; v2 measurement) | $0.474 |
| Fetch + re-describe 162 pages (Grok, 5 calls) | $0.077 |
| Embeddings (OpenAI) | $0.0003 |
| Propose top level (Grok, 1 call) | $0.053 |
| Name 110 sub-labels (Grok, 5 calls) | $0.089 |
| File into top labels (Jev) | $0.126 |
| Sub-label assignment (Jev) | $0.046 |
| Split veto, merge check, extra parents (Jev, 125 requests) | $0.003 |
| **Total** | **≈ $0.87** (Grok $0.69, Jev $0.175) |

Describing is 55% of the run and is paid once; a re-run after editing labels costs only the Jev rows it invalidates (at most $0.175). Asking only the 6 nearest top labels instead of all 19 would cut filing to about $0.04. The same decisions on the Grok stand-in cost about $0.65 in v2.

**Actual v3 experiment spend: $0.58 of the $3.00 budget** (`spend_ledger_v3.json`): Grok $0.302 (re-describe $0.077, two top-level proposals $0.109, naming $0.089, judge $0.028), Jev $0.279 (run $0.175, agreement study $0.103), embeddings < $0.001.

## 10. Evaluation without a gold set

- **Read the tree:** `taxonomy.md` and `items.csv` are the acceptance check; the pause is the first checkpoint.
- **Structure report on every run:** labels per level, size distribution, split-test table (which labels split, k, z, stability, which did not and why), merge candidates, multi-label rate against the threshold table, unclustered share by reason.
- **Review list:** items whose primary p or `choice` confidence is below 0.7, and extra labels with p within 0.05 of τ.
- **LLM-judge spot checks:** label coherence and name fit, extra-label correctness; 30+ samples for a release check, with a judge model other than the describer when available.
- **Stability:** re-run the split test with another seed and compare; adjusted Rand index across embedding backends.
- **Operational:** unsure rate, fetch success, Jev error and retry counts, tokens against §9.

## 11. Config, CLI, privacy, tests

- **Config:** `DescriptionConfig`; `EmbeddingConfig.backend = openai|local|fake`; `DecisionConfig {backend = typesafe|grok|off, model = jev-1.14.0, concurrency = 8, max_retries = 6}`; `TaxonomyConfig {extra_threshold = 0.85, primary_threshold = 0.5, candidates = 8, split_alpha = 0.05, split_stability = 0.6, null_replicates = 30, parent_threshold = 0.7, facets = []}`.
- **CLI:** `constella organize INPUT... --output-dir RUN [--embedding openai|local] [--decision typesafe|grok|off] [--extra-threshold 0.85] [--facets region] [--fetch on|off] [--dry-run]`, then `constella organize --resume RUN`.
- **Jev client:** plain HTTPS, questions as an object, exponential backoff with jitter on 429/529/5xx, non-retryable on 4xx, every response cached on disk, key-shaped strings redacted from stored error bodies.
- **Privacy:** Grok, Jev and OpenAI receive only the display name, the registrable domain, the generated description, and on fallback the public homepage title and meta description (titles containing an email address or phone number are stripped). Usernames, addresses, URL paths and query strings never leave the machine.
- **Tests (offline, fakes for chat, decision and embedding):** split test rejects structureless data and accepts planted clusters; no item is lost by split or merge; ancestors are never stored; at most one primary per item and never a facet; unsure-after-fetch never reaches clustering; resume re-asks only questions about edited labels; YAML validation errors name the offending label; backoff retries 429/529 and stops on 401/422.

## 12. Milestones

0. **Approval of this v3** and answers to §13. Nothing is built before this.
1. Core models (`Label`, `Assignment`), DAG utilities, split and merge tests, exports. Offline tests.
2. Description stage with batching, cache and homepage fallback.
3. `decide()` with `typesafe`, `grok` and `off` backends; Jev client with backoff and cache.
4. Top-level proposal, `labels_proposed.yaml`, `--resume`.
5. Label assignment, thresholds, review list; `constella.records` adapter and the CLI.
6. Real run on the full list (≈ $0.87), user review, README update.

## 13. Open questions for the user

1. **Extra-label threshold.** Which rate do you want after reading `multi_label_examples.md`?
   A) τ = 0.85, 9.8% of labeled items — **recommended**;
   B) τ = 0.80, 14.5%;
   C) τ = 0.90, 5.3%.
2. **Large labels with no stable sub-structure.** The split test left "AI tools and platforms" (82 items), "Identity and security" (48) and "Jobs and careers" (47) without sub-labels.
   A) leave them flat, as the data says — **recommended for the first release**;
   B) for labels above √N items that fail the test, let Grok propose sub-labels, marked `origin: grok` and shown at a second pause;
   C) lower the stability bar from 0.6 to 0.5 (splits the AI-tools label, with less reproducible children).
3. **Sub-label granularity.** The rule produced 110 sub-labels with a median of 7 items.
   A) keep (smallest stable cut within one null standard deviation of the best) — **recommended**;
   B) coarser: take the first cut that beats the null and recurse, giving fewer, larger sub-labels and a third level, at lower stability;
   C) keep the rule but raise the minimum useful size from ½·log₂N (5) to log₂N (10).
4. **Login and sign-in pages.** "Identity and security" became the primary label for sign-in pages of unrelated services.
   A) tell the describer to describe the service behind the domain, not the login page, and narrow the label definition to identity providers and password tools — **recommended**;
   B) keep the primary as is and rely on the extra label;
   C) add an `access` facet for login portals, like the region facet.
5. **Multi-parent labels.** Only one label reached the 0.7 bar for a second parent.
   A) keep 0.7 — **recommended**;
   B) lower to 0.5 (six labels, two of them wrong) and list them in the review file;
   C) let Grok propose second parents when it names the labels.
6. **A second pause.** Should the run also stop after the sub-labels are built and named?
   A) no, one pause; edit `labels.yaml` afterwards and `--resume` again if needed — **recommended**;
   B) yes, always pause twice.
7. **Region facet.** Grok proposed India and United States facets (168 and 176 items at p ≥ 0.8).
   A) off by default, enabled with `--facets region` — **recommended**;
   B) on by default;
   C) drop facets from the first release.
8. **Top-level size hint.** Without a hint Grok proposed 41 labels; with the √N hint, 19.
   A) keep the √N hint — **recommended**;
   B) no hint, and merge at the pause;
   C) add a `--top-level broad|fine` switch.

## Appendix A — Top labels of the v3 run

Counts are items carrying the label or any descendant (primary or extra).

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

m = primary members at filing; z = standardised gap at the chosen k; stability = mean adjusted Rand index over ten 80% subsamples; Jev = veto `noul` (≥ 0.5 allows the split).

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
| Jobs and careers | 52 | not split: no cut beats the null (p = 0.39) | – | 1.4 max | – | – |
| Fitness and wellness | 39 | not split (p = 0.16) | – | 2.0 max | – | – |
| Payments and transfers | 38 | not split (p = 0.23) | – | 1.5 max | – | – |
| Identity and security | 37 | not split (p = 0.36) | – | 1.3 max | – | – |
| Dating and matchmaking | 32 | not split (p = 0.45) | – | 1.0 max | – | – |
| Home and local services | 31 | not split (p = 0.16) | – | 2.2 max | – | – |
| Food delivery | 27 | split | 4 | 2.9 | 0.84 | 0.89 |
| Startups and crowdfunding | 25 | not split (p = 0.94) | – | −0.6 max | – | – |

k counts the clusters of the cut; clusters under 3 items are not turned into labels, so the number of sub-labels can be lower. All 22 sub-labels large enough to test (≥ 10 members) failed the null test, which is why there is no third level.

## Appendix C — Sources

- Jev: https://docs.typesafe.ai/introduction/quickstart, https://docs.typesafe.ai/models, https://docs.typesafe.ai/api/errors.md, https://api.typesafe.ai/openapi.json; hierarchical classification cookbook: https://docs.typesafe.ai/cookbooks/hierarchical_classification.md (notes gathered 2026-11-11).
- Grok pricing: https://docs.x.ai/developers/pricing.
- Experiment files in `~/acct_cleanup/experiments/`: `experiment_labels.md`, `multi_label_examples.md`, `jev_report.md`, `run_v3/` (`labels_proposed.yaml`, `labels_final.json`, `assignments.csv`, `items.csv`, `summary.json`, `build_report.json`), `agreement_v3.json`, `judge_v3.json`, `spend_ledger_v3.json`, scripts `10_`–`17_`.
