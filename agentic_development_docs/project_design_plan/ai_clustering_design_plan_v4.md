# Constella v4 — Design Plan

Status: **design only, awaiting approval.** No code accompanies this document. Written 2026-11-20. It replaces v3.1 (`ai_clustering_design_plan.md`), which stays in the repo as the record of the measurements.

What changed since v3.1, in one line each:

- The statistical split test, merge test, √N priors, label DAG and facets are gone. Sub-labels come from the agent drilling into one label.
- Labels form a folder-path tree, 1 to 4 levels deep. An item has one primary label and may carry extra labels.
- Constella reads items from a **source** (a file, a REST API such as Kosha's, or a Python object) and writes results back to it. Item text never passes through the agent.
- Seed labels, precomputed embeddings, daily new items and a relevance filter are supported.
- Seven tools became four (plus one optional image tool).
- Jev is asked one yes/no question (a `Noul`) per candidate label. The `Choice` with an "uncategorised" option is dropped (§2.3).

Numbers marked **measured** come from the v3 experiments on 1,322 account names (`~/acct_cleanup/experiments/`, 2026-11-11), from a re-analysis of those cached answers done today at no cost, or from the 100 X bookmarks sampled today. Numbers marked **estimated** are derived from them. Jev facts marked **docs** come from docs.typesafe.ai, read today.

## 1. The algorithm

### 1.1 In one paragraph

Constella copies the items (text and vectors) from the source into a local workspace. It shows the agent a small map: the items cut into at most 40 groups by embedding, each with a count and three short samples. The agent writes labels, each a name and a one-sentence definition, starting from the seed labels if there are any. Jev then answers, for each item, one yes/no question per candidate label (the 8 labels nearest to the item by embedding): "does this item belong under this path?". The item takes its highest label at p ≥ 0.8 as primary, every other label at p ≥ 0.8 as an extra, and otherwise stays pending. Pending items are grouped again and shown to the agent, who adds or rewrites labels; only questions never asked before are sent. The same three moves (overview, apply labels, score) repeat inside any label to make sub-labels, down to four levels. A new item gets the same questions walking down the tree, with no agent involved.

### 1.2 Step by step

| # | Step | Runs where | What the agent reads |
| --- | --- | --- | --- |
| 1 | **sync.** Pull items, vectors, labels and user-made assignments from the source into the workspace. Compute vectors only for items that arrive without one. | Source API; embedding model only if needed | ≤ 150 tokens: counts |
| 2 | **Describe thin items.** An item whose text is under 40 characters once URLs are removed (a bare name, a bare link) gets a one-line description from an LLM, 50 items per call. The description replaces the text for embedding and for Jev. Items the LLM cannot describe become `unplaceable`. Part of `sync`. | LLM (server-side, never the agent) | nothing |
| 3 | **Relevance filter** (optional, off for Kosha). One Jev `Noul` per item with a plain-language rule given at setup, for example "This is a personal or informative message, not an automated notification". Items under 0.5 become `skipped`. Part of `sync`. | Jev | nothing |
| 4 | **overview.** Cut the items into at most 40 groups with k-means. For each group: count, three samples (≤ 100 characters each), the two nearest groups, the nearest existing label. Above it, the label tree with counts. | Local, no API | ≤ 4,000 tokens |
| 5 | **apply_labels.** The agent adds, redefines, moves, merges or deletes labels. Constella embeds each new definition and returns an estimated size per label (nearest-label count). | Embedding model (a few hundred tokens) | ≤ 400 tokens |
| 6 | **score(sample=300).** A dry run on 300 random items. Returns the share placed, the count per label and a p histogram. The agent fixes the labels and repeats, usually once or twice. Nothing is written back. | Jev, about $0.015 per run | ≤ 400 tokens |
| 7 | **Approval**, only above the size threshold (§5.4). | The user, in the source's UI | – |
| 8 | **score().** One Jev request per item: the first 150 tokens of its text as the state, one `Noul` per candidate label. Apply the rule in §2.4. Write the accepted assignments back to the source. | Jev | ≤ 400 tokens |
| 9 | **Second look.** An unplaced item whose text was cut at 150 tokens is asked again with its first 500 tokens. Part of `score`. | Jev | nothing |
| 10 | **Pending loop.** `overview(scope="pending")` groups what is left. The agent adds or rewrites labels. `score()` asks only the new questions. At most 3 rounds (§2.5). | Local, then Jev | ≤ 4,000 + ≤ 400 per round |
| 11 | **Sub-labels.** `overview(scope="label:<id>")` groups one label's members. The agent adds child labels under it. `score()` asks those members about the children, with the full path in each question. Repeat per label worth splitting, down to 4 levels. | Local, then Jev | ≤ 4,000 + ≤ 800 per label |
| 12 | **New items.** `sync` then `score()`. Each new item walks the tree: root candidates, then the children of every label it passed. No agent, no LLM unless the item is thin. | Jev | ≤ 550 tokens for both calls |

### 1.3 Choices and reasons

**Embedding model.** OpenAI `text-embedding-3-small`, first 500 tokens (about 2,000 characters) of `title + text`. It is what the experiments measured. When the source supplies vectors, Constella uses them and embeds only label definitions and thin-item descriptions, with the same model (the source declares it). The older note in `clustering_analysis_and_recommendations.md` recommended a larger model because K-Means alone had to separate topics. In v4 the vectors never decide anything, so the small model is enough: Jev's best label is among the 8 nearest by embedding for 96.6% of items, and 98.8% of all labels Jev rates ≥ 0.8 are among the 8 nearest (measured).

**Clustering.** Mini-batch K-Means on L2-normalised vectors, with k fixed by the token budget: `k = min(max_groups, ceil(n / 8))`, `max_groups = 40`. Reasons:

- The groups are a sampling device, not a result. They make sure the agent sees every region of the data. Jev decides membership afterwards, so the choice of k cannot produce a wrong label.
- k must be set by what fits in 4,000 tokens. v3's `2·√N` gives 346 groups at 30,000 items, about 18,000 tokens.
- It takes seconds at 30,000 items and is repeatable with a fixed seed, so group ids stay stable between calls.
- Ward needs an n × n matrix (about 7 GB at 30,000). HDBSCAN returns an unpredictable number of groups and, in the v2 comparison, left 17–64% of items as noise (measured).
- Nothing selects k from the data. The silhouette/elbow search in the current `kmeans.py` is the part `TODO.md` reports as not working, and it is deleted.

The agent can zoom: `overview(scope="group:g17")` cuts one group into up to 40 smaller ones.

**LLM.** Used by Constella itself for one job: one-line descriptions of thin items. Any LiteLLM model id, kept in a constant. Labels are written by the calling agent. A later no-agent mode reuses the same LLM as the proposer (§5.6).

**Jev.** `jev-1.14.0`, pinned by id, because an alias such as `jev-latest` moves and answers can change (docs). One request per item, 8 requests in flight. Every answer is stored with a hash of the model id, the question text and the item state, so no question is paid for twice and a rewritten label definition is re-asked automatically.

**Jev question shape.** State: `{"title": …, "text": first 150 tokens}`. One `Noul` per candidate:

```json
{"type": "noul",
 "instructions": "Does this item belong under \"AI learning > RAG > LangChain\"? Definition: tutorials, examples and tools built on the LangChain library.",
 "criteria": {"true": "A reasonable person would look for this item under that path.",
              "false": "The path does not describe this item."}}
```

Jev reads literally (docs, "model jaggedness"), so the definition is the label. A label without a definition is flagged in the overview.

## 2. Critique of the proposed algorithm

The idea under review: Constella proposes a taxonomy (from one-line summaries, embeddings, clusters); Jev checks each post against its suggested label and rejects misfits; the rejected ones are labelled again and re-checked, including whether the hierarchy fits; every Jev `Choice` offers "uncategorised"; an assignment is accepted above a cut-off of about 0.8–0.9.

### 2.1 What holds

| Part of the idea | Evidence |
| --- | --- |
| An LLM proposes, Jev verifies each item | On the same 1,322 filing questions Jev cost about a quarter of Grok's price and gave the same answer for 87.8% of items. Where Jev is ≥ 0.9 confident the two agree 97–100% (measured). |
| Loop on the rejected items | Three rounds took pending from 227 to 173 on labels the LLM added or rewrote (measured, conservative count). The new labels were sensible. |
| Cut-off around 0.8 | Labels accepted at 0.8 were about 97% correct, weighted by band (measured, 20 audited items per band, so roughly ±20 points per band). At 0.9 a third of the items go back for about 3 more points. |
| Caller drives a few tools; Constella keeps the items out of the context | Measured in v3: 3,800–4,000 input tokens per loop round. |
| Embeddings and clusters as inputs to the proposal | An overview of 68 groups for 1,161 items was 3,600 tokens (measured). |

### 2.2 What breaks at 30,000 items

| Problem | Why | Change |
| --- | --- | --- |
| A one-line LLM summary for every item | Measured $0.36 per 1,000 items. That is $10.80 at 30,000, over the whole budget before any labelling. | Describe only thin items (about 5% of the bookmarks are a bare link; measured). |
| "Check each post against the suggested label" | It needs someone to pick one label per post first. If the cluster picks it, the pick is wrong often: the nearest label by embedding equals Jev's best label for only 69% of items (measured). One candidate also cannot find a second label. | Ask the 8 nearest labels in one request. Checking and assigning become one step. |
| Re-running the check over everything each round | Each full pass at 30,000 items costs $1.35–2.50 and takes about 10 minutes (estimated). | Settle the labels on a 300-item sample first. After that, only unanswered questions are sent. |
| Looping until nothing is rejected | Returns fall fast: 75, then 19, then 11 items placed per round (measured). | Stop after 3 rounds or when a round places under 10% of the pending items. Accept an unsorted remainder of 10–15%. |
| An overview that grows with N | `2·√N` groups is 18,000 tokens at 30,000 items. | Groups are capped by tokens (40). The agent zooms into a group when it needs detail. |
| Long tool calls | 30,000 Jev requests take about 10 minutes at the measured 49 requests per second. | `sync` and `score` run in the background and are resumable; calling again returns progress or the result. |
| Unstable proposals | The same prompt run twice gave 19 and 14 top labels; the two partitions agree at an adjusted Rand index of 0.45 (measured). | Seeds anchor the tree, the sample run shows coverage before money is spent, and the user approves large changes. |

### 2.3 "Uncategorised" as a Choice option — replaced

v3.1 asked one `Noul` per candidate plus one `Choice` that included "none of these". I re-read the 1,161 cached top-level answers today (free, from `run_v3/filing.json`):

| Finding | Count |
| --- | --- |
| Items whose best `Noul` is ≥ 0.8 | 781 |
| Of those, the `Choice` picked "none" | 0 |
| Of those, the `Choice` picked the same label as the best `Noul` | 756 (96.8%) |
| Of the 25 that differ, the `Choice`'s label also has a `Noul` ≥ 0.5 | 23 |
| Items where no `Noul` reaches 0.5 | 141 |
| Items where the `Choice` picked "none" | 29 |

So at the cut-off the `Choice` decided nothing the `Noul`s had not already decided, and its "none" option caught a fifth of the items that had no fitting label. The TypeSafe docs say the same: an `other` option competes with the real options for probability, and the recommended pattern for items that can match several options is one `Noul` per option, where "nothing fits" is simply every `Noul` coming back low (docs, "Companion Noul").

**Change: one `Noul` per candidate label and no `Choice`.** "Uncategorised" is not an option the model picks. It is the state of an item with no label at or above the cut-off.

### 2.4 How the questions the brief asks are answered

**Does this item fit this path?** Each level is checked on its own, conditional on the level above. An item is asked about a label's children only after it passed the label itself, and each child question carries the full path (`AI learning > RAG > LangChain`) and the child's definition. The item ends on the deepest node it passes. If no child passes, it stays directly on the parent, which is normal. Ancestors are implied and never stored. This is the beam search of TypeSafe's hierarchical-classification cookbook with `Noul`s in place of a `Choice`: every label that passes is followed, not only the best one.

**Is the tree itself right?** Jev is not asked. Both Jev and Grok said yes to all 10 proposed splits in v3 (measured), so a yes/no question about structure carries no information. The tree is judged from counts that `overview` already shows:

- a child that attracts few of its parent's members is a bad child;
- a label where most members stay directly on it after children were added has the wrong children;
- two sibling labels that share more than half of the smaller one's members are a merge candidate (printed as `overlap`);
- a label that passes for more than 25% of all items is too broad (printed as `broad`). v3 had one: "Identity and security" attracted the sign-in pages of unrelated services.

**How is multi-label decided?** Each `Noul` is independent, so several can pass. Primary = the highest p at or above `cutoff` (0.8). Extras = every other label at or above `extra_cutoff`, at most 4 per item. A label and its own ancestor never both count; the deepest wins. In v3 `extra_cutoff = 0.85` gave 9.8% of items a second label and 0.80 gave 14.5% (measured, account names). For Kosha, where several tags per bookmark are normal, the default is 0.80, to be set from the first real run.

**How are rejected items re-processed?** In this order, cheapest first:

1. Text was cut at 150 tokens → ask Jev again with 500 tokens.
2. Still unplaced → the item is `pending`. `overview(scope="pending")` groups the pending items and shows the nearest existing label per group.
3. The agent adds a label, or rewrites a definition that was too narrow. `score()` asks the new questions of every item that has the new or changed label among its 8 nearest, placed or pending. So a new label can also pull items out of an old one.

**When does the loop stop?** When a round places under 10% of the pending items, when the agent has no more operations, after 3 rounds, or when the next call would pass the spending cap. What remains keeps the status `pending` and shows as unsorted in the source. Its best label under the cut-off is kept in Constella but not written back (open question 2).

**How are wrong labels noticed later?**

- **User corrections.** When the user removes a tag Constella attached, or attaches another, the source reports it on the next `sync`. A removed tag is never re-attached. `overview` prints the correction rate per label; a label with many corrections needs a better definition, and rewriting it re-asks only that label.
- **The doubtful band.** Assignments with p in [0.8, 0.9) were right 17 times out of 20; those at ≥ 0.9, 20 out of 20 (measured). `overview(scope="label:<id>", band="doubtful")` shows samples from the lower band so the agent or the user can spot-check a label.
- **New labels re-open old decisions**, as in step 3 above.
- **Pinned model.** Answers do not drift under a moving alias. Moving to a new Jev version is a deliberate re-score.

### 2.5 What else I changed and why

| Change | Reason |
| --- | --- |
| No statistical tests anywhere | Fixed decision 1. Also the simplest thing that works: the agent looks and decides, Jev verifies. |
| Dry run on a 300-item sample before the full pass | It costs about $0.015 and 6 seconds and tells the agent whether the labels cover the data. Not in the original idea; it is what makes iteration affordable at 30,000 items. |
| `score()` means "ask every unanswered question" | One rule replaces the scopes `changed`, `pending` and `all`. Calling it twice costs nothing the second time. |
| A wider Jev window before an LLM description | For a long post that Jev could not place, reading 500 tokens instead of 150 costs about $0.00006. An LLM description of the same post costs about $0.002 and adds nothing the text does not already say (both estimated). This narrows fixed decision 3; see §10. |
| The constellation image is optional and last | The v3 test was one image run, and two text-only runs differed from each other as much as either differed from the image run (measured). The useful part of the picture, which groups sit next to each other, is given as text: each group line names its two nearest groups (about 6 tokens). |
| No `grok` or `off` decision backends | One backend (Jev) and a fake for tests. If the key is missing the tool says so and stops; it does not fall back to embeddings. |

## 3. MCP tools

Four tools, plus one optional. All take `dataset` (a workspace name). None returns the item list.

| Tool | Input | Output | Typical response |
| --- | --- | --- | --- |
| `sync` | `dataset`; on first use `source: {type: "file", path}` or `{type: "rest", base_url, token_env}`; optional `relevance: "<rule>"`, `limit` | `{items, new, changed, deleted, thin_described, unplaceable, skipped, labels: {approved, proposed, rejected}, user_corrections, cost_usd}` | 100–150 tokens |
| `overview` | `dataset`, `scope: "tree" \| "all" \| "pending" \| "label:<id>" \| "group:<id>"`, `max_groups = 40`, `samples = 3`, `band` | `{totals: {placed, pending, unplaceable, skipped, new}, tree: [{id, path, n, n_direct, flags[]}], groups: [{id, n, near: [group ids], nearest_label, covered_pct, samples[]}], hints: [{overlap \| broad \| no_definition, …}], cost_so_far_usd}` | `tree`: about 10 tokens per label. With groups: 3,000–4,000 tokens. Hard cap 4,000. |
| `apply_labels` | `dataset`, `ops: [{op: "add", name, definition, parent?} \| {op: "redefine", id, definition} \| {op: "rename", id, name} \| {op: "move", id, parent} \| {op: "merge", id, into} \| {op: "delete", id}]` | `{applied, rejected: [{op, reason}], labels: [{id, path, est_items}], needs_approval, message}` | 150–400 tokens |
| `score` | `dataset`, optional `sample: 300`, `label: <id>`, `confirm_usd` | `{asked_items, placed, pending, moved, multi_label_pct, by_label: [{id, n, gained}] (top 15), p_histogram[10], round_gain_pct, should_stop, cost_usd}` or `{status: "running", done, total}` or `{needs_confirm: true, est_usd}` | 200–400 tokens |
| `render` (optional, last phase) | `dataset`, `scope` | PNG ≤ 1000×800: 2-D map, group ids and counts, no item text | about 800 image tokens (measured) |

Notes:

- **The cap is enforced.** `overview` counts tokens before returning and drops samples, then groups, until it fits.
- **Samples are the only item text the agent sees:** at most 120 snippets of 100 characters per call.
- **`score` asks before spending.** If the estimate is above $0.50 it returns `needs_confirm` with the estimate, and runs only when called again with `confirm_usd` at or above it. The agent passes the number on to the user.
- **Long calls.** `sync` and `score` return `running` with progress after 20 seconds and continue in the background. A second call returns progress or the final summary. A killed run resumes from the stored answers.
- **Writing back** happens at the end of every `score` that is not a sample run. There is no export tool. For a file source the write-back is `labels.json` and `assignments.jsonl` in the workspace.
- **Dropped from v3.1:** `ingest` (now `sync`), `describe` (inside `sync`), `export` (inside `score`).

A whole first run, as the agent sees it:

```text
sync                                   150 tokens
overview(all)                        4,000
apply_labels(add ×20)                  400
score(sample=300)                      400      repeat the last two once or twice
-- user approves, if needed --
score()                                400
overview(pending) -> apply -> score  4,800      up to 3 rounds
per label worth splitting:
  overview(label) -> apply -> score  4,800
```

That is about 80,000 tokens of tool results at the caps for a tree of 20 top labels with 12 of them split, and less in practice because one label's overview is smaller than the cap. It depends on the number of labels, not on the number of items (estimated). Per v3, one round measured 3,800–4,000 input and 1,400–3,600 output tokens with Grok as the agent.

## 4. Data model and persistence

### 4.1 Objects

```text
Item        {id, text, title?, metadata?}                         what a source supplies
            + description?, content_hash, status, vector row      what Constella adds
            status: new | placed | pending | unplaceable | skipped

Label       {id, ext_id?, parent_id?, name, definition,
             status: proposed | approved | rejected,
             origin: seed | agent | user}

Score       {item_id, label_id, p, question_hash, scored_at}      every answer Jev gave
Assignment  {item_id, label_id, p, primary, source: jev | user}   derived from scores by the rule in §2.4,
                                                                  plus what the user set in the source
Ledger      {at, tool, requests, input_tokens, usd}               spend per call
```

A "run" is not an object. The workspace is the state, and every tool call is resumable because answers are stored as they arrive. The ledger replaces v3's manifest.

### 4.2 Workspace

One directory per dataset, `~/.constella/<dataset>/`:

| File | Content |
| --- | --- |
| `constella.db` | SQLite (standard library): items, labels, scores, assignments, ledger, cached groups |
| `vectors.npy` | float32 matrix, one row per item (184 MB at 30,000 × 1,536) |
| `config.json` | source, model ids, cut-offs, relevance rule. Tokens are named by environment variable, never stored. |

### 4.3 Who owns embeddings and cluster state — decided

**The source owns what it needs anyway. Constella owns what only Constella uses. Constella keeps a cache of the first, not a second copy.**

| Thing | Owner | In the other system |
| --- | --- | --- |
| Items and their text | Kosha | Constella caches the first 2,000 characters, keyed by `content_hash` |
| Item embeddings | Kosha (pgvector; needed for search whether or not Constella runs) | Constella caches them in `vectors.npy` |
| Tags and their tree | Kosha (`tags`) | Constella mirrors them by id on every call. Kosha wins every conflict. |
| Accepted assignments | Kosha (`item_tags`) | Constella derives them and writes them |
| Every Jev answer, including the ones under the cut-off | Constella | not in Kosha |
| Label definition vectors, cached groups, spend ledger | Constella | not in Kosha |

Reasons:

- **Why Kosha computes the embeddings.** It needs one per item at save time for search. If Constella computed them, Kosha would wait on Constella to make a bookmark searchable.
- **Why Constella still keeps the vectors locally.** Grouping and shortlisting need the whole matrix in memory. Fetching 30,000 vectors over REST is about 250 MB of base64, too much to repeat on every call. The cache is rebuilt from Kosha for free if the workspace is deleted.
- **Why the Jev answers stay in Constella.** They are 8 to 30 numbers per item that only Constella reads: to avoid paying twice, to change a cut-off without re-asking, and to compute the overlap and broad-label hints. Putting them in Kosha would put Constella's internals in Kosha's schema, and file datasets need the store anyway.
- **Why group state stays in Constella.** Groups are a view, recomputed from a fixed seed. They are cached only so that `group:g17` means the same thing in the next call.

What is lost if the workspace is deleted: the Jev answers (rebuilt by paying again, $1–6 at 10,000–30,000 items) and nothing else.

Rejected alternatives: Constella reading Kosha's Postgres directly (ties Constella to Kosha's schema; the REST decision was made); a stateless Constella with everything in Kosha (Kosha would have to store and serve the answer cache).

## 5. Kosha ↔ Constella contract

### 5.1 The source interface

Inside Constella a source is any object with five methods. That is the whole abstraction.

```python
class Source(Protocol):
    def meta(self) -> dict: ...
    def iter_items(self, updated_after: str | None) -> Iterator[dict]: ...
    def labels(self) -> list[dict]: ...
    def write_labels(self, ops: list[dict]) -> dict: ...        # returns new ids
    def write_assignments(self, batch: list[dict]) -> dict: ...
```

Two implementations ship: `FileSource` (JSONL or CSV in; `labels.json` and `assignments.jsonl` out) and `RestSource` (the endpoints below). A direct library caller passes its own object with these methods.

### 5.2 Endpoints Kosha must offer

Base path `/api/constella/v1`. Seven endpoints.

| # | Endpoint | Purpose |
| --- | --- | --- |
| 1 | `GET /meta` | `{"name": "kosha", "items": 12840, "embedding": {"model": "text-embedding-3-small", "dims": 1536}, "max_limit": 1000}`. `embedding` is `null` if Kosha sends no vectors. |
| 2 | `GET /items?updated_after=<cursor>&limit=500&include=embedding` | Changed items since the cursor, oldest first. Response `{"items": […], "next_cursor": "…" \| null}`. |
| 3 | `GET /labels` | All tags, unpaged: `[{"id", "name", "parent_id", "definition", "status", "origin"}]` |
| 4 | `POST /labels` | Create tags. Body `{"labels": [{"ref": "c1", "name", "definition", "parent_id" \| "parent_ref", "status": "proposed" \| "approved", "est_items": 412, "sample_item_ids": […]}]}` → `{"ids": {"c1": "tag_88"}}`. Idempotent on (parent, lower-cased name): a repeat returns the existing id. |
| 5 | `PATCH /labels/{id}` | `definition` and `est_items` at any time. `name` and `parent_id` only while the tag is `proposed` and was created by Constella. |
| 6 | `DELETE /labels/{id}` | Only a `proposed` tag created by Constella. Anything else returns 409. |
| 7 | `PUT /assignments` | Replace Constella's tags on up to 1,000 items per call. |

Item shape (endpoint 2):

```json
{"id": "x_1849302",
 "title": null,
 "text": "post text\n\nQuoted: …\n\nLink: page title — page description",
 "metadata": {"source": "x", "url": "…", "author": "…", "saved_at": "2026-11-19T08:12:00Z", "has_video": true},
 "content_hash": "9f2c…",
 "embedding": "<base64, float32 little-endian>",
 "user_labels": ["tag_12"],
 "removed_labels": ["tag_9"],
 "deleted": false}
```

- **`text` is composed by Kosha**, up to 4,000 characters: the post, the quoted post, the title and description of external links, the head of an article body. Constella truncates. Kosha knows these parts; Constella should not parse sources. This matters for the data: 11% of the sampled bookmarks quote another post and 41% carry a link with a title (measured).
- **`user_labels`**: tags attached by the user or by an import (X folders, Tweetsmash labels, Dewey folders, playlist names). These are the seed memberships and the corrections.
- **`removed_labels`**: tags Constella attached and the user removed. Kosha must keep that fact (a tombstone on the `item_tags` row).
- **Cursor**: opaque, a keyset on `(updated_at, id)`. `updated_at` must change when the text changes, when the user changes the item's tags, and when the item is deleted (`deleted: true`).
- **Paging**: `limit` up to `max_limit`. With vectors a page of 500 is about 4 MB.

Assignments (endpoint 7):

```json
{"items": [{"id": "x_1849302",
            "labels": [{"id": "tag_88", "p": 0.93, "primary": true},
                       {"id": "tag_12", "p": 0.86, "primary": false}]}]}
```

For each listed item Kosha replaces the `item_tags` rows whose `attached_by` is `constella` with the given set (`confidence = p`). It never touches rows attached by the user or an import, never re-creates a row listed in `removed_labels`, and keeps the user's primary flag if the user set one. Response `{"written": 1964, "skipped_user_removed": 3}`. The call is idempotent.

**Auth.** `Authorization: Bearer <token>`. The token is made in Kosha's settings with one scope: read items and tags, write proposed tags, write Constella's own assignments. Constella's config holds the base URL and the name of the environment variable that holds the token. HTTPS unless the host is localhost. Errors: 401, 409 (conflict, body carries the existing id), 429 with `Retry-After`.

### 5.3 What Constella writes back

| What | When | How |
| --- | --- | --- |
| New labels | `apply_labels` | `POST /labels`, `proposed` when approval is needed (§5.4), else `approved`; `origin: constella` |
| Definitions and estimated sizes | `apply_labels`, sample `score` | `PATCH /labels/{id}` |
| Assignments with confidence | end of each full `score` | `PUT /assignments`; only items whose set changed |
| Nothing else | | Thin-item descriptions and under-cut-off suggestions stay in Constella (open questions 2 and 5) |

Renaming, moving, merging or deleting a tag the user has approved is not Constella's to do. `apply_labels` rejects the operation with the reason "approved in Kosha; change it there", and the agent uses Kosha's own MCP tools. Constella sees the result on its next call: a merged or deleted tag disappears and its items are asked again about the remaining candidates; a renamed tag keeps its id.

### 5.4 Approval

A batch of label operations needs the user's approval when either holds:

- the dataset has more than 300 items and the batch adds, deletes or merges a top-level label;
- one operation is estimated to change the primary label of more than 200 items.

Everything else applies at once: sub-labels of a small label, labels for the daily trickle, any run on a small dataset. Both numbers are config constants.

Flow:

1. `apply_labels` creates the tags in Kosha as `proposed`, with estimated sizes and three sample item ids each, and returns `needs_approval: true`.
2. The agent may still edit them and run sample scores. Sample scores work on proposed labels and update the estimated sizes. A full `score` skips proposed labels and says so.
3. The user reviews in Kosha's UI: approve, rename, edit the definition, reject. Or the user tells the agent, which uses Kosha's MCP tool. Constella has no approve tool, so nothing reaches the items through Constella without Kosha recording an approval.
4. Constella's next call reads `GET /labels`. Approved tags go live. A rejected tag is kept as a tombstone, so the same name under the same parent is refused if the agent proposes it again. An edited definition drops the stored answers for that label only.
5. `score()` runs and writes the assignments.

For a file source the same flow uses `labels_proposed.yaml` in the workspace with `approved: true` set by hand, as in v3.1.

### 5.5 New items

```text
daily (cron, or the agent):   sync  ->  score
```

`sync` fetches what changed since the stored cursor: about 20 new bookmarks a day, with their vectors. `score` walks each new item down the tree, 1 to 4 Jev requests. Thin items get a description first. Assignments go back through `PUT /assignments`. Cost: about $0.002 a day (estimated). Nothing is re-run.

Items that find no label stay `pending`. `sync` and `score` report the pending count; when it passes 50 items or 5% of the dataset the response says `suggest: overview(scope="pending")`, and the next agent session adds labels for them. Deleted items are dropped from the workspace. An item whose `content_hash` changed is asked again.

### 5.6 Kosha calling Constella without an agent (later, not designed here)

What it would need:

- **Daily labelling:** nothing new. `constella.sync(dataset)` and `constella.score(dataset)` are plain Python functions; the MCP server only wraps them.
- **Proposing labels:** a function `constella.auto.organize(dataset)` in which an LLM API plays the agent: it receives the same overview text and returns the same operations as JSON. v3's `18_loop.py` did this with Grok at $0.014–0.028 per round (measured). About 100 lines and one prompt.
- **If Kosha is not Python or runs elsewhere:** four HTTP routes mirroring the four tools.
- **Progress and approval** surface in Kosha's UI through the proposed tags and the `running` status the tools already return.

## 6. Cost and tokens

### 6.1 Prices and facts

| Item | Value | Status |
| --- | --- | --- |
| Jev price | $0.042 per 1M input tokens; output free | Confirmed in the docs (models page) and matches the v3 ledger |
| Jev limits | 64k tokens per request, 32k per question; a `Choice` takes at most 255 options | Docs |
| Jev batch discount or batch API | None found. "Batching" in the docs means many questions in one request. | Not found |
| Is the state billed once per request or once per question? | **Not confirmed.** The docs say the state is ingested once, and their cookbook reports a 13-question request 12.2× cheaper than 13 single requests. The v3 report read its own token counts as "once per question", but its states were 15 words, too short to tell. | Open; measurement 1 settles it |
| Jev rate limits | Per organisation, shown in the developer console. v3 ran 49 requests per second with no 429. | Limit not confirmed |
| Jev tokens, 8 `Noul`s on a 15-word state | 942 per request (median) | Measured |
| OpenAI `text-embedding-3-small` | $0.02 per 1M tokens | Measured against the ledger |
| LLM description of a bare name | $0.36 per 1,000 items with `grok-4.7`, low reasoning, 50 per call | Measured |
| Agent round (overview in, operations out) | 3,800–4,000 input and 1,400–3,600 output tokens; $0.014–0.028 with Grok | Measured |

### 6.2 Assumptions (all estimated)

1. Kosha-like items. Jev state capped at 150 tokens. Jev request = 942 + 130 extra state tokens ≈ **1,070 tokens** if the state is billed once (expected), or ≈ **1,980** if billed per question (worst case).
2. Jev requests per item: 1.0 at the root; 0.65 at level 2 (80% placed, 70% of them in a label that has children, 15% carry an extra label); 0.25 at level 3; 0.05 at level 4; 0.25 for the pending rounds and changed labels; 0.2 for second looks at 500 tokens. **About 2.4 request-equivalents per item.**
3. Three sample runs of 300 items: about 1M tokens, $0.04 at any size ($0.07 in the worst case).
4. Thin items needing an LLM description: 15% (5% bare links measured; the rest is a guess for video posts with almost no text), at the measured $0.36 per 1,000 described.
5. Embeddings: 200 tokens per item on average (median text is 300 characters; 41% are long posts). Zero if Kosha supplies them.
6. No relevance filter (Kosha). Where used, add one request of about 300 tokens per item: $0.013 per 1,000 items.

### 6.3 Estimates

| | 1,000 items | 10,000 items | 30,000 items |
| --- | --- | --- | --- |
| Jev, expected | $0.15 | $1.12 | $3.28 |
| Jev, worst case | $0.27 | $2.07 | $6.06 |
| LLM descriptions of thin items | $0.05 | $0.54 | $1.62 |
| Embeddings (if Constella computes them) | $0.004 | $0.04 | $0.12 |
| **Constella total, expected** | **$0.20** | **$1.70** | **$5.02** |
| **Constella total, worst case** | **$0.33** | **$2.65** | **$7.80** |
| Jev wall time at 49 requests per second | 1 min | 8 min | 25 min |
| Agent's own tokens (tool results read) | about 30,000 | 60,000–80,000 | 60,000–80,000 |

- The target of under $10 for 5,000–10,000 items is met with a wide margin: $0.90–2.70.
- The agent's tokens are on the agent's bill (a Claude Code subscription, for example). If Constella's own LLM plays the agent, add about $0.50 per full run (estimated from the measured per-round cost and about 20 rounds).
- For comparison, the measured v3 run on 1,322 items cost $0.97, of which $0.55 was describing every item.
- Levers if the worst case holds: 6 candidates instead of 8 (recall 93.5% against 96.6%, measured); shorter questions; a cheaper model for descriptions.
- A new day's 20 items: about $0.002.

## 7. The existing code

About 2,650 lines. The flat, single-label, in-memory pipeline does not survive, but four pieces do.

| Module | Lines | Verdict | Why |
| --- | --- | --- | --- |
| `data/models.py` | 316 | **Delete** | `ContentUnitCollection` holds vectors as Python lists, is single-label and flat. `get_content()` builds the embedded text with `Id:`, `Path:`, `Size:` and `Cluster ID:` lines (`models.py:64-81`). Replaced by a small `Item` and the store. |
| `data/results.py` | 67 | **Delete** | Result dataclasses for the old pipeline. |
| `config/schemas.py` | 66 | **Rewrite** | One `Config` dataclass and module constants. Drops the hard-coded `/tmp` path and `gpt-4o-mini`. |
| `embeddings/adapters.py` | 216 | **Keep the core, trim** | The token-budget batching and thread pool are sound. Add retries and truncation at 2,000 characters; default to OpenAI; drop the Fireworks subclass. Becomes `embed.py`, about 80 lines. |
| `embeddings/base.py` | 26 | **Delete** | An abstract class for one implementation. Tests pass a fake function. |
| `clustering/kmeans.py` | 462 | **Delete** | The k search (silhouette, elbow, Davies–Bouldin, capped at 15, `schemas.py:17`, `kmeans.py:383-387`) is the part that does not work. Replaced by about 15 lines of fixed-k mini-batch K-Means. |
| `labeling/selection.py` | 109 | **Keep the idea, rewrite** | Nearest-to-centroid plus a seeded diverse pick is exactly how overview samples are chosen. About 20 lines on a numpy matrix with cosine similarity. |
| `labeling/auto_label.py`, `prompts.py`, `response_parsing.py`, `providers.py` | 718 | **Delete; keep one pattern** | One LLM call per cluster to name it is no longer a step. The LiteLLM call with a pydantic `response_format` and backoff is reused in `llm.py` for descriptions. |
| `pipelines/workflow.py` | 290 | **Delete** | The old orchestration. |
| `visualization/umap.py` | 242 | **Keep for `render`** | Remove the unused `streamlit` import. Moves to `render.py` behind an optional `viz` extra. |
| `visualization/html_templates.py`, `umap_hover.html`, `artifacts.py` | 91 + HTML | **Delete** | Kosha has its own UI. Git keeps them if a standalone viewer is wanted. |
| `records/` | – | **Delete** | Only stale `.pyc` files, no source. |
| `scripts/` | 217 | **Delete** | The Quora runner needs `psycopg` and the old pipeline. |
| `tests/` | 464 | **Rewrite** | They test the deleted modules and stub out UMAP, so nothing real is exercised. |
| `README.md` | 142 | **Rewrite in the last phase** | It imports `constella.labeling.llm.get_labels`, which does not exist, and its `VisualizationConfig` example raises. |
| `pyproject.toml` | – | **Fix** | Remove `numba`, `streamlit`, `psycopg`. Declare `pydantic` and `tiktoken`, used but undeclared today. Add `httpx` and `mcp`. Move `matplotlib` and `umap-learn` to a `viz` extra. Add `[project.scripts] constella = "constella.cli:main"`. Version 0.2.0. |
| `build/`, `dist/`, `*.egg-info`, `old_code/`, `artifacts/` | – | **Delete locally** | Build output and old runs. Add `dist/` to `.gitignore`. |

New layout, flat, about 1,300 lines (estimated):

```text
constella/
  __init__.py      public functions: sync, overview, apply_labels, score
  config.py        Config dataclass; constants (cut-offs, model ids, sizes, thresholds)
  store.py         the workspace: SQLite and vectors.npy
  sources.py       Source protocol, FileSource, RestSource
  embed.py         embed(texts): LiteLLM, batching, retries
  llm.py           describe(items): one-line descriptions
  jev.py           Jev client (stored answers, backoff) and question builders; ported from experiments/common_v3.py
  core.py          the algorithm: grouping, shortlist, the assignment rule, the four operations
  mcp_server.py    the tools: thin wrappers and token caps
  cli.py           constella mcp | sync | score
  render.py        optional PNG map
tests/
```

## 8. Build order

### 8.1 Measure first (about $0.40 in total)

| # | Measurement | Why it comes first | Expected cost |
| --- | --- | --- | --- |
| 1 | **Jev on real bookmarks.** The 100 sampled X bookmarks and about 10 labels (the X folders plus a few written by hand). (a) Token bill for 1, 8 and 16 questions on states of 30, 150 and 500 tokens: settles whether the state is billed once. (b) Each bookmark scored three ways: first 150 tokens, first 500 tokens, an LLM one-line description. Compare the best label and the share at ≥ 0.8, split by post type (long, video, bare link, quote). | Decides the two biggest cost lines and tests the 150-token rule on text that is nothing like 15-word account descriptions. | Jev < $0.05; 100 LLM descriptions about $0.15 |
| 2 | **Overview at scale.** 10,000 Reddit posts: embed, cut into 40 groups, build the overview, count its tokens. The agent proposes top labels from it. `score(sample=300)` reports the share placed at 0.8. Repeat once after `overview(pending)`. | "40 groups of 3 samples is enough to propose a top level for 10,000+ items" is the one scaling claim not yet measured. v3 measured 68 groups for 1,161 items. | Embeddings about $0.04; Jev about $0.03 |

Both need a go-ahead for the spend. Measurement 2 can be scripted in an afternoon from the v3 experiment code.

### 8.2 Phases

| Phase | Build | Test that it works |
| --- | --- | --- |
| 1 | `store`, `FileSource`, `embed`, grouping, `overview`. Delete the old modules; fix `pyproject.toml`. | Offline, with planted clusters and fake vectors: every item is in exactly one group; group ids are identical across two calls; `overview` stays under 4,000 tokens at 1,000, 10,000 and 30,000 synthetic items. |
| 2 | `jev`, `apply_labels`, `score` at the root (sample and full), the assignment rule, file write-back. | Offline with a fake Jev: a second `score()` sends nothing; a redefined label is re-asked and others are not; the spending check refuses. Real: the account dataset with the v3 top labels places about 781 items at 0.8, for about $0.05. |
| 3 | Sub-labels with path questions, the pending loop, thin-item descriptions, the second look, the relevance filter. | Offline: an item never holds both a label and its ancestor; an item is asked about children only under a label it passed; the loop stops under the 10% rule. Real: a full run on the account dataset for under $1, compared with the v3 numbers. |
| 4 | `mcp_server`, `cli`, background runs. | Every tool response respects its cap. A full run driven from Claude Code on the account dataset. A run killed mid-`score` resumes without re-asking. |
| 5 | `RestSource`, seeds, approval, user corrections, new items. A fake Kosha server of about 100 lines for the contract tests; it doubles as the reference for Kosha's implementation. | Contract tests against the fake: paging, cursor, idempotent writes, user rows never touched, removed tags never re-attached, rejected names refused. Real: the 100 bookmarks from JSONL with the X folders as seeds. |
| 6 | Real Kosha once its API exists. README with screenshots. Optional `render`. | 10,000 real bookmarks under $3. The next day's bookmarks labelled by `sync` and `score` alone. |

## 9. Open questions

| # | Question | Recommended default |
| --- | --- | --- |
| 1 | Extra-label cut-off for Kosha: 0.80 (more tags) or 0.85? | 0.80, then adjust after the first real run. It is a config constant; changing it re-asks nothing. |
| 2 | Items whose best label is between 0.5 and 0.8: write them to Kosha as suggestions, or leave them unsorted? | Leave them unsorted. That band was 45–60% correct in the audit. |
| 3 | Which LLM writes the thin-item descriptions? | `grok-4.7` at low reasoning for now, since its cost is measured. Try a cheaper model if measurement 1 shows more than 15% thin items. |
| 4 | Approval thresholds: more than 300 items in the dataset for top-level changes; more than 200 items moved by one operation. | Keep both. Lower them if the first run feels unsupervised. |
| 5 | Should Kosha receive the thin-item descriptions, for search and display? | Not in the first version. It is one optional field on `PUT /assignments` later. |
| 6 | Who enriches video-only and link-only posts (link title, video title, transcript)? | Kosha, at save time, into `text`. Constella marks what is still empty as `unplaceable` and reports the count. |
| 7 | May Jev contradict a seed? A bookmark the user filed under "Lawyers" that Jev rates low. | No. User and import assignments are never removed or overridden; Jev only adds. |
| 8 | Sampling Gmail (118k) and Reddit (457k) down to a subset. | The source does it: `FileSource` takes `limit` and a seed, and the relevance rule filters. Constella does not add a sampling tool. |
| 9 | Constella deletes the interactive D3 map and keeps only the optional PNG. | Yes. Restore from git if a standalone viewer is wanted. |

## 10. Fixed decisions: kept, narrowed, changed

| # | Decision | Status |
| --- | --- | --- |
| 1 | Agent drills into a label and proposes sub-labels; Jev scores them. Statistical tests, priors, DAG and facets removed. | Kept as given. |
| 2 | Folder-path tree, no multi-parent; one primary and extra labels when Jev is confident. | Kept. Added: at most 4 extras per item; a label and its ancestor never both count. |
| 3 | Embed the first ~500 tokens; give Jev the first ~150; an LLM one-liner only for thin items and for items Jev cannot place. | **Narrowed, with a reason.** For an item Jev cannot place, Constella first re-asks Jev with 500 tokens. The LLM describes only thin items, where there is nothing more to read. The wider window costs about 30 times less than a description and uses the item's own words. Measurement 1 checks it; if the description wins there, the original rule comes back. |
| 4 | Pause for review only above a size threshold. | Kept, and made concrete in §5.4. |
| 5 | Generic item `{id, text, title?, metadata?}`. | Kept. Sources may add `embedding`, `user_labels`, `removed_labels`, `content_hash`. |
| 6 | Seeds, precomputed embeddings, batch ingest, relevance filter, overview under ~4,000 tokens. | Kept. |
| 7 | 5,000–10,000 items under $10. | Met: $0.90–2.70 estimated. |
| 8 | A handful of tools; an algorithm explainable in a paragraph. | Four tools; §1.1. |

Outside the fixed list, the proposed "uncategorised" option in every Jev `Choice` is replaced by one `Noul` per candidate label (§2.3).

## Appendix — Sources

- TypeSafe docs, read 2026-11-20: `introduction.md`, `models.md` (price, context limits, aliases, rate limits), `api/errors.md`, `api/system-one.md`, `patterns/companion-noul.md`, `confidence.md`, `model-jaggedness.md`, `concepts/state.md`, and the cookbooks `hierarchical_classification.md`, `parallel_questions.md`, `infinite_cardinality_choice.md`, `classification_using_confidence.md`.
- Experiments in `~/acct_cleanup/experiments/`: `jev_report.md`, `jev_stats.json`, `agreement_v3.json`, `loop_v3.json`, `image_test_v3.json`, `comparison.md`, `spend_ledger.json`, `spend_ledger_v3.json`, `run_v3/filing.json`, scripts `12_top.py`, `18_loop.py`, `common_v3.py`.
- The `Noul`-against-`Choice` table in §2.3 was computed today from `run_v3/filing.json` (1,161 cached top-level answers). No API call was made.
- Existing code: an inventory of every module under `constella/`, `tests/`, `scripts/` and `pyproject.toml`, made today.
- Kosha data: 100 X bookmarks sampled today (figures quoted in §2.2, §5.2 and §6.2).
