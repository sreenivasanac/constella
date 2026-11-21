# Library v2: what changed from v4 and why

Written 2026-11-21. v4 (`ai_clustering_design_plan_v4.md`) stays as the record of the reasoning and the earlier measurements. The Kosha side of this decision is in Kosha's `docs/SEARCH_DESIGN.md`, section H.

## The change in one paragraph

v4 made Constella a service: its own MCP server, a SQLite workspace, a `Source` interface with a REST client, and seven endpoints Kosha had to offer. v2 makes it a library of functions that a host imports. Kosha already owns the items, the vectors, the tags, the Jev answer table and the MCP server, so everything v4 built to move those across a process boundary had nothing left to do. A file runner keeps Constella usable on a plain JSONL file.

The owner's view of embedding groups is now the rule: a group is a way to select items (group, pick one, group it again, combine with other operations). It is never a label applied to its members. Every label is checked per item by Jev.

## What exists

| Module | What it does |
| --- | --- |
| `constella.groups` | `group()`: mini-batch K-Means on unit vectors, k from the token budget, representatives per group, stable ids. `nearest_groups()`. |
| `constella.labels` | `Label`, `shortlist()` (nearest labels per item), `assign()` (primary, secondary, others, pending, doubtful band). |
| `constella.jev` | `Jev` client (sync and async), request packing, retries, spend, `estimate()`, `DiskCache`, `ask_cached()`. |
| `constella.loop` | `score_labels()`: one v4 round. `tree_hints()`: empty, too-broad and overlapping labels from counts. |
| `constella.runner` | `constella embed / overview / apply-labels / score / export / render` over a JSONL file and a workspace folder. |

## v4 parts that were not built, and why

| v4 | v2 | Why |
| --- | --- | --- |
| MCP server with four tools | None | One MCP server, Kosha's. Its `groups` and `score_labels` stages call this library. |
| `Source` protocol, `FileSource`, `RestSource`, seven REST endpoints | None | The host passes arrays and rows. The runner reads a file. |
| SQLite store for items, labels, scores, ledger | One small SQLite answer cache, optional | The host may pass its own cache (two methods). Kosha's `jev_answers` table is that cache. |
| LLM descriptions of thin items | None | Kosha writes summaries and media extracts. The runner sends the text it is given. |
| Relevance filter, approval thresholds, background runs, write-back | None | Host concerns: Kosha's filters, write previews and jobs. |
| Walking an item down the tree level by level | The host calls `score_labels` with a label's members and its children | The loop is one round; the order of rounds belongs to whoever drives it. |
| "A label and its ancestor never both count" inside `assign` | Not in `assign` | Kosha's `roles auto` applies it where the tree lives. `tree_hints` skips ancestor pairs when it looks for overlap. |
| Second look at 500 tokens for unplaced items | Always send 2,000 characters | Measured today: 500 tokens matches the full text; 150 tokens missed 4 of 8 and 8 of 33 yeses on two of three questions; and most posts are short enough that the cost is the same. |
| LiteLLM for embeddings | `httpx` against an OpenAI-compatible endpoint (OpenAI or Voyage) | Twenty lines replace a large dependency. A host with its own vectors never calls it. |

## Deleted

The silhouette/elbow/Davies–Bouldin search for k, the per-cluster one-shot LLM labelling, the `ContentUnitCollection` data model, the old pipeline, the D3 hover map and its HTML template, the Quora and Fireworks scripts, the old tests, and `build/`, `dist/`, `*.egg-info`. Dependencies dropped: `streamlit`, `psycopg`, `litellm`, `numba`. `umap-learn` and `matplotlib` moved to the optional `viz` extra for `constella render`.

## Decisions made from today's measurements

Details in `../measurements_2026-11-21_jev.md`.

- The state is billed once per request and every request costs about 260 tokens before anything else. So the client packs: 100 items per request when one question is asked of many items, and all of an item's label questions in one request when scoring labels.
- Label scoring measured 1,005 tokens per item for 8 labels ($0.042 per 1,000 items), which confirms v4's "expected" cost column.
- The two request layouts give slightly different probabilities (92 to 97 of 100 agree at 0.8). `score_labels` keeps the layout v4's 0.8 cut-off was measured with.

## Left for later

- No hand-labelled set yet, so the 0.8 cut-off is still v4's number from account names.
- `assign` has no cap on the number of extra labels (v4 said at most 4); with a shortlist of 8 it has not mattered.
- The cache stores a Score's value and confidence, not its per-level probabilities.
