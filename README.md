# Constella

A small Python library for organising a set of items (bookmarks, posts, email threads, account names) without sending every item to an LLM.

It works in two moves:

1. **Group to select.** Cut the items into at most 40 groups by embedding, read a few samples of each, pick a group, group it again. A group is a way to choose items. It is never applied as a label.
2. **Check every label per item.** An agent or a person writes labels, each with a one-sentence definition. [TypeSafe Jev](https://docs.typesafe.ai) answers one yes/no question per item and candidate label, and returns a probability. An item takes the labels at 0.8 or above; the rest stay pending for the next round.

The library is pure functions that a host application imports (Kosha does). A file runner covers a plain JSONL file.

## Install

```bash
uv add git+https://github.com/sreenivasanac/constella  # or: uv sync, inside a clone
export TYPESAFE_API_KEY=...                     # only for Jev calls
```

Python 3.11 or later. Dependencies: numpy, scikit-learn, httpx.

## Library

```python
import numpy as np
from constella import Jev, DiskCache, Label, Question, ask_cached, group, nearest_groups, score_labels, tree_hints

vectors = np.load("vectors.npy")            # float32, one row per item; the host computes these
items = [{"id": 1, "title": "…", "text": "…"}, ...]
ids = [i["id"] for i in items]

# 1. Group, look, zoom
groups = group(vectors, ids)                 # at most 40, largest first, same ids every run
g = groups[0]
g.size, g.cohesion, g.central, g.spread      # count, tightness, ids near the centre, ids at the edges
nearest_groups(groups)[g.id]                 # [(other group id, similarity), ...]
rows = [ids.index(i) for i in g.ids]
subgroups = group(vectors[rows], g.ids)      # a second level

# 2. Labels, checked per item
labels = [Label("rag", "AI/RAG", "Retrieval-augmented generation: retrieving documents to ground an LLM's answer.")]
label_vectors = embed([l.definition for l in labels])    # your embedding function, same model as the items
result = score_labels(items, labels, vectors, label_vectors, Jev(), DiskCache("answers.sqlite"))

result.assignments[1].primary                # ("rag", 0.93) or None
result.pending                               # ids with no label at 0.8 or above
result.spend                                 # requests, input_tokens, usd, asked, cached
tree_hints(result.assignments, labels)       # {"empty": [...], "broad": [...], "overlap": [...], "sizes": {...}}

# One question over many items, 100 items per request
q = Question("Is this post about robotics?")
answers, spend = ask_cached(Jev(), [(q, i["id"], i["text"][:2000]) for i in items], DiskCache("answers.sqlite"))
```

Calling `score_labels` again with the same cache sends only the questions that were never asked: a new label, a rewritten definition, a changed item. Changing the cut-off re-asks nothing.

| Function | What it returns |
| --- | --- |
| `group(vectors, ids, k=None, max_groups=40, seed=0)` | `Group`s: `id, ids, size, centroid, cohesion, central, spread` |
| `nearest_groups(groups, n=2)` | the n closest groups for each group |
| `shortlist(item_vectors, label_vectors, n=8)` | indices of the n nearest labels per item |
| `assign(scores, accept=0.8)` | per item: `primary, secondary, others, doubtful`, and `pending` |
| `score_labels(items, labels, vectors, label_vectors, jev, cache=None, accept=0.8)` | `assignments, pending, spend, scores` |
| `tree_hints(assignments, labels)` | empty, too-broad and overlapping labels, from counts |
| `Jev().ask(pairs, by="question")`, `await Jev().ask_async(...)` | answers and spend for `(Question, item id, text)` pairs |
| `estimate(pairs, by)` | the expected spend, without calling the API |
| `ask_cached(jev, pairs, cache, by)` | the same, asking only what the cache lacks |

A cache is any object with `get_many(keys)` and `put_many({key: answer})`; a key is `(question hash, item id, text hash)`. `DiskCache(path)` is a SQLite file.

Vectors are numpy float32 arrays. Python lists of floats are rejected.

## File runner

Items are JSON lines: `{"id": ..., "text": ..., "title": ..., "metadata": {...}}` (`title` and `metadata` optional).

```bash
uv run constella embed items.jsonl -w ws                 # OpenAI by default; --provider voyage; or --vectors pre.npy
uv run constella overview -w ws                          # groups, counts, three 100-character samples each
uv run constella overview -w ws --group 3                # group 3 again; --pending for unlabelled items
uv run constella apply-labels labels.json -w ws          # [{"id", "path", "definition", "seed"?}, ...]
uv run constella score -w ws --sample 300                # dry run, nothing saved
uv run constella score -w ws                             # refuses above $0.50 unless --max-usd is raised
uv run constella export -w ws -o labelled.jsonl
uv run constella render -w ws                            # optional PNG map; needs `uv sync --extra viz`
```

The overview is capped at about 4,000 tokens whatever the number of items, so an agent can read it.

## Cost

Measured on 719 X posts on 2026-11-21 (`agentic_development_docs/measurements_2026-11-21_jev.md`):

- Jev bills input tokens at $0.042 per million. Each request costs about 260 tokens before its content, and the shared state is billed once per request, so the client packs many questions into each request.
- One question over many items: about 170 tokens per item at 600 characters of text.
- Label scoring, 8 candidate labels per item: about 1,000 tokens per item, $0.04 per 1,000 items.

## Tests

```bash
uv run pytest
```

## Design notes

`agentic_development_docs/project_design_plan/library_v2_notes.md` says what changed from the earlier design and why. The older plans are kept next to it.
