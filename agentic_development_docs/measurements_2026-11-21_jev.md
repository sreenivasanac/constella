# Jev measurements, 2026-11-21

Real calls to `POST https://api.typesafe.ai/v1/systemone`, model `jev-1.14.0`. Data: 719 posts extracted from the raw X GraphQL captures in Kosha (`data/captures/x/*.json`); median 292 characters, 168 over 600 characters, 28 over 2,000. The extraction and probe scripts were throwaway and are not in either repo.

**Spent: $0.0624 of the $0.40 budget** (1,483,982 input tokens: $0.0320 on the probes below, $0.0304 on a full label-scoring run through the file runner). Embedding the 719 posts for that run cost about $0.002 on OpenAI.

## Summary

| Question | Answer |
| --- | --- |
| a. Packing | One request, the question as `state`, one Noul per item with the item text inside `instructions`. No limit on the number of items; the limit is 64,000 tokens per request. 719 items in one request worked (61,319 tokens, 0.57 s); 1,438 returned HTTP 400 `max_tokens_exceeded`. A 100-item request takes 0.3 to 0.5 s. |
| b. Billing | The API reports `usage.input_tokens`. **The state is billed once per request**, not once per question. Every request also carries a **fixed overhead of about 260 tokens**. Packing 50 items per request instead of 1 cost 2.2 times less per item at 150 words and 5 times less at 30 words. |
| c. Text length | The first 500 tokens (2,000 characters) give the same answers as the full text (99 to 100 of 100 agree at 0.8). The first 150 tokens never add a false yes but miss some: 0 of 13, 4 of 8 and 8 of 33 on the three questions (the sample is two-thirds long posts). A rule-built summary misses 21 of 33 on a question about what the post offers. |
| d. Calibration | For "about AI coding agents", 719 posts: 54% under 0.2, 40% between 0.2 and 0.8, 6% at 0.8 or above. With the item as the state (100 posts): 70%, 22%, 8%. |

## a. Request shape, limits, latency

Shape `by="question"` (what `constella.jev` sends when one question is asked of many items; TypeSafe's "infinite cardinality" cookbook uses the same layout):

```json
POST https://api.typesafe.ai/v1/systemone
Authorization: Bearer <TYPESAFE_API_KEY, redacted>
Content-Type: application/json

{
  "model": "jev-1.14.0",
  "state": "Is this post about building software with AI coding agents (for example Claude Code, Codex or Cursor)?",
  "questions": {
    "q0": {"type": "noul",
           "instructions": {"statement": "The answer to the question in the state is yes for this item.",
                            "item": "<text of item 0>"}},
    "q1": {"type": "noul",
           "instructions": {"statement": "The answer to the question in the state is yes for this item.",
                            "item": "<text of item 1>"}}
  }
}
```

Response:

```json
{"model": "jev-1.14.0",
 "answers": {"q0": {"type": "noul", "noul": 0.93}, "q1": {"type": "noul", "noul": 0.04}},
 "usage": {"input_tokens": 392, "output_tokens": …}}
```

`output_tokens` is reported at 17 to 21 per Noul and is not billed.

When the Noul has `true`/`false` criteria, the state becomes `{"question": …, "yes_means": …, "no_means": …}`. A Score sends its levels as `criteria` on every per-item question and `{"task": "Rate this item on the question in the state.", "item": …}` as instructions.

Shape `by="item"` (label scoring: one item, several questions):

```json
{
  "model": "jev-1.14.0",
  "state": "<title and the first 2,000 characters of the item>",
  "questions": {
    "q0": {"type": "noul",
           "instructions": "Does this item belong under \"AI/Coding agents\"? Definition: Building software with AI coding agents such as Claude Code, Codex, Cursor or Grok Bot: tips, skills, workflows and harnesses.",
           "criteria": {"true": "A reasonable person would look for this item under that path.",
                        "false": "The path does not describe this item."}},
    "q1": {"type": "noul", "instructions": "Does this item belong under \"Robotics\"? Definition: …", "criteria": {"true": "…", "false": "…"}}
  }
}
```

Limits and latency (all measured):

| Request | Input tokens | Wall time |
| --- | --- | --- |
| 1 Noul, short state | 281 | 0.12 to 0.25 s |
| 50 Nouls of 150 words | 11,112 | 0.19 s |
| 100 posts, up to 2,000 characters each | 31,792 | 0.31 to 0.48 s |
| 300 posts, 200 characters each | 26,504 | 0.38 s |
| 719 posts, 200 characters each | 61,319 | 0.57 s |
| 1,438 posts, 200 characters each | – | HTTP 400 `{"detail":{"error_type":"max_tokens_exceeded"}}` |
| 719 posts as 8 requests in flight at once | 122,434 | 0.39 s for all |
| 300 one-item requests (8 Nouls each), 8 in flight | 301,379 | about 3 s |

No 429 or 529 was returned at any point (about 1,130 requests, at most 8 in flight). The one response whose headers were inspected carried no rate-limit or retry header.

The client packs 100 items per request and keeps its own estimate under 48,000 tokens per request, because the estimate is rough and the hard limit is 64,000.

## b. Billing

Synthetic words, so the only thing that changes between rows is the thing being measured.

State = the item, k Nouls of about 14 words each:

| State | 1 question | 8 questions | 16 questions |
| --- | --- | --- | --- |
| 1 word | 281 | 428 | 602 |
| 30 words | 319 | 466 | 640 |
| 150 words | 475 | 622 | 796 |
| 500 words | 930 | 1,077 | 1,251 |

Adding 499 words of state adds 649 tokens whether the request holds 1, 8 or 16 questions. So the state is billed once. Each question adds about 21 tokens (its own 13 or so plus about 8). What is left with a one-word state and one question is about 260 tokens: a fixed charge per request.

State = the question (10 words), n Nouls each holding an item:

| Item | n = 1 | n = 2 | n = 10 | n = 50 | per item at 50 |
| --- | --- | --- | --- | --- | --- |
| 30 words | 332 | 392 | 872 | 3,312 | 66 |
| 150 words | 488 | 704 | 2,432 | 11,112 | 222 |

A 300-word state instead of a 10-word one added 377 tokens to every one of these requests, once.

Model that fits all of the above:

```text
input tokens = 260 + tokens(state) + sum over questions of (tokens(question) + 8)
```

No cost field is returned; dollars are `input_tokens × $0.042 / 1M`.

What it means:

- This settles the open question in v4 §6.1. v4's "expected" column (state billed once) is the right one.
- v4's 942 tokens for 8 Nouls on a 15-word state is the 260-token request charge plus eight label questions with path, definition and criteria.
- One question per request is the expensive way to use Jev. One question over 100 posts (first 600 characters) measured 167 tokens per post packed; sent one post per request the model above gives about 420.
- For label scoring, measured on all 719 posts with 12 labels and 8 questions per post: **1,005 tokens per item, $0.042 per 1,000 items**. The same pairs packed by label instead are estimated at 1,327 tokens per item, because each post's text is then sent 8 times. Packing by label wins only when the item text is shorter than the label question.
- `constella.jev.estimate` was 2 to 3% under the bill for `by="question"` and 11% over for `by="item"`.

## c. Text length

100 posts: 34 of at most 600 characters, 38 between 600 and 2,000, all 28 over 2,000. Long posts are over-represented on purpose (they are 23% of the 719), since short posts read the same in every variant except the summary. Each post is the state of its own request with three Nouls:

- **agents**: "Is this post about building software with AI coding agents (for example Claude Code, Codex or Cursor)?"
- **robot**: "Is this post about robotics, electronics or other physical hardware projects?"
- **oss**: "Does this post share an open-source project, free tool or free learning resource that the reader can use?"

Variants: first 600 characters (about 150 tokens), first 2,000 (about 500 tokens), the full text, and a rule-built one-liner (URLs removed, first two sentences, at most 240 characters; posts have no title).

Yes = p ≥ 0.8. There is no hand-labelled answer key; the full text is the reference.

| Question | Yes on full text | 500 tokens | 150 tokens | Summary |
| --- | --- | --- | --- | --- |
| agents | 8 | 8 (100 agree) | 4, misses 4 (96 agree) | 6, misses 3, adds 1 (96 agree) |
| robot | 13 | 13 (100 agree) | 13 (100 agree) | 11, misses 2 (98 agree) |
| oss | 33 | 32, misses 1 (99 agree) | 25, misses 8 (92 agree) | 12, misses 21 (79 agree) |

Mean absolute difference in p against the full text: 500 tokens 0.003 to 0.013; 150 tokens 0.016 to 0.074; summary 0.040 to 0.194.

Mean input tokens per request (3 questions): summary 377, 150 tokens 465, 500 tokens 616, full 734.

What it means:

- 500 tokens is as good as the whole post. Nothing longer is worth sending.
- 150 tokens never produced a false yes; it only misses, and on long posts it misses a lot. That is the behaviour v4's "second look at 500 tokens" was built for. `score_labels` now sends 2,000 characters from the start: for the 77% of posts under 600 characters it costs the same, and it removes a step.
- A summary made by rule is not a substitute for the text. It drops whatever the post says after its opening, and the `oss` question is about exactly that. Whether an LLM-written summary does better was not measured.

## d. Calibration

"agents" question, first 2,000 characters.

| Band | 719 posts, `by="question"` | 100 posts, `by="item"` |
| --- | --- | --- |
| 0 to 0.2 | 391 (54%) | 70 |
| 0.2 to 0.8 | 285 (40%); 214 in 0.2 to 0.5, 71 in 0.5 to 0.8 | 22 |
| 0.8 to 1.0 | 43 (6%); 13 at 0.9 or above | 8 |

Read by eye, as data only: the 22 highest-scoring posts in the 0.8 band are about coding agents with one or two arguable cases; the 0.5 to 0.8 band is mixed (things built with an AI model but not about the tooling); posts that are only a link sit near 0.42, which is the honest answer for "cannot tell".

The other two questions with the item as the state (100 posts): robot 77 / 10 / 13, oss 35 / 33 / 32.

## Also measured: the two shapes do not give identical answers

The same 100 posts and questions, first 2,000 characters, `by="question"` against `by="item"`:

| Question | Agree at 0.8 | Mean absolute difference | Yes: by question / by item |
| --- | --- | --- | --- |
| agents | 97 | 0.108 | 7 / 8 |
| robot | 97 | 0.044 | 10 / 13 |
| oss | 92 | 0.092 | 32 / 32 |

`by="question"` is more cautious: its probabilities sit closer to the middle. In the 14 disagreements read by eye, the item-as-state answer looked right more often for agents and robot (for example a post about brainstorming in Claude Code: 0.95 against 0.60), and it was mixed for oss. A third layout, all items in the state with questions pointing at them by key, agreed with item-as-state on 92 to 99 of 100 and cost about the same as `by="question"`, so it was not kept.

Consequence: `score_labels` uses `by="item"` (also the shape v4's 0.8 cut-off was measured with). `by="question"` is the default for one question over many items, where it costs about 2.5 times less; a cut-off tuned on one shape should be checked before it is used on the other.

A Score question works in both shapes; on the 5 posts compared, `score / top level` came back within 0.03 and `confidence` within 0.08.

## The full run through the file runner

`constella embed` → `overview` → `apply-labels` (12 labels written after reading the overview) → `score`:

- The overview of 719 posts: 40 groups, 3 samples each, 13,861 characters (about 3,470 tokens).
- `score --sample 300`: 300 requests, 2,400 questions, 301,379 tokens, $0.0127, about 3 s.
- `score`: the other 419 items, 3,352 questions, 421,108 tokens, $0.0177; the 2,400 answers from the sample came from the cache.
- `score` again: 0 requests, 5,752 answers from the cache.
- 387 of 719 placed at 0.8 with these first-draft labels (54%), 115 with a second label, 177 with a label in the doubtful band. No pending round was run.

## Not measured

- Accuracy against hand labels. Everything here is agreement between variants.
- An LLM-written summary as the text Jev reads.
- The organisation's rate limits: nothing was hit at 8 requests in flight.
