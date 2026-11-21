"""One round of the label-scoring loop (design plan v4), and count-based hints about the tree."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from constella.jev import Question, Spend, ask_cached
from constella.labels import ACCEPT, PATH_SEP, SHORTLIST, Assignment, Label, assign, shortlist

JEV_TEXT_CHARS = 2000  # about 500 tokens: answers match the full text; 150 tokens missed up to half the yeses on long posts (measured)
LABEL_TRUE = "A reasonable person would look for this item under that path."
LABEL_FALSE = "The path does not describe this item."
BROAD_SHARE = 0.25  # a label holding more than this share of all items is too broad
OVERLAP_SHARE = 0.5  # two labels sharing more than this share of the smaller one are merge candidates
MIN_OVERLAP_ITEMS = 3  # below this a shared item or two is noise


@dataclass
class LabelScores:
    assignments: dict[object, Assignment]
    pending: list  # ids of items with no label at or above the cut-off
    spend: Spend
    scores: dict[object, dict[str, float]]  # every answer used, {item id: {label id: p}}


def label_question(label: Label) -> Question:
    """The yes/no question for one label. Jev reads literally, so the path and the definition are both in it."""
    return Question(f'Does this item belong under "{label.path}"? Definition: {label.definition}', LABEL_TRUE, LABEL_FALSE)


def item_text(item: Mapping) -> str:
    """What Jev reads for an item: its title and the head of its text."""
    title = item.get("title")
    return ((title + "\n" if title else "") + (item.get("text") or ""))[:JEV_TEXT_CHARS]


def score_labels(
    items: Sequence[Mapping],
    labels: Sequence[Label],
    vectors: np.ndarray,
    label_vectors: np.ndarray,
    jev,
    cache=None,
    accept: float = ACCEPT,
    n: int = SHORTLIST,
) -> LabelScores:
    """Shortlist the n nearest labels per item, ask Jev only the questions the cache
    has no answer for, and apply the assignment rule.

    ``items`` are {id, text, title?} rows aligned with ``vectors``; ``labels`` are aligned
    with ``label_vectors``. Calling it twice with the same cache costs nothing the second time.
    """
    if not len(items) or not len(labels):
        return LabelScores({i["id"]: Assignment(i["id"]) for i in items}, [i["id"] for i in items], Spend(), {})
    questions = [label_question(label) for label in labels]
    nearest = shortlist(vectors, label_vectors, n)
    pairs, where = [], []
    for row, item in enumerate(items):
        text = item_text(item)
        for j in nearest[row]:
            pairs.append((questions[j], item["id"], text))
            where.append((item["id"], labels[j].id))
    # one request per item: its text is billed once for all of its label questions
    answers, spend = ask_cached(jev, pairs, cache, by="item")
    scores: dict = {item["id"]: {} for item in items}
    for (item_id, label_id), answer in zip(where, answers):
        scores[item_id][label_id] = answer.value
    assignments = assign(scores, accept)
    return LabelScores(assignments, [i for i, a in assignments.items() if a.pending], spend, scores)


def tree_hints(assignments: Mapping[object, Assignment], labels: Sequence[Label]) -> dict:
    """What the counts say about the tree, with no model call:

    empty   label ids no item was accepted under
    broad   (label id, share of all items) above BROAD_SHARE
    overlap (label a, label b, shared items, share of the smaller) above OVERLAP_SHARE;
            a label and its own ancestor are not compared
    """
    members: dict[str, set] = {label.id: set() for label in labels}
    for item_id, a in assignments.items():
        for label_id in a.label_ids():
            members.setdefault(label_id, set()).add(item_id)
    path = {label.id: label.path for label in labels}
    total = len(assignments) or 1

    def related(a, b):
        pa, pb = path.get(a, a), path.get(b, b)
        return pa.startswith(pb + PATH_SEP) or pb.startswith(pa + PATH_SEP)

    overlap = []
    for a, b in combinations(sorted(members), 2):
        shared = len(members[a] & members[b])
        smaller = min(len(members[a]), len(members[b]))
        if shared >= MIN_OVERLAP_ITEMS and shared / smaller > OVERLAP_SHARE and not related(a, b):
            overlap.append((a, b, shared, round(shared / smaller, 2)))
    return {
        "empty": [lid for lid, m in members.items() if not m],
        "broad": [(lid, round(len(m) / total, 2)) for lid, m in members.items() if len(m) / total > BROAD_SHARE],
        "overlap": overlap,
        "sizes": {lid: len(m) for lid, m in members.items()},
    }
