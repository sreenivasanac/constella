import numpy as np

from constella.jev import Answer, DiskCache, Spend
from constella.labels import Assignment, Label
from constella.loop import score_labels, tree_hints

LABELS = [
    Label("cats", "Animals/Cats", "Posts about cats."),
    Label("dogs", "Animals/Dogs", "Posts about dogs."),
    Label("cars", "Cars", "Posts about cars.", seed=True),
]
LABEL_VECTORS = np.eye(3, dtype=np.float32)
ITEMS = [
    {"id": "a", "title": "Tabby", "text": "my cat sleeps"},
    {"id": "b", "text": "my dog and my cat"},
    {"id": "c", "text": "tax forms"},
]
VECTORS = np.array([[1, 0.2, 0], [0.6, 1, 0], [0.1, 0.1, 0.2]], dtype=np.float32)


class FakeJev:
    """Answers 0.95 when the label's last path word (cat, dog, car) is in the text Jev was given."""

    model = "fake"

    def __init__(self):
        self.calls = []

    def ask(self, pairs, by="question"):
        self.calls.append((list(pairs), by))
        answers = []
        for question, _, text in pairs:
            word = question.instructions.split('"')[1].split("/")[-1].lower().rstrip("s")
            answers.append(Answer(0.95 if word in text.lower() else 0.05))
        return answers, Spend(requests=len({p[1] for p in pairs}), asked=len(pairs), input_tokens=100, usd=0.001)


def test_round_assigns_and_reports_pending(tmp_path):
    jev = FakeJev()
    result = score_labels(ITEMS, LABELS, VECTORS, LABEL_VECTORS, jev, DiskCache(tmp_path / "c.sqlite"), n=2)
    assert result.assignments["a"].primary == ("cats", 0.95) and result.assignments["a"].secondary is None
    assert result.assignments["b"].label_ids() == ["cats", "dogs"]
    assert result.pending == ["c"]
    assert result.spend.asked == 6 and result.spend.usd == 0.001  # 3 items x the 2 nearest labels
    pairs, by = jev.calls[0]
    assert by == "item" and pairs[0][2] == "Tabby\nmy cat sleeps"  # the title is part of what Jev reads
    assert "Posts about cats." in pairs[0][0].instructions


def test_second_round_asks_only_new_questions(tmp_path):
    jev, cache = FakeJev(), DiskCache(tmp_path / "c.sqlite")
    score_labels(ITEMS, LABELS, VECTORS, LABEL_VECTORS, jev, cache, n=3)
    again = score_labels(ITEMS, LABELS, VECTORS, LABEL_VECTORS, jev, cache, n=3)
    assert len(jev.calls) == 1 and again.spend.asked == 0 and again.spend.cached == 9

    # the agent adds a label and rewrites one definition: only those two labels are asked again
    labels = [LABELS[0], Label("dogs", "Animals/Dogs", "Posts about dogs and puppies."), LABELS[2], Label("tax", "Money/Tax", "Tax paperwork.")]
    vectors = np.vstack([LABEL_VECTORS, [[0, 0, 0.5]]]).astype(np.float32)
    third = score_labels(ITEMS, labels, VECTORS, vectors, jev, cache, n=4)
    asked = {(p[1], p[0].instructions.split('"')[1]) for p in jev.calls[1][0]}
    assert asked == {(i, path) for i in "abc" for path in ("Animals/Dogs", "Money/Tax")}
    assert third.pending == [] and third.assignments["c"].primary[0] == "tax"


def test_accept_changes_nothing_that_was_paid_for(tmp_path):
    jev, cache = FakeJev(), DiskCache(tmp_path / "c.sqlite")
    score_labels(ITEMS, LABELS, VECTORS, LABEL_VECTORS, jev, cache)
    strict = score_labels(ITEMS, LABELS, VECTORS, LABEL_VECTORS, jev, cache, accept=0.99)
    assert len(jev.calls) == 1 and sorted(strict.pending) == ["a", "b", "c"]


def test_no_labels_means_everything_is_pending():
    result = score_labels(ITEMS, [], VECTORS, np.zeros((0, 3), dtype=np.float32), FakeJev())
    assert result.pending == ["a", "b", "c"] and result.spend.asked == 0


def test_tree_hints_from_counts():
    def a(item, *label_ids):
        pairs = [(lid, 0.9) for lid in label_ids]
        return Assignment(item, pairs[0] if pairs else None, pairs[1] if len(pairs) > 1 else None, pairs[2:])

    labels = LABELS + [Label("pets", "Pets", "Pets."), Label("animals", "Animals", "Animals.")]
    assignments = {n: a(n, "cats", "pets", "animals") for n in range(4)}
    assignments.update({n: a(n, "pets") for n in range(4, 6)})
    assignments.update({n: a(n) for n in range(6, 20)})
    hints = tree_hints(assignments, labels)
    assert set(hints["empty"]) == {"dogs", "cars"}
    assert hints["broad"] == [("pets", 0.3)]  # 6 of 20 items
    # cats/pets and animals/pets overlap; cats/animals do not count, one is the other's ancestor
    assert [(x, y) for x, y, _, _ in hints["overlap"]] == [("animals", "pets"), ("cats", "pets")]
    assert hints["sizes"]["cats"] == 4
