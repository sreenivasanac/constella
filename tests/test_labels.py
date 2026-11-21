import numpy as np

from constella.labels import assign, shortlist


def test_shortlist_nearest_first():
    labels = np.eye(4, dtype=np.float32)
    items = np.array([[1, 0.5, 0.1, 0], [0, 0, 0.2, 5]], dtype=np.float32)
    assert shortlist(items, labels, n=2).tolist() == [[0, 1], [3, 2]]
    assert shortlist(items, labels, n=8).shape == (2, 4)  # never more than there are labels


def test_assign_roles():
    out = assign({
        "a": {"x": 0.95, "y": 0.85, "z": 0.82, "w": 0.3},
        "b": {"x": 0.79, "y": 0.1},
        "c": {"y": 0.8},
        "d": {},
    })
    a = out["a"]
    assert a.primary == ("x", 0.95) and a.secondary == ("y", 0.85) and a.others == [("z", 0.82)]
    assert a.doubtful == ["y", "z"]  # accepted, but under 0.9
    assert not a.pending and a.label_ids() == ["x", "y", "z"]
    assert out["b"].pending and out["b"].primary is None and out["b"].doubtful == []
    assert out["c"].primary == ("y", 0.8) and out["c"].secondary is None and out["c"].doubtful == ["y"]
    assert out["d"].pending


def test_assign_cutoff_is_a_parameter():
    assert assign({"b": {"x": 0.79}}, accept=0.7)["b"].primary == ("x", 0.79)
