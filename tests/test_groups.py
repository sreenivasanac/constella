import numpy as np
import pytest

from constella.groups import choose_k, group, nearest_groups


def test_every_item_in_exactly_one_group(blobs):
    vectors, ids = blobs
    groups = group(vectors, ids, k=3)
    assert sorted(i for g in groups for i in g.ids) == sorted(ids)
    assert [g.size for g in groups] == [40, 40, 40]
    assert all(0.9 < g.cohesion <= 1.0 for g in groups)
    assert all(abs(np.linalg.norm(g.centroid) - 1) < 1e-5 for g in groups)


def test_deterministic(blobs):
    vectors, ids = blobs
    a, b = group(vectors, ids), group(vectors, ids)
    assert [(g.id, g.ids, g.central, g.spread) for g in a] == [(g.id, g.ids, g.central, g.spread) for g in b]


def test_representatives_come_from_the_group(blobs):
    vectors, ids = blobs
    for g in group(vectors, ids, k=3):
        assert len(g.central) == 3 and len(g.spread) == 2
        assert set(g.central) | set(g.spread) <= set(g.ids)
        assert not set(g.central) & set(g.spread)


def test_recursion_on_a_subset(blobs):
    vectors, ids = blobs
    first = group(vectors, ids, k=3)[0]
    rows = [ids.index(i) for i in first.ids]
    second = group(vectors[rows], first.ids, k=4)
    assert sorted(i for g in second for i in g.ids) == sorted(first.ids)
    assert len(second) == 4


def test_k_comes_from_the_budget():
    assert choose_k(30_000) == 40
    assert choose_k(100) == 13
    assert choose_k(3) == 1


def test_small_inputs_and_bad_types(blobs):
    vectors, ids = blobs
    assert group(vectors[:0], []) == []
    assert group(vectors[:1], ids[:1])[0].ids == ids[:1]
    assert len(group(vectors[:2], ids[:2], k=5)) <= 2
    with pytest.raises(TypeError):
        group(vectors.tolist(), ids)


def test_nearest_groups(blobs):
    vectors, ids = blobs
    groups = group(vectors, ids, k=3)
    near = nearest_groups(groups, n=2)
    assert set(near) == {0, 1, 2}
    assert all(len(v) == 2 and g not in [j for j, _ in v] for g, v in near.items())
