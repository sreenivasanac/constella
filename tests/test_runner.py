import json

import numpy as np

from constella import runner


def workspace(tmp_path, n=2000):
    rng = np.random.default_rng(1)
    items = tmp_path / "items.jsonl"
    items.write_text("".join(json.dumps({"id": i, "title": f"Item {i}", "text": "word " * 80}) + "\n" for i in range(n)))
    np.save(tmp_path / "pre.npy", rng.standard_normal((n, 12)).astype(np.float32))
    ws = tmp_path / "ws"
    runner.main(["embed", str(items), "-w", str(ws), "--vectors", str(tmp_path / "pre.npy")])
    return ws


def test_overview_stays_under_the_token_cap(tmp_path, capsys):
    ws = workspace(tmp_path)
    capsys.readouterr()
    runner.main(["overview", "-w", str(ws)])
    text = capsys.readouterr().out
    assert text.startswith("2000 items in 40 groups")
    assert len(text) / runner.CHARS_PER_TOKEN <= runner.OVERVIEW_TOKENS
    assert "g0  n=" in text and "  - Item " in text


def test_overview_of_a_group_and_labels_and_export(tmp_path, capsys):
    ws = workspace(tmp_path, n=400)
    runner.main(["overview", "-w", str(ws), "--group", "0"])
    assert "g0.0  n=" in capsys.readouterr().out

    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps([{"id": "x", "path": "X", "definition": "x things"}, {"id": "y", "path": "Y", "definition": "y things", "seed": True}]))
    np.save(tmp_path / "lv.npy", np.eye(2, 12, dtype=np.float32))
    runner.main(["apply-labels", str(labels), "-w", str(ws), "--vectors", str(tmp_path / "lv.npy")])
    assert "est_items=" in capsys.readouterr().out

    runner.write_jsonl(ws / "assignments.jsonl", [{"id": 0, "primary": ["x", 0.9], "secondary": None, "others": [["y", 0.8]], "doubtful": []}])
    runner.main(["export", "-w", str(ws)])
    first = json.loads(capsys.readouterr().out.splitlines()[0])
    assert first == {"id": 0, "title": "Item 0", "primary": "X", "p": 0.9, "secondary": None, "others": ["Y"]}
