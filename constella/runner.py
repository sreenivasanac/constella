"""File runner: the library over a JSONL file of {id, text, title?, metadata?} and a workspace folder.

    constella embed items.jsonl -w ws        # or: --vectors precomputed.npy
    constella overview -w ws                 # groups with counts and samples
    constella apply-labels labels.json -w ws
    constella score -w ws
    constella export -w ws -o labelled.jsonl

Workspace files: items.jsonl, vectors.npy, config.json, labels.json, label_vectors.npy,
jev_cache.sqlite, assignments.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

import httpx
import numpy as np

from constella.groups import group, nearest_groups
from constella.jev import DiskCache, Jev, estimate
from constella.labels import ACCEPT, SHORTLIST, Label, shortlist
from constella.loop import item_text, label_question, score_labels, tree_hints

# provider -> (endpoint, key variable, default model). Both speak the OpenAI embeddings format.
EMBED_PROVIDERS = {
    "openai": ("https://api.openai.com/v1/embeddings", "OPENAI_API_KEY", "text-embedding-3-small"),
    "voyage": ("https://api.voyageai.com/v1/embeddings", "VOYAGE_API_KEY", "voyage-4-lite"),
}
EMBED_TEXT_CHARS = 2000  # about 500 tokens of title + text
EMBED_BATCH = 100
OVERVIEW_TOKENS = 4000
OVERVIEW_SAMPLES = 3
SAMPLE_CHARS = 100
CHARS_PER_TOKEN = 4
MAX_USD = 0.50  # `score` refuses to spend more than this without --max-usd


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path, rows) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def embed_texts(texts: list[str], provider: str, model: str) -> np.ndarray:
    """Embed texts through an OpenAI-compatible endpoint, EMBED_BATCH at a time."""
    url, key_env, _ = EMBED_PROVIDERS[provider]
    key = os.environ.get(key_env)
    if not key:
        sys.exit(f"{key_env} is not set")
    rows = []
    with httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=120, transport=httpx.HTTPTransport(retries=3)) as client:
        for start in range(0, len(texts), EMBED_BATCH):
            batch = [t or " " for t in texts[start : start + EMBED_BATCH]]
            r = client.post(url, json={"model": model, "input": batch})
            r.raise_for_status()
            rows += [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])]
    return np.asarray(rows, dtype=np.float32)


def load(ws: Path):
    items = read_jsonl(ws / "items.jsonl")
    vectors = np.load(ws / "vectors.npy")
    if len(items) != len(vectors):
        sys.exit(f"{len(items)} items but {len(vectors)} vectors in {ws}")
    return items, vectors


def load_labels(ws: Path):
    if not (ws / "labels.json").exists():
        return [], None
    labels = [Label(**row) for row in json.loads((ws / "labels.json").read_text())]
    return labels, np.load(ws / "label_vectors.npy")


def cmd_embed(args):
    ws = Path(args.workspace)
    ws.mkdir(parents=True, exist_ok=True)
    items = read_jsonl(args.items)
    model = args.model or EMBED_PROVIDERS[args.provider][2]
    if args.vectors:
        vectors = np.load(args.vectors).astype(np.float32)
        if len(vectors) != len(items):
            sys.exit(f"{len(items)} items but {len(vectors)} rows in {args.vectors}")
    else:
        vectors = embed_texts([item_text(i)[:EMBED_TEXT_CHARS] for i in items], args.provider, model)
    shutil.copyfile(args.items, ws / "items.jsonl")
    np.save(ws / "vectors.npy", vectors)
    # label definitions must later be embedded with the same model as the items
    (ws / "config.json").write_text(json.dumps({"provider": args.provider, "model": model}))
    print(f"{len(items)} items, vectors {vectors.shape}, model {model} -> {ws}")


def select(ws: Path, items, vectors, path: str | None, pending: bool):
    """Narrow to the pending items and/or walk down a dotted group path such as "3.1"."""
    keep = np.arange(len(items))
    if pending:
        waiting = {r["id"] for r in read_jsonl(ws / "assignments.jsonl") if not r["primary"]}
        keep = np.array([i for i in keep if items[i]["id"] in waiting], dtype=int)
    for step in (path or "").split("."):
        if step:
            members = set(group(vectors[keep], keep.tolist())[int(step)].ids)
            keep = np.array([i for i in keep if i in members], dtype=int)
    return keep


def cmd_overview(args):
    ws = Path(args.workspace)
    items, vectors = load(ws)
    labels, label_vectors = load_labels(ws)
    keep = select(ws, items, vectors, args.group, args.pending)
    groups = group(vectors[keep], keep.tolist())
    near = nearest_groups(groups)
    prefix = f"{args.group}." if args.group else ""

    def render(n_samples: int, n_groups: int) -> str:
        lines = [f"{len(keep)} items in {len(groups)} groups (tight = mean cosine to the group's centre)"]
        for g in groups[:n_groups]:
            head = f"g{prefix}{g.id}  n={g.size}  tight={g.cohesion:.2f}  near=" + ",".join(f"g{prefix}{j}" for j, _ in near[g.id])
            if labels:
                head += "  label=" + labels[int(shortlist(g.centroid[None, :], label_vectors, 1)[0, 0])].path
            lines.append(head)
            for row in (g.central + g.spread)[:n_samples]:
                lines.append("  - " + " ".join(item_text(items[row]).split())[:SAMPLE_CHARS])
        if n_groups < len(groups):
            lines.append(f"({len(groups) - n_groups} smaller groups not shown)")
        return "\n".join(lines)

    # the cap is enforced: drop samples first, then the smallest groups
    tries = [(s, len(groups)) for s in range(args.samples, -1, -1)] + [(0, n) for n in range(len(groups) - 1, 0, -1)]
    for n_samples, n_groups in tries:
        text = render(n_samples, n_groups)
        if len(text) / CHARS_PER_TOKEN <= OVERVIEW_TOKENS:
            break
    print(text)


def cmd_apply_labels(args):
    """Replace the workspace's labels with a JSON list of {id, path, definition, seed?} and embed the definitions."""
    ws = Path(args.workspace)
    rows = json.loads(Path(args.labels).read_text())
    labels = [Label(**row) for row in rows]
    if len({label.id for label in labels}) != len(labels):
        sys.exit("label ids must be unique")
    if args.vectors:
        label_vectors = np.load(args.vectors).astype(np.float32)
    else:
        config = json.loads((ws / "config.json").read_text())
        label_vectors = embed_texts([f"{label.path}: {label.definition}" for label in labels], config["provider"], config["model"])
    if len(label_vectors) != len(labels):
        sys.exit(f"{len(labels)} labels but {len(label_vectors)} label vectors")
    (ws / "labels.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    np.save(ws / "label_vectors.npy", label_vectors)
    _, vectors = load(ws)
    sizes = np.bincount(shortlist(vectors, label_vectors, 1)[:, 0], minlength=len(labels))
    for label, size in zip(labels, sizes):
        print(f"{label.id}  {label.path}  est_items={size}")


def cmd_score(args):
    ws = Path(args.workspace)
    items, vectors = load(ws)
    labels, label_vectors = load_labels(ws)
    if not labels:
        sys.exit("no labels yet: run apply-labels first")
    rows = list(range(len(items)))
    if args.sample:  # a dry run on a random sample; nothing is written
        rows = sorted(random.Random(args.seed).sample(rows, min(args.sample, len(rows))))
    items, vectors = [items[i] for i in rows], vectors[rows]

    # worst case: nothing is cached yet
    questions = [label_question(label) for label in labels]
    worst = estimate([(questions[j], item["id"], item_text(item)) for item, near in zip(items, shortlist(vectors, label_vectors, SHORTLIST)) for j in near], by="item")
    if worst.usd > args.max_usd:
        sys.exit(f"estimated up to ${worst.usd:.2f} for {worst.asked} questions; pass --max-usd to allow it")

    result = score_labels(items, labels, vectors, label_vectors, Jev(), DiskCache(ws / "jev_cache.sqlite"), accept=args.accept)
    if not args.sample:
        write_jsonl(ws / "assignments.jsonl", [
            {"id": a.item_id, "primary": a.primary, "secondary": a.secondary, "others": a.others, "doubtful": a.doubtful}
            for a in result.assignments.values()])
    best = [max(s.values()) for s in result.scores.values()]
    hints = tree_hints(result.assignments, labels)
    path = {label.id: label.path for label in labels}
    print(f"items={len(items)} placed={len(items) - len(result.pending)} pending={len(result.pending)}"
          f" doubtful={sum(bool(a.doubtful) for a in result.assignments.values())}"
          f" multi_label={sum(a.secondary is not None for a in result.assignments.values())}" + (" (sample, not saved)" if args.sample else ""))
    print("best p histogram (0.0-0.1 … 0.9-1.0):", np.histogram(best, bins=10, range=(0, 1))[0].tolist())
    print("by label:", ", ".join(f"{path[lid]}={n}" for lid, n in sorted(hints["sizes"].items(), key=lambda x: -x[1])[:15]))
    for kind in ("broad", "overlap", "empty"):
        if hints[kind]:
            print(f"{kind}:", hints[kind])
    s = result.spend
    print(f"asked={s.asked} cached={s.cached} requests={s.requests} input_tokens={s.input_tokens} usd={s.usd:.4f}")


def cmd_export(args):
    """Items with their accepted label paths, one JSON object per line."""
    ws = Path(args.workspace)
    labels, _ = load_labels(ws)
    path = {label.id: label.path for label in labels}
    assigned = {r["id"]: r for r in read_jsonl(ws / "assignments.jsonl")}
    out = []
    for item in read_jsonl(ws / "items.jsonl"):
        a = assigned.get(item["id"]) or {"primary": None, "secondary": None, "others": []}
        out.append({
            "id": item["id"], "title": item.get("title"),
            "primary": path[a["primary"][0]] if a["primary"] else None,
            "p": a["primary"][1] if a["primary"] else None,
            "secondary": path[a["secondary"][0]] if a["secondary"] else None,
            "others": [path[lid] for lid, _ in a["others"]],
        })
    if args.output:
        write_jsonl(args.output, out)
        print(f"{len(out)} rows -> {args.output}")
    else:
        for row in out:
            print(json.dumps(row, ensure_ascii=False))


def cmd_render(args):
    """Optional 2-D map of the groups as a PNG. Needs the `viz` extra (umap-learn, matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import umap
    except ImportError:
        sys.exit("render needs the viz extra: uv sync --extra viz")
    ws = Path(args.workspace)
    items, vectors = load(ws)
    groups = group(vectors, list(range(len(items))))
    colour = np.empty(len(items), dtype=int)
    for g in groups:
        colour[g.ids] = g.id
    xy = umap.UMAP(random_state=0).fit_transform(vectors)
    plt.figure(figsize=(10, 8))
    plt.scatter(xy[:, 0], xy[:, 1], c=colour, cmap="gist_ncar", s=6)
    for g in groups:
        plt.annotate(f"g{g.id} ({g.size})", xy[g.ids].mean(axis=0), fontsize=7)
    plt.axis("off")
    out = args.output or str(ws / "map.png")
    plt.savefig(out, dpi=100, bbox_inches="tight")
    print(out)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="constella", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name, fn, **kw):
        p = sub.add_parser(name, help=fn.__doc__ or kw.get("help"))
        p.add_argument("-w", "--workspace", required=True)
        p.set_defaults(fn=fn)
        return p

    p = command("embed", cmd_embed, help="copy items into the workspace and embed them")
    p.add_argument("items")
    p.add_argument("--vectors", help="precomputed .npy, one row per item, instead of calling an embedding API")
    p.add_argument("--provider", choices=EMBED_PROVIDERS, default="openai")
    p.add_argument("--model")

    p = command("overview", cmd_overview, help="groups with counts and samples, capped at about 4,000 tokens")
    p.add_argument("--group", help='group a group again: "3", or "3.1" for a third level')
    p.add_argument("--pending", action="store_true", help="only items the last score left without a label")
    p.add_argument("--samples", type=int, default=OVERVIEW_SAMPLES)

    p = command("apply-labels", cmd_apply_labels)
    p.add_argument("labels")
    p.add_argument("--vectors", help="precomputed .npy of label definition vectors")

    p = command("score", cmd_score, help="ask Jev about each item's nearest labels; only unanswered questions are sent")
    p.add_argument("--sample", type=int, help="dry run on this many random items")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--accept", type=float, default=ACCEPT)
    p.add_argument("--max-usd", type=float, default=MAX_USD)

    p = command("export", cmd_export)
    p.add_argument("-o", "--output")

    p = command("render", cmd_render)
    p.add_argument("-o", "--output")

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
