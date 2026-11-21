"""A small client for TypeSafe Jev: Noul (probability of yes) and Score questions.

One call takes many (question, item) pairs and packs them into as few requests
as possible, because every request carries a fixed token overhead. Numbers in
the constants below were measured on 2026-11-21, see
``agentic_development_docs/measurements_2026-11-21_jev.md``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import httpx

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-1.14.0"  # pinned: an alias such as jev-latest moves and answers change with it
API_KEY_ENV = "TYPESAFE_API_KEY"
USD_PER_TOKEN = 0.042 / 1e6  # input tokens; output is free
REQUEST_OVERHEAD_TOKENS = 260  # billed once per request, whatever it holds (measured)
QUESTION_OVERHEAD_TOKENS = 8  # per question, on top of its own text (measured)
CHARS_PER_TOKEN = 4.0  # English prose (measured on X posts); used only for estimates
ITEMS_PER_REQUEST = 100  # one question per item, as TypeSafe's search cookbook packs them
MAX_REQUEST_TOKENS = 48_000  # the API allows 64,000 per request; estimates are rough, so stay under
CONCURRENCY = 8
TIMEOUT_S = 120.0
MAX_RETRIES = 5
BACKOFF_FIRST_S = 0.5
BACKOFF_MAX_S = 30.0
RETRY_STATUS = {429, 500, 502, 503, 504, 529}  # rate limit, overload and server errors

# Shape "question": the question is the state, each item is one Noul/Score whose instructions hold the item.
ITEM_FRAME = {
    "noul": "The answer to the question in the state is yes for this item.",
    "score": "Rate this item on the question in the state.",
}


@dataclass(frozen=True)
class Question:
    """A Noul (yes/no) by default; a Score when ``levels`` lists its level descriptions, low to high."""

    instructions: str
    true: str | None = None  # Noul only: what a yes means
    false: str | None = None  # Noul only: what a no means
    levels: tuple[str, ...] = ()

    @property
    def kind(self) -> str:
        return "score" if self.levels else "noul"

    def criteria(self):
        """The API's ``criteria`` field, or None when the instructions stand alone."""
        if self.levels:
            return list(self.levels)
        return {"true": self.true, "false": self.false} if self.true and self.false else None

    def key(self, model: str) -> str:
        """Changes when the model, the wording or the criteria change, so a reworded question is asked again."""
        raw = json.dumps([model, self.kind, self.instructions, self.criteria()])
        return hashlib.sha256(raw.encode()).hexdigest()[:24]


class Answer(NamedTuple):
    value: float  # Noul: probability of yes. Score: score / top level, so both run 0 to 1.
    confidence: float | None = None  # Score only


@dataclass
class Spend:
    requests: int = 0
    input_tokens: int = 0
    usd: float = 0.0
    asked: int = 0  # questions sent
    cached: int = 0  # questions answered from the cache

    def __add__(self, other: "Spend") -> "Spend":
        return Spend(*(getattr(self, f) + getattr(other, f) for f in ("requests", "input_tokens", "usd", "asked", "cached")))


Pair = tuple[Question, object, str]  # (question, item id, the exact text Jev reads)


class JevError(RuntimeError):
    """A request failed for good. ``partial`` holds what the other requests of the call returned
    (answers with None for the failed ones, and the spend), so paid answers are not lost."""

    partial: tuple[list, Spend] | None = None


def _plan(pairs: Sequence[Pair], by: str) -> list[tuple[object, dict, list[int]]]:
    """Pack pairs into requests: (state, questions, index of each question's pair).

    by="question": one request per question and up to ITEMS_PER_REQUEST items; cheapest when
        a question is asked of many items (about text + 17 tokens per item).
    by="item": one request per item with all its questions; the item's text is billed once,
        cheapest when an item gets several long questions (label scoring).
    """
    if by not in ("question", "item"):
        raise ValueError('by must be "question" or "item"')
    buckets: dict = {}
    for i, (q, item_id, text) in enumerate(pairs):
        buckets.setdefault(q if by == "question" else (item_id, text), []).append(i)

    plans = []
    for key, idxs in buckets.items():
        chunk, tokens = [], 0
        for i in idxs:
            q, _, text = pairs[i]
            cost = len(text if by == "question" else q.instructions) / CHARS_PER_TOKEN + 40
            if chunk and (tokens + cost > MAX_REQUEST_TOKENS or (by == "question" and len(chunk) >= ITEMS_PER_REQUEST)):
                plans.append((key, chunk))
                chunk, tokens = [], 0
            chunk.append(i)
            tokens += cost
        plans.append((key, chunk))

    out = []
    for key, chunk in plans:
        questions = {}
        for n, i in enumerate(chunk):
            q, _, text = pairs[i]
            body = {"type": q.kind}
            if by == "question":
                body["instructions"] = {"statement" if q.kind == "noul" else "task": ITEM_FRAME[q.kind], "item": text}
            else:
                body["instructions"] = q.instructions
            if q.criteria() and (by == "item" or q.kind == "score"):
                body["criteria"] = q.criteria()  # a Noul's criteria travel in the state when by="question"
            questions[f"q{n}"] = body
        if by == "question":
            state = key.instructions
            if key.kind == "noul" and key.criteria():
                state = {"question": key.instructions, "yes_means": key.true, "no_means": key.false}
        else:
            state = key[1]
        out.append((state, questions, chunk))
    return out


def estimate(pairs: Sequence[Pair], by: str = "question") -> Spend:
    """What a call would cost, from character counts. No network. Within about 10% on English text (measured)."""
    spend = Spend()
    for state, questions, chunk in _plan(pairs, by):
        chars = len(json.dumps(state, ensure_ascii=False)) + sum(len(json.dumps(q, ensure_ascii=False)) for q in questions.values())
        spend.requests += 1
        spend.asked += len(chunk)
        spend.input_tokens += int(REQUEST_OVERHEAD_TOKENS + chars / CHARS_PER_TOKEN + QUESTION_OVERHEAD_TOKENS * len(chunk))
    spend.usd = spend.input_tokens * USD_PER_TOKEN
    return spend


def _read(answer: dict) -> Answer:
    if answer["type"] == "noul":
        return Answer(float(answer["noul"]))
    top = max(1, len(answer["probabilities"]) - 1)
    return Answer(float(answer["score"]) / top, float(answer["confidence"]))


class Jev:
    """Synchronous (``ask``) and asynchronous (``ask_async``) access to Jev. No cache; see ``ask_cached``."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = JEV_MODEL,
        concurrency: int = CONCURRENCY,
        timeout: float = TIMEOUT_S,
        transport: httpx.AsyncBaseTransport | None = None,  # tests pass httpx.MockTransport
    ):
        self._key = api_key or os.environ.get(API_KEY_ENV)
        if not self._key:
            raise JevError(f"{API_KEY_ENV} is not set")
        self.model = model
        self.concurrency = concurrency
        self.timeout = timeout
        self._transport = transport

    async def _post(self, client: httpx.AsyncClient, state, questions: dict) -> dict:
        """One request with retries. Waits as long as Retry-After says, else doubles the wait with jitter."""
        body = {"model": self.model, "state": state, "questions": questions}
        last = ""
        for attempt in range(MAX_RETRIES + 1):
            try:
                r = await client.post(JEV_URL, json=body)
            except httpx.TransportError as ex:
                last = type(ex).__name__
            else:
                if r.status_code == 200:
                    return r.json()
                last = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code not in RETRY_STATUS:
                    break
                retry_after = r.headers.get("retry-after", "")
                if retry_after.replace(".", "", 1).isdigit():
                    await asyncio.sleep(min(BACKOFF_MAX_S, float(retry_after)))
                    continue
            if attempt < MAX_RETRIES:
                await asyncio.sleep(min(BACKOFF_MAX_S, BACKOFF_FIRST_S * 2**attempt) * (0.5 + random.random()))
        raise JevError(f"Jev request failed: {last}")

    async def ask_async(self, pairs: Sequence[Pair], by: str = "question") -> tuple[list[Answer], Spend]:
        """Answer every (question, item id, text) pair. Returns answers in the order given, and the spend."""
        answers: list = [None] * len(pairs)
        spend = Spend()
        gate = asyncio.Semaphore(self.concurrency)
        headers = {"Authorization": f"Bearer {self._key}"}

        async def run(state, questions, chunk):
            async with gate:
                data = await self._post(client, state, questions)
            for n, i in enumerate(chunk):
                answers[i] = _read(data["answers"][f"q{n}"])
            tokens = int(data.get("usage", {}).get("input_tokens", 0))
            spend.requests += 1
            spend.asked += len(chunk)
            spend.input_tokens += tokens
            spend.usd += tokens * USD_PER_TOKEN

        async with httpx.AsyncClient(headers=headers, timeout=self.timeout, transport=self._transport) as client:
            results = await asyncio.gather(*(run(*p) for p in _plan(pairs, by)), return_exceptions=True)
        errors = [r for r in results if isinstance(r, BaseException)]
        if errors:
            err = errors[0] if isinstance(errors[0], JevError) else JevError(repr(errors[0]))
            err.partial = (answers, spend)
            raise err
        return answers, spend

    def ask(self, pairs: Sequence[Pair], by: str = "question") -> tuple[list[Answer], Spend]:
        """``ask_async`` for callers without an event loop."""
        return asyncio.run(self.ask_async(pairs, by))


# --- answer cache -------------------------------------------------------------------------

CacheKey = tuple[str, str, str]  # (question hash, item id, text hash)


def cache_key(question: Question, item_id, text: str, model: str) -> CacheKey:
    return question.key(model), str(item_id), hashlib.sha256(text.encode()).hexdigest()[:24]


class DiskCache:
    """Answers in one SQLite file. A host may pass any object with these two methods instead."""

    def __init__(self, path):
        self._db = sqlite3.connect(str(path))
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS answers (q TEXT, item TEXT, text TEXT, value REAL, confidence REAL,"
            " PRIMARY KEY (q, item, text))"
        )

    def get_many(self, keys: Sequence[CacheKey]) -> dict[CacheKey, Answer]:
        found = {}
        for key in keys:
            row = self._db.execute("SELECT value, confidence FROM answers WHERE q=? AND item=? AND text=?", key).fetchone()
            if row:
                found[key] = Answer(*row)
        return found

    def put_many(self, answers: dict[CacheKey, Answer]) -> None:
        with self._db:
            self._db.executemany("INSERT OR REPLACE INTO answers VALUES (?,?,?,?,?)", [(*k, *a) for k, a in answers.items()])


def ask_cached(jev, pairs: Sequence[Pair], cache=None, by: str = "question") -> tuple[list[Answer], Spend]:
    """Ask only what the cache does not hold, and store the new answers.

    ``jev`` is anything with ``ask(pairs, by)`` and a ``model`` attribute. If a request
    fails, the answers that did arrive are stored before the error is raised again.
    """
    if cache is None:
        return jev.ask(pairs, by)
    keys = [cache_key(q, item_id, text, jev.model) for q, item_id, text in pairs]
    known = cache.get_many(list(dict.fromkeys(keys)))
    todo, seen = [], set()  # the same pair given twice is asked once
    for i, k in enumerate(keys):
        if k not in known and k not in seen:
            seen.add(k)
            todo.append(i)
    spend = Spend()
    if todo:
        try:
            new, spend = jev.ask([pairs[i] for i in todo], by)
        except JevError as err:
            if err.partial:
                cache.put_many({keys[i]: a for i, a in zip(todo, err.partial[0]) if a is not None})
            raise
        fresh = {keys[i]: a for i, a in zip(todo, new)}
        cache.put_many(fresh)
        known.update(fresh)
    spend.cached = len(pairs) - len(todo)
    return [known[k] for k in keys], spend
