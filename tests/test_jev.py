import json

import httpx
import pytest

import constella.jev as jev_module
from constella.jev import Answer, DiskCache, Jev, JevError, Question, Spend, ask_cached, cache_key, estimate

Q = Question("Is this about cats?")


def server(calls, fail_first=0, status=429):
    """A fake Jev endpoint: says yes (0.9) when the text mentions a cat."""

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if len(calls) <= fail_first:
            return httpx.Response(status, headers={"retry-after": "0"}, text="slow down")
        answers = {}
        for qid, q in body["questions"].items():
            text = json.dumps([body["state"], q["instructions"]])
            if q["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": 0.9 if "cat" in text.replace("cats?", "") else 0.1}
            else:
                answers[qid] = {"type": "score", "score": 1.5, "confidence": 0.4, "probabilities": {"0": 0.2, "1": 0.1, "2": 0.7}}
        return httpx.Response(200, json={"model": body["model"], "answers": answers, "usage": {"input_tokens": 1000, "output_tokens": 1}})

    return httpx.MockTransport(handle)


def test_packs_many_items_into_one_request_per_question():
    calls = []
    jev = Jev(api_key="k", transport=server(calls))
    pairs = [(Q, n, "a cat" if n % 2 else "a dog") for n in range(250)]
    answers, spend = jev.ask(pairs)
    assert [a.value for a in answers[:4]] == [0.1, 0.9, 0.1, 0.9]
    assert len(calls) == 3 and sorted(len(c["questions"]) for c in calls) == [50, 100, 100]
    assert calls[0]["state"] == Q.instructions and calls[0]["model"] == "jev-1.14.0"
    assert spend.requests == 3 and spend.asked == 250 and spend.input_tokens == 3000
    assert spend.usd == pytest.approx(3000 * 0.042 / 1e6)


def test_by_item_sends_the_text_once_with_all_its_questions():
    calls = []
    jev = Jev(api_key="k", transport=server(calls))
    q2 = Question("Is this about dogs?", "It is.", "It is not.")
    answers, spend = jev.ask([(Q, 1, "a cat"), (q2, 1, "a cat"), (Q, 2, "a dog")], by="item")
    assert len(calls) == 2 and spend.requests == 2
    first = next(c for c in calls if c["state"] == "a cat")
    assert len(first["questions"]) == 2
    assert {"true": "It is.", "false": "It is not."} in [q.get("criteria") for q in first["questions"].values()]
    assert [a.value for a in answers] == [0.9, 0.9, 0.1]


def test_score_is_scaled_to_0_1():
    jev = Jev(api_key="k", transport=server([]))
    (answer,), _ = jev.ask([(Question("How good?", levels=("bad", "ok", "good")), 1, "x")])
    assert answer == Answer(0.75, 0.4)


def test_retries_rate_limits_then_succeeds(monkeypatch):
    calls = []
    jev = Jev(api_key="k", transport=server(calls, fail_first=2))
    answers, spend = jev.ask([(Q, 1, "a cat")])
    assert len(calls) == 3 and answers[0].value == 0.9 and spend.requests == 1


def test_bad_request_is_not_retried():
    calls = []
    jev = Jev(api_key="k", transport=server(calls, fail_first=99, status=422))
    with pytest.raises(JevError, match="422"):
        jev.ask([(Q, 1, "a cat")])
    assert len(calls) == 1


def test_missing_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(JevError):
        Jev()


def test_estimate_counts_one_overhead_per_request():
    one = estimate([(Q, 1, "x" * 400)])
    hundred = estimate([(Q, n, "x" * 400) for n in range(100)])
    assert one.requests == hundred.requests == 1
    assert hundred.input_tokens < 100 * one.input_tokens / 2  # packing is what saves money
    assert estimate([(Q, n, "x") for n in range(100)], by="item").requests == 100


class FakeJev:
    model = "fake"

    def __init__(self):
        self.asked = []

    def ask(self, pairs, by="question"):
        self.asked.append(list(pairs))
        return [Answer(0.9)] * len(pairs), Spend(requests=1, asked=len(pairs), input_tokens=10, usd=0.01)


def test_cache_asks_only_what_is_missing(tmp_path):
    cache, jev = DiskCache(tmp_path / "c.sqlite"), FakeJev()
    pairs = [(Q, 1, "a"), (Q, 2, "b"), (Q, 1, "a")]  # one duplicate
    answers, spend = ask_cached(jev, pairs, cache)
    assert len(jev.asked[0]) == 2 and len(answers) == 3 and spend.cached == 1

    answers, spend = ask_cached(jev, pairs + [(Q, 3, "c")], DiskCache(tmp_path / "c.sqlite"))  # reopened from disk
    assert [p[1] for p in jev.asked[1]] == [3] and spend.cached == 3 and answers[0] == Answer(0.9)


def test_cache_key_changes_with_question_text_and_model():
    base = cache_key(Q, 1, "a", "m")
    assert cache_key(Question("Is this about dogs?"), 1, "a", "m") != base  # reworded question
    assert cache_key(Q, 1, "a changed", "m") != base  # changed item text
    assert cache_key(Q, 1, "a", "m2") != base  # another model
    assert cache_key(Question("Is this about cats?", "yes", "no"), 1, "a", "m") != base  # criteria added


def test_host_cache_needs_only_two_methods():
    class Host(dict):
        def get_many(self, keys):
            return {k: self[k] for k in keys if k in self}

        put_many = dict.update

    host, jev = Host(), FakeJev()
    ask_cached(jev, [(Q, 1, "a")], host)
    ask_cached(jev, [(Q, 1, "a")], host)
    assert len(jev.asked) == 1 and len(host) == 1


def test_partial_answers_survive_a_failed_request(tmp_path):
    class Half(FakeJev):
        def ask(self, pairs, by="question"):
            err = JevError("boom")
            err.partial = ([Answer(0.7), None], Spend())
            raise err

    cache = DiskCache(tmp_path / "c.sqlite")
    with pytest.raises(JevError):
        ask_cached(Half(), [(Q, 1, "a"), (Q, 2, "b")], cache)
    assert list(cache.get_many([cache_key(Q, 1, "a", "fake"), cache_key(Q, 2, "b", "fake")]).values()) == [Answer(0.7)]
