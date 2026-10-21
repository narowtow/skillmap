"""Weighted accounting, concurrent cooldowns, cache validation, and retries."""

from __future__ import annotations

import json
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest
import requests
from conftest import FakeClock, FakeResponse, FakeSession, payload

from skillmap import config, fetch
from skillmap.synthetic import make_sites


@pytest.fixture
def chunk():
    return fetch.Chunk("gn-test", 1.0, 2.0, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"))


def run_chunk(chunk, tmp_path, session, clock=None, **kwargs):
    clock = clock or FakeClock()
    limiter = fetch.WeightedRateLimiter(clock=clock, sleep=clock.sleep)
    gate = fetch.PauseGate(clock, clock.sleep)
    return fetch.fetch_chunk(chunk, session=session, limiter=limiter, gate=gate,
                             cache_dir=tmp_path, clock=clock, sleep=clock.sleep, jitter=lambda: 0, **kwargs)


def test_weight():
    assert fetch.request_weight(1, 365, 5) == pytest.approx(13.035714)
    assert fetch.request_weight(1, 14, 5) == 1
    assert fetch.request_weight(2, 28, 20, 2) == 16
    with pytest.raises(ValueError):
        fetch.request_weight(0, 365, 5)


def test_plan_order_and_params():
    sites = pd.concat([make_sites(), make_sites().assign(site_id=lambda x: x.site_id + "b")])
    chunks = fetch.plan_chunks(sites)
    assert [c.site_id for c in chunks[::11]][:5] == make_sites().site_id.tolist()
    assert [c.year for c in chunks[:11]] == list(range(config.START.year, config.END.year + 1))
    assert all(c.year == config.START.year for c in fetch.plan_chunks(sites, "year-major")[:10])
    assert chunks[0].params()["daily"] == ",".join(config.VARIABLES)
    assert chunks[0].params()["models"] == config.WEATHER_MODEL
    assert chunks[0].params()["wind_speed_unit"] == "ms"
    assert chunks[0].params()["timezone"] == "auto"
    assert chunks[0].params()["temperature_unit"] == "celsius"


@pytest.mark.parametrize("failure", [FakeResponse(500), requests.Timeout(), requests.ConnectionError()])
def test_retry_then_success(chunk, tmp_path, failure):
    session = FakeSession([failure, FakeResponse(body=payload(chunk.start, chunk.end))])
    clock = FakeClock()
    result = run_chunk(chunk, tmp_path, session, clock)
    assert result.status == "done" and result.attempts == 2
    assert len(session.calls) == 2 and clock.sleeps == [1]
    assert session.calls[0][1]["timeout"] == (10, 90)
    ledger = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert len(ledger) == 1 and ledger[0]["weight"] == pytest.approx(chunk.weight * 2)
    assert fetch.read_cache(fetch.cache_path(tmp_path, chunk), chunk) is not None
    assert not list(tmp_path.rglob("*.tmp"))


def test_400_not_retried(chunk, tmp_path):
    session = FakeSession([FakeResponse(400)])
    result = run_chunk(chunk, tmp_path, session)
    assert result.status == "http_400" and result.attempts == 1
    assert not fetch.cache_path(tmp_path, chunk).exists()
    assert json.loads((tmp_path / "ledger.jsonl").read_text())["status"] == "http_400"


def test_five_attempt_limit(chunk, tmp_path):
    session = FakeSession([FakeResponse(503)] * 6)
    clock = FakeClock()
    result = run_chunk(chunk, tmp_path, session, clock)
    assert result.attempts == 5 and clock.sleeps == [1, 2, 4, 8]


def test_resume_validates_and_writes_ledger_once(chunk, tmp_path):
    session = FakeSession([FakeResponse(body=payload(chunk.start, chunk.end))])
    path = fetch.cache_path(tmp_path, chunk)
    path.parent.mkdir(parents=True)
    path.write_text('{"daily":{}}')
    assert run_chunk(chunk, tmp_path, session).status == "done"
    before = (tmp_path / "ledger.jsonl").read_bytes()
    assert run_chunk(chunk, tmp_path, session).status == "cached"
    assert len(session.calls) == 1
    assert (tmp_path / "ledger.jsonl").read_bytes() == before


@pytest.mark.parametrize("bad", ["time", "variable", "length", "numeric"])
def test_invalid_payload(chunk, bad):
    body = payload(chunk.start, chunk.end)
    variable = next(iter(config.VARIABLES))
    if bad == "time":
        body["daily"]["time"] = body["daily"]["time"][::-1]
    elif bad == "variable":
        del body["daily"][variable]
    elif bad == "length":
        body["daily"][variable].pop()
    else:
        body["daily"][variable][0] = "bad"
    with pytest.raises(ValueError):
        fetch.validate_response(body, chunk)


def test_concurrent_limiter_all_windows():
    clock = FakeClock()
    caps = {"minute": 4, "hour": 10, "day": 14}
    limiter = fetch.WeightedRateLimiter(caps, clock=clock, sleep=clock.sleep)

    class RecordingDeque(deque):
        def __init__(self):
            super().__init__()
            self.history = []

        def append(self, item):
            self.history.append(item)
            super().append(item)

    recorded = RecordingDeque()
    limiter.events["day"] = recorded
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(limiter.acquire, [2.0] * 24))
    assert len(recorded.history) == 24
    for timestamp, _ in recorded.history:
        for name, cap in caps.items():
            window = fetch.WINDOW_SECONDS[name]
            total = sum(w for t, w in recorded.history if timestamp - window < t <= timestamp)
            assert total <= cap


def test_seeded_budget_and_torn_ledger(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({"fetched_at": "1970-01-01T00:00:00+00:00", "weight": 9}) + '\n{"bad')
    clock = FakeClock(1)
    limiter = fetch.WeightedRateLimiter({"hour": 10, "day": 10}, clock=clock, sleep=clock.sleep,
                                       past_usage=fetch.ledger_usage(ledger))
    limiter.acquire(2)
    assert clock() == 86400
    with pytest.raises(ValueError):
        limiter.acquire(11)


def test_429_pauses_other_threads(chunk, tmp_path):
    clock = FakeClock()
    waiting, release, other_waiting = threading.Event(), threading.Event(), threading.Event()
    sleeper_count = 0
    sleeper_lock = threading.Lock()

    def sleep(seconds):
        nonlocal sleeper_count
        with sleeper_lock:
            sleeper_count += 1
            (waiting if sleeper_count == 1 else other_waiting).set()
        assert release.wait(5)
        clock.sleep(seconds)

    session = FakeSession([FakeResponse(429, headers={"Retry-After": "7"}),
                           FakeResponse(body=payload(chunk.start, chunk.end)),
                           FakeResponse(body=payload(chunk.start, chunk.end))])
    limiter = fetch.WeightedRateLimiter(clock=clock, sleep=sleep)
    gate = fetch.PauseGate(clock, sleep)

    def work(site_id):
        other = fetch.Chunk(site_id, chunk.lat, chunk.lon, chunk.start, chunk.end)
        return fetch.fetch_chunk(other, session=session, limiter=limiter, gate=gate, cache_dir=tmp_path,
                                 clock=clock, sleep=sleep)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(work, "gn-first")
        assert waiting.wait(5)
        second = executor.submit(work, "gn-second")
        try:
            assert other_waiting.wait(5)
            assert len(session.calls) == 1  # second location cannot send while cooldown is pending
        finally:
            release.set()
        assert first.result().status == second.result().status == "done"
    assert clock() >= 7


def test_retry_after_date_and_fallback():
    assert fetch._retry_after("Thu, 01 Jan 1970 00:01:00 GMT", 1) == 59
    assert fetch._retry_after(None, 0) == 120
    assert fetch._retry_after("invalid", 0) == 120


def test_download_budget_and_resume(tmp_path):
    sites = make_sites()
    chunks = fetch.plan_chunks(sites)
    # Auto-respond using the requested dates, under a lock for concurrent workers.
    class Session(FakeSession):
        def get(self, url, **kwargs):
            with self.lock:
                self.calls.append((url, kwargs))
            params = kwargs["params"]
            return FakeResponse(body=payload(params["start_date"], params["end_date"]))

    session = Session()
    clock = FakeClock()
    limit = sum(c.weight for c in chunks[:3])
    result = fetch.download_all(sites, max_workers=4, budget=limit, cache_dir=tmp_path,
                                session=session, clock=clock, sleep=clock.sleep)
    assert sum(r.weight for r in result) <= limit
    assert len(session.calls) == 3
    assert len((tmp_path / "ledger.jsonl").read_text().splitlines()) == 3
    result = fetch.download_all(sites, max_workers=2, budget=0, cache_dir=tmp_path,
                                session=session, clock=clock, sleep=clock.sleep)
    assert sum(r.status == "cached" for r in result) == 3
    assert len(session.calls) == 3


def test_retry_cannot_overspend_budget(chunk, tmp_path):
    result = run_chunk(chunk, tmp_path, FakeSession([FakeResponse(500)]),
                       budget=fetch.DownloadBudget(chunk.weight))
    assert result.status == "budget" and result.attempts == 1


def test_uncached_benchmark_fetch_writes_nothing(chunk, tmp_path):
    session = FakeSession([FakeResponse(body=payload(chunk.start, chunk.end))])
    assert run_chunk(chunk, tmp_path, session, use_cache=False).status == "done"
    assert not list(tmp_path.iterdir())
