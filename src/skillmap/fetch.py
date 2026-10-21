"""Resumable site-year requests with shared weighted budgets and cooldowns."""

from __future__ import annotations

import json
import logging
import math
import os
import random
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import requests

from skillmap import config

LOG = logging.getLogger(__name__)
WINDOW_SECONDS = {"minute": 60.0, "hour": 3600.0, "day": 86400.0}
LEDGER_LOCK = threading.Lock()


def request_weight(n_locations: int, n_days: int, n_variables: int, n_models: int = 1) -> float:
    """Open-Meteo weighted calls, including the location and model multipliers."""
    if min(n_locations, n_days, n_variables, n_models) <= 0:
        raise ValueError("Request dimensions must be positive")
    return n_locations * max(1.0, (n_variables * n_models / 10) * max(1.0, n_days / 14))


@dataclass(frozen=True)
class Chunk:
    """One location and an inclusive local calendar interval."""

    site_id: str
    lat: float
    lon: float
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def year(self) -> int:
        return self.start.year

    @property
    def weight(self) -> float:
        return request_weight(1, (self.end - self.start).days + 1, len(config.VARIABLES))

    def params(self) -> dict:
        return {
            "latitude": self.lat, "longitude": self.lon,
            "start_date": self.start.strftime("%Y-%m-%d"), "end_date": self.end.strftime("%Y-%m-%d"),
            "daily": ",".join(config.VARIABLES), "models": config.WEATHER_MODEL,
            "timezone": "auto", "wind_speed_unit": "ms", "temperature_unit": "celsius",
        }


def order_sites(sites: pd.DataFrame) -> pd.DataFrame:
    """Interleave groups while retaining selection order within each group."""
    if "koppen_group" not in sites:
        return sites.reset_index(drop=True)
    ordered = sites.copy()
    ordered["_rank"] = ordered.groupby("koppen_group", sort=False).cumcount()
    ordered = ordered.sort_values(["_rank", "koppen_group"], kind="stable").drop(columns="_rank")
    return ordered.reset_index(drop=True)


def plan_chunks(
    sites: pd.DataFrame,
    order: str = "site-major",
    start: pd.Timestamp = config.START,
    end: pd.Timestamp = config.END,
) -> list[Chunk]:
    """Calendar-year chunks, clipped to the configured study interval."""
    if order not in {"site-major", "year-major"}:
        raise ValueError("order must be site-major or year-major")
    chunks = [
        Chunk(row.site_id, row.lat, row.lon, max(start, pd.Timestamp(year, 1, 1)),
              min(end, pd.Timestamp(year, 12, 31)))
        for row in order_sites(sites).itertuples()
        for year in range(start.year, end.year + 1)
    ]
    return sorted(chunks, key=lambda c: c.year) if order == "year-major" else chunks


class WeightedRateLimiter:
    """Reserve weighted calls atomically in three independent sliding windows.

    Clock values and past_usage timestamps are epoch seconds. Sleeps happen
    outside the lock, then reservations are rechecked to handle competing threads.
    """

    def __init__(
        self,
        limits: dict[str, float] | None = None,
        *,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        past_usage: Iterable[tuple[float, float]] = (),
    ) -> None:
        self.limits = dict(config.RATE_LIMITS if limits is None else limits)
        if not self.limits or any(not math.isfinite(v) or v <= 0 for v in self.limits.values()):
            raise ValueError("Rate caps must be finite and positive")
        self.clock, self.sleep = clock, sleep
        history = sorted(past_usage)
        self.events = {name: deque(history) for name in self.limits}
        self.lock = threading.Lock()

    def acquire(self, weight: float) -> None:
        """Block until reserving weight cannot exceed any configured cap."""
        if not math.isfinite(weight) or weight <= 0 or any(weight > cap for cap in self.limits.values()):
            raise ValueError("Weight must be positive and fit within every rate cap")
        while True:
            with self.lock:
                now = self.clock()
                delay = 0.0
                for name, cap in self.limits.items():
                    events = self.events[name]
                    window = WINDOW_SECONDS[name]
                    while events and events[0][0] <= now - window:
                        events.popleft()
                    used = sum(w for _, w in events)
                    if used + weight > cap + 1e-10:
                        for timestamp, old_weight in events:
                            used -= old_weight
                            if used + weight <= cap + 1e-10:
                                delay = max(delay, timestamp + window - now)
                                break
                if delay <= 0:
                    for events in self.events.values():
                        events.append((now, weight))
                    return
            self.sleep(delay)


class PauseGate:
    """A 429 extends a common deadline observed by all request threads."""

    def __init__(self, clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.clock, self.sleep = clock, sleep
        self.until = 0.0
        self.lock = threading.Lock()

    def pause(self, seconds: float) -> None:
        with self.lock:
            self.until = max(self.until, self.clock() + max(0, seconds))

    def wait(self) -> bool:
        """Wait for the latest deadline, returning whether any sleep was needed."""
        paused = False
        while True:
            with self.lock:
                delay = self.until - self.clock()
            if delay <= 0:
                return paused
            paused = True
            self.sleep(delay)


class DownloadBudget:
    """Shared per-invocation budget, charging every HTTP attempt including retries."""

    def __init__(self, limit: float | None = None) -> None:
        if limit is not None and (not math.isfinite(limit) or limit < 0):
            raise ValueError("Budget must be finite and nonnegative")
        self.limit, self.spent = limit, 0.0
        self.lock = threading.Lock()

    def reserve(self, weight: float) -> bool:
        with self.lock:
            if self.limit is not None and self.spent + weight > self.limit + 1e-10:
                return False
            self.spent += weight
            return True


def ledger_usage(path: Path) -> list[tuple[float, float]]:
    """Recover completed/error request usage; ignore a torn final ledger line."""
    if not path.exists():
        return []
    usage = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            timestamp = datetime.fromisoformat(row["fetched_at"]).timestamp()
            weight = float(row["weight"])
            if math.isfinite(weight) and weight > 0:
                usage.append((timestamp, weight))
        except (ValueError, TypeError, KeyError):
            LOG.warning("Ignoring malformed ledger entry in %s", path)
    return usage


def validate_response(payload: dict, chunk: Chunk) -> None:
    """Require exact dates, all daily arrays, and numeric-or-null values."""
    if not isinstance(payload, dict) or not isinstance(payload.get("daily"), dict):
        raise ValueError("Missing daily object")
    daily = payload["daily"]
    expected = pd.date_range(chunk.start, chunk.end).strftime("%Y-%m-%d").tolist()
    if daily.get("time") != expected:
        raise ValueError("Response dates do not match requested calendar interval")
    for variable in config.VARIABLES:
        values = daily.get(variable)
        if not isinstance(values, list) or len(values) != len(expected):
            raise ValueError(f"Missing or misaligned daily variable: {variable}")
        if any(value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)))
               for value in values):
            raise ValueError(f"Non-numeric daily variable: {variable}")


def cache_path(cache_dir: Path, chunk: Chunk) -> Path:
    return cache_dir / chunk.site_id / f"{chunk.year}.json"


def read_cache(path: Path, chunk: Chunk) -> dict | None:
    """Return only a fully validated cached response."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_response(payload, chunk)
        return payload
    except (OSError, ValueError, TypeError):
        return None


def _atomic_write(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _retry_after(value: str | None, now: float) -> float:
    if value:
        try:
            seconds = float(value)
            if math.isfinite(seconds):
                return max(0, seconds)
        except ValueError:
            try:
                return max(0, parsedate_to_datetime(value).timestamp() - now)
            except (ValueError, TypeError, OverflowError):
                pass
    return WINDOW_SECONDS["minute"] * 2


@dataclass(frozen=True)
class FetchResult:
    chunk: Chunk
    status: str
    attempts: int = 0
    weight: float = 0.0


def fetch_chunk(
    chunk: Chunk,
    *,
    session: requests.Session,
    limiter: WeightedRateLimiter,
    gate: PauseGate,
    cache_dir: Path = config.OPENMETEO_CACHE,
    budget: DownloadBudget | None = None,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] = random.random,
    use_cache: bool = True,
) -> FetchResult:
    """Fetch at most five times, write original bytes atomically, then journal once.

    Ledger weights include retries; their final timestamp conservatively seeds
    subsequent processes' windows. Concurrent processes sharing a cache are unsupported.
    """
    path = cache_path(cache_dir, chunk)
    if use_cache and read_cache(path, chunk) is not None:
        return FetchResult(chunk, "cached")
    budget = DownloadBudget() if budget is None else budget
    attempts, status = 0, "error"
    for attempt in range(1, 6):
        gate.wait()
        if not budget.reserve(chunk.weight):
            status = "budget"
            break
        while True:
            limiter.acquire(chunk.weight)
            if not gate.wait():
                break
            # A cooldown can outlive a rate reservation. Reserve again at the
            # new send time, conservatively retaining the earlier reservation.
        attempts += 1
        try:
            response = session.get(config.ARCHIVE_URL, params=chunk.params(), timeout=(10, 90))
            if response.status_code == 429:
                gate.pause(_retry_after(response.headers.get("Retry-After"), clock()))
                status = "http_429"
                continue
            if response.status_code >= 500:
                status = f"http_{response.status_code}"
            elif response.status_code >= 400:
                status = f"http_{response.status_code}"
                LOG.error("%s %s: %s", chunk.site_id, chunk.year, status)
                break
            else:
                payload = response.json()
                validate_response(payload, chunk)
                if use_cache:
                    _atomic_write(path, response.content)
                status = "done"
                break
        except (requests.Timeout, requests.ConnectionError) as error:
            status = type(error).__name__
        except (ValueError, TypeError) as error:
            status = "invalid_response"
            LOG.warning("%s %s: %s", chunk.site_id, chunk.year, error)
        if attempt < 5:
            sleep(2 ** (attempt - 1) + jitter())
    weight = attempts * chunk.weight
    if attempts and use_cache:
        record = {"site_id": chunk.site_id, "year": chunk.year, "weight": weight, "status": status,
                  "fetched_at": datetime.fromtimestamp(clock(), UTC).isoformat(), "attempts": attempts}
        cache_dir.mkdir(parents=True, exist_ok=True)
        with LEDGER_LOCK, (cache_dir / "ledger.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    if status not in {"done", "budget"}:
        LOG.error("Failed %s %s after %d attempts: %s", chunk.site_id, chunk.year, attempts, status)
    return FetchResult(chunk, status, attempts, weight)


def download_all(
    sites: pd.DataFrame,
    max_workers: int = 4,
    budget: float | None = None,
    limiter: WeightedRateLimiter | None = None,
    *,
    cache_dir: Path = config.OPENMETEO_CACHE,
    order: str = "site-major",
    session: requests.Session | None = None,
    gate: PauseGate | None = None,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> list[FetchResult]:
    """Download planned chunks with bounded scheduling and a shared retry budget."""
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    limiter = limiter or WeightedRateLimiter(clock=clock, sleep=sleep,
                                             past_usage=ledger_usage(cache_dir / "ledger.jsonl"))
    gate = gate or PauseGate(clock, sleep)
    usage = DownloadBudget(budget)
    own_session = session is None
    session = requests.Session() if session is None else session
    chunks = plan_chunks(sites, order)
    results, pending = [], []
    for chunk in chunks:
        if read_cache(cache_path(cache_dir, chunk), chunk) is not None:
            results.append(FetchResult(chunk, "cached"))
        else:
            pending.append(chunk)
    iterator = iter(pending)
    began = clock()

    def submit_next(executor, futures) -> None:
        if usage.limit is not None and usage.limit - usage.spent + 1e-10 < min_weight:
            return
        chunk = next(iterator, None)
        if chunk is not None:
            future = executor.submit(fetch_chunk, chunk, session=session, limiter=limiter, gate=gate,
                                     cache_dir=cache_dir, budget=usage, clock=clock, sleep=sleep)
            futures.add(future)

    min_weight = min((c.weight for c in pending), default=0)
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures: set = set()
            for _ in range(max_workers):
                submit_next(executor, futures)
            while futures:
                future = next(as_completed(futures))
                futures.remove(future)
                result = future.result()
                results.append(result)
                completed = sum(r.status in {"done", "cached"} for r in results)
                fetched = sum(r.status == "done" for r in results)
                elapsed = clock() - began
                eta = elapsed / fetched * (len(chunks) - completed) if fetched else float("inf")
                LOG.info("Chunks %d/%d; weight %.2f; ETA %.0fs", completed, len(chunks), usage.spent, eta)
                submit_next(executor, futures)
    finally:
        if own_session:
            session.close()
    return results
