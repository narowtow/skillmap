"""Offline fixtures for the data pipeline."""

from __future__ import annotations

import json
import threading

import pandas as pd
import pytest
import requests

from skillmap import config


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Tests must not use real HTTP")
    monkeypatch.setattr(requests.Session, "request", denied)


def payload(start, end) -> dict:
    dates = pd.date_range(start, end).strftime("%Y-%m-%d").tolist()
    return {"latitude": 1.0, "longitude": 2.0, "elevation": 3.0,
            "timezone": "Test/Local", "utc_offset_seconds": 3600,
            "daily": {"time": dates, **{variable: [float(i)] * len(dates)
                                       for i, variable in enumerate(config.VARIABLES)}}}


class FakeClock:
    def __init__(self, now: float = 0.0):
        self.now = now
        self.sleeps = []
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            return self.now

    def sleep(self, seconds):
        with self.lock:
            self.sleeps.append(seconds)
            self.now += seconds


class FakeResponse:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code = status
        self.body = body
        self.headers = headers or {}
        self.content = json.dumps(body).encode()

    def json(self):
        return self.body


class FakeSession:
    def __init__(self, responses=()):
        self.responses = iter(responses)
        self.calls = []
        self.lock = threading.Lock()

    def get(self, url, **kwargs):
        with self.lock:
            self.calls.append((url, kwargs))
            response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response
