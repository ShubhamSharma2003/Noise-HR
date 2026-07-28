import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("FRESHTEAM_API_KEY", "test-key")
os.environ.setdefault("FRESHTEAM_SUBDOMAIN", "test")

import requests
import pytest

import hr_system.freshteam as ft
from hr_system.freshteam import FreshteamClient


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(ft.time, "sleep", lambda s: None)


def _ok_response(payload):
    return SimpleNamespace(
        status_code=200,
        headers={"total-pages": "1"},
        json=lambda: payload,
        raise_for_status=lambda: None,
    )


class TestConnectionRetry:
    def test_get_retries_transient_connection_error_then_succeeds(self, monkeypatch):
        calls = {"n": 0}

        def flaky_get(url, **kw):
            calls["n"] += 1
            if calls["n"] < 3:
                raise requests.exceptions.ConnectionError("boom")
            return _ok_response([])

        monkeypatch.setattr(ft.requests, "get", flaky_get)
        resp = FreshteamClient()._get("/job_postings")
        assert resp.status_code == 200
        assert calls["n"] == 3   # retried twice, succeeded on the third

    def test_get_job_postings_degrades_on_persistent_network_error(self, monkeypatch):
        def always_timeout(url, **kw):
            raise requests.exceptions.Timeout("timed out")

        monkeypatch.setattr(ft.requests, "get", always_timeout)
        # Must not raise — degrades to empty list so the app offers manual entry.
        assert FreshteamClient().get_job_postings() == []

    def test_get_job_posting_returns_stub_on_network_error(self, monkeypatch):
        def always_conn_err(url, **kw):
            raise requests.exceptions.ConnectionError("down")

        monkeypatch.setattr(ft.requests, "get", always_conn_err)
        assert FreshteamClient().get_job_posting(42) == {"id": 42}
