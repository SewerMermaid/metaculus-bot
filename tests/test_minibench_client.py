"""Unit tests for MetaculusClient construction (no network)."""

import pytest

from metaculus_bot.minibench_analysis.client import MetaculusClient


def test_token_trailing_newline_is_stripped_from_header():
    """A secret stored with a trailing newline must not poison the auth header.

    Regression: an unstripped "Token <...>\\n" made requests raise
    "ValueError: Invalid header value" before any call, so the whole run
    crashed and produced no report files.
    """
    client = MetaculusClient(token="abc123\n")
    assert client.token == "abc123"
    auth = client._headers["Authorization"]
    assert auth == "Token abc123"
    assert "\n" not in auth and "\r" not in auth


def test_token_surrounding_whitespace_is_stripped():
    client = MetaculusClient(token="  abc123  ")
    assert client.token == "abc123"


def test_token_from_env_is_stripped(monkeypatch):
    monkeypatch.setenv("METACULUS_TOKEN", "envtoken\n")
    client = MetaculusClient()
    assert client.token == "envtoken"


def test_missing_token_raises(monkeypatch):
    monkeypatch.delenv("METACULUS_TOKEN", raising=False)
    with pytest.raises(ValueError):
        MetaculusClient()


def test_whitespace_only_token_raises(monkeypatch):
    monkeypatch.delenv("METACULUS_TOKEN", raising=False)
    with pytest.raises(ValueError):
        MetaculusClient(token="   \n")


def _client(_get):
    c = MetaculusClient(token="t")
    c._get = _get  # type: ignore[assignment]
    return c


def test_list_tournaments_uses_dedicated_minibench_endpoint_and_sorts_oldest_first():
    calls = []

    def fake_get(path, params=None):
        calls.append((path, params))
        return [
            {"id": 2, "slug": "minibench", "name": "MiniBench", "start_date": "2026-06-15"},
            {"id": 1, "slug": "minibench-2026-06-01", "name": "MiniBench", "start_date": "2026-06-01"},
        ]

    got = _client(fake_get).list_minibench_tournaments()
    assert [t["slug"] for t in got] == ["minibench-2026-06-01", "minibench"]
    assert calls == [("/projects/minibenches/", None)]


def test_list_tournaments_tolerates_bare_list_response():
    def fake_get(path, params=None):
        # No "results" wrapper — a bare list.
        return [{"id": 3, "slug": "minibench-archive", "name": "Mini Bench May"}]

    got = _client(fake_get).list_minibench_tournaments()
    assert [t["slug"] for t in got] == ["minibench-archive"]


def test_list_tournaments_falls_back_to_slug_when_dedicated_listing_is_empty():
    def fake_get(path, params=None):
        if path == "/projects/tournaments/minibench/":
            return {"id": 42, "slug": "minibench", "name": "MiniBench"}
        return []

    got = _client(fake_get).list_minibench_tournaments()
    assert [t["slug"] for t in got] == ["minibench"]


def test_list_tournaments_empty_when_listing_and_fallback_both_empty():
    def fake_get(path, params=None):
        return None  # nothing from the listing, nothing from the slug fetch

    assert _client(fake_get).list_minibench_tournaments() == []


def test_get_leaderboard_uses_project_leaderboard_endpoint_and_nested_entries():
    calls = []

    def fake_get(path, params=None):
        calls.append((path, params))
        return [
            {
                "id": 7,
                "entries": [
                    {"rank": 2, "user": {"id": 12, "username": "second"}, "score": 4.0},
                    {"rank": 1, "user": {"id": 11, "username": "first"}, "score": 5.0},
                ],
            }
        ]

    got = _client(fake_get).get_leaderboard(33074)

    assert calls == [("/leaderboards/project/33074/", {"primary_only": "true", "with_entries": "true"})]
    assert [entry["username"] for entry in got] == ["first", "second"]
    assert [entry["user_id"] for entry in got] == [11, 12]


def test_get_leaderboard_returns_empty_when_no_entries():
    assert _client(lambda path, params=None: [{"id": 7, "entries": []}]).get_leaderboard(33074) == []
