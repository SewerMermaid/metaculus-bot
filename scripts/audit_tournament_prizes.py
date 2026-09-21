"""Print sanitized leaderboard prize fields for the authenticated Metaculus account."""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import requests


BASE_URL = "https://www.metaculus.com/api"


def _get(path: str, token: str, params: dict[str, str] | None = None) -> Any:
    for attempt in range(4):
        response = requests.get(
            f"{BASE_URL}{path}",
            headers={"Authorization": f"Token {token}"},
            params=params,
            timeout=30,
        )
        if response.status_code != 429:
            response.raise_for_status()
            return response.json()
        if attempt < 3:
            retry_after = response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 10.0 * (attempt + 1)
            time.sleep(min(max(delay, 1.0), 60.0))
    response.raise_for_status()
    raise RuntimeError("Unreachable")


def _entry_user_id(entry: dict[str, Any]) -> Any:
    user = entry.get("user")
    nested = user if isinstance(user, dict) else {}
    return entry.get("user_id") or nested.get("id")


def main() -> int:
    token = os.environ["METACULUS_TOKEN"]
    me = _get("/users/me/", token)
    user_id = me.get("id")
    results: list[dict[str, Any]] = []

    for tournament in sys.argv[1:]:
        project = _get(f"/projects/tournaments/{tournament}/", token)
        project_id = project["id"]
        payload = _get(
            f"/leaderboards/project/{project_id}/",
            token,
            {"primary_only": "true", "with_entries": "true"},
        )
        leaderboards = payload if isinstance(payload, list) else [payload]
        entries = [
            entry
            for leaderboard in leaderboards
            if isinstance(leaderboard, dict)
            for entry in leaderboard.get("entries", [])
            if isinstance(entry, dict)
        ]
        mine = next((entry for entry in entries if str(_entry_user_id(entry)) == str(user_id)), None)
        numeric_takes = [float(entry["take"]) for entry in entries if isinstance(entry.get("take"), (int, float))]
        numeric_prizes = [float(entry["prize"]) for entry in entries if isinstance(entry.get("prize"), (int, float))]
        results.append(
            {
                "tournament": tournament,
                "project_id": project_id,
                "authenticated_username": me.get("username"),
                "leaderboard_entry_count": len(entries),
                "own_entry": mine,
                "take_sum": sum(numeric_takes),
                "reported_prize_sum": sum(numeric_prizes),
                "positive_reported_prizes": sum(1 for value in numeric_prizes if value > 0),
            }
        )

    rendered = json.dumps(results, indent=2, sort_keys=True)
    with open("tournament_audit.json", "w", encoding="utf-8") as output:
        output.write(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
