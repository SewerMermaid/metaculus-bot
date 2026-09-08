"""End-to-end MiniBench CLI test with a fully mocked Metaculus client (no network)."""

import pandas as pd
import pytest

from metaculus_bot.minibench_analysis import cli
from metaculus_bot.minibench_analysis.aggregate import summarize_bot
from metaculus_bot.minibench_analysis.parse import verdict_from_question
from metaculus_bot.minibench_analysis.report import (
    my_bot_accuracy_records,
    my_bot_answered_records,
    top10_records,
)


class _FakeClient:
    """Stand-in for MetaculusClient returning canned tournaments/posts/leaderboard."""

    def __init__(self):
        self.tournaments = [
            {"id": 1, "slug": "minibench-a", "name": "MiniBench A", "start_date": "2026-05-04"},
            {"id": 2, "slug": "minibench-b", "name": "MiniBench B", "start_date": "2026-05-18"},
            {"id": 3, "slug": "minibench", "name": "MiniBench C", "start_date": "2026-06-01"},
        ]

    def get_me(self):
        return {"id": 99, "username": "my-bot"}

    def list_minibench_tournaments(self):
        return self.tournaments

    def get_leaderboard(self, project_id):
        return [
            {"rank": 1, "username": "alpha", "user_id": 10, "score": 55.0, "take": 500, "peer_score": 12.0},
            {"rank": 2, "username": "beta", "user_id": 11, "score": 40.0, "take": 300, "peer_score": 8.0},
        ]

    def get_resolved_posts(self, tournament):
        # Two binary (one hit, one miss), one numeric uniform (IQR hit, not beat-chance).
        return [
            {
                "id": 101,
                "slug": "q-one",
                "title": "Question one?",
                "question": {
                    "id": 1,
                    "type": "binary",
                    "resolution": "yes",
                    "my_forecasts": {"latest": {"forecast_values": [0.2, 0.8]}},
                },
            },
            {
                "id": 102,
                "slug": "q-two",
                "title": "Question two?",
                "question": {
                    "id": 2,
                    "type": "binary",
                    "resolution": "no",
                    "my_forecasts": {"latest": {"forecast_values": [0.3, 0.7]}},
                },
            },
            {
                "id": 103,
                "slug": "q-three",
                "title": "Question three?",
                "question": {
                    "id": 3,
                    "type": "numeric",
                    "resolution": "50",
                    "scaling": {"range_min": 0.0, "range_max": 100.0, "zero_point": None},
                    "my_forecasts": {"latest": {"forecast_values": [i / 200 for i in range(201)]}},
                },
            },
        ]


class _NullLatestClient(_FakeClient):
    """Include the API shape used for resolved questions the bot did not answer."""

    def get_resolved_posts(self, tournament):
        posts = super().get_resolved_posts(tournament)
        posts.append(
            {
                "id": 104,
                "slug": "q-unanswered",
                "title": "Question not answered?",
                "question": {
                    "id": 4,
                    "type": "binary",
                    "resolution": "yes",
                    "my_forecasts": {"latest": None},
                },
            }
        )
        return posts


def test_two_sessions_ago_writes_files_and_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "MetaculusClient", lambda *a, **k: _FakeClient())
    rc = cli.main(["--mode", "two-sessions-ago", "--output-dir", str(tmp_path), "--session-offset", "2"])
    assert rc == 0

    answered = pd.read_csv(tmp_path / "my_bot_answered.csv").iloc[0]
    assert answered["total_answered"] == 3
    assert answered["binary_answered"] == 2
    assert answered["numeric_answered"] == 1

    acc = pd.read_csv(tmp_path / "my_bot_accuracy.csv").iloc[0]
    # Binary: one hit of two scorable -> 50%.
    assert acc["binary_beatchance_pct"] == 50.0
    # Numeric uniform: IQR hit (tier2 100%) but beat-chance 0%.
    assert acc["numeric_tier2_pct"] == 100.0
    assert acc["numeric_beatchance_pct"] == 0.0
    # Binary Brier: (0.8-1)^2 and (0.7-0)^2, averaged.
    assert acc["binary_brier_n"] == 2
    assert acc["binary_brier_mean"] == pytest.approx(0.265)

    top = pd.read_csv(tmp_path / "top_bots_accuracy.csv")
    assert list(top["bot"]) == ["alpha", "beta"]
    assert not top["per_question_available"].any()  # other bots' forecasts not exposed

    questions = pd.read_csv(tmp_path / "my_bot_questions.csv")
    assert len(questions) == 3
    assert set(questions["question_url"]) == {
        "https://www.metaculus.com/questions/101/q-one/",
        "https://www.metaculus.com/questions/102/q-two/",
        "https://www.metaculus.com/questions/103/q-three/",
    }
    assert (questions["question_url"] == questions["my_answer_url"]).all()
    q1 = questions[questions["question_id"] == 1].iloc[0]
    assert q1["my_prediction"] == "80% yes"
    assert q1["title"] == "Question one?"
    assert q1["accurate"] == "yes"  # 80% yes, resolved yes -> directional hit
    # Q2 forecast 70% yes but resolved no -> inaccurate.
    q2 = questions[questions["question_id"] == 2].iloc[0]
    assert q2["accurate"] == "no"
    assert q2["brier_score"] == pytest.approx(0.49)


def test_both_report_modes_tolerate_null_latest(tmp_path, monkeypatch):
    fake = _NullLatestClient()
    monkeypatch.setattr(cli, "MetaculusClient", lambda *a, **k: fake)

    top_dir = tmp_path / "top"
    rc = cli.main(["--mode", "two-sessions-ago", "--output-dir", str(top_dir), "--session-offset", "2"])
    assert rc == 0
    assert pd.read_csv(top_dir / "my_bot_answered.csv").iloc[0]["total_answered"] == 3

    history_dir = tmp_path / "history"
    rc = cli.main(["--mode", "all-except-current", "--output-dir", str(history_dir)])
    assert rc == 0
    assert list(pd.read_csv(history_dir / "my_bot_history_answered.csv")["total_answered"]) == [3, 3]


def test_two_sessions_ago_targets_correct_tournament(tmp_path, monkeypatch):
    """offset=2 with 3 tournaments -> the oldest (index -3) is analyzed."""
    captured = {}
    fake = _FakeClient()
    original = fake.get_resolved_posts

    def _capture(tid):
        captured["tid"] = tid
        return original(tid)

    fake.get_resolved_posts = _capture  # type: ignore
    monkeypatch.setattr(cli, "MetaculusClient", lambda *a, **k: fake)
    cli.main(["--mode", "two-sessions-ago", "--output-dir", str(tmp_path), "--session-offset", "2"])
    assert captured["tid"] == 1  # MiniBench A, two sessions before current (id=3)


def test_all_except_current_excludes_latest(tmp_path, monkeypatch):
    fake = _FakeClient()
    seen = []
    orig = fake.get_resolved_posts

    def _track(tid):
        seen.append(tid)
        return orig(tid)

    fake.get_resolved_posts = _track  # type: ignore
    monkeypatch.setattr(cli, "MetaculusClient", lambda *a, **k: fake)
    rc = cli.main(["--mode", "all-except-current", "--output-dir", str(tmp_path)])
    assert rc == 0
    assert seen == [1, 2]  # current (id=3) excluded

    hist = pd.read_csv(tmp_path / "my_bot_history_answered.csv")
    assert len(hist) == 2
    assert list(hist["total_answered"]) == [3, 3]

    q_hist = pd.read_csv(tmp_path / "my_bot_history_questions.csv")
    assert len(q_hist) == 6  # 3 questions x 2 past minibenches
    assert set(q_hist["minibench"]) == {"MiniBench A", "MiniBench B"}


def test_all_except_current_single_tournament_degrades(tmp_path, monkeypatch):
    """With only one visible tournament, analyze it (with a note) instead of writing nothing."""
    fake = _FakeClient()
    fake.tournaments = [{"id": 1, "slug": "minibench", "name": "MiniBench", "start_date": "2026-06-01"}]
    monkeypatch.setattr(cli, "MetaculusClient", lambda *a, **k: fake)

    summary = cli.run_all_except_current(fake, str(tmp_path))
    assert "only one MiniBench tournament" in summary
    assert "current included" in summary

    hist = pd.read_csv(tmp_path / "my_bot_history_answered.csv")
    assert len(hist) == 1  # a file is produced, not an empty dir
    assert hist.iloc[0]["total_answered"] == 3


def test_all_except_current_no_tournaments_writes_message(monkeypatch, tmp_path):
    fake = _FakeClient()
    fake.tournaments = []
    summary = cli.run_all_except_current(fake, str(tmp_path))
    assert "No MiniBench tournaments found" in summary


def test_explicit_tournaments_analyzes_given_ids(tmp_path, monkeypatch):
    """--tournaments bypasses discovery and analyzes the supplied ids/slugs."""
    fake = _FakeClient()
    fetched = []
    seen_posts = []
    fake.get_tournament = lambda t: fetched.append(t) or {"id": t, "slug": f"mb-{t}", "name": f"MB {t}"}
    orig_posts = fake.get_resolved_posts
    fake.get_resolved_posts = lambda tid: seen_posts.append(tid) or orig_posts(tid)
    monkeypatch.setattr(cli, "MetaculusClient", lambda *a, **k: fake)

    rc = cli.main(["--mode", "all-except-current", "--output-dir", str(tmp_path), "--tournaments", " 501 , 502 "])
    assert rc == 0
    assert fetched == ["501", "502"]  # whitespace trimmed, both fetched
    assert seen_posts == ["501", "502"]  # discovery was NOT used

    hist = pd.read_csv(tmp_path / "my_bot_history_answered.csv")
    assert len(hist) == 2
    assert set(hist["minibench"]) == {"MB 501", "MB 502"}

    ranking = pd.read_csv(tmp_path / "my_bot_history_ranking.csv")
    assert list(ranking["minibench"]) == ["MB 501", "MB 502"]


def test_history_records_authenticated_bot_leaderboard_rank(tmp_path):
    fake = _FakeClient()
    original = fake.get_leaderboard

    def _leaderboard(project_id):
        return original(project_id) + [
            {"rank": 7, "username": "my-bot", "user_id": 99, "score": 12.5, "take": 80, "peer_score": 3.0}
        ]

    fake.get_leaderboard = _leaderboard  # type: ignore
    fake.get_tournament = lambda t: {"id": t, "slug": f"mb-{t}", "name": f"MB {t}"}
    cli.run_explicit_tournaments(fake, str(tmp_path), ["501"], me=fake.get_me())

    ranking = pd.read_csv(tmp_path / "my_bot_history_ranking.csv").iloc[0]
    assert ranking["bot"] == "my-bot"
    assert ranking["rank"] == 7
    assert ranking["leaderboard_score"] == 12.5

    workbook = pd.ExcelFile(tmp_path / "my_bot_history.xlsx")
    assert workbook.sheet_names == ["answered", "accuracy", "ranking", "questions"]


def test_explicit_tournament_not_found_falls_back_to_bare_id(tmp_path, monkeypatch):
    fake = _FakeClient()
    fake.get_tournament = lambda t: None  # not found via API
    monkeypatch.setattr(cli, "MetaculusClient", lambda *a, **k: fake)
    summary = cli.run_explicit_tournaments(fake, str(tmp_path), ["mb-99"])
    assert "explicit list" in summary
    hist = pd.read_csv(tmp_path / "my_bot_history_answered.csv")
    assert list(hist["minibench"]) == ["mb-99"]  # bare id used as label


def test_report_records_shapes():
    posts_q = {
        "id": 1,
        "type": "binary",
        "resolution": "yes",
        "my_forecasts": {"latest": {"forecast_values": [0.1, 0.9]}},
    }
    v = verdict_from_question(posts_q, is_my_bot=True)
    s = summarize_bot("my-bot", [v], rank=1)
    assert my_bot_answered_records(s, label="X")["binary_answered"] == 1
    assert my_bot_accuracy_records(s, label="X")["binary_beatchance_pct"] == 100.0
    assert top10_records([s], {"my-bot": {"leaderboard_score": 10}})[0]["leaderboard_score"] == 10
