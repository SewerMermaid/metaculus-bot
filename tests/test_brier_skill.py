import pytest

from metaculus_bot.minibench_analysis.aggregate import QuestionVerdict, TypeSummary, summarize_bot
from metaculus_bot.minibench_analysis.parse import my_bot_question_detail, verdict_from_question
from metaculus_bot.minibench_analysis.report import my_bot_accuracy_records


def question(options, probabilities, resolution):
    return {
        "id": 1,
        "type": "multiple_choice" if options else "binary",
        "options": options,
        "resolution": resolution,
        "my_forecasts": {"latest": {"forecast_values": probabilities}},
    }


@pytest.mark.parametrize("probability,expected", [(1, 1), (0.5, 0), (0.1, -2.24)])
def test_binary_skill(probability, expected):
    q = question([], [1 - probability, probability], "yes")
    v = verdict_from_question(q, is_my_bot=True)
    row = my_bot_accuracy_records(summarize_bot("bot", [v]))
    assert row["binary_brier_skill"] == pytest.approx(expected)
    assert my_bot_question_detail({"id": 1}, q)["brier_skill"] == pytest.approx(expected)


def test_mixed_option_counts_use_ratio_of_sums():
    questions = [question(["A", "B"], [1, 0], "B"), question(["A", "B", "C", "D"], [1, 0, 0, 0], "A")]
    verdicts = [verdict_from_question(q, is_my_bot=True) for q in questions]
    verdicts += [
        QuestionVerdict(2, "multiple_choice", False, True, brier_score=2, brier_uniform_baseline=0.5),
        QuestionVerdict(3, "multiple_choice", True, False, brier_score=2, brier_uniform_baseline=0.5),
    ]
    summary = summarize_bot("bot", verdicts)
    row = my_bot_accuracy_records(summary)
    assert row["mc_brier_n"] == 2
    assert row["mc_brier_skill"] == pytest.approx(1 - 2 / (0.5 + 0.75))
    assert not summary.overall.brier_baselines


def test_empty_or_incomplete_baselines_are_not_zero_skill():
    assert TypeSummary().brier_skill is None
    assert TypeSummary(brier_scores=[0.1]).brier_skill is None
    assert TypeSummary(brier_scores=[0.1, 0.2], brier_baselines=[0.25]).brier_skill is None


def test_uniform_mc_skill():
    q = question(["A", "B", "C"], [1 / 3] * 3, "B")
    row = my_bot_accuracy_records(summarize_bot("bot", [verdict_from_question(q, is_my_bot=True)]))
    assert row["mc_brier_skill"] == pytest.approx(0)
