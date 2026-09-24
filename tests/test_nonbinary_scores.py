import pandas as pd
import pytest

from metaculus_bot.minibench_analysis.aggregate import QuestionVerdict, summarize_bot
from metaculus_bot.minibench_analysis.parse import my_bot_question_detail, verdict_from_question
from metaculus_bot.minibench_analysis.report import my_bot_accuracy_records, write_csv, write_xlsx
from metaculus_bot.minibench_analysis.scoring import NumericCdf, bounded_crps, multiclass_brier


def test_mc_sum_convention():
    assert multiclass_brier([0.6, 0.3, 0.1], 0) == pytest.approx(0.26)
    assert multiclass_brier([1.0, 0.0], 0) == 0
    assert multiclass_brier([1.0, 0.0], 1) == 2


@pytest.mark.parametrize("probs", [[0.2, 0.3], [float("nan"), 1], [-0.1, 1.1]])
def test_invalid_mc(probs):
    with pytest.raises(ValueError):
        multiclass_brier(probs, 0)


@pytest.mark.parametrize("outcome,expected", [(0, 1 / 3), (0.25, 7 / 48), (0.5, 1 / 12), (1, 1 / 3)])
def test_crps_uniform_analytic(outcome, expected):
    # Analytic CRPS for U(0,1), including a resolution between grid points.
    assert bounded_crps(NumericCdf([0, 1], [0, 1]), outcome) == pytest.approx(expected)


def test_nonuniform_axis_and_scale():
    assert bounded_crps(NumericCdf([0, 0.1, 1], [0, 0.1, 1]), 0.5) == pytest.approx(1 / 12)
    assert bounded_crps(NumericCdf([0, 10, 100], [0, 0.1, 1]), 50) == pytest.approx(100 / 12)
    # Open tails: flat 0.5 over a unit interval has bounded score 0.25.
    assert bounded_crps(NumericCdf([0, 1], [0.5, 0.5]), 0.3) == pytest.approx(0.25)


@pytest.mark.parametrize(
    "cdf",
    [
        NumericCdf([0, 0], [0, 1]),
        NumericCdf([0, 1], [0.8, 0.2]),
        NumericCdf([0, 1], [0]),
        NumericCdf([0, 1], [0, float("nan")]),
    ],
)
def test_bad_cdf_rejected(cdf):
    with pytest.raises(ValueError):
        bounded_crps(cdf, 0.5)


def test_nonbinary_pipeline_and_exports(tmp_path):
    mc = {
        "id": 1,
        "type": "multiple_choice",
        "options": ["A", "B", "C"],
        "resolution": "A",
        "my_forecasts": {"latest": {"forecast_values": [0.6, 0.3, 0.1]}},
    }
    numeric = {
        "id": 2,
        "type": "numeric",
        "resolution": "50",
        "scaling": {"range_min": 0, "range_max": 100},
        "my_forecasts": {"latest": {"forecast_values": [0, 0.5, 1]}},
    }
    verdicts = [verdict_from_question(q, is_my_bot=True) for q in (mc, numeric)]
    verdicts.append(QuestionVerdict(3, "multiple_choice", answered=False, scorable=True))
    summary = summarize_bot("bot", verdicts)
    accuracy = my_bot_accuracy_records(summary)
    assert accuracy["mc_brier_mean"] == pytest.approx(0.26)
    assert accuracy["mc_brier_n"] == 1
    assert accuracy["mc_scored"] == 1  # A resolved but unanswered question is not a score.
    assert accuracy["numeric_normalized_bounded_crps_mean"] == pytest.approx(1 / 12)
    assert accuracy["numeric_normalized_bounded_crps_n"] == 1
    assert not summary.overall.brier_scores
    details = [my_bot_question_detail({"id": q["id"]}, q) for q in (mc, numeric)]
    assert details[0]["brier_score"] == pytest.approx(0.26)
    assert details[1]["bounded_crps"] == pytest.approx(100 / 12)
    csv = tmp_path / "accuracy.csv"
    write_csv([accuracy], str(csv))
    assert pd.read_csv(csv).iloc[0]["mc_brier_mean"] == pytest.approx(0.26)
    xlsx = tmp_path / "scores.xlsx"
    assert write_xlsx({"accuracy": [accuracy], "questions": details}, str(xlsx))
    assert pd.read_excel(xlsx, sheet_name="accuracy").iloc[0]["numeric_normalized_bounded_crps_mean"] == pytest.approx(
        1 / 12
    )
    assert "unknown tails" in pd.read_excel(xlsx, sheet_name="scoring_notes").iloc[0, 0]


@pytest.mark.parametrize("resolution", [None, "annulled", "ambiguous", "above_upper_bound", "150"])
def test_unscorable_numeric_is_blank(resolution):
    q = {
        "type": "numeric",
        "resolution": resolution,
        "scaling": {"range_min": 0, "range_max": 100},
        "my_forecasts": {"latest": {"forecast_values": [0, 0.5, 1]}},
    }
    v = verdict_from_question(q, is_my_bot=True)
    assert v.bounded_crps is None
    assert v.normalized_bounded_crps is None
