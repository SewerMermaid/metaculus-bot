"""Paid, non-publishing smoke test for the proposed four-model ensemble."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta

from forecasting_tools import BinaryQuestion

from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.fallback_openrouter import (
    check_deprecation_alerts_and_exit,
    get_donated_404_fallback_count,
)
from metaculus_bot.forecaster import TemplateForecaster
from metaculus_bot.llm_configs import (
    DISAGREEMENT_ANALYZER_LLM,
    PARSER_LLM,
    RESEARCHER_LLM,
    STACKER_LLM,
    SUMMARIZER_LLM,
    build_smoke_forecaster_llms,
)

logger = logging.getLogger(__name__)


async def _fixed_research(question: BinaryQuestion) -> str:
    """Use only the user-supplied context, avoiding separate research API calls."""
    return question.background_info or "No additional background information was provided."


def build_smoke_forecaster() -> TemplateForecaster:
    """Build the production forecasting architecture with only the proposed base lineup substituted."""
    return TemplateForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=False,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=False,
        aggregation_strategy=AggregationStrategy.CONDITIONAL_STACKING,
        research_provider=_fixed_research,
        max_questions_per_run=1,
        llms={
            "forecasters": build_smoke_forecaster_llms(),
            "stacker": STACKER_LLM,
            "analyzer": DISAGREEMENT_ANALYZER_LLM,
            "summarizer": SUMMARIZER_LLM,
            "parser": PARSER_LLM,
            "researcher": RESEARCHER_LLM,
        },
    )


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must be supplied for the custom-question smoke test")
    return value


def _make_smoke_question() -> BinaryQuestion:
    question_text = _required_env("SMOKE_QUESTION_TEXT")
    resolution_criteria = _required_env("SMOKE_RESOLUTION_CRITERIA")
    # Prompt rendering currently compares these values with naive ``datetime.now()``.
    now = datetime.now()
    return BinaryQuestion(
        question_text=question_text,
        id_of_post=999_999_001,
        id_of_question=999_999_001,
        background_info=os.environ.get("SMOKE_BACKGROUND_INFO", "").strip(),
        resolution_criteria=resolution_criteria,
        fine_print=os.environ.get("SMOKE_FINE_PRINT", "").strip(),
        open_time=now - timedelta(days=1),
        scheduled_resolution_time=now + timedelta(days=1),
    )


async def run_smoke() -> float:
    bot = build_smoke_forecaster()
    models = [llm.model for llm in bot._forecaster_llms]
    logger.info("Running non-publishing ensemble smoke test with models: %s", models)

    reports = await bot.forecast_questions([_make_smoke_question()], return_exceptions=True)
    if len(reports) != 1:
        raise RuntimeError(f"Expected one smoke report, received {len(reports)}")

    report = reports[0]
    if isinstance(report, BaseException):
        raise RuntimeError(f"Smoke forecast failed: {report}") from report

    prediction = getattr(report, "prediction", None)
    if not isinstance(prediction, (int, float)) or not 0.0 < float(prediction) < 1.0:
        raise RuntimeError(f"Smoke forecast returned an invalid binary prediction: {prediction!r}")

    TemplateForecaster.log_report_summary(reports)
    if bot.alertable_count:
        raise RuntimeError(f"Smoke forecast completed with {bot.alertable_count} alertable degradation event(s)")
    donated_404 = get_donated_404_fallback_count()
    if donated_404:
        raise RuntimeError(f"Smoke forecast used personal-key fallback after {donated_404} donated-key 404(s)")

    check_deprecation_alerts_and_exit()

    summary = "\n".join(
        [
            "# Proposed ensemble smoke test",
            "",
            "- Publication: disabled",
            "- Question: custom binary question supplied for this workflow run",
            "- Research: user-supplied background plus Gemini web research via OpenRouter",
            f"- Gemini research backend: {os.environ.get('GEMINI_SEARCH_BACKEND', 'google')}",
            f"- Gemini research model: {os.environ.get('GEMINI_SEARCH_MODEL', 'disabled')}",
            f"- Gemini fallback model: {os.environ.get('GEMINI_SEARCH_FALLBACK_MODEL', 'none')}",
            "- Aggregation: conditional stacking with production-equivalent stacking flags",
            f"- Models: {', '.join(models)}",
            f"- Aggregate binary prediction: {float(prediction):.4f}",
            "",
        ]
    )
    logger.info(summary)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as summary_file:
            summary_file.write(summary + "\n")

    return float(prediction)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(run_smoke())


if __name__ == "__main__":
    main()
