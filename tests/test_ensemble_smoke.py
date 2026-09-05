from pathlib import Path
from types import SimpleNamespace

from metaculus_bot import ensemble_smoke, llm_configs
from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.prompts import binary_prompt


def test_smoke_question_uses_required_user_inputs(monkeypatch) -> None:
    monkeypatch.setenv("SMOKE_QUESTION_TEXT", "Will the event happen by year-end?")
    monkeypatch.setenv("SMOKE_BACKGROUND_INFO", "Relevant facts supplied by the user.")
    monkeypatch.setenv("SMOKE_RESOLUTION_CRITERIA", "Resolves YES if the event happens by December 31.")
    monkeypatch.setenv("SMOKE_FINE_PRINT", "Use the official announcement time.")

    question = ensemble_smoke._make_smoke_question()

    assert question.question_text == "Will the event happen by year-end?"
    assert question.background_info == "Relevant facts supplied by the user."
    assert question.resolution_criteria == "Resolves YES if the event happens by December 31."
    assert question.fine_print == "Use the official announcement time."
    assert question.open_time.tzinfo is None
    assert question.scheduled_resolution_time.tzinfo is None
    assert "Will the event happen by year-end?" in binary_prompt(question, "Research context")


def test_smoke_question_rejects_missing_required_input(monkeypatch) -> None:
    monkeypatch.delenv("SMOKE_QUESTION_TEXT", raising=False)
    monkeypatch.setenv("SMOKE_RESOLUTION_CRITERIA", "Resolves YES if it happens.")

    try:
        ensemble_smoke._make_smoke_question()
    except RuntimeError as exc:
        assert "SMOKE_QUESTION_TEXT" in str(exc)
    else:
        raise AssertionError("Expected missing question text to fail")


def test_smoke_workflow_can_toggle_gemini_research_and_uses_funded_key() -> None:
    workflow = Path(".github/workflows/smoke_test_next_ensemble.yaml").read_text(encoding="utf-8")

    assert "GOOGLE_API_KEY:" not in workflow
    assert "OPENROUTER_API_KEY: ${{ secrets.OAI_ANTH_OPENROUTER_KEY }}" in workflow
    assert "include_gemini_research:" in workflow
    assert "GEMINI_SEARCH_ENABLED: ${{ inputs.include_gemini_research }}" in workflow
    assert "GEMINI_SEARCH_BACKEND: 'openrouter'" in workflow
    assert "GEMINI_SEARCH_MODEL: 'gemini-3.8-flash'" in workflow
    assert "GEMINI_SEARCH_FALLBACK_MODEL: 'gemini-3.7-flash'" in workflow
    assert "GEMINI_USE_DONATED_OPENROUTER_KEY: 'true'" in workflow
    assert "NATIVE_SEARCH_REASONING_EFFORT: 'high'" in workflow


def test_production_lineup_is_unchanged() -> None:
    assert [llm.model for llm in llm_configs.FORECASTER_LLMS] == [
        "openrouter/openai/gpt-5.4",
        "openrouter/openai/gpt-5.5",
        "openrouter/anthropic/claude-opus-4.8",
        "openrouter/anthropic/claude-opus-4.6",
    ]


def test_smoke_lineup_uses_requested_models_at_high_effort(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_builder(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(model=kwargs["model"])

    monkeypatch.setattr(llm_configs, "build_llm_with_openrouter_fallback", fake_builder)

    llms = llm_configs.build_smoke_forecaster_llms()

    assert [llm.model for llm in llms] == [
        "openrouter/openai/gpt-5.6-terra",
        "openrouter/openai/gpt-5.6-sol",
        "openrouter/anthropic/claude-fable-5",
        "openrouter/anthropic/claude-opus-5",
    ]
    assert all(call["reasoning"] == {"effort": "high"} for call in calls)
    assert all(call["max_tokens"] == 64_000 for call in calls)
    assert all("temperature" not in call and "top_p" not in call for call in calls)
    assert calls[2]["extra_body"] == {"verbosity": "high"}
    assert calls[3]["extra_body"] == {"verbosity": "high"}


def test_smoke_forecaster_preserves_production_architecture_without_publishing(monkeypatch) -> None:
    captured: dict = {}
    smoke_llms = [SimpleNamespace(model="smoke-model")]
    sentinel = object()

    def fake_forecaster(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(ensemble_smoke, "build_smoke_forecaster_llms", lambda: smoke_llms)
    monkeypatch.setattr(ensemble_smoke, "TemplateForecaster", fake_forecaster)

    result = ensemble_smoke.build_smoke_forecaster()

    assert result is sentinel
    assert captured["llms"]["forecasters"] is smoke_llms
    assert captured["aggregation_strategy"] is AggregationStrategy.CONDITIONAL_STACKING
    assert captured["research_reports_per_question"] == 1
    assert captured["predictions_per_research_report"] == 1
    assert captured["publish_reports_to_metaculus"] is False
    assert captured["skip_previously_forecasted_questions"] is False
    assert captured["max_questions_per_run"] == 1
    assert captured["research_provider"] is ensemble_smoke._fixed_research
