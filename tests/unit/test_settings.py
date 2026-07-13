"""
Tests for src/config/settings.py's per-agent model tiering (roadmap #24).

MAEDASettings fields use aliases (e.g. "MAEDA_PLANNER_MODEL") as their
constructor/env-var keys, so constructing MAEDASettings(planner_llm_model=...)
directly with the plain attribute name silently no-ops (extra="ignore"
swallows the unrecognized kwarg rather than raising). Mutating the real
settings singleton via monkeypatch -- as src/persistence tests already do
for runs_db_path -- sidesteps that entirely and is how this is actually
read throughout the app (`from src.config.settings import settings`).
"""
from src.config.settings import settings


def test_resolved_planner_model_defaults_to_stronger_openai_model(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "planner_llm_model", None)
    assert settings.resolved_planner_model == "gpt-4o"


def test_resolved_planner_model_defaults_to_stronger_anthropic_model(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "planner_llm_model", None)
    assert settings.resolved_planner_model == "claude-3-5-sonnet-20241022"


def test_resolved_planner_model_respects_explicit_override(monkeypatch):
    monkeypatch.setattr(settings, "planner_llm_model", "gpt-4-turbo")
    assert settings.resolved_planner_model == "gpt-4-turbo"


def test_resolved_planner_model_differs_from_default_agent_model(monkeypatch):
    """The whole point of tiering: the planner shouldn't silently end up
    on the same cost-optimized model as everything else."""
    monkeypatch.setattr(settings, "llm_model", "gpt-4o-mini")
    monkeypatch.setattr(settings, "planner_llm_model", None)
    assert settings.resolved_planner_model != settings.llm_model


def test_resolved_guardrail_model_defaults_to_stronger_openai_model(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "guardrail_llm_model", None)
    assert settings.resolved_guardrail_model == "gpt-4o"


def test_resolved_guardrail_model_defaults_to_stronger_anthropic_model(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "guardrail_llm_model", None)
    assert settings.resolved_guardrail_model == "claude-3-5-sonnet-20241022"


def test_resolved_guardrail_model_respects_explicit_override(monkeypatch):
    monkeypatch.setattr(settings, "guardrail_llm_model", "claude-opus-4-6")
    assert settings.resolved_guardrail_model == "claude-opus-4-6"


def test_planner_and_guardrail_overrides_are_independent(monkeypatch):
    monkeypatch.setattr(settings, "planner_llm_model", "gpt-4-turbo")
    monkeypatch.setattr(settings, "guardrail_llm_model", "gpt-4o")
    assert settings.resolved_planner_model == "gpt-4-turbo"
    assert settings.resolved_guardrail_model == "gpt-4o"


def test_tiering_does_not_affect_eval_model_resolution(monkeypatch):
    """resolved_eval_model has its own (provider-diversity) logic --
    setting a planner/guardrail override must not leak into it."""
    monkeypatch.setattr(settings, "planner_llm_model", "gpt-4-turbo")
    monkeypatch.setattr(settings, "guardrail_llm_model", "gpt-4-turbo")
    assert settings.resolved_eval_model != "gpt-4-turbo"
