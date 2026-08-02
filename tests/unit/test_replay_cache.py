"""
Tests for src/eval/replay_cache.py -- the fingerprint + cache that lets
Step 2 (eval v2, docs/eval_v2_plan.md) iterate on scoring logic without
regenerating reports from the real pipeline every time.
"""
from pathlib import Path

import pytest

from src.eval.replay_cache import ReplayCache, generation_fingerprint


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    """A minimal src/ + data/ tree, isolated from the real repo, so
    fingerprint tests don't depend on (or get slowed down by) hashing the
    actual project source."""
    _write(tmp_path / "src" / "agents" / "foo.py", "x = 1\n")
    _write(tmp_path / "src" / "eval" / "metrics.py", "y = 1\n")  # must be excluded
    _write(
        tmp_path / "src" / "config" / "agent_prompts.py",
        'PLANNER_SYSTEM = """\\\noriginal planner prompt\n"""\n'
        'EVAL_RELEVANCE_SYSTEM = """\\\noriginal eval prompt\n"""\n',
    )
    _write(tmp_path / "data" / "demo" / "x.csv", "a,b\n1,2\n")

    monkeypatch.setattr("src.eval.replay_cache._REPO_ROOT", tmp_path)
    monkeypatch.setattr("src.eval.replay_cache._GENERATION_SRC_ROOT", tmp_path / "src")
    monkeypatch.setattr("src.eval.replay_cache._EXCLUDED_SRC_SUBTREE", tmp_path / "src" / "eval")

    from src.config.settings import settings
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "llm_model", "gpt-4o-mini")
    monkeypatch.setattr(settings, "llm_temperature", 0.0)
    monkeypatch.setattr(settings, "planner_llm_model", None)
    monkeypatch.setattr(settings, "guardrail_llm_model", None)
    monkeypatch.setattr(settings, "max_tokens_per_call", 4096)

    return tmp_path


def test_fingerprint_is_stable_across_repeated_calls(fake_repo):
    assert generation_fingerprint() == generation_fingerprint()


def test_fingerprint_changes_when_generation_source_changes(fake_repo):
    before = generation_fingerprint()
    _write(fake_repo / "src" / "agents" / "foo.py", "x = 2\n")
    assert generation_fingerprint() != before


def test_fingerprint_ignores_src_eval_subtree(fake_repo):
    """Changing src/eval/ must never invalidate cached reports -- that's
    exactly the code Step 2 iterates on."""
    before = generation_fingerprint()
    _write(fake_repo / "src" / "eval" / "metrics.py", "y = 999\n")
    assert generation_fingerprint() == before


def test_fingerprint_ignores_eval_prefixed_prompt_but_sees_other_prompts(fake_repo):
    """agent_prompts.py mixes generation prompts with EVAL_RELEVANCE_SYSTEM
    (the judge's prompt). Editing the judge prompt must not force a
    regeneration; editing a generation prompt must."""
    before = generation_fingerprint()

    _write(
        fake_repo / "src" / "config" / "agent_prompts.py",
        'PLANNER_SYSTEM = """\\\noriginal planner prompt\n"""\n'
        'EVAL_RELEVANCE_SYSTEM = """\\\nCOMPLETELY DIFFERENT eval prompt now\n"""\n',
    )
    assert generation_fingerprint() == before, "editing only EVAL_* prompt changed the fingerprint"

    _write(
        fake_repo / "src" / "config" / "agent_prompts.py",
        'PLANNER_SYSTEM = """\\\nCOMPLETELY DIFFERENT planner prompt\n"""\n'
        'EVAL_RELEVANCE_SYSTEM = """\\\noriginal eval prompt\n"""\n',
    )
    assert generation_fingerprint() != before, "editing a generation prompt did not change the fingerprint"


def test_fingerprint_changes_when_demo_data_changes(fake_repo):
    before = generation_fingerprint()
    _write(fake_repo / "data" / "demo" / "x.csv", "a,b\n1,2\n3,4\n")
    assert generation_fingerprint() != before


def test_fingerprint_changes_when_generation_model_setting_changes(fake_repo, monkeypatch):
    from src.config.settings import settings
    before = generation_fingerprint()
    monkeypatch.setattr(settings, "llm_model", "gpt-4o")
    assert generation_fingerprint() != before


def test_fingerprint_ignores_eval_only_settings(fake_repo, monkeypatch):
    """eval_judge_samples / eval_llm_model are Step 2's territory --
    changing them must not force regeneration of cached reports."""
    from src.config.settings import settings
    before = generation_fingerprint()
    monkeypatch.setattr(settings, "eval_judge_samples", 1)
    monkeypatch.setattr(settings, "eval_llm_model", "some-other-model")
    assert generation_fingerprint() == before


def test_cache_put_and_get_roundtrip(fake_repo, tmp_path):
    cache = ReplayCache(path=tmp_path / "cache.json")
    cache.put("D01", {"report": "hello"}, elapsed_s=12.0, cost_usd=0.02)
    entry = cache.get("D01")
    assert entry is not None
    assert entry.state == {"report": "hello"}
    assert entry.cost_usd == 0.02


def test_cache_get_returns_none_for_missing_case(fake_repo, tmp_path):
    cache = ReplayCache(path=tmp_path / "cache.json")
    assert cache.get("NOPE") is None


def test_cache_get_returns_none_when_fingerprint_is_stale(fake_repo, tmp_path):
    cache = ReplayCache(path=tmp_path / "cache.json")
    cache.put("D01", {"report": "hello"})
    _write(fake_repo / "src" / "agents" / "foo.py", "x = 2\n")  # invalidating change
    assert cache.get("D01") is None


def test_cache_persists_to_disk_and_reloads(fake_repo, tmp_path):
    path = tmp_path / "cache.json"
    cache = ReplayCache(path=path)
    cache.put("D01", {"report": "hello"})
    cache.save()

    reloaded = ReplayCache(path=path)
    entry = reloaded.get("D01")
    assert entry is not None
    assert entry.state == {"report": "hello"}


def test_stale_or_missing_reports_both_kinds(fake_repo, tmp_path):
    cache = ReplayCache(path=tmp_path / "cache.json")
    cache.put("D01", {"report": "hello"})
    assert cache.stale_or_missing(["D01", "D02"]) == ["D02"]

    _write(fake_repo / "src" / "agents" / "foo.py", "x = 2\n")
    assert cache.stale_or_missing(["D01", "D02"]) == ["D01", "D02"]
