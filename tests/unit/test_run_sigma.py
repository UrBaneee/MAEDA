"""附录 GZ: sigma runner void rules."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import run_sigma as rs  # noqa: E402


def test_clean_run_is_not_void():
    assert rs.row_void({"mcp_modes": ["mcp"]}, {}, "A normal report", "B|D01|1", 900) is None


def test_mcp_fallback_is_void():
    assert rs.row_void({"mcp_modes": ["fallback"]}, {}, "r", "k", 900) == "mcp_fallback"


def test_rule_based_report_signature_is_void():
    rep = "# MAEDA Analysis Report\n\n## Executive Summary\nAnalysis completed for query: *x*\n"
    assert rs.row_void({"mcp_modes": ["mcp"]}, {}, rep, "k", 900) == "llm_report_fallback"


def test_logged_llm_fallback_is_void_even_with_normal_text():
    h = rs._FallbackHandler(level=logging.WARNING)
    tok = rs.CURRENT_KEY.set("C|X|1")
    try:
        h.emit(logging.LogRecord("n", logging.WARNING, "f", 1,
                                 "Report generation LLM failed: Connection error - using rule-based fallback", (), None))
    finally:
        rs.CURRENT_KEY.reset(tok)
    assert rs.row_void({"mcp_modes": ["mcp"]}, {}, "fine text", "C|X|1", 900) == "llm_report_fallback"
    rs.LLM_FALLBACKS.clear()


def test_run_longer_than_the_timeout_is_void():
    assert rs.row_void({"mcp_modes": ["mcp"], "elapsed_s": 33958}, {}, "fine", "k", 900) == "timeout"
    assert rs.row_void({"mcp_modes": ["mcp"], "elapsed_s": 840}, {}, "fine", "k", 900) is None


# 附录 HK / HL.5 / HM.5: holdout reveal mode. Nothing here runs a case or calls a model.
import pytest  # noqa: E402

from src.config.settings import settings  # noqa: E402
from src.eval.reveal_guard import REVEAL_ENV, RevealNotAuthorized  # noqa: E402


@pytest.fixture
def reveal_settings(monkeypatch):
    monkeypatch.setattr(settings, "rag_db_path", "/x/tb5_full.db")
    monkeypatch.setattr(settings, "intent_refine_trigger", "always")
    monkeypatch.setenv(REVEAL_ENV, "1")
    return monkeypatch


def test_holdout_preflight_passes_only_when_all_three_hold(reveal_settings):
    assert rs.preflight("holdout") is None


def test_holdout_refused_without_reveal_flag(reveal_settings):
    reveal_settings.delenv(REVEAL_ENV)
    assert REVEAL_ENV in rs.preflight("holdout")


def test_holdout_refused_unless_refine_always(reveal_settings):
    reveal_settings.setattr(settings, "intent_refine_trigger", "if_unresolved")
    assert "always" in rs.preflight("holdout")


def test_holdout_refused_on_dev_store(reveal_settings):
    reveal_settings.setattr(settings, "rag_db_path", "/x/tb5_dev.db")
    assert "tb5_full.db" in rs.preflight("holdout")


def test_dev_still_requires_dev_store(monkeypatch):
    monkeypatch.setattr(settings, "rag_db_path", "/x/tb5_full.db")
    assert "tb5_dev.db" in rs.preflight("dev")
    monkeypatch.setattr(settings, "rag_db_path", "/x/tb5_dev.db")
    assert rs.preflight("dev") is None


def test_loading_holdout_is_refused_without_flag(monkeypatch):
    monkeypatch.delenv(REVEAL_ENV, raising=False)
    with pytest.raises(RevealNotAuthorized):
        rs.load_sample("holdout")


def test_holdout_has_119_test_cases_when_authorized(monkeypatch):
    monkeypatch.setenv(REVEAL_ENV, "1")
    cases = rs.load_sample("holdout")
    assert len(cases) == 119 and all(c.split == "test" for c in cases)


def test_arm_a_runs_twice_on_holdout_others_at_k():
    assert rs.repeats_for("A", 3, "holdout") == 2
    assert rs.repeats_for("CR", 3, "holdout") == 3
    assert rs.repeats_for("A", 3, "dev") == 3
    assert rs.ARMS["A"] == ("auto", "auto")
    assert rs.DEFAULT_ARMS["holdout"] == ["B", "C", "R", "CR", "A"]
    assert rs.DEFAULT_ARMS["dev"] == ["B", "C", "R", "CR"]
