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
