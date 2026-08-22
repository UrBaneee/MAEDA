"""
Terminal run states — ECOSYSTEM_INTEGRATION_PLAN.md 阶段 3 剩余量 E3
(执行顺序表轮次 5, 附录 CU).

E3 asks for six terminal states: success / safe refusal / pipeline error /
mcp error / environment error / guardrail rejection. This module defines
**five**, and the missing one is a deliberate, evidence-backed refusal to
create a second name for something that already has one:

  guardrail rejection IS the safe refusal, in this repository.
      src/graph/nodes.py::handle_error_node derives `safe_refusal` from
      exactly one condition -- `guardrail_checks[-1]["overall_verdict"]
      == "fail"` -- and src/graph/builder.py routes to handle_error from
      run_guardrails on exactly one key ("fail", route_after_guardrails).
      There is no input-side refusal path: src/agents/guardrail_agent.py's
      checks all run on `report`/`insights`/`analysis_results`, i.e. after
      generation. So "guardrail rejection" and "safe refusal" name the
      same event today, and splitting them would produce one state that is
      structurally unreachable plus two names for one thing -- the
      同名不同义 hazard 硬约束 7 exists to prevent, in mirror image.
      CONDITION THAT WOULD CHANGE THIS: an input-side / pre-generation
      refusal path (declining an out-of-scope or unsafe query before
      analysis). That is a genuinely different event from "we produced a
      report and the guardrail would not let it out", and when one lands,
      SAFE_REFUSAL should split rather than be overloaded.

REUSE, NOT A SECOND ERROR MATRIX
    The sub-classification carried in `terminal_detail` is NOT a new
    taxonomy. For every failure that came out of a sub-system call it is
    verbatim `SubSystemHardFailure.error_class`, i.e. one of the four
    values src/mcp_client/fallback.py::_classify already produces
    ("connection" / "data_input" / "contract" / "internal_unknown", see
    that file's 错误处理矩阵 comments). `_DETAIL_TO_TERMINAL` below maps
    those four onto terminal states; it does not re-derive them.
    tests/unit/test_e3_terminal_states.py drives fallback._classify itself
    and asserts every class it can return is a key here, so a fifth error
    class cannot be added there and silently fall through to
    "pipeline_error" here.

WHY data_input MAPS TO environment_error AND NOT mcp_error
    The 阶段 3 line under E3 wants `mcp_error`/`environment_error` counted
    separately from agent reasoning failures precisely so that MCP
    可用率/超时率/fallback 率 can be read off them. A `data_input` failure
    means the call round-tripped fine and the sub-system correctly answered
    "this file is unreadable" -- counting it as an MCP error would
    understate MCP availability by blaming the transport for a bad input.
    So: mcp_error = the call itself failed; environment_error = the inputs
    or environment this run was handed are unusable (no data source at all,
    or a file no one can read).
"""
from __future__ import annotations

from typing import Optional

SUCCESS = "success"
SAFE_REFUSAL = "safe_refusal"
PIPELINE_ERROR = "pipeline_error"
MCP_ERROR = "mcp_error"
ENVIRONMENT_ERROR = "environment_error"

TERMINAL_STATES = (SUCCESS, SAFE_REFUSAL, PIPELINE_ERROR, MCP_ERROR, ENVIRONMENT_ERROR)

# `terminal_detail` values MAEDA sets itself (everything else in this table
# comes from fallback.py, see the module docstring).
DETAIL_NO_DATA_SOURCE = "no_data_source"
DETAIL_GUARDRAIL_FAIL = "guardrail_fail"
DETAIL_SCOPE_FINGERPRINT_MISMATCH = "scope_fingerprint_mismatch"

_DETAIL_TO_TERMINAL = {
    # src/mcp_client/fallback.py::_classify's four error_class values.
    "connection": MCP_ERROR,
    "contract": MCP_ERROR,
    "internal_unknown": MCP_ERROR,
    "data_input": ENVIRONMENT_ERROR,
    # MAEDA-side.
    DETAIL_NO_DATA_SOURCE: ENVIRONMENT_ERROR,
    # 附录 U.3's cross-round scope check (nodes.py::_check_scope_fingerprint).
    # Detected on our side, but what it detects is the two processes
    # disagreeing about the intent scope -- a contract violation, same
    # family as fallback.py's "contract".
    DETAIL_SCOPE_FINGERPRINT_MISMATCH: MCP_ERROR,
    DETAIL_GUARDRAIL_FAIL: SAFE_REFUSAL,
}


def resolve_terminal_state(state: dict) -> tuple[str, Optional[str]]:
    """
    THE gate for "how did this run end" -- one judgment site, several
    projections, the same pattern as `cleaner_should_attempt_clean`
    (src/graph/router.py, 附录 CC.2) and `resolve_glossary`
    (src/tools/glossary.py, 附录 CQ).

    Idempotent: if `state["terminal_state"]` is already set (handle_error_node
    stores it), that value is returned unchanged. Everything downstream --
    EvalRunner.score, persist_run_node, scripts/run_eval.py -- calls this
    rather than re-deriving the rule, so there is exactly one place where
    an MCP failure becomes `mcp_error`.

    Returns (terminal_state, terminal_detail). `terminal_detail` is None for
    a success and for a failure that carries no classifiable detail.
    """
    stored = state.get("terminal_state")
    if stored in TERMINAL_STATES:
        return stored, state.get("terminal_detail")

    detail = state.get("terminal_detail")

    if not state.get("error") and state.get("current_phase") != "error":
        return SUCCESS, None

    # A guardrail rejection is recognisable from the checks themselves even
    # when nothing set a detail (e.g. a state assembled by a caller rather
    # than by handle_error_node) -- same derivation handle_error_node used
    # before E3, kept here so the two can't disagree.
    checks = state.get("guardrail_checks") or []
    if checks and checks[-1].get("overall_verdict") == "fail":
        return SAFE_REFUSAL, detail or DETAIL_GUARDRAIL_FAIL

    return _DETAIL_TO_TERMINAL.get(detail, PIPELINE_ERROR), detail


def legacy_error_type(terminal_state: str) -> Optional[str]:
    """
    `state["error_type"]` projected from the terminal state, for the
    consumers frozen against its original two-value vocabulary:
    src/eval/metrics.py::score_system_metrics (safe_refusal / error_rate),
    scripts/run_eval.py's crash reporting, and the `error_type` column of
    every row already in logs/runs.db.

    E3 NARROWS THE MEANING OF THE STORED VALUE `pipeline_error`, and that
    has to be registered rather than discovered later (same class of
    silent-semantics-drift as 附录 BO.5's intent_accuracy note): rows
    written BEFORE E3 use `pipeline_error` for every non-guardrail failure,
    including the MCP and environment failures that now get their own
    terminal state. `terminal_state IS NULL` is the marker for a
    pre-E3 row -- do not read an old `pipeline_error` as "agent reasoning
    failure".
    """
    if terminal_state == SUCCESS:
        return None
    if terminal_state == SAFE_REFUSAL:
        return "safe_refusal"
    return "pipeline_error"
