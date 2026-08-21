"""
TB3+TB4 tests — ECOSYSTEM_INTEGRATION_PLAN.md 附录 U (frozen contract) /
附录 V (cleaner's real response shapes).

Covers the MAEDA-side landing: P.1 deterministic column resolution, P.2
intent payload construction, U.3 same-round scope_fingerprint verification
(including the re-profile call T.2 called out as the easy-to-forget one),
U.4.3 critical_basis, R.3's cleaning_applied_level, and U.7's Insight
caveats disclosure chain.

Run with: pytest tests/unit/test_tb3_tb4.py -v
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.tools.data_connector import ColumnInfo


def _col(name, is_datetime=False, is_numeric=False):
    return ColumnInfo(
        name=name, dtype="object", null_pct=0.0, unique_count=1,
        sample_values=[], is_numeric=is_numeric, is_datetime=is_datetime,
    )


# ─── P.1: _resolve_intent_columns ──────────────────────────────────────────────

class TestResolveIntentColumns:
    def test_exact_match(self):
        from src.graph.nodes import _resolve_intent_columns
        resolved, unresolved, status = _resolve_intent_columns(
            {"target_metrics": ["revenue"]}, [_col("revenue"), _col("region")],
        )
        assert resolved == [{"name": "revenue", "role": "metric", "matched_from": "revenue", "match": "exact"}]
        assert unresolved == []
        assert status == "full"

    def test_case_insensitive_match(self):
        from src.graph.nodes import _resolve_intent_columns
        resolved, unresolved, status = _resolve_intent_columns(
            {"dimensions": ["Region"]}, [_col("region")],
        )
        assert resolved == [{"name": "region", "role": "dimension", "matched_from": "Region", "match": "case_insensitive"}]
        assert status == "full"

    def test_no_match_goes_to_unresolved(self):
        from src.graph.nodes import _resolve_intent_columns
        resolved, unresolved, status = _resolve_intent_columns(
            {"target_metrics": ["net_income"]}, [_col("revenue")],
        )
        assert resolved == []
        assert unresolved == [{"text": "net_income", "role": "metric", "reason": "no_match", "candidates": []}]
        # 附录 U.4.2: cleaner's _resolve_scope() treats empty columns as
        # "absent" regardless of what this field says -- nothing resolved
        # must report "absent", not "partial".
        assert status == "absent"

    def test_partial_when_some_resolve_and_some_dont(self):
        from src.graph.nodes import _resolve_intent_columns
        resolved, unresolved, status = _resolve_intent_columns(
            {"target_metrics": ["revenue", "net_income"]}, [_col("revenue")],
        )
        assert [r["name"] for r in resolved] == ["revenue"]
        assert [u["text"] for u in unresolved] == ["net_income"]
        assert status == "partial"

    def test_absent_when_no_mentions_at_all(self):
        from src.graph.nodes import _resolve_intent_columns
        resolved, unresolved, status = _resolve_intent_columns({}, [_col("revenue")])
        assert resolved == unresolved == []
        assert status == "absent"

    def test_absent_when_schema_extraction_failed(self):
        """P.1's documented exception case: connect_and_profile_node passes
        an empty column list when schema extraction raised."""
        from src.graph.nodes import _resolve_intent_columns
        resolved, unresolved, status = _resolve_intent_columns(
            {"target_metrics": ["revenue"]}, [],
        )
        assert resolved == []
        assert unresolved[0]["reason"] == "no_match"
        assert status == "absent"

    def test_datetime_column_gets_time_role_regardless_of_source_field(self):
        """role is upgraded to "time" purely from the column's own dtype --
        ParsedIntent has no separate list of time mentions to read from."""
        from src.graph.nodes import _resolve_intent_columns
        resolved, _, _ = _resolve_intent_columns(
            {"dimensions": ["order_date"]}, [_col("order_date", is_datetime=True)],
        )
        assert resolved[0]["role"] == "time"

    def test_filters_column_field_extracted_with_filter_role(self):
        from src.graph.nodes import _resolve_intent_columns
        resolved, _, _ = _resolve_intent_columns(
            {"filters": [{"column": "region", "op": "==", "value": "North"}]},
            [_col("region")],
        )
        assert resolved == [{"name": "region", "role": "filter", "matched_from": "region", "match": "exact"}]

    def test_sort_by_extracted_as_dimension_mention(self):
        from src.graph.nodes import _resolve_intent_columns
        resolved, _, _ = _resolve_intent_columns({"sort_by": "revenue"}, [_col("revenue")])
        assert resolved == [{"name": "revenue", "role": "dimension", "matched_from": "revenue", "match": "exact"}]

    def test_same_column_mentioned_twice_yields_one_entry(self):
        from src.graph.nodes import _resolve_intent_columns
        resolved, _, _ = _resolve_intent_columns(
            {"target_metrics": ["revenue"], "sort_by": "revenue"}, [_col("revenue")],
        )
        assert len(resolved) == 1


# ─── P.2: _build_intent_payload ────────────────────────────────────────────────

class TestBuildIntentPayload:
    def test_frozen_shape(self):
        from src.graph.nodes import _build_intent_payload
        resolved = [{"name": "revenue", "role": "metric", "matched_from": "revenue", "match": "exact"}]
        unresolved = [{"text": "net_income", "role": "metric", "reason": "no_match", "candidates": []}]
        payload = _build_intent_payload(
            "/data/x.csv", {"query_type": "diagnostic"}, resolved, unresolved, "partial",
        )
        assert payload == {
            "intent_contract_version": "1",
            "dataset_path": "/data/x.csv",
            "detected_issues": [],
            "analysis": {
                "query_type": "diagnostic",
                "resolution_status": "partial",
                "resolved_columns": resolved,
                "unresolved_mentions": unresolved,
            },
            "column_scope": {
                "columns": ["revenue"],
                # 附录 U.2/U.9 frozen literal values -- not configurable.
                "profiling": "restrict",
                "mutation": "advisory",
            },
            "glossary": [],
            # 阶段 3 轮次 4 (附录 CQ): additive top-level key, permitted by
            # 附录 U.11 without moving intent_contract_version. Defaults to
            # "absent" and is NEVER omitted -- an empty `glossary` alone can't
            # tell "no curated definition exists" apart from "the glossary
            # wasn't attached", and the second reading is what lets a planner
            # invent column semantics unchallenged.
            "glossary_coverage": "absent",
        }

    def test_frozen_u2_keys_are_untouched_by_the_glossary_round(self):
        """附录 U.11 allows additive changes only. Whatever 轮次 4 added, every
        key 附录 U.2 froze must still be present and unchanged in meaning --
        this pins the frozen surface separately from the exact dict above, so
        a future additive change can't quietly drop one of them."""
        from src.graph.nodes import _build_intent_payload
        payload = _build_intent_payload("/data/x.csv", {}, [], [], "absent")
        for key in ("intent_contract_version", "dataset_path", "detected_issues",
                    "analysis", "column_scope", "glossary"):
            assert key in payload
        assert payload["intent_contract_version"] == "1"   # unchanged: additive only
        assert set(payload["analysis"]) == {
            "query_type", "resolution_status", "resolved_columns", "unresolved_mentions",
        }
        assert set(payload["column_scope"]) == {"columns", "profiling", "mutation"}

    def test_query_type_defaults_to_exploratory(self):
        from src.graph.nodes import _build_intent_payload
        payload = _build_intent_payload("/x.csv", {}, [], [], "absent")
        assert payload["analysis"]["query_type"] == "exploratory"


# ─── U.3: _scope_fingerprint / _check_scope_fingerprint ────────────────────────

class TestScopeFingerprint:
    def test_deterministic_regardless_of_input_order(self):
        from src.graph.nodes import _scope_fingerprint
        assert _scope_fingerprint(["b", "a"]) == _scope_fingerprint(["a", "b"])

    def test_matches_frozen_algorithm(self):
        """附录 U.3: sha256(json.dumps(sorted(columns), ensure_ascii=False,
        separators=(",", ":")))[:16] -- must byte-for-byte match cleaner's
        own _scope_fingerprint() (mcp_app.py), not just "look like a hash"."""
        import hashlib
        import json as _json
        from src.graph.nodes import _scope_fingerprint

        columns = ["region", "revenue"]
        expected = hashlib.sha256(
            _json.dumps(sorted(columns), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        assert _scope_fingerprint(columns) == expected

    def test_empty_columns_is_a_stable_constant(self):
        from src.graph.nodes import _scope_fingerprint
        assert _scope_fingerprint([]) == _scope_fingerprint([])
        assert len(_scope_fingerprint([])) == 16


class TestCheckScopeFingerprint:
    def test_fallback_mode_never_checked(self):
        from src.graph.nodes import _check_scope_fingerprint
        from src.state.graph_state import initial_state
        err = _check_scope_fingerprint(initial_state("q"), "profile_dataset", "abc123", "different", "fallback")
        assert err is None

    def test_none_actual_means_server_predates_tb3(self):
        from src.graph.nodes import _check_scope_fingerprint
        from src.state.graph_state import initial_state
        err = _check_scope_fingerprint(initial_state("q"), "profile_dataset", "abc123", None, "mcp")
        assert err is None

    def test_matching_fingerprints_pass(self):
        from src.graph.nodes import _check_scope_fingerprint
        from src.state.graph_state import initial_state
        err = _check_scope_fingerprint(initial_state("q"), "profile_dataset", "abc123", "abc123", "mcp")
        assert err is None

    def test_mismatch_returns_error_message(self):
        """This is the actual T.2/U.3 failure mode: a call in this round
        got a different (or missing) intent than the round's other calls."""
        from src.graph.nodes import _check_scope_fingerprint
        from src.state.graph_state import initial_state
        err = _check_scope_fingerprint(initial_state("q"), "profile_dataset(re-profile)", "abc123", "xyz789", "mcp")
        assert err is not None
        assert "abc123" in err and "xyz789" in err


# ─── R.3: _cleaning_applied_level ──────────────────────────────────────────────

class TestCleaningAppliedLevel:
    def test_none_when_never_applied(self):
        from src.graph.nodes import _cleaning_applied_level
        from src.state.graph_state import initial_state
        assert _cleaning_applied_level(initial_state("q")) == "none"

    def test_full_when_applied_and_not_blocked(self):
        from src.graph.nodes import _cleaning_applied_level
        from src.state.graph_state import initial_state
        state = initial_state("q")
        state["cleaning_applied"] = True
        state["cleaning_stop_reason"] = "passed"
        assert _cleaning_applied_level(state) == "full"

    def test_blocked_needs_review_takes_priority(self):
        from src.graph.nodes import _cleaning_applied_level
        from src.state.graph_state import initial_state
        state = initial_state("q")
        state["cleaning_applied"] = True  # e.g. a prior round succeeded
        state["cleaning_stop_reason"] = "needs_review"
        assert _cleaning_applied_level(state) == "blocked_needs_review"


# ─── Integration: connect_and_profile_node builds + reuses one intent ─────────

class TestIntentFlowsThroughCleaningLoop:
    """Drives connect_and_profile_node directly (same harness pattern as
    TestCleaningLoop in test_phase4.py) to prove the intent payload built
    from a real ParsedIntent is the literal same object sent to every MCP
    call in the round, and that cleaner's echoed scope_fingerprint is
    actually being verified, not just plumbed through unchecked."""

    @staticmethod
    def _mock_mcp(**method_side_effects):
        from src.mcp_client.fallback import SubSystemWithFallback
        mock = MagicMock(spec=SubSystemWithFallback)
        for name, effect in method_side_effects.items():
            setattr(mock, name, AsyncMock(side_effect=effect))
        return mock

    @staticmethod
    def _log(tool):
        return {"system": "data_cleaner", "tool": tool,
                "mode": "mcp", "args": {}, "duration_ms": 1.0}

    @staticmethod
    def _run(state, mock_mcp):
        import src.graph.nodes as nodes
        from src.tools.data_connector import DataConnector
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Summary."))
        old_connector = nodes._data_connector
        old_mcp = nodes._subsystem_client
        nodes._data_connector = DataConnector(llm=mock_llm)
        nodes._subsystem_client = mock_mcp
        try:
            return asyncio.run(nodes.connect_and_profile_node(state))
        finally:
            nodes._data_connector = old_connector
            nodes._subsystem_client = old_mcp

    @pytest.fixture(autouse=True)
    def _artifact_root_in_tmp(self, tmp_path, monkeypatch):
        from src.config.settings import settings as _settings
        monkeypatch.setattr(_settings, "maeda_artifact_root", str(tmp_path))

    def test_same_intent_object_sent_to_every_call_this_round(self, tmp_path):
        from src.graph.nodes import _scope_fingerprint
        from src.mcp_client.models import CleaningResult, DataQualityReport, QualityValidation
        from src.state.graph_state import initial_state

        dirty = tmp_path / "dirty.csv"
        dirty.write_text("revenue,region\n,North\n,South\n1,East\n2,West\n3,North\n")

        state = initial_state("q", data_sources=[{"type": "csv", "path": str(dirty)}])
        state["parsed_intent"] = {"query_type": "diagnostic", "target_metrics": ["revenue"]}

        fp = _scope_fingerprint(["revenue"])  # what the node will actually expect
        d = tmp_path / f"{state['run_id']}_clean1" / "stage2" / "data_versions"
        d.mkdir(parents=True)
        cleaned = d / "cleaned.csv"
        cleaned.write_text("revenue,region\n0,North\n0,South\n1,East\n2,West\n3,North\n")

        dirty_report = DataQualityReport(row_count=5, columns=[], quality_issues=[], has_critical_issues=True, scope_fingerprint=fp)
        clean_result = CleaningResult(cleaned_path=str(cleaned), changes_summary={}, rows_affected=0, scope_fingerprint=fp)
        clean_report = DataQualityReport(row_count=5, columns=[], quality_issues=[], has_critical_issues=False, scope_fingerprint=fp)
        validation = QualityValidation(passed=True, score=1.0, issues=[], details={}, scope_fingerprint=fp)

        mock_mcp = self._mock_mcp(
            profile_dataset=[(dirty_report, self._log("profile_dataset")),
                              (clean_report, self._log("profile_dataset"))],
            clean_dataset=[(clean_result, self._log("clean_dataset"))],
            validate_quality=[(validation, self._log("validate_quality"))],
        )

        result = self._run(state, mock_mcp)

        assert result["cleaning_stop_reason"] == "passed"
        assert result.get("error") is None

        intents_seen = (
            [c.kwargs.get("intent") for c in mock_mcp.profile_dataset.call_args_list]
            + [mock_mcp.clean_dataset.call_args.kwargs.get("intent")]
            + [mock_mcp.validate_quality.call_args.kwargs.get("intent")]
        )
        assert all(i is not None for i in intents_seen)
        assert all(i is intents_seen[0] for i in intents_seen), (
            "every call this round must carry the literal same intent object (附录 U.3)"
        )
        assert intents_seen[0]["column_scope"]["columns"] == ["revenue"]
        assert result["cleaning_intent"]["column_scope"]["columns"] == ["revenue"]
        assert result["cleaning_execution_plan"] == {}

    def test_reprofile_fingerprint_mismatch_is_a_hard_failure(self, tmp_path):
        """附录 T.2's exact scenario: re-profile (the second profile_dataset
        call this round) returns a different scope_fingerprint than the
        round's other calls -- this must be caught, not silently accepted."""
        from src.graph.nodes import _scope_fingerprint
        from src.mcp_client.models import CleaningResult, DataQualityReport, QualityValidation
        from src.state.graph_state import initial_state

        dirty = tmp_path / "dirty.csv"
        dirty.write_text("revenue,region\n,North\n,South\n1,East\n2,West\n3,North\n")

        state = initial_state("q", data_sources=[{"type": "csv", "path": str(dirty)}])
        state["parsed_intent"] = {"target_metrics": ["revenue"]}

        d = tmp_path / f"{state['run_id']}_clean1" / "stage2" / "data_versions"
        d.mkdir(parents=True)
        cleaned = d / "cleaned.csv"
        cleaned.write_text("revenue,region\n0,North\n0,South\n1,East\n2,West\n3,North\n")

        # correct_fp is what the node will actually expect (real algorithm,
        # real resolved columns) -- using a made-up value here would make
        # the very first profile_dataset call fail the check, never
        # reaching clean_dataset/re-profile at all, and this test would
        # stop proving what its name says it proves.
        correct_fp = _scope_fingerprint(["revenue"])
        wrong_fp = "0000000000000000"
        assert wrong_fp != correct_fp
        dirty_report = DataQualityReport(row_count=5, columns=[], quality_issues=[], has_critical_issues=True, scope_fingerprint=correct_fp)
        clean_result = CleaningResult(cleaned_path=str(cleaned), changes_summary={}, rows_affected=0, scope_fingerprint=correct_fp)
        # re-profile "forgot" the round's intent and got a different scope back
        reprofile_report = DataQualityReport(row_count=5, columns=[], quality_issues=[], has_critical_issues=False, scope_fingerprint=wrong_fp)

        mock_mcp = self._mock_mcp(
            profile_dataset=[(dirty_report, self._log("profile_dataset")),
                              (reprofile_report, self._log("profile_dataset"))],
            clean_dataset=[(clean_result, self._log("clean_dataset"))],
            validate_quality=[(QualityValidation(passed=True, score=1.0, issues=[]), self._log("validate_quality"))],
        )

        result = self._run(state, mock_mcp)

        assert result["current_phase"] == "error"
        assert "scope_fingerprint mismatch" in result["error"]
        mock_mcp.validate_quality.assert_not_awaited(), "must not proceed past a contract violation"


# ─── U.7: _build_cleaning_caveats + Insight.caveats ────────────────────────────

class TestBuildCleaningCaveats:
    def test_no_execution_plan_returns_empty(self):
        from src.agents.insight_agent import _build_cleaning_caveats
        assert _build_cleaning_caveats(None, []) == []
        assert _build_cleaning_caveats({}, []) == []

    def test_lossy_step_with_impact_produces_info_caveat(self):
        from src.agents.insight_agent import _build_cleaning_caveats
        plan = {"steps": [{
            "mcp_tool": "handle_missing", "lossy": True, "risk_tier": "lossy_reversible",
            "escalated_by": [], "impact": {"rows_removed": 0, "rows_modified": 12, "cells_changed": 12},
        }]}
        caveats = _build_cleaning_caveats(plan, [])
        assert len(caveats) == 1
        assert caveats[0]["code"] == "imputed_values"
        assert caveats[0]["severity"] == "info"

    def test_lossy_step_with_zero_impact_produces_no_caveat(self):
        """R.2: disclosure is forced for operations that actually did
        something -- a lossy tool that ran and touched nothing isn't lying
        by omission if it stays silent."""
        from src.agents.insight_agent import _build_cleaning_caveats
        plan = {"steps": [{
            "mcp_tool": "deduplicate", "lossy": True, "risk_tier": "lossy_reversible",
            "escalated_by": [], "impact": {"rows_removed": 0, "rows_modified": 0, "cells_changed": 0},
        }]}
        assert _build_cleaning_caveats(plan, []) == []

    def test_safe_step_produces_no_caveat(self):
        from src.agents.insight_agent import _build_cleaning_caveats
        plan = {"steps": [{
            "mcp_tool": "normalize_schema", "lossy": False, "risk_tier": "safe",
            "escalated_by": [], "impact": {"rows_removed": 0, "rows_modified": 0, "cells_changed": 5},
        }]}
        assert _build_cleaning_caveats(plan, []) == []

    def test_high_risk_step_produces_warning_caveat(self):
        from src.agents.insight_agent import _build_cleaning_caveats
        plan = {"steps": [{
            "mcp_tool": "handle_outliers", "lossy": True, "risk_tier": "high_risk",
            "escalated_by": ["impact_threshold"], "impact": {"rows_removed": 0, "rows_modified": 0, "cells_changed": 0},
            "confidence": 0.4,
        }]}
        caveats = _build_cleaning_caveats(plan, [])
        assert len(caveats) == 1
        assert caveats[0]["code"] == "cleaning_step_blocked"
        assert caveats[0]["severity"] == "warning"
        assert caveats[0]["evidence"]["escalated_by"] == ["impact_threshold"]

    def test_scope_hit_attributes_columns(self):
        from src.agents.insight_agent import _build_cleaning_caveats
        plan = {"steps": [{
            "mcp_tool": "handle_missing", "lossy": True, "risk_tier": "lossy_reversible",
            "escalated_by": ["scope_hit"], "impact": {"rows_removed": 0, "rows_modified": 9, "cells_changed": 9},
        }]}
        caveats = _build_cleaning_caveats(plan, ["revenue"])
        assert caveats[0]["columns"] == ["revenue"]

    def test_no_scope_hit_leaves_columns_empty(self):
        from src.agents.insight_agent import _build_cleaning_caveats
        plan = {"steps": [{
            "mcp_tool": "handle_missing", "lossy": True, "risk_tier": "lossy_reversible",
            "escalated_by": [], "impact": {"rows_removed": 0, "rows_modified": 9, "cells_changed": 9},
        }]}
        caveats = _build_cleaning_caveats(plan, ["revenue"])
        assert caveats[0]["columns"] == []

    def test_step_can_produce_both_caveats(self):
        """A step blocked by the confidence gate (rule 4) is both lossy-typed
        and high_risk, but never produced impact -- only the block caveat
        should fire, not a false "modified the data" disclosure."""
        from src.agents.insight_agent import _build_cleaning_caveats
        plan = {"steps": [{
            "mcp_tool": "handle_missing", "lossy": True, "risk_tier": "high_risk",
            "escalated_by": ["confidence_gate"], "impact": {"rows_removed": 0, "rows_modified": 0, "cells_changed": 0},
        }]}
        caveats = _build_cleaning_caveats(plan, [])
        assert len(caveats) == 1
        assert caveats[0]["code"] == "cleaning_step_blocked"


class TestInsightCaveatsField:
    def test_defaults_to_empty_list(self):
        from src.agents.insight_agent import Insight
        ins = Insight(finding="x", evidence=[], confidence=0.7, domain_context="",
                       impact="medium", recommendation="", sources=[])
        assert ins.caveats == []

    def test_from_dict_reads_caveats(self):
        from src.agents.insight_agent import Insight
        d = {"finding": "x", "confidence": 0.8, "caveats": [{"code": "imputed_values"}]}
        ins = Insight.from_dict(d)
        assert ins.caveats == [{"code": "imputed_values"}]

    def test_to_dict_round_trip_includes_caveats(self):
        from src.agents.insight_agent import Insight
        ins = Insight(
            finding="x", evidence=[], confidence=0.7, domain_context="",
            impact="medium", recommendation="", sources=[],
            caveats=[{"code": "imputed_values", "severity": "info"}],
        )
        d = ins.to_dict()
        assert d["caveats"] == [{"code": "imputed_values", "severity": "info"}]


def test_generate_attaches_cleaning_caveats_to_every_insight():
    """End-to-end through InsightAgent.generate(): cleaning_execution_plan
    in state must reach every insight's caveats field, not just sit in
    decision_trace (附录 U.7's whole point)."""
    import json as _json

    from src.agents.insight_agent import InsightAgent

    mock_llm = MagicMock()
    insight_response = MagicMock()
    insight_response.content = _json.dumps([
        {"finding": "A", "evidence": "e", "confidence": 0.8, "recommendation": "r"},
        {"finding": "B", "evidence": "e", "confidence": 0.8, "recommendation": "r"},
    ])
    insight_response.usage_metadata = {"input_tokens": 1, "output_tokens": 1}
    report_response = MagicMock()
    report_response.content = "# report"
    report_response.usage_metadata = {"input_tokens": 1, "output_tokens": 1}
    mock_llm.ainvoke = AsyncMock(side_effect=[insight_response, report_response])

    from src.state.graph_state import initial_state
    state = initial_state("q")
    state["analysis_results"] = []
    state["rag_context"] = []
    state["rag_sources"] = []
    state["cleaning_execution_plan"] = {"steps": [{
        "mcp_tool": "handle_missing", "lossy": True, "risk_tier": "lossy_reversible",
        "escalated_by": [], "impact": {"rows_removed": 0, "rows_modified": 12, "cells_changed": 12},
    }]}
    state["cleaning_intent"] = {"column_scope": {"columns": []}}

    agent = InsightAgent(llm=mock_llm)
    result = asyncio.run(agent.generate(state))

    assert len(result["insights"]) == 2
    for insight in result["insights"]:
        assert insight["caveats"] == [{
            "code": "imputed_values", "severity": "info", "columns": [],
            "detail": "handle_missing modified the data: "
                      "{'rows_removed': 0, 'rows_modified': 12, 'cells_changed': 12}",
            "source": "cleaner",
            "evidence": {"risk_tier": "lossy_reversible", "rows_removed": 0, "rows_modified": 12, "cells_changed": 12},
        }]
