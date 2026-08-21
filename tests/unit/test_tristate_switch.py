"""
Three-state subsystem switch tests — 定案 #15, ECOSYSTEM_INTEGRATION_PLAN.md
阶段 3 收尾执行计划轮次 1, 附录 CB (design) / CC (frozen decisions) / CD
(implementation record).

Covers, per CC's frozen decisions:
  - settings defaults (auto/auto) and value validation
  - router.cleaner_should_attempt_clean: the single force_on/force_off/auto
    gate shared by profile_and_clean and route_after_profiling (CC.2/CC.3)
  - profile_and_clean: force_off makes NO clean_dataset call, regardless of
    has_critical_issues (CC.3 -- "不发起调用", not "发起但丢弃"); force_on
    bypasses has_critical_issues=False and calls clean_dataset anyway
    (CC.2 -- bypasses the JUDGEMENT ITSELF)
  - retrieve_knowledge_node: force_off makes NO RAG call; auto/force_on both
    call unconditionally (documented RAG "auto" degeneration, CC.7 裁定 4 --
    the conditional edge added by 轮次 3 does not change this, see
    tests/unit/test_rag_tier_and_routing.py)
  - `mode="skipped"` is recorded (not just an absent log entry) and is
    distinguishable from both "mcp" and "fallback" (附录 CB.1.3)
  - arm reporting: EvalResult top-level fields + state snapshot (per-report/
    per-row reporting lives in scripts/run_eval.py, covered separately in
    tests/unit/test_run_eval_script.py)
  - mcp_strict_mode x three-state interaction (附录 CB.3.3's 6-combination
    table): no combination raises or produces contradictory state

No real LLM/MCP calls anywhere in this file — SubSystemWithFallback and its
client methods are mocked throughout, the same harness pattern as
tests/unit/test_tb3_tb4.py / test_phase4.py's TestCleaningLoop.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest


# ─── settings ───────────────────────────────────────────────────────────────

class TestSettingsDefaults:
    def test_cleaner_mode_defaults_to_auto(self, monkeypatch):
        monkeypatch.delenv("MAEDA_CLEANER_MODE", raising=False)
        from src.config.settings import MAEDASettings
        assert MAEDASettings().maeda_cleaner_mode == "auto"

    def test_rag_mode_defaults_to_auto(self, monkeypatch):
        monkeypatch.delenv("MAEDA_RAG_MODE", raising=False)
        from src.config.settings import MAEDASettings
        assert MAEDASettings().maeda_rag_mode == "auto"

    def test_invalid_cleaner_mode_value_rejected(self, monkeypatch):
        monkeypatch.setenv("MAEDA_CLEANER_MODE", "sometimes")
        from src.config.settings import MAEDASettings
        with pytest.raises(Exception):
            MAEDASettings()

    def test_invalid_rag_mode_value_rejected(self, monkeypatch):
        monkeypatch.setenv("MAEDA_RAG_MODE", "sometimes")
        from src.config.settings import MAEDASettings
        with pytest.raises(Exception):
            MAEDASettings()

    def test_two_switches_are_independent_not_a_shared_enum(self, monkeypatch):
        """附录 CB.3.2: 定案 #15 says "统一 auto|force_on|force_off 三态"
        (a shared vocabulary), not one combined enum -- the two settings
        fields must be able to hold different values simultaneously."""
        monkeypatch.setenv("MAEDA_CLEANER_MODE", "force_on")
        monkeypatch.setenv("MAEDA_RAG_MODE", "force_off")
        from src.config.settings import MAEDASettings
        s = MAEDASettings()
        assert s.maeda_cleaner_mode == "force_on"
        assert s.maeda_rag_mode == "force_off"


# ─── router.cleaner_should_attempt_clean (the shared gate) ────────────────────

class TestCleanerShouldAttemptClean:
    def test_auto_follows_has_critical_issues(self, monkeypatch):
        from src.config.settings import settings as _settings
        from src.graph.router import cleaner_should_attempt_clean
        monkeypatch.setattr(_settings, "maeda_cleaner_mode", "auto")
        assert cleaner_should_attempt_clean(True) is True
        assert cleaner_should_attempt_clean(False) is False

    def test_force_on_bypasses_the_judgement_itself(self, monkeypatch):
        """CC.2: force_on must return True even when has_critical_issues is
        False -- not merely "unblocked", genuinely bypassing the judgement.
        This is the exact case CC.2 found matters: 12/13 real demo runs
        have has_critical_issues=False, so a force_on that only removed a
        blocking gate would behave identically to auto on real data."""
        from src.config.settings import settings as _settings
        from src.graph.router import cleaner_should_attempt_clean
        monkeypatch.setattr(_settings, "maeda_cleaner_mode", "force_on")
        assert cleaner_should_attempt_clean(False) is True
        assert cleaner_should_attempt_clean(True) is True

    def test_force_off_bypasses_the_judgement_the_other_way(self, monkeypatch):
        from src.config.settings import settings as _settings
        from src.graph.router import cleaner_should_attempt_clean
        monkeypatch.setattr(_settings, "maeda_cleaner_mode", "force_off")
        assert cleaner_should_attempt_clean(True) is False
        assert cleaner_should_attempt_clean(False) is False


class TestRouteAfterProfilingModes:
    @staticmethod
    def _state(**overrides):
        from src.state.graph_state import initial_state
        state = initial_state("q")
        state.update(overrides)
        return state

    def test_auto_unchanged_behavior(self, monkeypatch):
        """The default path must be byte-for-byte the pre-定案-#15 gate."""
        from src.config.settings import settings as _settings
        from src.graph.router import route_after_profiling
        monkeypatch.setattr(_settings, "maeda_cleaner_mode", "auto")
        critical = self._state(data_quality_report={"has_critical_issues": True}, iteration_count=0)
        assert route_after_profiling(critical) == "clean"
        not_critical = self._state(data_quality_report={"has_critical_issues": False}, iteration_count=0)
        assert route_after_profiling(not_critical) == "ready"

    def test_force_on_routes_clean_even_when_report_says_not_critical(self, monkeypatch):
        from src.config.settings import settings as _settings
        from src.graph.router import route_after_profiling
        monkeypatch.setattr(_settings, "maeda_cleaner_mode", "force_on")
        state = self._state(data_quality_report={"has_critical_issues": False}, iteration_count=0)
        assert route_after_profiling(state) == "clean"

    def test_force_on_still_respects_max_iterations(self, monkeypatch):
        from src.config.settings import settings as _settings
        from src.graph.router import route_after_profiling
        monkeypatch.setattr(_settings, "maeda_cleaner_mode", "force_on")
        state = self._state(data_quality_report={"has_critical_issues": True}, iteration_count=3)
        assert route_after_profiling(state) == "ready"

    def test_force_off_routes_ready_even_when_report_says_critical(self, monkeypatch):
        from src.config.settings import settings as _settings
        from src.graph.router import route_after_profiling
        monkeypatch.setattr(_settings, "maeda_cleaner_mode", "force_off")
        state = self._state(data_quality_report={"has_critical_issues": True}, iteration_count=0)
        assert route_after_profiling(state) == "ready"

    def test_existing_stop_reason_still_short_circuits_in_every_mode(self, monkeypatch):
        """A round that already hit a B.4 terminal stop condition must
        still route "ready" regardless of MAEDA_CLEANER_MODE -- the shared
        gate only matters for the has_critical_issues branch below it."""
        from src.config.settings import settings as _settings
        from src.graph.router import route_after_profiling
        for mode in ("auto", "force_on", "force_off"):
            monkeypatch.setattr(_settings, "maeda_cleaner_mode", mode)
            state = self._state(
                cleaning_stop_reason="passed",
                data_quality_report={"has_critical_issues": True}, iteration_count=1,
            )
            assert route_after_profiling(state) == "ready", mode


# ─── src.mcp_client.fallback.make_skipped_call_record ──────────────────────────

class TestMakeSkippedCallRecord:
    def test_shape_and_mode(self):
        from src.mcp_client.fallback import make_skipped_call_record
        rec = make_skipped_call_record(
            "data_cleaner", "clean_dataset", {"a": 1}, "MAEDA_CLEANER_MODE=force_off",
        )
        assert rec["mode"] == "skipped"
        assert rec["system"] == "data_cleaner"
        assert rec["tool"] == "clean_dataset"
        assert rec["error"] is None  # nothing failed -- the reason lives in result_summary
        assert "force_off" in rec["result_summary"]

    def test_skipped_is_distinguishable_from_mcp_and_fallback(self):
        """The exact aggregation scripts/run_eval.py's meta["mcp_modes"]
        performs (`sorted({c.get("mode","mcp") for c in mcp_call_log})`) --
        附录 CB.1.3's requirement that "skipped" not collapse into either
        neighbor."""
        from src.mcp_client.fallback import make_skipped_call_record
        log = [
            {"mode": "mcp"},
            {"mode": "fallback"},
            make_skipped_call_record("rag_server", "retrieve_with_metadata", {}, "x"),
        ]
        modes = sorted({c.get("mode", "mcp") for c in log})
        assert modes == ["fallback", "mcp", "skipped"]


# ─── profile_and_clean: force_on / force_off at the node level ────────────────

class TestProfileAndCleanForceOff:
    def test_force_off_never_calls_clean_dataset_even_when_critical(self, tmp_path, monkeypatch):
        """CC.3: "不发起调用" -- must be a call that never happens, not one
        that happens and gets discarded. has_critical_issues=True here on
        purpose: force_off must ignore it entirely, not merely "not be
        blocked by" a false verdict."""
        import src.graph.nodes as nodes
        from src.config.settings import settings as _settings
        from src.mcp_client.fallback import SubSystemWithFallback
        from src.mcp_client.models import DataQualityReport
        from src.state.graph_state import initial_state

        monkeypatch.setattr(_settings, "maeda_artifact_root", str(tmp_path))
        monkeypatch.setattr(_settings, "maeda_cleaner_mode", "force_off")

        csv_path = tmp_path / "sales.csv"
        csv_path.write_text("revenue,region\n,North\n,South\n1,East\n2,West\n3,North\n")

        state = initial_state("q", data_sources=[{"type": "csv", "path": str(csv_path)}])
        state["effective_dataset_path"] = str(csv_path)
        state["schema_columns"] = []

        critical_report = DataQualityReport(
            row_count=5, columns=[], quality_issues=[], has_critical_issues=True,
        )
        _log = lambda tool: {"system": "data_cleaner", "tool": tool,
                              "mode": "mcp", "args": {}, "duration_ms": 1.0}
        mock_mcp = MagicMock(spec=SubSystemWithFallback)
        mock_mcp.profile_dataset = AsyncMock(return_value=(critical_report, _log("profile_dataset")))

        old_mcp = nodes._subsystem_client
        nodes._subsystem_client = mock_mcp
        try:
            result = asyncio.run(nodes.profile_and_clean(state))
        finally:
            nodes._subsystem_client = old_mcp

        mock_mcp.clean_dataset.assert_not_called()
        assert result["cleaning_applied"] is False
        assert result["cleaning_stop_reason"] == "force_off"
        assert result["cleaning_applied_level"] == "none"

        modes = [c["mode"] for c in result["mcp_call_log"]]
        assert modes == ["mcp", "skipped"]  # real profile_dataset + skipped clean_dataset
        skipped = result["mcp_call_log"][-1]
        assert skipped["tool"] == "clean_dataset"
        assert skipped["system"] == "data_cleaner"

    def test_force_off_still_calls_profile_dataset(self, tmp_path, monkeypatch):
        """profile_dataset is diagnostic, not "cleaning" -- unconditional
        in every mode, same as before 定案 #15 existed (附录 CB.1)."""
        import src.graph.nodes as nodes
        from src.config.settings import settings as _settings
        from src.mcp_client.fallback import SubSystemWithFallback
        from src.mcp_client.models import DataQualityReport
        from src.state.graph_state import initial_state

        monkeypatch.setattr(_settings, "maeda_artifact_root", str(tmp_path))
        monkeypatch.setattr(_settings, "maeda_cleaner_mode", "force_off")

        csv_path = tmp_path / "sales.csv"
        csv_path.write_text("revenue,region\n1,North\n2,South\n")
        state = initial_state("q", data_sources=[{"type": "csv", "path": str(csv_path)}])
        state["effective_dataset_path"] = str(csv_path)
        state["schema_columns"] = []

        report = DataQualityReport(row_count=2, columns=[], quality_issues=[], has_critical_issues=False)
        _log = lambda tool: {"system": "data_cleaner", "tool": tool,
                              "mode": "mcp", "args": {}, "duration_ms": 1.0}
        mock_mcp = MagicMock(spec=SubSystemWithFallback)
        mock_mcp.profile_dataset = AsyncMock(return_value=(report, _log("profile_dataset")))

        old_mcp = nodes._subsystem_client
        nodes._subsystem_client = mock_mcp
        try:
            asyncio.run(nodes.profile_and_clean(state))
        finally:
            nodes._subsystem_client = old_mcp

        mock_mcp.profile_dataset.assert_awaited_once()


class TestProfileAndCleanForceOn:
    def test_force_on_calls_clean_dataset_even_when_report_says_not_critical(self, tmp_path, monkeypatch):
        """CC.2: the defining behavior -- force_on must trigger
        clean_dataset even though profile_dataset itself reported
        has_critical_issues=False. If this only removed a blocking gate
        (rather than bypassing the judgement), this test would fail
        because nothing would ever call clean_dataset here."""
        import src.graph.nodes as nodes
        from src.config.settings import settings as _settings
        from src.mcp_client.fallback import SubSystemWithFallback
        from src.mcp_client.models import CleaningResult, DataQualityReport
        from src.state.graph_state import initial_state

        monkeypatch.setattr(_settings, "maeda_artifact_root", str(tmp_path))
        monkeypatch.setattr(_settings, "maeda_cleaner_mode", "force_on")

        csv_path = tmp_path / "sales.csv"
        csv_path.write_text("revenue,region\n1,North\n2,South\n")
        state = initial_state("q", data_sources=[{"type": "csv", "path": str(csv_path)}])
        state["effective_dataset_path"] = str(csv_path)
        state["schema_columns"] = []

        not_critical_report = DataQualityReport(
            row_count=2, columns=[], quality_issues=[], has_critical_issues=False,
        )

        clean_dir = tmp_path / f"{state['run_id']}_clean1"
        clean_dir.mkdir(parents=True, exist_ok=True)
        cleaned_path = clean_dir / "sales.csv"
        # Byte-identical to input -> the round terminates on "no_diff",
        # which is fine here: this test only needs to prove clean_dataset
        # was actually INVOKED, not that a multi-round loop completed.
        cleaned_path.write_text(csv_path.read_text())

        clean_result = CleaningResult(cleaned_path=str(cleaned_path), changes_summary={}, rows_affected=0)

        _log = lambda tool: {"system": "data_cleaner", "tool": tool,
                              "mode": "mcp", "args": {}, "duration_ms": 1.0}
        mock_mcp = MagicMock(spec=SubSystemWithFallback)
        mock_mcp.profile_dataset = AsyncMock(return_value=(not_critical_report, _log("profile_dataset")))
        mock_mcp.clean_dataset = AsyncMock(return_value=(clean_result, _log("clean_dataset")))

        old_mcp = nodes._subsystem_client
        nodes._subsystem_client = mock_mcp
        try:
            result = asyncio.run(nodes.profile_and_clean(state))
        finally:
            nodes._subsystem_client = old_mcp

        mock_mcp.clean_dataset.assert_awaited_once()
        assert result["cleaning_applied"] is True
        assert result["cleaning_stop_reason"] == "no_diff"


# ─── retrieve_knowledge_node: RAG mode ─────────────────────────────────────────

class TestRetrieveKnowledgeNodeModes:
    @staticmethod
    def _wire(mock_mcp):
        import src.graph.nodes as nodes
        mock_insight = MagicMock()
        mock_insight.build_retrieval_query = MagicMock(return_value="revenue by region")
        old_mcp, old_insight = nodes._subsystem_client, nodes._insight_agent
        nodes._subsystem_client = mock_mcp
        nodes._insight_agent = mock_insight
        return nodes, old_mcp, old_insight

    def test_force_off_never_calls_retrieve_and_logs_skipped(self, monkeypatch):
        import src.graph.nodes as nodes
        from src.config.settings import settings as _settings
        from src.mcp_client.fallback import SubSystemWithFallback
        from src.state.graph_state import initial_state

        monkeypatch.setattr(_settings, "maeda_rag_mode", "force_off")
        mock_mcp = MagicMock(spec=SubSystemWithFallback)
        nodes, old_mcp, old_insight = self._wire(mock_mcp)
        state = initial_state("q")
        try:
            result = asyncio.run(nodes.retrieve_knowledge_node(state))
        finally:
            nodes._subsystem_client = old_mcp
            nodes._insight_agent = old_insight

        mock_mcp.retrieve_knowledge.assert_not_called()
        assert result["rag_context"] == []
        assert result["rag_sources"] == []
        modes = [c["mode"] for c in result["mcp_call_log"]]
        assert modes == ["skipped"]
        assert result["mcp_call_log"][0]["tool"] == "retrieve_with_metadata"
        assert result["mcp_call_log"][0]["system"] == "rag_server"

    @pytest.mark.parametrize("mode", ["auto", "force_on"])
    def test_auto_and_force_on_both_call_retrieve_unconditionally(self, mode, monkeypatch):
        """附录 CC.7 裁定 4: "auto" degenerates to unconditional retrieval.
        Since 轮次 3 the conditional edge itself DOES exist
        (route_after_viz, tests/unit/test_rag_tier_and_routing.py), but
        裁定 4 kept `auto`'s own skip judgement out of scope, so auto and
        force_on still both retrieve. This test locks that degeneration at
        the node level so it isn't silently "fixed" without also updating
        the settings docstring that documents it."""
        import src.graph.nodes as nodes
        from src.config.settings import settings as _settings
        from src.mcp_client.fallback import SubSystemWithFallback
        from src.mcp_client.models import RAGChunk
        from src.state.graph_state import initial_state

        monkeypatch.setattr(_settings, "maeda_rag_mode", mode)
        chunk = RAGChunk(content="c", source_file="f.pdf", page=1, chunk_id="1", score=0.9)
        _log = {"system": "rag_server", "tool": "retrieve_with_metadata",
                "mode": "mcp", "args": {}, "duration_ms": 1.0}
        mock_mcp = MagicMock(spec=SubSystemWithFallback)
        mock_mcp.retrieve_knowledge = AsyncMock(return_value=([chunk], _log))
        nodes, old_mcp, old_insight = self._wire(mock_mcp)
        state = initial_state("q")
        try:
            result = asyncio.run(nodes.retrieve_knowledge_node(state))
        finally:
            nodes._subsystem_client = old_mcp
            nodes._insight_agent = old_insight

        mock_mcp.retrieve_knowledge.assert_awaited_once()
        assert len(result["rag_context"]) == 1
        modes = [c["mode"] for c in result["mcp_call_log"]]
        assert modes == ["mcp"]


# ─── arm reporting: EvalResult + state snapshot (CB.3.4) ──────────────────────

class TestArmOnEvalResult:
    def test_defaults_to_none(self):
        from src.eval.metrics import MetricScore
        from src.eval.runner import EvalResult
        result = EvalResult("r1", "q", [MetricScore("error_rate", 1.0, "pass")], 0.9)
        assert result.cleaner_mode is None
        assert result.rag_mode is None
        d = result.to_dict()
        assert d["cleaner_mode"] is None
        assert d["rag_mode"] is None

    def test_to_dict_carries_arm_when_set(self):
        from src.eval.metrics import MetricScore
        from src.eval.runner import EvalResult
        result = EvalResult(
            run_id="r1", query="q", scores=[MetricScore("error_rate", 1.0, "pass")],
            aggregate_score=0.9, cleaner_mode="force_on", rag_mode="force_off",
        )
        d = result.to_dict()
        assert d["cleaner_mode"] == "force_on"
        assert d["rag_mode"] == "force_off"

    def test_not_reachable_by_aggregate_score(self):
        """结构性保证 (附录 CB.3.4, 沿用 intent_refined 先例):
        _aggregate_score only ever consumes `scores: list[MetricScore]` --
        cleaner_mode/rag_mode are separate dataclass fields it never
        touches, so an extreme value here cannot move the aggregate."""
        from src.eval.metrics import MetricScore
        from src.eval.runner import EvalResult, _aggregate_score
        scores = [MetricScore("error_rate", 1.0, "pass")]
        base = _aggregate_score(scores)
        result = EvalResult(
            run_id="r1", query="q", scores=scores, aggregate_score=base,
            cleaner_mode="force_on", rag_mode="force_off",
        )
        assert _aggregate_score(result.scores) == base


class TestArmOnState:
    def test_initial_state_snapshots_settings(self, monkeypatch):
        from src.config.settings import settings as _settings
        monkeypatch.setattr(_settings, "maeda_cleaner_mode", "force_on")
        monkeypatch.setattr(_settings, "maeda_rag_mode", "force_off")
        from src.state.graph_state import initial_state
        state = initial_state("q")
        assert state["cleaner_mode"] == "force_on"
        assert state["rag_mode"] == "force_off"

    def test_default_state_is_auto_auto(self, monkeypatch):
        from src.config.settings import settings as _settings
        monkeypatch.setattr(_settings, "maeda_cleaner_mode", "auto")
        monkeypatch.setattr(_settings, "maeda_rag_mode", "auto")
        from src.state.graph_state import initial_state
        state = initial_state("q")
        assert state["cleaner_mode"] == "auto"
        assert state["rag_mode"] == "auto"


def _routing_mock_llm(relevance_response, groundedness_response):
    """Local copy of tests/unit/test_phase9.py's helper of the same name
    (no cross-test-file import precedent in this repo) -- a single mock LLM
    that returns different canned responses depending on which of the two
    independent judge system prompts it was called with."""
    from src.config.agent_prompts import EVAL_GROUNDEDNESS_SYSTEM, EVAL_RELEVANCE_SYSTEM

    async def _ainvoke(messages):
        system_content = messages[0].content
        if system_content == EVAL_RELEVANCE_SYSTEM:
            return relevance_response
        if system_content == EVAL_GROUNDEDNESS_SYSTEM:
            return groundedness_response
        raise AssertionError(f"unexpected system prompt: {system_content!r}")

    mock = MagicMock()
    mock.ainvoke = AsyncMock(side_effect=_ainvoke)
    return mock


def _mock_relevance_response(score, reasoning="r"):
    resp = MagicMock()
    resp.content = json.dumps({"answer_relevance": score, "reasoning": reasoning})
    resp.usage_metadata = {"input_tokens": 10, "output_tokens": 10}
    return resp


def _mock_groundedness_response(claims, reasoning="r"):
    resp = MagicMock()
    resp.content = json.dumps({"claims": claims, "reasoning": reasoning})
    resp.usage_metadata = {"input_tokens": 10, "output_tokens": 10}
    return resp


def test_eval_runner_score_reads_arm_from_state():
    """The last link in the chain: EvalRunner.score() must actually read
    state["cleaner_mode"]/state["rag_mode"] (not just have the dataclass
    fields exist unused) -- same pattern as the existing
    test_eval_runner_carries_cleaning_fields_from_state in test_phase9.py."""
    from src.eval.runner import EvalRunner
    from src.state.graph_state import initial_state

    mock_llm = _routing_mock_llm(
        _mock_relevance_response(0.9, "ok"),
        _mock_groundedness_response([{"claim": "x", "supported": True, "evidence": "e"}]),
    )
    runner = EvalRunner(llm=mock_llm)
    state = initial_state("Show revenue by region")
    state["report"] = "# Report\n\n## Findings\nx.\n"
    state["analysis_results"] = [{"method": "groupby", "result_summary": "x", "failed": False}]
    state["parsed_intent"] = {"query_type": "descriptive", "confidence": 0.9}
    state["rag_context"] = []
    state["charts"] = []
    state["cleaner_mode"] = "force_on"
    state["rag_mode"] = "force_off"

    result = asyncio.run(runner.score(state))
    assert result.cleaner_mode == "force_on"
    assert result.rag_mode == "force_off"


# ─── mcp_strict_mode x three-state interaction (附录 CB.3.3) ──────────────────

class TestMcpStrictModeInteraction:
    """附录 CB.3.3 found all 6 (three-state x strict/degraded) combinations
    for each subsystem well-defined with no contradiction: the two
    dimensions answer independent questions ("whether to call" vs "what to
    do if the call fails/errors"). These tests exercise the ones with real
    code paths -- force_off is orthogonal to mcp_strict_mode by
    construction (the call, and therefore the strict/degraded branch in
    src/mcp_client/fallback.py::_call_with_matrix, never happens), so
    there is nothing further to assert for force_off beyond
    TestProfileAndCleanForceOff/TestRetrieveKnowledgeNodeModes above."""

    @pytest.mark.parametrize("strict_mode", ["strict", "degraded"])
    def test_force_on_plus_either_strict_mode_still_calls_clean_dataset(
        self, strict_mode, tmp_path, monkeypatch,
    ):
        import src.graph.nodes as nodes
        from src.config.settings import settings as _settings
        from src.mcp_client.fallback import SubSystemWithFallback
        from src.mcp_client.models import CleaningResult, DataQualityReport
        from src.state.graph_state import initial_state

        monkeypatch.setattr(_settings, "maeda_artifact_root", str(tmp_path))
        monkeypatch.setattr(_settings, "maeda_cleaner_mode", "force_on")
        monkeypatch.setattr(_settings, "mcp_strict_mode", strict_mode)

        csv_path = tmp_path / "sales.csv"
        csv_path.write_text("revenue,region\n1,North\n2,South\n")
        state = initial_state("q", data_sources=[{"type": "csv", "path": str(csv_path)}])
        state["effective_dataset_path"] = str(csv_path)
        state["schema_columns"] = []

        not_critical = DataQualityReport(row_count=2, columns=[], quality_issues=[], has_critical_issues=False)
        clean_dir = tmp_path / f"{state['run_id']}_clean1"
        clean_dir.mkdir(parents=True, exist_ok=True)
        cleaned_path = clean_dir / "sales.csv"
        cleaned_path.write_text(csv_path.read_text())
        clean_result = CleaningResult(cleaned_path=str(cleaned_path), changes_summary={}, rows_affected=0)

        _log = lambda tool: {"system": "data_cleaner", "tool": tool,
                              "mode": "mcp", "args": {}, "duration_ms": 1.0}
        mock_mcp = MagicMock(spec=SubSystemWithFallback)
        mock_mcp.profile_dataset = AsyncMock(return_value=(not_critical, _log("profile_dataset")))
        mock_mcp.clean_dataset = AsyncMock(return_value=(clean_result, _log("clean_dataset")))

        old_mcp = nodes._subsystem_client
        nodes._subsystem_client = mock_mcp
        try:
            result = asyncio.run(nodes.profile_and_clean(state))
        finally:
            nodes._subsystem_client = old_mcp

        # mcp_strict_mode governs SubSystemWithFallback's own internal
        # failure handling (src/mcp_client/fallback.py::_call_with_matrix),
        # which this mock bypasses entirely -- the point of this test is
        # that force_on's "attempt the call" decision doesn't itself branch
        # on strict_mode, i.e. the same call happens either way.
        mock_mcp.clean_dataset.assert_awaited_once()
        assert result["cleaning_applied"] is True
