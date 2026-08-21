"""
RAG retrieval-tier validation + conditional routing —
ECOSYSTEM_INTEGRATION_PLAN.md 阶段 3 收尾执行计划轮次 3, 附录 CI.3 (MAEDA
half), 附录 CK.3 (the three tightened conditions), 附录 CC.7 裁定 4.

Three things are covered here, in the order they happen at runtime:

  1. the echo fields rag-framework has always returned on retrieve /
     retrieve_with_metadata (`retrieval_mode`, `embedding_provider`,
     `reranker_provider`, `degraded_reason`) are now parsed, carried into
     state["mcp_call_log"], and VALIDATED — 附录 U.3's rule that an
     un-validated echoed field in a cross-process contract is worse than
     no field at all, since it makes people believe a guarantee exists;
  2. under MAEDA_RAG_MODE=force_on, a retrieval that did not actually run
     at the asserted tier — wrong tier, degraded, fallback, or hard
     failure — invalidates the trial so it cannot be averaged into the
     on-arm mean (附录 CK.3 conditions 2 and 3). The test that matters
     most in this file is
     test_force_on_rag_failure_removes_the_trial_from_pass_at_k_and_variance:
     it proves the trial leaves the AGGREGATION, not merely that a line
     was written to a log;
  3. `generate_viz` → `retrieve_domain_knowledge` is a real conditional
     route with two reachable destinations, and the routing decision
     itself lands in the decision trace (轮次 3). `auto` still always
     retrieves — 裁定 4's adjudicated degeneration — and that is locked
     by a test here so it can neither be "fixed" silently nor forgotten.

One finding recorded as an executable test rather than prose:
test_embedding_provider_unavailable_is_a_FALLBACK_in_degraded_mode.
附录 CK.3 located the danger in nodes.py's `except SubSystemHardFailure`
branch; under the DEFAULT mcp_strict_mode="degraded" that branch is not
the path a missing OPENAI_API_KEY actually takes.

No real LLM/MCP calls anywhere in this file — same mocked-transport /
mocked-SubSystemWithFallback harness as tests/unit/test_phase3.py and
tests/unit/test_tristate_switch.py.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.mcp_client.client import MCPClient, MCPToolError
from src.mcp_client.models import RetrievalTier

# A response body shaped like rag-framework's RetrieveOutput
# (rag/app/mcp_server/server.py:432-436 / :595-602).
_HYBRID = {
    "chunks": [{"content": "Domain context.", "score": 0.88, "source_file": "policy.pdf",
                "page": 2, "chunk_id": "c_7"}],
    "retrieval_mode": "hybrid",
    "embedding_provider": "openai",
    "reranker_provider": "voyage",
    "degraded_reason": None,
}
_BM25_ONLY = {
    "chunks": [{"content": "Something plausible.", "score": 0.015, "source_file": "memo.md",
                "page": 1, "chunk_id": "c_1"}],
    "retrieval_mode": "bm25_only",
    "embedding_provider": None,
    "reranker_provider": None,
    "degraded_reason": None,
}


def _mock_rag_transport(tool_responses: dict) -> MCPClient:
    transport = MagicMock(spec=MCPClient)

    async def call_tool(name, args, **kwargs):
        resp = tool_responses[name]
        if isinstance(resp, Exception):
            raise resp
        return resp

    transport.call_tool = AsyncMock(side_effect=call_tool)
    return transport


# ─── 1. RetrievalTier parsing + degradation reporting ────────────────────────

class TestRetrievalTier:
    def test_parses_all_four_echo_fields(self):
        tier = RetrievalTier.from_mcp_response(_HYBRID)
        assert tier.retrieval_mode == "hybrid"
        assert tier.embedding_provider == "openai"
        assert tier.reranker_provider == "voyage"
        assert tier.degraded_reason is None
        assert tier.echoed is True

    def test_response_without_echo_fields_is_marked_not_echoed(self):
        """A server that says nothing about its tier must not read as
        healthy — that is precisely the pre-CI.3 state of the world."""
        tier = RetrievalTier.from_mcp_response({"chunks": []})
        assert tier.echoed is False
        assert tier.degradations("hybrid") == [
            "response carried no retrieval_mode/embedding_provider echo fields"
        ]

    @pytest.mark.parametrize("mode", ["hybrid", "hybrid_rerank"])
    def test_vector_modes_satisfy_hybrid_expectation(self, mode):
        """A reranker on top is a strictly stronger tier, not a violation."""
        tier = RetrievalTier(retrieval_mode=mode, embedding_provider="openai", echoed=True)
        assert tier.degradations("hybrid") == []

    def test_bm25_only_violates_hybrid_expectation(self):
        tier = RetrievalTier.from_mcp_response(_BM25_ONLY)
        reasons = tier.degradations("hybrid")
        assert len(reasons) == 2
        assert any("retrieval_mode='bm25_only'" in r for r in reasons)
        assert any("embedding_provider is null" in r for r in reasons)

    def test_degraded_reason_violates_even_the_bm25_only_expectation(self):
        """Declaring a BM25-only deployment waives the tier check, not
        rag's own "this call was degraded" signal."""
        tier = RetrievalTier(retrieval_mode="bm25_only", degraded_reason="index dim mismatch",
                             echoed=True)
        assert tier.degradations("bm25_only") == [
            "rag reported degraded_reason='index dim mismatch'"
        ]

    def test_declared_bm25_only_deployment_is_not_a_degradation(self):
        tier = RetrievalTier.from_mcp_response(_BM25_ONLY)
        assert tier.degradations("bm25_only") == []

    def test_unavailable_short_circuits(self):
        tier = RetrievalTier.unavailable("rag_server unreachable")
        assert tier.degradations("hybrid") == ["no tier reported: rag_server unreachable"]
        assert tier.degradations("bm25_only") == ["no tier reported: rag_server unreachable"]

    def test_round_trips_through_the_call_log_dict(self):
        """The tier is validated in nodes.py by reading it back off the
        persisted log entry, so to_dict/from_dict must be lossless."""
        tier = RetrievalTier.from_mcp_response(_HYBRID)
        assert RetrievalTier.from_dict(tier.to_dict()) == tier


# ─── 2. Client layer: the echo fields survive the trip ───────────────────────

class TestClientCarriesTier:
    def test_retrieve_with_metadata_tiered_returns_the_tier(self):
        from src.mcp_client.rag_server import RAGServerClient
        client = RAGServerClient(_mock_rag_transport({"retrieve_with_metadata": _HYBRID}))
        chunks, tier = asyncio.run(client.retrieve_with_metadata_tiered("q"))
        assert len(chunks) == 1
        assert tier.retrieval_mode == "hybrid"

    def test_retrieve_tiered_returns_the_tier(self):
        """CK.3 condition 1 names `retrieve` as well as
        `retrieve_with_metadata`; both echo the same fields."""
        from src.mcp_client.rag_server import RAGServerClient
        client = RAGServerClient(_mock_rag_transport({"retrieve": _BM25_ONLY}))
        chunks, tier = asyncio.run(client.retrieve_tiered("q"))
        assert len(chunks) == 1
        assert tier.retrieval_mode == "bm25_only"

    def test_plain_methods_keep_returning_just_chunks(self):
        """Back-compat: the non-tiered signatures are unchanged, so no
        existing caller has to know this feature exists."""
        from src.mcp_client.rag_server import RAGServerClient
        client = RAGServerClient(_mock_rag_transport(
            {"retrieve": _HYBRID, "retrieve_with_metadata": _HYBRID}
        ))
        assert len(asyncio.run(client.retrieve("q"))) == 1
        assert len(asyncio.run(client.retrieve_with_metadata("q"))) == 1

    def test_retrieve_knowledge_puts_the_tier_on_the_call_log(self):
        from src.mcp_client.data_cleaner import DataCleanerClient
        from src.mcp_client.fallback import SubSystemWithFallback
        from src.mcp_client.rag_server import RAGServerClient

        rag = RAGServerClient(_mock_rag_transport({"retrieve_with_metadata": _BM25_ONLY}))
        fb = SubSystemWithFallback(MagicMock(spec=DataCleanerClient), rag)
        _, log = asyncio.run(fb.retrieve_knowledge("q"))
        assert log["mode"] == "mcp"
        assert log["retrieval_tier"]["retrieval_mode"] == "bm25_only"
        assert log["retrieval_tier"]["echoed"] is True

    def test_fallback_path_reports_an_explicitly_unavailable_tier(self):
        """An empty context from a degraded call must stay
        distinguishable from an empty context from a healthy server that
        legitimately matched nothing (附录 CH.2, one layer down)."""
        from src.mcp_client.client import MCPConnectionError
        from src.mcp_client.data_cleaner import DataCleanerClient
        from src.mcp_client.fallback import SubSystemWithFallback
        from src.mcp_client.rag_server import RAGServerClient

        rag = RAGServerClient(_mock_rag_transport(
            {"retrieve_with_metadata": MCPConnectionError("rag down")}
        ))
        fb = SubSystemWithFallback(MagicMock(spec=DataCleanerClient), rag)
        chunks, log = asyncio.run(fb.retrieve_knowledge("q"))
        assert chunks == []
        assert log["mode"] == "fallback"
        assert log["retrieval_tier"]["unavailable_reason"]

    def test_embedding_provider_unavailable_is_a_FALLBACK_in_degraded_mode(self, monkeypatch):
        """**The channel 附录 CK.3 under-counted.**

        CK.3 places the danger in nodes.py's `except SubSystemHardFailure`
        branch. But rag's EMBEDDING_PROVIDER_UNAVAILABLE is classified
        `internal_unknown` (fallback.py::_RAG_ERROR_CODE_CLASS), and
        internal_unknown is fallback-eligible in degraded mode — which is
        the DEFAULT (settings.mcp_strict_mode). So with stock
        configuration, rag's loud failure over a missing OPENAI_API_KEY
        arrives at the node as a perfectly successful call returning zero
        chunks, and the hard-failure branch is never entered.

        Same corrupted on-arm mean, quieter path — which is why the
        node's force_on guard keys on `mode == "fallback"` too, not only
        on the exception."""
        from src.config.settings import settings as _settings
        from src.mcp_client.data_cleaner import DataCleanerClient
        from src.mcp_client.fallback import SubSystemHardFailure, SubSystemWithFallback
        from src.mcp_client.rag_server import RAGServerClient

        err = MCPToolError(
            "MCP tool error: Error executing tool retrieve_with_metadata: "
            "EMBEDDING_PROVIDER_UNAVAILABLE: OPENAI_API_KEY is not set"
        )
        rag = RAGServerClient(_mock_rag_transport({"retrieve_with_metadata": err}))
        fb = SubSystemWithFallback(MagicMock(spec=DataCleanerClient), rag)

        monkeypatch.setattr(_settings, "mcp_strict_mode", "degraded")
        chunks, log = asyncio.run(fb.retrieve_knowledge("q"))
        assert chunks == []
        assert log["mode"] == "fallback"          # NOT an exception
        assert log["error_class"] == "internal_unknown"

        monkeypatch.setattr(_settings, "mcp_strict_mode", "strict")
        with pytest.raises(SubSystemHardFailure):
            asyncio.run(fb.retrieve_knowledge("q"))


# ─── 3. The node: force_on arm integrity (CK.3 conditions 2 and 3) ───────────

def _wire_node(mock_mcp):
    """Same wiring helper shape as test_tristate_switch.py's."""
    import src.graph.nodes as nodes
    mock_insight = MagicMock()
    mock_insight.build_retrieval_query = MagicMock(return_value="revenue by region")
    old_mcp, old_insight = nodes._subsystem_client, nodes._insight_agent
    nodes._subsystem_client = mock_mcp
    nodes._insight_agent = mock_insight
    return nodes, old_mcp, old_insight


def _run_node(mock_mcp, state):
    nodes, old_mcp, old_insight = _wire_node(mock_mcp)
    try:
        return asyncio.run(nodes.retrieve_knowledge_node(state))
    finally:
        nodes._subsystem_client = old_mcp
        nodes._insight_agent = old_insight


def _mcp_client_returning(chunks, log):
    from src.mcp_client.fallback import SubSystemWithFallback
    mock = MagicMock(spec=SubSystemWithFallback)
    mock.retrieve_knowledge = AsyncMock(return_value=(chunks, log))
    return mock


def _log(mode="mcp", tier=None, **extra):
    d = {"system": "rag_server", "tool": "retrieve_with_metadata", "mode": mode,
         "args": {}, "duration_ms": 1.0, "error": None, "error_class": None}
    d["retrieval_tier"] = (tier if tier is not None
                           else RetrievalTier.from_mcp_response(_HYBRID).to_dict())
    d.update(extra)
    return d


class TestForceOnArmIntegrity:
    def test_hybrid_tier_under_force_on_is_a_valid_arm(self, monkeypatch):
        from src.config.settings import settings as _settings
        from src.mcp_client.models import RAGChunk
        from src.state.graph_state import initial_state

        monkeypatch.setattr(_settings, "maeda_rag_mode", "force_on")
        chunk = RAGChunk(content="c", source_file="f.pdf", page=1, chunk_id="1", score=0.9)
        result = _run_node(_mcp_client_returning([chunk], _log()), initial_state("q"))

        assert result["rag_arm_invalid_reason"] is None
        assert len(result["rag_context"]) == 1

    def test_bm25_only_tier_under_force_on_invalidates_the_trial(self, monkeypatch):
        """附录 CK.3 condition 2. Note the retrieval SUCCEEDED and returned
        a chunk — the dangerous case is the plausible one, not the empty
        one (附录 CI.2)."""
        from src.config.settings import settings as _settings
        from src.mcp_client.models import RAGChunk
        from src.state.graph_state import initial_state

        monkeypatch.setattr(_settings, "maeda_rag_mode", "force_on")
        monkeypatch.setattr(_settings, "maeda_rag_expected_retrieval_mode", "hybrid")
        chunk = RAGChunk(content="plausible", source_file="memo.md", page=1,
                         chunk_id="1", score=0.015)
        tier = RetrievalTier.from_mcp_response(_BM25_ONLY).to_dict()
        result = _run_node(_mcp_client_returning([chunk], _log(tier=tier)), initial_state("q"))

        assert result["rag_arm_invalid_reason"]
        assert "bm25_only" in result["rag_arm_invalid_reason"]
        # The run still completed normally — invalidation is a bookkeeping
        # verdict, not a pipeline abort.
        assert len(result["rag_context"]) == 1
        assert any(t["action"] == "rag_arm_invalidated" for t in result["decision_trace"])

    def test_declared_bm25_only_expectation_keeps_the_trial_valid(self, monkeypatch):
        """settings.maeda_rag_expected_retrieval_mode="bm25_only" is a
        pre-registration statement: the tier was declared up front, not
        discovered afterwards."""
        from src.config.settings import settings as _settings
        from src.state.graph_state import initial_state

        monkeypatch.setattr(_settings, "maeda_rag_mode", "force_on")
        monkeypatch.setattr(_settings, "maeda_rag_expected_retrieval_mode", "bm25_only")
        tier = RetrievalTier.from_mcp_response(_BM25_ONLY).to_dict()
        result = _run_node(_mcp_client_returning([], _log(tier=tier)), initial_state("q"))
        assert result["rag_arm_invalid_reason"] is None

    def test_fallback_mode_under_force_on_invalidates_the_trial(self, monkeypatch):
        """The degraded-mode channel from
        test_embedding_provider_unavailable_is_a_FALLBACK_in_degraded_mode,
        now seen from the node's side."""
        from src.config.settings import settings as _settings
        from src.state.graph_state import initial_state

        monkeypatch.setattr(_settings, "maeda_rag_mode", "force_on")
        log = _log(mode="fallback", tier=RetrievalTier.unavailable("rag unreachable").to_dict(),
                   error_class="internal_unknown", error="EMBEDDING_PROVIDER_UNAVAILABLE: no key")
        result = _run_node(_mcp_client_returning([], log), initial_state("q"))
        assert result["rag_arm_invalid_reason"]
        assert "fallback" in result["rag_arm_invalid_reason"]

    def test_missing_echo_fields_under_force_on_invalidates_the_trial(self, monkeypatch):
        """An older rag build, or any other RAG-MCP-Server, that reports
        no tier cannot be scored as a confirmed on-arm run — 附录 U.3's
        point is that an unvalidatable field guarantees nothing."""
        from src.config.settings import settings as _settings
        from src.state.graph_state import initial_state

        monkeypatch.setattr(_settings, "maeda_rag_mode", "force_on")
        tier = RetrievalTier.from_mcp_response({"chunks": []}).to_dict()
        result = _run_node(_mcp_client_returning([], _log(tier=tier)), initial_state("q"))
        assert result["rag_arm_invalid_reason"]
        assert "echo fields" in result["rag_arm_invalid_reason"]

    def test_hard_failure_under_force_on_invalidates_but_still_completes(self, monkeypatch):
        """附录 CK.3 condition 3. Both halves matter: the enrichment-only
        degradation stays (CLAUDE.md — MAEDA must run standalone), AND the
        trial stops being aggregatable."""
        from src.config.settings import settings as _settings
        from src.mcp_client.fallback import SubSystemHardFailure, SubSystemWithFallback
        from src.state.graph_state import initial_state

        monkeypatch.setattr(_settings, "maeda_rag_mode", "force_on")
        exc = SubSystemHardFailure("contract mismatch", "contract",
                                   _log(mode="mcp", error="contract mismatch"))
        mock = MagicMock(spec=SubSystemWithFallback)
        mock.retrieve_knowledge = AsyncMock(side_effect=exc)
        result = _run_node(mock, initial_state("q"))

        assert result["rag_context"] == []          # degraded, as before
        assert result["current_phase"] != "error"   # NOT aborted
        assert result["rag_arm_invalid_reason"]
        assert "hard failure" in result["rag_arm_invalid_reason"]

    @pytest.mark.parametrize("mode", ["auto", "force_off"])
    def test_non_force_on_modes_never_invalidate(self, mode, monkeypatch):
        """auto/force_off make no claim about RAG's contribution, so a
        degraded tier is logged but must not shrink anyone's denominator.
        Guards against the invalidation quietly becoming a general trial
        filter — that would be a selection effect of its own."""
        from src.config.settings import settings as _settings
        from src.state.graph_state import initial_state

        monkeypatch.setattr(_settings, "maeda_rag_mode", mode)
        tier = RetrievalTier.from_mcp_response(_BM25_ONLY).to_dict()
        result = _run_node(_mcp_client_returning([], _log(mode="fallback", tier=tier)),
                           initial_state("q"))
        assert result["rag_arm_invalid_reason"] is None

    def test_tier_is_recorded_in_the_decision_trace_even_when_valid(self, monkeypatch):
        from src.config.settings import settings as _settings
        from src.state.graph_state import initial_state

        monkeypatch.setattr(_settings, "maeda_rag_mode", "force_on")
        result = _run_node(_mcp_client_returning([], _log()), initial_state("q"))
        trace = [t for t in result["decision_trace"] if t["action"] == "retrieve_knowledge"][-1]
        assert "retrieval_mode='hybrid'" in trace["reasoning"]
        assert "MAEDA_RAG_MODE=force_on" in trace["reasoning"]


# ─── 4. The exclusion actually reaches the aggregation ───────────────────────

def _eval_result_row(case_id: str, label: str, score: float, **extra) -> dict:
    """One `per_trial[trial][case]` row in run_eval.py's report shape."""
    result = {
        "test_case_id": case_id,
        "cleaning_applied_level": "none",
        "rag_arm_invalid_reason": None,
        "scores": [{"metric": "answer_relevance", "score": score, "label": label,
                    "reasoning": "", "valid": True}],
    }
    result.update(extra)
    return {"test_case_id": case_id, "eval_result": result, "meta": {}}


class TestInvalidTrialLeavesTheAggregation:
    def test_is_applicable_excludes_a_rag_invalid_row(self):
        from src.eval.trials import is_applicable, not_applicable_reason

        assert is_applicable({"rag_arm_invalid_reason": None}) is True
        assert is_applicable({}) is True
        assert is_applicable({"rag_arm_invalid_reason": "retrieval_mode='bm25_only'"}) is False
        assert not_applicable_reason(
            {"rag_arm_invalid_reason": "retrieval_mode='bm25_only'"}
        ).startswith("rag_arm_invalid:")

    def test_blocked_needs_review_still_excludes(self):
        """附录 AX.1's condition is untouched by CK.3's addition."""
        from src.eval.trials import is_applicable
        assert is_applicable({"cleaning_applied_level": "blocked_needs_review"}) is False
        assert is_applicable({"cleaning_applied_level": "none"}) is True

    def test_force_on_rag_failure_removes_the_trial_from_pass_at_k_and_variance(self):
        """**The test 附录 CK.3 asks for by name.**

        Three trials of one case: two clean passes and one whose RAG
        degraded to BM25-only under force_on and, unsurprisingly, scored
        badly. If the bad trial were merely logged and scored as usual,
        it would drag both pass@1 and the continuous mean down — an
        off-arm run averaged into the on-arm number, exactly 附录 CI.2's
        "credible wrong answer".

        The assertion is that it leaves the DENOMINATOR: n_scored == 2,
        pass@1 == 1.0, and the continuous mean == 0.9 rather than 0.63.
        """
        from src.eval.trials import summarize_report

        good_a = _eval_result_row("C01", "pass", 0.9)
        good_b = _eval_result_row("C01", "pass", 0.9)
        degraded = _eval_result_row(
            "C01", "fail", 0.1,
            rag_arm_invalid_reason="retrieval_mode='bm25_only', expected vector retrieval",
        )
        summary = summarize_report([[good_a], [good_b], [degraded]], k_values=[1])
        case = summary["per_case"]["C01"]

        assert case["n_trials"] == 3
        assert case["n_applicable"] == 2
        assert case["n_not_applicable"] == 1
        binary = case["binary"]["answer_relevance"]
        assert binary["n_scored"] == 2 and binary["c"] == 2
        assert binary["pass_at_k"][1] == 1.0
        assert case["continuous"]["answer_relevance"]["summary"]["mean"] == pytest.approx(0.9)

    def test_the_exclusion_is_reported_with_its_reason_not_just_counted(self):
        """An exclusion visible only as a shrinking denominator is the
        same class of problem as the silent degradation it exists to
        catch — so summarize_case names the reasons."""
        from src.eval.trials import summarize_report

        rows = [
            [_eval_result_row("C01", "pass", 0.9)],
            [_eval_result_row("C01", "fail", 0.1, rag_arm_invalid_reason="bm25_only")],
            [_eval_result_row("C01", "fail", 0.2, cleaning_applied_level="blocked_needs_review")],
        ]
        case = summarize_report(rows, k_values=[1])["per_case"]["C01"]
        assert case["not_applicable_reasons"] == {
            "rag_arm_invalid: bm25_only": 1,
            "blocked_needs_review": 1,
        }

    def test_eval_runner_copies_the_reason_off_state(self, monkeypatch):
        """The link between the node and the aggregation: without this,
        the node's verdict would never reach trials.py. Same pattern as
        test_tristate_switch.py::test_eval_runner_score_reads_arm_from_state.
        """
        import json

        from src.eval.runner import EvalRunner
        from src.state.graph_state import initial_state

        def _resp(payload):
            r = MagicMock()
            r.content = json.dumps(payload)
            r.usage_metadata = {"input_tokens": 10, "output_tokens": 10}
            return r

        responses = [
            _resp({"score": 0.9, "reasoning": "ok"}),
            _resp({"claims": [{"claim": "x", "supported": True, "evidence": "e"}],
                   "reasoning": "ok"}),
        ]
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=responses * 4)

        state = initial_state("Show revenue by region")
        state["report"] = "# Report\n\n## Findings\nx.\n"
        state["analysis_results"] = [{"method": "groupby", "result_summary": "x", "failed": False}]
        state["parsed_intent"] = {"query_type": "descriptive", "confidence": 0.9}
        state["rag_arm_invalid_reason"] = "rag hard failure (contract): boom"

        result = asyncio.run(EvalRunner(llm=mock_llm).score(state))
        assert result.rag_arm_invalid_reason == "rag hard failure (contract): boom"
        assert result.to_dict()["rag_arm_invalid_reason"] == "rag hard failure (contract): boom"


# ─── 5. 轮次 3: the conditional route out of generate_viz ────────────────────

class TestConditionalRouting:
    def test_force_off_routes_to_skip(self, monkeypatch):
        from src.config.settings import settings as _settings
        from src.graph.router import route_after_viz
        from src.state.graph_state import initial_state

        monkeypatch.setattr(_settings, "maeda_rag_mode", "force_off")
        assert route_after_viz(initial_state("q")) == "skip"

    @pytest.mark.parametrize("mode", ["auto", "force_on"])
    def test_auto_and_force_on_route_to_retrieve(self, mode, monkeypatch):
        """裁定 4: the conditional edge now exists, but `auto` still
        always retrieves — its query-type judgement is deliberately NOT
        implemented. This locks the documented degeneration so it can't
        be silently "fixed" into a skip without revisiting 裁定 4."""
        from src.config.settings import settings as _settings
        from src.graph.router import RAG_RETRIEVE_AUTO_DEGENERATE, rag_retrieval_decision, route_after_viz
        from src.state.graph_state import initial_state

        monkeypatch.setattr(_settings, "maeda_rag_mode", mode)
        state = initial_state("what is 2+2 on this column")  # purely computational
        assert route_after_viz(state) == "retrieve"
        should, reason = rag_retrieval_decision(state)
        assert should is True
        if mode == "auto":
            assert reason == RAG_RETRIEVE_AUTO_DEGENERATE
            assert "NOT implemented" in reason

    def test_graph_has_both_destinations_wired(self):
        """Asserted against the COMPILED graph, not the builder source:
        the claim 轮次 3 makes is that a route which can decline to
        retrieve exists in the executable graph. `conditional=True` on
        both branches is what distinguishes this from the plain
        `add_edge` that was here before."""
        from src.graph.builder import build_graph

        g = build_graph()
        assert "skip_retrieval" in g.nodes
        out = {e.data: (e.target, e.conditional)
               for e in g.get_graph().edges if e.source == "generate_viz"}
        assert out == {
            "retrieve": ("retrieve_domain_knowledge", True),
            "skip": ("skip_retrieval", True),
        }
        assert any(e.source == "skip_retrieval" and e.target == "generate_insights"
                   for e in g.get_graph().edges)

    def test_skip_node_records_a_skipped_call_and_the_routing_decision(self, monkeypatch):
        from src.config.settings import settings as _settings
        from src.graph.nodes import skip_retrieval_node
        from src.graph.router import RAG_SKIP_FORCE_OFF
        from src.state.graph_state import initial_state

        monkeypatch.setattr(_settings, "maeda_rag_mode", "force_off")
        result = asyncio.run(skip_retrieval_node(initial_state("q")))

        assert [c["mode"] for c in result["mcp_call_log"]] == ["skipped"]
        assert result["mcp_call_log"][0]["result_summary"] == RAG_SKIP_FORCE_OFF
        assert result["rag_context"] == [] and result["rag_sources"] == []
        routing = [t for t in result["decision_trace"] if t["action"] == "route_after_viz"]
        assert len(routing) == 1
        assert RAG_SKIP_FORCE_OFF in routing[0]["reasoning"]

    def test_node_backstop_skips_even_when_invoked_directly(self, monkeypatch):
        """The conditional edge should already have diverted force_off, but
        the node must not depend on the topology being right — same shared
        gate, so the two can't drift (附录 CC.2)."""
        from src.config.settings import settings as _settings
        from src.mcp_client.fallback import SubSystemWithFallback
        from src.state.graph_state import initial_state

        monkeypatch.setattr(_settings, "maeda_rag_mode", "force_off")
        mock = MagicMock(spec=SubSystemWithFallback)
        result = _run_node(mock, initial_state("q"))

        mock.retrieve_knowledge.assert_not_called()
        assert [c["mode"] for c in result["mcp_call_log"]] == ["skipped"]

    def test_streaming_label_exists_for_the_skip_node(self):
        from src.graph.streaming import NODE_LABELS
        assert "skip_retrieval" in NODE_LABELS
