# MAEDA Architecture & Design Decisions

## Why LangGraph?

Most agent frameworks (AutoGen, CrewAI) use implicit loops or sequential chains. LangGraph gives us:
- **Explicit state transitions** — every edge condition is readable code
- **Typed state** — `MAEDAState` TypedDict prevents schema drift across 8 agents
- **Cycles with guardrails** — the guardrail→retry loop is a first-class graph construct, not a hack
- **Checkpointing** — full state replay at any node for debugging (`.stream()` mode)

The tradeoff: LangGraph's abstraction adds indirection. We mitigate this with `decision_trace` (every node logs entry/exit) and `.stream()` during development.

## Why MCP for sub-system communication?

We could have imported the Data Cleaner and RAG Server directly. We didn't, because:
1. **Deployment independence** — each project deploys separately; MAEDA just needs HTTP
2. **Language agnostic** — future sub-systems can be in Go, Rust, etc.
3. **Claude Desktop compatible** — MAEDA itself can be registered as an MCP server
4. **Graceful degradation** — `SubSystemWithFallback` catches `MCPConnectionError` and returns pandas fallbacks; MAEDA runs standalone out of the box

## State design

`MAEDAState` is a single TypedDict that all agents read from and write to. Key design choices:
- **Append-only lists** (`decision_trace`, `mcp_call_log`, `analysis_results`, `charts`) — no agent overwrites another's work
- **`current_phase`** — drives routing; agents set it before returning
- **`iteration_count`** — guardrail retry counter; prevents infinite loops
- **`token_usage`** — each agent updates this dict; the eval module reads it for cost scoring

## RAG routing and the force_on arm

Retrieval used to be an unconditional edge: `generate_viz` always fell
into `retrieve_domain_knowledge`. Since 阶段 3 收尾执行计划轮次 3 it is a
conditional route (`route_after_viz` → `rag_retrieval_decision`, both in
`src/graph/router.py`) with two reachable destinations — the real MCP
call, or `skip_retrieval`, which records a `mode="skipped"` call-log
entry plus the routing decision in the decision trace and falls through
to insight generation. The skip is a node rather than a bare edge on
purpose: a skip that left no record would be indistinguishable from a run
where retrieval simply wasn't triggered.

**`auto` still always retrieves.** This is a known, adjudicated,
temporary degeneration (ECOSYSTEM_INTEGRATION_PLAN.md 附录 CC.7 裁定 4),
not an oversight and not a missing implementation nobody noticed. What
轮次 3 owed was the testable route; the judgement that would let `auto`
skip retrieval for purely-computational queries is separate work that has
not been done. Today only `MAEDA_RAG_MODE=force_off` takes the skip
branch. When that judgement lands, `rag_retrieval_decision`'s `auto`
branch is the only place that changes — `force_on`/`force_off` keep their
meaning.

**Retrieval tier validation (force_on only).** rag-framework picks its
embedding/reranker providers from its own environment; nothing MAEDA
sends selects a tier. It echoes what it actually used
(`retrieval_mode`/`embedding_provider`/`reranker_provider`/
`degraded_reason`) on every retrieve response, and MAEDA now parses,
records and *validates* those fields instead of dropping them —
un-validated, an echoed field in a cross-process contract only creates
the impression of a guarantee.

The reason this matters is measurement, not operation: a retrieval that
silently fell back to BM25-only returns fewer/weaker but entirely
plausible chunks (附录 AL.3 measured dev Recall@5 0.7083 vs 0.9444
hybrid), so an on-arm trial that degraded looks exactly like evidence
that RAG isn't worth much. Under `MAEDA_RAG_MODE=force_on` — and only
there — a retrieval that ran below `MAEDA_RAG_EXPECTED_RETRIEVAL_MODE`,
degraded to the fallback path, or hard-failed sets
`state["rag_arm_invalid_reason"]`, which travels onto `EvalResult` and
makes `src/eval/trials.py::is_applicable` drop the trial from pass@k and
from the continuous summaries, with the reason reported per case.

The pipeline behaviour itself does not fork by mode: RAG is
enrichment-only, so a retrieval failure still degrades to "no domain
context, keep going" exactly as before. What changes is only whether the
resulting run is allowed to count as an on-arm data point.

## Guardrail design

The guardrail pipeline runs on every output, never optional. Design principle: fail safe.

- Rule-based checks (PII regex, SQL safety, completeness) are fast and deterministic
- LLM-as-judge (hallucination, claim grounding) catches subtle fabrications
- Critical failures trigger retry (up to `max_retries=2`); warnings attach caveats
- The retry loop is a real graph cycle: `guardrail → [retry path] → insight_agent → guardrail`

## Eval as a first-class concern

Most demo projects have no eval. MAEDA runs eval on every execution:
- LLM-as-judge for answer_relevance + groundedness (single batched call)
- Rule-based factual_accuracy (number overlap as proxy)
- Per-agent metrics derived from state (success rate, step count, etc.)
- Regression detection compares runs against a stored baseline

This means every demo run produces a quantitative quality score — a concrete interview talking point.

## Cost tracking

Every LLM call records model name, input/output tokens, and derived cost. `CostTracker` is injected into every agent via `BaseAgent`. Aggregate cost is surfaced in the Streamlit sidebar and eval scores. Budget limits (`max_cost_per_session`) raise `BudgetExceededError` before making the call.

## Graceful degradation

Three-tier fallback:
1. MCP call succeeds → use result
2. MCP unavailable → `SubSystemWithFallback` uses built-in pandas/rule-based logic
3. LLM call fails → every agent has a `_rule_based_*` fallback that produces deterministic output

This means MAEDA can run a full demo with no API key and no sub-systems running.
