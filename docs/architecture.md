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
