# CLAUDE.md — MAEDA Project

## Project Context

MAEDA (Multi-Agent Enterprise Data Analyst) is a production-grade multi-agent system built with LangGraph. It is the meta-orchestrator of a 3-project AI agent ecosystem:

- **MAEDA** (this project): LangGraph orchestrator + Intent Parser + Analysis Agent + Viz Agent + Guardrails + Eval
- **Agentic Data Cleaner** (external, via MCP): Handles data quality profiling and cleaning
- **RAG-MCP-Server** (external, via MCP): Handles domain knowledge retrieval

MAEDA delegates data quality work to Data Cleaner and knowledge retrieval to RAG-MCP-Server via MCP protocol. It NEVER rebuilds what those sub-systems already do.

## Full Specification

Read `DEV_SPEC.md` for complete architecture, state definitions, graph construction, and all 88 tasks across 13 phases.

## Key Architecture Rules

1. **LangGraph state graph** is the orchestration backbone. All agent flow uses conditional edges, not ad-hoc function calls.
2. **MAEDAState TypedDict** is the single source of truth. All inter-agent data passes through it.
3. **Every agent decision** must be logged to `decision_trace` with: agent_name, action, reasoning, inputs, outputs, confidence, timestamp.
4. **Every LLM call** must track token usage via `cost_tracker`.
5. **Sub-system calls go through MCP client** (`src/mcp_client/`). Never import from Data Cleaner or RAG-MCP-Server directly.
6. **Graceful degradation**: If a sub-system MCP is unavailable, fall back to basic built-in alternatives. MAEDA must be able to run standalone.
7. **Guardrail checks run on EVERY output** before user delivery. No exceptions.
8. **Eval module runs on every execution** — it's not optional.
9. All system prompts live in `src/config/agent_prompts.py`. Never hardcode prompts in agent files.
10. All configuration uses Pydantic Settings in `src/config/settings.py`.

## Development Workflow

1. Work through phases sequentially (Phase 1 → 2 → 3 → ...)
2. Each phase has numbered tasks — complete them in order
3. Write tests as you go, not at the end
4. After each phase, verify the graph still compiles and routes correctly
5. Phase 3 (MCP Integration) requires running Data Cleaner and RAG-MCP-Server — use fallback mode if they're not available during development

## Sub-System MCP Interfaces

### Data Cleaner MCP Tools (already implemented in external project)

```
profile_dataset:  input {path} → {row_count, columns: [{name, dtype, null_pct, unique_count, sample_values}], quality_issues: [...]}
get_cleaning_plan: input {path} → {steps: [{operation, target_column, rationale, estimated_impact}]}
clean_dataset:    input {path, plan?} → {cleaned_path, changes_summary, rows_affected}
validate_quality: input {path} → {passed, score, issues}
```

### RAG-MCP-Server Tools (already implemented in external project)

```
retrieve:               input {query, top_k} → {chunks: [{content, score, metadata}]}
retrieve_with_metadata: input {query, top_k} → {chunks: [{content, score, source_file, page, chunk_id}]}
list_collections:       input {} → {collections: [{name, doc_count, description}]}
```

## Current Progress

- [x] Phase 1: Project Scaffold & LangGraph Foundation
- [x] Phase 2: Intent Parser Agent
- [x] Phase 3: MCP Integration Layer
- [x] Phase 4: Data Source Connection
- [x] Phase 5: Analysis Agent
- [x] Phase 6: Visualization Agent
- [x] Phase 7: Insight Agent (RAG via MCP)
- [x] Phase 8: Guardrail Agent
- [x] Phase 9: Evaluation Module
- [x] Phase 10: MAEDA MCP Server
- [x] Phase 11: Streamlit UI
- [x] Phase 12: Ecosystem Demo & Datasets
- [x] Phase 13: Documentation & Interview Prep
