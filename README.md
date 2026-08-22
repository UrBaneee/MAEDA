# MAEDA — Multi-Agent Enterprise Data Analyst

[![CI](https://github.com/UrBaneee/MAEDA/actions/workflows/ci.yml/badge.svg)](https://github.com/UrBaneee/MAEDA/actions/workflows/ci.yml)

> A LangGraph-orchestrated plan-and-execute pipeline that takes a natural language question about enterprise data, connects to a data source, has an LLM planner emit a typed, dependency-ordered analysis plan, executes it, and delivers a charted insight report — with every output passing through a 9-check guardrail and a self-scored eval before it's returned.

**On the name**: this project is called MAEDA and the code is organized into five single-purpose LLM-calling classes ("agents") — but it is not an autonomous multi-agent system in the sense of agents that call tools in a loop, decide when they're done, or negotiate with each other. There is no `bind_tools`/`ToolNode`/`create_react_agent` anywhere in this codebase. The Analysis Planner emits its full step plan once; a separate, non-LLM executor runs it. What *is* real and demonstrated: a LangGraph state machine with 3 bounded retry loops (clarification, data-cleaning, guardrail-triggered re-execution), a typed step-plan the Planner writes and the Executor runs without further LLM involvement, and every output scored by an eval harness with measured human-agreement numbers (see [Evaluation](#evaluation) below) — see [docs/roadmap.md](docs/roadmap.md) for the full, current list of what's built vs. not.

<!-- Demo recording: save as docs/demo.gif (or .mp4) and uncomment the line below -->
<!-- ![MAEDA demo](docs/demo.gif) -->

---

## Architecture

12 LangGraph nodes, 3 conditional branches, 3 bounded retry loops, one typed
`MAEDAState` threaded through all of it:

```
┌──────────────────────────────────────────────────────────────┐
│                    MAEDA (This Project)                      │
│  LangGraph state machine · Eval · Guardrails · Viz          │
│                                                              │
│  parse_intent → connect_and_profile_data → plan_analysis     │
│       ↓ (↻ clarify, max 1)      ↓ (↻ clean, max 3)          │
│  execute_analysis → generate_viz → retrieve_domain_knowledge │
│       ↑ (↻ guardrail retry, max 2)         ↓                │
│  run_guardrails ←──────────────── generate_insights          │
│       ↓                                                      │
│  run_eval → persist_run → END                                 │
│                                                              │
│  ┌──────────────────┐    ┌──────────────────────────────┐    │
│  │  Agentic Data    │    │     RAG-MCP-Server            │    │
│  │  Cleaner (MCP)   │    │     (MCP)                     │    │
│  │  Data quality    │    │     Domain knowledge          │    │
│  │  100% fallback   │    │     100% fallback             │    │
│  │  in this env     │    │     in this env               │    │
│  └──────────────────┘    └──────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

**Key Design Decisions:**
- **LangGraph state graph** — all flow via conditional edges, typed `MAEDAState` as single source of truth; see [docs/roadmap.md](docs/roadmap.md) for the exact node/edge/loop inventory
- **Plan-and-execute, not an agentic tool-calling loop** — the Analysis Planner (LLM) emits a full typed step plan (tool, parameters, dependencies) once; a separate, non-LLM Executor runs it topologically. A failed step gets one LLM-assisted repair attempt using the tool's real error message, not a re-plan.
- **MCP protocol** for sub-system communication — designed so MAEDA delegates rather than reimplements data-quality profiling and domain retrieval. Verified end-to-end against real Data Cleaner + RAG-MCP-Server processes (not mocks): a live dirty dataset goes profile → clean → re-profile → validate and converges in-graph, cleaned files and quality reports are genuinely consumed by later rounds, and real OpenAI-embedded hybrid retrieval with a Voyage/CrossEncoder reranker returns re-ranked, source-attributed chunks. The strict/degraded error matrix (connection, data-input, format, contract-version, RAG zero-hit vs internal failure) has also been exercised live, not just unit-tested. The pandas/empty-context fallback path still exists and is exercised whenever a sub-system is genuinely unreachable, but it is no longer the *only* path that's been run.
- **Every LLM call tracked** — per-request cost tracker, decision trace on every node, eval scores on every run
- **Per-agent model tiering** — planner and guardrail judge run on a stronger model; high-volume generation stays on the cheap default
- **Every run persisted** — decision trace / MCP call log / eval scores land in SQLite via a terminal graph node, auditable after the process exits
- **Multi-turn conversation memory** — follow-ups like "now break that down by quarter" resolve against the prior turn's intent
- **Real streaming progress** — the UI reflects which graph node actually just finished, not a canned animation

---

## Quick Start

### 1. Install
```bash
git clone https://github.com/UrBaneee/MAEDA.git
cd MAEDA
pip install poetry && poetry install
cp .env.example .env   # add your API key
```

### 2. Generate demo data
```bash
python scripts/generate_demo_data.py
```

### 3. Run the Streamlit UI
```bash
streamlit run ui/app.py
```

### 4. Full ecosystem (Docker)
```bash
docker-compose up maeda                          # MAEDA standalone
docker-compose --profile full-ecosystem up       # with Data Cleaner + RAG Server
```

### 5. Use as an MCP server (Claude Desktop)
Add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "maeda": {
      "command": "python",
      "args": ["-m", "src.mcp_server.server"],
      "cwd": "/path/to/MAEDA"
    }
  }
}
```

---

## Demo Scenarios

| # | Scenario | Dataset | Showcases |
|---|----------|---------|-----------|
| 1 | The Full Stack | `sales_data.csv` | End-to-end pipeline, Data Cleaner integration |
| 2 | Q3 Revenue Drop | `sales_data.csv` | Diagnostic multi-step analysis |
| 3 | Churn Spike | `churn_data.csv` | Root cause analysis + RAG enrichment |
| 4 | Marketing ROI | `marketing_campaigns.csv` | Comparative analysis + charts |
| 5 | Graceful Degradation | `ecommerce_orders.db` | MAEDA without sub-systems |

```bash
python scripts/demo_scenarios.py --scenario 1
python scripts/demo_scenarios.py --all
```

---

## Project Structure

```
src/
├── agents/         Intent parser, Analysis, Viz, Insight, Guardrail
├── eval/           EvalRunner, metrics, 100-case golden suite (59 dev / 41 held-out test), regression detection
├── graph/          LangGraph builder, nodes, router, streaming
├── mcp_client/     Data Cleaner + RAG Server MCP clients with fallbacks
├── mcp_server/     MAEDA-as-MCP-server (FastMCP)
├── persistence/    SQLite run store — every run's trace survives the process
├── state/          MAEDAState TypedDict
├── tools/          Chart tool, data connector, SQL, stats, anomaly, time-series
├── config/         Settings (Pydantic), all agent prompts
└── utils/          Logger, cost tracker

ui/                 Streamlit app (Phase 11)
scripts/            Demo data generator, demo scenarios, eval harness
tests/
├── unit/           547 tests, all phases covered
└── eval/           Golden test suite JSON
data/demo/          Sales, churn, marketing, ecommerce datasets
```

---

## Pipeline

Plan-and-execute, not an agentic loop: the Planner decides the *what* once;
the Executor is a plain topological-sort runner with no further LLM
involvement per step (only a targeted, error-message-driven repair on
failure). `(LLM)` marks the four stages where a model actually decides
something; everything else is deterministic code.

```
User Query
    │
    ▼
Intent Parser      (LLM) → parsed_intent, clarification if needed (↻ max 1)
    │
    ▼
Connect + Profile  → schema, NL summary  ──→  Data Cleaner MCP, or local
                                              pandas fallback (↻ clean, max 3)
    │
    ▼
Analysis Planner   (LLM) → typed step plan: tool, parameters, depends_on
    │
    ▼
Analysis Executor  → runs the plan topologically: pandas / SQL / stats /
                     anomaly / time-series / comparison tools
    │
    ▼
Viz Agent          → recommend chart → static PNG + Plotly JSON → LLM caption
    │
    ▼
Retrieve Knowledge → RAG-MCP-Server, or skipped if unreachable
    │
    ▼
Insight Agent      (LLM) → grounded insights + markdown report
    │
    ▼
Guardrail Agent    → 9 checks: 5 rule-based (PII, SQL safety, numerical
                     consistency, completeness, population-claim grounding)
                     + 4 LLM-as-judge (factual accuracy, hallucination, PII
                     leakage, misleading framing) (↻ retry execute_analysis,
                     max 2, on a critical failure)
    │
    ▼
Eval Module        → answer_relevance, groundedness (independent LLM judges),
                     factual_accuracy, intent_accuracy, tool_selection, ...
    │
    ▼
Final Report + Charts + Eval Scores
```

---

## Evaluation

Every run is automatically scored against a 100-case golden suite
(59 dev / 41 held-out test, stratified by query type and difficulty) via
`scripts/run_eval.py --split dev` (the 41-case test split is a one-time
official reveal, gated behind `--split test`, not for iterating against).

| Metric | Weight | Method |
|--------|--------|--------|
| answer_relevance | 3.0 | LLM-as-judge, independent call |
| groundedness | 3.0 | LLM-as-judge: extracts a claim list, scored as supported/total in code (not asked as a bare float) |
| factual_accuracy | 2.0 | Number overlap between report and analysis results |
| intent_accuracy | 1.5 | Confidence + type match |
| tool_selection | 1.0 | Checked against each case's expected tool(s) |
| chart_appropriateness | 0.5 | Checked against each case's expected chart type(s) |
| token_cost, retry_count, error_rate, total_latency, step_success_rate, safe_refusal | 0.0 | Reported and tracked for regression detection, deliberately excluded from the aggregate — a healthy system's error_rate/step_success_rate *should* sit at a constant 1.0, so folding a non-discriminating metric into a quality average only dilutes it |

Regression detection alerts on any metric drop > 5% vs baseline.

### Failed runs are scored too, on less

Every run that reaches a terminal node is classified into one of five
terminal states (`src/state/terminal_state.py`): `success`, `safe_refusal`,
`pipeline_error`, `mcp_error`, `environment_error`. `mcp_error` and
`environment_error` are split out from the old catch-all `pipeline_error`
so sub-system availability can be read separately from agent reasoning
failures; the sub-classification carried alongside is the MCP client's own
`error_class`, not a second taxonomy.

A run that did not end in `success` is still scored — on the metrics it can
support. Its `error_rate`, `safe_refusal`, latency and cost are real
measurements; the metrics that score an answer it never produced are marked
`not_applicable` (`valid=False`) rather than scored zero, and the LLM judge
is not invoked for them at all. Such a run therefore has **no aggregate
score**: `aggregate_score` is `null`, not `0.0`, because its surviving
metrics are a different set than a successful run's. `overall_aggregate` in
a saved report is a mean over the successful cases only, with
`n_cases_without_aggregate` recording how many were left out.

### Human-vs-judge calibration (eval v2)

The harness rebuild above answers "does the pipeline work." A separate,
later effort answers a question that matters more for an eval-focused
role: **does the LLM judge scoring it actually agree with a human?**
120 human labels across 5 independent sessions (4 batches + a
self-consistency retest) found the judge's `groundedness` score agrees
with itself when re-labeling blind two days later (QWK = 0.775,
"acceptable") but agrees with the judge only weakly (QWK = 0.04–0.26
across every one of 5 comparisons) — ruling out human-rating noise as the
explanation and confirming a real, reproducible judge-calibration gap.
The specific mechanism — the judge verifies a cited number is real but
never checks whether a *comparative* claim built on it ("X is the
highest") is actually true against other values in the same
findings — was independently found in 4 separate labeled cases, fixed
(`EVAL_GROUNDEDNESS_SYSTEM`), and re-measured (pooled groundedness
QWK 0.141 → 0.185 — a real, partial improvement, reported honestly as
not a full fix, and explicitly caveated as *not* a blind-holdout
measurement). Two further pipeline bugs (a Planner defect and a
Guardrail truncation bug) were found the same way and fixed with a
noise-calibrated before/after (refusal rate 33.9% → 11.9%, checked
against a measured noise floor before being trusted as signal, not just
eyeballed). Full methodology, every batch's numbers, and the specific
evidence behind each finding: **[docs/judge_calibration.md](docs/judge_calibration.md)**
and **[docs/eval_v2_plan.md](docs/eval_v2_plan.md)**.

### The eval-first debugging pass (earlier, pre-eval-v2 history)

The system was originally iterated exclusively through its own eval harness,
predating the calibration work above: establish a baseline on the (then)
20-case golden suite, root-cause one failure, fix it, re-run, record the
delta. The specific numbers below are historical — the golden suite, its
scoring, and the baseline have all since changed (see above) — kept as a
record of the debugging methodology itself. **35 documented fixes** came out
of this pass — each with root
cause, code location, and before/after verification in
**[docs/eval_report.md](docs/eval_report.md)** — spanning silent
tool-parameter mismatches, a guardrail severity misclassification that let
hallucinated reports through, an MCP transport bug that had never actually
spoken the protocol, an eval judge that couldn't match numbers formatted
differently, a 33× cost-pricing overcount, and a cross-request cost leak in
the singleton agents.

The early arc, as a taste of why raw aggregate can't be read naively:

| Baseline | Aggregate score | Cases blocked by guardrail |
|---|---|---|
| Before fixes | 0.71 | 0 / 20 *(inflated — guardrail wasn't actually catching fabrication)* |
| After guardrail severity fix | 0.67 | 7 / 20 *(score drop = guardrail correctly blocking hallucinated reports for the first time)* |
| After the first 10 fixes | 0.76 | 2 / 20 *(both benign — a completeness complaint and a judge false-positive)* |

The same pattern repeated at the end: upgrading the guardrail's judge to a
stronger model moved the aggregate 0.751 → 0.725 while safe-refusals rose
4 → 6 — cross-referencing per-case reasoning traces showed all six were real
fabrications the cheaper judge had silently passed. Score deltas were only
ever trusted after reading the traces behind them.

Full run history and archived reports: `logs/eval_runs/`.
Prioritized gap analysis and what was deliberately *not* built:
**[docs/roadmap.md](docs/roadmap.md)**.

---

## Tests

```bash
pytest tests/unit/ -v          # 547 tests, all phases
pytest tests/unit/test_phase9.py -v   # eval module only
```

---

## Sub-System MCP Interfaces

**Agentic Data Cleaner** (`http://localhost:8001`):
- `profile_dataset(path)` → quality report
- `get_cleaning_plan(path)` → cleaning steps
- `clean_dataset(path, plan)` → cleaned file
- `validate_quality(path)` → pass/fail score

**RAG-MCP-Server** (`http://localhost:8002`):
- `retrieve(query, top_k)` → chunks with scores
- `retrieve_with_metadata(query, top_k)` → chunks with source attribution
- `list_collections()` → available knowledge bases
