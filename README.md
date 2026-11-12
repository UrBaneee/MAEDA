# MAEDA — Multi-Agent Enterprise Data Analyst

[![CI](https://github.com/UrBaneee/MAEDA/actions/workflows/ci.yml/badge.svg)](https://github.com/UrBaneee/MAEDA/actions/workflows/ci.yml)

> A LangGraph-orchestrated plan-and-execute pipeline that takes a natural language question about enterprise data, connects to a data source, has an LLM planner emit a typed, dependency-ordered analysis plan, executes it, and delivers a charted insight report — with every output passing through a 9-check guardrail and a self-scored eval before it's returned.

**On the name**: this project is called MAEDA and the code is organized into five single-purpose LLM-calling classes ("agents") — but it is not an autonomous multi-agent system in the sense of agents that call tools in a loop, decide when they're done, or negotiate with each other. There is no `bind_tools`/`ToolNode`/`create_react_agent` anywhere in this codebase. The Analysis Planner emits its full step plan once; a separate, non-LLM executor runs it. What *is* real and demonstrated: a LangGraph state machine with 2 bounded retry loops (data-cleaning, guardrail-triggered re-execution), a typed step-plan the Planner writes and the Executor runs without further LLM involvement, and every output scored by an eval harness with measured human-agreement numbers (see [Evaluation](#evaluation) below) — see [docs/roadmap.md](docs/roadmap.md) for the full, current list of what's built vs. not.

**Headline result**: in a pre-registered A/B test, the data-cleaning sub-system raised numeric accuracy by +0.189 and the retrieval sub-system raised cause-attribution accuracy by +0.653, both significant after multiple-comparison correction ([details](#do-the-two-sub-systems-actually-help-a-pre-registered-ab-test-phase-4)).

<!-- Demo recording: save as docs/demo.gif (or .mp4) and uncomment the line below -->
<!-- ![MAEDA demo](docs/demo.gif) -->

---

## Architecture

14 LangGraph nodes, 4 conditional branches, 2 bounded retry loops, one typed
`MAEDAState` threaded through all of it:

```
┌──────────────────────────────────────────────────────────────┐
│                    MAEDA (This Project)                      │
│  LangGraph state machine · Eval · Guardrails · Viz          │
│                                                              │
│  parse_intent → connect_schema → (refine_intent) →           │
│       profile_and_clean (↻ clean, max 3) → plan_analysis     │
│       ↓                                                      │
│  execute_analysis → generate_viz → retrieve_domain_knowledge │
│       ↑ (↻ guardrail retry, max 2)   or skip_retrieval       │
│  run_guardrails ←──────────────── generate_insights          │
│       ↓                    (failure → handle_error)          │
│  run_eval → persist_run → END                                │
│                                                              │
│  ┌──────────────────┐    ┌──────────────────────────────┐    │
│  │  Agentic Data    │    │     RAG-MCP-Server            │    │
│  │  Cleaner (MCP)   │    │     (MCP)                     │    │
│  │  Data quality    │    │     Domain knowledge          │    │
│  │  Pandas fallback │    │     Fallback if server is     │    │
│  │  if unavailable  │    │     unavailable               │    │
│  └──────────────────┘    └──────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

**Key Design Decisions:**
- **LangGraph state graph** — all flow via conditional edges, typed `MAEDAState` as single source of truth; the exact node/edge/loop wiring is in [`src/graph/builder.py`](src/graph/builder.py)
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
├── eval/           EvalRunner, metrics, noise/significance tests, H2 summary, regression detection
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
├── unit/           1,078 tests, all phases covered
├── integration/    Live MCP integration tests
└── eval/           130-case dev golden suite, phase 4 H1 holdout (119, revealed) and H2 cases
data/demo/          Sales, churn, marketing, ecommerce datasets
```

---

## Pipeline

Plan-and-execute, not an agentic loop: the Planner decides the *what* once;
the Executor is a plain topological-sort runner with no further LLM
involvement per step (only a targeted, error-message-driven repair on
failure). `(LLM)` marks the stages where a model actually decides
something; everything else is deterministic code.

```
User Query
    │
    ▼
Intent Parser      (LLM) → parsed_intent
    │
    ▼
Connect schema     → schema extraction + table selection; optional schema-aware
                     second intent pass (Refine Intent, LLM)
    │
    ▼
Profile + Clean    → Data Cleaner MCP (profile → clean → re-profile → validate),
                     or local pandas fallback (↻ clean, max 3). In auto mode,
                     cleaning runs only if the profile reports critical issues.
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
Retrieve Knowledge → RAG-MCP-Server. In auto mode, only for diagnostic or
                     exploratory questions; otherwise skipped (and logged).
                     Degrades to empty context if the server is unreachable.
    │
    ▼
Insight Agent      (LLM) → grounded insights + markdown report
    │
    ▼
Guardrail Agent    → 10 checks: 6 rule-based (PII, SQL safety, numerical
                     consistency, completeness, population-claim and
                     superlative-claim grounding)
                     + 4 LLM-as-judge (factual accuracy, hallucination, PII
                     leakage, misleading framing) (↻ retry execute_analysis,
                     max 2, on a critical failure)
    │
    ▼
Eval Module        → answer_relevance, groundedness (LLM judges on a separate
                     provider), factual_accuracy, intent_accuracy, ...
    │
    ▼
Final Report + Charts + Eval Scores
```

---

## Evaluation

### Do the two sub-systems actually help? A pre-registered A/B test (phase 4)

MAEDA's two MCP sub-systems were tested in a pre-registered experiment: the
hypotheses, samples, metrics, decision rules and code were frozen on
2026-11-11 ([`docs/phase4_preregistration.md`](docs/phase4_preregistration.md) §10.15),
and the sealed test sets were revealed once, on 2026-11-12. Five arms switch
only the cleaner and retrieval: B (both off), C (cleaning on), R (retrieval on),
CR (both on), A (auto).

| Hypothesis | Comparison | Primary metric | Effect | Test (α = 0.0167, Bonferroni over 3) | Verdict |
|---|---|---|---|---|---|
| H1: cleaning net gain | C − B | `factual_accuracy` (scored by code) | +0.189 | 98.3% CI [+0.114, +0.272] | **Holds** |
| H2: retrieval net gain | R − B | `event_attribution_accuracy` (LLM judge, calibrated against humans) | +0.653 | exact sign-flip over 12 events, p = 0.002 | **Holds** |
| H3: interaction | (CR − R) − (C − B) | `factual_accuracy` | −0.009 | 98.3% CI [−0.047, +0.028] | Not detected |

What it means, and its limits:

- Cleaning fixes numbers, retrieval supplies causes, and they don't interfere.
  With retrieval off, no report named an event's true cause.
- The absolute level is still low: with cleaning on, about 26% of numeric
  questions are answered correctly, because the cleaner does not repair ×10
  revenue outliers.
- Auto mode never triggered cleaning, so it gets none of the H1 gain. This was
  predicted before the reveal: the cleaner's gate cannot fire on this data.
- All data is synthetic (one demo sales table with injected defects, one
  scripted event set), so these results do not carry over directly to real data.

1,382 runs, 0 voided, about $42 in API cost. Full results, deviations and
reproduction files: **[docs/phase4_results.md](docs/phase4_results.md)**.

### The eval harness

Every run is scored automatically. The everyday suite is
`tests/eval/test_suite.json`: 130 dev cases, stratified by query type and
difficulty, run with `scripts/run_eval.py`. It is for iterating against, not
a holdout: the original 41-case test split was consumed in August 2026 and
retired to dev, and phase 4's one-time holdout (119 H1 cases) was revealed
on 2026-11-12 (see above).

| Metric | Method | Human agreement | Aggregate weight |
|--------|--------|-----------------|------------------|
| factual_accuracy | Code: numbers in the report vs the case's ground truth (1% tolerance); falls back to overlap with the analysis results when a case has no ground truth | no judge involved | 2.0 |
| answer_relevance | LLM judge, median of 3 calls | met the pre-registered bar (QWK 0.73 vs human retest 0.40) | 3.0 |
| groundedness | LLM judge extracts a claim list; supported/total computed in code | **did not meet** the bar (QWK 0.50 vs human retest 0.72) | 3.0 |
| event_attribution_accuracy | LLM judge: correct / wrong / none against the event's documented cause (H2 only) | met its bar before the freeze | 0 (exists on event cases only) |
| intent_accuracy | Code: query type 0.4, metrics 0.3, dimensions 0.3, against each case's expectations | no judge involved | 1.5 |
| tool_selection | Code: against each case's expected tool(s) | no judge involved | 1.0 |
| chart_appropriateness | Code: against each case's expected chart type(s) | no judge involved | 0.5 |
| token_cost, retry_count, error_rate, total_latency, step_success_rate, safe_refusal | Operational measurements | not applicable | 0 |

**About the weights.** `aggregate_score` (`src/eval/runner.py::_aggregate_score`)
still exists: it gives one number per run for the console summary and for
regression detection, which alerts on any metric or the aggregate dropping by
more than 0.05 vs a baseline (more than 0.20 is critical). The weights were
chosen by hand and have never been validated, and the largest one goes to
`groundedness`, the judge that did not meet the human-agreement bar. So the
aggregate is a quick-look number only. **No conclusion in this project rests
on it**: phase 4 bound each hypothesis to exactly one primary metric and
reported everything else descriptively, precisely to avoid a hand-weighted
composite. Operational metrics stay at weight 0 because a healthy system's
error_rate and step_success_rate sit at a constant 1.0 and would only dilute
a quality average.

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
pytest tests/unit/ -v          # 1,078 tests, all phases
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
