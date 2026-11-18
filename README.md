# MAEDA

[![CI](https://github.com/UrBaneee/MAEDA/actions/workflows/ci.yml/badge.svg)](https://github.com/UrBaneee/MAEDA/actions/workflows/ci.yml)

MAEDA answers natural-language questions about tabular business data and returns a
written report with charts. It is built as a LangGraph pipeline that calls two
independent services over MCP: a data cleaner
([agentic-data-cleaner-v2](https://github.com/UrBaneee/agentic-data-cleaner-v2)) and a
retrieval server ([rag-framework](https://github.com/UrBaneee/rag-framework)).

The main point of the project is the evaluation around it. I ran a pre-registered A/B
experiment to find out whether those two services actually make the reports better,
and I checked the LLM judges against human labels before trusting their scores.

## Results

Hypotheses, samples, metrics, decision rules and code were frozen on 2026-11-11
([pre-registration](docs/phase4_preregistration.md), §10.15). The sealed test sets were
revealed once, on 2026-11-12. The arms differ only in the two services: B (both off),
C (cleaning forced on), R (retrieval forced on), CR (both on), A (auto).

| Hypothesis | Comparison | Primary metric | Effect | Test (α = 0.0167, Bonferroni over 3) | Verdict |
|---|---|---|---|---|---|
| H1: cleaning helps | C − B | `factual_accuracy`, scored by code | +0.189 | 98.3% CI [+0.114, +0.272], 119 questions | **Holds** |
| H2: retrieval helps | R − B | `event_attribution_accuracy`, LLM judge | +0.653 | exact sign-flip over 12 events, p = 0.002 | **Holds** |
| H3: interaction | (CR − R) − (C − B) | `factual_accuracy` | −0.009 | 98.3% CI [−0.047, +0.028] | Not detected |

What the numbers mean:

- **Cleaning fixes numbers, retrieval supplies causes, and the two do not interfere.**
  With retrieval off, none of the 72 reports named an event's true cause.
- **The absolute level is still low.** With cleaning on, about 26% of numeric questions
  are answered correctly (0.067 with it off), because the cleaner does not repair the
  ×10 revenue outliers.
- **Auto mode never triggered cleaning**, so it gets none of the H1 gain. This was
  predicted before the reveal: the cleaner's trigger thresholds cannot be reached on
  this dataset.
- **The data is synthetic.** It is one generated sales table with injected defects
  (missing values, duplicate rows, ×10 outliers, negative revenue) plus a scripted set
  of business events and documents. The results say nothing directly about real data.
- **The H2 judge only just cleared its bar.** Its agreement with human pass 1
  (κ = 0.616) is above the human test-retest agreement (κ = 0.544), but the difference
  is not significant at n = 60, and against human pass 2 the judge's κ is only 0.258.
  Which pass to compare against was settled after both numbers were known. The
  pre-registration (§7.2.1) discloses all of this.

1,382 runs, 0 voided, about $42 in API cost. Full results, deviations and the files
needed to recompute every number offline: **[docs/phase4_results.md](docs/phase4_results.md)**.

## How it works

MAEDA is plan-and-execute, not an autonomous agent loop. An LLM planner writes the
whole analysis plan once, and plain Python code runs it. There is no tool-calling loop
anywhere in the code. "Agents" here means single-purpose classes that each make one
kind of LLM call.

```
question
  │
  ▼
parse_intent ........ LLM: query type, metrics, dimensions, filters
connect_schema ...... load the data source, read its schema
  (refine_intent) ... LLM: optional second intent pass, now aware of the schema
profile_and_clean ... cleaner over MCP: profile → clean → re-profile → validate
                      (retries up to 3 times; local pandas fallback if unreachable)
plan_analysis ....... LLM: typed step plan (tool, parameters, depends_on)
execute_analysis .... code: runs the plan in dependency order
generate_viz ........ chart (PNG + Plotly JSON) and an LLM caption
retrieve_domain_knowledge or skip_retrieval
                      retrieval server over MCP
generate_insights ... LLM: written report grounded in the results
run_guardrails ...... 10 checks; a critical failure re-runs execute_analysis (max 2)
run_eval ............ scores the run
persist_run ......... trace, MCP call log and scores go to SQLite
```

That is 14 nodes in [`src/graph/builder.py`](src/graph/builder.py), with one typed state
object (`MAEDAState`) passed through all of them.

**Sub-system switches.** Each service runs as `auto`, `force_on` or `force_off`
(`MAEDA_CLEANER_MODE`, `MAEDA_RAG_MODE`). In `auto`, cleaning runs only when the profile
reports critical issues, and retrieval runs only for diagnostic or exploratory
questions. The forced modes are what the A/B experiment uses.

**Guardrails.** Six rule-based checks: PII, SQL safety, numerical consistency,
completeness, population-claim grounding and superlative-claim grounding. Four
LLM-judged checks: factual accuracy, hallucination, PII leakage and misleading framing.
A run that fails a critical check ends as a `safe_refusal` rather than returning the
report.

**Failure handling.** Every run ends in one of five terminal states: `success`,
`safe_refusal`, `pipeline_error`, `mcp_error` or `environment_error`. When a service is
unreachable, MAEDA falls back (pandas profiling, empty retrieval context) and records
the fallback. A strict mode turns fallbacks into hard failures instead.

## Evaluation

Every run is scored. The everyday suite is `tests/eval/test_suite.json`, 130 dev cases
run with `scripts/run_eval.py`. It is for iterating against, not a holdout. The phase 4
holdout of 119 cases has already been revealed and is not reused for tuning.

| Metric | How it is scored | Checked against humans? |
|---|---|---|
| `factual_accuracy` | Code: numbers in the report vs computed ground truth, 1% tolerance | No judge involved |
| `event_attribution_accuracy` | LLM judge: correct / wrong / none against the event's documented cause | Met its bar on the point estimate (see Results) |
| `answer_relevance` | LLM judge, median of 3 calls | Met: QWK 0.73 vs human test-retest 0.40 |
| `groundedness` | LLM judge lists the claims; supported / total is computed in code | **Not met**: QWK 0.50 vs human test-retest 0.72 |
| `intent_accuracy`, `tool_selection`, `chart_appropriateness` | Code, against each case's expected values | No judge involved |
| cost, latency, retries, error rate, refusals | Measured | Not applicable |

The bar for a judge is relative, not absolute: its agreement with my labels must reach
my own agreement with myself when I relabel the same reports blind. Because
`groundedness` did not pass, no conclusion in this project depends on it.

A run that did not succeed is still scored on what it can support (cost, latency,
refusal). Metrics that would grade an answer it never produced are marked not
applicable instead of being scored 0.

There is also a weighted `aggregate_score`, used only for the console summary and for
regression alerts. Its weights were chosen by hand and never validated, and the largest
one goes to `groundedness`, so treat it as a quick-look number. Phase 4 deliberately
tied each hypothesis to a single metric instead.

More detail:
[judge calibration](docs/judge_calibration.md) ·
[noise floor](docs/noise_floor.md) ·
[eval v2 plan](docs/eval_v2_plan.md) ·
[eval-driven bug fixes](docs/eval_report.md) ·
[roadmap and known gaps](docs/roadmap.md)

## Quick start

Requires Python 3.11+ and an OpenAI or Anthropic API key.

```bash
git clone https://github.com/UrBaneee/MAEDA.git
cd MAEDA
pip install poetry && poetry install
cp .env.example .env      # add your API key(s)
```

The demo datasets are already in `data/demo/`. Don't run
`scripts/generate_demo_data.py` on a fresh clone: since phase 4 the sales table is
rendered by rag-framework and only copied in after a hash check, so the script fails
without that repository next to this one.

```bash
streamlit run ui/app.py                       # web UI
python scripts/demo_scenarios.py --scenario 1 # one demo scenario (1 to 5, or --all)
python scripts/run_eval.py --limit 5          # score the first 5 dev cases (paid LLM calls)
pytest tests/unit/                            # 1,078 unit tests, all LLM calls mocked
```

Without the two services running, MAEDA uses its fallbacks. To use them, start each
from its own repository (cleaner on `:8001`, retrieval on `:8002`) and set
`DATA_CLEANER_MCP_URL` and `RAG_SERVER_MCP_URL`. The `full-ecosystem` profile in
`docker-compose.yml` expects images built from those two repositories; they are not
published.

**As an MCP server.** MAEDA exposes `analyze_data`, `connect_data_source` and
`get_eval_report`:

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

## MCP tools MAEDA calls

| Service | Tools |
|---|---|
| Data cleaner (`:8001`) | `profile_dataset`, `clean_dataset`, `validate_quality` |
| Retrieval server (`:8002`) | `retrieve`, `retrieve_with_metadata`, `list_collections` |

Responses are checked for required fields and a contract version. A mismatch is an
error, not something to guess around.

## Repository layout

```
src/
  agents/        intent parser, analysis planner/executor, viz, insight, guardrail
  graph/         LangGraph builder, nodes, router, streaming progress
  mcp_client/    cleaner and retrieval clients, fallbacks, error classes
  mcp_server/    MAEDA as an MCP server (FastMCP)
  eval/          runner, metrics, agreement, noise and trial statistics, reveal guard
  tools/         data connector, SQL, stats/pandas, charts, glossary
  state/         MAEDAState and terminal states
  persistence/   SQLite run store
  config/        settings and all prompts
scripts/         eval runners, phase 4 run/analysis scripts, labeling export tools
tests/unit/      unit tests (no network, no API key)
tests/eval/      dev suite, phase 4 holdout and H2 cases
docs/            pre-registration, results, calibration, roadmap
ui/              Streamlit app
```

## Limitations

- One synthetic dataset and one scripted event set. The evaluation is careful, but its
  scope is narrow.
- The H2 test pool contains only events whose cause is documented in the corpus. How
  often the system blames a documented event that had no effect was not measured
  confirmatorily.
- The cleaner's auto trigger misses the defects that matter on this data, so auto mode
  gains nothing from cleaning. Changing the trigger is a contract change for a future
  round.
- The `groundedness` judge does not agree with humans well enough to rely on.

## License

MIT
