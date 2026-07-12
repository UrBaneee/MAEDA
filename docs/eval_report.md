# MAEDA Eval Report

This is a record of an eval-first debugging pass: establish a baseline on the
golden suite, find out why the score is what it is, fix root causes, and
re-measure. All numbers below come from real `graph.invoke()` runs (fallback
mode — both sub-system MCP servers offline), not unit tests with mocked LLMs.

## How to reproduce

```bash
poetry run python scripts/run_eval.py                       # full 20-case suite
poetry run python scripts/run_eval.py --limit 3              # smoke test
poetry run python scripts/run_eval.py --case DG01 --case C04
poetry run python scripts/run_eval.py --compare logs/eval_runs/baseline_fallback.json
```

Each run scores all 20 golden cases (`tests/eval/test_suite.json`) with
`EvalRunner` and writes a timestamped report to `logs/eval_runs/`.

## Baseline timeline

| Report | Overall aggregate | Cases blocked by guardrail | What changed |
|---|---|---|---|
| `baseline_fallback.json` | 0.71 | 0 / 20 | Starting point — no fixes applied |
| `after_fixes.json` | 0.67 | 7 / 20 | Bugs 1–3 fixed: guardrail starts correctly catching fabrication it previously waved through |
| `after_all_fixes.json` | 0.77 | 2 / 20 | Bugs 4–6 fixed: Planner grounded in real columns, honest predictive boundary, derived columns |
| `after_insight_fix.json` | 0.78 | 3 / 20 | Bug 7 fixed: Insight Agent sees real tool output instead of a one-line summary |
| `final_baseline.json` | 0.76 | 2 / 20 | Bugs 8–10 fixed: anomaly/groupby data surfaced, filter op validated, evidence-level classification |
| `after_safe_refusal.json` | 0.80 | 2 / 20 | Phase A #11: `error_rate` no longer conflates safe refusals with crashes |
| `phase_a_ground_truth.json` | 0.70 | 5 / 20 | `ground_truth` backfilled with real computed values (see Phase A below) — score drop reflects `factual_accuracy` finally checking something real, and immediately catching bugs #12 and one instance of #14 |
| `phase_a_final.json` | 0.76 | 3 / 20 | Bugs 12–15 fixed |
| `phase_a_judge_calibration.json` | 0.71 | 4 / 20 | Phase A #17: judge upgraded to a stronger, independent model + 3-sample median scoring — see below |
| `phase_a_tool_audit.json` | 0.73 | 3 / 20 | Phase A #18: `statistical_tool`/`anomaly_tool`/`comparison_tool` audited for the same silent-default pattern fixed earlier in `pandas_tool`/`sql_tool` — roughly flat, as expected (these fixes turn wrong-but-silent behavior loud, not more queries into successes) |
| `phase_b_joins.json` | 0.71 | 5 / 20 | Phase B #1/#2/#4: cross-table joins, date-part derivation — see below. Roughly flat overall (removing D02's automatic data-mismatch pass and subjecting it to real scoring accounts for most of the movement); D02 itself went from structurally unanswerable to aggregate=0.91 |
| `baseline_b_rag_online.json` | 0.72 | 6 / 20 | Phase B #10: RAG-MCP-Server connected and confirmed genuinely live for all 20 cases (`mcp_modes` includes `'mcp'`, not just `'fallback'`); Data Cleaner intentionally left offline — see below. Aggregate flat; per-case swings in both directions largely cancel out |

**The score dropping from 0.71 to 0.67 is not a regression.** The initial 0.71
was inflated by silent failures: guardrail's own severity mapping hardcoded
hallucination to "warning" (never blocking), so fabricated reports were
delivered and scored by lenient proxy metrics. Once guardrail started
correctly blocking those reports, `error_rate` — which cannot distinguish "the
pipeline correctly refused a fabricated answer" from "the pipeline crashed" —
penalized both identically. The subsequent climb to 0.76–0.78 reflects real
quality improvement: by the final run, only 2/20 cases are blocked, and both
are benign (one completeness complaint, one guardrail false-positive on an
already-verified-correct forecast number) rather than genuine fabrication.

## Bugs found and fixed

Each entry: root cause, fix location, how it was verified.

### 1. `pandas_tool` silently defaulted on parameter mismatch
**Root cause:** `pandas_transform` dispatched on `parameters.get("operation", "groupby")`. When the Planner's step parameters didn't include an `"operation"` key (which was most of the time — the Planner used semantically-named keys like `filter_column`, `current_quarter`, `target_columns` that don't match the tool's contract), every step silently fell into the "groupby" branch with heuristically-guessed columns, ignoring the step's actual intent. Only steps that happened to pass a real column name under a recognized key (`group_by`) surfaced as a hard `KeyError`.
**Fix:** [src/tools/stats_tool.py](../src/tools/stats_tool.py) — `_infer_operation()` only recognizes an explicit `operation` key or a known key for that operation; unrecognized parameter sets raise `ValueError` instead of defaulting. `_require_columns()` validates every referenced column exists.
**Verified:** DG01/C01/C04 steps that previously produced identical, wrong default output now fail loudly with a message naming the exact bad parameter/column.

### 2. `sql_tool` fallback query builder had the same disease
**Root cause:** When no raw `query` was given, the builder silently ignored unrecognized keys (e.g. `"aggregate"`, `"table"` instead of `"table_name"`) and built a degraded `SELECT * ... LIMIT 100` that dropped the intended aggregation.
**Fix:** [src/tools/sql_tool.py](../src/tools/sql_tool.py) — `_SQL_BUILDER_KEYS` whitelist raises on unrecognized keys; referenced columns are validated against the dataframe. Column validation only applies to bare identifiers (`_is_bare_identifier`), so legitimate SQL expressions like `"AVG(unit_price - cost) AS margin"` in `select_columns` aren't rejected — this was a self-inflicted false-positive caught during verification and fixed the same day.
**Verified:** C01/C04 SQL steps now fail clearly instead of silently degrading; the fixed version still executes a valid aggregate query with expressions in `select_columns`.

### 3. Guardrail hardcoded hallucination to "warning" severity
**Root cause:** [src/agents/guardrail_agent.py](../src/agents/guardrail_agent.py) `_parse_llm_checks()` only escalated `pii`/`safety`-named checks to `critical`; hallucination/fabrication findings were always `warning` (caveat + deliver), contradicting DEV_SPEC's own failure-handling table ("Critical fail (hallucination): Block output, retry"). A pre-existing unit test (`test_llm_judge_flags_hallucination`) explicitly asserted the warning behavior with a comment noting it was a deliberate prior change — not an oversight, but still spec-violating.
**Fix:** Added `_CRITICAL_CHECK_KEYWORDS` (`hallucin`, `fabricat`, `claim_ground`, `grounding`, plus existing `pii`/`safety`) so these checks escalate to critical and correctly trigger the retry/fail path. Updated the test to assert the spec-correct behavior.
**Verified:** A direct test constructing a failed "Hallucination" check now produces `overall_verdict="retry"` via `_aggregate()`. In the DG01 case that originally exposed this, the guardrail's own findings (fabricated Q3/Q4 comparison) had been correctly identified but not acted on; after the fix, equivalent fabrication is blocked.

### 4. Analysis Planner never saw real column names
**Root cause:** [src/agents/analysis_agent.py](../src/agents/analysis_agent.py) `plan()` only passed `state["schema_summary"]` (an LLM-generated prose paragraph) to the Planner — never the actual column list. The Planner had to guess/paraphrase identifiers from prose, producing plausible-but-wrong names (`category`→`product_category`, `channel`→`marketing_channel`, `tenure_months`→`"tenure in months"`).
**Fix:** Added `_column_manifest()` which renders `state["active_source"]["schema"]["columns"]` (name + dtype) and injects it as an "### Available Columns" section, explicitly marked authoritative. `ANALYSIS_PLANNER_SYSTEM` ([src/config/agent_prompts.py](../src/config/agent_prompts.py)) now instructs the Planner to use these names verbatim and never invent one.
**Verified:** Re-running C04/DG04/D05 after the fix showed correct real column names (`category`, `unit_price`, `cost`, `tenure_months`, `order_date`, etc.) in every generated plan across multiple regenerations.

### 5. Predictive queries always invented a nonexistent ML pipeline
**Root cause:** MAEDA has no model-training capability, but nothing told the Planner that. For every "predictive" query it planned a `feature_engineering → train_test_split → model_training → model_evaluation → forecast` sequence with no corresponding tool implementation — 100% failure rate on this query type (P01–P03, 15% of the golden suite).
**Fix:** Extended `analyze_time_series()` ([src/tools/stats_tool.py](../src/tools/stats_tool.py)) with a `forecast_periods` parameter that extrapolates the already-computed linear trend (slope/intercept), returning predicted values plus an explicit caveat that this is a naive extrapolation, not a trained model. `ANALYSIS_PLANNER_SYSTEM` now states plainly that no ML training exists and predictive queries must use `time_series` + `forecast_periods`.
**Verified:** P02 now plans a single honest `time_series` step with `forecast_periods=30` instead of a 5-step fake ML pipeline.

### 6. No way to compute a derived column before aggregating on it
**Root cause:** Queries like "which product categories have the highest margin" require `margin = unit_price - cost` before grouping by category — a two-step reasoning chain the tool set didn't support at all.
**Fix:** Added a `"derive"` operation to `pandas_tool` ([src/tools/stats_tool.py](../src/tools/stats_tool.py)) supporting `+`,`-`,`*`,`/` between two columns or a column and a constant (deliberately not a general `eval()` — no arbitrary expressions). Documented in the Planner's tool contract with an explicit instruction to derive before referencing.
**Verified:** C04 now reliably plans `derive(margin = unit_price - cost) → groupby(category, mean(margin))` and produces correct, verifiable output (Books highest at 217.83, Office Supplies lowest at 43.67 — checked against the raw data by hand).

### 7. Insight Agent only saw a one-line summary, not real tool output
**Root cause:** `_format_findings()` ([src/agents/insight_agent.py](../src/agents/insight_agent.py)) passed only `result_summary` to the Insight Agent — a terse paraphrase that dropped the actual forecast values, trend statistics, and tool-emitted caveats living in the full `result` dict. With nothing concrete to cite, the LLM invented plausible-sounding numbers (a fabricated "85% confidence level" appeared in both P01 and P02, unconnected to anything in the data).
**Fix:** Added `_extract_result_detail()` to surface `trend`, `forecast` (with caveat), `significance_test`, and a bounded generic fallback into the findings text. Strengthened `INSIGHT_GENERATOR_SYSTEM`/`REPORT_WRITER_SYSTEM` to forbid stating any number not present in the findings, and to require carrying tool caveats forward verbatim.
**Verified:** P01 aggregate score 0.85→0.99, P02 0.83→0.87. Both reports now cite the real predicted value and the "naive linear extrapolation, not a trained model" caveat, with confidence scores (0.3–0.4) explicitly justified by low r².

### 8. `_extract_result_detail` didn't cover anomaly detection or groupby/filter results
**Root cause:** The fix in #7 only special-cased `trend`/`forecast`/`caveat`/`significance_test` keys. Anomaly detection results (`n_outliers`, `outlier_values`) had no special case and only survived via a generic "<400 chars, dump verbatim" fallback — anything larger was silently dropped. List-shaped results (groupby/filter/sql — the majority of steps) hit `isinstance(result, dict)` and returned empty immediately.
**Fix:** Added explicit handling for list results (bounded row sample) and anomaly-shaped dicts (`n_outliers`, `outlier_pct`, sample `outlier_values`). The generic fallback now truncates instead of silently dropping when over the length threshold.
**Verified:** E02 (anomaly detection) and E03 (customer purchasing patterns, groupby-heavy) both went from guardrail-blocked (fabricated outlier/date claims) to clean, correctly-grounded reports.

### 9. `pandas_filter` silently no-opped on unrecognized operators
**Root cause:** The Planner used `"op": "="` (single equals); `pandas_filter`'s if/elif chain only matches `"=="` and has no `else`, so the condition was silently skipped and the "filtered" dataframe stayed unfiltered — discovered while investigating DG02's "last month" filter, which returned all 5,000 rows instead of a filtered subset.
**Fix:** Replaced the if/elif chain with an `_FILTER_OPS` dict lookup that raises `ValueError` for any unrecognized operator.
**Verified:** Unrecognized ops now fail the step (caught by the existing retry/failure path) instead of silently returning unfiltered data. Column-existence behavior for `pandas_filter` itself was left untouched — that's validated one layer up by `pandas_tool`, and an existing test intentionally covers the direct-call case.

### 10. Insight Agent generalized single sample rows into population claims
**Root cause:** When an aggregate step failed, the Insight Agent still had access to raw preview rows from an earlier successful filter/derive step (which include real column values for a handful of individual records). It would cite one customer's plan type or one product's price as if it were evidence of a group-wide pattern ("churned customers are more likely to be on the Premium plan") — not fabricating numbers, but treating an anecdote as a statistic.
**Fix:** `_format_findings()` now tags every successful step `[AGGREGATE]`, `[ROW-LEVEL SAMPLE]`, or `[UNKNOWN]` based on the tool/operation that produced it (groupby/pivot/statistical_test/anomaly_detection/time_series/comparison → aggregate; filter/derive → row-level). `INSIGHT_GENERATOR_SYSTEM` explicitly restricts group-level claims to `[AGGREGATE]`-tagged evidence and requires `[ROW-LEVEL SAMPLE]` data to be framed as an example, never a pattern. Failed steps are now included in the findings text (previously filtered out entirely) so the Insight Agent knows what wasn't answered instead of silently substituting something else.
**Verified:** Re-running DG02 across two plan variations — one where the aggregate step failed (report now says "one customer... insufficient data to analyze churn factors" instead of a population claim) and one where a real `comparison` (ANOVA) step succeeded (report correctly cites "p-value of 0.0, test statistic of 32.8209").

## Phase A — trustworthiness follow-ups (see [roadmap.md](roadmap.md))

Continuation of the same debugging methodology, working through Phase A of
the roadmap: make the eval numbers themselves trustworthy before touching
the data layer.

### 11. `error_rate` conflated safe refusals with crashes
**Root cause:** [src/graph/nodes.py](../src/graph/nodes.py) `handle_error_node()` set `state["error"]` for two entirely different situations — guardrail exhausting retries on a genuinely bad output (a correct, safe refusal) and a real pipeline failure (no data source, unhandled exception) — with no way to tell them apart downstream.
**Fix:** Added `state["error_type"]` (`"safe_refusal"` | `"pipeline_error"`), set based on whether `handle_error_node` was reached via the guardrail "fail" path. `score_system_metrics()` ([src/eval/metrics.py](../src/eval/metrics.py)) now only fails `error_rate` for genuine crashes, and reports a separate informational `safe_refusal` metric (weight 0 in the aggregate — see `_aggregate_score` in [src/eval/runner.py](../src/eval/runner.py)).
**Verified:** A live run with 2 guardrail-blocked cases showed `error_rate=1.0 (pass)` + `safe_refusal=1.0` for both, instead of the old `error_rate=0.0 (fail)`; overall aggregate rose from 0.76 to 0.80 purely from this reclassification, with regression detection showing no false alarms.

### 12. `ground_truth` was empty for all 20 golden cases
**Root cause:** `tests/eval/test_suite.json` shipped with `"ground_truth": {}` everywhere, so `factual_accuracy` always fell back to a loose number-overlap heuristic (or an automatic 0.8 "no numbers to check" pass) instead of checking against a real value.
**Fix:** Computed real ground truth directly from `data/demo/*` with pandas for all 16 answerable cases (e.g. `north_region_revenue: 1363760.55`, `books_avg_margin: 217.83`); the 4 known data-mismatch/predictive cases (D02, DG04, C03, P03/P01/P02) carry an explicit `"_note"` explaining why instead. `src/eval/runner.py`'s `_builtin_golden_suite()` fallback mirrors the same values; a test asserts the two stay in sync.
**Verified:** `factual_accuracy` now reports real overlap percentages ("Ground truth overlap: 67%") instead of the old heuristic messages. This single change is what surfaced bugs #13 and one instance of #14 below — both were invisible under the old empty-ground-truth scoring.

### 13. SQL fallback builder: null `limit` and ambiguous `GROUP BY *`
**Root cause A:** `{"limit": null}` in the Planner's JSON output becomes Python `None`; `parameters.get("limit", 100)` only applies its default when the key is *absent*, so an explicit `None` rendered as the literal (invalid) SQL text `LIMIT None`.
**Root cause B:** A `group_by` with no `select_columns` had no earlier check, so a step whose parameters got stripped down by the retry-simplification logic could silently degrade into `SELECT * FROM data GROUP BY x` — valid but meaningless SQLite (one arbitrary row per group, not an aggregate). This is exactly how ground-truth backfill caught D01: the report confidently cited *per-transaction* revenue values as if they were the region-level `SUM(revenue)` totals the plan asked for.
**Fix:** [src/tools/sql_tool.py](../src/tools/sql_tool.py) treats an explicit `None` limit the same as a missing one, and raises if `group_by` is given without `select_columns`.
**Verified:** D01 now fails loudly at the tool level with a message naming exactly what's missing, and the report honestly states no data was available — instead of confidently citing wrong numbers.

### 14. Insight/Viz Agents used a hardcoded non-zero temperature
**Root cause:** Every other agent either hardcodes `temperature=0.0` or reads `settings.llm_temperature` (which defaults to 0.0); `insight_agent.py` and `viz_agent.py` hardcoded `0.3` and `0.2` respectively, bypassing the setting entirely. Replaying an identical prompt (real, non-empty, 100-row analysis result) against the Insight Agent's LLM reproduced both a correct 3-insight response and, on other calls, an empty `[]` — from the exact same input.
**Fix:** Both agents now build their LLM with `temperature=settings.llm_temperature` like the rest of the pipeline.
**Verified:** The same replay prompt at `temperature=0.0` produced consistent, grounded insights across 3 repeated calls (previously flaky at 0.3).

### 15. `execute()` didn't respect `depends_on` when passing dataframes between steps
**Root cause:** [src/agents/analysis_agent.py](../src/agents/analysis_agent.py) `execute()` kept one shared `df` variable, unconditionally reassigned after *every* step regardless of that step's declared `depends_on`. An independent step (`depends_on=[]`) — e.g. planning "revenue by region" and "units by product" as two unrelated facets of an "overview" query — would silently receive whatever column-reduced dataframe the *previous* step in plan order happened to leave behind, rather than the original dataset, and fail with a spurious "column not found."
**Fix:** Track each step's output dataframe by step number (`result_dfs: dict[int, pd.DataFrame]`). A new `_select_input_dataframe()` gives independent steps the original full dataset and dependent steps the most recent declared dependency's output — actually implementing what the old code's comment ("rolling update for dependent steps") claimed it already did.
**Verified:** A new regression test (`test_execute_independent_steps_each_get_original_dataframe`) constructs exactly this shape and fails without the fix. Live: E01 went from succeeding roughly 1 run in 4 to 4/4 clean runs with real insights.

### 16. Fixing #13 introduced its own false positive: `ORDER BY` on a `SELECT`-defined alias
**Root cause:** The column-existence check added for #13 validated `order_by` as a bare identifier against the dataframe's real columns — but `ORDER BY total_revenue` after `SELECT ..., SUM(revenue) AS total_revenue` is valid SQL referencing an alias defined earlier in the same query, not an original column. The validator rejected it as "not found."
**Fix:** `_select_aliases()` extracts `AS <name>` aliases from `select_columns` and excludes them from the missing-column check.
**Verified:** A regression test confirms `ORDER BY` on a `select_columns` alias no longer raises; this was the actual step-1 failure blocking one of the E01 runs in #15's verification.

### 17. Eval judge reused the agent's own (weaker) model, and a single sample
**Root cause:** `_build_eval_llm()` ([src/eval/metrics.py](../src/eval/metrics.py)) used `settings.llm_model`/`settings.llm_provider` — the exact same model that generates the reports it's judging (self-preference risk), with one judge call per check. DEV_SPEC's own env-var spec originally called for a separate `EVAL_MODEL`, but it was never wired up. A single sample meant the same case could score `relevance=0.5` one run and `1.0` the next on an identical report.
**Fix:** Added `settings.resolved_eval_provider`/`resolved_eval_model` ([src/config/settings.py](../src/config/settings.py)) — prefers a provider *different* from the agent's own if a real (non-placeholder) API key exists for it, else falls back to a stronger model on the same provider (`gpt-4o` vs. the agents' `gpt-4o-mini`). `score_relevance_and_groundedness()` now makes `EVAL_JUDGE_SAMPLES` (default 3) independent calls in parallel and aggregates by median; when the spread across samples is ≥0.3 it's appended to the reasoning text so disagreement is visible in the report, not silently averaged away.
**Verified:** `tests/integration/test_eval_judge_calibration.py` makes real judge calls against three hand-written reports (grounded/on-topic, fabricated numbers, off-topic) — the upgraded judge correctly scores grounded high, fabricated low on groundedness, off-topic low on relevance, and ranks grounded strictly above fabricated. A live 20-case run flagged judge disagreement (spread ≥0.3) on 6/20 cases — variance that was previously invisible under single-sample scoring. Overall aggregate moved 0.76→0.71; this reflects the stronger judge being more critical and multi-sample scoring surfacing real disagreement, not a system regression (no code outside `metrics.py`/`settings.py` changed between these two runs).

### 18. Audited `statistical_tool`/`anomaly_tool`/`comparison_tool` for the pattern fixed earlier in `pandas_tool`/`sql_tool`
Five instances of the same family of bug, all in [src/tools/stats_tool.py](../src/tools/stats_tool.py):
- **`statistical_tool`** defaulted `test` to `"correlation"` whenever the key was omitted — exactly how the original DG04 bug happened (the Planner sent `metric`/`grouping_variable`, neither recognized, and got a silent correlation-across-everything instead of an error). Now infers the test from whichever recognized keys are present (mirroring `pandas_tool`'s `_infer_operation`) and raises if none match.
- **`compute_correlation`** silently dropped any requested column that didn't exist (`[c for c in columns if c in num_df.columns]`) — the exact pattern the roadmap called out. Now raises listing the missing columns, while still defaulting to "all numeric columns" when `columns` isn't given at all (that behavior is legitimate and has its own test).
- **`run_ttest` and `analyze_time_series`** return `{"error": ...}` instead of raising for "too few groups"/"too few data points" — a reasonable contract for direct callers (and their existing unit tests rely on it), but `statistical_tool`/`timeseries_tool` never checked for it, so `_execute_step` defaulted `"failed"` to `False` and reported these as successful steps. Both dispatchers now check `result.get("error")` and raise.
- **`anomaly_tool`** let a missing/misspelled `column` reach pandas as a bare `KeyError` with no indication of what to fix. Now validates the required parameter and its existence explicitly, matching every other tool's error format.
- **`compare_segments`**: `top_segment` was hardcoded to `grouped["mean"].idxmax()` regardless of the requested `agg` — asking "which channel has the highest total spend" (`agg="sum"`) would silently rank by average spend instead, since the `agg` parameter was validated into an `agg_fn` variable that was then never used. Now ranks by whichever aggregation was actually requested, and rejects unrecognized `agg` values instead of silently falling back to `"mean"`.

**Verified:** 9 new regression tests, one per bug (`tests/unit/test_phase5.py`), including a case (`test_compare_segments_top_segment_respects_agg`) with data specifically constructed so mean-ranking and sum-ranking disagree. A live 20-case run stayed flat (0.71→0.73, 4→3 safe refusals) — expected, since these fixes turn wrong-but-silent behavior into loud failures rather than turning failures into successes.

### 19. Eval ran the LLM judge twice per harness case
**Root cause:** `run_eval_node` scores every pipeline run inside the graph itself (CLAUDE.md's "eval runs on every execution" rule), and `scripts/run_eval.py` scores the same completed state again afterward with the golden `test_case` attached for ground-truth comparison. `answer_relevance`/`groundedness` don't depend on `test_case` at all — [src/eval/metrics.py](../src/eval/metrics.py)'s `score_relevance_and_groundedness()` signature never takes one — so the second pass was calling the exact same judge with the exact same inputs a second time. After #17 added 3-sample median scoring, that meant 6 judge calls per case through the harness instead of 3, for zero additional signal.
**Fix:** `EvalRunner.score()` ([src/eval/runner.py](../src/eval/runner.py)) now checks whether `state["eval_scores"]` already has `answer_relevance`/`groundedness` (i.e. `run_eval_node` already ran) and reuses those `MetricScore`s directly instead of re-invoking the judge. `factual_accuracy`/`intent_accuracy` (which *do* need `test_case`, and are cheap — no LLM call) are still computed fresh.
**Verified:** `test_eval_runner_reuses_existing_relevance_groundedness_instead_of_rejudging` wires the mock judge to raise if called at all, and passes. Live: `run_eval.py --case D01` shows identical `answer_relevance=1.00 groundedness=1.00` on both the in-graph and harness scoring log lines — the harness pass is now free.

### 20. `tests/integration/` was empty; no CI
**Fix:** Added `.github/workflows/ci.yml` with two jobs: `unit-tests` (the 357 mocked unit tests, no secrets needed, runs on every push/PR) and `integration-tests` (the live judge-calibration tests from #17 plus a single-case eval smoke run through the real pipeline, gated on `secrets.OPENAI_API_KEY`). The integration job skips gracefully via `_looks_like_real_key()` if the secret isn't configured (e.g. PRs from forks never receive repo secrets). Added a CI status badge to the README. A follow-up fix made `scripts/run_eval.py` itself exit non-zero on a genuine `pipeline_error` (previously it always exited 0, so this CI job went green the first time it ran even though every case hit a 401 auth error before `OPENAI_API_KEY` was configured as a repo secret).

## Phase B — real data (see [roadmap.md](roadmap.md))

Moving from Tier 2 (precision/trustworthiness) to Tier 1 (the data layer). First three items, done together since #2 (pushdown) falls out of #1 (joins) for free once real SQL runs against the real database.

### 21. Date-part derivation (#4)
**Root cause:** The `derive` operation only supported binary arithmetic (`+-*/`). "Quarter"/"month" never exist as real columns in any demo dataset, so a diagnostic query like "why did revenue drop in Q3" had no way to get there other than raw date-range filtering — every attempt at this either invented a nonexistent "quarter" column (failing loudly, post earlier fixes) or worked around it clumsily.
**Fix:** `pandas_derive` ([src/tools/stats_tool.py](../src/tools/stats_tool.py)) now also accepts unary date-part extraction — `op` ∈ `{year, quarter, month, week, day, dayofweek}` — via pandas' `.dt` accessor, with `right` no longer required for these. `ANALYSIS_PLANNER_SYSTEM` teaches the derive-then-filter/group_by pattern, and explicitly states date-part columns are **integers** (quarter: 1–4), not strings like `"Q3"` — comparing a string against an integer column silently matches zero rows, which is exactly what happened on the first live test before this line was added.
**Verified:** 10 new unit tests. Live: DG01 now plans `derive(quarter) → filter(quarter==3) → groupby(quarter, sum)` and computes Q3 total revenue = **$1,178,006.80** across two regenerated runs — matching the ground-truth value computed independently from the raw CSV (`data/demo/sales_data.csv`, summed by hand via pandas).

### 22. Cross-table joins + compute pushdown (#1, #2)
**Root cause:** `_load_dataframe()`/`_column_manifest()` only ever exposed the single "active" table to the Planner — for a SQL source with multiple tables (`ecommerce_orders.db`), there was no way to tell the Planner other tables even existed, so any query needing data from more than one table (order value by product category, customer attributes crossed with order history) was structurally unanswerable. D02 and C03 were explicitly flagged as known data mismatches for exactly this reason.
**Fix:** `list_related_tables()` ([src/tools/data_connector.py](../src/tools/data_connector.py)) does a schema-only introspection (no rows loaded) of every table in a SQL source via SQLAlchemy's inspector. `_related_tables_manifest()` ([src/agents/analysis_agent.py](../src/agents/analysis_agent.py)) renders that as a `### Related Tables` prompt section — table names, columns, and the real connection string — only when the active source is SQL-backed; empty/absent for CSV/JSON/Excel sources, so single-table planning is unaffected. `ANALYSIS_PLANNER_SYSTEM` now teaches that a cross-table question needs a single `sql_query` step with a raw JOIN query and `"connection_string"` set to the real value, executed directly against the database — the join AND the aggregation both happen in SQLite, not by loading every row into pandas (this is where #2, pushdown, comes for free: it was never a separate mechanism to build, just a consequence of running real SQL against the real engine instead of routing everything through a single in-memory table).
**Verified:** 9 new unit tests (`list_related_tables` schema introspection + exclusion, `_related_tables_manifest` graceful degradation for non-SQL/missing/broken sources, `plan()` prompt inclusion/omission). Live: D02 ("average order value per product category") now plans `SELECT p.category, AVG(o.revenue) AS average_order_value FROM orders o JOIN products p ON o.product_id = p.product_id GROUP BY p.category` — a real join, executed directly against `ecommerce_orders.db` — stable across 3 runs (identical `Books=1543.26 ... Office Supplies=770.88`, matching independently-computed ground truth), aggregate score 0.91 (up from being permanently capped as a data mismatch). C03 ("new vs returning customers") also now executes a real `orders JOIN customers` — the join mechanism works — but there's no literal new/returning flag in the data, so the Planner substitutes the closest available dimension (`customer.segment`) instead; this answers a related but not identical question, so C03 stays flagged as a (now partial, not structural) data mismatch. `KNOWN_DATA_MISMATCH` in `scripts/run_eval.py` and D02's `ground_truth` were updated accordingly.

### 23. MCP transport was never protocol-tested against a real server; RAG-MCP-Server connected for the first time (Phase B #10)
**Root cause:** [src/mcp_client/client.py](../src/mcp_client/client.py) hand-rolled `httpx.AsyncClient` POSTs of raw JSON-RPC to `/mcp` with no session negotiation at all. Against a real, spec-compliant streamable-http server (rag-framework's FastMCP-based RAG-MCP-Server) this failed immediately: `406 Not Acceptable` (missing the required `Accept: application/json, text/event-stream` header), then `400 Missing session ID` once the header was fixed — MCP's streamable-http transport requires an `initialize` handshake that negotiates a session ID, carried on every subsequent request. This is almost certainly why Phase 10 ("MAEDA MCP Server") and every fallback-mode baseline before this one never caught it: `SubSystemWithFallback` treats any transport error as "sub-system unavailable" and silently degrades, so a fundamentally broken client looked identical to "the servers just aren't running."
**Fix:** Rewrote `MCPClient` to use the official `mcp` SDK (`ClientSession` + `streamable_http_client`) instead of hand-rolled HTTP — opens a fresh session (handshake included) per call, matching MAEDA's per-node `asyncio.run()` architecture (no event loop to hold a persistent session open across). Separately, `retrieve`/`retrieve_with_metadata` in [src/mcp_client/rag_server.py](../src/mcp_client/rag_server.py) were sending flat `{"query":..., "top_k":...}` arguments, but rag-framework's actual tool signature (`def retrieve_with_metadata(input: RetrieveInput)`) takes one Pydantic-model parameter named `input` — FastMCP maps tool-call arguments onto Python parameter names, so this failed with a Pydantic "Field required: input" error until arguments were wrapped as `{"input": {"query":..., "top_k":...}}`. `list_collections` needed no such change (its signature takes a flat `db_path: str`).
**Verified:** Direct `curl` diagnostics confirmed each failure mode before and after the fix. `tests/unit/test_phase3.py`'s `TestMCPClient` (7 tests) rewritten to mock at the new SDK boundary (`_FakeSession` + patched `streamable_http_client`/`ClientSession`), all passing. Live end-to-end: started a real RAG-MCP-Server (`rag-framework`, port 8002) with the Wake Apparel business guide ingested (BM25 index, 9 chunks) alongside 9 pre-existing unrelated documents (resumes, financial PDFs) in the same unscoped knowledge base — no collection-level isolation exists in rag-framework today. Ran the full 20-case golden suite (`baseline_b_rag_online.json`) with Data Cleaner intentionally left in fallback (not yet connected — separate decision) and RAG genuinely online; every case's `mcp_call_log.mcp_modes` includes `'mcp'`, confirming real protocol calls rather than silent fallback.

**Result — not a clean win:** Aggregate moved 0.71 → 0.72 (flat). Individual cases swung up to ±0.23 in both directions, and inspecting the judge's `groundedness` reasoning text separates two different effects:
- **Real, attributable harm on 4 cases** (E01 -0.18, C01 -0.14, P02 -0.12, D03 -0.11): the judge reasoning explicitly names irrelevant retrieved content as the problem — e.g. D03 ("how many customers per country") scored down because the report "includes information about **investment behavior** that is not relevant to the user's question," a near-certain leak from one of the 9 unrelated documents sharing the unscoped knowledge base.
- **Unattributable gains on 7 cases** (E04 +0.23, DG02 +0.18, C03/C02/DG01 +0.10, P01 +0.09, D04 +0.07): none of these cases' groundedness reasoning mentions RAG or Wake Apparel content as a factor. Given `llm_temperature=0.0` throughout the report-generation pipeline, this is more likely ordinary judge-side sampling variance (the eval judge itself is still a non-zero-temperature LLM call, now median-of-3 per #17 but not eliminated) than a genuine RAG-grounding benefit.

**Conclusion:** the MCP integration bug is real and now fixed — this is the first time either sub-system has been protocol-tested end-to-end. But this baseline does not show that connecting RAG improved report quality; it shows concrete evidence of noise contamination on queries unrelated to the one ingested document, with no comparably concrete evidence of benefit. The fix to make going forward is scoping retrieval (collection filtering) — currently blocked upstream in rag-framework, which doesn't thread `RetrieveInput.collection` through to the query at all (confirmed by reading `rag/app/mcp_server/server.py`).

## Known limitations (not yet fixed)

- **`factual_accuracy` is still a brittle exact-match proxy.** Large numbers formatted with thousands separators (e.g. an LLM writing `$1,363,760.55`) won't match the raw ground-truth value `1363760.55` via the current regex-based extraction — several of the `0%` scores are this, not a real analysis error. Needs a tolerant numeric-match (strip separators, allow rounding) rather than exact string equality.
- **Data Cleaner is still fallback-mode by choice** (not yet mature enough to connect — a deliberate decision, not a bug). RAG-MCP-Server is now connected and verified live (see #23), but its measured effect on report quality was a wash, not a clear improvement — see #23's conclusion on retrieval scoping.
- **The guardrail's own LLM-as-judge is untouched.** #17 only fixed the eval harness's judge (`src/eval/metrics.py`); `guardrail_agent.py`'s live pass/fail judge still uses the agent's own model with a single sample. Deliberately out of scope here — changing it affects every live pipeline run's latency/cost, not just eval runs, and the guardrail already has its own retry loop as a partial mitigation.
- **3 golden cases remain data mismatches** (DG04, C03, P03) — customer LTV and plan-upgrade history don't exist anywhere in the demo data, and C03's "new vs returning" flag doesn't either (segment is the closest proxy). These will never score perfectly regardless of code quality.
- **Cross-table joins only work for genuinely SQL-backed sources.** CSV/JSON/Excel sources have no "other tables" concept — this is specific to `ecommerce_orders.db`-style multi-table databases (roadmap #3, real Postgres/warehouse connectors, is still open).
- **The `_select_input_dataframe` fix only handles single-parent chaining** for `pandas_transform` steps. A step depending on multiple prior `pandas_transform` steps (`depends_on=[1, 2]`) gets the *last* listed dependency's dataframe, not a merge of both — the real multi-table join capability (#22 above) is a separate mechanism (a single `sql_query` step), not a fix to this chaining behavior.

All reports referenced above are archived in `logs/eval_runs/`. 377 unit tests (plus 4 live judge-calibration tests in `tests/integration/`) passed throughout this entire sequence of changes.
