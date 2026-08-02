# MAEDA Roadmap — From Demo to Production

This is an honest list of what separates the current state of MAEDA (a working
multi-agent pipeline, eval-first debugged — see [eval_report.md](eval_report.md))
from something a real team could depend on. "Production-ready" only means
something once you know *who* uses it and *for what* — the items below are
split into things that are true regardless of use case, and things that only
matter once a specific scenario is picked.

## Tier 1 — Data layer (the biggest gap between demo and real use)

This is the hardest ceiling today. Nothing else matters much until this moves.

1. ✅ **Done — cross-table joins.** See eval_report.md #22. The Planner now
   sees a `### Related Tables` section (real table names, columns, and
   connection string) for SQL-backed sources and can write a single
   `sql_query` step with a raw JOIN executed directly against the database.
   D02 ("average order value per product category") went from a structural
   data mismatch to a real, correctly-joined, stable answer
   (aggregate=0.91). C03 ("new vs returning customers") now executes a real
   join too, but the data has no literal new/returning flag — the Planner
   substitutes the closest available dimension (segment), so it stays a
   partial (not structural) mismatch. Scoped to SQL sources only — CSV/JSON/
   Excel have no "other tables" concept.
2. ✅ **Done (for SQL sources) — compute pushdown.** See eval_report.md #22.
   Fell out of #1 for free: once the Planner writes real SQL against the
   real database via `connection_string`, the JOIN and the aggregation both
   execute inside SQLite, not by loading every row into pandas first. Not a
   separate mechanism — doesn't apply to flat CSV/JSON/Excel sources, which
   have no external engine to push down to (see #3, still open, for
   Postgres/warehouse-scale pushdown).
3. **Real data source connectors.** Postgres is nominally P1 in DEV_SPEC but
   was never actually exercised. Needs Snowflake/BigQuery support, incremental
   pulls, connection pooling, and schema caching.
4. ✅ **Done — date-part derivation.** See eval_report.md #21. `derive` now
   supports unary date-part extraction (year/quarter/month/week/day/
   dayofweek) via pandas' `.dt` accessor, not just binary arithmetic. DG01
   ("why did revenue drop in Q3") now reliably derives quarter, filters,
   and aggregates to the correct, ground-truth-matching total.
5. **A semantic layer over schema.** Real production columns look like
   `cust_seg_cd`, not `customer_segment`. Without a mapping from column names
   to business meaning and metric definitions, Intent Parser and Planner
   accuracy will drop sharply on real data. This is exactly where MAEDA and
   the separate knowledge-lifecycle project (business-glossary knowledge base)
   would intersect.

## Tier 2 — Precision & trustworthiness (extends "Known limitations" in eval_report.md)

These are the loose ends this session's debugging pass explicitly left open,
and the natural continuation of the eval-first narrative.

6. ✅ **Done — `safe_refusal` classification.** See eval_report.md #11.
   `error_rate` no longer scores a correctly-blocked fabricated report the
   same as a pipeline crash; a separate informational `safe_refusal` metric
   (excluded from the weighted aggregate) tracks refusal rate on its own.
7. ✅ **Done — backfilled `ground_truth`.** See eval_report.md #12. All
   answerable golden cases now have real values computed from
   `data/demo/*`; the data-mismatch/predictive ones carry an explicit
   `"_note"` instead (originally 4 — D02's cross-table join fix in #22
   later made it answerable, leaving 3: DG04, C03, P03). This immediately
   caught two live bugs (eval_report.md #13, #14) that empty ground truth
   had been hiding.
   Remaining gap (now closed, see eval_report.md #25): the exact-string-match
   scoring used to be brittle against thousands-separator formatting and
   LLM rounding — replaced with tolerant numeric comparison.
8. ✅ **Done (eval judge only) — LLM-judge reliability.** See
   eval_report.md #17. The eval harness's judge now auto-resolves to a
   provider/model independent from the agent being evaluated (prefers a
   different provider if a real key exists, else a stronger same-provider
   model), scores via 3-sample median instead of one call, and flags
   high inter-sample disagreement in the reasoning text. Backed by a live
   calibration test (`tests/integration/test_eval_judge_calibration.py`)
   that checks the judge actually ranks grounded > fabricated reports.
   Explicitly NOT done: the guardrail's own live pass/fail judge
   (`guardrail_agent.py`) still uses the agent's model with a single
   sample — deliberately deferred since it affects every live run's
   cost/latency, not just eval runs.
9. ✅ **Done — audited the remaining tools for the same silent-default
   pattern.** See eval_report.md #18. Found and fixed 5 instances across
   `statistical_tool`, `anomaly_tool`, and `comparison_tool`: a silently
   defaulted `test` type, `compute_correlation` dropping nonexistent
   columns, two "soft error dict" results getting reported as successful
   steps, an unhelpful `anomaly_tool` error on a bad column, and
   `compare_segments` ranking `top_segment` by the wrong aggregation.
10. ✅ **Done (partially, by choice) — real "baseline B/C" with RAG-MCP-Server
    online.** See eval_report.md #23–24. Found and fixed a fundamental MCP
    transport bug — `MCPClient` had never actually been protocol-tested
    against a real server (406 → 400 errors); rewritten on the official
    `mcp` SDK. Data Cleaner deliberately stays offline (not mature enough
    yet — the user's call, not a bug). #23 found concrete noise-contamination
    harm from an unscoped shared knowledge base; #24 fixed collection
    isolation upstream in rag-framework (ingest never tagged chunks with
    `collection`, so the filter that already existed on the retrieval side
    had nothing to match) and added `settings.rag_collection` on MAEDA's
    side to request it. Verified via reasoning-trace evidence that the
    contamination is gone — but aggregate score still doesn't move at
    MAEDA's current judge-noise level, since only 2/20 cases were affected.
    Answers the original question: sub-system coordination bugs (transport,
    then collection isolation) were both real and are now fixed; MAEDA's own
    orchestration code was never the bottleneck here.
11. ✅ **Done — close the error→retry loop with the actual error message.**
    See eval_report.md #28. New `_repair_step()` sends the tool's real
    error message + authoritative column list to the LLM for a targeted
    parameter fix, falling back to the old `_simplify_step` (kept as a
    safety net) when the model isn't confident. Live: 10/10 tool failures
    recovered in one 20-case run — column renames, a missing required
    parameter, a wrong SQL table name — none of which blind parameter-
    stripping could ever have fixed. Also fixed a latent `_parse_json`
    bug surfaced by reusing it for the repair response (an object whose
    only array is nested got truncated to just that array).
12. ✅ **Done — harden the "single sample row → population claim" defense.**
    See eval_report.md #29. New guardrail check `_check_population_claim_grounding`:
    population-generalizing language in the report escalates to critical
    unless at least one analysis step actually aggregated over the data
    (reuses the same evidence classifier that produces the prompt-level
    [AGGREGATE]/[ROW-LEVEL SAMPLE] tags). Verified correct at the unit
    level (6 tests); live on the 20-case suite it fired with zero false
    positives but also zero observed genuine catches — reported honestly
    rather than overclaimed, plausibly because #15/#28 already reduce how
    often a report is left with only row-level evidence in the first place.
25. **Planner sometimes runs a comparison/significance test on
    already-aggregated data.** Found via the user's own judge-calibration
    labeling session (eval v2 Step 3, see
    [judge_calibration.md](judge_calibration.md), case C01) — this is a
    genuine Planner defect, not just the nan-significance display bug
    fixed alongside it. C01's plan: step 2 groupby-aggregates revenue down
    to one row per quarter, then step 3 runs `comparison` (ANOVA) grouped
    by quarter *again*, over data that's already one-row-per-group — every
    group has count=1, so mean/median/sum all collapse to the same single
    value and std/F-statistic/p-value are all structurally undefined
    (`nan`), regardless of what the underlying data actually looks like.
    The fix already shipped (`_significance_flag()` in
    `src/tools/stats_tool.py`, see `judge_calibration.md`) stops the
    system from *reporting* an undefined test as a real null result, but
    doesn't stop the Planner from writing this kind of step sequence in
    the first place — comparison/significance-test steps need to run
    against row-level data, not a prior step's aggregated output.
    Deliberately not fixed now: touching `analysis_agent.py`'s planning
    logic is generation-relevant and would invalidate the replay corpus
    the user is actively labeling against (`logs/replay_cache/corpus.json`).
    **Pick up after Step 4** (eval_v2_plan.md) once the golden suite has
    expanded past 20 cases and isn't a fixed target being actively labeled
    — likely fix direction: either have the Planner check `depends_on` for
    an already-aggregated intermediate before emitting a `comparison`
    step, or have `compare_segments` itself validate `count > 1` per group
    and refuse (loudly, not `nan`) rather than let scipy silently produce
    an undefined result.
26. **`GuardrailAgent._llm_judge`'s own hallucination check has the same
    truncation bug the eval judge had — never fixed, because it's a
    different code path.** Found via the user's own labeling, case C02
    (see [judge_calibration.md](judge_calibration.md)): the report's
    Automated Caveats section flatly asserted "the provided data does not
    include any information about the Search channel," which is false —
    Search's numbers (conversion rate 6.49%, the actual top segment) are
    in `analysis_results`, just past where the guardrail's own truncation
    cuts them off. Confirmed by direct reproduction:
    [src/agents/guardrail_agent.py:214-223](../src/agents/guardrail_agent.py#L214)
    joins `result_summary` across all steps and caps it at **800 chars**
    (for C02, `"comparison: top=Search"` — literally the one substring
    that would have told the guardrail Search's data existed — sits at
    character 950, past the cut) and separately caps the report at
    **1500 chars**. This is a sibling of the bug fixed in `src/eval/
    metrics.py`'s `_build_judge_prompt` (`render_findings`/no-truncation,
    see `judge_calibration.md`'s bug #1) — same shape, same root cause
    (an LLM-as-judge call built from a truncated view of the same
    findings/report everything else now sees in full), but a completely
    separate function in a separate file that the earlier fix never
    touched. Practical consequence: the guardrail can (and, in this
    observed case, did) fabricate a hallucination finding — accusing a
    report of citing data that doesn't exist, when the data exists but
    the guardrail itself couldn't see it — which is arguably worse than
    missing a real hallucination, since it teaches the pipeline to
    distrust correct output. **Not fixed now**, same reason as #25:
    `guardrail_agent.py` is generation-relevant, changing it invalidates
    the corpus mid-labeling. **Pick up alongside #25, after Step 4** —
    likely fix: reuse `render_findings`/`render_rag_context` from
    `src/eval/metrics.py` here too (untruncated, single source of truth
    for "what does an LLM judge of this pipeline's output get to see"),
    rather than `_llm_judge` maintaining its own truncated,
    independently-drifting context-builder.

## Tier 3 — Engineering robustness

13. ✅ **Done — clean up the async architecture.** See eval_report.md #30.
    All 9 I/O-bound graph nodes converted from sync wrappers around
    individual `asyncio.run()` calls to real `async def` functions sharing
    one event loop; every call site (`src/mcp_server/server.py`,
    `scripts/run_eval.py`, `scripts/demo_scenarios.py`, `ui/app.py`)
    switched from `graph.invoke()` to `graph.ainvoke()`. Confirmed live:
    zero `"Event loop is closed"` occurrences in a full 20-case run
    (previously present). This was also the real prerequisite for #14
    (streaming) — `.astream()` needs a genuine async execution path, not
    a graph where each node opens and closes its own loop.
14. ✅ **Done — streaming output.** See eval_report.md #31. Replaced the
    chat handler's fake progress (a background thread + a fixed 7-phase
    timer with no relationship to which node was actually running) with
    `src/graph/streaming.py`'s `astream_pipeline()`/`run_pipeline_streaming()`,
    driving `graph.astream(state, stream_mode="updates")` directly and
    invoking a callback as each node genuinely completes. Verified live in
    a real browser session: the status line advanced through real node
    names as they finished, not a canned animation. Kept free of any
    Streamlit dependency so the logic is independently unit-tested (10
    tests) rather than only checkable by eye.
15. ✅ **Done — stop double-running eval.** See eval_report.md #19.
    `EvalRunner.score()` now reuses `run_eval_node`'s in-graph
    answer_relevance/groundedness scores instead of re-invoking the judge,
    since neither depends on `test_case`. Only the cheap, test_case-aware
    metrics (factual_accuracy, intent_accuracy) get recomputed.
16. ✅ **Done — integration tests + CI.** See eval_report.md #20.
    `.github/workflows/ci.yml`: a free `unit-tests` job (385 mocked tests,
    no secrets) and an `integration-tests` job (live judge calibration +
    a 1-case eval smoke run, gated on `secrets.OPENAI_API_KEY`, skips
    gracefully without it). README has a CI status badge.
17. ✅ **Done — multi-turn conversation memory.** See eval_report.md #32.
    `conversation_history` (a state field that already existed since Phase
    1 but was never read back) now actually resolves follow-ups: the
    Intent Parser sees the last few turns and carries forward whichever
    of query_type/target_metrics/dimensions/filters the current query
    doesn't override, while everything downstream stays unaware multi-turn
    exists at all. Verified live with a real two-turn browser session —
    "Now break that down by quarter too" (a query naming neither "revenue"
    nor "region") correctly resolved to `dimensions=['region', 'quarter']`
    at full confidence. `src/memory/` stays an empty stub deliberately —
    a capped list threaded through existing state was sufficient; no
    separate memory abstraction was needed for what this required. Not
    done: no multi-turn golden eval suite yet (scripts/run_eval.py still
    runs all 20 cases independently) — a real follow-up, not silently
    skipped.
18. ✅ **Done — concurrency and session isolation.** See eval_report.md #35.
    Agents are module-level singletons, and each held a persistent
    `self._cost_tracker` that only ever accumulated (`+=`, never reset) —
    confirmed live that a second query in the same process showed
    query1+query2's token_usage combined, and under genuinely concurrent
    `asyncio.gather()` requests this is real cross-session leakage, not
    just drift. `decision_trace` was already correctly scoped (stateless
    per call). Fixed by rehydrating a tracker from `state["token_usage"]`
    itself (unique per request) instead of accumulating on `self`;
    `self._cost_tracker` removed entirely. Verified live under both
    sequential and genuinely concurrent (`asyncio.gather`) requests — each
    now reports only its own cost.

## Tier 4 — Product layer (scenario-dependent — don't build ahead of a use case)

19. **Access control** — read-only DB accounts, table allowlists (this pairs
    naturally with the existing SQL-safety guardrail check).
20. ✅ **Done — persist `decision_trace`/`mcp_call_log`.** See
    eval_report.md #33. New `src/persistence/run_store.py` (SQLite,
    path configurable via `settings.runs_db_path`) wired in as a genuine
    graph node — `persist_run_node` is the terminal node before `END` on
    every path (both `run_eval` and `handle_error` route through it), so
    every invocation is audited automatically, success or failure,
    without any other node needing to know it exists. Verified live: a
    separate process queried the store after the originating run's
    process had exited and found the full trace. The eval harness's
    20-case runs now get persisted for free too, since it goes through
    the same graph.
21. **Cost quotas per user/day**, not just a single-session cap.
22. **Componentize the UI** (`ui/components/` is still empty) and get the
    trace viewer / eval dashboard to actually meet the DEV_SPEC acceptance
    criteria.
23. **Scheduling + push delivery** — e.g. a weekly ops report pushed to
    Slack/Feishu. If the target scenario is recurring reporting, this is the
    core delivery mechanism, not an add-on.
24. ✅ **Done — per-agent model tiering.** See eval_report.md #34.
    `AnalysisAgent`'s planner/step-repair and `GuardrailAgent`'s live judge
    now resolve to a stronger model (`resolved_planner_model`/
    `resolved_guardrail_model`, default `gpt-4o`) via new settings, while
    `IntentParser`/`VizAgent`/`InsightAgent` stay on the cheap default.
    Found and fixed two real bugs in passing while live-verifying this: a
    `cost_tracker.py` pricing bug that silently overcounted every
    gpt-4o-mini call at gpt-4o's ~33x rate (substring-match ordering), and a
    systemic `token_usage` overwrite bug where only the last agent to run
    in a given pipeline ever had its cost visible in the final state (two
    agents never wrote it at all). Live 20-case suite showed aggregate
    0.751→0.725 and safe_refusals 4/20→6/20 — cross-referenced against the
    prior baseline's per-case reasoning traces, this is the stronger judge
    correctly catching real fabrications the cheaper judge previously
    missed, not a quality regression. Cost adds ~$0.023/query (~12x on the
    two upgraded agents), still under 3 cents/query total.

## Suggested phasing

For a solo effort, roughly 2–4 weeks per phase:

**Phase A — Make "trustworthy" solid** (#6, #7, #8, #9, #15, #16): ✅ done.
All six eval/quality items completed — see eval_report.md #11–20. This was
the safety net for every later change. #10 rolls into Phase B below. #11
(retry-loop feedback) and #12 (population-claim guardrail) — the two
smaller Tier 2 items not bundled into a named phase — are also now done,
picked up opportunistically. See eval_report.md #28–29.

**Phase B — Make "real data" actually work** (#1, #2, #4, #10): ✅ done.
#1, #2, #4 (joins, pushdown, date-part derivation — see eval_report.md
#21–22). #10 (baseline B/C/D, see eval_report.md #23–25): RAG-MCP-Server
connected and verified genuinely live; Data Cleaner intentionally still
offline. Answer to this debugging pass's original question: two real
sub-system coordination bugs were found and fixed — the MCP transport
layer (#23) and collection isolation upstream in rag-framework (#24,
verified via reasoning-trace evidence, not just aggregate score) — plus
one eval-harness bug of its own (`factual_accuracy`'s exact-match
brittleness, #25, worth +0.03 aggregate points, confirmed twice
independently). None of these were MAEDA's own orchestration logic;
the harness itself needed as much debugging as the pipeline it measures.
As the interim alternative on the Data Cleaner side, the built-in pandas
fallback profiler was strengthened from one check to six (eval_report.md
#26) — which immediately surfaced 238 previously-unknown duplicate rows in
the primary demo dataset — and a dead-key bug was fixed that had kept every
profiler finding out of the report's quality caveat.

**Phase C — Make it pleasant to use** (#13, #14, #17): ✅ done — see
eval_report.md #30–32. #13 was the real prerequisite for #14 (streaming
needs a genuine async execution path, not one loop per node). #17 turned
out to be a two-agent, prompt-and-state-threading change rather than the
large new subsystem the "largest, most novel-feature-shaped item" framing
suggested going in — intent resolution absorbed all of the multi-turn
complexity, so nothing downstream needed to change.

**Product layer (#19–24) waits until a target scenario is picked** — recurring
ops reporting prioritizes #23; an analyst-copilot scenario prioritizes #17
and #14.
