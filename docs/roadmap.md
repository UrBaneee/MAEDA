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
25. ✅ **Done (2026-08-02) — Planner no longer runs a comparison/significance
    test on already-aggregated data.** Found via the user's own
    judge-calibration labeling session (eval v2 Step 3, see
    [judge_calibration.md](judge_calibration.md), case C01) — a genuine
    Planner defect, not just the nan-significance display bug fixed
    alongside it at the time. C01's plan: step 2 groupby-aggregates revenue
    down to one row per quarter, then step 3 runs `comparison` (ANOVA)
    grouped by quarter *again*, over data that's already one-row-per-group
    — every group has count=1, so std/F-statistic/p-value are all
    structurally undefined (`nan`) regardless of what the data looks like.
    Deliberately deferred past Step 4 (touching `analysis_agent.py`'s
    planning logic would have invalidated the replay corpus being actively
    labeled against); picked up once all 100 cases were labeled. Fix:
    `_select_input_dataframe` in `src/agents/analysis_agent.py` now detects
    this shape directly — a `comparison` step whose chosen dependency has
    already been reduced to ≤1 row per value of the step's own
    `segment_col` — and falls back to the original, un-aggregated dataset
    instead (only when `segment_col`/`value_col` exist there too, so a
    truly-derived column that only exists on the aggregating step's output
    is left alone rather than silently recomputed against the wrong data).
    Deterministic, not reliant on Planner prompt engineering. Verified live
    against the real pipeline: C01 regenerated post-fix now shows
    `significance_test: {"test": "one-way ANOVA", "statistic": 129.97,
    "p_value": 0.0, "significant": true}` with `count: 2927` per quarter
    (real per-transaction rows), not `count: 1`. New regression test:
    `test_comparison_step_falls_back_to_original_data_when_dependency_is_pre_aggregated`
    in `tests/unit/test_phase5.py`.
26. ✅ **Done (2026-08-02) — `GuardrailAgent._llm_judge`'s own truncation
    bug fixed, same day as #25.** Found via the user's own labeling, case
    C02 (see [judge_calibration.md](judge_calibration.md)): the report's
    Automated Caveats section flatly asserted "the provided data does not
    include any information about the Search channel," which was false —
    Search's numbers (conversion rate 6.49%, the actual top segment) were
    in `analysis_results`, just past where the guardrail's own truncation
    cut them off (`result_summary` joined across steps and capped at 800
    chars; report separately capped at 1500). Same shape, same root cause
    as the eval judge's bug #1 above, but a completely separate function
    in a separate file that fix never touched. Fix: `_llm_judge` now builds
    its context with `render_findings`/`render_rag_context`/
    `render_data_quality` from `src/eval/metrics.py` — the exact same
    functions the eval judge's own prompt uses — instead of its own inline,
    truncated context-builder, so the two views can never diverge again.
    Verified live: C02 regenerated post-fix now correctly discusses
    Search's real conversion rate (ANOVA statistic 994.85, p=0.0, matching
    the actual data) with `guardrail_passed: true` and zero fabricated
    caveats — the false accusation is gone. New regression test:
    `test_llm_judge_context_is_not_truncated` in `tests/unit/test_phase8.py`.
    Both fixes are generation-relevant, so the replay corpus (all 100
    cases) was regenerated afterward — see `docs/eval_v2_plan.md` for the
    before/after baseline this produced.
27. ✅ **Done (2026-08-02) — guardrail severity no longer waves through a
    failed Factual accuracy check.** Found while interpreting the #25/#26
    before/after numbers: `_parse_llm_checks` in
    `src/agents/guardrail_agent.py` only escalated a failed check to
    `"critical"` (block + retry) if its name contained one of
    `("pii", "safety", "hallucin", "fabricat", "claim_ground", "grounding")`
    — anything else, including a check the LLM itself named
    `"Factual accuracy"`, was `"warning"` (deliver with a caveat, never
    blocks), no matter how wrong the finding. `GUARDRAIL_SYSTEM`
    ([agent_prompts.py:348](../src/config/agent_prompts.py:348)) asks the
    LLM to check four things, and across all 100 post-#25/#26 cases it
    reliably named them exactly `Hallucination`, `PII leakage`,
    `Factual accuracy`, `Misleading framing` every time — the first two
    matched the keyword list and blocked; the second two structurally
    couldn't, regardless of severity. Confirmed live: 16 of 100 cases had a
    failed `Factual accuracy` and/or `Misleading framing` check delivered
    anyway (e.g. case P05: "the report says May 2023 had the highest churn
    rate at 13.74%, but March 2024's 39.07% is actually highest" — a real,
    specific factual error, waved through with only a caveat).
    **Fix**: added `"factual"`/`"accuracy"` to `_CRITICAL_CHECK_KEYWORDS`,
    so a failed Factual accuracy check now escalates to critical (block +
    retry) exactly like Hallucination. **Deliberately did NOT fold in
    `"Misleading framing"`**: spot-checked its own 3 standalone failures
    (DG13, D18, P17) and found a genuinely mixed bag — a real tone/urgency
    overstatement (P17: correctly non-critical), a comparison mix-up
    (DG13), and a possibly-mislabeled fabrication (D18) — not the same
    clean, consistent pattern Factual accuracy showed, so escalating it
    isn't well-evidenced yet. **Verification method, and why it differs
    from #25/#26**: this is a pure severity-classification fix (no
    generation-content dependency), so a mocked-LLM-response unit test
    (`test_llm_judge_factual_accuracy_failure_is_critical_not_warning` in
    `tests/unit/test_phase8.py`) is a *more* reliable proof than trying to
    reproduce the exact same hallucination via a live regeneration — tried
    regenerating P05 anyway as a spot-check, and (being stochastic) it
    happened not to reproduce the specific error this time, which is
    itself the reason a deterministic mock test is the right tool here,
    not a gap in verification. **Corpus regeneration deferred**: this
    change is generation-relevant (changes which reports get blocked vs.
    delivered), but a full 100-case regen is deliberately not done yet —
    more pipeline-layer work is queued (MCP connection, real agent tool
    use) that would invalidate it again; regenerate once, after all of
    that lands, before the eventual test-split reveal.
28. ✅ **Done (2026-08-02) — the false-superlative pattern flagged at P05 is
    real and recurring, not a one-off; fixed at both the generation and
    guardrail layers.** Follow-up investigation: wrote a script to grep all
    100 cached reports for a superlative claim citing a specific number
    ("X achieved the highest/lowest Y at/of Z") and cross-check the named
    entity against the matching row-level analysis result. 10 such claims
    were found; 6 were checkable against a matching data column, and **all
    6 named the wrong entity** — 4 distinct cases (C02, C09 ×2 phrasings,
    C14, DG09), all in the marketing-campaign dataset's "individual
    campaign spotlight" narrative pattern (e.g. C02: "Campaign CAM0001
    achieved the highest ROI of 2023.7" when CAM1939's roi=19741.6, ~10x
    higher, is the true max). The other 2 checkable claims (DG02, DG03)
    were genuinely correct and, as expected, were not flagged by anything
    built here.

    **Root cause, found by inspecting what the Insight Agent actually
    sees**: `_extract_result_detail` in `src/agents/insight_agent.py`
    showed row-level (list-typed) analysis results only as `result[:5]` —
    an *unsorted* 5-row sample, always the first 5 rows in file order
    (`CAM0001`–`CAM0005` in every affected case, since none of these steps
    sort their output). The LLM then reads whichever value is locally
    largest within that arbitrary 5-row peek and reports it as the
    dataset's global "highest" — confirmed directly: `CAM0001` is real
    data (`roi=2023.7` matches exactly) but ranks 236th of 429 same-channel
    campaigns by ROI, nowhere near the top. `INSIGHT_GENERATOR_SYSTEM`
    ([agent_prompts.py:290](../src/config/agent_prompts.py#L290)) already
    has an explicit rule that a [ROW-LEVEL SAMPLE]-tagged step "may [only]
    cite ... as an example ... never generalize ... into a claim" — this
    is the cheap generation-tier model (roadmap #24) violating an existing
    instruction, not an absent one.

    **Fix, generation layer**: new `_numeric_column_ranges()` in
    `src/agents/insight_agent.py` computes the true min/max (with which
    row achieves it) per numeric column across *all* rows of a row-level
    result, appended to the prompt alongside the existing 5-row sample.
    `INSIGHT_GENERATOR_SYSTEM`'s [ROW-LEVEL SAMPLE] rule extended to name
    this explicitly: the sample is never sorted and must not be used for a
    superlative claim, but the "True column min/max" line is a real
    computed aggregate and may be cited. **Verified live, and honestly
    incomplete**: regenerating C02 after this fix shows the SAME report
    correctly use the true max for `conversion_rate` (now says
    "~13.00%", the real value) in one paragraph, then *still* fabricate
    the identical wrong `CAM0001` ROI claim in another paragraph of the
    same report — giving the model correct data measurably helps but does
    not reliably stop a cheap model from ignoring it, so this alone isn't
    a fix, only a mitigation.

    **Fix, guardrail layer (the actual backstop)**: new deterministic
    check `_check_superlative_claim_grounding` in
    `src/agents/guardrail_agent.py`, registered alongside
    `_check_population_claim_grounding` (#12) — same coarse-but-precise
    design philosophy: regex-matches a superlative claim with an explicit
    number, resolves its metric to a column by keyword overlap against any
    row-level result available, and flags `critical` (block + retry) if
    the named entity is not that column's true extreme by more than 1%.
    Silently passes anything it can't confidently resolve (no matching
    column, entity not found, or a genuinely correct claim like DG02's
    "Basic plan has the highest churn rate", which really is true). Live
    on the same corpus: **4/4 known-wrong claims flagged critical, 0/100
    false positives** across the whole cached corpus (including the 2
    known-correct claims and the ~50 other cases with non-numeric
    superlative language this regex doesn't even attempt to check). Live
    end-to-end on C02's regenerated state: `GuardrailAgent.process()` now
    returns `overall_verdict="retry"`, `guardrail_passed=False`, with
    `retry_reason` naming the exact fabricated claim — where the LLM
    judge's own `Factual accuracy` check passed it through unflagged on
    that same run, this is the deterministic check doing precisely the
    backstop job it was built for.

    New tests: `test_numeric_column_ranges_*` and
    `test_extract_result_detail_*` in `tests/unit/test_phase7.py`;
    `test_superlative_claim_*` in `tests/unit/test_phase8.py`. Full suite:
    559 passed (was 548). **Corpus regeneration deferred**, same reasoning
    as #27: this changes `INSIGHT_GENERATOR_SYSTEM` (generation-relevant,
    invalidates the replay fingerprint for all 100 cases) but more
    pipeline-layer work is queued that would invalidate it again —
    regenerate once, after that lands, before the eventual test-split
    reveal. Only C02 was spot-check-regenerated live, as verification, the
    same pattern #27 used for P05.
29. **Every QWK/before-after number produced so far was measured with a
    judge that shares a provider with the pipeline it's judging — the
    self-preference safeguard exists in code but is currently disabled by
    environment, not design.** `settings.resolved_eval_provider`
    ([settings.py:108](../src/config/settings.py#L108)) is explicitly built
    to prefer a provider *different* from the agent's own (`llm_provider`)
    so the judge isn't scoring a sibling of itself — but it falls back to
    the agent's own provider if no real key exists for the other one, and
    in this environment `ANTHROPIC_API_KEY` isn't a real key
    (`_looks_like_real_key` returns `False`), so `resolved_eval_provider`
    has resolved to `openai` — same as the pipeline — for every measurement
    in this project: the original noise floor, all 5 judge-calibration
    labeling passes, A2's before/after, and #25/#26's before/after. None of
    those results are wrong because of this (the human-labeling work
    measured real agreement against a real judge, and the mechanism fixes
    were verified by reading the judge's actual reasoning, not just
    trusting a score) — but none of them can currently rule out
    self-preference as a contributing factor either, and this project
    specifically built the judge-tiering design to guard against exactly
    that. Previously only a scattered caveat across `noise_floor.md` and
    `judge_calibration.md`; formalized here as a tracked item. **Blocked
    on a real `ANTHROPIC_API_KEY`** — once available, cheapest next step is
    the judge-model comparison table already scaffolded in
    `judge_calibration.md` ("Reusing the same labels to settle the
    judge-model question"): re-score the existing 120 human-labeled cases
    with `claude-sonnet-5` as the judge and recompute QWK against the
    labels already collected — no new labeling round needed, turns "why
    gpt-4o" from an assumption into a measured comparison.

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
