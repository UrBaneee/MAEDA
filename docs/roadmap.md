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
7. ✅ **Done — backfilled `ground_truth`.** See eval_report.md #12. All 16
   answerable golden cases now have real values computed from
   `data/demo/*`; the 4 unanswerable ones (data mismatch / predictive)
   carry an explicit `"_note"` instead. This immediately caught two live
   bugs (eval_report.md #13, #14) that empty ground truth had been hiding.
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
11. **Close the error→retry loop with the actual error message.** Tool errors
    now include actionable detail (e.g. the exact list of available columns),
    but `_simplify_step`'s retry just strips parameters down rather than
    feeding that message back to the Planner for a corrected replan. This
    would meaningfully raise the recovery rate.
12. **Harden the "single sample row → population claim" defense.** The current
    fix is a prompt-level evidence tag ([AGGREGATE]/[ROW-LEVEL SAMPLE]) on the
    generation side. A stronger version adds a guardrail check: any
    population-level phrasing in the report must trace back to an aggregate
    step, or it gets escalated to critical.

## Tier 3 — Engineering robustness

13. **Clean up the async architecture.** Each graph node calls `asyncio.run()`
    independently, producing constant "event loop is closed" noise in the
    logs (visible throughout this session's runs). Should be a single
    end-to-end async execution path.
14. **Streaming output.** Queries take 22–137 seconds with zero feedback until
    completion — not acceptable for an interactive user. LangGraph's
    `.stream()` mode should push node progress and intermediate artifacts to
    the UI as they're produced.
15. ✅ **Done — stop double-running eval.** See eval_report.md #19.
    `EvalRunner.score()` now reuses `run_eval_node`'s in-graph
    answer_relevance/groundedness scores instead of re-invoking the judge,
    since neither depends on `test_case`. Only the cheap, test_case-aware
    metrics (factual_accuracy, intent_accuracy) get recomputed.
16. ✅ **Done — integration tests + CI.** See eval_report.md #20.
    `.github/workflows/ci.yml`: a free `unit-tests` job (357 mocked tests,
    no secrets) and an `integration-tests` job (live judge calibration +
    a 1-case eval smoke run, gated on `secrets.OPENAI_API_KEY`, skips
    gracefully without it). README has a CI status badge.
17. **Multi-turn conversation memory.** `src/memory/` is an empty
    `__init__.py`. Follow-up questions ("now break that down by region")
    aren't supported at all, but real analysis is inherently conversational.
18. **Concurrency and session isolation.** Agents are module-level singletons;
    two simultaneous Streamlit users will cross-contaminate state.

## Tier 4 — Product layer (scenario-dependent — don't build ahead of a use case)

19. **Access control** — read-only DB accounts, table allowlists (this pairs
    naturally with the existing SQL-safety guardrail check).
20. **Persist `decision_trace`/`mcp_call_log`** — currently discarded after
    each run, so nothing is auditable after the fact.
21. **Cost quotas per user/day**, not just a single-session cap.
22. **Componentize the UI** (`ui/components/` is still empty) and get the
    trace viewer / eval dashboard to actually meet the DEV_SPEC acceptance
    criteria.
23. **Scheduling + push delivery** — e.g. a weekly ops report pushed to
    Slack/Feishu. If the target scenario is recurring reporting, this is the
    core delivery mechanism, not an add-on.
24. **Per-agent model tiering** — everything runs on gpt-4o-mini today;
    planner/judge quality likely benefits from a stronger model while the
    step executor stays cheap.

## Suggested phasing

For a solo effort, roughly 2–4 weeks per phase:

**Phase A — Make "trustworthy" solid** (#6, #7, #8, #9, #15, #16): ✅ done.
All six eval/quality items completed — see eval_report.md #11–20. This was
the safety net for every later change. #10 rolls into Phase B below; #11
and #12 (retry-loop feedback, harder population-claim guardrail) are
smaller Tier 2 items not bundled into a named phase — pick up opportunistically.

**Phase B — Make "real data" actually work** (#1, #2, #4, #10): ✅ done.
#1, #2, #4 (joins, pushdown, date-part derivation — see eval_report.md
#21–22). #10 (baseline B, see eval_report.md #23): RAG-MCP-Server connected
and verified genuinely live; Data Cleaner intentionally still offline.
Answer to this debugging pass's original question: the transport layer
itself was a real bug (now fixed), but with it fixed, precision didn't
move — the remaining gap is retrieval scoping upstream in rag-framework,
not MAEDA's own orchestration.

**Phase C — Make it pleasant to use** (#13, #14, #17): streaming, multi-turn,
async cleanup.

**Product layer (#19–24) waits until a target scenario is picked** — recurring
ops reporting prioritizes #23; an analyst-copilot scenario prioritizes #17
and #14.
