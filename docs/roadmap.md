# MAEDA Roadmap — From Demo to Production

This is an honest list of what separates the current state of MAEDA (a working
multi-agent pipeline, eval-first debugged — see [eval_report.md](eval_report.md))
from something a real team could depend on. "Production-ready" only means
something once you know *who* uses it and *for what* — the items below are
split into things that are true regardless of use case, and things that only
matter once a specific scenario is picked.

## Tier 1 — Data layer (the biggest gap between demo and real use)

This is the hardest ceiling today. Nothing else matters much until this moves.

1. **Cross-table joins.** A query today only ever operates on one flat
   DataFrame. ~20% of the golden suite (queries needing customer↔order↔product
   joins) is structurally unanswerable. Real enterprise questions ("which
   customer segment has the highest order value?") almost always need a join.
   Requires exposing multi-table schema to the Planner and letting `sql_tool`
   run real multi-table queries instead of loading one table into pandas.
2. **Compute pushdown.** The whole table is currently read into pandas before
   any aggregation happens. Fine at demo scale (5k–12k rows); falls over on a
   real warehouse (millions of rows). Aggregation should push down to
   SQL/warehouse execution; pandas should only touch the already-aggregated
   result.
3. **Real data source connectors.** Postgres is nominally P1 in DEV_SPEC but
   was never actually exercised. Needs Snowflake/BigQuery support, incremental
   pulls, connection pooling, and schema caching.
4. **Date-part derivation.** Today's workaround is telling the Planner to use
   date-range filters instead of a "quarter" column. The `derive` operation
   should support extracting year/quarter/month directly — this is one of the
   most common derived-column needs in analysis.
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
   Remaining gap: the exact-string-match scoring is brittle against
   thousands-separator formatting — see eval_report.md's known limitations.
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
9. **Audit the remaining tools for the same silent-default pattern** fixed in
   `pandas_tool`/`sql_tool` this session — `statistical_tool` (silently drops
   requested columns that don't exist rather than erroring), `anomaly_tool`,
   and `comparison_tool` haven't been checked yet.
10. **A real "baseline B" with MCP sub-systems online.** Every baseline run so
    far has both Data Cleaner and RAG-MCP-Server offline. The original
    question this debugging pass set out to answer — how much of the
    precision problem comes from sub-system coordination — is still
    unanswered. The MCP server (Phase 10) has also never been exercised by a
    real client.
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
15. **Stop double-running eval.** `run_eval_node` (inside the graph) and the
    harness (`scripts/run_eval.py`) each make their own LLM-judge call today,
    doubling eval cost per case.
16. **Integration tests + CI.** `tests/integration/` is currently empty. Add a
    GitHub Actions workflow running the 331 unit tests plus one eval smoke
    case — a public repo can carry that as a badge.
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

**Phase A — Make "trustworthy" solid** (#6, #7, #8, #9, #15, #16): all eval/quality
work, cheap, no external dependencies, directly strengthens the interview
narrative, and is the safety net for every later change — without reliable
eval, touching the data layer next is just guessing.

**Phase B — Make "real data" actually work** (#1, #2, #4, #10): joins, pushdown,
baseline B. Only after this does MAEDA earn the right to say "usable" instead
of "demoable."

**Phase C — Make it pleasant to use** (#13, #14, #17): streaming, multi-turn,
async cleanup.

**Product layer (#19–24) waits until a target scenario is picked** — recurring
ops reporting prioritizes #23; an analyst-copilot scenario prioritizes #17
and #14.
