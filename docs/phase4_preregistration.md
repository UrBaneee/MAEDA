# 阶段 4 Pre-registration — **DRAFT, NOT FROZEN**

Status: **draft**. This file is the pre-registration artifact the
`ECOSYSTEM_INTEGRATION_PLAN.md` "冻结前必办清单" writes into. It is frozen only
when that checklist is empty; until then every section below is editable and
nothing here is a commitment.

Authority: `ECOSYSTEM_INTEGRATION_PLAN.md` is the source of truth for how each
statement here was decided. This file restates the decisions in the form an
outside reader of the 阶段 4 report needs them, and adds nothing that was not
adjudicated there. Where a section is still open it says so, names the item it
is waiting on, and **does not fill in a default**.

Every measurement in this file was re-taken on the date noted next to it
(附录 CX.5's rule: live numbers in historical appendices are not quotable as
current fact). Where a re-measurement disagrees with an earlier appendix, both
values are shown and the disagreement is called out.

---

## 0. What is being pre-registered

Two experiments, both defined in `ECOSYSTEM_INTEGRATION_PLAN.md` 阶段 4:

- **实验 1 — component net gain.** `force_on` vs `force_off`, per sub-system.
  For the cleaner arm the comparison is **"forced cleaning vs no cleaning"**,
  not "auto routing vs none" (附录 CC.2): routing accuracy was deliberately
  split out into 实验 2, because a single arm mixing the trigger and the action
  misattributes whichever one fails. `force_off` means **no call is issued**,
  not "call and discard" (附录 CC.3) — so quality metrics and latency metrics
  are **not from the same source** in the off arm and must not be presented in
  one net-gain table.
- **实验 2 — routing accuracy.** Requires the conditional-routing work landed
  in 附录 CM (`3b4f88a`).

**Open, must be closed before freeze:**

| Open item | Waiting on | Rule |
|---|---|---|
| 实验 1 RAG on-arm retrieval tier — (a) real embeddings / `hybrid`, or (b) declared `bm25_only` | **CM.7 — user decision.** lead recommends (a) (附录 CN.5), with 附录 CP.5's "report corpus-backed and corpus-free events separately" as an added condition | **No default may be written here.** `settings.maeda_rag_expected_retrieval_mode` defaults to `"hybrid"` (`src/config/settings.py:153`), and that default must not be allowed to become the answer by omission |
| Number of paired trials `k` at the reveal | σ_hetero measured on the dev split; k=5 is a candidate only (附录 CA/CC) | Candidate draft, not a pre-registered value |

---

## 1. 硬约束 1 — the one reveal

### 1.1 There are two holdouts, not one

| Holdout | Where | Size (measured 2026-08-22) | Driven from |
|---|---|---|---|
| MAEDA eval holdout | `tests/eval/test_suite.json`, entries with `split == "test"` | 41 of 100 cases | **MAEDA** |
| TB5 retrieval holdout | rag `eval/fixtures/tb5_retrieval/test.json`, derived from the frozen test-split `probe_queries` in `eval/fixtures/event_script/event_script.yaml` | 8 derived entries; the guard's digest set holds **10** frozen test-split probe queries | **rag** |

Conflating the two is easy and wrong: the rag split guard (附录 CZ) protects
the second one only. See §1.3.

### 1.2 How the reveal is executed — who, in which process, with which command

**(A) MAEDA holdout — driven from MAEDA.**

```
cd ~/MAEDA
MAEDA_INTENT_REFINE_TRIGGER=always \
MAEDA_CLEANER_MODE=<arm> MAEDA_RAG_MODE=<arm> \
poetry run python scripts/run_eval.py --split test --trials <k>
```

- `--split` defaults to `dev`; `test` must be typed (`scripts/run_eval.py:251`)
  and prints `*** Running against the HOLDOUT TEST SPLIT. ***` (`:330`).
- The rag MCP server is a **separate process** reached over streamable-http
  (`settings.rag_server_mcp_url`, default `http://localhost:8002`;
  `docker-compose.yml:26` sets `http://rag-server:8002`). MAEDA does not start
  it in-process and does not pass it any reveal declaration.
- `MAEDA_INTENT_REFINE_TRIGGER=always` is **required on the command line**, see
  §3.2. Nothing in `scripts/run_eval.py` sets it, and nothing will: as of
  冻结前必办 (15) / 附录 DD.3 the script **refuses** a `--split test` run whose
  parsed `settings.intent_refine_trigger` is not `"always"`, exiting 2 before
  the banner above, before the suite is loaded and before the graph is built —
  i.e. before this process can issue a billable call (`scripts/run_eval.py:294-329`).
  It refuses rather than setting the value so that the command recorded here
  stays the command that produced the data.

**(B) TB5 retrieval holdout — driven from rag.**

```
cd ~/rag-framework
poetry run python -m rag.cli.eval --split test
```

That path enters `rag.eval.split_guard.declare_test_split_reveal(...)`
in-process (附录 CZ.5), which is the recognised declaration. **No environment
variable is involved.**

**Neither run needs `RAG_TEST_SPLIT_REVEAL`, and it must be left unset.**
This answers 附录 CZ.8's question. The reason run (A) does not need it is not
that the guard is satisfied — it is that **the guard never sees a frozen probe
query on that path** (§1.3).

### 1.3 What the guard guarantees, and what it does not — read both halves

**It guarantees:** a frozen test-split probe query will not be *executed* as a
retrieval by a rag process that has not declared a reveal. The check is by
content digest, sits on the first executable line of `QueryPipeline.query()`,
fires before anything is written to the trace store or any index is touched,
and does not care how the caller obtained the string (附录 CZ.1/CZ.4, verified
by lead in 附录 DB.1).

**It does not guarantee that the test split has not been observed.** It
protects **10 frozen strings**, not the concept "test split". All of the
following are outside it *by design*, not by omission:

1. Writing one's own query aimed at a test event and comparing the hits with
   that event's `expected_sources` — a test-split observation the guard cannot
   see (附录 CZ.6(6), ruled non-closable in 附录 DB.2).
2. **Reading** test-split content — corpora, event metadata, `expected_sources`
   — which is not a reveal and must stay possible (附录 CR.5).
3. Rephrasing a probe query: digests tolerate whitespace/case/NFKC, not rewording.
4. Retrieval issued straight at an index object (`mgr.bm25.search()`), which
   bypasses the pipeline entirely — **and is the only bypass that leaves no
   trace row** (附录 DB.2(2)).
5. Rows already in the trace store from earlier runs; the guard cannot retract history.

**And the half that concerns MAEDA directly: the rag guard does not protect
MAEDA's holdout at all.** Measured 2026-08-22 against
`rag.eval.split_guard.is_test_probe_query`: of the 100 queries in
`tests/eval/test_suite.json`, **0** are recognised as test-split probe queries
(0 of the 41 test-split ones); the guard's digest set holds 10 strings, none of
them a suite query. MAEDA does not even send the user's query verbatim — the
text handed to rag is synthesised by
`InsightAgent.build_retrieval_query` (`src/agents/insight_agent.py:125`) from
intent plus analysis findings. What protects MAEDA's holdout is: `--split`
defaulting to `dev` and `test` having to be typed, the banner at
`scripts/run_eval.py:270`, `scripts/measure_noise.py full --split` defaulting
to `dev` (附录 CC.1/CD), and **discipline**.

*A reader who takes away "硬约束 1 is enforced by a guard" and stops there will
overestimate the protection by a wide margin. That is why this section is
stated positively rather than as "everything else rests on discipline".*

### 1.4 `RAG_TEST_SPLIT_REVEAL` — one-shot only, if it is ever needed at all

Per §1.2 the planned reveal needs no such variable. If a future execution plan
does route frozen probe queries through a rag server driven from MAEDA, then:

- the variable may be set **only on the process environment of that one run** —
  a command prefix, or a one-off `-e` on that single `docker compose run`;
- it must **not** be written into `.env`, into a `docker-compose.yml`
  `environment:` block, into `.env.example`, or into any file that is committed
  or persists after the run.

Reason (附录 DB.3): rag's `docker-compose.yml:15-16` and `:47-48` give both
services `env_file: - .env`, so one line in `.env` becomes a permission compose
re-applies on every start, and several rag entry points call
`load_dotenv(override=False)`. **硬约束 1 says "exactly once", not "the first
time".** A one-off authorisation that lands in a long-lived file stops being
one-off. rag prints a banner the first time the variable actually opens the
gate (附录 CZ.3) — that is detection, and it does not substitute for this rule.

### 1.5 Auditability of the unprotectable case

The bypass in §1.3(1) cannot be prevented, but it is **partly auditable after
the fact** (附录 DB.2(2)): every retrieval that really executes goes through
`QueryPipeline.query()`, which immediately writes the query text into the trace
store. Enumerating the trace store therefore answers "did anyone run a
self-written query against a test event before the reveal?" — with exactly one
hole, `mgr.bm25.search()` (§1.3(4)), which writes no trace.

**Enumeration performed 2026-08-22 (orchestrator).** Scope: every trace store
in the rag repository — `data/**/*.db` plus `rag/app/studio/**/*.db`, 14 files.

| Measure | Value |
|---|---|
| `runs` rows across all 14 stores | 22,387 |
| rows carrying query text | 16,516 |
| rows whose query is a frozen test-split probe query | **6** |
| distinct probe queries involved | **2** (of the 10 frozen digests) |
| when | all six at `2026-08-18 19:39:51`, one batch, `data/tb5.db` |

Pulling the whole `19:39:00–19:41:00` window gives 21 rows = 7 distinct queries
× 3 run types (`query` / `query_pre_rerank` / `query_complete`), of which
exactly 2 are frozen test-split probe queries. That reconciles line-for-line
with the disclosure in §1.6: 附录 CP.2 recorded "7 query records, timestamp
2026-08-18 19:40:07" — 7 distinct queries, timestamp taken from the later end of
the batch.

**Result: apart from the single occasion disclosed in §1.6, no test-split probe
query has ever been executed.** This changes the evidential status of that
disclosure: it was previously a self-report accepted by the lead, and is now a
statement checked against every trace row on disk.

**Two limits, which must be read together with the result above.**

1. `mgr.bm25.search()` (§1.3(4)) writes no trace, so a retrieval taken that way
   is invisible here.
2. **Pointing the trace store at a copy outside the repository removes the row
   from this audit.** That is not hypothetical — it is what 附录 CO did, on
   purpose, to honour a "zero writes to the repository" commitment: the trace
   DBs were copied to a scratch directory and `db_path` pointed at the copy.
   The improvement that removed the writes also removed this audit's ability to
   see them. That episode is nonetheless on the record, which is the point:
   **what makes it checkable is the disclosure, not the trace.**

Therefore this enumeration is a backstop over repository-local traces only. It
must not be read as "nothing was found, therefore nothing happened."

**Note for anyone re-running it.** The column is `metadata_json`, not
`metadata`. The first attempt at this enumeration used the latter, silently
matched no column in any of the 14 stores, and returned a clean-looking
all-zero table — a wrong answer whose direction happened to be reassuring. The
re-run asserts the column exists and fails loudly if it does not. Any automated
check of this kind must first demonstrate that it can see anything at all; a
zero it produces is not evidence until that is established.

> Text authored by the orchestrator in ECOSYSTEM_INTEGRATION_PLAN.md 附录 DE; transcribed verbatim.

### 1.6 Prior observations of the test split

**One prior observation of the test split has occurred. It is disclosed here in
full rather than assessed as harmless by the parties involved** (冻结前必办 (1),
附录 CO.0 / CP.2).

| | |
|---|---|
| When | 2026-08-18, trace rows timestamped 19:39:51–19:40:07 |
| What was observed | the 2 `probe_queries` of `EVT-2024-03-SOUTH-UNITS-SPIKE`, an event whose `split` is `test` |
| Retrieval tier | BM25-only |
| What was seen | the top-5 chunks and their scores |
| Who | rag-owner, during the 附录 CJ.3 BM25-only verification |
| How it happened | the probe script read `event_script.yaml`'s `probe_queries` directly; at that time the reveal guard was mounted only on `rag.cli.eval`, so this path did not pass through it (附录 CP.4) |

**It was self-reported.** No external check produced it: the rag-owner went
looking through SQLite timestamps and disclosed it unprompted, having already
narrowed its own scope, and stated explicitly that it should count against the
reveal-discipline record rather than be quietly cancelled by that narrowing.

**Adjudication (附录 CP.2): this does not consume 硬约束 1's single reveal.**
The reasoning is not "the damage was small" — assessing damage after the fact
is how a discipline breach gets retroactively approved. It is:

1. **No quantity comparable to ground truth was produced.** That event has no
   corpus by design and an empty `expected_sources`, so there is no recall, no
   hit rate, no pass/fail to compute. What was seen is which 5 chunks BM25
   returned for a query that has no correct answer.
2. **Neither conclusion drawn that round depends on the observed data.** Both
   are properties read out of source and true for any query: `min_results=1`
   never returns an empty set (`query_pipeline.py:358`), and `final_score` is
   identically `RRF(rank)` with no reranker (`rrf_fusion.py:76`). **A conclusion
   that is invariant under the observation received no information from it.**
3. **No parameter, threshold, prompt, or criterion was changed because of it.**
   Every commit in all three repositories after 附录 CJ was checked; none
   relates to that event.

**Independently verified 2026-08-22** by enumerating all 22,387 trace rows in
the repository (§1.5): the observation is exactly as disclosed, and there is no
other. Read §1.5's two limits alongside that.

**No claim of harmlessness is being made here.** The reader is given the time,
the object, the content, and the falsifiable assertion in point 3, and may draw
their own conclusion.

> Text authored by the orchestrator in ECOSYSTEM_INTEGRATION_PLAN.md 附录 DE; transcribed verbatim.

---

## 2. Known covariate: `glossary_coverage` (清单 (9))

### 2.1 Per-data-source values — re-measured 2026-08-22

Method: offline and deterministic, no LLM. `glossary.resolve_glossary(...)` was
called with the same arguments `src/graph/nodes.py::connect_schema` (`:641-645`)
passes it, with column lists read directly (CSV header / `PRAGMA table_info`)
so that no `DataConnector` schema-summary request is issued at all.

| Data source | cases (dev / test) | `glossary_coverage` | covered columns |
|---|---|---|---|
| `data/demo/sales_data.csv` | 17 (9 / 8) | `full` | 7/7 |
| `sqlite:///data/demo/ecommerce_orders.db` `table=orders` | 29 (19 / 10) | `full` | 7/7 |
| `data/demo/churn_data.csv` | 25 (14 / 11) | `full` | 8/8 |
| `data/demo/marketing_campaigns.csv` | 16 (6 / 10) | `full` | 9/9 |
| `sqlite:///…ecommerce_orders.db` `table=products` | 7 (6 / 1) | **`absent`** | 0/5 — `reason='no glossary entry for data source ecommerce_orders.db::products'` |
| `sqlite:///…ecommerce_orders.db` `table=customers` | 6 (5 / 1) | **`absent`** | 0/6 — `reason='no glossary entry for data source ecommerce_orders.db::customers'` |

Re-measurement agrees with 附录 CU.1 and 附录 CV.1 value for value, including
the `reason` strings.

The values are **correct, not a misconfiguration**: `config/glossary.yaml`
covers 4 data sources / 31 column entries, and `glossary.py::_select_block`
deliberately does not fall back across tables of the same `.db` file.

### 2.2 It is **not constant** inside the experiment

- Overall: **87 / 100 cases `full`, 13 / 100 `absent`** (products 7 + customers 6).
- On the MAEDA holdout specifically: **39 / 41 `full`, 2 / 41 `absent`**.
- On dev: 48 / 59 `full`, 11 / 59 `absent`.

Consequences that must be read together with the table:

1. This is a **known, declared, per-case covariate**, not a constant condition.
   The answer to 附录 CS.2 condition 1 ("state whether the value is constant
   within the experiment") is **no**.
2. It is **constant within a matched pair** — both arms of a paired comparison
   run the same case against the same data source — so it does not bias the
   paired difference. What it does do is make any *stratified* claim about the
   `absent` stratum unsupportable on the holdout: **n = 2**. No conclusion may
   be reported by coverage stratum from the test split.
3. 附录 CV.3 explicitly **rejected** the alternative of writing glossary entries
   for `products`/`customers` so that all six sources become `full`: the
   decision rule (附录 CT.3) was fixed before the measurement, and changing the
   data to change the result after seeing it is the thing pre-registration
   exists to prevent. Filling those two entries remains a legitimate proposal
   **after** 阶段 4.

### 2.3 The cleaner's planning prompt contains a fixed paragraph about this field

Verified 2026-08-22 by reading the cleaner repository (read-only; cleaner HEAD
`c3639dc`):

- Since cleaner commit **`349a68f`**, `agent/llm_planner.py` defines
  `_GLOSSARY_COVERAGE_GUIDANCE` (`:327-339`) and inserts it into the **user**
  prompt between `## Intent` and `## Global Constraints`.
- The insert is gated on **key presence, not value**:
  `_glossary_coverage_section()` (`:341-345`) returns the constant iff
  `"glossary_coverage" in intent`. The constant contains no interpolation of
  the value.
- MAEDA always sends the key: `src/graph/nodes.py:470` emits
  `"glossary_coverage": glossary_coverage or glossary.COVERAGE_ABSENT`.

**Therefore all 100 cases receive byte-identical guidance text**; the only
per-case difference is the value inside the intent JSON. The paragraph states
the field's meaning (availability of *written business definitions* — not data
quality, volume or trustworthiness), forbids using it to change *which* steps
are planned, forbids inventing a column's business meaning from its name when
coverage is `absent`, and explicitly defers to the cleaner's existing Critical
Rules 13/14 (whose name-based `skip_columns` heuristics are safety checks about
how a column may be *treated*, not claims about what it *means*).

Why this is disclosed rather than treated as an implementation detail: the
cleaner arm runs with `data_cleaner_planner_mode="llm"`
(`src/config/settings.py:69`, ruled mandatory in 附录 CT.3), so this field
reaches a live planning model. An unexplained key can be read as "context is
thin here", and a planner's natural response to thin context is to plan fewer
cleaning steps — which is the very quantity 实验 1 measures (附录 CS.2). The
paragraph exists to remove that reading; its presence is a **behaviour change
introduced before the freeze and declared here**.

---

## 3. Known behaviour changes and measurement breakpoints (清单 (6))

Everything in this section changed **before** the freeze and is declared so that
a reader of the 阶段 4 report does not attribute it to the experiment.

### 3.1 Glossary alias matching changes intent resolution (附录 CQ.4 #1)

Since `a7f808d`, alias matching can resolve mentions that previously failed:
`resolution_status` moves from `absent` to `full`/`partial` on affected queries,
`column_scope.columns` becomes non-empty, and `scope_fingerprint` changes with
it. Downstream this changes the cleaner's `required` scope, which under 附录
U.4.2 previously degraded to "all".

Bound on exposure measured on dev only (附录 CQ.4): **0** of the 59 dev queries
contain CJK characters, so Chinese aliases cannot fire there; ASCII alias words
appear as text in dev queries at most `churn` 6, `charges` 4, `sales` 2,
`quantity` 2, others ≤1. That is a **text-level upper bound, not a trigger
count** — actual mentions come from the intent parser's output.

### 3.2 Refine fires less often under the default trigger (附录 CQ.4 #2)

`route_after_schema`'s pre-check shares the alias index with the real
reconciliation, so mentions the glossary can now resolve no longer trigger a
refine round under `intent_refine_trigger="if_unresolved"` — which is the
**default** (`src/config/settings.py:189-190`).

**Nothing forces `always`, and nothing will — but as of 2026-08-22 a run that
omits it on the holdout is refused rather than silently different**
(冻结前必办 (15) / 附录 DD.3). `scripts/run_eval.py` still never *sets*
`intent_refine_trigger`; until (15) the requirement lived only in that field's
docstring (`settings.py:203-208`), so forgetting the variable produced a
complete, plausible-looking run whose cases silently differed in whether they
got a second intent pass. That is recoverable on dev (rerun it) and not
recoverable on the holdout, which gets one reveal (硬约束 1). The gate is
therefore scoped to `--split test`: it refuses, and it does not touch dev runs
at all.

Dev runs are covered the other way, because a refusal there would be wrong:
every eval report and every `measure_noise.py full` report now carries a
top-level `settings_snapshot` with the parsed `intent_refine_trigger`
(`scripts/run_eval.py::_settings_snapshot`). This matters for the σ_hetero /
`k` measurement (清单 (5)), which is taken on **dev**, where the gate cannot
reach: a noise floor measured under `if_unresolved` while the reveal runs under
`always` describes a different code path than the one it is used to qualify,
and before this key nothing in a saved report said which one produced it.
What cannot be prevented must at least be identifiable afterwards.

### 3.3 Cleaner impact keys appear verbatim in user-visible caveats (附录 CT.5)

`src/agents/insight_agent.py:517-534` consumes
`execution_plan.steps[].impact`: `has_impact` decides whether a lossy step
produces a mandatory-disclosure caveat (附录 R.2), and `:534` formats the whole
dict into the caveat text (`f"{tool} modified the data: {impact}"`), with the
same dict also copied into `evidence`. The cleaner's 附录 CS.4 additive
extension (`rows_added` plus column-level counts) therefore **changed the
literal text of that caveat** in the 阶段 4 report, with no MAEDA code change
and no contract version bump. Additive at the contract level; not additive in
what a reader sees.

### 3.4 U.5 escalation and the `row_alignment` measurement breakpoint (附录 CX.6)

Cleaner commit `c3639dc` fixed diff row alignment (previously DataFrame-index
based, so survivors of a row-removing step shifted up and were counted as
modified). Effects to pre-register:

- **Report the corrected behaviour.** Measured by the cleaner owner across 12
  lossy steps (附录 CW.4): `high_risk` **4 → 3**, i.e. escalation rate
  **33.3% → 25.0%**, all of the change from U.5 rule 2. The single step that
  de-escalated did so because its `cells_changed_ratio` fell from
  7360/9450 ≈ 0.78 to 281/9450 ≈ 0.030 (threshold 0.20) — it had been
  `high_risk` on the strength of an inflated input. The three steps that remain
  `high_risk` do so on `rows_removed_ratio` (0.40 / 0.20 / 0.30), and
  `rows_removed` was not touched: **the fix removed inflated "modified" counts
  without weakening disclosure of real row deletion.**
- **The inflation was layout-dependent, not a constant factor**: measured from
  ×1 (a 5-row fixture) up to ×26 (a 1050-row fixture), and exactly zero on two
  fixtures whose deleted rows were all at the tail. It can therefore neither be
  corrected by a coefficient after the fact nor waived as "our data is probably
  fine".
- **Detectable marker.** A diff whose `changes` object **has no
  `row_alignment` key** was produced by the pre-fix code path; its
  `rows_modified` / `cells_changed` are inflated on row-removing steps and must
  not be compared with post-fix numbers. `row_alignment` (and `rows_compared`)
  are additive keys added by the fix (`mcp_server/tools/generate_diff.py:195-196`).
- **Correction to 附录 CX.6's phrasing.** CX.6 located the stale artefacts in
  "`logs/`". Verified 2026-08-22: MAEDA's `logs/` contains **no diff files at
  all** (`eval_runs/`, `noise_runs/`, `replay_cache/`, `runs.db`). The pre-fix
  artefacts live in the **cleaner** repository under
  `artifacts/**/diffs/*.diff.json`; measured there today: 1103 diff-shaped JSON
  files, of which **414 carry `row_alignment` and 689 do not**. Both cohorts are
  on disk right now, which is exactly why the marker — rather than a date — is
  the discriminator.
- These keys **do not leak into `impact`**: `mcp_app.py::_impact_from_diff`
  reads keys explicitly, and the cleaner added a test pinning `impact`'s key set,
  precisely because of §3.3.

---

## 4. Reading rules for the 阶段 4 numbers (清单 (14))

From 附录 CU.8 (1)–(4) and 附录 CV.6 (5). Items (1)–(2) of CU.8 are covered in
§2 and below.

1. **`glossary_coverage` per data source, and it is not constant** — §2.
2. **`overall_aggregate` changed meaning**: it is now the mean over **successful
   cases**, not over all cases. Any report that quotes it **must** quote
   `n_cases_without_aggregate` alongside (`scripts/run_eval.py:352`, `:371`) —
   without it, a shrinking denominator is invisible in the archived report.
3. **Failed runs are scored on a smaller metric set**, so they carry
   `aggregate_score = None` and do not enter aggregate comparisons — but they
   **remain in the denominators of `error_rate` and pass@k**. Dropping the whole
   row would remove failures from the failure rate's own denominator, i.e. the
   metric would improve as the system got worse (附录 CU.5 #1 / CV.4).
4. **`error_type` has a breakpoint at E3 (`d2b2b86`).** Rows written before E3
   collapse today's `mcp_error` and `environment_error` into `pipeline_error`.
   The discriminator is the `terminal_state` column.
   **Refinement measured 2026-08-22:** the live `logs/runs.db` holds 988 rows
   and **does not have that column at all** — it is added by `RunStore`'s
   additive migration (`src/persistence/run_store.py:64-72`) the first time a
   store opens the file, which has not happened since E3 landed. So the correct
   test for a pre-E3 row is **"the column is absent, or present and NULL"**;
   `SELECT … WHERE terminal_state IS NULL` on today's file raises
   `no such column` rather than returning them.
5. **`safe_refusal` measures post-hoc interception, not entry-side judgement**
   (附录 CV.2). Every guardrail check runs on `report` / `insights` /
   `analysis_results`, i.e. after generation, so a refused run costs
   approximately what a successful run costs in tokens and latency, and the
   metric must not be read as "the system knew not to take this on".

Also carried from earlier adjudications and repeated here because it belongs in
the same list: **corpus-backed and corpus-free events must be reported
separately, never as one combined score** (附录 CP.5 #4), and an on-arm
hallucination rate on `unexplained` events that is **not lower** than the off
arm is a **pre-declared possible outcome** with a known mechanism, not evidence
of a bug (附录 CP.5 #2).

### 4.1 Two design premises carried over from 附录 CP.5

These are **premises, not findings** — stated in advance so that the numbers
they constrain are not misread later, and so that a class of natural-sounding
remedies is ruled out before someone proposes it mid-experiment.

**#1 — On `unexplained` events, the hallucination rate measures the generation
layer. There is no "correct retrieval behaviour" to measure at that layer.**
Such events have no corpus by design, and `min_results=1` guarantees a non-empty
result set, so retrieval will always return something. This is a property of the
design, not a defect, and the retrieval layer must not be scored on these
events.

**#2 — "Have MAEDA decide whether to answer based on the retrieval score" is
not implementable under the current contract, and is ruled out for 阶段 4.**
The `score` the contract exposes is a ranking function (`final_score ≡ RRF(rank)`
with no reranker), not a confidence; `metadata["source_path"]` is the empty
string on every hit (附录 CO.5), so the caller cannot even see which document it
got. Measured cosine similarity additionally overlaps across the two classes —
`UNEXPLAINED-DIP#1` scored 0.4224, **above** the corpus-backed control
`SOUTH-ONLINE-POLICY#2` at 0.3842 — so no safe threshold exists even if the
value were exposed. That comparison is 2 vs 4 samples, directional only, and
must not be extrapolated; it is recorded because it is sufficient to rule the
remedy out, not to characterise the distribution.

> Text authored by the orchestrator in ECOSYSTEM_INTEGRATION_PLAN.md 附录 DE; transcribed verbatim.

### 4.2 Two reading traps in the cleaner's `impact` numbers (附录 CS.6)

Both are properties of what the numbers mean, not defects introduced for 阶段 4.
They matter here because the D4 demo and the 阶段 4 report quote these figures
directly.

**Trap 1 — `impact` aggregation is `sum_over_steps`, not a net difference.**
A cell modified by two steps is counted twice. `aggregation="sum_over_steps"` is
an explicit field, not an implicit convention. It answers **"how much cleaning
work was done"**; it does **not** answer "how much did the data change relative
to the input". A report that wants the second quantity must not quote
`cells_changed`.

**Trap 2 — with multi-round cleaning, these numbers cover the last round only.**
`agent/runner.py::_write_manifest()` points `run_log_stage2_path` at the final
round's run log (`:449-457`), and `diff_paths` globs only the final round's
`stage2*/diffs/`. `executed_steps` and `execution_plan` have always had the same
coverage. It is **detectable rather than fixed**: `rounds_counted` and
`total_rounds` are both returned, and any inequality between them marks a
partial count. `clean_dataset`'s `max_rounds` defaults to 1, so the default path
has `rounds_counted == total_rounds`.

**A third trap from 附录 CS.6 has since been fixed and is not repeated here.**
Row-deleting steps used to report inflated `rows_modified` / `cells_changed`
because `_build_changes()` aligned on DataFrame index; that was corrected in the
cleaner repository (`c3639dc`). Its consequence for comparability across the fix
is a measurement breakpoint, and is stated in §3.4 — not here.

> Text authored by the orchestrator in ECOSYSTEM_INTEGRATION_PLAN.md 附录 DE; transcribed verbatim.

---

## 5. Spending discipline

**Local paid-test intent gate (清单 (8) / 附录 CR.2) — implemented.**
`tests/paid_calls.py` gates every suite under `tests/integration/`: a real API
key is necessary but not sufficient; `MAEDA_ALLOW_PAID_TESTS` must also carry an
affirmative value, or the suite skips. Verified on this machine 2026-08-22,
where `.env` supplies real-looking keys: `poetry run pytest tests/integration`
reports `4 skipped` and issues no calls. The opt-in is read from the process
environment only — never through `Settings`/`.env` — for the same reason
§1.4 gives for `RAG_TEST_SPLIT_REVEAL`, and it is deliberately **not** added to
`.env.example` (附录 CK.4/CZ.3: a line in a version-controlled template
self-replicates).

### 5.1 Disclosure: one unauthorised paid run, 2026-08-21

**This is disclosed for the same reason as §1.6: the legitimate basis for
trusting a spending discipline is disclosure, not self-certification.**

On 2026-08-21 the maeda-owner ran `poetry run pytest tests/integration` during a
verification pass, in violation of an explicit instruction for that turn not to
use any paid or external service. The machine had real API keys configured, so
`tests/integration/test_eval_judge_calibration.py` did not skip: 3 passed and 1
xpassed, roughly 6 calls to `score_relevance_and_groundedness`, each asking two
questions, i.e. **approximately 12 real `claude-sonnet-5` calls**, estimated at
**$0.05–0.10**. It was self-reported at the top of 附录 CQ, unprompted.

**Boundary established at the time, and unchanged:** that suite uses only
short synthetic reports hardcoded in the file. It does not read any fixture, does
not read `test_suite.json`, and touches no split. `logs/runs.db` timestamps were
unchanged. **硬约束 1 was not affected.**

**This is the failure mode that 清单 (8) exists to close.** Before the gate in
§5, the suite was safe only in environments that happened to have no key —
i.e. the safety came from an accident of configuration rather than from a
declaration of intent. The gate now requires `MAEDA_ALLOW_PAID_TESTS` in
addition to a key. Re-verified on this machine 2026-08-22 with real keys and no
opt-in: `4 skipped`, zero calls.

> Text authored by the orchestrator in ECOSYSTEM_INTEGRATION_PLAN.md 附录 DE; transcribed verbatim.

### 5.2 A known physical defect in the demo data: 76 rows of negative revenue

**Not a spending item; recorded here because it is a data fact the 阶段 4
numbers rest on** (冻结前必办 (10), 附录 CQ.6.1 / CR.5).

`data/demo/sales_data.csv` contains **76 rows with negative `revenue`** out of
48,960 (**0.16%**, minimum −256.8). Cause: `generate_demo_data.py:255` clamps at
zero **before** the event coefficients at `:256-257` are applied, so a `down`
event combined with the lead-product ×3 weight can drive the multiplier
negative. Negative revenue in sales data is physically impossible.

**Where it falls:** all 76 rows are in `North / Gadget Pro / 2024-05` (50 rows)
and `2024-06` (26 rows), i.e. the window of `EVT-2024-05-NORTH-GADGETPRO-EOL`,
whose `split` is **`dev`**. The affected window is not in the test split.
(Reading an event's `split` is structural metadata and is a precondition for the
guard to operate at all; it is not an observation of the test split — see §1.3.)

**Not fixed before 阶段 4, deliberately.** Fixing it means changing the render
order, which means re-rendering `sales_data.csv`, which means recomputing every
fixture already rebuilt in two prior rounds. The frozen `event_script.yaml` is
unaffected — this is a defect in the rendering parameters, outside the frozen
surface.

**Why it belongs in a pre-registration and not only in a backlog:** TB5's
`verify.py` checks detectability, **not physical plausibility**, so no existing
gate stops this class of defect; and the cleaner's `handle_outliers` applies IQR
treatment to `revenue`, so how these negative values are treated lands directly
on the cleaner arm of experiment 1.

> Text authored by the orchestrator in ECOSYSTEM_INTEGRATION_PLAN.md 附录 DE; transcribed verbatim.

---

## 6. Checklist coverage of this file

| 冻结前必办 | Written here? |
|---|---|
| (1) CP.2 prior-observation disclosure | **Yes** — §1.6(text from 附录 DE.2)|
| (2) split guard sunk below every entry point | Closed in the rag repo (附录 CZ/DB); its **boundary** is stated in §1.3 |
| (3) CP.5 four clauses | **Yes** — #2/#4 在 §4,#1/#3 在 §4.1(DE.3) |
| (4) CM.7 on-arm tier | **No — user decision pending.** §0 placeholder, no default |
| (5) `k` | **No** — candidate only, §0 |
| (6) behaviour changes | **Yes** — §3 (CQ.4 ×2, CT.5, CX.6) |
| (7) CQ.0 disclosure | **Yes** — §5.1(DE.4) |
| (8) paid-test intent gate | **Yes** — implemented, §5 |
| (9) pre-registration write-up incl. DB.2(1) and DB.3 | **Yes** — §2 (+ §1.3, §1.4) |
| (10) CR.5 negative revenue | **Yes** — §5.2(DE.5) |
| (12) CS.6 reading traps | **Yes** — §4.2(DE.6) |
| (14) CU.8 / CV.6 five items | **Yes** — §4 |
| (15) `--split test` refuses a non-`always` `intent_refine_trigger` (附录 DD.3) | **Yes** — implemented in `scripts/run_eval.py`; stated in §1.2 and §3.2 |
