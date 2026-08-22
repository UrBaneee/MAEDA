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

- `--split` defaults to `dev`; `test` must be typed (`scripts/run_eval.py:225`)
  and prints `*** Running against the HOLDOUT TEST SPLIT. ***` (`:270`).
- The rag MCP server is a **separate process** reached over streamable-http
  (`settings.rag_server_mcp_url`, default `http://localhost:8002`;
  `docker-compose.yml:26` sets `http://rag-server:8002`). MAEDA does not start
  it in-process and does not pass it any reveal declaration.
- `MAEDA_INTENT_REFINE_TRIGGER=always` is **required on the command line**, see
  §3.2 — nothing in `scripts/run_eval.py` sets it (verified 2026-08-22:
  `intent_refine_trigger` appears in `src/config/settings.py`,
  `src/graph/router.py` and unit tests only).

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

**Action, owner = experiment designer / lead:** run that enumeration once
before freeze and record the result here.

### 1.6 Prior observations of the test split — **NOT YET WRITTEN**

Reserved for 冻结前必办 (1) / 附录 CP.2 (the 2026-08-18 19:40:07 BM25-only
observation of `EVT-2024-03-SOUTH-UNITS-SPIKE`'s two probe queries). Not
written in this pass; that item has not been dispatched.

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

**Verified 2026-08-22: nothing forces `always` for eval runs.**
`scripts/run_eval.py` never sets `intent_refine_trigger`; the requirement lives
only in that field's docstring (`settings.py:203-208`). It is therefore a
**run-configuration obligation of this pre-registration**, not a mechanism: the
reveal command in §1.2 carries `MAEDA_INTENT_REFINE_TRIGGER=always` so that
every case takes the same code path in every arm. If any arm is run without it,
that arm's cases silently differ in whether they got a second intent pass.

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

**Not yet written:** 附录 CQ.0's disclosure of the 2026-08-21 unauthorised paid
run (清单 (7)), and 附录 CR.5's negative-revenue fact (清单 (10)).

---

## 6. Checklist coverage of this file

| 冻结前必办 | Written here? |
|---|---|
| (1) CP.2 prior-observation disclosure | **No** — §1.6 placeholder |
| (2) split guard sunk below every entry point | Closed in the rag repo (附录 CZ/DB); its **boundary** is stated in §1.3 |
| (3) CP.5 four clauses | Partly — #2 and #4 in §4; #1/#3 not yet written |
| (4) CM.7 on-arm tier | **No — user decision pending.** §0 placeholder, no default |
| (5) `k` | **No** — candidate only, §0 |
| (6) behaviour changes | **Yes** — §3 (CQ.4 ×2, CT.5, CX.6) |
| (7) CQ.0 disclosure | **No** — §5 |
| (8) paid-test intent gate | **Yes** — implemented, §5 |
| (9) pre-registration write-up incl. DB.2(1) and DB.3 | **Yes** — §2 (+ §1.3, §1.4) |
| (10) CR.5 negative revenue | **No** — §5 |
| (12) CS.6 reading traps | **No** — `impact` aggregation is `sum_over_steps`, not a net difference; not yet written up |
| (14) CU.8 / CV.6 five items | **Yes** — §4 |
