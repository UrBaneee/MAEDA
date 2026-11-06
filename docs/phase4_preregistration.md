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
| Number of paired trials `k` at the reveal | σ_hetero measured on the dev split; **and the size N of the new holdout, which σ_hetero also sets** (清单 (5)) | Candidate draft, not a pre-registered value |
| Size N of the new MAEDA holdout | σ_hetero (清单 (5)); the suite must then be authored (附录 DQ) | Not yet determined. No case may be authored into it before N is fixed |

**Closed:**

**实验 1 RAG on-arm retrieval tier — decided.** The on-arm uses **hybrid
retrieval** (vector + BM25), the `hybrid` tier reached on the established
`docker-compose` path (§1.2). `MAEDA_RAG_EXPECTED_RETRIEVAL_MODE=bm25_only` is
**not** set.

- **Provenance**: this value comes from a **user adjudication dated
  2026-08-22** (`ECOSYSTEM_INTEGRATION_PLAN.md` 附录 DG.4, user's own words
  「做 a 没有问题」) — it is **not** a default written in by omission.
  `settings.maeda_rag_expected_retrieval_mode` happens to default to
  `"hybrid"` as well (`src/config/settings.py:153`), so the adjudicated value
  and the default coincide; that coincidence is exactly why this provenance
  line has to be written down explicitly, per the "no default may be written
  here" rule this table already carries.
- **Basis**: 附录 AL.3's measured retrieval-tier gap on the dev split —
  BM25-only Recall@5 **0.7083** vs hybrid **0.9444**, and 2 of the 12 dev
  probe queries were not retrieved at all under BM25-only (cross-language
  term mismatch). 硬约束 1 gives one reveal; it should not be spent on a tier
  already known to trail by this much and that is not the shipped
  configuration.
- **Attached condition** (附录 CP.5 #4): corpus-backed and corpus-free events
  must be reported separately in the 阶段 4 report; no combined total score.

**Arm count — decided.** The reveal collects **five** conditions: the 2×2
factorial of 实验 1 ({`cleaner_mode`: `force_on` / `force_off`} ×
{`rag_mode`: `force_on` / `force_off`}, four cells including the interaction),
plus 实验 2's `(cleaner_mode=auto, rag_mode=auto)`.

- **Provenance**: user adjudication, 2026-08-23. Not a default.
- **Two conditions attached, both binding**: (i) **separate reporting** — 实验 1's
  net-gain table and 实验 2's routing precision/recall/confusion matrix are
  reported as two tables and never combined into one score; `fallback` is
  classified separately and never counted as `force_off`. (ii) **the set of
  pairwise comparisons is fixed here, in advance** — five arms admit ten
  pairwise comparisons where four admit six, and 附录 AP.4 records a real run
  in which 9 of 28 pairwise comparisons came out falsely significant against a
  calibrated 1/28. The comparisons actually to be made must be listed before
  any result is seen; any comparison not listed here is exploratory and is
  reported as such.

**硬约束 1 — reading and budget, decided.** Read per holdout, not per project;
re-draw budget one. See §1.6.

> Text authored by the orchestrator in ECOSYSTEM_INTEGRATION_PLAN.md 附录 DW.6; transcribed verbatim.

---

## 1. 硬约束 1 — the one reveal

### 1.1 There are three sets now, and only two of them are holdouts

| Set | Where | Size | Status |
|---|---|---|---|
| **Retired MAEDA holdout** | `tests/eval/test_suite.json`, `split == "test"` | 41 of 100 | **Not a holdout.** Reveal adjudicated CONSUMED 2026-08-23 (§1.6) |
| **New MAEDA holdout** | not yet constructed | N — **NOT YET DETERMINED**, set by σ_hetero (清单 (5)) | Holdout, one reveal, re-draw budget spent |
| **TB5 retrieval holdout** | rag `eval/fixtures/tb5_retrieval/test.json`, from the frozen test-split `probe_queries` in `event_script.yaml` | 8 derived entries; the guard's digest set holds 10 frozen test-split probe queries | Holdout, **reveal unspent** (§1.5, 附录 CP.2) |

Conflating any two of these is the error §1.6 records. The rag split guard
(附录 CZ) protects the third row only; it has never protected either MAEDA set.
See §1.3.

The new MAEDA holdout excludes `predictive` cases by decision (user, 2026-08-23):
`scripts/compute_ground_truth.py` structurally skips predictive /
data_mismatch / ambiguous cases, so their ground truth cannot be audited.
**阶段 4's conclusions therefore do not cover predictive queries.** This is a
deliberate narrowing of scope, not an omission.

> Text authored by the orchestrator in ECOSYSTEM_INTEGRATION_PLAN.md 附录 DW.4; transcribed verbatim.

**Two other query types are also skipped by `scripts/compute_ground_truth.py`
and are nevertheless retained.** The script skips predictive / data_mismatch /
ambiguous because none has a checkable **numeric** ground truth
(`_KNOWN_UNANSWERABLE_CASES`, `tests/unit/test_phase9.py:1262`). That reason is
not the same in all three cases:

- **predictive** — the fact is not in the data at all. No method recovers it.
  **Excluded** (user adjudication, 2026-08-23, above).
- **data_mismatch** — the query asks for something the dataset does not
  contain. Whether it contains it is trivially decidable at authoring time;
  what is missing is only that the correct answer is a refusal rather than a
  number. **Retained.**
- **ambiguous** — whether a query is ambiguous given the schema is likewise
  decidable at authoring time. **Retained.**

The two retained types carry a **behavioural** ground truth
(`expected_behavior`) and are scored by `safe_refusal` and `error_rate`, not by
`factual_accuracy`. They are **not** part of H1/H3's sample under any value of
`cleaning_relevant`.

**Why they are not dropped along with predictive**: `safe_refusal` is the one
metric that answers "does the system invent an answer when the data cannot
support one". A holdout with no absent-data cases cannot measure it. The RAG
side protects exactly this property deliberately — TB5's event script carries
two `unexplained` events for it, and 附录 AA.2 raised an alarm when IQR
clipping threatened one of them. Removing the MAEDA-side equivalent would leave
the two halves asymmetric for no reason.

**Prerequisite, not yet met.** `score_factual_accuracy` returns a score for
every case rather than `valid=False` when no numeric ground truth exists
(`src/eval/metrics.py:714-727`: a case with no comparable summary numbers
returns 0.8 / "pass" / "No numbers to cross-check"), and it is **not** covered
by E3's not-applicable gate, which reaches only `answer_relevance` and
`groundedness` (`src/eval/runner.py:272-275`). Until corrected these cases
would contribute a fabricated `factual_accuracy` to any sample they entered —
and the same defect lets a run whose analysis steps all failed score 0.8 on
H1/H3's primary metric. **The correction is required before the reveal** and is
tracked in the pre-reveal patch batch.

> Adjudications 2026-10-28 (query-type scope) and 2026-10-30 (the
> `score_factual_accuracy` defect, found while verifying the prerequisite).

### 1.2 How the reveal is executed — who, in which process, with which command

**(A) MAEDA holdout — SUSPENDED.** The command below targeted the retired
holdout and no longer points at one (§1.6). The reveal command for the new
MAEDA holdout is **NOT YET WRITTEN**: its form depends on how the new holdout
is labelled, which is 清单 (21) — if it reuses `split == "test"` the two
existing gates (清单 (15) on `run_eval.py`, 清单 (16) on `measure_noise.py`)
continue to apply unchanged; if it is labelled differently, both gates go
silently dead and must be changed with it. The block below is retained as the
record of what the retired reveal would have been, not as an instruction.

**(B) below is unaffected and remains in force.**

> Text authored by the orchestrator in ECOSYSTEM_INTEGRATION_PLAN.md 附录 DW.5; transcribed verbatim.

```
cd ~/MAEDA
MAEDA_INTENT_REFINE_TRIGGER=always \
MAEDA_CLEANER_MODE=<arm> MAEDA_RAG_MODE=<arm> \
poetry run python scripts/run_eval.py --split test --trials <k>
```

- `--split` defaults to `dev`; `test` must be typed (`scripts/run_eval.py:251`)
  and prints `*** Running against the HOLDOUT TEST SPLIT. ***` (`:330`).
- The rag MCP server is a **separate process** reached over streamable-http
  (`settings.rag_server_mcp_url`, default `http://localhost:8002`; **MAEDA's
  own** `docker-compose.yml:26` sets `http://rag-server:8002`). MAEDA does not
  start it in-process and does not pass it any reveal declaration.
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

Reason (附录 DB.3): rag's `docker-compose.yml` gives both the `studio` and
`mcp` services `env_file: - .env`, so one line in `.env` becomes a permission compose
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
with the disclosure in §1.7: 附录 CP.2 recorded "7 query records, timestamp
2026-08-18 19:40:07" — 7 distinct queries, timestamp taken from the later end of
the batch.

**Result: apart from the single occasion disclosed in §1.7, no test-split probe
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

### 1.6 The first MAEDA holdout was burned. This section is its disclosure.

**Status: 硬约束 1's reveal on the first MAEDA holdout is adjudicated
CONSUMED** (user adjudication, 2026-08-23). The 41 cases carrying
`split == "test"` in `tests/eval/test_suite.json` **are no longer a holdout**
and must not be reported as one.

This section replaces an earlier version stating that one prior observation had
occurred and that "there is no other". **That statement was false.** It was
written by the orchestrator on the strength of the §1.5 audit, which enumerated
every trace store *in the rag repository* and therefore answered a question
about the TB5 retrieval holdout only. The MAEDA holdout had never been audited
at all. §1.1 warns that conflating the two is easy and wrong; the earlier text
did exactly that.

**What the audit found once it was actually run** (附录 DM / DN / DO / DP):

1. The 41 cases were run in full **at least three times**: 2026-08-02
   19:52–20:04 and 2026-08-03 22:17–22:48, both recorded only in
   `logs/runs.db` — which is not under version control and which no
   file-listing audit could see — and 2026-08-03 15:48, which produced
   `logs/eval_runs/v1_test_reveal.json` carrying
   `overall_aggregate = 0.7365189129975717`, an aggregate score over the
   entire holdout.
2. `data/annotations/answer_key_claude_sonnet_5_100.json` carries real judge
   scores with per-case reasoning for all 100 cases, all 41 included.
3. **Three layers of the evaluation chain were changed after looking at these
   cases:**
   - **the system under test** — `164b78d` (2026-08-02 21:38, 1h34m after the
     run covering all 41) fixed false-superlative claims, citing C14 and DG09
     as its confirming evidence; both are holdout cases. Its new guardrail was
     validated as "0/100 false positives across the full cached corpus", i.e.
     including the holdout.
   - **the judge** — `8a6e7f7` (2026-08-01) changed `EVAL_GROUNDEDNESS_SYSTEM`
     on the strength of human labels pooled across all 100 cases, all 41
     included.
   - **the ground truth** — the 2026-08-17 rewrites changed `ground_truth` on
     **69 of 100** cases, **30 of them in the holdout**, after
     `v1_test_reveal.json` had already scored `factual_accuracy` against the
     older values.
4. **A disclosure to this effect already existed in the repository and never
   reached this document.** `docs/judge_calibration.md`, written 2026-08-01/02,
   states that the judge fix was measured "not a blind holdout", that "all 120
   collected labels had already been read in full while diagnosing the bug",
   and that the fix should be validated "once a genuine (unread-by-Claude)
   holdout exists". DG09 is discussed there by name.
5. **No uncontaminated case remains in the suite.** All 100 were observed.
   Excluding `logs/runs.db` entirely, every case still appears in **at least
   seven** independent channels, and four files each cover 100/100 — every
   case carries both judge scores and human labels.

**The adjudication, and what it does not rest on.** The user ruled the reveal
CONSUMED rather than ruling it harmless. No assessment of how much the
observations mattered is offered here and none should be inferred: assessing
damage after the fact is how a discipline breach gets retroactively approved,
which is the reasoning 附录 CP.2 itself used and which applies here with more
force, not less.

**One fact that cuts against a comfortable reading, stated deliberately.** It
is tempting to say the contamination was ordinary development that happened to
precede the split's creation on 2026-08-01, and the timeline partly supports
that. **That defence is weak.** `164b78d` landed on 2026-08-02, after the split
existed and 1h34m after a run covering all 41 cases, and it changed the system
on evidence read off two of them. That is "looked at holdout behaviour, then
changed the system" — precisely what 硬约束 1 forbids in the same breath as it
defines itself ("只揭盲一次、不反复回看调参"). **What follows is a re-draw
after a breach, not a clarification of a non-breach.**

**Scope of the adjudication.** 硬约束 1 reads "test split 只进行一次正式
reveal … 只揭盲一次、不反复回看调参". Its subject is *a* test split and its
stated purpose is to prevent an iterate-on-feedback loop, not to impose a
project-lifetime quota. It is therefore read **per holdout**: the retired
holdout's reveal is spent; a newly constructed holdout carries its own single
reveal. **The TB5 retrieval holdout is a separate holdout and its reveal is
unspent** — §1.5's audit is what establishes that, and it is the question that
audit actually answered. **The two budgets are not interchangeable**, and the
TB5 reveal must not be spent early on the reasoning that the MAEDA side is
being redone anyway.

**Re-draw budget: one.** This is a pre-registered limit, fixed before any
result from the new holdout is seen. The new MAEDA holdout gets one reveal.
**If it is contaminated in turn, it may not be re-drawn on the authority of
this document**: the reason the first one was burned must be disclosed and the
situation re-adjudicated by the user. Without this limit the per-holdout
reading degenerates into "burn, re-draw, repeat, report the best" — the same
overfitting the constraint exists to prevent, at a slower cadence.

> Text authored by the orchestrator in ECOSYSTEM_INTEGRATION_PLAN.md 附录 DW.3; transcribed verbatim.

### 1.7 The TB5 retrieval holdout: one prior observation, and why its reveal is still unspent

This section holds material that was §1.6's entire content until 附录 DW.3
replaced §1.6 to disclose the MAEDA holdout burn. It is **about the other
holdout** (§1.1, row 3) and is kept here because §1.1 and §1.6 both assert that
the TB5 reveal is unspent, and this is the evidence for that assertion. Its
removal without a new home was a drafting error by the orchestrator, recorded in
附录 DY.

**One prior observation of the TB5 retrieval holdout has occurred. It is
disclosed here in full rather than assessed as harmless by the parties
involved** (冻结前必办 (1), 附录 CO.0 / CP.2).

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

**Adjudication (附录 CP.2): this does not consume the TB5 holdout's single
reveal.** The reasoning is not "the damage was small" — assessing damage after
the fact is how a discipline breach gets retroactively approved. It is:

1. **No quantity comparable to ground truth was produced.** That event has no
   corpus by design and an empty `expected_sources`, so there is no recall, no
   hit rate, no pass/fail to compute. What was seen is which 5 chunks BM25
   returned for a query that has no correct answer.
2. **Neither conclusion drawn that round depends on the observed data.** Both
   are properties read out of source and true for any query: `final_score` is
   identically `RRF(rank)` with no reranker (the `1.0 / (k + rank)` accumulation
   in `RRFFusion`), and the language filter applied after fusion
   (`_language_filter(..., min_results=1)`) cannot empty a non-empty candidate
   list. State the second one precisely: it is **not** a guarantee that retrieval
   never returns an empty set — that filter is a no-op for non-CJK queries, and
   its contract is `length >= min(min_results, len(candidates))`. What actually
   made the observed query return something is that the index is non-empty, not
   that fallback. 附录 CP.2's adjudication does not rest on it either way:
   point 1 — the event has no corpus and no `expected_sources`, so no quantity
   comparable to ground truth exists — is what carries it. **A conclusion that is
   invariant under the observation received no information from it.**
3. **No parameter, threshold, prompt, or criterion was changed because of it.**
   Every commit in all three repositories after 附录 CJ was checked; none
   relates to that event.

**Independently verified 2026-08-22** by enumerating all 22,387 trace rows in
the rag repository (§1.5): **no test-split probe query has ever been executed
apart from this one occasion.** Read §1.5's two limits alongside that, and read
the scope of that sentence precisely — **it is a statement about the TB5
retrieval holdout only.** An earlier version of this text said "there is no
other" without that qualifier, and was thereby read as covering the MAEDA
holdout, which §1.5 never examined. That error is what §1.6 now discloses; the
qualifier here is the correction (附录 DN / DP).

**No claim of harmlessness is being made here.** The reader is given the time,
the object, the content, and the falsifiable assertion in point 3, and may draw
their own conclusion.

> Table, adjudication and points 1–3 authored by the orchestrator in
> ECOSYSTEM_INTEGRATION_PLAN.md 附录 DE, restored and scope-corrected in 附录 DY.2;
> transcribed verbatim.

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

### 3.0 `insight.sources` is removed before the calibration re-run

`insight.sources` is never produced by the model — `INSIGHT_GENERATOR_SYSTEM`'s
output schema has no such field — and `src/agents/insight_agent.py:199-203`
fills it with the run's top three retrieved filenames, identically on every
insight. It then reaches the report-writer prompt through
`to_dict()` (`:255`). Removal is adjudicated (路线 A, user, 2026-10-26).

**It must land before the calibration re-run**, so that calibration and reveal
score the same report shape; landing it afterwards would calibrate on reports
carrying the fabricated field and reveal on reports without it.

**Measured scope of the change: none of the 20 stored calibration reports cites
any source file** (zero hits for a filename pattern across all 20), so the
report writer is already ignoring the field and its removal is not expected to
alter report text. *A bound, not a proof*: the field's presence in the prompt
could still affect wording in ways a filename search does not detect, and these
20 reports predate the 2026-08-17 data re-render.

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
  are additive keys added by the fix, in `_build_changes()` (cleaner repo,
  `mcp_server/tools/generate_diff.py`).
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
Such events have no corpus by design and an empty `expected_sources`, so no
quantity comparable to ground truth exists at that layer. Retrieval will still
return something, but state the reason precisely: it is because the index being
queried is non-empty, and **not** because of the `min_results=1` fallback in the
language filter applied after fusion (`_language_filter` in the rag query
pipeline). That filter is a no-op for non-CJK queries, and its contract is only
`length >= min(min_results, len(candidates))`: it cannot empty a non-empty
candidate list, and it cannot fill an empty one. This is a property of the
design, not a defect, and the retrieval layer must not be scored on these
events.

> Correction, 2026-08-22: the original wording of this paragraph attributed the
> non-empty result set to `min_results=1`. That attribution was wrong and is
> withdrawn; the premise itself is unchanged, because it never depended on that
> mechanism. Authored by the orchestrator in ECOSYSTEM_INTEGRATION_PLAN.md
> 附录 DH.3 (same error, same origin as the §1.6 correction in 附录 DG.2);
> transcribed verbatim.

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
| (1) CP.2 prior-observation disclosure | **Yes** — §1.7(restored and scope-corrected per 附录 DY.2;原文出自 附录 DE) |
| (2) split guard sunk below every entry point | Closed in the rag repo (附录 CZ/DB); its **boundary** is stated in §1.3 |
| (3) CP.5 four clauses | **Yes** — #2/#4 在 §4,#1/#3 在 §4.1(DE.3; premise #1 corrected per 附录 DH.3, 冻结前必办 (17)) |
| (4) CM.7 on-arm tier | **Yes** — §0 "Closed" (user adjudication 2026-08-22, 附录 DG.4/DJ.4, 冻结前必办 (19)) |
| (5) `k` and N | **No** — `k` still candidate only (§0); **N (new MAEDA holdout size) is a newly introduced open item, NOT YET DETERMINED**, both wait on σ_hetero, which 附录 DV records as never measured (pilot completed 2026-08-23, repeats=1, no σ obtained) |
| (6) behaviour changes | **Yes** — §3 (CQ.4 ×2, CT.5, CX.6) |
| (7) CQ.0 disclosure | **Yes** — §5.1(DE.4) |
| (8) paid-test intent gate | **Yes** — implemented, §5 |
| (9) pre-registration write-up incl. DB.2(1) and DB.3 | **Yes** — §2 (+ §1.3, §1.4) |
| (10) CR.5 negative revenue | **Yes** — §5.2(DE.5) |
| (12) CS.6 reading traps | **Yes** — §4.2(DE.6) |
| (14) CU.8 / CV.6 five items | **Yes** — §4 |
| (15) `--split test` refuses a non-`always` `intent_refine_trigger` (附录 DD.3) | **Yes** — implemented in `scripts/run_eval.py`; stated in §1.2 and §3.2. **Now targets a retired holdout** (§1.6) — still mechanically active (protects the `test` label), but see (21) |
| (20) part 2 — pre-registration holdout identity rebuild (附录 DU.3/DW) | **Partially done.** §1.6 rewritten in full (附录 DW.3); §1.1 rewritten in full as a three-set table (附录 DW.4); §1.2 (A) marked SUSPENDED with the retired command retained as record only, (B) untouched (附录 DW.5); §0 `k` row rewritten, N row added, arm-count/硬约束-1 "Closed" block appended (附录 DW.6). **Still open, genuinely, not just unwritten**: the new MAEDA holdout's size N (blocked on σ_hetero, 清单 (5) — 附录 DV: dev-split pilot completed 2026-08-23 but measured `repeats=1`, so σ_hetero was never obtained) and the new reveal command for §1.2 (A) (blocked on 清单 (21)'s labelling decision). Both markers are left as **NOT YET DETERMINED** / **NOT YET WRITTEN** deliberately — they are unresolved facts, not missing prose. §1.7 has restored the CP.2 disclosure that 附录 DW.3 had deleted from this file, so the gap that this row's earlier phrasing implicitly left open is repaired. |
| (21) new holdout label vs. guard consistency (附录 DU.4) | **Not started.** If the new MAEDA holdout reuses `split == "test"`, gates (15) and (16) continue to protect it unchanged; if it is labelled differently, both gates go silently dead (§1.2 (A) annotation, 附录 DW.5) and must be updated together with a test proving the new label is still caught (判据 CK.6). |
| (22) pre-registered comparisons and decision rules (附录 DW.6(3)(ii), EB, EC.3, ED.2) | **Yes** — §7. Arms, the three confirmatory hypotheses (H1/H2/H3), the routing-accuracy analysis, sample-eligibility fields, stratification rule, and multiplicity correction are all written. **7.6's decision rule is conditional on N**, the new MAEDA holdout's size, **which is not yet determined** (blocked on σ_hetero, 清单 (5)) — if N comes in materially smaller than the retired holdout's 41, 7.6 must be recomputed rather than carried over as written. **Amended 2026-10-30**: `groundedness` does not meet §8.1's bar, so **H2 is exploratory and the confirmatory set is H1 and H3** (§8.3(1)); §7.6's multiplicity is recomputed at two tests, 97.5%. §7.3 is expanded with 实验 2's labels, a pre-derived prediction for the cleaner half, the RAG routing predicate, and a per-arm k. §7.4 gains `cleaning_material`. |
| (23) judge validity threshold, its pre-committed consequences, and the human labelling rubric | **Yes** — §8 and §9. The threshold is relative to the human reference's own test-retest reliability, measured at `answer_relevance` 0.4032 / `groundedness` 0.7167, replacing the absolute 0.60 bar (**stricter** for groundedness, **looser** for relevance). Four consequences pre-committed in §8.3; offline re-scoring bounded in §8.5; judge output specification and offline scoring rules in §8.6; the rubric, derived from the annotator's contemporaneous notes, in §9. **Provenance**: user adjudications 2026-10-28 and 2026-10-30; §8.1 carries its own disclosure that the rule was formulated after the machine scores were known. |

---

## 7. Pre-registered comparisons and decision rules (清单 (22))

**Fixed before any result from the new holdout is seen.** 附录 AP.4 records a
real run in which 9 of 28 pairwise comparisons came out falsely significant
against a calibrated rate of 1/28. Five arms admit ten pairwise comparisons, and
each case yields **twelve** metrics (`answer_relevance`, `groundedness`,
`factual_accuracy`, `intent_accuracy`, `tool_selection`, `step_success_rate`,
`chart_appropriateness`, `token_cost`, `retry_count`, `error_rate`,
`safe_refusal`, `total_latency`) plus a composite `aggregate_score`. Unconstrained
that is 120 tests before stratification, at which scale "something came out
significant" is a mathematical certainty rather than a finding. **Each
confirmatory hypothesis below is therefore bound to exactly one primary metric,
and everything else is descriptive.** H2's primary metric (§7.2.1) is a thirteenth
metric, computed for H2 only.

### 7.1 Arms

| Label | `cleaner_mode` | `rag_mode` | Role |
|---|---|---|---|
| **B** | `force_off` | `force_off` | shared baseline |
| **C** | `force_on` | `force_off` | 实验 1 cleaner arm |
| **R** | `force_off` | `force_on` | 实验 1 RAG arm |
| **CR** | `force_on` | `force_on` | interaction cell |
| **A** | `auto` | `auto` | 实验 2 |

### 7.2 Confirmatory hypotheses — three, and only three

| | Test | Primary metric | Sample |
|---|---|---|---|
| **H1** | `C − B` — cleaner net gain | **`factual_accuracy`** | cases with `cleaning_relevant == true` |
| **H2** | `R − B` — RAG net gain | **`event_attribution_accuracy`** (§7.2.1) | cases linked to an event with `event_type == normal` |
| **H3** | `(CR − R) − (C − B)` — interaction | **`factual_accuracy`** | cases with `cleaning_relevant == true` |

The two subsystems are bound to **different** primary metrics deliberately. The
cleaner's claim is that dirty data produces wrong answers and cleaning fixes
them, which `factual_accuracy` measures against ground truth; RAG's claim is
that retrieved knowledge lets the report name the real cause of an anomaly,
which `event_attribution_accuracy` measures against the frozen event script. A single
composite would dilute both. **H3 is a contrast, not two pairwise comparisons:
it is one test.**

**H2's confirmatory status is conditional on the scorer bar in §7.2.1 and is
currently not met: the scorer does not exist yet.** *(Status 2026-11-03, 附录 FN:
the scorer exists and is frozen; machine-vs-human κ on pass 1 is 0.616; the bar
cannot be judged until pass 2 gives the human test-retest κ. Still not met.)* Until the bar is met, H2 is
reported as exploratory and the confirmatory set is H1 and H3. Were the scorer to
meet the bar before the freeze, H2 returns to confirmatory and §7.6's multiplicity
reverts to three tests at 98.3%.

**Disclosure: H2's metric was changed on 2026-11-01 (user, 附录 FI/FJ).** It was
`groundedness`. The change came after that metric was seen to miss its §8.1 bar on
the calibration set, so that sequence is stated here. The stated reason is
construct validity, not the miss: `groundedness` measures traceability to the
evidence shown, the evidence set differs by arm, a distractor document makes a
wrong cause count as supported, and all 20 calibration samples had empty
`rag_context` (附录 FH). No `R − B` result on any metric had been observed, on dev
or holdout, when the change was made. `groundedness` stays in the report as a
descriptive metric.

#### 7.2.1 `event_attribution_accuracy`

**Unit.** The event is the independent unit. Each `normal` event in the frozen
event script contributes one score per arm; several cases on one event are
averaged first and never counted as separate observations.

**Trial score.** 1 if the label is `correct`, otherwise 0. The scorer reads the
report and assigns one of three labels against the event's cause in the frozen
script:

| Label | Meaning | Trial score |
|---|---|---|
| `correct` | names the event's cause as an explanation of the anomaly, and no other cause as the main one | 1 |
| `wrong` | names a different cause as the main one | 0 |
| `none` | describes the anomaly and offers no cause | 0 |

Hedged wording ("possibly because of ...") counts as naming the cause, since this
metric does not measure stated confidence. The arm is hidden from the scorer and
from the human labeller. String matching is not a permitted scorer (附录 X.8).

**Refused runs (decided 2026-11-02, 附录 FM).** A run that ends in `safe_refusal`
delivers no report to the user, so in the primary analysis its trial score is
**0**, whatever its text says. Its text is still labelled, by the human and the
judge, and a **sensitivity analysis** recomputes H2 from the text labels with the
refusal ignored. The per-arm refusal rate is reported descriptively. Dropping
refused runs is not permitted: refusals differ by arm, so dropping them would
select different subsets per arm (informative censoring). If the primary and
sensitivity results disagree, the difference is attributed to the guardrail, not
to retrieval, and is reported as such.

**Test.** Exact event-level sign-flip test on the per-event `R − B` means (random
sign-flips only if the number of events makes enumeration impractical), at the
§7.6 α. This replaces `paired_bootstrap_ci` for H2 only: on simulated null data
the bootstrap ran at 0.072 at n = 5 and 0.041 at n = 10 against a nominal 0.0167
(附录 FI). The bootstrap interval is reported descriptively. The 0.028 practical
floor in §7.6 still applies.

**Scorer bar, same logic as §8.1.** Machine-vs-human Cohen's κ on the three labels
must reach the human test-retest κ on the same reports, 100% of the ceiling. At
least 60 dev reports, from arms B and R, labelled blind. The rubric is frozen
before the first label is written. If the bar is missed, the remedy is more
reports, not more scorer variants on the same reports (§8.4).

**Reported separately, descriptive only (附录 CP.5).** Wrong-attribution rate on
`distractor` events and hallucination rate on `unexplained` events. The holdout
has one of each today, which supports no test. An on-arm hallucination rate not
below the off arm's is a foreseeable result (附录 CO, CP.5).

**N is open.** At least 7 events are required for any result to reach α = 0.0167
at all (2/2^7 = 0.0156), so the 3 `normal` events now in the test split cannot
support H2 whatever the effect. The planning value is **about 12**; the final
number is set from a dev pilot of the two arms' rates (附录 FI).

### 7.3 实验 2: routing accuracy, and a prediction made before the reveal

Arm A's result is the **precision / recall / confusion matrix of its routing
decisions against the labelled "should have fired" ground truth**, reported in
its own table. `fallback` is classified separately and is **never** counted as
`force_off` — "tried and degraded" is not "decided not to call". **实验 2 does
not consume any of the confirmatory budget in 7.2.**

**The labels** are the authoring-time fields of §7.4, not a separate annotation
pass: `cleaning_material` is "cleaning should have fired", `corpus_backed` is
"retrieval should have fired".

#### 7.3.1 Cleaner half: a falsifiable prediction derived in advance

The cleaner's trigger is `has_critical_issues`, frozen at 附录 U.4.1 as three
dimensions over the intent's `required` column scope:

| Dimension | Threshold | Reachable maximum on the frozen data |
|---|---|---|
| missing | mean `null_ratio` > 0.10 | **0.03** |
| duplicate | `duplicate_row_ratio` > 0.05 | **≈0.0196** |
| schema / type inconsistency | any hit | **unreachable** |

`outlier` is deliberately **not** a trigger dimension (附录 B.1, carried into v2
unchanged).

The maxima come from the generator, not from observation:
`scripts/generate_demo_data.py:279-288` injects quality noise into
**`sales_data.csv` only** — 3% missing `revenue`, 2% duplicated rows, 1% outlier
revenues (×10). The other three demo datasets receive none. Because only
`revenue` carries nulls, the arithmetic mean of `null_ratio` over any column
scope is **at most 0.03**, reached only when the scope is exactly
`["revenue"]`; any wider scope is diluted by zero-null columns. The duplicate
figure is 2% of the original rows appended, i.e. 2/102 of the final file.

**Prediction, recorded before the reveal:** `has_critical_issues` cannot be true
for any query against this frozen corpus. 实验 2's cleaner half will therefore
report **recall = 0** and **precision undefined** (the system emits no
positives, so the precision denominator is zero). **If the reveal shows
otherwise, this derivation or the frozen inputs have changed, and that
discrepancy must be investigated before the routing numbers are read.**

**What this measures is a coverage gap, not a tuning error.** The data's three
defect classes are: low-rate missing (3%, below a threshold set deliberately
high), **negative `revenue` on 76 rows** (§5.2) which **no dimension covers at
any threshold**, and outliers (excluded by design, and correctly so — 附录 AM
records the llm planner selecting `skip_columns=["revenue","units","rep_id"]`
with the reason "extreme values are legitimate", a judgement that requires
intent and therefore cannot live in a pre-intent gate).

**The criterion is not changed to make this experiment produce a different
number.** Widening it (a lower missing threshold, or a new dimension covering
negative values) is a 附录 U.4 contract change requiring a TB4 v2 design round,
and is recorded as future work. Changing the system under test so the
measurement improves is the inversion this pre-registration exists to prevent.

#### 7.3.2 RAG half

RAG's `auto` is currently equivalent to "always retrieve" (附录 CC.7 裁定 4 — an
adjudicated, documented degeneration, not a missing feature). Measured as is,
the RAG half returns recall = 1 and precision = the base rate: both are
determined by the design, not by the system, and carry no information.

**A minimal conditional criterion is therefore added MAEDA-side before the
reveal**, declared here in full:

> **Retrieval is attempted when, and only when, the parsed intent's
> `query_type` is `diagnostic` or `exploratory`. It is skipped for
> `descriptive` and `comparative`.** (`predictive` does not occur: the new
> holdout excludes it, §1.1.)

The predicate lives in the **shared gate** `rag_retrieval_decision(state)`,
which both the conditional edge and `retrieve_knowledge_node`'s defensive
backstop already call, so the routing decision and the backstop cannot drift
apart — the same construction 附录 CC.2 required for the cleaner gate.

**This governs `auto` only.** `force_on` still retrieves unconditionally and
`force_off` still skips unconditionally; the criterion is bypassed in both.
实验 1's R and CR arms are therefore unchanged by this addition, and remain
comparable to every measurement taken before it.

**The two binding constraints are met:**

1. **Decidable from runtime state alone.** `query_type` comes from the parsed
   intent. The gate never reads `corpus_backed`, `event_id`, or any other
   authoring-time field — the system cannot see its own label.
2. **Declared before any reveal data is seen**, in this frozen document.

**One source of trial-to-trial variation, deliberately.** `query_type` is
produced by the LLM intent parser, so it can differ between trials and from the
case's authored `expected_query_type`. This is why arm A runs at k = 2 rather
than k = 1 (§7.3.3).

**A decomposition that costs nothing.** Both the parsed and the authored
`query_type` are recorded on every row, so each routing error splits into two
causes without additional measurement:

| Parsed vs authored | What the error means |
|---|---|
| differ | **intent classification error** — the router behaved correctly on the label it was given |
| agree | **routing-logic error** — the criterion itself is wrong for this case |

**This decomposition is reported, and is descriptive.** It supports no
confirmatory claim.

#### 7.3.3 Arm A does not need the same k as the net-gain arms

附录 DT.2 noted this without developing it. Routing is a binary decision, and on
the cleaner side it is **deterministic**: `has_critical_issues` is computed from
the profile, so for fixed data and a fixed column scope it does not vary between
trials. The only source of trial-to-trial variation is the intent parse, which
can move the column scope.

**Arm A therefore runs at k = 2**, not at the k chosen for the net-gain arms.
Per-arm k values and this reasoning are frozen here. At N = 41 this saves about
$16 against a uniform k = 5; at N = 74, about $29.

#### 7.3.4 The confusion matrix is partly an authoring choice, and is pre-registered as such

The routing predicate reads `query_type`; the label is `corpus_backed`. **Both
are set when the case is authored.** If every `diagnostic`/`exploratory` case
were authored as corpus-backed and every `descriptive`/`comparative` case as
not, precision and recall would both be 1 by construction, and 实验 2 would
report nothing but the authoring convention.

**The intended joint distribution is therefore fixed here**, as the sixth
quantity in the case-authoring spec (§7.4):

- **About one quarter of cases are deliberately discordant**, split evenly
  between the two directions:
  - `corpus_backed = true` with a `descriptive`/`comparative` query — a question
    about an event-affected period that the criterion will skip
    (**expected FN**);
  - `corpus_backed = false` with a `diagnostic`/`exploratory` query — an
    explanatory question with no corpus behind it (**expected FP**).
- The remaining three quarters are concordant.

**Without the discordant quarter there are no off-diagonal cells and the matrix
is a tautology.** Exact counts are fixed once N is known (清单 (5)).

### 7.4 Sample eligibility — fixed at case-authoring time, never recomputed

These three fields are written when the case is authored, from frozen inputs,
**without running the system on the case**:

- **`cleaning_relevant`** — determined by a deterministic script from the frozen
  `data/demo/` files: does this case's data slice contain at least one injected
  defect class (missing revenue, negative revenue, outliers)?
- **`event_id`** and the derived **`corpus_backed`** — the case is explicitly
  linked at authoring time to an event in the frozen `event_script.yaml`, or
  declared unlinked; `corpus_backed` follows from that event's `corpus_evidence`
  / `event_type`.
- **`cleaning_material`** — determined by the same deterministic script from
  the same frozen `data/demo/` files: does the defect **change the answer to the
  question this case asks**? Computed by running the case's own ground-truth
  function twice — once on the raw slice, once on the slice repaired by a fixed,
  declared rule (drop null-`revenue` rows, drop negative-`revenue` rows, clip to
  the declared IQR bounds) — and comparing the two results against the **1%
  scoring tolerance** used by `_numbers_match`. True iff they differ by more
  than that.

**`cleaning_relevant` and `cleaning_material` are different fields answering
different questions, and are used in different places:**

| Field | Question | Used by |
|---|---|---|
| `cleaning_relevant` | Is a defect **present** in this slice? | H1 / H3 sample eligibility (§7.2) |
| `cleaning_material` | Does the defect **change this case's answer**? | 实验 2's "should have fired" label (§7.3) |

The wider field stays bound to H1 deliberately: the trade-off note below says a
case whose data is dirty but whose cleaner criteria do not fire must stay in
H1's sample, because that dilution is real and **"that is precisely what 实验 2
measures; it must not be quietly removed from 实验 1's sample."** Narrowing H1
to `cleaning_material` would remove exactly those cases.

The narrower field is what 实验 2 needs, because scoring routing against
`cleaning_relevant` would mark the system wrong for a judgement that is
defensibly right: the frozen data carries 3% missing `revenue`, and the
cleaner's criterion deliberately does not fire below 10% (附录 B.1: the threshold
is set so that only "坏到一半以上" triggers a lossy modification). Labelling
those cases "should have fired" would score correct restraint as a miss.

**All three are pre-treatment.** None requires running the system on the case,
so none reintroduces the post-treatment selection bias this section was written
to avoid (附录 EC.1). All are frozen with this pre-registration and are **not
recomputed at reveal time**.

**A deliberate trade-off, stated rather than hidden.** `cleaning_relevant` is
defined by **whether the defect is present in the data**, not by **whether the
cleaner actually acted**. These are not equivalent: a case whose data is dirty
but whose cleaner criteria do not fire stays in H1's sample and shows up in arm C
as "cleaning did not help". This is intentional for two reasons. Defining
eligibility by whether the cleaner acted would require running the cleaner on
holdout cases before the reveal — which is itself an observation of the holdout,
and would burn it (附录 EC.1). It would also be **selection on a post-treatment
variable**, which is a bias in its own right. Defining it by a data property
fixes the sample before treatment assignment. The cost is that H1's effect size
is diluted by "should have fired but did not" cases — **and that is precisely
what 实验 2 measures; it must not be quietly removed from 实验 1's sample.**

### 7.5 Stratified reporting is not stratified testing

附录 CP.5 #4 requires corpus-backed and corpus-free events to be reported
separately. **Reporting and testing are separated here deliberately**, because
folding strata into the confirmatory set silently doubles it:

- **H2's restriction to `normal` events is a sample definition, not a stratified
  test.**
- **H1 and H3 are run on all eligible cases and are not stratified for testing.**
  Stratified figures are reported and labelled descriptive.
- **Any stratified comparison not listed in 7.2 is exploratory.**

### 7.6 Multiplicity and the decision rule

- **Two confirmatory tests.** H1 and H3. **H2 is exploratory**: the metric it is
  bound to (`event_attribution_accuracy`) has no validated scorer yet (§7.2.1).
  Bonferroni: α = 0.05 / 2 = 0.025, i.e. report **97.5% paired bootstrap
  confidence intervals** using the project's existing `paired_bootstrap_ci`.
  - Both remaining confirmatory tests are bound to `factual_accuracy`, computed
    by a pure function against frozen ground truth
    (`src/eval/metrics.py::score_factual_accuracy`, no LLM call). **The
    confirmatory core therefore does not depend on judge validity.**
  - Had H2 remained confirmatory the multiplier would be 2.394 (98.3%) rather
    than 2.241 (97.5%); every MDE in 附录 CA's table improves by about 6% as a
    result. This is a **consequence** of dropping a test, not a reason to drop
    it.
  - **附录 CA's MDE table is optimistic on two further counts** and must not be
    read as written. It uses the 2σ multiplier (≈95%), not the Bonferroni one;
    and it uses n = 41, the whole retired holdout, while **no hypothesis is
    tested on all of it** — H1/H3 run on the `cleaning_relevant` subset, H2 on
    `normal`-event cases (§7.2). At σ_hetero = 0.05 and k = 5 the middle
    cell moves from CA's 0.0196 to 0.0220 at 97.5%, and to 0.0307 if the
    eligible subset is half of 41. **The subset sizes, not N, are what must
    clear the threshold** (清单 (5)).
- **Directions declared in advance**: H1 and H2 are expected positive; H3 is
  two-sided with no expected direction.
- **"The subsystem is worth it" requires BOTH**: the interval excludes zero,
  **and** the point estimate exceeds the project's standing noise floor of
  **0.028** (附录 CA). Significance alone is not sufficient: 附录 CA's MDE at
  k = 5 is ≈ 0.020, **below** the noise floor, so a statistically significant
  difference can be one that is practically indistinguishable from noise.
- **Everything else is descriptive**: the other seven pairwise comparisons and
  the other ten metrics are reported with values and intervals, carry **no**
  significance claims, support **no** conclusion, and are labelled `exploratory`
  in the report.
- **Cost and reliability metrics** (`token_cost`, `total_latency`,
  `retry_count`, `error_rate`) are always reported in their own table and are
  never combined with quality metrics.

### 7.7 What this section does not solve

- **H2 needs events that do not exist yet.** No case in `tests/eval/test_suite.json`
  carries an `event_id`, and new `normal` events must come from a new event-script
  version, which re-renders `data/demo/` and therefore moves ground truth (附录 FI).
  The scorer is also calibrated on few events: dev has 4 `normal`.
- It cannot prevent the primary metrics being changed after the fact. Its only
  defence is that it is frozen before the reveal and that the frozen version is
  traceable.
- **The decision rule in 7.6 assumes the new holdout's size N supports 附录 CA's
  k and MDE derivation. N is NOT YET DETERMINED** (清单 (5); σ_hetero was never
  obtained — 附录 DV). **If N turns out materially smaller than the retired
  holdout's 41, 7.6 must be recomputed rather than carried over.**

> Text authored by the orchestrator in ECOSYSTEM_INTEGRATION_PLAN.md 附录 ED.2;
> transcribed verbatim.

---

## 8. Judge validity: the threshold, and what happens if it is not met

### 8.1 The threshold is relative to the human reference's own reliability

A machine-vs-human agreement statistic is bounded by how reliably the human
reference labels the same material twice. Measured on the non-predictive 20
(`data/annotations/human_pass1_vs_pass2_nonpredictive20_agreement.json`,
n = 20):

| Metric | Human test-retest QWK | Spearman | Exact-match rate |
|---|---|---|---|
| `answer_relevance` | **0.4032** | 0.4919 | 0.35 |
| `groundedness` | **0.7167** | 0.7648 | 0.45 |

**Pre-registered rule.** A judge is treated as validated for a metric when its
QWK against the frozen human reference is **at least the human's own test-retest
QWK for that metric**. The rule carries no free parameter: meeting it means the
judge is not statistically distinguishable from a second pass by the human
annotator.

This **replaces** the previous absolute bar of QWK ≥ 0.60, which ignored the
reliability of the reference it was measured against. The replacement is
**stricter for `groundedness`** (0.7167 vs 0.60) and **looser for
`answer_relevance`** (0.4032 vs 0.60). It is a recalibration in both
directions, not a relaxation.

**Provenance**: user adjudication, 2026-10-28. **Disclosure**: the rule was
formulated after the machine scores in 附录 FB/EZ were known. The specific form
"100% of the human ceiling" was chosen *because* it admits no tunable fraction;
fractional variants were examined, and a bar at 65% of the ceiling would flip
`groundedness` to passing. That is precisely why no fraction is used here.

**Status under this rule, on the evidence as of 附录 FB/EZ:**

| Metric | Judge | QWK (strict / lenient) | Bar | Status |
|---|---|---|---|---|
| `answer_relevance` | Qwen3.8-Max | 0.726 / 0.637 | 0.4032 | **Met** |
| `groundedness` | GPT-5.6 Sol | 0.495 / 0.475 | 0.7167 | **Not met** |

**Both the judge QWK and the human ceiling are estimated at n = 20 and must be
reported with bootstrap CIs** (附录 FD, 第 8.7 Q7). The ceiling is itself an
estimate; a wide CI on the ceiling widens the uncertainty on this verdict.

### 8.2 What is reported is the number, not a verdict

Every 阶段 4 report of a judged metric carries, in this order: the point
estimate, its bootstrap CI, the human test-retest ceiling for that metric, and
whether the bar in 8.1 was met. **A bare PASS/FAIL is not an acceptable
presentation** — a reader cannot tell a near miss from a gross one, and the
distance to the ceiling is the part that carries the meaning.

### 8.3 Pre-committed consequences when the bar is not met

These apply per metric, are fixed here in advance, and are **not** contingent on
what the 阶段 4 numbers turn out to be.

1. **The hypothesis bound to that metric drops to exploratory.** Since
   2026-11-01 `groundedness` binds no hypothesis, so missing the bar affects its
   descriptive reporting only. H2 is governed by its own scorer bar (§7.2.1); until
   that is met **H2 is exploratory** and the confirmatory set is **H1 and H3
   only**. Both are bound to `factual_accuracy`, computed by
   a pure function against frozen ground truth
   (`src/eval/metrics.py::score_factual_accuracy` — no LLM call), so the
   confirmatory core does not depend on judge validity at all.
   - Multiplicity is recomputed accordingly: two confirmatory tests, Bonferroni
     α = 0.05 / 2 = 0.025, i.e. **97.5% paired bootstrap CIs** (§7.6).
2. **The metric's paired difference is still reported**, labelled exploratory.
   **Only the difference is interpreted; the absolute level is not.** Reported
   alongside it, mandatorily: **the per-arm distribution of claim counts**, so
   that a difference arising from a shift in the denominator rather than in
   groundedness is visible rather than inferred.
3. **The machine score is also reported against both frozen human reference
   bounds** (`human_reference_nonpredictive20_strict_lower.json` and
   `..._lenient_upper.json`, 附录 EW). Only differences that hold under **both**
   bounds are stated descriptively; a difference holding under one bound only is
   reported as such and supports nothing.
4. **Post-hoc validation on the reveal data itself.** After the single reveal,
   20 holdout reports are sampled at random and labelled by hand, and the
   machine-vs-human agreement on that sample is reported. This costs no
   additional API calls and is not a second reveal: no run is repeated, no trial
   is added, and the result is already locked.
   **It is pre-committed here that this check will not be used to revise any
   reported conclusion**, under any outcome. It is a limitation report, not a
   gate — otherwise it would be a back door to "look, then adjust".

### 8.4 What this section does not solve

- It does not make an uncalibrated judge trustworthy. It makes the confirmatory
  core independent of the judge, and it makes the uncertainty legible.
- The ceiling is measured on the same non-predictive 20 that has now been used
  to compare four judge models (附录 FD.4's warning about selection degrees of
  freedom). **The remedy is to enlarge the calibration set, not to keep testing
  variants on the same 20**; enlarging it is dev-side and touches neither
  holdout.
- **The calibration reports may predate the 2026-08-17 data re-render.** C07's
  report states North 2022 revenue as $439,901.01 where the current fixture is
  $1,784,443.68 — a ratio of about 4.06, against the TB5 row-count change of
  12,000 → 48,960 (4.08). This does not affect groundedness itself, which is
  judged against the findings shown rather than against current fixtures, but it
  means **the judge is calibrated on reports produced by an older pipeline over
  older data**, and the transfer of that calibration to the new holdout is an
  assumption rather than a measurement. Confirming the generation provenance of
  these 20 reports is tracked in the pre-reveal patch batch.
- It does not establish that `factual_accuracy` is a *good* metric, only that it
  is a deterministic one. Its ground truth is reproducible
  (`scripts/compute_ground_truth.py`, re-verified 2026-10-24: 76 checked,
  0 mismatch, 0 error, 24 skipped), but reproducibility is internal consistency,
  not evidence that each derivation formula is correct (附录 BN.2/BN.3).

### 8.5 Offline re-scoring: free before the freeze, forbidden after the reveal

Judge output is stored as a per-claim record
(`data/annotations/answer_key_*.json`), so **any scoring rule can be re-applied
to it with no API call and no re-run**
(`scripts/regrade_groundedness_offline.py`). This is a capability, and therefore
a back door: **none of the existing guards sees it.** The rag split guard
refuses a second run against a test split; nothing refuses a second *scoring* of
the first run's output.

**The governing principle:**

> A paid run is needed **only to obtain information the judge has never
> emitted**. Everything downstream of emitted information is free and may be
> recomputed any number of times — before the freeze.

**Corollary, which §8.6 is built on: tag, do not filter; emit, do not decide.**
A decision made inside the judge call costs another paid run to revise. The same
decision made offline, over an emitted field, costs nothing.

**Before the freeze — permitted and preferred.** Candidate scoring rules are
compared offline on stored output. **Limit:** offline re-scoring cannot invent
information the judge never produced. Section attribution and failure-report
identification can be **approximated** from claim wording; the three-way
`verdict` **cannot be recovered at all**, because what is stored is a boolean.

**On the dev split during the σ run — permitted, counted.** Each offline
re-scoring pass is a selection-degree-of-freedom expenditure in the sense of
附录 FD.4, recorded with what was chosen and why.

**After the reveal — forbidden, under every outcome including an inconclusive
one.** The scoring rule is frozen before the reveal, **identified by the commit
hash of the code that implements it**, and each primary metric is computed once,
by that frozen code.

**The line:** statistics computed *alongside* a frozen number are permitted
(per-arm claim-count distributions, per-section unsupported rates, §8.3(4)'s
post-hoc agreement check). **Changing how a frozen number is computed is not.**

### 8.6 Judge output specification, and scoring rules applied to it

> **Naming**: this is **the pre-registration's §8.6**. It supersedes the
> *handoff's* 第 8.6 节 (`~/MAEDA/docs/handoff_maeda_to_subsystems.md`), which
> proposed the same three changes in a single undifferentiated layer. Cite both
> in full; 附录 AF and 附录 C/CA record what label collisions cost here.

The handoff's 第 8.6 节 proposed three changes: (a) restrict the judge's input
to three sections, (b) make `verdict` three-way, (c) add a two-value
`claim_type`. **(a) and (c) are decisions taken inside the judge call and are
therefore re-split here into an emitted field plus an offline rule.** Only (b)
is genuinely new information the judge must produce.

#### 8.6.1 Layer 1 — what the judge must emit (requires the paid run)

Per claim, five fields. **The judge filters nothing and weights nothing.**

| Field | Content |
|---|---|
| `claim` | the assertion, terse |
| `section` | the report heading it was taken from, verbatim |
| `verdict` | `supported` / `contradicted` / `not_addressed` |
| `references` | which of this case's `expected_metrics`, `expected_dimensions` or `ground_truth` keys the claim concerns, or null |
| `evidence` | the supporting or refuting text, or the reason none was found |

`section` and `references` are **offline-auditable**: the tagged section must
contain the claim's text, and the named field must appear in the claim. This is
the property `claim_type` never had, and the reason it is replaced.

**The full report is supplied.** Nothing is withheld from the judge; section
scope becomes an offline filter (8.6.2), so the coverage holes that input
restriction would have created — cleaning disclosures in Data Quality Notes and
Automated Caveats (U.7/R.2's forced disclosure chain) — do not open.

#### 8.6.2 Layer 2 — scoring rules, applied offline

Candidates, applied cumulatively to Layer 1's output:

| Step | Rule |
|---|---|
| 0 | baseline: `supported_count / total_claims`, all claims, boolean verdict |
| 1 | **restrict to the human-read sections** (Executive Summary, Key Findings, Analysis Details) via `section` |
| 2 | **exempt honest failure reports**: a `not_addressed` claim that reports a tool or step failure is not a deduction (§9 R2.2) |
| 2a | **remove chart claims** (§9 R5): a claim about a chart, plot or other visual enters neither score. Applied offline by a fixed pattern over `claim` (`CHART` in `scripts/regrade_incremental_offline.py`), frozen with the rest of the rule by commit hash |
| 3 | **three-way verdict**: `contradicted` always deducts; `not_addressed` deducts, and is counted and warned separately as possible evidence-coverage failure |
| 4 | **centrality weighting** by `references` |

**Step 2a was added after steps 0–2 had been computed on the calibration set**
(user adjudication, 2026-10-31, 附录 FG). The rule itself predates that result
(§9 R5, 2026-10-30) and has no tunable parameter, but its placement in this table
was decided after seeing that steps 1 and 2 did not close the gap, and after a
post-hoc diagnostic on the same 20 cases (附录 FF.3). Its effect there is
therefore not evidence for it; only data the rule has not seen can be.

**The stopping rule is pre-registered, not chosen afterwards.** QWK against the
human reference is computed with bootstrap CI at each step. **The frozen rule is
the earliest step after which the next step's QWK CI lower bound does not exceed
the current step's point estimate** — the curve is treated as flat unless the
improvement clears the noise. Steps beyond it are not applied, and the fact that
they were evaluated and rejected is reported.

**This is why the `claim_type` / centrality question needed no prior
adjudication.** It is no longer an argument about whether the axis is right; it
is step 4 of a curve, kept only if it earns its place. The C07 diagnostic
already suggests it may not: steps 1 and 2 alone close an estimated 60% of that
case's machine-human gap (0.586 → ≈0.818 against a human 0.75), **with no
weighting of any kind**.

#### 8.6.3 Order, and what each step costs

| Step | Cost |
|---|---|
| Offline re-score existing output: baseline → 1 → 2 | **zero** |
| `pandas_groupby` rendering fix (附录 FD.3) + rubric written down (§9) | **zero**, both prerequisites |
| One judge run emitting all five fields over the calibration set | **about $7–10** |
| Offline curve over the new output, steps 0–4 | **zero** |
| Freeze the chosen rule by commit hash | **zero** |

**The paid run buys emitted information, not a scoring decision.** If the
offline pass over existing output already closes the gap, the run's scope
narrows to obtaining `verdict` alone.

---

## 9. Human labelling rubric

**Status**: written down 2026-10-30, **after** the labels it describes were
produced (2026-08-25) and after machine scores for the same cases were known.
附录 FD 第 8.8 names that order as a fault: a rubric reverse-engineered from
memory once the scores are visible is not a constraint on anything.

**What makes this version better than the one 8.8 warned about, and where it is
still weak.** It is derived from the annotator's own **contemporaneous written
notes** on each case
(`data/annotations/human_labels_dmxapi_calibration20_nonpredictive_relabel_decoded.json`),
not from recollection: every rule below cites the case notes that support it.
That is stronger evidence than recall and weaker than a rubric written in
advance. **It does not retroactively make the 20 labels pre-registered.**

**What it is for.** Two things, and not a third:

1. the scoring rules in §8.6.2 are written to match it, not the other way round;
2. it makes the reference auditable — a label it cannot explain is a recorded
   gap (R6);
3. **it is not used to relabel.** See R6.

### R1. What text is judged

**Executive Summary, Key Findings, and Analysis Details only.**

Later sections — Recommendations, Data Quality Notes, Automated Caveats,
Visualization — are **not read** by the annotator (direct answer, 2026-08-25,
附录 FD.2). The annotation page renders them; the annotator does not use them.

**This is a property of the reference, not an instruction to the judge.** The
judge is shown the whole report and tags each claim with its section (§8.6.1);
the restriction is applied offline (§8.6.2 step 1). Measured consequence on
C07: of 12 machine deductions, **at least 3 came from sections the annotator
never read**.

### R2. Groundedness: when a statement is deducted

A statement is deducted when the report **asserts something the evidence does
not license**. Four observed forms.

**R2.1 Contradicted by the evidence.** A figure, ordering or characterisation
the analysis results refute.
*C02: stated conversion rates "wrong". D05: "March recorded the highest order
volume … January the lowest" — "this claim is wrong". E23: "most customers have
been with the company for 1 to 5 months" — wrong. C07: 2022–2024 historical
actuals described as "projected", three times, in the Executive Summary and both
Key Findings.*

**R2.2 False claim of inability.** The report says it could not compute
something the evidence shows it could.
*C16: "the analysis did not yield specific revenue figures for the West and East
regions" — "wrong. it DID yield". D05: "not possible due to the limited data
available" — "however, the data IS available".*

> **Not a deduction when the report honestly reports a tool or step failure.**
> "due to a failure in analysis steps" is an accurate statement about what
> happened; the resulting gap in the answer is an `answer_relevance` deduction
> (R4), not a groundedness one. *Added 2026-10-30 from the C07 review; it
> changes that case's second flagged item from a deduction to a non-deduction,
> and it reached **4 of C07's 12 machine deductions**.*

> **Severity depends on whether the denied content is delivered elsewhere in the
> report.** *C07 denies being able to compute a growth rate while listing every
> figure needed — peripheral. C16 denies yielding the requested regional figures
> and does not provide them — core.*

**R2.3 Unlicensed inference.** A causal or evaluative conclusion the evidence
cannot support, **even when nothing contradicts it**.
*C20: t-test numbers correct but the direction reversed, then read as "higher
charges 导致 churn / pricing influences retention" — "这些都没有被表格支持".
D07: "efficiency" / "effectiveness" / "does not correlate" — "没有直接由两张汇总
表支持". DG12: "后半部分引入了多个没有数据支持的结论". DG03: "strong revenue
potential". C02: "Social channels" did not "show promising conversion rates".*

**This is the single largest category in the recorded notes**, and the one a
two-value `empirical | interpretive` split would have mis-handled: these are
interpretive claims, the evidence does not refute them, and the annotator
deducted for all of them.

**R2.4 Superlative from a partial view.** "Highest" or "lowest" asserted when
the visible evidence does not cover the whole population.
*C08: "not the highest cuz we cannot see the whole data". C04: "since we can't
see the whole picture, we can't get the conclusion that Product 2 generates the
highest margin".*

**R2.5 What is NOT deducted: a claim the evidence merely does not mention.** No
recorded note deducts on that basis.

**This is the largest divergence from the machine judge**, which recorded
`no evidence found` on **193 of its 194** unsupported verdicts (附录 FD.2) —
including, on C07, the three `projected` claims that are in fact R2.1
contradictions. The judge identified the right problem and could only express it
in the wrong category. §8.6.1's three-way `verdict` exists for this.

### R2.6 Do not complete the report's reasoning on its behalf

Judge only what the report **states**. When the question asks for a comparison,
a trend or a conclusion and the report lists the inputs but never states it,
that is a **failure to answer**; it does not become an answer because the reader
can perform the step mentally.

**Operational check**: before crediting a conclusion, locate the sentence that
states it. If there is no such sentence, the conclusion was not made.

**Why this is written down.** The annotator identified the risk in their own
practice (2026-10-30): when the arithmetic is easy enough to do at a glance, the
gap closes unnoticed and the report is credited for a step it skipped. The bias
has a **fixed direction** — it inflates reports that list raw figures without
synthesising them — and that is precisely the distinction between a data dump
and an analysis.

**It corrupts the reference, not the judge.** A judge that correctly deducts for
a missing synthesis would be scored as *disagreeing with the human*. Any
diagnosis of "the judge is too harsh" has to rule this out first.

**Which axis it belongs to**: a missing required conclusion is an
`answer_relevance` deduction, **not** a groundedness one. If the report states
nothing unsupported, groundedness is unaffected. **The same defect is never
penalised on both axes.**

### R3. Groundedness: how the band is chosen

The band reflects **whether the error reaches the answer to the question**, not
how many statements were flagged.

| Band | Criterion |
|---|---|
| **1.00** | Nothing flagged |
| **0.75** | Flagged statements are peripheral; the core answer stands |
| **0.50** | A real error reaches the answer; part of the answer survives |
| **0.25** | The requested content is substantially wrong or unsupported |
| **0.00** | The central conclusion is fabricated, or the data is fundamentally misread |

**Count is not the criterion.** *C07 carries two flagged statements and scores
0.75. DG01 carries one and scores 0.00, because its central conclusion — "a
significant drop compared to previous quarters" — rests on a comparison the data
cannot contain.*

This is the functional the machine's equal-weight `supported / total` cannot
reproduce, and why §8.6.2 treats weighting as a candidate step to be earned
rather than an assumption.

### R4. `answer_relevance`: deductions and bands

Relevance asks **whether the report addressed what was asked**, independent of
whether what it said was supported.

**Deducted for:**

- **an omitted required element** *(D01: "doesn't mention south region"; D11:
  "didn't mention Widget A")*
- **an omitted required metric** *(D08: "didn't mention 'average tenure' at
  all")*
- **the required operation not actually performed** *(C07: "doesn't really
  compare revenue growth")*
- **partial coverage** *(D05: "only show monthly order volume for 5 months")*
- **answering a different question** *(DG12: 用户问的是为什么 Gold tier churn
  更少,"报告基本没有回答,只是在解释'分析不了',然后转去分析 plan")*

| Band | Criterion |
|---|---|
| **1.00** | Fully addressed |
| **0.75** | One required element omitted |
| **0.50** | The required operation was not really performed, or coverage is partial |
| **0.25** | The main ask was not answered, or a different question was answered |
| **0.00** | Unrelated to the question as asked |

**Lower stakes than R2/R3**: `answer_relevance`'s judge meets the §8.1 validity
bar (QWK 0.637–0.726 against a human test-retest ceiling of 0.4032), so this
rubric is recorded for consistency and because R2.2 and R2.6 route deductions
here. It is not the subject of §8.6's redesign.

### R5. Out of scope, stated rather than implied

Neither metric covers these. Each is also declared out of scope in §8.3.

- **Chart correctness.** The annotation page contains no rendered chart (no
  `<img>`, no embedded image data), so it is **structurally absent** from the
  reference. Chart claims are identified offline (§8.6.2 step 2a), counted and
  reported descriptively; they do not enter either score.
- **Citation attribution.** This means which source file a report cites. It is
  **not** event attribution, whether the report names the right cause of an
  anomaly, which is H2's primary metric (§7.2.1). Whether a cited source is the
  right one is not judged, and today it could not be: `insight.sources` is never produced by the
  model (`INSIGHT_GENERATOR_SYSTEM`'s output schema has no such field), and
  `src/agents/insight_agent.py:199-203` fills it with the run's top three
  retrieved filenames, identically on every insight. **Removal is adjudicated
  (路线 A, user, 2026-10-26) and scheduled in the pre-reveal patch batch; it must
  land before the calibration re-run** so that calibration and reveal score the
  same report shape (§3.0). **Measured scope: none of the 20 stored calibration
  reports cites any source file** (zero hits for a filename pattern across all
  20), so the field is already ignored by the report writer and its removal is
  not expected to alter report text. *A bound, not a proof.*

### R5.1 Event attribution labels (§7.2.1)

Confirmed by the user 2026-11-02 (附录 FL). Labelled separately from R2–R4, one
label per report, with the arm hidden. **The whole report is read, by the
labeller and by the judge alike**; R1's three-section restriction does not apply
here, because a reading-scope difference between the two was a source of
disagreement in the `groundedness` calibration. `EVAL_EVENT_ATTRIBUTION_SYSTEM`
carries the same definitions.

| Situation in the report | Label |
|---|---|
| Names the documented cause, in any wording, including hedged ("possibly because of ...") | `correct` |
| Names the documented cause plus a second, minor one | `correct` |
| Two causes side by side with no main one, the documented cause among them | `correct` |
| Two causes side by side with no main one, neither of them the documented cause | `wrong` |
| A different cause is the main one, even if the documented cause is also named | `wrong` |
| A cause is asserted in any section, even if another section only suggests checking causes | judged by the assertion |
| Decomposition only ("driven by Widget A", "fewer units sold lowered revenue") | `none` |
| Size, timing or location only | `none` |
| Only suggests checking a cause ("verify whether stock ran out") | `none` |
| Three or more guesses side by side with no main one | `none` |
| No documented cause exists, and the report asserts a specific cause | `wrong` |
| No documented cause exists, and the report says the cause is unknown or that there was no clear effect | `none` |

**`distractor` events have no documented cause.** Their corpus describes an event
that left no trace in the data; a report crediting it with an effect is `wrong`.
A report that only says the event had no clear effect is `none`; one that goes on
to explain *why* there was no effect ("the campaign lacked a product focus, which
may have contributed") asserts a cause and is `wrong`.

**Per label the labeller also records**, none of which is scored: for `correct`
or `wrong`, the sentence that decided it; an "uncertain" flag; optional notes.
They exist so that a disagreement with the judge can be traced without
relabelling.

**Procedure.** Six reports outside the 60 are labelled first as a dry run. The
rubric may change only after the dry run and is frozen before the first of the
60 is labelled. Pass 2 relabels all 60 at least 7 days after pass 1 ends, in a
new order, without access to pass 1.

**Dry run done and rubric frozen, 2026-11-02 (附录 FM).** The user labelled
D01–D06; the drafter's independent reading agreed on 5 of 6. The one difference
(D04, a hedged "likely contributed" cause in Key Findings while Recommendations
listed three things to check) was a labeller slip, confirmed by the user, not a
rubric gap. The dry run added two rows, both already the drafter's reading before
the labels: two non-documented causes side by side → `wrong` (D02), and an
assertion anywhere decides over a check-list elsewhere (D04). The distractor
paragraph gained the "explaining why there was no effect" sentence (D05). **No
row may change from here until pass 2 ends.**

### R6. Self-consistency check, not relabelling

Per 附录 FD 第 8.8: this rubric is checked **against the existing 20 labels** to
find labels it cannot explain. **The 20 cases are not relabelled** — that would
consume the calibration set a second time and, worse, would encode the rubric's
drafter into the reference.

Labels the rubric cannot account for are **reported as rubric gaps**, not
corrected.

#### R6.1 Known gap: C07

Reviewed 2026-10-30. After R2.2's honest-failure-report exemption, one
groundedness deduction survives: historical data described as "projected", three
times, in the Executive Summary and both Key Findings. That is clearly not
peripheral, but **R3 does not resolve whether it reaches "a real error reaches
the answer" (0.50) or leaves "the core answer standing" (0.75)**.

**The label stands at 0.75, unchanged**, because (a) 8.8 prescribes a
consistency check rather than relabelling; (b) the corrections this review
produced reach other cases, so applying them to C07 alone would make the
reference non-uniform; (c) the annotator was indifferent between the two bands,
so any change would record the drafter's reading rather than theirs.

**The ambiguity is itself the finding**: the reference has limited resolution in
this band region, and that limit is reported rather than papered over.

---

## 10. Amendment 2026-11-05 (附录 GB): the v2 event script, and what it discloses

Appended; nothing above is rewritten. Where this section conflicts with earlier
text, the earlier text stays as written and this section says which sentence it
supersedes. Decided in 附录 FQ, FR, FS and delivered in 附录 FZ; received here in
附录 GB. No test-split event content appears in this section, only counts and
hashes.

### 10.1 What changed in the data

`event-script-v2` (rag local tag, hash chain in 10.6) adds 18 `normal` events to
the 12 v1 events: 30 in total (`normal` 25, `distractor` 3, `unexplained` 2),
dev 13 and test 17. The v2 dev events (6) are public in the tag's
`event_script.yaml`. The v2 test events (12) are in a local-only sealed file in
rag and are not readable from this repository. `data/demo/sales_data.csv` is now
the v2 render (sha256 `b863b60173f3c0bf30c4582738b85ba3304a7a212794d66ab14aa35dce422719`,
48960 data rows), copied by hash from rag. Every earlier mention in this file of
the 12-event script, of `sales_data.csv` hash `a0da755e…`, or of "the 3 `normal`
events now in the test split" describes the v1 state.

### 10.2 H2 test pool, and the exception to "no case before N" (附录 FQ.2 item 1)

**Pool definition.** The H2 test pool is the events with `split == test`,
`event_type == normal` and `batch == v2`: **12 events**. The 3 v1 `normal` test
events are **excluded** from the H2 pool. They keep `split == test` and remain
part of the TB5 retrieval holdout, whose reveal is unspent (§1.7). The reason is
the one given in 附录 FS.2 item 1: those three events' probes and corpus define
that holdout, and the H2 probes for them would have to be added to events whose
text a retrieval reveal depends on. The dev `normal` events are 10 (v1 4, v2 6).

**Exception.** The open-items table at the top of this file says no case may be authored into
the new MAEDA holdout before N is fixed. That rule was written for the H1/H3
case suite and stays in force for it. The H2 pool is not a case suite: its
independent unit is an event, test events can only be written when G3 writes
them into the data, and the dev pilot comes after G3, so by the time a pilot
could inform N the pool is already fixed. The pool size was therefore set at G3
from the planning value in 附录 FI (about 12, 附录 FR.6.2), and the dev pilot can
only **check** that size's power, not set it. If the pilot says 12 is
underpowered, the only options are to accept lower power and report it, or to
write a v3 and render again; neither is a default chosen here. The v2 events were
written and frozen before any arm's report existed for any v2 event (the only
reports so far, 附录 FM, cover v1 dev events), so the exception does not open a path
for results to shape the pool.

### 10.3 The data signal of test events is visible in the CSV (附录 FR.4 item 6)

Each test event's data signal (region, window, magnitude) is rendered into
`sales_data.csv`. Anyone analysing the data can find where the anomalies are.
What is sealed is their cause text, corpus, probes and ids. H2 measures whether a
system attributes an already-found anomaly to the correct cause in the corpus;
the location alone does not give the cause, so this is not judged to invalidate
the holdout. It is a disclosed residual, not a proven absence of effect. The
system under test sees the CSV in both arms; the disclosure matters for any
human or model that has seen both the CSV and the cause text.

### 10.4 The TB5 retrieval holdout is not comparable to v1-corpus numbers (附录 FR.6.1)

The TB5 retrieval holdout is still the same 8 derived entries from the 10 frozen
v1 probe digests (`test.json` unchanged byte for byte through G3). But the corpus
it is retrieved against grows at reveal from the 10 v1 event directories to a
larger library that includes v2 events, so there are more distractors. Retrieval
numbers measured on the v1 corpus (including every dev retrieval number taken
before G3) are **not comparable** to the holdout's reveal number. Read the §1.1
TB5 retrieval row with this sentence attached. The v2 test probes serve H2 only
and are not added to `test.json`.

### 10.5 R-arm retrieval store for the H2 dev pilot and σ runs

The H2 dev pilot and σ runs use `tb5_dev.db` (16 event directories: 10 v1 with
corpus plus 6 v2 dev; 28 documents, 38 chunks; embedding model
`text-embedding-3-small`, 附录 FZ.3). They must **not** use a store that contains
v2 test corpus (`tb5_full.db`, not built; it is built at reveal by the user's
command). Because the dev store and the reveal store differ, the dev-to-reveal
step has a corpus-size change on top of the data change (10.4); this is a known
difference between σ and reveal conditions.

The database file contains timestamps and is not byte-reproducible, so it is
recorded by **content hash** (附录 GA.2): sha256 over every chunk ordered by
chunk id, each as (document source path, chunk id, chunk text), together with the
embedding model name. Value recorded on 2026-11-05:
`bc7f85fc47cf0d57bdfd6c3bc17c009cef1339b5b4e5868ad6a090998238045e`. The source
path is the absolute path on this machine, so the hash is a same-machine
identity, not a portable one. `scripts/run_eval.py` now writes it into
`settings_snapshot.data_version` (`src/eval/data_provenance.py`) when
`MAEDA_RAG_DB_PATH` points at the file; it must be equal before and after a σ run
and a pilot.

### 10.6 Added to the EJ.2 freeze list (附录 FQ.2 item 3)

Recorded per run in `settings_snapshot.data_version` and checked by hand at
freeze:

| Item | Value now |
|---|---|
| event script git tag | `event-script-v2` (rag, local, not pushed) |
| manifest `sha256` = `public_sha256` | `46e7f23b489e35961af33317046e073e2740b5fff9608520acce6565948f3083` |
| `sealed_sha256` | `4d3eae4096a165cb763cfe614b70059233fa5ec2085ac3cc42b0767d1ed0ad9a` |
| `combined_sha256` | `efc9e5acabf3a636321add28b4dc600eadc39d227aa10cadf5dc81219a97db9e` |
| `slot_plan_sha256` | `359eb5ed04a9730a570dc491f2e5ed2b8f7027f046eb9cd5cc0e13eaee73dfd5` |
| `sales_data.csv` sha256 | `b863b60173f3c0bf30c4582738b85ba3304a7a212794d66ab14aa35dce422719` |
| `tb5_dev.db` content sha256 | `bc7f85fc47cf0d57bdfd6c3bc17c009cef1339b5b4e5868ad6a090998238045e` |

The `input_hashes` block in `provenance` already covers all of `data/demo/` and
the golden suite file; `data_version` adds the three items that live outside this
repository.

### 10.7 Consequences for earlier measurements in this file

- Fixtures: `tests/eval/test_suite.json` ground truth was recomputed on the new
  CSV (11 of 76 checkable cases changed, all sales-derived), and the golden-hash
  guard now pins the new CSV hash. The churn, marketing and e-commerce hashes did
  not change.
- Any number in this file derived from the v1 sales render (§2, §3, §4 and the
  FM report set) describes the v1 data. The 60 reports of 附录 FM were generated
  on the v1 render with the v1 `probe_queries` wording; the H2 dev cases now use
  the year-bearing `h2_probes`. Those reports remain judge-calibration material
  only (附录 FQ.2) and are not planning values for H2.
- C1b (附录 FQ.1) changed report generation: non-sum aggregations are rendered
  as `<col>_<func>`. Reports generated before this change and after it are not
  comparable (EJ.2); generate H2 pilot and σ reports only after it.

### 10.8 Correction to §10.2 (附录 GD, 2026-11-05)

§10.2 gave the wrong reason for excluding the 3 v1 `normal` test events from the
H2 pool. That sentence said adding H2 probes to them would alter the events the
TB5 retrieval reveal depends on. That is not the ruling, and it is contradicted
by 附录 FR.3, which keeps the v1 `probe_queries` byte-identical and puts the v1
test events' `h2_probes` in the sealed overlay. The sentence is left as written
above; this subsection replaces its reason.

**Correct reason.** The MAEDA side reviewed those three events' causes, segments
and windows item by item in 附录 AC (2026-08-16), and the MAEDA renderer later
read them. Under the user's confidentiality rule they are therefore no longer
blind to MAEDA, so they cannot serve as H2 test events. Sources: 附录 FQ.3 item
5, 附录 FR.1 item 5, 附录 FS.2 item 1, 附录 AC. The exclusion is a contamination
exclusion, not a technical limitation.

**The pool definition is unchanged:** `split == test`, `event_type == normal`,
`batch == v2`, 12 events. The three v1 events keep `split == test` and remain
part of the TB5 retrieval holdout.

### 10.9 Disclosure: median fill of missing revenue cells, and a standing data rule (附录 GG)

**Median fill (附录 GE.4 item 5).** The cleaner's `handle_missing` fills about 3%
of revenue cells with the full-table median. Inside an event window those filled
rows lose the event signal, which attenuates the window's lift by roughly 3% in
magnitude. This happens outside the clipping gate (附录 FW.2, GE), and it also
existed in v1. Only its magnitude was estimated; there is no exact per-event
attenuation figure. For an event whose window level is far from the full-table
median it can matter more than the average suggests. Both arms see the same
cleaned data, so it does not favour either arm, but it is a real signal loss in
the R arm's input.

**Standing rule (附录 GF).** Any `sales_data.csv` whose sha256 differs from
`b863b60173f3c0bf30c4582738b85ba3304a7a212794d66ab14aa35dce422719` requires
redoing 附录 GE.1 to GE.3 (the clipping re-check) before any run. A σ run, a
pilot or the reveal that finds a different hash stops until that is done.

### 10.10 Correction to §10.9 (附录 GI)

§10.9 contains a sentence added by its drafter that is not in 附录 GE.4 and does
not hold: "Both arms see the same cleaned data, so it does not favour either arm,
but it is a real signal loss in the R arm's input." It is **retracted**.

H2 reports run with the cleaner off in both arms: `scripts/generate_h2_reports.py`
line 6 ("Arms B and R only, both with the cleaner forced off") and line 80
(`settings.maeda_cleaner_mode = "force_off"`). Neither `handle_missing` nor the
IQR clipping touches the input of an H2 report, so the median fill and the
clipping gate do not apply to H2. The §10.9 disclosure applies to arms where
cleaning actually runs: the H1/H3 cleaner arms, and cleaner-on arms in σ runs.
The standing CSV-hash rule in §10.9 is unchanged and applies to every run, since
the same CSV is the input everywhere.

### 10.11 H1 ground truth is computed on a clean table (附录 GS; decided in 附录 GO.6)

**What changed.** Until now the sales ground truth in `tests/eval/test_suite.json`
was computed by `scripts/compute_ground_truth.py` on `data/demo/sales_data.csv`,
the table with injected defects (3% missing revenue, 2% duplicate rows, 1% revenue
multiplied by 10). On that basis a cleaner arm that cleans correctly is scored as
wrong and an arm that does not clean is scored as right (附录 GO.3: for example the
North region's revenue moves by −7.19% under rule-style cleaning, far outside the
1% tolerance), which is the opposite of what H1 claims to test. The user ruled
that truth is computed on the table before the defects are injected.

**The clean table.** The v2 render with the quality-noise block switched off and
every event kept: `data/truth/sales_data_clean.csv`, sha256
`6c95334422ccfefd4a3a4815944418dd995c3bcdd9513b3aa4db0be76f7d771d`, 48000 rows, no
missing values and no duplicates. Source: rag-framework
`artifacts/g3/sales_data_clean.csv` with receipt `render_receipt_clean.json`
(renderer `d557e52`, renderer sha256 `ed223821…`; 附录 GQ). Consistency with the
pipeline input was measured (GQ.2): the first 48000 rows of
`data/demo/sales_data.csv` equal the clean table in every column except revenue,
and in revenue differ only where the value is missing (1440 cells) or exactly ten
times larger (462 cells); the appended 960 rows are duplicates of clean rows. The
file sits outside `data/demo/` on purpose and is **a truth basis only**.

**The pipeline input does not change.** Every arm still reads
`data/demo/sales_data.csv` (sha256 `b863b601…2719`). Only the numbers the answers
are compared with change. `scripts/compute_ground_truth.py` verifies the clean
table's hash before computing anything.

**What H1 now means.** H1 asks whether the answer produced from the cleaned data is
closer to the truth on the clean table than the answer produced from the dirty
data. It does not ask whether cleaning reproduces the clean table. The cleaner
fills missing revenue with the full-table median, removes exact duplicates and
clips by IQR, so a cleaner arm will not match the clean truth exactly in every
case (clipping can cut real event peaks, 附录 GE); some of the distance between a
cleaned answer and the truth is the cleaner's own approximation error. That is part
of what is measured, and it is fixed here so it cannot be read as a definition
problem after the reveal. The scoring tolerance (1%) is unchanged.

**Ground truth changed for the existing cases.** All 12 sales cases with numeric
truth were recomputed (D01, D04, D10, D11, DG01, DG03, C01, C07, C11, C16, E01,
E02); the e-commerce, churn and marketing cases are unaffected. Revenue truths
fell by roughly 5 to 8%, units truths by roughly 2%, and in D10 the top sales rep
changed (rep 1011 on the dirty table, rep 1027 on the clean one), because the
multiplied rows had inflated rep 1011. E01 and E02 are discussed separately below.

**E01 and E02 are not settled by this entry.** E01 (row count) and E02 (IQR outlier
counts) ask about defects of the delivered file; their clean-basis truth (48000
rows; 135 revenue outliers) is a property of a table no arm sees. The treatment
(exclude them from the H1 sample, or rewrite them) awaits the user's ruling and is
recorded when made.

**Confidentiality note (附录 GQ.4).** The clean table lacks the multiplied rows
that make the dirty table noisy, so the data signal of the test events (region,
window, size) is slightly easier to detect in it than in the dirty table. The same
signal was already visible in the dirty table (§10.3); no new kind of information
is exposed (cause text, corpus, probes and ids stay sealed), but the disclosure in
§10.3 should be read with this in mind. Nothing in this repository reads the sealed
file.

**New cases.** 28 new dev sales cases (S01 to S28) with truths from the clean table
were added for the σ sample (`scripts/h1_sales_cases.py`). All are `split: dev`; no
holdout case was written or changed.
