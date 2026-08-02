# Judge Calibration — Eval v2 Step 3

Status 2026-08-02: **STEP 3 COMPLETE, A2 FIX APPLIED AND MEASURED.** The
groundedness judge's missed-superlative-claims bug is fixed
(`EVAL_GROUNDEDNESS_SYSTEM` in `src/config/agent_prompts.py`) and
re-scored against all previously-collected labels (not a blind holdout —
see the "A2" section below for that caveat stated in full). Mechanism
check: all 4 known Product-2-margin cases now correctly flag the claim as
unsupported. Aggregate: pooled n=100 groundedness QWK 0.141 → 0.185, a
real but partial improvement. Jump to [A2](#a2--fixing-eval_groundedness_system-2026-08-02)
for the full writeup, or read on for how the underlying problem was found.

Five independent labeling
sessions, 120 total labels (100 distinct cases + a self-consistency
replication of the original 20). Original 20-case pass: the user's
labeling of C01 surfaced 3 bugs (all fixed, corpus/export regenerated);
labeling C02 surfaced a 4th (the guardrail's own hallucination check has
the same truncation problem as bug #1, in a code path the fix never
touched — tracked as [roadmap.md #26](roadmap.md), deliberately not fixed
yet, see below). Results: `answer_relevance` QWK=0.676 (acceptable),
`groundedness` QWK=0.178 (weak). Batch 2 (20 new cases) replicated the
same direction and found it worse (`answer_relevance` QWK=0.551,
`groundedness` QWK=0.040), and pinned down a precise root cause: the
groundedness judge repeatedly misses false comparative/superlative claims
("X is the highest") even when the contradicting number sits in the same
visible table, though not always (one case, C07, shows it catching the
identical error pattern). Batch 3 (20 more) triangulated that finding a
third time (DG11's "Product 2 generates the highest margin," the exact
same claim batch 1's C04 and batch 2's DG07 had independently caught).
Batch 4+5 merged (40 cases, the largest single pass) made it a fourth
independent replication (E15), sharpening both patterns at double the
sample size: `answer_relevance` QWK=0.763 (the best of any pass),
`groundedness` QWK=0.040 with Spearman effectively zero (p=0.921).
**The self-consistency retest closes the loop**: relabeling the original
20 blind, two days later, gives human-vs-human groundedness QWK=**0.775**
("acceptable") against a human-vs-judge groundedness QWK that sits at
0.04-0.26 across *every one* of the five comparisons made. The human is
internally consistent; the judge disagrees with her consistently. That
rules out human noise as the explanation and confirms this is a real,
reproducible judge calibration gap. See [Results](#results),
[Results — Batch 2](#results--batch-2-20-new-cases-step-4-expansion-first-pass-2026-07-31),
[Results — Batch 3](#results--batch-3-20-new-cases-step-4-expansion-2026-08-01),
[Results — Batch 4+5 merged](#results--batch-45-merged-40-new-cases-2026-08-01),
and [Results — Self-consistency retest](#results--self-consistency-retest-original-20-relabeled-blind-2026-08-02)
below — see [docs/eval_v2_plan.md](eval_v2_plan.md) for the full 4-step
plan this is step 3 of, and the note there on what Step 3 being fully
labeled means for validating the eventual `EVAL_GROUNDEDNESS_SYSTEM` fix.

Labeling tools are kept deliberately separate so they don't contaminate
each other:
- **Self-consistency retest, pass 2** (same original 20 cases as pass 1,
  blind, fresh storage key) — `https://claude.ai/code/artifact/334c6970-b750-4b06-ad7e-1d6c2831c034`
  — labeled 2026-08-02, results below. **All labeling tools now complete.**
- **Batch 2** (20 new cases from Step 4's first expansion) — labeled
  2026-07-31, results below. Answer key:
  `data/annotations/answer_key_1785525212.json`.
- **Batch 3** (20 more new cases from Step 4's second expansion) —
  `https://claude.ai/code/artifact/c5db9af0-1be7-4e6f-b1a9-17f987c28114`,
  answer key at `data/annotations/answer_key_1785544815.json` — labeled
  2026-08-01, results below.
- **Batch 4** (20 cases from Step 4's third expansion) — standalone link
  (`https://claude.ai/code/artifact/736b0913-7db2-4390-86e8-0c0c051e0959`)
  **superseded, do not use** — merged into Batch 4+5 below at the user's
  request rather than labeled separately. Kept only for its answer key
  (`answer_key_1785548910.json`), reused (not re-billed) in the merge.
- **Batch 4+5 merged** (40 cases: batch 4's 20 + a new batch 5's 20, Step
  4's third and fourth expansions, suite now 100 cases, the user's
  original target) — one combined blind tool:
  `https://claude.ai/code/artifact/f6adb892-e2b8-4f2b-82cc-737cfee7a53f`,
  merged answer key at `data/annotations/answer_key_batch4_5_merged.json`
  — labeled 2026-08-01, results below.

## What this measures and why

Every prior step in this project improved the eval harness by inspecting
it, reasoning about it, or measuring its noise — but none of that answers
the one question an eval-role interviewer will ask first: **does the LLM
judge actually agree with a human?** Without this, "the judge scored it
0.8" is a number nobody outside the system can trust. This step produces
that evidence: a human independently rates the same (query, report,
findings) tuples the judge scores, and we measure agreement directly.

## Known limitation, stated up front: n=20, not the recommended 60-100

The original plan called for a stratified sample of 60-100 cases. The
golden suite currently has only **20 distinct queries** (Step 4 will expand
it to 100-200) — there is no larger pool to sample down from yet. Rather
than manufacture more samples by regenerating each of the 20 queries
multiple times (which would cost real API time/money and only diversify
generation noise, not add genuinely new content), the decision — made with
the user 2026-07-30 — was to run this pass on all 20 cases now, as a
pilot, and revisit sample size once Step 4 lands. This means:

- The confidence interval on QWK/Spearman below will be wide. A single
  extra disagreement can move QWK noticeably at n=20.
- All 20 cases are used (no stratified sub-sampling needed since it's the
  full available population) — the natural query_type and score-band
  distribution across those 20 is reported in
  `scripts/export_for_annotation.py`'s console output at export time,
  not chosen by any sampling procedure.
- This is a legitimate first pass, not the final answer. Report it as
  such — a small-n pilot with an honest CI is more credible than a larger
  claim without one.

## Tooling built

- **`scripts/export_for_annotation.py`** — loads all 20 cases from the
  replay cache (`logs/replay_cache/corpus.json`, built in Step 1/2), scores
  each with the *current* judge to build a hidden answer key, and
  generates a self-contained labeling HTML page with only the blind
  material (query, report, findings, RAG context — no judge scores
  anywhere in that file). Two outputs, kept deliberately separate:
  - `data/annotations/answer_key_<ts>.json` — the judge's scores +
    reasoning. **Never look at this before labeling** — seeing the judge's
    score first anchors the human rating and inflates measured agreement.
  - `data/annotations/labeling_tool_<ts>.html` — the blind labeling page,
    also published as a private Claude Artifact for easy access
    (favicon ⚖️, title "MAEDA Judge Calibration — Labeling").
- **`scripts/_annotation_template.html`** — the page template
  `export_for_annotation.py` fills in with the blind sample data. Verified
  interactively (band selection, progress tracking, prev/next nav,
  localStorage autosave across a real page reload, JSON export, dark mode)
  before handing off — see the browser-verification pass this session.
- **`src/eval/agreement.py`** + **`scripts/compute_agreement.py`** — QWK
  (primary; penalizes a 2-band miss more than a 1-band one, correct for
  ordered categories, unlike plain Cohen's κ), Spearman correlation, MAE,
  exact-band-match rate, and a confusion matrix, computed independently for
  `answer_relevance` and `groundedness`. Both continuous judge scores and
  the human's 5-band picks are discretized into 5 ordered bands
  (`discretize()`) before QWK; Spearman/MAE use the raw continuous values
  directly. Also supports `--labels2` instead of `--answer-key`, for the
  self-consistency check below. Verified end-to-end with synthetic
  smoke-test data (not real labels, deleted immediately after) before
  being handed off for real use.

## The rating rubric shown in the tool

Both dimensions use 5 anchored bands (1.0 / 0.75 / 0.5 / 0.25 / 0.0),
expanding on the judge's own 3-anchor rubric in `agent_prompts.py` — a
3-point scale left too much mass stuck in one middle bucket to discriminate
anything, which was itself one of the earlier audit's findings. The human
rates the same two things the judge is asked to rate, and is shown the
exact anchor text used for each band (visible in the tool, and in
`scripts/_annotation_template.html`'s `RELEVANCE_BANDS`/`GROUNDEDNESS_BANDS`
constants).

## How to actually do the labeling

1. Open the labeling tool (Artifact link, or the local HTML file).
2. Read the query, report, and findings for each case. Rate
   **answer_relevance** and **groundedness** independently — a report can
   be perfectly grounded and still not answer the question, or vice versa.
3. Recommended: label the first ~5, then pause and reconsider any band
   definitions that felt ambiguous before continuing — don't let the
   standard drift partway through (this is the plan's "calibrate before
   batching" step, compressed for n=20).
4. `Download labels` at any point for a `human_labels_<ts>.json` file.
   Progress autosaves to the browser as you go either way.
5. **Self-consistency (required, since this is a single annotator so
   inter-annotator agreement isn't measurable)**: 24h+ later, reload the
   tool fresh — don't look at the first session's file — and label all 20
   again into a second file. Compare the two with
   `compute_agreement.py --labels2`.
6. Run:
   ```
   poetry run python scripts/compute_agreement.py \
       --labels data/annotations/human_labels_<ts>.json \
       --answer-key data/annotations/answer_key_1785455589.json
   ```

## Bugs found by the user's first labeling attempt, fixed 2026-07-30

The user started labeling case C01 against the *first* export
(`answer_key_1785447682.json` / `labeling_tool_1785447682.html` — both
since deleted) and, in the course of asking careful questions about what
exactly she was looking at, surfaced three real bugs. Labeling paused,
these were fixed, and the corpus + export were regenerated before handing
the tool back. This is exactly the "if agreement comes back low" loop
described below, one level earlier: catching measurement problems *before*
spending hours labeling against a broken setup, not after.

1. **Judge and human were shown different material.** `_build_judge_prompt`
   truncated findings to 600 chars and the report to 1200 chars; the export
   script showed the human the full text. For case C01 specifically, the
   600-char cut landed mid-way through the findings and sliced off the
   entire groupby result (the 4 quarterly revenue figures) and comparison
   step -- the judge had no way to verify numbers the report cited, which
   a human reading the same case could see plainly. **Fixed**: removed the
   truncation entirely (`_build_judge_prompt` now uses the full findings
   and full report), and unified the rendering so the judge's prompt and
   the human's blind export are built from the *same* functions
   (`render_findings`/`render_rag_context`/`render_data_quality` in
   `src/eval/metrics.py`) instead of two independently-maintained
   implementations that had already drifted. Measured cost of removing the
   truncation: +$0.12 per full 20-case replay run ($0.35 → $0.47, +34%) --
   judged worth it, since the alternative was an invalid measurement.
2. **Report claims that couldn't be checked against anything shown.** The
   user asked where "238 fully duplicated rows" came from -- it was real
   (`state["data_quality_report"]`), but neither the judge nor the blind
   export surfaced it at all. **Fixed**: `render_data_quality()` renders
   `data_quality_report`'s flagged issues; both the judge's prompt (a new
   `### Data Quality` section) and the labeling tool (a new "Data quality"
   panel) now show it.
3. **A real analysis bug, not just a visibility gap**: the report claimed
   "a one-way ANOVA test indicated no statistically significant
   differences," but the test's own p-value was `nan` (`compare_segments`
   in `src/tools/stats_tool.py` was run on data already reduced to one row
   per quarter by a prior groupby step, so within-group variance -- and
   therefore the F-statistic -- was undefined). `"significant": bool(p_val
   < 0.05)` evaluates `nan < 0.05` as `False` in Python, so an
   uncomputable test and a genuinely-computed null result were
   indistinguishable in the output -- the same silent-fail-open pattern as
   the rate-limit bugs found in Step 1, just in the statistics layer
   instead of the API layer. **Fixed**: a shared `_significance_flag()`
   helper returns `None` (not `False`) when the p-value is `nan`, applied
   to all 4 call sites in `stats_tool.py` (t-test, chi-square, time-series
   trend, segment comparison), plus a diagnostic `note` explaining why on
   the ANOVA path. Regenerating C01 after the fix produced a genuinely
   different (the pipeline is stochastic) report that said the test
   "could not be computed... due to insufficient data" -- an honest
   statement instead of a fabricated one. New regression test:
   `test_compare_segments_on_already_aggregated_data_reports_undefined_not_false`
   in `tests/unit/test_phase5.py`, which reproduces the exact C01 input
   shape.

   **The nan-significance display bug is fixed, but the root cause
   producing it isn't**: the Planner shouldn't have written a plan that
   runs `comparison` (ANOVA) on step 2's already-aggregated
   one-row-per-quarter output in the first place — every group having
   count=1 makes the test structurally meaningless regardless of the data.
   Tracked as [roadmap.md #25](roadmap.md), deliberately deferred to after
   Step 4 since fixing `analysis_agent.py`'s planning logic would
   invalidate the corpus this labeling session is actively using.

The full 20-case corpus and the annotation export were regenerated after
all three fixes; the user's original C01 rating isn't wasted (the question
itself hasn't changed), but she should re-read case C01 in the *new*
export before trusting a rating made against the old, incomplete view.

## A fourth bug, found labeling case C02: the guardrail has the same truncation bug, in a code path bug #1's fix never touched

Labeling continued after the fixes above (C01 re-read, then C02). C02's
report accused itself of hallucination it hadn't committed: its own
Automated Caveats section claimed "the provided data does not include any
information about the Search channel" — false. Search's numbers
(conversion rate 6.49%, the actual top-performing channel) are right there
in `analysis_results`.

Root cause, confirmed by direct reproduction:
[`src/agents/guardrail_agent.py:214-223`](../src/agents/guardrail_agent.py#L214)
(`GuardrailAgent._llm_judge`, the hallucination-detection call) builds its
own context by joining `result_summary` across steps and capping it at
**800 chars** — for C02, the substring `"comparison: top=Search"` sits at
character 950 of the joined text, past the cut, so the guardrail's own
LLM call never saw it. The report text is separately capped at 1500 chars.

This is the *same bug* as #1 above, in spirit — an LLM-as-judge call built
from a truncated view of material everything else now sees in full — but
in a *different function in a different file* that bug #1's fix never
touched (`_build_judge_prompt` in `src/eval/metrics.py` is the eval
judge's context-builder; `_llm_judge`'s inline context-building in
`guardrail_agent.py` is entirely separate code). Fixing one did not fix
the other. Consequence is arguably worse than a missed hallucination: the
guardrail fabricated a false accusation against a report that was, on
this specific claim, correct — teaching the pipeline to distrust valid
output.

**Not fixed now** — `guardrail_agent.py` is generation-relevant, same as
the Planner bug found on C01; fixing it now invalidates the corpus
mid-labeling. Tracked as [roadmap.md #26](roadmap.md), deferred to the
same after-Step-4 window as #25. Likely fix: have `_llm_judge` reuse
`render_findings`/`render_rag_context` from `src/eval/metrics.py` instead
of maintaining its own truncated, independently-drifting context-builder.

## Results

First pass, 2026-07-31 (`data/annotations/human_labels_1785476672693.json`
vs. `data/annotations/answer_key_1785455589.json`, all 20 cases,
`scripts/compute_agreement.py`):

| Metric | n | QWK | Spearman (p) | MAE | Exact-band-match |
|---|---|---|---|---|---|
| `answer_relevance` | 20 | **0.676** (acceptable) | 0.664 (p=0.001) | 0.200 | 35% |
| `groundedness` | 20 | **0.178** (weak) | 0.541 (p=0.014) | 0.247 | 35% |

Full report with confusion matrices:
`data/annotations/human_labels_1785476672693_agreement.json`.

**Headline finding: the judge agrees with the human reasonably well on
whether a report answers the question, but agrees poorly on whether the
report is actually supported by the data — which is the dimension eval v2
Step 2 specifically rebuilt (claim-list scoring) to make more objective.**
That rebuild fixed the *mechanism* (no more bare-float groundedness, no
more shared reasoning string with relevance) but not, apparently, the
judge's actual reliability at the one thing groundedness is supposed to
catch: fabricated claims.

The groundedness confusion matrix (bands b0-b4 = 0/0.25/0.5/0.75/1.0) shows
the shape of the disagreement clearly — the judge lands in the top band
(b4, fully grounded) on 15 of 20 cases, while the human's ratings spread
across b1-b4. This isn't random noise; it's a directional bias, and the
human's free-text notes name specific claims the judge apparently didn't
penalize enough:

- **DG02**: report says a "notably high churn rate" is linked to login
  frequency; the underlying correlation is -0.0113. Human: "not notably
  high." Judge: still landed near the top band.
- **DG03**: report claims the Online channel has the lowest revenue among
  top channels; human notes it should be Retail.
- **D05**: report claims a specific date/order-id had the highest single-order
  revenue; human notes both specific claims are wrong.
- **E03**: "Consumer segment outperforms Corporate" and a claimed seasonal
  pattern, both flagged by the human as unsupported by the findings shown.

Every one of these is a *specific, checkable numeric or factual claim* —
exactly the kind of thing the Step 2 claim-list rebuild was meant to catch
by forcing the judge to enumerate claims and check each one individually
rather than emit an impressionistic float. The weak QWK says that
mechanism alone wasn't sufficient; the judge's per-claim support-checking
is still too lenient in practice. That's a concrete, evidenced next step
(likely: tighten `EVAL_GROUNDEDNESS_SYSTEM` in `agent_prompts.py` to
require the judge to explicitly compute/verify cited numbers against the
findings rather than accepting a plausible-sounding paraphrase — the
current failures all follow the pattern "claim resembles the data but
gets a specific number or direction wrong").

`answer_relevance` fared better (QWK 0.676) and the disagreement pattern
is different in kind, not just degree: the judge's discretized scores
never land in bands b1 or b3 at all (column sums: b0=3, b1=0, b2=9, b3=0,
b4=8), so most of the mismatch is the judge clustering around b2/b4 while
the human used the full 5-band range — some of that gap is the judge
under-scoring relative to the human (2 of the human's 7 b4 cases got only
b2 from the judge) and some is over-scoring (3 cases the human placed at
b2/b3 got b4 from the judge). It reads more like reduced discrimination in
the judge's relevance scores than a one-directional bias — worth checking
whether `score_answer_relevance`'s prompt is nudging the judge toward a
narrower effective range than the 5 bands it's meant to use.

**One labeling-protocol note worth recording for reproducibility**: on
D01, the human explicitly stated she does not factor the report's own
"Automated Caveats" section into her relevance/groundedness scoring,
treating it as separate from the report content being judged — even where
a caveat is itself wrong (as it was here). This is a reasonable and
consistently-applied convention, but it means the human labels score the
*report body*, not the report-plus-self-critique bundle; worth keeping in
mind if `EVAL_GROUNDEDNESS_SYSTEM`/`EVAL_RELEVANCE_SYSTEM` prompts are ever
changed to explicitly include or exclude caveats from the judge's own
view.

Both metrics show 35% exact-band-match — on a 5-band scale that's better
than the ~20% a random labeler would hit by chance, but confirms QWK
(which credits near-misses) is the right primary statistic here, not raw
accuracy.

**n=20 caveat still applies** — these are the same 20 cases as everywhere
else in this doc, not an independent larger sample; a single case moving a
band could shift groundedness QWK meaningfully. The self-consistency
retest (below) is the next piece of evidence that would either confirm
this is a real judge-reliability gap or reveal it's partly human-rating
noise at small n.

## Results — Batch 2 (20 new cases, Step 4 expansion), first pass, 2026-07-31

Independent replication on a genuinely different set of 20 cases (never
scored or labeled before — `D06-D09`, `DG05-DG08`, `C05-C08`, `P04-P07`,
`E05-E08`), against `data/annotations/answer_key_1785525212.json`:

| Metric | n | QWK | Spearman (p) | MAE | Exact-band-match |
|---|---|---|---|---|---|
| `answer_relevance` | 20 | **0.551** (weak) | 0.557 (p=0.011) | 0.188 | 55% |
| `groundedness` | 20 | **0.040** (weak) | 0.326 (p=0.161, not significant) | 0.182 | 50% |

Full report: `data/annotations/human_labels_batch2_1785532153573_agreement.json`.

Both numbers are *worse* than batch 1 (relevance 0.676→0.551, groundedness
0.178→0.040 — essentially chance-level, and the Spearman correlation isn't
even statistically significant at n=20). The judge's groundedness
distribution shows the same top-band skew as batch 1: 13 of 20 cases land
in the top band (b4) by the judge's discretized score, vs. a human
distribution spread across b1-b4 — same directional pattern, replicated on
completely independent cases.

**Traced two of these to a precise, verified root cause — not just "the
judge is lenient" but a specific gap in what it checks.** Pulled the
judge's actual claim-list reasoning from the answer key for the cases with
the biggest gaps:

- **D07** ("total spend and revenue by channel"): report claims "The Email
  channel generated the highest revenue at approximately $60,607,080.18."
  False — Search's revenue ($73,111,916.05) is higher, and both numbers
  sit in the *same 5-row groupby table*, fully visible, no truncation
  involved. The judge's groundedness call still scored this 1.0 (7/7
  claims supported); its own reasoning for this exact claim reads "✓ ...
  (evidence: channel=Email, revenue=60607080.18)" — it checked that the
  *number* $60,607,080.18 correctly is Email's revenue (true), but never
  checked whether the word "highest" was actually true given the other
  numbers in the same table it had already cited two lines earlier for a
  different claim.
- **DG07** ("why does Product 8 have a negative margin"): report claims
  "Product 2 has the highest profit margin at $362.98." False — Product 7
  has $429.23, sitting in the very same rendered findings (confirmed by
  direct inspection: `render_findings` output for this case is 2,728
  characters, and "product_id': 7 ... profit_margin: 429.23" appears at
  character 1,483 — inside the visible text, not past any truncation
  point). The judge scored this 1.0 too (4/4 claims supported); its
  "evidence" for the Product-2 claim is a byte-for-byte echo of the
  report's own sentence back at itself — not independent verification at
  all, just confirming the claim exists in the report.
- **C07** is the useful control case here, and worth including precisely
  *because* it complicates a simpler story: same report shape, same kind
  of false superlative claim ("the East region generated the highest
  revenue in 2022" — actually North, at $439,901 vs. East's $401,708, both
  fully visible in the same findings table), and this time **the judge did
  catch it** — flagged "✗ no evidence found" for that exact claim, landing
  at groundedness=0.6 against the human's 0.5. Good agreement, not a miss.

So the accurate claim is narrower than "the judge over-scores
groundedness in general": on this batch it specifically, repeatedly missed
*comparative/superlative* claims ("highest," "lowest") even when
fully-grounded contradicting data sat in the same visible table — but it's
inconsistent, not universal, since it caught the structurally identical
error in C07. That inconsistency is itself informative: it suggests the
claim-verification step isn't doing a real "is this superlative true
across all the values shown" check, and only sometimes stumbles into
catching the contradiction anyway (plausibly when the wrong value and the
right value are closer together in the prompt, or the claim is phrased in
a way that happens to prompt a fuller re-read). Concrete next step, more
specific than batch 1's: strengthen `EVAL_GROUNDEDNESS_SYSTEM` to require
explicit comparison against *all* co-listed values before accepting a
"highest/lowest/most" claim, not just confirming the cited number exists
somewhere in the findings.

**One byproduct worth flagging, not fixing now**: DG07's report also
states "The confidence level in this finding is 100%, indicating high
reliability" — a fabricated precision figure with no corresponding field
anywhere in `analysis_results`, which the human flagged separately on D06
and C06 too ("confidence level" / "moderate at 60%" appearing "out of
nowhere"). This looks like a real, repeatable generation-side pattern (the
Insight Agent inventing a numeric confidence score with no underlying
computation) distinct from the comparative-claim issue above — worth its
own roadmap item once Step 4 is otherwise wrapped up, not chased down now.

## Results — Batch 3 (20 new cases, Step 4 expansion), 2026-08-01

Third independent replication (`D10-D13`, `DG09-DG12`, `C09-C12`,
`P08`/`P09`/`P11`, `E09-E13` — never scored or labeled before), against
`data/annotations/answer_key_1785544815.json`:

| Metric | n | QWK | Spearman (p) | MAE | Exact-band-match |
|---|---|---|---|---|---|
| `answer_relevance` | 20 | **0.577** (weak) | 0.626 (p=0.003) | 0.212 | 40% |
| `groundedness` | 20 | **0.263** (weak) | 0.449 (p=0.047) | 0.216 | 50% |

Full report: `data/annotations/human_labels_batch3_1785553830125_agreement.json`.

Numerically, groundedness is the best of the three passes so far (0.178,
0.040, 0.263) — but "weak" across all three, and the variability itself
(0.178 → 0.040 → 0.263) is a useful data point on its own: at n=20 per
batch, QWK moves around a lot, which is exactly why a single pass was
never going to be the final word and why growing the suite (now at 100)
matters.

**The headline finding from this batch isn't a new failure mode — it's
confirmation, by triangulation across three fully independent human
labeling sessions, of the exact mechanism found in batch 2.** DG11 asks
"why does the Office Supplies category have the lowest average product
margin" (true premise: Office Supplies' average is 43.67, genuinely the
lowest of the 5 categories). The report's answer contains two false
superlative claims, and the judge marked *both* as supported:

- **"Product 2 generates the highest margin at $362.98"** — false, and
  not a new discovery: the real answer (Product 7, $429.23) was
  independently found by the human on **C04 in batch 1** and again on
  **DG07 in batch 2**. This is the third distinct human-labeled case where
  this exact same fabricated claim appears in a generated report, and the
  third time the judge's groundedness check calls it "✓ supported"
  because $362.98 genuinely is Product 2's margin (the number is real —
  the comparative word "highest" is what's false, and nothing in the
  judge's claim-verification checks that).
- **"Product 6 stands out with the highest profit margin ... among office
  supplies"** — also false, verified directly against `data/demo/*`:
  within Office Supplies specifically, Product 26 has the highest margin
  ($363.32), Product 6 is second ($201.60). Same mechanism, same report,
  a second instance the judge also marked supported.

Three independent human sessions, three independent cases, one root
cause: the groundedness judge checks whether a cited *number* traces back
to the findings, but never checks whether a *comparative claim* attached
to that number holds against the other values sitting in the same table.
This is no longer a hypothesis — it's the most reproducible finding in
this entire eval-v2 effort, and the clearest, most concrete target for the
`EVAL_GROUNDEDNESS_SYSTEM` prompt fix once a genuine (unread-by-Claude)
holdout exists to validate the fix against.

Two more of the human's notes are worth a direct check before trusting
them at face value, matching this doc's standing practice — pulled from
the judge's own reasoning in the answer key:

- **DG09**: the report claims "Campaign CAM0002 achieved the highest
  click-through rate at 3.52%." The judge's own reasoning marked this
  "✗ no evidence found" — so this is actually a case the judge *caught*,
  not missed, landing at groundedness=0.6 against the human's 0.75. A
  useful reminder alongside batch 2's C07: the judge's miss rate on
  superlative claims is real but not total.
- **C12**: judge groundedness=0.333 (2 of 3 claims marked unsupported,
  including a specific fabricated date), close to the human's 0.25 — good
  agreement, not a divergence worth flagging further.

## Results — Batch 4+5 merged (40 new cases), 2026-08-01

Largest single labeling pass so far (n=40, double every prior batch),
against `data/annotations/answer_key_batch4_5_merged.json`:

| Metric | n | QWK | Spearman (p) | MAE | Exact-band-match |
|---|---|---|---|---|---|
| `answer_relevance` | 40 | **0.763** (acceptable) | 0.748 (p<0.001) | 0.144 | 55% |
| `groundedness` | 40 | **0.040** (weak) | -0.016 (p=0.921, not significant) | 0.231 | 35% |

Full report: `data/annotations/human_labels_batch4_5_1785624268674_agreement.json`.

At double the sample size, both patterns from batches 1-3 sharpen rather
than regress to the mean: `answer_relevance` is now the *best* result of
any pass (0.763, clearly "acceptable"), while `groundedness` lands exactly
at batch 2's near-chance QWK=0.040 — and this time the Spearman
correlation is essentially zero (-0.016) and nowhere near significant
(p=0.921). At n=40 that's a much stronger statement than any single
20-case batch could make: the judge's groundedness score and the human's
groundedness score are, across this batch, statistically indistinguishable
from unrelated.

**The single most important result in this whole document**: the exact
same fabricated claim tracked since batch 1 — *"Product 2 has the highest
profit margin"* — appears **a fourth independent time**, on **E15** (a
correlation case, "what's the relationship between unit price and
margin"), and the judge marks it supported a fourth time. Pulled straight
from the answer key:

> ✓ Product 2 has the highest profit margin at 362.98. (evidence: Product
> 2 details: profit_margin=362.98.)

Four independent human labeling sessions (batch 1's C04, batch 2's DG07,
batch 3's DG11, now batch 4+5's E15), four independently-generated
reports, and the judge's "evidence" is the same shallow check every single
time: confirm the cited number is real (it is — $362.98 genuinely is
Product 2's margin), never check whether "highest" is true against the
other values already sitting in the same findings (Product 7's $429.23,
visible in every one of these four cases' rendered findings, per the
earlier byte-offset check on DG07 in the batch-2 results above). This has
gone from "a finding" to the load-bearing piece of evidence in this
project's entire eval-v2 effort — it is reproducible on demand, in a
report the pipeline hasn't been touched to fix (deliberately, so as not to
invalidate the label sets), across every batch generated since Step 3
began.

Two more checks worth recording, since not every miss is the same shape:

- **C16** shows a related but *distinct* failure mode, closer to the
  original bug #1 (batch-1 C01) than to the Product-2 pattern: "On January
  1, 2023, the East region's Widget A achieved the highest revenue of
  $421.51 through online sales" — the judge's evidence only confirms
  $421.51 is a real row in a *filtered, unsorted* preview of
  `sales_data.csv` (12,240 rows), not that it's actually the maximum of
  anything. This is "a plausible-looking row from a preview the model
  never sorted or aggregated," not "a real number attached to a false
  comparison" — same broad category (unverified superlatives), different
  specific mechanism.
- **D14** is a genuine control case, and a good one: the report claims
  "Clothing has the highest total quantity ordered" (false — Furniture's
  12,507 beats Clothing's 11,120, both visible in the same findings), and
  this time **the judge caught it** — marked "✗ no evidence found," landing
  at groundedness=0.8, consistent with batch 2's C07 and batch 3's DG09 as
  further proof the miss rate is real and large but not total.

## Results — Self-consistency retest (original 20, relabeled blind), 2026-08-02

**Step 3 is now complete.** The user relabeled all 20 original cases blind
(fresh storage key, no visibility into her first-pass answers, per the
protocol above), roughly two days after the first pass. Two comparisons:

**Human vs. human (self-consistency)** — pass 2 vs. pass 1
(`data/annotations/human_labels_1785476672693.json`), the question this
whole retest exists to answer: is the human's own rating stable, or is
some of the weak human-vs-judge agreement actually human noise?

| Metric | n | QWK | Spearman (p) | MAE | Exact-band-match |
|---|---|---|---|---|---|
| `answer_relevance` | 20 | **0.683** (acceptable) | 0.682 (p=0.001) | 0.138 | 60% |
| `groundedness` | 20 | **0.775** (acceptable) | 0.773 (p<0.001) | 0.125 | 50% |

**Human vs. judge** — pass 2 vs. the original answer key
(`answer_key_1785455589.json`), replicating the very first measurement
with a second, independent labeling instance of the same 20 reports:

| Metric | n | QWK | Spearman (p) | MAE | Exact-band-match |
|---|---|---|---|---|---|
| `answer_relevance` | 20 | **0.766** (acceptable) | 0.782 (p<0.001) | 0.163 | 40% |
| `groundedness` | 20 | **0.183** (weak) | 0.483 (p=0.031) | 0.230 | 35% |

Full reports: both saved alongside `human_labels_retest_1785630231929.json`.

**This is the decisive result for the whole judge-calibration effort.**
Line them up:

| Comparison | groundedness QWK |
|---|---|
| Human vs. human (self-consistency) | **0.775** |
| Human vs. judge, pass 1 (original 20) | 0.178 |
| Human vs. judge, batch 2 | 0.040 |
| Human vs. judge, batch 3 | 0.263 |
| Human vs. judge, batch 4+5 (n=40) | 0.040 |
| Human vs. judge, pass 2 (original 20, this retest) | 0.183 |

The human's own groundedness rating, on the exact same 20 cases, two days
apart, with no memory aid and no visibility into her first answers, is
**"acceptable" (0.775)** — she rates a report's groundedness the same way
twice. Every single human-vs-judge measurement, across five independent
comparisons spanning 100 different cases plus this replication, sits in
the 0.04-0.26 "weak" band. That decomposition rules out the most
convenient explanation for the low agreement: it is *not* that human
groundedness judgments are inherently noisy or subjective at n=20-40. The
human is consistent. The disagreement is specifically and repeatedly
between the human and the judge — a real calibration gap, not measurement
noise on either side, and the Product-2-margin mechanism documented above
(traced through four of these five batches) is the concrete, verified
explanation for a large share of it.

`answer_relevance` tells a gentler version of the same story: self-consistency
(0.683) and human-vs-judge (0.676-0.766 across passes) are close to each
other — consistent with the batch-2/4+5 finding that relevance
disagreement looks more like reduced judge discrimination than a
systematic miss, a smaller and different problem than groundedness's.

One data point worth naming directly: on **C04**, the user flagged
*"Product 2 generates the highest margin"* as wrong in **both** passes,
independently, two days apart, with no memory aid — the same case, so it
isn't a fifth independent triangulation of the underlying bug, but it is
strong evidence that this specific human judgment is itself highly
reliable, not a one-off reading.

## A2 — Fixing `EVAL_GROUNDEDNESS_SYSTEM`, 2026-08-02

**Methodology caveat, stated up front and not to be glossed over**: this
is **not** a blind-holdout evaluation. By this point Claude had read all
120 collected labels (100 cases across 4 batches, plus the retest) in
full detail while diagnosing the Product-2-margin mechanism above. The
numbers below measure whether the fix improves agreement on
*previously-collected* labels, which is weaker evidence than a fix
validated against data nobody involved in writing the fix had seen. This
was an explicit, informed tradeoff — see `docs/eval_v2_plan.md` Step 4 for
the three options that were on the table and why this one was chosen.
Take the improvement below as real but not as strong a claim as a fresh
holdout would give.

### The fix

Added an explicit instruction to `EVAL_GROUNDEDNESS_SYSTEM`
(`src/config/agent_prompts.py`): a comparative/superlative claim
("highest", "lowest", "most", "best", "X outperforms Y", ...) is not
"supported" just because the cited number is real — the judge must also
check every *other* visible value for that same metric and mark the claim
unsupported if any of them contradicts the claimed extreme. This is
excluded from `generation_fingerprint()` (only `EVAL_*` prompts are
excluded, by design — see `src/eval/replay_cache.py`), so the replay
cache stayed valid; no reports needed regenerating, only re-scoring.

### Mechanism check — did it actually fix the specific bug?

Direct, before-any-statistics confirmation on the four cases where the
Product-2-margin claim was found (`data/annotations/answer_key_postfix_100_clean.json`):

| Case | New judge verdict on "Product 2 has the highest margin" |
|---|---|
| C04 | ✗ unsupported — "no evidence found" |
| DG07 | ✗ unsupported — "no evidence found" |
| DG11 | ✗ unsupported (and its second false claim, "Product 6 highest among office supplies," also now ✗) |
| E15 | ✗ unsupported — "no evidence found" |

All four. The fix works exactly as designed on the specific mechanism it
targets.

### Aggregate check — does it move the QWK numbers?

Pooled across all 100 distinct labeled cases (`human_labels_pooled100.json`,
combining pass 1 + batch 2 + batch 3 + batch 4+5 — the retest excluded
since it duplicates pass 1's case ids), before vs. after:

| Metric | Before (pre-fix) | After (post-fix) |
|---|---|---|
| `answer_relevance` | 0.670 | 0.684 |
| `groundedness` | **0.141** | **0.185** |

`answer_relevance` is flat (expected — that prompt wasn't touched; the
small movement is ordinary judge-call variance). `groundedness` moved from
0.141 to 0.185 — a real, ~31% relative improvement, though it stays in the
"weak" QWK band rather than crossing into "acceptable." One data quality
note on the way to this number: the first full re-scoring pass hit
OpenAI's rate limit on 19 of 100 cases (`"valid": false`, correctly
surfaced rather than silently defaulted per the Step 1 fix — see
`call_with_rate_limit_retry`), and those 19 were re-scored individually
before this comparison was run; the numbers above are from the fully
clean pass.

Per-batch, for completeness (n=20 or 40 each, noisier than the pooled
number — some individual batches even show a small dip, which is exactly
the kind of noise the pooled n=100 number exists to average out):

| Batch | groundedness QWK, before → after |
|---|---|
| Pass 1 (original 20) | 0.178 → 0.156 |
| Batch 2 | 0.040 → 0.033 |
| Batch 3 | 0.263 → 0.280 |
| Batch 4+5 (n=40) | 0.040 → 0.225 |
| Retest (pass 2, original 20) | 0.183 → 0.215 |

**Honest read of this result**: the fix demonstrably closes the exact gap
it targeted (4/4 on the mechanism check) and produces a real, if modest,
aggregate improvement (+0.044 QWK, ~31% relative) that survives pooling
across all available data. It does not fully close the human-vs-judge gap
— groundedness is still "weak," meaning superlative claims were a
significant but not the only source of disagreement; other, unidentified
sources of judge/human divergence remain. That's a more credible finding
than either "one prompt tweak solved it" or "nothing moved" would be, and
it's exactly the shape of result that matters for an eval-engineering
portfolio: a specific, verified bug, a specific fix, a measured
before/after, and an honest statement of what's still unexplained.

Once real labels exist, the same 20 (query, report, findings) tuples can
score additional candidate judge models without re-labeling anything —
just re-run `score_answer_relevance`/`score_groundedness` with a different
model and recompute agreement against the existing human labels. This
turns "why does the judge use gpt-4o" from a justification into an
experimental result, e.g.:

| Judge model | QWK (answer_relevance) | QWK (groundedness) | Cost/run |
|---|---|---|---|
| gpt-4o | ? | ? | ? |
| claude-sonnet-5 | ? | ? | ? |
| claude-haiku-4-5 | ? | ? | ? |

Also worth revisiting once labels exist: `settings.resolved_eval_provider`
currently resolves to `openai` (same provider as the agent pipeline) because
`ANTHROPIC_API_KEY` in this environment isn't a real key — the
self-preference concern the judge-tiering design was meant to avoid
(judge shouldn't share weights/training with what it's scoring) is
presently live. Fix that before treating a same-provider judge's agreement
numbers as fully independent evidence.

## If agreement comes back low

Not a failure — the standard loop (see `docs/eval_v2_plan.md` Step 3): read
the confusion matrix, find the cell with the most disagreement, usually
traces to an under-specified band ("mostly grounded, some unsupported" —
how many unsupported claims counts as "some"?), rewrite that band with a
concrete example, then **re-run the judge on the same labeled reports and
recompute QWK — no re-labeling needed**. "QWK went from 0.45 to 0.71 by
rewriting the mid-band rubric" is, on its own, a complete demonstration of
eval-engineering work.
