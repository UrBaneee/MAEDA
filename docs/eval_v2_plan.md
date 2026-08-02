# Eval v2 Plan — from a scoreboard to an instrument

Written 2026-07-29, after a mock-interview-style audit surfaced that the
current eval harness (aggregate=0.725 on `phase_d_model_tiering.json`) has
six zero-variance metrics carrying 33% of the aggregate's weight, a judge
that shares one reasoning string across two supposedly independent metrics,
a "trust the reasoning, not the aggregate" methodology whose reasoning trace
turned out to be misaligned for `groundedness`, and 35 documented fixes
validated against the same 20 cases they were tuned on, with no noise floor
established and no holdout.

This plan is the fix, in four steps, in dependency order. Steps 3 and 4 can
run in parallel with each other; both depend on Step 2 being done first
(scores from the old metric definitions aren't comparable to anything built
on this plan). Step 1 must run before Step 2's fixes land, because it's the
only way to know, later, whether Step 2 actually changed anything or just
moved noise around.

```
Step 1  Measure noise            ┐ must go first — old baselines aren't
        judge-only (cheap)       │ comparable once Step 2 lands
        full-pipeline (8 reruns) ┘
             ↓
Step 2  2a  0.5 fallback → explicit failure   (do first — precondition for
        2b  split judge into 2 independent calls   everything measured after)
        2c  groundedness → claim-list, score derived
        2d  rebuild the 6 zero-variance metrics
             ↓
        rebuild baseline — old 0.725 is not comparable, say so explicitly
             ↓
Step 3  Human-label 60-100 samples, QWK/Spearman     ┐ can run in parallel
        reuse the same labels to pick a judge model  │ with Step 4 — pulls
                                                       │ from logs/eval_runs/,
                                                       │ doesn't need new cases
Step 4  Collapse 4 duplicate golden-suite sources    ┘
        Expand to 100-200 cases (generated + reviewed)
        Stratified dev/test split, test never peeked at until one reveal
```

---

## Step 1 — Measure the noise floor

**STATUS: DONE (2026-07-30).** Full results in
[docs/noise_floor.md](noise_floor.md); raw data in
`logs/noise_runs/judge_noise_1785385878.json` and
`logs/noise_runs/full_noise_1785391747.json`. Headline numbers, since the
guess in this section's original "Deliverable" example below turned out to
be wrong in an interesting way:

- Judge noise on a fixed input is close to zero (10/12 case×metric cells
  had exactly 0 std across 20 repeats) — the opposite of the original
  guess that the judge would be the dominant noise source.
- Full-pipeline noise (clean, isolated run, 8 reruns of the 20-case suite):
  `overall_aggregate` std=0.014, **2σ=0.028**. Per-case noise is much less
  even: median std=0.060, ranging from <0.01 (C04, P02, D02) to 0.137
  (E01, the open-ended "give me an overview" query).
- Bootstrap calibration validated: 1/28 pairwise trial comparisons flagged
  "significant" under pure noise, matching the ~1.4/28 expected false-positive
  rate at 95% CI.
- Unplanned but load-bearing side effect: a first attempt at this
  measurement was contaminated by running two API-heavy jobs concurrently
  against a low-tier OpenAI rate limit (30k TPM/500 RPM), which surfaced two
  real, unrelated bugs — `src/agents/analysis_agent.py`'s planner and
  `src/agents/guardrail_agent.py`'s guardrail judge both silently failed
  open (empty plan / auto-pass) on any exception, including a transient
  429, with no retry. The guardrail one was a live safety concern (fails
  open exactly under load). **Fixed 2026-07-30**: both now go through
  [`src/utils/retry.py`](../src/utils/retry.py)'s shared
  `call_with_rate_limit_retry` — the same pattern already applied to the
  eval judge in `src/eval/metrics.py`, now consolidated into one helper all
  three call sites share instead of three near-copies. See noise_floor.md's
  incident section for full detail and the new tests that verify it.

**Goal:** produce one number — "a score change smaller than X is not
evidence" — that every later comparison gets judged against.

### Why this matters even at temperature=0

`llm_temperature` defaults to 0.0 and the judge call is also
`temperature=0.0`. That is not the same as deterministic. Batch-order
floating point accumulation, MoE routing, backend hardware differences all
still produce different outputs at temperature 0 on commercial APIs. Say
this explicitly in the writeup — it's the first thing a careful interviewer
will ask about.

### Two noise sources, measured separately

| Source | How to isolate | Why separate |
|---|---|---|
| **Judge noise** | Take one saved `state` (a real report from a past run), re-invoke the judge N times, nothing else changes | Cheap, minutes to run. If this alone is huge, everything downstream needs rethinking first |
| **Full-pipeline noise** | Rerun the entire 20-case suite N times, same config, nothing changed | This is the real number comparisons get judged against — generation noise + judging noise stacked |

Measure judge noise first — it's cheap and gates the priority of everything else.

### Build `scripts/measure_noise.py`

Two modes:
```
--mode judge   # pull N saved states from logs/eval_runs/*.json, re-judge each K times
--mode full --repeat 8   # rerun the full 20-case suite 8x, same config
```

**Watch out for the reuse optimization in `EvalRunner.score`**
([src/eval/runner.py:113](../src/eval/runner.py:113)): it reuses
`state["eval_scores"]` if already present, to avoid double-judging the same
state. Correct behavior for normal runs, but `--mode full` needs a
`force_rejudge=True` override — otherwise you're measuring "how much does
the pipeline vary" without re-sampling the judge at all, which isn't the
number you want.

### Report per-metric, not one aggregate number

```
metric                mean    std     min    max   2σ threshold
answer_relevance      0.62   0.081   0.45   0.75   0.16
groundedness           0.50   0.104   0.35   0.68   0.21
factual_accuracy       0.51   0.033   0.47   0.56   0.07
...
overall_aggregate      0.725  0.021  0.698  0.751   0.042
```

Different metrics have genuinely different noise — a single global threshold
either lets real signal in low-noise metrics get dismissed as noise, or lets
real noise in high-noise metrics get read as a finding.

### Prefer paired bootstrap over a flat 2σ rule

The real comparisons in this project are paired (same 20 cases, before vs.
after a change) — paired tests have much more power than treating the two
runs as independent samples. Take the per-case deltas, bootstrap resample
10,000x, report the 95% CI. If it crosses zero, not significant. `scipy.stats.bootstrap`
handles this in ~15 lines.

### Cost

Cost tracking was just fixed (previously silently reported $0 every run —
see [src/eval/metrics.py:325](../src/eval/metrics.py:325)), so this is the
first run where the real number will actually show up. Estimated $1-3 per
20-case pass; 8 passes ≈ $8-25. Judge-only noise is a few dollars. This is
the highest-leverage spend in the whole plan — don't cut corners here.

### Deliverable — DONE, see docs/noise_floor.md

The original guess for this paragraph (below, struck through in spirit)
assumed the judge would be the dominant noise source. The measured result
was the opposite: judge noise was close to zero, and full-pipeline noise
(generation, not judging) turned out to be what needs guarding against.
Actual result:

> Repeated 8x at identical config (isolated, no concurrent API load —
> see the contamination incident in noise_floor.md), `overall_aggregate` has
> std=0.014 (2σ=0.028). Judge-only repetition (20x per fixed input, 6 inputs)
> gave std=0 on 10/12 case×metric cells — the judge is *not* the dominant
> noise source; full-pipeline generation noise is. Per-case noise varies
> 40x across the suite (0.003 to 0.137 std) — a flat regression threshold
> is wrong at the per-case level; use paired bootstrap there instead.
> Re-auditing the 35 historical fixes against this baseline is still open —
> flagged as a natural follow-up in noise_floor.md, not yet done.

---

## Step 2 — Fix the metrics themselves

**STATUS: DONE (2026-07-30).** All four sub-tasks (2a-2d) landed together
in one implementation pass, verified with 522 unit tests plus a real
replay run against the cached 20-case corpus
(`logs/eval_runs/replay_1785446988.json`). Headline results:

- **Every metric feeding the aggregate now has measured variance** — the
  acceptance check below is fully satisfied. `tool_selection` and
  `chart_appropriateness` (previously always 1.00) now show real {0.0, 1.0}
  splits; `groundedness` (previously stuck on {0, 0.5, 1.0}) now takes 8
  distinct values in a single 20-case run.
- **New baseline: `overall_aggregate` = 0.658**, down from Step 1's clean
  0.716. Not comparable to the pre-Step-2 numbers in `docs/eval_report.md`
  or `docs/noise_floor.md` — the ruler changed, not just the system (see
  the "aggregate will drop" warning below, which is exactly what happened).
  **Superseded, not reproducible as-is**: this number predates the Step 3
  fixes below (`src/tools/stats_tool.py`'s nan-significance bug, judge
  prompt truncation) — both are generation-relevant changes that
  invalidated and regenerated the corpus, so re-running `replay_eval.py`
  today won't reproduce exactly 0.658. The delta is a real fix, not noise;
  a fresh baseline wasn't re-captured since it wasn't asked for and this
  document's job here is the *method*, not chasing a moving number after
  every subsequent bugfix.
- **Design deviation from this doc's original wording, decided during
  implementation:** `tool_selection`/`chart_appropriateness` score a
  **binary hit** (1.0 if any actual tool/chart is in the expected set, 0.0
  otherwise), not the literal `|actual ∩ expected| / |expected|` fraction
  written below. `expected_tools`/`expected_chart_types` were authored as
  acceptable *alternatives* ("either time_series or pandas_transform is
  fine"), and a multi-step plan routinely mixes a directly-relevant tool
  with supporting steps -- the literal fraction formula would score using
  one of two acceptable alternatives as "only half right," and would
  penalize a normal correct plan for including a legitimate supporting
  step. Binary hit matches this doc's own stated *rationale* ("more than
  one tool choice can be legitimate") better than its literal formula did.
- **Bug found live, not in mocks:** the first real replay run crashed
  several cases' groundedness scoring with `'list' object has no attribute
  'get'`. Root cause: `_build_eval_llm()`'s `max_tokens=256` was sized for
  the old bare-float response and silently truncated claim-list JSON for
  reports with several claims; `_parse_json`'s `[...]` fallback then
  parsed the truncated fragment as a bare list instead of failing cleanly.
  Fixed by (1) raising groundedness's budget to 1024 tokens and (2)
  deleting the `[...]` fallback entirely, since every prompt in this
  module asks for a top-level object and a bare list was never a
  legitimate response, only a symptom of truncation. New tests:
  `test_parse_json_*` in `tests/unit/test_phase9.py`.

**Infrastructure now in place (built 2026-07-30, see `src/eval/replay_cache.py`,
`scripts/generate_replay_corpus.py`, `scripts/replay_eval.py`):** every
sub-task below only touches `src/eval/`, which never needs a different
generated report. Run `generate_replay_corpus.py` once (full 20-case suite
cached at `logs/replay_cache/corpus.json`, ~7 min / $0.35 at the current
OpenAI Tier 2 concurrency), then `replay_eval.py` after each metric change
— scores the same cached reports against whatever `src/eval/metrics.py`
currently does, in ~1 minute instead of ~1 hour (56x measured on the full
suite: 63.8s replay vs. 3554s original generation). The cache's fingerprint
covers all generation-relevant source/data/settings and is deliberately
blind to `src/eval/` and to `EVAL_`-prefixed prompts inside
`src/config/agent_prompts.py`, so editing judge logic here never triggers
an unwanted, expensive regeneration — verified both in
`tests/unit/test_replay_cache.py` and live against the real repo (touching
`src/eval/metrics.py` left the fingerprint unchanged; touching
`src/agents/analysis_agent.py` changed it, as intended).

Four sub-tasks, in dependency order.

### 2a. Replace the 0.5 fallback with an explicit failure flag (30 min, do first) — DONE

**Current bug:** [src/eval/metrics.py:99](../src/eval/metrics.py:99)
`raw.get("answer_relevance", 0.5)` makes "the judge genuinely scored this
0.5" and "the JSON response was missing the field" indistinguishable — and
0.5 is the single most common value in the data (16/20 cases for
groundedness in the last run).

**Fix:** add a `valid: bool = True` field to `MetricScore`. A missing field
sets `valid=False`, `label="error"`. `_aggregate_score` skips
`valid=False` entries and the report prints "N metrics failed to score this
run" explicitly. This has to land before Step 1's noise runs are trusted —
otherwise parsing failures are silently mixed into what you're calling noise.

### 2b. Split the judge into two independent calls (1 hour) — DONE

**Current bug:** one call produces both scores and **one shared reasoning
string** — verified: 20/20 cases in the last run have identical reasoning
text for `answer_relevance` and `groundedness`. The groundedness reasoning
for case D03 explains whether the report answers the question, not whether
claims are traceable.

**Fix:** split `score_relevance_and_groundedness` into two functions, each
with its own prompt and its own reasoning. Split
`EVAL_RELEVANCE_SYSTEM` in `src/config/agent_prompts.py` into
`EVAL_RELEVANCE_SYSTEM` and `EVAL_GROUNDEDNESS_SYSTEM`.

**Also update:** the reuse check at
[src/eval/runner.py:113-122](../src/eval/runner.py:113) currently gates on
`"answer_relevance" in existing and "groundedness" in existing` together —
needs to check each independently now that they're separate calls.

**Cost impact:** 3 judge calls/case → 6. 20 cases = 120 calls, still a few
dollars. Don't economize here.

### 2c. Turn groundedness into a claim-list score (half day — highest-value change here) — DONE

Instead of asking the judge for a raw 0-1 groundedness score, have it emit
a structured claim list:

```json
{
  "claims": [
    {"claim": "North region revenue $1,363,760.55", "supported": true,  "evidence": "findings: North=1363760.55"},
    {"claim": "growth driven by new product line",   "supported": false, "evidence": null}
  ]
}
```

`groundedness = supported_count / total_claims`, computed in code.

Why this is worth the effort:
1. Score becomes genuinely continuous, not stuck on {0, 0.5, 1.0}.
2. Reasoning and score can no longer misalign — the claim list *is* the reasoning.
3. Directly auditable during Step 3 labeling — a human checks whether the
   claim list is right, instead of guessing why a 0.5 was assigned.
4. Extraction-and-counting is a task LLMs are meaningfully more reliable at
   than free-form 0-1 judgment.

Same technique applies to `factual_accuracy` later if useful, but that one
already has variance (rule-based number overlap) — lower priority.

### 2d. Rebuild the six zero-variance metrics (half day) — DONE (see binary-hit deviation noted above)

`tool_selection`, `plan_efficiency`, `chart_appropriateness`, `token_cost`,
`retry_count`, `error_rate` were all 1.00 on every one of 20 cases in the
last run — 33% of aggregate weight carrying zero information. Split into
two groups.

**Group A — rename honestly, move out of the aggregate.**
`error_rate` / `retry_count` / `token_cost` are operational/diagnostic
signals, not quality signals — a healthy system's `error_rate` *should* be
constantly 1.0; folding it into a quality average just dilutes it. Drop
these from `_aggregate_score`'s weight table, print them in a separate
"operational panel" section of the report instead. Keep `error_rate` as a
gate (alerts if it drops) without it counting toward the quality score.

Also delete the `completeness: 1.5` weight-table entry — it's a dead line;
`completeness` is documented in `metrics.py`'s docstring but never
implemented anywhere.

Delete `plan_efficiency` outright — "1-6 steps = full score" measures
nothing a genuinely better metric wouldn't already imply.

**Group B — actually rebuild.**

- **`tool_selection`**: currently `1 - failed_steps/total_steps`, i.e. "did
  nothing throw." Rename that to `step_success_rate`, move it to the
  operational panel (it has real diagnostic value, just not as
  "tool_selection"). Build a *new* `tool_selection` that checks whether the
  planner picked the right tool: add `expected_tools` to each golden case
  (from the 6 `TOOL_REGISTRY` keys — `sql_query`, `pandas_transform`,
  `statistical_test`, `anomaly_detection`, `time_series`, `comparison`),
  score by set overlap (`|actual ∩ expected| / |expected|`) since more than
  one tool choice can be legitimate for a given query.

- **`chart_appropriateness`**: `recommend_chart` in
  [src/tools/chart_tool.py:94](../src/tools/chart_tool.py:94) is
  rule-based and always returns a valid type — that's why this metric is
  structurally always 1.0. Add `expected_chart_types` per case (from `bar`,
  `horizontal_bar`, `line`, `scatter`, `histogram`, `box`, `heatmap`,
  `pie`). **Do not** derive the expectation from the same data-shape rules
  `recommend_chart` uses internally — that's grading the function against
  itself. The expected value has to come from an independent source (the
  case author's judgment).

**Resulting aggregate weight table** — only metrics with real variance:

| Metric | Weight |
|---|---|
| answer_relevance | 3.0 |
| groundedness | 3.0 |
| factual_accuracy | 2.0 |
| intent_accuracy | 1.5 |
| tool_selection (rebuilt) | 1.0 |
| chart_appropriateness (rebuilt) | 0.5 |
| **Total** | **11.0** |

### ⚠️ The aggregate will drop, and that's the point — CONFIRMED

Predicted: removing ~33% of free weight would likely pull the aggregate
from ~0.73 to somewhere near 0.5. Actual, measured 2026-07-30: **0.716 →
0.658** (a real single run, `logs/eval_runs/replay_1785446988.json`) — a
smaller drop than guessed, since some of the removed weight's slack was
offset by groundedness becoming a more literal, stricter claim-by-claim
score rather than a lenient holistic judgment. Either way:

> Post-refactor scores are not comparable to pre-refactor scores. The old
> 0.725/0.716 included ~33% weight from six zero-variance metrics (constant
> at full score). The rebuilt aggregate (0.658) contains only metrics with
> measured variance. This is a change of ruler, not a regression.

### Acceptance check — PASSED

Real replay run, 2026-07-30: every metric feeding the aggregate shows
std > 0 across the 20 cases (`answer_relevance` std=0.294,
`groundedness` std=0.424 across 8 distinct values, `factual_accuracy`
std=0.470, `intent_accuracy` std=0.151, `tool_selection` std=0.224,
`chart_appropriateness` std=0.410). No constant metric remains in the
aggregate. The operational metrics correctly still show std=0
(`retry_count`, `error_rate`) or near-0 (`token_cost`) — expected, since
they're weight=0 and not supposed to discriminate case difficulty.

---

## Step 3 — Human labeling and judge agreement

**STATUS: COMPLETE (2026-08-02).** Five independent labeling sessions
(original 20, batch 2's 20, batch 3's 20, batch 4+5's merged 40, and a
self-consistency retest of the original 20), 120 total labels across 100
distinct cases. Capstone result: human-vs-human self-consistency on
groundedness is 0.775 (acceptable) while every human-vs-judge groundedness
comparison lands at 0.04-0.26 (weak) — the human is reliable, the judge
disagrees with her consistently, and the mechanism is identified and
verified 4 times independently (a specific class of false superlative
claim the groundedness judge doesn't check). Full writeup, all numbers,
and the fix candidate for `EVAL_GROUNDEDNESS_SYSTEM` are in
[docs/judge_calibration.md](judge_calibration.md) — that's the living doc
for this step, this section stays as the original plan.
`scripts/export_for_annotation.py` + `scripts/_annotation_template.html`
(blind labeling page, published as a Claude Artifact) +
`src/eval/agreement.py` + `scripts/compute_agreement.py` (QWK/Spearman/MAE/
confusion matrix) are all built and verified (interactive browser
verification of the labeling page; synthetic smoke-test data, deleted
after, for the agreement CLI). **Labeling itself is explicitly the user's
work, not something to fabricate** — an AI-generated "human label" would
make this step measure agreement between two LLM outputs instead of real
human-model agreement, defeating the entire point.

The single highest-leverage step for an Eval/Model-Quality interview — the
one thing this project currently has zero evidence for.

### Free resource already sitting in the repo — CORRECTED, this was wrong

The original assumption below turned out to be false: `logs/eval_runs/*.json`
only stores computed scores (`EvalResult.to_dict()`), never the raw report/
findings/rag_context text that produced them — confirmed both here and
during Step 1's replay-cache work. There is no free 360-tuple pool sitting
in git history. The actual source used:
`logs/replay_cache/corpus.json` (built in Step 1/2), which *does* hold full
pipeline states for the current 20 golden cases — but only 20, not 360, and
all from the current code/data, not 18 historical variants of it.

<details><summary>Original (incorrect) plan text, kept for the record</summary>

No need to rerun anything to get raw material. `logs/eval_runs/` already
has ~18 runs × 20 cases ≈ 360 real (query, findings, report, judge score,
judge reasoning) tuples. Sample from history.

</details>

### What to label

Not "what should this case's final score be" — label what the judge is
actually asked to judge: given (query, findings, report), a human scores
relevance and groundedness independently.

Sample **60-100**. Fewer than 60 makes the agreement confidence interval too
wide to be useful.

### Sampling must be stratified

Two axes:
- **query_type** — descriptive / diagnostic / comparative / predictive / exploratory, proportionally represented
- **judge score band** — high / mid / low / refusal, all represented

The second axis matters more than it looks. Sampling only high-scoring
cases inflates agreement — easy cases are easy for everyone to agree on.
Disagreement lives in the middle band; that's what needs measuring.

### Three non-negotiable process details

1. **Blind labeling.** The export script must not show the judge's score.
   Seeing it anchors the human rating and inflates measured agreement.
2. **Calibrate before batching.** Label 20 → stop → write down edge cases
   encountered as rubric addenda → label the remaining 80. Batching without
   this calibration pass means the standard drifts partway through.
3. **Re-label 20 for self-consistency.** Single annotator, so
   inter-annotator agreement (IAA) is not measurable — the honest substitute
   is test-retest: re-label 20 of the same cases 24+ hours later and report
   self-agreement. State this limitation explicitly in the writeup — say it
   before being asked.

### Statistics

Scores are ordered 5-point categories, not nominal classes:

| Statistic | Why |
|---|---|
| **Quadratic Weighted Kappa (QWK)** | primary metric — penalizes a 2-band disagreement more than a 1-band one, matching ordered-data intuition. Plain Cohen's κ treats all disagreements equally, wrong tool here |
| **Spearman correlation** | secondary — measures rank agreement, insensitive to systematic offset |
| **MAE + exact-match rate** | intuitive supplements |
| **Confusion matrix** | most useful diagnostic — shows exactly where disagreement concentrates |

Rough benchmark: QWK > 0.6 acceptable, > 0.8 good.

### If agreement is low — this is the valuable branch, not a failure

Say QWK comes back at 0.45. Standard loop:

1. Read the confusion matrix, find the cell with the most disagreement.
2. Usually traces to an under-specified middle band ("mostly grounded, some
   unsupported" — how many unsupported claims counts as "some"?).
3. Rewrite that band with a concrete example.
4. Re-run the judge on the same labeled set (no re-labeling needed) and
   recompute QWK.
5. Report the delta.

"QWK went from 0.45 to 0.71 by rewriting the mid-band rubric" is, on its
own, a complete demonstration of eval-engineering work.

### Reuse the same labels to settle the judge-model question

Once labeled, the same 100 examples score multiple candidate judge models
without re-labeling:

| Judge model | QWK | Cost/run |
|---|---|---|
| claude-sonnet-5 | ? | ? |
| claude-haiku-4-5 | ? | ? |
| gpt-4o-mini | ? | ? |

This turns the "why this model" question from a justification into an
experimental result: "haiku-4.5 QWK 0.68, sonnet-5 QWK 0.74 at 5x the cost —
0.06 of agreement doesn't justify 5x spend, so the judge runs on haiku."

### Deliverables

- `data/annotations/human_labels.json` (checked into the repo)
- `scripts/export_for_annotation.py` (stratified sample + blind export)
- `scripts/compute_agreement.py` (QWK / Spearman / confusion matrix)
- `docs/judge_calibration.md` (results + explicit methodology limitations)

### Time

~3-5 min/label (reading the report + checking against findings), 100 labels
≈ 5-8 hours. Split across two days — labeling quality degrades with fatigue
late in a single session.

---

## Step 4 — Expand cases and establish a dev/test split

**STATUS (2026-08-01): consolidation done, suite expanded to 100 cases
across 5 batches (the user's original target), dev/test split done
(36→49→59 dev as the suite grew), ground-truth audit script done.**

Consolidation completed exactly per the plan below:
- `data_source` moved into `tests/eval/test_suite.json` per case; `CASE_DATA_SOURCES`
  deleted from `scripts/run_eval.py` (and the two scripts that imported it,
  `generate_replay_corpus.py`/`measure_noise.py`, now read `tc.data_source`
  directly).
- `KNOWN_DATA_MISMATCH` deleted; replaced by a `"data_mismatch"` tag on
  each case (`DG04`, `C03`, `P03`, plus new case `P06` — see below).
- `_builtin_golden_suite()` deleted from `src/eval/runner.py`.
  `load_golden_suite()` now raises `FileNotFoundError` if the JSON is
  missing instead of silently falling back to a hardcoded copy.
- `GoldenTestCase` gained a `split: Optional[str] = None` field.

**80 new cases added across four batches**, bringing the suite to 100 —
the user's originally requested target:

- **Batch 2** (`D06-D09`, `DG05-DG08`, `C05-C08`, `P04-P07`, `E05-E08` — 20
  cases): a second false-premise case (`DG08` — Corporate vs. Consumer
  segment revenue are within 0.4% of each other, not "significantly
  less"), a fourth absent-data case (`P06` — no CSAT/satisfaction column
  exists anywhere), the first deliberately ambiguous case (`E05` — "Show
  me the numbers.", no metric or dimension named), and three cross-table
  joins that (unlike `C03`) actually resolve (`D06`, `D09`, `C06`).
- **Batch 3** (`D10-D13`, `DG09-DG12`, `C09-C12`, `P08`/`P09`/`P11`,
  `E09-E13` — 20 cases): another false-premise case (`DG09` — Display's
  click-through rate is statistically indistinguishable from every other
  channel's, 2.70%-2.79%, not a real underperformance), a fifth
  absent-data case (`DG12` — no loyalty-tier column exists anywhere), and
  a second ambiguous case (`E13` — "How are we doing?"). Also opened up
  previously-untouched columns/tables: sales rep performance (`rep_id`),
  units sold, order discounts, plan-tier monthly charges, customer
  cohort/join dates.
- **Batch 4** (`D14-D17`, `DG13-DG16`, `C13-C16`, `P12-P14`, `E14`/`E15`/`E17`-`E19`
  — 20 cases, 2026-08-01): a third false-premise case (`DG13` — Books
  actually generates slightly *more* revenue than Furniture, and a higher
  margin, despite the query's framing), a sixth absent-data case (`DG16`
  — no payment-method column anywhere), a third ambiguous case (`E19` —
  "What's next?"). Opened up `order.quantity` (never touched before, and
  quantized 1-10 so genuinely has zero IQR outliers — a clean "correctly
  reports nothing anomalous" case rather than another data-mismatch),
  campaign counts per channel, subscription-plan distribution, and two new
  correlation pairs (unit price vs. margin, tenure vs. monthly charges).
  Also surfaced a real, unprompted guardrail catch worth knowing about: one
  case's report leaked raw customer names/emails from a `customers` table
  join, and the guardrail correctly refused it as PII exposure --
  interesting pipeline behavior, not something fixed here.
- **Batch 5** (`D18-D21`, `DG17-DG20`, `C17-C20`, `P15-P17`, `E20-E24` — 20
  cases, 2026-08-01, requested together with batch 4 to reach 100): a
  fourth false-premise case (`DG19` — order discount and order quantity
  have essentially zero correlation, 0.0083, contradicting the query's
  framing that larger orders get bigger discounts), a seventh absent-data
  case (`DG20` — no referral-source/acquisition-channel column exists
  anywhere). `DG17`/`C20` (monthly charges vs. churn) is a genuine, if
  weak, real pattern worth noting as a *non*-false-premise diagnostic case
  for contrast: correlation -0.0864, retained customers pay more on
  average ($63.58 vs. $56.58) -- the premise holds, just weakly. No new
  ambiguous case this batch (3 already in the suite was judged enough).
  **Batches 4 and 5 were merged into a single 40-case blind labeling tool
  at the user's request** (`export_for_annotation.py`'s existing `--cases`
  filter made this direct — batch 5 scored fresh, batch 4's *already-scored*
  answer key and blind samples reused rather than re-billing the judge to
  rescore unchanged cached reports) rather than kept as two separate
  20-case tools like batches 2 and 3 were. See
  [judge_calibration.md](judge_calibration.md) for the merged tool link;
  batch 4's original standalone link is superseded, not to be used.

Ground truth for every non-"unanswerable" case across all four batches
was computed directly from `data/demo/*` (pandas/sqlite, not hand-guessed)
before the case was written, and batches 4-5's cases were registered
directly in `scripts/compute_ground_truth.py` (below) as they were
authored rather than retrofitted afterward.

**STATUS (2026-07-31, later): dev/test split done.** `scripts/assign_dev_test_split.py`
stratifies on (query_type, is_hard_case) at a 60:40 target, seeded and
persisted in `data/eval_split_manifest.json` (the source of truth --
`test_suite.json`'s `"split"` field is a synced copy). Designed to be
growth-safe: re-running it after more cases are added only assigns splits
to the *new* case ids and never touches an existing assignment, so a case
already used for tuning can't silently flip into the holdout later.
Current suite (after batches 4-5, below): 59 dev / 41 test, 100 total --
re-running the script for each new batch only assigned splits to that
batch's new case ids (verified both times: every bucket's "already
assigned" dev/test counts matched the pre-batch totals exactly), no
existing case's assignment was ever touched. The gate itself is live in both
`scripts/run_eval.py` and `scripts/replay_eval.py` (`--split dev` default,
`--split test` required explicitly, prints a "this should be an official
reveal" warning, and the split used is recorded in the output report
JSON); `replay_eval.py` also accepts `--split all` for split-agnostic
work. `--cases` is still bounded by `--split` (a case outside the
requested split is skipped with a message, not silently included) so it
can't be used as a side-door around the gate.

**Important caveat this split does NOT solve on its own**: it stratifies
*by case*, cutting across labeling batches, not aligned with them. The
split is valuable structural hygiene going forward (all future
case-authoring, prompt changes, and pipeline changes should respect it),
but it was never going to solve the *specific* problem of validating the
groundedness-prompt fix (Step 3's A2) against data Claude hasn't already
reasoned about.

**RESOLVED (2026-08-02): user chose option (b).** Fixed
`EVAL_GROUNDEDNESS_SYSTEM` and re-scored the existing 100+20 labels,
explicitly framed as re-evaluation against previously-collected labels,
not a blind holdout — see [judge_calibration.md](judge_calibration.md)'s
"A2 — Fixing `EVAL_GROUNDEDNESS_SYSTEM`" section for the caveat, the fix
itself, the direct mechanism check (4/4 of the known Product-2-margin
cases now correctly flagged), and the aggregate result (pooled n=100
groundedness QWK 0.141 → 0.185, a real but modest ~31% relative
improvement that doesn't fully close the gap).

**STATUS (2026-08-01): `scripts/compute_ground_truth.py` done, extended
cleanly through batches 4-5** -- one small registered function per case id
(76 of 100 have checkable numeric ground truth; the other 24 are
predictive/data_mismatch/ambiguous and are intentionally skipped, matching
`_KNOWN_UNANSWERABLE_CASES` in `test_phase9.py`), each recomputing that
case's `ground_truth` directly from `data/demo/*`. `--check` (default)
reports mismatches without writing; `--write` overwrites. Running it as a
first audit against the (then-60-case) suite (including the original 20,
never previously re-verified in this project) found two real,
previously-unnoticed errors, both fixed:
- **C02**: stored `search_conversion_rate_pct: 6.41` / `display_conversion_rate_pct: 1.22`
  didn't match either of two independent recomputations (aggregate
  conversions/clicks, or mean of per-campaign rate), both of which agree
  on 6.49% / 1.20%. No way to recover the original derivation; corrected
  to the freshly verified, reproducible value via `--write`.
- **D09** ("which single order generated the most revenue"): `top_order_id: 6553`
  turned out to be one side of a genuine 2-way tie at the max revenue
  value ($4,992.20) -- a second order (`5810`) has the identical revenue.
  Both tied orders are UK customers, so `top_order_revenue` and
  `top_order_customer_country` are both stable regardless of tie-break
  order, but `top_order_id` isn't a well-defined single answer -- it just
  depends on which row a merge/`idxmax()` happens to return. Dropped
  `top_order_id` from the case's ground truth rather than pin it to
  implementation-detail row ordering (the query itself never asked for
  the order id specifically).

Neither fix touches any already-collected human label or judge score --
`ground_truth` only feeds `score_factual_accuracy` (checks whether cited
numbers appear in the report), which is fully independent of the
groundedness/relevance judge calls the 40 collected labels were measuring
agreement against. It does mean future `run_eval.py`/`replay_eval.py`
runs will score C02/D09's `factual_accuracy` more accurately than before.

Registering batches 4-5 in `compute_ground_truth.py` as they were authored
(rather than retrofitting afterward, as batches 1-3 needed) meant their
first `--check` run came back clean immediately -- 0 mismatches, no
C02/D09-style surprises. All four new batches were generated via the real
pipeline (`scripts/generate_replay_corpus.py --cases ...`, ~$0.30-0.36 per
20-case batch) and scored with the current judge to build answer keys
(`answer_key_1785525212.json` batch 2, `answer_key_1785544815.json` batch
3, `answer_key_1785548910.json` batch 4, `answer_key_1785549448.json`
batch 5 -- batches 4+5 also merged into `answer_key_batch4_5_merged.json`
for the combined tool). Batches 2 and 3 got their own separate blind
labeling tools; batches 4 and 5 were merged into one 40-case tool at the
user's request instead (`export_for_annotation.py`'s `--cases` filter,
present since batch 2, made this a direct case of scoring batch 5 fresh
and reusing batch 4's already-computed answer key rather than re-billing
the judge). All tools are kept apart from the original 20's
self-consistency retest so none of them contaminate each other — see
[judge_calibration.md](judge_calibration.md) for every live link.

**DONE (2026-08-02)**: [roadmap.md #25](roadmap.md) and
[#26](roadmap.md) — the Planner sometimes ran a comparison/significance
test on data a prior step already aggregated to one row per group (found
via the user's own judge-calibration labeling, case C01 — see
[judge_calibration.md](judge_calibration.md)), and the Guardrail's own
hallucination check had a truncation bug the eval judge's fix never
touched (case C02). Both deliberately deferred until all 100 cases were
labeled, then fixed together; both are generation-relevant, so the full
replay corpus was regenerated afterward ($1.82, ~15 min, 100/100 cases
fresh). Full detail in [roadmap.md](roadmap.md)'s #25/#26 entries.

**Before/after on the dev split (59 cases), same replay-mode measurement
both times** (`scripts/replay_eval.py`, no other code changed between the
two runs):

| | Before (pre-#25/#26) | After (post-#25/#26) |
|---|---|---|
| Overall aggregate | 0.751 | 0.779 |
| `answer_relevance` (mean) | 0.737 | 0.797 |
| `groundedness` (mean) | 0.789 | 0.822 |
| `factual_accuracy` (mean) | 0.655 | 0.689 |
| Safe refusals | 20/59 (33.9%) | **7/59 (11.9%)** |

**The aggregate move (+0.028) is not, on its own, distinguishable from
noise — check this against a measured floor before trusting it.**
`docs/noise_floor.md` measured `overall_aggregate`'s 2σ at **0.028** across
8 identical reruns of the same 20-case suite with nothing changed. The
observed +0.028 sits exactly at that boundary. Reported as a genuine
finding anyway, but explicitly *not* as "the fix improved the aggregate" —
this is exactly the kind of number Step 1 exists to keep people from
over-claiming.

**The refusal-rate drop is the real signal, but it also needed calibrating
against noise before treating it as one — same process, more careful this
time.** Refusal counts from those same 8 identical dev-suite reruns
(`logs/noise_runs/full_noise_1785391747.json`, n=20 each): 7, 7, 7, 8, 9,
10, 10, 11 refusals — mean 8.6, std 1.6, i.e. **refusal rate itself
naturally swings ~35%-55% with zero code change**, and 10 of the 20 cases
flip between "refused" and "passed" across otherwise-identical runs.
Scaling that std to n=59 puts the noise band at roughly **±9 points**. The
observed drop, 33.9% → 11.9% (22 points), is about **2.4x that band** —
outside it, a real effect, not a coincidence of which run happened to
land where. The single strongest piece of evidence for this: **C02, DG03,
and E03 were refused in 8/8 identical pre-fix reruns** (not
probabilistically — every single time) and now pass cleanly; a
deterministic failure flipping to a pass isn't explainable by rerun noise.

Of the 15 cases that stopped being refused, direct inspection of the
guardrail's actual before/after reasoning (not just the pass/fail label)
sorts into two groups:
- **12 genuinely fixed** by #26 (C02, DG04, DG11, DG03, E03, E12, E13, P03,
  P07, P09, plus 2 more) — the pre-fix refusal reason was some form of "X
  isn't supported by the findings," and direct verification confirms the
  cited number *was* real, just outside the old 800-char/`result_summary`-
  only context (e.g. DG04's guardrail context, reconstructed exactly:
  correlation coefficients 0.732/0.6142/0.024/-0.0096 appear zero times in
  the old inline context — `result_summary` never included the `result`
  dict at all, a worse gap than "truncated," it was structurally absent —
  and all four appear in the new `render_findings`-based context).
- **2 not clearly attributable to the fix** (C19: a rounding-precision
  complaint, 10.00% vs. 9.9968%, unrelated to truncation; D19: fails a
  separate rule-based check, `_check_population_claim_grounding`, that
  never calls the LLM at all and #26 cannot have touched — its
  pass/fail here is plausibly just this run's own stochasticity).

**A structural gap in the guardrail surfaced while doing this
verification, now tracked as [roadmap.md #27](roadmap.md):** severity
(blocking vs. warning-only) is decided by *substring-matching the LLM's
own self-chosen check name* against a fixed keyword list
(`pii`/`safety`/`hallucin`/`fabricat`/`claim_ground`/`grounding`). Of
`GUARDRAIL_SYSTEM`'s four requested checks, only `Hallucination` and `PII
leakage` match; `Factual accuracy` and `Misleading framing` structurally
can never block, no matter how wrong the finding is. Confirmed live: 16 of
100 post-fix cases have a failed `Factual accuracy`/`Misleading framing`
check that was delivered anyway with just a caveat — including case P05,
where the report claims "May 2023 had the highest churn rate at 13.74%"
when March 2024's 39.07% is actually highest (the guardrail's own
`Factual accuracy` check correctly caught this, then didn't block on it).
That same P05 case is also tracked as [roadmap.md #28](roadmap.md): a
false-superlative claim caught live in the *generation* layer (the Insight
Agent), not just the eval judge — suggesting the pattern A2 fixed on the
judging side (missing "highest"/"lowest" claims that contradict other
visible values) may also exist upstream in what gets generated in the
first place. Neither #27 nor #28 fixed here — found while interpreting
these results, not chased down yet.

**Caveat, stated for the same reason as the earlier A2 caveat**: this
before/after uses the dev split as defined (established *after* #25/#26
were already known bugs, though before they were fixed), so it's a real,
useful signal — but the formal "official reveal" is the held-out test
split, not run yet as of this writing; see the note above about what the
dev/test split can and can't prove given when it was created relative to
this project's own iteration history.

### Fix a structural problem before expanding anything

The golden suite is currently defined in **four places** that will
desynchronize the moment the suite grows past 20 cases:

| Location | Content | Problem |
|---|---|---|
| `tests/eval/test_suite.json` | 20 cases | the real source |
| [src/eval/runner.py:263](../src/eval/runner.py:263) `_builtin_golden_suite()` | the *same* 20, hardcoded in Python | pure liability — editing the JSON doesn't touch this |
| [scripts/run_eval.py:47](../scripts/run_eval.py:47) `CASE_DATA_SOURCES` | case id → data source, hardcoded dict | every new case needs edits in two files |
| [scripts/run_eval.py:70](../scripts/run_eval.py:70) `KNOWN_DATA_MISMATCH` | hardcoded set | same |

Also: `GoldenTestCase` already has a `data_source` field
([src/eval/runner.py:41](../src/eval/runner.py:41)) that the harness never
actually reads — it uses the separate hardcoded dict instead. Same disease
as the unused `completeness` weight.

**Consolidation (half day, must happen before expanding case count):**
- Move `data_source` into the JSON, delete `CASE_DATA_SOURCES`.
- Replace `KNOWN_DATA_MISMATCH` with a `data_mismatch` tag in each case's `tags` list.
- **Delete `_builtin_golden_suite()`** — fail loudly if the JSON is missing.
  Keeping a hardcoded fallback copy in sync has no upside.
- Add fields to `GoldenTestCase`: `split` (dev/test), `expected_tools`,
  `expected_chart_types` (needed by Step 2d).

### Growing to 100-200 cases

Don't hand-write 200 cases.

1. **Systematic generation**: 5 query types × 4 datasets × templates,
   machine-generate candidates, human-review to keep the sensible ones.
2. **Ground truth computed by a checked-in script**: add
   `scripts/compute_ground_truth.py`. Currently ground truth values only
   have a comment pointing at "the pandas computation in test_suite.json" —
   not reproducible or auditable. A script means ground truth can be
   recomputed whenever the underlying data changes.
3. **Deliberately include hard cases:**

| Hard-case type | Tests | Current count |
|---|---|---|
| asks about something absent from the data | refusal behavior | 3 (DG04/C03/P03) — expand |
| genuinely ambiguous query | the clarification path (`route_after_intent`'s `clarify` branch is barely exercised today) | 0 |
| requires a cross-table join | Phase B capability | few |
| requires a derived column | date-part extraction etc. | few |
| embeds a false premise | whether the system pushes back or fabricates agreement (DG03 — "why is North underperforming" when North is actually the top region) | 1 — expand |

The false-premise category is the most valuable one already in the suite.
DG03 is the strongest case in the current 20 precisely because it tests
whether the system resists a leading question instead of confabulating. More
of these make the suite look meaningfully more sophisticated.

### Splitting

- Stratification key = (query_type, is_hard_case)
- **dev : test = 60:40** or 50:50
- **Fixed seed, split recorded in the JSON** (`"split": "dev"`), not
  re-randomized per run — otherwise it's not reproducible
- Test-set ground truth is computed the same way, but its score is not
  looked at during development

### Engineering gate to enforce discipline

```
python scripts/run_eval.py                 # dev only, default
python scripts/run_eval.py --split test     # must be explicit
```

Print a warning on `--split test`: "You are running against the holdout
set — this result should be recorded as an official reveal." Not
ceremony — it prevents accidental peeking, and a holdout set that's been
peeked at once is compromised for good. The report JSON should record which
split was used, so runs don't get confused after the fact.

### Deliverable

> Dev set N cases, test set M, stratified split on (query_type,
> is_hard_case). All metric tuning and validation of the 35 historical
> fixes happened on dev only; the test set was never scored during that
> process. Final reveal: dev aggregate = X, test aggregate = Y, gap = Z.

The dev/test gap is itself the most persuasive number here:
- small gap (within noise floor) → improvements generalized, not overfit
- large gap → overfitting happened, and it was caught

A large gap, reported honestly, is a stronger interview signal than no gap
reported at all — it demonstrates the ability to detect the failure mode,
which is the actual skill being evaluated.

---

## If time is short: minimum viable version (3 days)

1. **2a + 2b + 2d** (1 day) — get metrics to show real variance, baseline decency
2. **Step 1, judge-noise only** (half day) — cheap, produces one citable number
3. **Step 3, compressed to ~50 labels** (1.5 days) — wider CI on QWK, but
   having the number at all is the difference that matters

Cut Step 4. State honestly in the roadmap: "No holdout established yet —
current findings should not be read as claims of generalization." The
existing roadmap's habit of stating limitations plainly is a strength —
keep doing that here.
