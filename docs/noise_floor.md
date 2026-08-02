# Noise Floor — Eval v2, Step 1

Measured 2026-07-30 using `scripts/measure_noise.py` (see
[docs/eval_v2_plan.md](eval_v2_plan.md) for the full 4-step plan this is
step 1 of). Raw data: `logs/noise_runs/judge_noise_1785385878.json`,
`logs/noise_runs/full_noise_1785391747.json`.

## TL;DR

- **Judge noise, on a fixed input, is near zero.** 10/12 (case, metric)
  combinations had exactly 0 variance across 20 repeats. The two that
  didn't were fully absorbed by the harness's existing median-of-3
  procedure. Judge randomness is not a meaningful noise source here.
- **Full-pipeline noise (generation + judging stacked) is the real number
  to compare future score changes against**: `overall_aggregate` std=0.014,
  **2σ = 0.028** across 8 identical reruns of the full 20-case suite. A
  score change smaller than ~0.03 on the top-line aggregate is not
  distinguishable from noise at this sample size.
- **Per-case noise is larger and uneven**: median std=0.060 (2σ≈0.12)
  across the 20 cases, ranging from near-zero (C04, P02, D02: std<0.01) to
  0.137 (E01, an open-ended "give me an overview" exploratory query).
  Comparisons of a *specific* case's score should use the paired-bootstrap
  method (below), not a flat threshold.
- **A real bug surfaced while measuring this**, unrelated to noise
  per se: two silent fail-open fallbacks in the pipeline (planner, guardrail)
  — see "Incident" below. **Fixed 2026-07-30** (retry-with-backoff, same
  pattern as the judge fix, see `src/utils/retry.py`).
- The historical regression-detection threshold in `src/eval/runner.py`
  (flat 5% drop = warning) turns out to be roughly in the right ballpark for
  the top-line aggregate (measured 2σ≈2.8%) — not something to be
  embarrassed about, but also not something that was previously validated
  against anything.

## How this was measured

Two noise sources, isolated separately (see `scripts/measure_noise.py`):

- **`judge` mode**: 6 fixed (query, report, analysis_results, rag_context)
  tuples — one real pipeline run each, cached once — then the judge
  re-invoked 20x per tuple, both as a single raw call (`n_samples=1`) and as
  the harness's actual median-of-3 procedure. Isolates judge-only noise
  from generation noise.
- **`full` mode**: the entire 20-case golden suite rerun 8x end-to-end,
  identical config, nothing changed between runs. Measures the number that
  actually matters for judging whether a future score change is real:
  generation noise + judging noise stacked, exactly as a before/after
  comparison would see it.

## Judge-only noise (fixed input, repeated judging)

6 cases spanning query types (D01 descriptive, DG01/DG03 diagnostic — DG03
is the false-premise case, C02 comparative, P01 predictive, E02
exploratory/anomaly), 20 repeats each.

| Case | Single-call std (rel / gnd) | Median-of-3 std (rel / gnd) |
|---|---|---|
| D01  | 0.000 / 0.000 | 0.000 / 0.000 |
| DG01 | 0.000 / 0.000 | 0.000 / 0.000 |
| DG03 | 0.000 / 0.112 | 0.000 / 0.000 |
| C02  | 0.000 / 0.000 | 0.000 / 0.000 |
| P01  | 0.000 / 0.000 | 0.000 / 0.000 |
| E02  | 0.183 / 0.000 | 0.000 / 0.000 |

10/12 cells: exactly zero variance across 20 repeats. The two non-zero
cells (DG03's groundedness, E02's relevance) show variance only at the
*single-call* level, and in both cases the existing median-of-3 aggregation
already reduces that variance to zero. **Conclusion: for this
model/provider (gpt-4o via the resolved eval provider — see caveat below),
judge randomness at temperature=0 is not the dominant driver of the
historical 0.70–0.78 baseline swings documented in
[eval_report.md](eval_report.md).** That leaves generation noise as the
main suspect, measured next.

A pooled-across-cases view of the same data shows much larger std
(0.19–0.20) — that number mixes real between-case differences (D01 should
score near 1.0, DG01/DG03/C02/P01 are meant to be harder) with actual noise,
and is *not* a noise-floor number. `scripts/measure_noise.py`'s reporting
was fixed mid-measurement to lead with per-case numbers instead, after this
distinction became clear (see git history on that file).

**Caveat**: `settings.resolved_eval_provider` currently resolves to
`openai` (the same provider as the agent pipeline itself), not `anthropic`,
because `ANTHROPIC_API_KEY` in this environment isn't a real key (see
`_looks_like_real_key`). The self-preference concern the judge-tiering
design was meant to avoid (`resolved_eval_provider`'s docstring in
`src/config/settings.py`) is presently live. Unrelated to noise, but worth
fixing before trusting judge scores as fully independent evidence.

## Full-pipeline noise (generation + judging, clean run)

8 full reruns of the 20-case suite, concurrency=2, nothing changed between
runs. Total cost: $2.93.

**Top-line:**

```
overall_aggregate: mean=0.716  std=0.014  2σ=0.028  range=[0.688, 0.733]
trial values: [0.7173, 0.7273, 0.7064, 0.7142, 0.6882, 0.724, 0.7325, 0.7193]
```

**Per-metric (mean-across-20-cases-per-trial, variation across the 8 trials):**

| Metric | mean | std | 2σ |
|---|---|---|---|
| answer_relevance | 0.613 | 0.030 | 0.060 |
| groundedness | 0.487 | 0.042 | 0.085 |
| factual_accuracy | 0.524 | 0.020 | 0.040 |
| intent_accuracy | 0.784 | 0.010 | 0.020 |
| tool_selection | 0.971 | 0.033 | 0.065 |
| plan_efficiency | 0.998 | 0.005 | 0.011 |
| chart_appropriateness | 0.994 | 0.012 | 0.023 |
| token_cost | 0.982 | 0.001 | 0.001 |
| retry_count | 1.000 | 0.000 | 0.000 |
| error_rate | 1.000 | 0.000 | 0.000 |
| safe_refusal | 0.431 | 0.080 | 0.160 |
| total_latency | 0.569 | 0.035 | 0.070 |

Note `tool_selection`/`plan_efficiency`/`chart_appropriateness` are *not*
perfectly constant here even though the original audit
(`phase_d_model_tiering.json`, a single run) found them constant across the
20 *different* cases within that one run. Both things are true at once:
these metrics don't discriminate between an easy and a hard case within a
run (the original critique, still valid — see `docs/eval_v2_plan.md` Step
2d), but they do have real run-to-run flakiness (occasional tool-step
failures that get repaired, occasional chart-generation hiccups) that a
single run can't reveal. Neither observation cancels the other; both matter
for the Step 2 redesign.

`safe_refusal` mean here is 0.431 (43%), with real spread (std=0.080) — a
meaningfully different (and less precise) number than the "30% refusal
rate" quoted from the single historical run earlier in this project. The
true refusal rate on this suite is closer to 35–50% and is itself noisy;
don't quote a single-run refusal percentage as if it were exact.

**Per-case noise** (how much does one case's *own* aggregate score vary
across the 8 identical reruns):

| Case | std | 2σ | Case | std | 2σ |
|---|---|---|---|---|---|
| C01 | 0.085 | 0.169 | DG02 | 0.055 | 0.110 |
| C02 | 0.048 | 0.097 | DG03 | 0.119 | 0.237 |
| C03 | 0.113 | 0.225 | DG04 | 0.076 | 0.152 |
| C04 | 0.003 | 0.006 | E01  | 0.137 | 0.274 |
| D01 | 0.077 | 0.155 | E02  | 0.063 | 0.127 |
| D02 | 0.005 | 0.009 | E03  | 0.035 | 0.069 |
| D03 | 0.086 | 0.172 | E04  | 0.048 | 0.096 |
| D04 | 0.056 | 0.112 | P01  | 0.127 | 0.255 |
| D05 | 0.054 | 0.108 | P02  | 0.004 | 0.009 |
| DG01 | 0.050 | 0.099 | P03  | 0.080 | 0.160 |

Median per-case std = 0.060 (2σ ≈ 0.12). Highest: E01 ("give me an
overview of this dataset" — open-ended, so the report's content and
therefore its judged scores vary the most between generations). Lowest:
C04, P02, D02 (std < 0.01 — narrow, well-constrained queries with an
essentially fixed correct answer). **When investigating whether a specific
case's score change is real, use the per-case paired-bootstrap check below
— a flat threshold is wrong here because the noise itself varies 40x
case-to-case.**

## Bootstrap calibration check

`src/eval/noise.py`'s `paired_bootstrap_ci` was validated by running it on
all C(8,2)=28 possible pairs among the 8 identical trials (nothing actually
differs between any pair — this is a null comparison by construction).

**Clean run**: 1/28 pairs flagged "significant" at 95% CI — matching the
expected ~1.4/28 false-positive rate for a correctly-calibrated 95% test
under pure noise. The method is calibrated correctly.

## Incident: concurrent measurement jobs contaminated the first full-pipeline run, and surfaced a real bug

The first attempt at this full-pipeline measurement was run *concurrently*
with the judge-noise measurement, both hitting the same OpenAI org's rate
limit (30,000 TPM / 500 RPM — a low ceiling discovered during this work).
That run's pairwise bootstrap calibration came back at **9/28 pairs
flagged significant** — far above the ~1.4 expected under pure noise, with
trial 0 significantly different from every other trial (7/7 comparisons).

Root-caused by timestamp correlation: 81 of 83 rate-limit hits in that run
landed between 04:31:46–04:46:22, exactly the window the concurrent
judge-noise job was also running. Once that job finished, the full-pipeline
run hit zero further rate limits for its remaining trials — strong evidence
the effect was contention between the two jobs, not organic noise.

**The 81 contaminated hits exposed two silent fail-open fallbacks that had
nothing to do with the eval judge and were, at the time, still live**:

- `src/agents/analysis_agent.py`'s `plan()`: any exception during plan
  generation (including a transient 429) was silently replaced with an
  **empty plan** — no retry.
- `src/agents/guardrail_agent.py`'s `_llm_judge()`: any exception during
  the guardrail's LLM judge call (including a transient 429) **silently
  defaulted `hallucination_check` and `claim_grounding` to pass** — no
  retry.

The second one was a genuine safety concern, not just a measurement
nuisance: the guardrail's entire purpose is catching hallucinated/unsafe
output, and under load — exactly the condition where a system is most
likely to be producing degraded output — it silently waved everything
through instead of blocking or failing closed.

**Fixed 2026-07-30.** Both now retry a rate-limit-classed exception with
exponential backoff via a shared helper,
[`src/utils/retry.py`](../src/utils/retry.py)'s
`call_with_rate_limit_retry` (also what `src/eval/metrics.py`'s
`_judge_once` was refactored onto, consolidating what used to be three
near-identical inline implementations). Only a non-rate-limit failure, or a
rate limit still failing after retries are exhausted, falls through to the
original fallback (empty plan / default-pass) — unchanged from before,
since that part was never the problem. Verified with new tests:
`tests/unit/test_retry.py` (the shared helper in isolation),
`tests/unit/test_phase5.py::test_plan_retries_on_rate_limit_instead_of_going_straight_to_empty_plan`,
and `tests/unit/test_phase8.py::test_llm_judge_retries_on_rate_limit_instead_of_defaulting_to_pass`
— the last one specifically proves a real hallucination finding survives a
rate-limited retry instead of being silently overwritten with "pass".

Practical lesson for future noise/regression measurements: **don't run
multiple API-heavy jobs concurrently against this account** — the
resulting contention doesn't just add generic noise, it can trigger
specific, non-random failure modes that masquerade as a real trend
(9/28 "significant" pairs looked like *something changed*, when the
real cause was two of my own scripts fighting over a 30K-token-per-minute
budget).

The contaminated run's report file was deleted after diagnosis; the
supporting log excerpts above are preserved in this document instead.

## What this means for Step 2 and beyond

- **Regression threshold**: the existing flat 5% warning threshold in
  `detect_regressions` (`src/eval/runner.py`) is closer to correct for the
  *top-line aggregate* than it had any right to be (measured 2σ≈2.8%) — it
  was never validated before, and now it can be. Recommend tightening it to
  something nearer 3% for the aggregate specifically, and switching
  per-case regression checks to the paired-bootstrap method rather than a
  flat percentage, given how much per-case noise varies (0.006 to 0.274).
- **The 35 historical fixes** in `eval_report.md` were validated against a
  baseline timeline swinging 0.70–0.78 (a 0.08 range). This clean
  measurement's 2σ≈0.028 doesn't fully explain that range — either several
  of those fixes had real (if individually modest) effects, or some
  historical runs experienced contamination similar to what was just
  diagnosed here (this account's rate limit is low enough that any run
  sharing it with other work is a candidate). This is now an answerable
  question rather than a guess: rerunning specific historical before/after
  pairs with the paired-bootstrap method would settle it case by case, time
  permitting.
- **Judge independence caveat** (self-preference risk, provider currently
  same as the agent's) should be fixed before Step 3's human-labeling work,
  since Step 3 explicitly measures judge trustworthiness and an
  unintentional same-provider setup undermines that.
- **The two silent fail-open fallbacks** are a legitimate, separate
  follow-up — most naturally fixed the same way `_judge_once` was fixed
  (retry-with-backoff on rate-limit-class errors, don't catch-and-degrade
  everything else identically), given `_is_rate_limit_error` in
  `src/eval/metrics.py` is now generic enough to reuse.
