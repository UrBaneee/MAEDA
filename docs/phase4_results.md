# Phase 4 Results: Does Cleaning and Retrieval Improve MAEDA's Reports?

*Reveal completed 2026-11-12. Pre-registration: [`phase4_preregistration.md`](phase4_preregistration.md), frozen 2026-11-11 (§10.15, commit `0a9edbc`).*

## Summary

Of the three pre-registered confirmatory hypotheses, two hold and one was not detected. Data cleaning
significantly improves numeric accuracy, retrieval significantly improves cause attribution, and no
interaction between the two was detected.

| Hypothesis | Comparison | Primary metric | Effect | Test | Verdict |
|---|---|---|---|---|---|
| H1: cleaning net gain | C − B | `factual_accuracy` | +0.189 | 98.3% CI [+0.114, +0.272] | **Holds** (worth it) |
| H2: retrieval net gain | R − B | `event_attribution_accuracy` | +0.653 | exact sign-flip, p = 0.002 | **Holds** |
| H3: interaction | (CR − R) − (C − B) | `factual_accuracy` | −0.009 | 98.3% CI [−0.047, +0.028] | Not detected |

- Multiplicity: Bonferroni over three tests, α = 0.0167. H1 and H2 must also clear a 0.028 practical
  floor; both clear it by a wide margin.
- The effects are significant, but the absolute level is still low. With cleaning on, only about 26% of
  numeric questions are answered correctly, because the cleaner does not repair the ×10 revenue outliers.
- Auto mode (arm A) never triggered cleaning, so on numeric questions it performs like all-off. This
  matches a prediction written before the reveal (see Experiment 2).

## Design

The system under test is MAEDA, a LangGraph multi-agent data analyst. It calls two independent services
over MCP: data cleaning ([agentic-data-cleaner-v2](https://github.com/UrBaneee/agentic-data-cleaner-v2)) and
retrieval ([rag-framework](https://github.com/UrBaneee/rag-framework)).
The experiment only switches these two services; everything else is frozen.

| Arm | Cleaning | Retrieval | Used for |
|---|---|---|---|
| B | off | off | baseline for H1, H2, H3 |
| C | forced on | off | H1, H3 |
| R | off | forced on | H2, H3 |
| CR | forced on | forced on | H3 |
| A | auto | auto | Experiment 2 (routing); uses none of the confirmatory budget |

**Samples**

- **H1 / H3:** the 119 test questions of the H1 holdout (all `cleaning_relevant`), 2 runs per arm per
  question. The data is a synthetic demo sales table generated with injected defects: 3% missing
  revenue, 2% duplicate rows, 1% ×10 revenue outliers, and 76 rows of negative revenue.
- **H2:** 12 sealed test `normal` events with 2 probe questions each, 24 questions in total. Arms B and
  R ran 3 times per question (72 runs each); arm A ran twice (48 runs). Each event's cause text and
  corpus stayed sealed locally until the reveal.

**Metrics**

- **`factual_accuracy`:** whether the numbers in the report match ground truth (1% tolerance). Scored by
  code, with no model call.
- **`event_attribution_accuracy`:** whether the report names the event's true cause, labelled
  `correct` / `wrong` / `none`; only `correct` scores 1. The judge is deepseek-v4-flash-0731 (via
  dmxapi), 3 samples per report. Before the freeze, the judge met the pre-registered agreement bar
  against blind human labels.
- **Refusals:** a run ending in `safe_refusal` scores 0 and is not voided. A run is void only if its row
  says so, its report falls back to the rule-based template, or it ran longer than 900 s.

**Pre-registration and reveal**

1. The pre-registration was frozen on 2026-11-11. After the freeze, MAEDA commits may touch only `docs/`.
2. The reveal followed the six steps of §10.15 item 9 in order: build the reveal retrieval store, export
   the H2 test questions, start the MCP services, run H1/H3, run H2, run the judge.
3. The user authorized each paid step separately, with a budget. All six steps ran on 2026-11-11 and 2026-11-12.
4. All analysis used the frozen rules and the project's existing functions. New analysis scripts were
   kept outside this repository to respect the freeze.

## H1 and H3: cleaning helps, with no interaction with retrieval

Cleaning raises numeric accuracy from 0.067 to 0.256, so H1 holds. Turning retrieval on as well does not
change the effect of cleaning, so H3 was not detected.

| Arm | Mean `factual_accuracy` |
|---|---|
| C (cleaning only) | **0.256** |
| CR (both on) | **0.247** |
| A (auto; cleaning never triggered) | 0.069 |
| R (retrieval only) | 0.068 |
| B (both off) | 0.067 |

*119 questions × 2 runs per arm, refusals scored 0.*

**H1 (C − B):** +0.1887, 98.3% CI [+0.1139, +0.2723], n = 119. The interval excludes 0 and the point
estimate is above 0.028, so by §7.6 cleaning is "worth it". Per question: 31 improved, 87 unchanged,
1 worse.

**H3 ((CR − R) − (C − B)):** −0.0092, 98.3% CI [−0.0472, +0.0279]; the interval includes 0. Per question:
10 larger, 12 smaller, 97 unchanged.

Descriptive checks (not part of the formal test) all point the same way as H1:

- On the 116 `cleaning_material` questions (where cleaning actually changes the ground truth):
  C − B = +0.194.
- With refusals excluded instead of scored 0: C − B = +0.222.
- Exploratory differences: R − B +0.0004, CR − B +0.180, A − B +0.001. Retrieval does not help numeric
  questions, as expected.

The CIs use the project's existing `paired_bootstrap_ci` with 100,000 resamples and seed 0 (the
function's default). Each question's 2 runs are averaged before arms are compared.

## H2: retrieval lets reports name the cause

With retrieval on, event-level attribution accuracy rises by 0.653 on average. The exact sign-flip test
gives p = 0.002, so H2 holds.

| Event | R | B |
|---|---|---|
| EVT-V2-008 | 1.00 | 0 |
| EVT-V2-009 | 1.00 | 0 |
| EVT-V2-012 | 1.00 | 0 |
| EVT-V2-015 | 1.00 | 0 |
| EVT-V2-018 | 1.00 | 0 |
| EVT-V2-003 | 0.83 | 0 |
| EVT-V2-006 | 0.83 | 0 |
| EVT-V2-017 | 0.50 | 0 |
| EVT-V2-005 | 0.33 | 0 |
| EVT-V2-014 | 0.33 | 0 |
| EVT-V2-002 | 0.00 | 0 |
| EVT-V2-011 | 0.00 | 0 |

*Per-event `event_attribution_accuracy`, refusals scored 0 (primary analysis).*

- **Primary analysis** (refusals scored 0): of 12 events, 10 are positive, 2 are zero, and none are
  negative. p = 2/2^10 = 0.00195, below α 0.0167 and above the 0.028 floor. Descriptive 98.3% bootstrap
  interval: [+0.375, +0.903].
- **Sensitivity analysis** (refusals scored from their report text): +0.681, p = 0.00195. This agrees
  with the primary analysis, so the result is not an artifact of the guardrail.
- **With retrieval off, no report got the cause right.** Of arm B's 72 reports, 63 described the anomaly
  without offering a cause (`none`) and 9 gave a wrong cause (`wrong`). Without external material, the
  system has no way to know the event's cause.
- Arm R labels: `correct` 49, `wrong` 14, `none` 9. The two events where R scored 0 are EVT-V2-002 and
  EVT-V2-011; their cause has not been investigated separately.

| Arm | Refusal rate | Correct and not refused |
|---|---|---|
| B | 26.4% (19/72) | 0/72 |
| R | 8.3% (6/72) | 47/72 |
| A | 22.9% (11/48) | 23/48 |

The refusal rates and arm A figures are descriptive. Refusals drop markedly with retrieval on, possibly
because the guardrail judges claims "unsupported" less often once material is available; this has not
been tested. `distractor` and `unexplained` events are not in the test pool, so their wrong-attribution
rates cannot be reported.

## Experiment 2: routing in auto mode

Auto mode never triggered cleaning, exactly as predicted before the reveal. For retrieval, every run that
should have retrieved did, and all 6 unnecessary retrievals came from intent-classification errors. This
section is descriptive and uses none of the confirmatory budget. Arm A ran 286 times: 238 on the H1
holdout and 48 on the H2 probes.

| Half | Label (should fire) | TP | FP | FN | TN | Precision | Recall |
|---|---|---|---|---|---|---|---|
| Cleaning (H1 only) | `cleaning_material` | 0 | 0 | 232 | 6 | undefined | 0 |
| Retrieval (H1 + H2) | `corpus_backed` | 48 | 6 | 0 | 232 | 0.889 | 1.0 |

**Cleaning half.** Every run profiles the data first, then the `has_critical_issues` gate decides that
no cleaning is needed. Pre-registration §7.3.1 derived this in advance: the missing ratio is at most 0.03
(threshold 0.10), the duplicate ratio is about 0.0196 (threshold 0.05), and negative values and outliers
are not trigger dimensions, so the gate cannot be true on this data. The result, recall 0 and precision
undefined, matches the prediction. This is a gap in what the gate covers, not a tuning error. Changing
the gate is a contract change and is left for future work.

**Retrieval half.** Auto mode retrieves only when the parsed `query_type` is `diagnostic` or
`exploratory`. All 48 H2 runs retrieved. Of the 238 H1 runs, 6 retrieved unnecessarily; all were
questions authored as `comparative` and parsed as `diagnostic` (H1T-015, H1T-088 and H1T-101, twice
each). Under the §7.3.2 decomposition, all 6 are **intent-classification errors**: the router responded
correctly to the label it was given. The other 232 runs parsed the same type as authored and correctly
skipped retrieval.

The retrieval figures in this table mostly reflect how the sample was composed, not how good the routing
rule is (see Limitations and deviations).

## Run quality and cost

All 1,382 report generations completed, with 0 voided and 0 citing foreign documents. Total paid cost was
about $42, within the $55 the user approved.

| Step | What | Runs | Void | Cost |
|---|---|---|---|---|
| 1 | Build reveal retrieval store (49 documents, 59 chunks) | n/a | n/a | about $0.0003 |
| 2–3 | Export the 24 H2 questions, start MCP services | n/a | n/a | $0 |
| 4 | H1 / H3 and arm A (H1 holdout) | 1,190 | 0 | $37.28 |
| 5 | H2 and arm A (probes) | 192 | 0 | $4.66 |
| 6 | Attribution judge | 192 scored | 0 invalid | ¥0.44 (about $0.06) |

- Step 4 refusals: B 18, C 26, R 35, CR 31, A 29 (238 runs per arm). Refusals score 0 by rule and are
  included in the H1/H3 results.
- During step 4 the judge provider's balance briefly ran out (UTC 06:28–06:29); the user topped it up at
  once. Six runs had `answer_relevance` and `groundedness` recorded as errors, and a few were scored from
  1 of 3 samples. Both are descriptive metrics; the primary metric `factual_accuracy` is scored by code
  and was unaffected.
- The step 1 store-building script crashed while printing statistics after the build finished (a
  `rep.status_counts()` type error). rag-framework was frozen, so it was not fixed. The store was checked
  directly instead: 28 public and 21 sealed documents, 59 FAISS entries, all 12 sealed events present,
  content hash `efbcd523…ca`.
- Costs not counted separately: cleaning triggered by arm A (in practice it never triggered, so close to
  0), and small embedding and rerank costs on the rag side.
- Longest single run: 85 s in step 5, far below the 900 s void line.

## Limitations and deviations

None of the deviations changes the three confirmatory conclusions. All are recorded in the integration log.

**Deviations from, or gaps in, the pre-registration**

- **Bootstrap seed:** the pre-registration does not state a seed. The default of `paired_bootstrap_ci`
  (0) was used. H1's lower bound of +0.114 is far from 0, so a different seed would not change the verdict.
- **No dedicated field for the parsed `query_type`:** §7.3.2 says every row records it; in fact none do.
  For H1 rows it was recovered from the `intent_accuracy` explanation text (all 238 rows recovered). H2
  rows carry no such text, so H2 has no decomposition.
- **`corpus_backed` label for H2 probes:** the pre-registration does not list it per question. Every
  probe is linked to a test event with a corpus, so this report treats it as true.
- **rag API key in step 3:** the rag server was first started with MAEDA's OpenAI key, whereas the σ runs
  used rag's own key. This was corrected and the server restarted before step 4, so the reveal runs match σ.

**Limits of the sample and measurement**

- **Experiment 2 has no discordant cases:** §7.3.4 planned for about a quarter of questions to have a
  query type that disagrees with whether a corpus exists. In the actual test sets, H1 is all
  `descriptive`/`comparative` with no corpus, and H2 is all `diagnostic` with a corpus, so there are no
  off-diagonal cells at the authoring level. The retrieval half's precision and recall are therefore
  largely a product of sample structure.
- **Synthetic data, one dataset:** all conclusions apply to this demo sales table with injected defects
  and this event script. They do not carry over directly to real business data.
- **Low absolute level in H1:** the cleaner deduplicates and imputes missing revenue but does not repair
  ×10 outliers. Also, §7.4's defect list does not name duplicate rows, so `cleaning_material` does not
  account for deduplication.
- **H2 uses judge labels only:** the reveal reports were not human-labelled. The pre-registration does not
  require it, and the judge met the human-agreement bar before the freeze.
- **H2's arm B is all zero:** the result means "without external material the system cannot name the
  cause". The size of the effect depends on the cause existing only in the corpus.

**Records and reproducibility**

- The reveal analysis and operations scripts live in the integration control directory, outside the
  three repositories. That directory is not under version control.
- Reveal outputs sit in gitignored directories of each repository; there is only one local copy.

## Interpretation and next steps

Each tool helps on the metric it is responsible for, and they do not interfere with each other. The
biggest weakness is that auto mode cannot make use of cleaning.

- **Cleaning handles numbers, retrieval handles causes.** Cleaning improves only `factual_accuracy`,
  where retrieval has no effect; retrieval acts only on attribution. H3 detected no interaction, so
  turning both on does not cancel either out.
- **Auto mode's cleaning gate does nothing on this data.** Forcing cleaning on gains +0.189, but auto
  mode never gets it, because the gate's three dimensions do not cover negative values or low-rate
  missing data.
- **The retrieval routing itself showed no problem.** The visible errors came from intent parsing
  misreading `comparative` as `diagnostic`.

**Suggested follow-up work** (all belongs to a new round; none of it can revise these results)

1. As §7.3.1 states, treat extending the cleaning gate (a lower missing threshold, a dimension that
   covers negative values) as a contract change, through a new design round.
2. Make the cleaner handle ×10-style outliers; they are the main reason H1's absolute level is low.
3. Add the parsed `query_type` field to run records, delivering the record §7.3.2 promised.
4. In the next routing experiment, actually include discordant cases as §7.3.4 intended.
5. Put the integration control directory (the shared log and reveal scripts) under version control or back it up.

## Appendix: files

Every number can be recomputed offline from the files below, with no further paid calls. Paths are
relative to each repository; data files are gitignored and kept locally.

| Content | Location | Identifier |
|---|---|---|
| Pre-registration (frozen) | `docs/phase4_preregistration.md` §10.15 | MAEDA `0a9edbc` |
| Reveal retrieval store | rag-framework `data/tb5_full.db` | content hash `efbcd523…ca` |
| H2 test question export | `data/h2_reveal/h2_test_attribution_cases.json` | sha256 `37b548b2…4d14` |
| H1 / H3 runs | `data/reveal_h1/{br,c,cr,a}/runs.jsonl` | 1,190 rows |
| H1 eligibility fields | `tests/eval/h1_holdout_eligibility.json` | sha256 `6246cdff…` |
| H2 reports | `data/h2_reveal/reports_reveal.jsonl` | sha256 `8f49ddc3…c2` |
| H2 judge labels | `data/h2_reveal/judge_reveal.jsonl` | sha256 `9e61a293…7029` |
| H2 summary | `data/h2_reveal/h2_summary_reveal.json` | output of frozen `scripts/h2_summary.py` |

Repository commits the reveal ran on (§10.15 item 7): MAEDA `368436f` (code) with `0a9edbc` (docs),
rag-framework `fe908d4`, agentic-data-cleaner-v2 `c3639dc`.
