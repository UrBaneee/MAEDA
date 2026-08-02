"""
Noise-floor utilities (eval v2, Step 1 — see docs/eval_v2_plan.md).

Two things live here:
  1. `paired_bootstrap_ci` — significance test for "did this change actually
     move the score, or is it noise" comparisons on the same golden cases
     before/after a change. Paired because the two runs share the same 20
     (or however many) cases; a paired test has much more power here than
     treating the two runs as independent samples.
  2. `summarize` — plain descriptive stats (mean/std/min/max) for a set of
     repeated measurements of the same quantity, used by
     scripts/measure_noise.py to characterize judge noise and full-pipeline
     noise separately.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats as _scipy_stats


@dataclass
class BootstrapResult:
    mean_delta: float
    ci_low: float
    ci_high: float
    significant: bool  # True iff the CI excludes 0
    n: int


def paired_bootstrap_ci(
    deltas: list[float],
    confidence: float = 0.95,
    n_resamples: int = 10000,
    seed: int = 0,
) -> BootstrapResult:
    """
    deltas: per-case (after - before) score differences on the *same* cases.
    Returns the bootstrap CI on the mean delta. If the CI excludes 0, the
    change is significant at this confidence level; if it straddles 0, it
    is not distinguishable from noise at this sample size.
    """
    arr = np.asarray(deltas, dtype=float)
    if len(arr) < 2:
        raise ValueError("paired_bootstrap_ci needs at least 2 paired deltas")

    rng = np.random.default_rng(seed)
    res = _scipy_stats.bootstrap(
        (arr,), np.mean, confidence_level=confidence,
        n_resamples=n_resamples, method="percentile", random_state=rng,
    )
    lo, hi = float(res.confidence_interval.low), float(res.confidence_interval.high)
    return BootstrapResult(
        mean_delta=float(arr.mean()), ci_low=lo, ci_high=hi,
        significant=not (lo <= 0.0 <= hi), n=len(arr),
    )


@dataclass
class NoiseSummary:
    label: str
    n: int
    mean: float
    std: float
    min: float
    max: float
    two_sigma: float  # 2 * std — the "don't trust a change smaller than this" threshold

    def to_dict(self) -> dict:
        return {
            "label": self.label, "n": self.n, "mean": round(self.mean, 4),
            "std": round(self.std, 4), "min": round(self.min, 4),
            "max": round(self.max, 4), "two_sigma": round(self.two_sigma, 4),
        }


def summarize(label: str, values: list[float]) -> Optional[NoiseSummary]:
    """None if fewer than 2 values — std is undefined for n<2, and a
    single-sample noise estimate would be misleading to report at all."""
    if len(values) < 2:
        return None
    std = statistics.stdev(values)
    return NoiseSummary(
        label=label, n=len(values), mean=statistics.mean(values),
        std=std, min=min(values), max=max(values), two_sigma=2 * std,
    )
