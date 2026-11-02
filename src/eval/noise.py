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
    n_resamples: int = 100000,
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


@dataclass
class VarianceDecomposition:
    """附录 CA.1 / EJ.3: Var(d_i) = σ²_hetero + (σ²_a + σ²_b) / k. Trials shrink
    only the second term, which is why the two must be estimated apart."""
    n_cases: int
    n_dropped: int
    mean_delta: float
    var_delta: float
    repeat_var: float        # mean over cases of s²_a + s²_b, per single trial
    mean_k: float
    sigma_hetero: float
    sigma_hetero_ci: tuple[float, float]

    def var_at(self, k: int, sigma_hetero: Optional[float] = None) -> float:
        hetero = self.sigma_hetero if sigma_hetero is None else sigma_hetero
        return hetero ** 2 + self.repeat_var / k


def _sigma_hetero(deltas: np.ndarray, measurement: np.ndarray) -> np.ndarray:
    """Vectorised over the last axis. Floored at zero: sampling error can push
    the raw estimate negative, and a variance cannot be."""
    return np.sqrt(np.maximum(deltas.var(axis=-1, ddof=1) - measurement.mean(axis=-1), 0.0))


def decompose_paired_variance(
    arm_a: dict[str, list[float]],
    arm_b: dict[str, list[float]],
    confidence: float = 0.95,
    n_resamples: int = 10000,
    seed: int = 0,
) -> VarianceDecomposition:
    """Per-case repeated scores for two arms, keyed by case id. A case needs at
    least two repeats on both sides, otherwise it says nothing about
    measurement noise. The interval resamples cases, the independent unit."""
    usable = sorted(c for c in set(arm_a) & set(arm_b)
                    if len(arm_a[c]) >= 2 and len(arm_b[c]) >= 2)
    if len(usable) < 3:
        raise ValueError(f"need at least 3 usable cases, got {len(usable)}")
    n_dropped = len(set(arm_a) | set(arm_b)) - len(usable)
    a = [np.asarray(arm_a[c], dtype=float) for c in usable]
    b = [np.asarray(arm_b[c], dtype=float) for c in usable]
    deltas = np.array([x.mean() - y.mean() for x, y in zip(a, b)])
    measurement = np.array([x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y) for x, y in zip(a, b)])
    repeat_var = float(np.mean([x.var(ddof=1) + y.var(ddof=1) for x, y in zip(a, b)]))

    idx = np.random.default_rng(seed).integers(0, len(usable), size=(n_resamples, len(usable)))
    boot = _sigma_hetero(deltas[idx], measurement[idx])
    tail = (1 - confidence) / 2
    low, high = np.quantile(boot, [tail, 1 - tail])
    return VarianceDecomposition(
        n_cases=len(usable), n_dropped=n_dropped, mean_delta=float(deltas.mean()),
        var_delta=float(deltas.var(ddof=1)), repeat_var=repeat_var,
        mean_k=float(np.mean([len(x) + len(y) for x, y in zip(a, b)]) / 2),
        sigma_hetero=float(_sigma_hetero(deltas, measurement)),
        sigma_hetero_ci=(float(low), float(high)),
    )


def required_n(var_delta: float, mde: float = 0.028, multiplier: float = 2.241,
               margin: float = 1.2) -> int:
    """Cases needed for `mde` at the 97.5% multiplier (§7.6), with FE.8's ×1.2."""
    return int(np.ceil(margin * multiplier ** 2 * var_delta / mde ** 2))


def sigma_separates_n(decomposition: VarianceDecomposition, k: int = 5,
                      candidates: tuple[int, int] = (30, 60)) -> bool:
    """附录 EJ.4(4): the σ run is invalid when its interval is so wide that both
    candidate sizes fit inside the N it implies."""
    low, high = decomposition.sigma_hetero_ci
    n_low = required_n(decomposition.var_at(k, low))
    n_high = required_n(decomposition.var_at(k, high))
    return not (n_low <= candidates[0] and candidates[1] <= n_high)


@dataclass
class SignFlipResult:
    n: int
    mean_delta: float
    p_value: float
    exact: bool


def event_sign_flip_test(deltas: list[float], n_random: int = 100000, seed: int = 0) -> SignFlipResult:
    """预注册 §7.2.1: two-sided sign-flip test on per-event R−B means. Exact
    up to 16 events; beyond that a random sample of flips, counting the observed
    assignment so p can never be zero. Returned p is compared with §7.6's α."""
    arr = np.asarray(deltas, dtype=float)
    n = arr.size
    if n == 0:
        raise ValueError("no events")
    observed = abs(arr.sum())
    exact = n <= 16
    if exact:
        signs = 1 - 2 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
        extreme = int((np.abs(signs @ arr) >= observed - 1e-12).sum())
        p_value = extreme / len(signs)
    else:
        rng = np.random.default_rng(seed)
        extreme = 0
        for start in range(0, n_random, 10000):
            signs = rng.choice([-1.0, 1.0], size=(min(10000, n_random - start), n))
            extreme += int((np.abs(signs @ arr) >= observed - 1e-12).sum())
        p_value = (extreme + 1) / (n_random + 1)
    return SignFlipResult(n=n, mean_delta=float(arr.mean()), p_value=float(p_value), exact=exact)


def average_by_event(case_scores: dict[str, float], case_events: dict[str, str]) -> dict[str, float]:
    """Several cases on one event are one observation (附录 FI)."""
    grouped: dict[str, list[float]] = {}
    for case, score in case_scores.items():
        grouped.setdefault(case_events[case], []).append(score)
    return {event: float(np.mean(scores)) for event, scores in grouped.items()}
