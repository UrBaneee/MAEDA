"""
Judge-vs-human agreement statistics — eval v2 Step 3 (docs/eval_v2_plan.md).

The judge's own scores are continuous (groundedness, since Step 2c) or a
coarse 3-point scale (answer_relevance, unchanged) in [0, 1]. Quadratic
Weighted Kappa is defined over discrete ordered categories, so both human
and judge scores are discretized into 5 bands (matching the plan's "scores
are ordered 5-point categories" framing) before computing it. Spearman and
MAE are computed on the raw continuous scores directly -- no binning
needed, and binning would throw away real information for those two.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats as _scipy_stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix

# 5 ordered bands: 0=fail .. 4=excellent. Matches the 5-anchor rubric
# recommended during the initial audit (3 anchors left too much mass
# stuck in one middle bucket to discriminate anything).
_N_BANDS = 5


def discretize(score: float, n_bands: int = _N_BANDS) -> int:
    """Map a continuous [0, 1] score to an ordered integer band
    0..n_bands-1. score=1.0 lands in the top band, not spilling into a
    phantom n_bands-th one."""
    score = min(max(score, 0.0), 1.0)
    band = int(score * n_bands)
    return min(band, n_bands - 1)


@dataclass
class AgreementReport:
    metric: str
    n: int
    qwk: float
    spearman: float
    spearman_p: float
    mae: float
    exact_match_rate: float  # within-band exact match, i.e. same discretized band
    confusion: list[list[int]]  # rows=human band, cols=judge band
    n_bands: int

    def to_dict(self) -> dict:
        return {
            "metric": self.metric, "n": self.n, "qwk": round(self.qwk, 4),
            "spearman": round(self.spearman, 4), "spearman_p": round(self.spearman_p, 4),
            "mae": round(self.mae, 4), "exact_match_rate": round(self.exact_match_rate, 4),
            "confusion": self.confusion, "n_bands": self.n_bands,
        }

    def summary_line(self) -> str:
        verdict = "good" if self.qwk >= 0.8 else "acceptable" if self.qwk >= 0.6 else "weak"
        return (f"{self.metric}: n={self.n}  QWK={self.qwk:.3f} ({verdict})  "
                f"Spearman={self.spearman:.3f} (p={self.spearman_p:.3f})  "
                f"MAE={self.mae:.3f}  exact-band-match={self.exact_match_rate:.0%}")


def compute_agreement(
    human_scores: list[float],
    judge_scores: list[float],
    metric: str,
    n_bands: int = _N_BANDS,
) -> AgreementReport:
    """human_scores and judge_scores must be paired (same sample, same
    order) -- e.g. human_scores[i] and judge_scores[i] both score sample i.
    """
    if len(human_scores) != len(judge_scores):
        raise ValueError(f"paired lists must be the same length: {len(human_scores)} vs {len(judge_scores)}")
    if len(human_scores) < 2:
        raise ValueError("need at least 2 paired samples to compute agreement")

    human_bands = [discretize(s, n_bands) for s in human_scores]
    judge_bands = [discretize(s, n_bands) for s in judge_scores]

    qwk = cohen_kappa_score(human_bands, judge_bands, weights="quadratic")
    spearman_r, spearman_p = _scipy_stats.spearmanr(human_scores, judge_scores)
    mae = statistics.mean(abs(h - j) for h, j in zip(human_scores, judge_scores))
    exact = sum(1 for hb, jb in zip(human_bands, judge_bands) if hb == jb) / len(human_bands)
    cm = confusion_matrix(human_bands, judge_bands, labels=list(range(n_bands)))

    return AgreementReport(
        metric=metric, n=len(human_scores), qwk=float(qwk),
        spearman=float(spearman_r), spearman_p=float(spearman_p),
        mae=mae, exact_match_rate=exact, confusion=cm.tolist(), n_bands=n_bands,
    )


def print_confusion_matrix(report: AgreementReport) -> None:
    band_labels = [f"b{i}" for i in range(report.n_bands)]
    header = "human\\judge".ljust(12) + "".join(f"{b:>6s}" for b in band_labels)
    print(header)
    for i, row in enumerate(report.confusion):
        print(f"{band_labels[i]:<12s}" + "".join(f"{v:>6d}" for v in row))
