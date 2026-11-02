#!/usr/bin/env python3
"""
Offline σ_hetero decomposition between two arm reports (附录 EJ.3/EJ.4(4)).
Reads saved reports from scripts/run_eval.py --trials N or scripts/measure_noise.py
full. Makes no LLM call.

    python scripts/decompose_variance.py arm_a.json arm_b.json --metric factual_accuracy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.noise import decompose_paired_variance, required_n, sigma_separates_n
from src.eval.trials import not_applicable_reason


def trial_rows(report: dict) -> list[dict]:
    if "trials_raw" in report:
        return [row for trial in report["trials_raw"] for row in trial.values()]
    if "per_trial" in report:
        return [row for trial in report["per_trial"] for row in trial["cases"]]
    return list(report["cases"])


def case_scores(report: dict, metric: str) -> dict[str, list[float]]:
    """Excluded trials and invalid metric entries contribute nothing."""
    scores: dict[str, list[float]] = {}
    for row in trial_rows(report):
        result = row["eval_result"]
        if not_applicable_reason(result):
            continue
        for s in result["scores"]:
            if s["metric"] == metric and s.get("valid", True):
                scores.setdefault(result["test_case_id"], []).append(s["score"])
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm_a", type=Path)
    parser.add_argument("arm_b", type=Path)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--k", type=int, default=5, help="planned trials per case at the reveal")
    args = parser.parse_args()

    a = case_scores(json.loads(args.arm_a.read_text()), args.metric)
    b = case_scores(json.loads(args.arm_b.read_text()), args.metric)
    d = decompose_paired_variance(a, b)
    print(json.dumps({
        "metric": args.metric, "n_cases": d.n_cases, "n_dropped": d.n_dropped,
        "mean_delta": round(d.mean_delta, 4), "var_delta": round(d.var_delta, 6),
        "repeat_var": round(d.repeat_var, 6), "sigma_hetero": round(d.sigma_hetero, 4),
        "sigma_hetero_ci": [round(x, 4) for x in d.sigma_hetero_ci],
        "n_required_at_k": required_n(d.var_at(args.k)),
        "ej4_4_valid": sigma_separates_n(d, args.k),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
