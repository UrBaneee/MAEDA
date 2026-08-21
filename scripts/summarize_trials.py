"""
D0 (阶段 3 / 附录 AW block 3) — CLI wrapper around src/eval/trials.py.

Reads a multi-trial report JSON produced by
`scripts/run_eval.py --trials N` (N > 1; see that script's report shape)
and prints per-case pass@k/pass^k (binary metrics) and cross-trial
variance (continuous metrics), plus a suite-level rollup. Writes the same
data as JSON alongside the printed summary.

Usage:
    poetry run python scripts/run_eval.py --trials 8 --split dev
    poetry run python scripts/summarize_trials.py logs/eval_runs/eval_trials8_....json
    poetry run python scripts/summarize_trials.py <report.json> --k 1 --k 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.trials import summarize_report  # noqa: E402


def _load_per_trial(report_path: Path) -> list[list[dict]]:
    report = json.loads(report_path.read_text())
    if "per_trial" not in report:
        raise ValueError(
            f"{report_path} does not look like a multi-trial report (no 'per_trial' key) -- "
            "run scripts/run_eval.py with --trials > 1 to produce one. A single-trial report "
            "(the default, --trials 1) has nothing to compute pass@k across."
        )
    return [t["cases"] for t in report["per_trial"]]


def _print_summary(summary: dict) -> None:
    print("\n" + "=" * 92)
    print(f"D0 trial summary — {summary['n_cases']} case(s), k={summary['k_values']}")
    print("=" * 92)

    for cid, case in sorted(summary["per_case"].items()):
        # 附录 CK.3: there are two exclusion conditions now
        # (blocked_needs_review and a force_on arm whose RAG retrieval
        # degraded), so print the actual reasons rather than the one
        # hard-coded label this line used to assume.
        reasons = case.get("not_applicable_reasons") or {}
        excluded_note = (
            f" ({case['n_not_applicable']} excluded: "
            + "; ".join(f"{r} x{n}" for r, n in sorted(reasons.items()))
            + ")"
            if case["n_not_applicable"] else ""
        )
        print(f"\n[{cid}] n_trials={case['n_trials']} n_applicable={case['n_applicable']}{excluded_note}")
        for metric, b in sorted(case["binary"].items()):
            pk = ", ".join(f"@{k}={v:.2f}" if v is not None else f"@{k}=n/a"
                            for k, v in b["pass_at_k"].items())
            phk = ", ".join(f"^{k}={v:.2f}" if v is not None else f"^{k}=n/a"
                             for k, v in b["pass_hat_k"].items())
            print(f"    {metric:22s} n={b['n_scored']:3d} c={b['c']:3d}  pass{pk}  pass{phk}")
        for metric, c in sorted(case["continuous"].items()):
            s = c["summary"]
            if s is None:
                print(f"    {metric:22s} (fewer than 2 applicable trials -- no variance to report)")
            else:
                print(f"    {metric:22s} mean={s['mean']:.3f} std={s['std']:.3f} 2σ={s['two_sigma']:.3f}")

    print("\n" + "-" * 92)
    print("Suite-level rollup (mean across cases where defined):")
    for metric, per_k in sorted(summary["suite_binary"].items()):
        for k, v in per_k.items():
            if v["mean_pass_at_k"] is None:
                continue
            print(
                f"  {metric:22s} k={k}  mean_pass@k={v['mean_pass_at_k']:.3f}  "
                f"mean_pass^k={v['mean_pass_hat_k']:.3f}  "
                f"({v['n_cases_defined']}/{v['n_cases_total']} cases)"
            )
    for metric, v in sorted(summary["suite_continuous"].items()):
        if v["mean_of_case_means"] is None:
            continue
        print(
            f"  {metric:22s} mean_of_case_means={v['mean_of_case_means']:.3f}  "
            f"({v['n_cases_with_variance_data']}/{v['n_cases_total']} cases)"
        )
    print("=" * 92)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report", type=str, help="Path to a multi-trial report JSON (--trials > 1)")
    parser.add_argument("--k", type=int, action="append",
                        help="k value(s) for pass@k/pass^k (repeatable). Default: [1].")
    parser.add_argument("--out", type=str, default=None,
                        help="Write the JSON summary here (default: alongside the input report)")
    args = parser.parse_args()

    report_path = Path(args.report)
    per_trial = _load_per_trial(report_path)
    summary = summarize_report(per_trial, k_values=args.k)

    _print_summary(summary)

    out_path = Path(args.out) if args.out else report_path.with_name(report_path.stem + "_trial_summary.json")
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()
