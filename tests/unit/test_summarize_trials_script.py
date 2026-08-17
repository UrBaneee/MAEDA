"""
Smoke test for scripts/summarize_trials.py — the CLI wrapper around
src/eval/trials.py::summarize_report(). Exercises the actual multi-trial
report shape scripts/run_eval.py --trials N writes (not a hand-shaped
stand-in), end to end: read report -> compute -> print -> write summary
JSON. No graph/LLM calls.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def _write_multi_trial_report(path: Path) -> None:
    def _case(cid: str, score: float, label: str):
        return {
            "test_case_id": cid,
            "eval_result": {
                "run_id": cid, "query": "q", "test_case_id": cid,
                "cleaning_applied_level": "full", "cleaning_stop_reason": "passed",
                "scores": [{"metric": "answer_relevance", "score": score, "label": label,
                            "reasoning": "", "valid": True}],
                "aggregate_score": score, "timestamp": 0.0,
            },
            "meta": {"run_id": f"{cid}-real-run-id"},
        }

    report = {
        "timestamp": 0.0, "n_cases": 1, "split": "dev",
        "trials": 2, "concurrency": 1,
        "per_trial": [
            {"trial_index": 0, "n_cases": 1, "overall_aggregate": 0.9, "cases": [_case("A", 0.9, "pass")]},
            {"trial_index": 1, "n_cases": 1, "overall_aggregate": 0.2, "cases": [_case("A", 0.2, "fail")]},
        ],
    }
    path.write_text(json.dumps(report))


def test_summarize_trials_end_to_end(tmp_path, capsys):
    import summarize_trials

    report_path = tmp_path / "eval_trials2_123.json"
    _write_multi_trial_report(report_path)

    argv = ["summarize_trials.py", str(report_path)]
    old_argv = sys.argv
    sys.argv = argv
    try:
        summarize_trials.main()
    finally:
        sys.argv = old_argv

    captured = capsys.readouterr()
    assert "D0 trial summary" in captured.out
    assert "answer_relevance" in captured.out

    out_path = tmp_path / "eval_trials2_123_trial_summary.json"
    assert out_path.exists()
    summary = json.loads(out_path.read_text())
    assert summary["n_cases"] == 1
    assert summary["per_case"]["A"]["binary"]["answer_relevance"]["c"] == 1
    assert summary["per_case"]["A"]["binary"]["answer_relevance"]["n_scored"] == 2


def test_summarize_trials_rejects_single_trial_report(tmp_path):
    import summarize_trials

    single_trial_report = {
        "timestamp": 0.0, "n_cases": 1, "overall_aggregate": 0.9,
        "split": "dev", "cases": [],
    }
    report_path = tmp_path / "eval_single.json"
    report_path.write_text(json.dumps(single_trial_report))

    try:
        summarize_trials._load_per_trial(report_path)
        assert False, "expected ValueError for a single-trial report"
    except ValueError as exc:
        assert "per_trial" in str(exc)
