#!/usr/bin/env python3
"""
Run the event-attribution judge (预注册 §7.2.1, R5.1) over saved H2 reports.
PAID: settings.eval_judge_samples judge calls per report, on the eval provider.

Reads only the report text plus the event's anomaly and documented cause, as
score_event_attribution does; the arm and refusal status never reach the judge.
Refused runs are judged too: the sensitivity analysis needs their text labels
(附录 FM). Each verdict is appended to the JSONL as soon as it exists and a rerun
skips reports already judged. Stops before a report when the running cost would
pass the budget.

The output is not shown to the human labeller before pass 2 ends: per-report
verdicts would leak into the retest (R5.1 procedure).

    python scripts/judge_h2_reports.py --set formal --budget-usd 1
    python scripts/judge_h2_reports.py --set formal --dir data/h2_pilot_v2 --budget-usd 0.2

--dir only chooses where reports_<set>.jsonl is read and judge_<set>.jsonl is
written (附录 HJ); the judge prompt and scoring are unchanged.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import settings  # noqa: E402
from src.eval.metrics import (  # noqa: E402
    _build_eval_llm,
    _judge_usage,
    _judge_usage_summary,
    score_event_attribution,
)

CASES = Path("tests/eval/h2_dev_attribution_cases.json")
DEFAULT_DIR = Path("data/h2_attribution")
CNY_PER_USD = 7.1  # budget check only; the CNY figure is what is recorded


class _UsageRecorder:
    """Pass-through to the judge model that keeps each call's token usage."""

    def __init__(self, llm):
        self._llm = llm
        self.calls: list[dict] = []

    async def ainvoke(self, messages, **kwargs):
        response = await self._llm.ainvoke(messages, **kwargs)
        self.calls.append({"usage": _judge_usage(response)})
        return response


async def main(args) -> int:
    cases_path = Path(args.cases) if args.cases else CASES
    rows = json.loads(cases_path.read_text())
    from src.eval.reveal_guard import require_reveal_authorization
    require_reveal_authorization(rows)  # split=test cases need the reveal flag (附录 HQ.2)
    cases = {c["id"]: c for c in rows}
    out_dir = Path(args.dir)
    reports = [json.loads(line) for line in (out_dir / f"reports_{args.set}.jsonl").open()]
    out_path = out_dir / f"judge_{args.set}.jsonl"
    done = {}
    if out_path.exists():
        done = {r["job_id"]: r for r in map(json.loads, out_path.open())}
    spent_cny = sum(r["usage"].get("cost_cny", 0) for r in done.values())
    todo = [r for r in reports if r["job_id"] not in done]
    print(f"{len(done)} judged, {len(todo)} to judge, model {settings.resolved_eval_provider}/"
          f"{settings.resolved_eval_model}, samples {settings.eval_judge_samples}")
    per_report = None
    for rep in todo:
        if per_report is not None and (spent_cny + per_report) / CNY_PER_USD > args.budget_usd:
            print(f"STOP: next report would pass the budget ({spent_cny:.4f} CNY spent)")
            return 3
        case = cases[rep["case_id"]]
        recorder = _UsageRecorder(_build_eval_llm(metric="event_attribution_accuracy"))
        score = await score_event_attribution(
            rep["report"], case["event_anomaly"], case["event_cause"], llm=recorder)
        usage = _judge_usage_summary(recorder.calls, "event_attribution_accuracy")
        row = {"job_id": rep["job_id"], "label": score.raw_value, "valid": score.valid,
               "reasoning": score.reasoning, "usage": usage}
        with out_path.open("a") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        spent_cny += usage.get("cost_cny", 0)
        per_report = usage.get("cost_cny", 0) or per_report
        print(f"{rep['job_id']}: {row['label']} valid={row['valid']} "
              f"calls={usage['calls']} total {spent_cny:.4f} CNY")
        if not score.valid:
            print(f"STOP: judge failed ({score.reasoning})")
            return 4
    print(f"done, {spent_cny:.4f} CNY (~${spent_cny / CNY_PER_USD:.4f})")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--set", choices=["dry", "formal"], required=True)
    parser.add_argument("--budget-usd", type=float, required=True)
    parser.add_argument("--cases", default=None,
                        help="cases JSON (default: the dev attribution cases); the reveal export for the H2 reveal")
    parser.add_argument("--dir", default=str(DEFAULT_DIR),
                        help="directory holding reports_<set>.jsonl; judge_<set>.jsonl is written there")
    sys.exit(asyncio.run(main(parser.parse_args())))
