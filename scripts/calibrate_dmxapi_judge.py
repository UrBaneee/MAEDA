#!/usr/bin/env python3
"""Run MAEDA's real judge prompts against three hand-written dev probes."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from langchain_openai import ChatOpenAI

from src.eval.metrics import score_relevance_and_groundedness


MODEL = "deepseek-v4-flash-0731"
ALLOW_ENV = "MAEDA_ALLOW_PAID_TESTS"
KEY_ENV = "DMX_API_KEY"

QUERY = "What is total revenue by region?"
FINDINGS = [{
    "result_summary": (
        "pandas/groupby -> 5 rows. North=1363760.55, South=1006074.27, "
        "East=1164964.55, West=1271795.18, Central=821174.97"
    ),
    "failed": False,
}]
CASES = {
    "grounded": """# Executive Summary
North leads with $1,363,760.55, followed by West at $1,271,795.18, East at
$1,164,964.55, South at $1,006,074.27, and Central at $821,174.97.""",
    "fabricated": """# Executive Summary
North generated $50 million, a 300% year-over-year increase driven by a new
premium product line. Partnership channel revenue tripled.""",
    "off_topic": """# Executive Summary
Customer churn spiked in March because support tickets rose 40% and a
competitor launched a 20%-off promotion.""",
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--base-url", default="https://www.dmxapi.cn/v1")
    parser.add_argument("--model", default=MODEL, choices=[MODEL])
    parser.add_argument("--output-dir", type=Path, default=Path("logs/dmxapi_smoke"))
    return parser.parse_args()


async def _run(args: argparse.Namespace, key: str) -> dict:
    llm = ChatOpenAI(
        model=args.model,
        api_key=key,
        base_url=args.base_url,
        temperature=0.2,
        max_tokens=2048,
        extra_body={"thinking": {"type": "disabled"}},
    )
    results = {}
    for name, report in CASES.items():
        rel, gnd = await score_relevance_and_groundedness(
            QUERY, report, FINDINGS, [], llm=llm, n_samples=1,
        )
        results[name] = {
            "answer_relevance": {"score": rel.score, "valid": rel.valid, "reasoning": rel.reasoning},
            "groundedness": {"score": gnd.score, "valid": gnd.valid, "reasoning": gnd.reasoning},
        }
    return {
        "probe": "dmxapi_maeda_judge_calibration_v1",
        "model": args.model,
        "thinking": "disabled",
        "samples_per_metric": 1,
        "expected": {
            "grounded": "relevance >= 0.7 and groundedness >= 0.7",
            "fabricated": "groundedness <= 0.4",
            "off_topic": "relevance <= 0.4 and groundedness valid and <= 0.4",
        },
        "results": results,
    }


def main() -> int:
    args = _args()
    if not args.execute:
        raise SystemExit("Refusing paid calls: pass --execute")
    if os.environ.get(ALLOW_ENV, "").lower() not in {"1", "true", "yes", "on"}:
        raise SystemExit(f"Refusing paid calls: set {ALLOW_ENV}=1 for this process")
    key = os.environ.get(KEY_ENV, "").strip()
    if not key or key.endswith("..."):
        raise SystemExit(f"Refusing paid calls: {KEY_ENV} is missing or a placeholder")

    result = asyncio.run(_run(args, key))
    scores = result["results"]
    passed = (
        scores["grounded"]["answer_relevance"]["valid"]
        and scores["grounded"]["answer_relevance"]["score"] >= 0.7
        and scores["grounded"]["groundedness"]["valid"]
        and scores["grounded"]["groundedness"]["score"] >= 0.7
        and scores["fabricated"]["groundedness"]["valid"]
        and scores["fabricated"]["groundedness"]["score"] <= 0.4
        and scores["off_topic"]["answer_relevance"]["valid"]
        and scores["off_topic"]["answer_relevance"]["score"] <= 0.4
        and scores["off_topic"]["groundedness"]["valid"]
        and scores["off_topic"]["groundedness"]["score"] <= 0.4
    )
    result["passed"] = passed
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"dmxapi_maeda_calibration_{int(time.time())}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for name, values in scores.items():
        print(
            f"{name}: relevance={values['answer_relevance']['score']:.3f} "
            f"groundedness={values['groundedness']['score']:.3f}"
        )
    print(f"{'PASS' if passed else 'FAIL'} artifact={path}")
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"DMXAPI MAEDA calibration FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
