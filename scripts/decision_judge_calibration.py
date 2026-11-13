"""
Exploratory trial (branch experiment/gemini-decision): does Google's
Gemini Decision 1 (`gemini-decision-1`, Decisions API) agree with the
human reference better than the current LLM judges?

Not part of phase 4. The phase 4 results on `main` are untouched; this
script only reads cached reports and existing human labels.

What it does
------------
For each case it rebuilds the exact judge input the current judges see
(`src.eval.metrics._build_judge_prompt` over the case's own cached
replay state, so the report is the one the humans labelled), then asks
Decision 1 two `score` questions with five levels that match the human
label grid (0, 0.25, 0.5, 0.75, 1.0):

  * groundedness      (rubric from prereg §9: R1 sections, R2 deductions)
  * answer_relevance

Both questions go in ONE request per case (shared input billed once).
Decision 1 returns a probability per level. Pre-specified primary score
= the most probable level, mapped to level/4 (directly comparable to the
discrete human labels). The probability-weighted score is reported as a
descriptive secondary, never as the headline.

Primary comparison: the 20-case calibration set behind prereg §8.1
(human reference strict_lower and lenient_upper), with the same QWK
(`src.eval.agreement`, 5 equal-width bands) and the same bars
(groundedness 0.7167, answer_relevance 0.4032). Earlier judges on that
set: groundedness best 0.495/0.475 (GPT-5.6 Sol), answer_relevance best
0.726/0.637 (Qwen via dmxapi). Optional descriptive extra: the 100-case
pooled human labels.

Modes
-----
  python scripts/decision_judge_calibration.py                 # dry run (default): no API call,
                                                               # prints requests, token and cost estimate
  MAEDA_ALLOW_PAID_TESTS=1 python scripts/decision_judge_calibration.py --execute [--set pooled100]
  python scripts/decision_judge_calibration.py --analyze OUT.json   # offline QWK on a saved run

Requires GEMINI_API_KEY in .env only for --execute.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL = "gemini-decision-1"
PRICE_PER_M_INPUT = 0.10  # USD, output billed at 0 (Google AI Studio pricing, 2026-11)
ANN = Path("data/annotations")
CAL20_KEY = ANN / "answer_key_openai_gpt_5_6_sol_groundedness_xhigh_16k_nonpredictive20_complete.json"
REFS = {
    "strict": ANN / "human_reference_nonpredictive20_strict_lower.json",
    "lenient": ANN / "human_reference_nonpredictive20_lenient_upper.json",
}
POOLED = ANN / "human_labels_pooled100.json"
BARS = {"groundedness": 0.7167, "answer_relevance": 0.4032}
PRIOR_BEST = {
    "groundedness": {"strict": 0.495, "lenient": 0.475, "judge": "GPT-5.6 Sol xhigh"},
    "answer_relevance": {"strict": 0.726, "lenient": 0.637, "judge": "Qwen3.8 Max (dmxapi)"},
}
ALLOW_ENV = "MAEDA_ALLOW_PAID_TESTS"
# Conservative token estimate: ~3.5 chars per token for this English/number-heavy
# text, plus a fixed per-request overhead measured on the 2026-11-13 smoke call
# (154 tokens billed for ~150 chars of input + questions).
CHARS_PER_TOKEN = 3.5
REQUEST_OVERHEAD_TOKENS = 110

from src.config.agent_prompts import DECISION_JUDGE_QUESTIONS as QUESTIONS  # noqa: E402

METRICS = list(QUESTIONS)


# ---------------------------------------------------------------- inputs
def case_ids(which: str) -> list[str]:
    if which == "cal20":
        return list(json.loads(CAL20_KEY.read_text())["metadata"]["case_ids"])
    if which == "pooled100":
        return sorted(json.loads(POOLED.read_text())["labels"])
    raise ValueError(which)


def build_input(state: dict) -> str:
    from src.eval.metrics import _build_judge_prompt
    return _build_judge_prompt(
        state.get("user_query", ""),
        state.get("report") or "",
        state.get("analysis_results") or [],
        state.get("rag_context") or [],
        data_quality_report=state.get("data_quality_report"),
    )


def build_request(input_text: str) -> dict:
    return {"model": MODEL, "input": input_text, "questions": QUESTIONS}


def estimate_tokens(request: dict) -> int:
    q_chars = len(json.dumps(request["questions"]))
    return math.ceil((len(request["input"]) + q_chars) / CHARS_PER_TOKEN) + REQUEST_OVERHEAD_TOKENS


def load_requests(which: str) -> dict[str, dict]:
    from src.eval.replay_cache import ReplayCache
    cache = ReplayCache()
    out = {}
    for cid in case_ids(which):
        entry = cache._entries.get(cid)
        if entry is None:
            raise SystemExit(f"case {cid} missing from replay cache")
        out[cid] = build_request(build_input(entry.state))
    return out


# ---------------------------------------------------------------- scoring
def parse_answer(ans: dict) -> dict:
    """Decision `score` answer -> {'argmax': level/4, 'expected': score/4, 'probs': [...] }."""
    probs = ans["probabilities"]
    n = len(probs)
    p = [float(probs[str(i)]) for i in range(n)]
    top = max(range(n), key=lambda i: (p[i], -i))  # ties -> lower level (conservative)
    return {"argmax": top / (n - 1), "expected": float(ans["score"]) / (n - 1),
            "probs": p, "confidence": float(ans.get("confidence", max(p)))}


def human_labels(which: str) -> dict[str, dict[str, dict]]:
    if which == "cal20":
        return {name: json.loads(p.read_text())["labels"] for name, p in REFS.items()}
    return {"pooled": json.loads(POOLED.read_text())["labels"]}


def analyze(results: dict, which: str) -> dict:
    from src.eval.agreement import compute_agreement
    refs = human_labels(which)
    out = {}
    for metric in METRICS:
        for ref_name, labels in refs.items():
            ids = sorted(c for c in results if c in labels and metric in results[c].get("answers", {}))
            for variant in ("argmax", "expected"):
                h = [float(labels[c][metric]) for c in ids]
                j = [results[c]["answers"][metric][variant] for c in ids]
                rep = compute_agreement(h, j, metric).to_dict()
                rep["bar"] = BARS[metric]
                rep["meets_bar"] = rep["qwk"] >= BARS[metric]
                out[f"{metric}/{ref_name}/{variant}"] = rep
    return out


def print_analysis(an: dict, which: str) -> None:
    print(f"\nAgreement with human reference ({which}); primary = argmax")
    for key, r in an.items():
        metric, ref, variant = key.split("/")
        prior = PRIOR_BEST[metric].get(ref)
        tag = "PRIMARY" if variant == "argmax" else "descr."
        print(f"  {tag:7s} {metric:16s} {ref:8s} {variant:8s} n {r['n']:3d}  QWK {r['qwk']:+.3f}"
              f"  bar {r['bar']:.4f} {'MET' if r['meets_bar'] else 'missed'}"
              + (f"  (prior best {prior:.3f})" if prior is not None and which == "cal20" else ""))


# ---------------------------------------------------------------- run
def dry_run(which: str) -> dict:
    reqs = load_requests(which)
    toks = {c: estimate_tokens(r) for c, r in reqs.items()}
    total = sum(toks.values())
    cost = total / 1e6 * PRICE_PER_M_INPUT
    first = next(iter(reqs))
    print(f"DRY RUN, no API call. set={which}, model={MODEL}, {len(reqs)} requests "
          f"(1 per case, {len(METRICS)} questions each)")
    print(f"  example case {first}: input {len(reqs[first]['input'])} chars, ~{toks[first]} tokens")
    print(f"  estimated input tokens: total {total:,}, max per request {max(toks.values()):,}, "
          f"mean {total // len(toks):,}")
    print(f"  estimated cost at ${PRICE_PER_M_INPUT}/1M input (output $0): ${cost:.4f}"
          f"  (free tier: $0)")
    print(f"  free-tier pacing: 50,000 input tokens/min -> about {math.ceil(total / 50000)} min")
    return {"n": len(reqs), "tokens": total, "cost_usd": cost, "max_tokens": max(toks.values())}


def execute(which: str, out_path: Path, client=None, sleep=time.sleep) -> dict:
    if os.environ.get(ALLOW_ENV, "").strip().lower() not in {"1", "true", "yes", "on"}:
        raise SystemExit(f"Refusing API calls: set {ALLOW_ENV}=1 for this process")
    if out_path.suffix != ".json" or out_path.is_dir():
        raise SystemExit(f"--out must be a .json file: {out_path}")
    reqs = load_requests(which)
    if client is None:
        from dotenv import dotenv_values
        from google import genai
        key = os.environ.get("GEMINI_API_KEY") or dotenv_values(".env").get("GEMINI_API_KEY")
        if not key:
            raise SystemExit("Refusing API calls: GEMINI_API_KEY missing")
        client = genai.Client(api_key=key, http_options={"api_version": "v1beta"})
    done = json.loads(out_path.read_text())["results"] if out_path.exists() else {}
    q_sha = hashlib.sha256(json.dumps(QUESTIONS, sort_keys=True).encode()).hexdigest()
    used = 0
    for cid, req in reqs.items():
        if cid in done:
            continue
        for attempt in range(4):
            try:
                resp = client.decisions.create(**req)
                break
            except Exception as exc:  # rate limits on the free tier: back off, then give up
                if attempt == 3:
                    raise
                print(f"  {cid}: {type(exc).__name__}, retry in {15 * (attempt + 1)}s")
                sleep(15 * (attempt + 1))
        d = resp.model_dump() if hasattr(resp, "model_dump") else resp
        toks = int(d["usage"]["total_input_tokens"])
        used += toks
        done[cid] = {"answers": {m: parse_answer(d["answers"][m]) for m in METRICS},
                     "input_tokens": toks, "input_sha256": hashlib.sha256(req["input"].encode()).hexdigest()}
        out_path.write_text(json.dumps({"metadata": {"model": MODEL, "set": which, "questions_sha256": q_sha,
                                                     "primary": "argmax"}, "results": done}, indent=1))
        print(f"  {cid}: {toks} tokens")
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=["cal20", "pooled100"], default="cal20")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--analyze", default=None, help="saved run JSON to analyze offline")
    a = ap.parse_args(argv)
    if a.analyze:
        saved = json.loads(Path(a.analyze).read_text())
        print_analysis(analyze(saved["results"], saved["metadata"]["set"]), saved["metadata"]["set"])
        return 0
    if not a.execute:
        dry_run(a.set)
        return 0
    out = Path(a.out or f"data/gemini_decision/decision_{a.set}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    res = execute(a.set, out)
    print_analysis(analyze(res, a.set), a.set)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
