#!/usr/bin/env python3
"""Small, auditable DMXAPI judge qualification probe.

The script is inert unless both ``--execute`` and
``MAEDA_ALLOW_PAID_TESTS=1`` are present.  It sends one warm-up request and
then two identical requests concurrently so cache behaviour can be observed
without changing the judge prompt.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from src.config.settings import settings


MODEL = "deepseek-v4-flash-0731"
ALLOWED_MODELS = {MODEL, "deepseek-v4-pro", "qwen3.8-max"}
BASE_URL = "https://www.dmxapi.cn"
ALLOW_ENV = "MAEDA_ALLOW_PAID_TESTS"
KEY_ENV = "DMX_API_KEY"
TRUTHY = {"1", "true", "yes", "on"}

SYSTEM_PROMPT = """You are a bilingual evaluation judge. Return JSON only.
Score whether the answer is supported by the evidence. Use this exact shape:
{"score": 0.0, "reason": "brief explanation"}
score must be a number from 0 to 1. reason must be at most 30 words."""

USER_PROMPT = """Evidence / 证据:
2024年华东区 Widget A 的营收为 120 万元，同比增长 20%。

Answer to judge / 待评答案:
华东区 Widget A 在 2024 年营收为 120 万元，同比增长 20%。"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Actually send three paid calls")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--model", default=MODEL, choices=sorted(ALLOWED_MODELS))
    parser.add_argument("--input-price", type=float, required=True, help="CNY per 1M uncached input tokens")
    parser.add_argument("--cached-input-price", type=float, required=True, help="CNY per 1M cached input tokens")
    parser.add_argument("--output-price", type=float, required=True, help="CNY per 1M output tokens")
    parser.add_argument("--max-cost-cny", type=float, default=0.05)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path, default=Path("logs/dmxapi_smoke"))
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> str:
    if not args.execute:
        raise SystemExit("Refusing paid calls: pass --execute after reviewing the command.")
    if os.environ.get(ALLOW_ENV, "").strip().lower() not in TRUTHY:
        raise SystemExit(f"Refusing paid calls: set {ALLOW_ENV}=1 for this process only.")
    key = (os.environ.get(KEY_ENV) or settings.dmx_api_key or "").strip()
    if not key or key.endswith("..."):
        raise SystemExit(f"Refusing paid calls: {KEY_ENV} is missing or looks like a placeholder.")
    if args.max_cost_cny <= 0:
        raise SystemExit("--max-cost-cny must be positive")
    for name in ("input_price", "cached_input_price", "output_price"):
        if getattr(args, name) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative")
    return key


def _usage_cost(usage: dict[str, Any], args: argparse.Namespace) -> float:
    prompt = int(usage.get("prompt_tokens", 0))
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0)
    completion = int(usage.get("completion_tokens", 0))
    uncached = max(0, prompt - cached)
    return (
        uncached * args.input_price
        + cached * args.cached_input_price
        + completion * args.output_price
    ) / 1_000_000


def _model_alias_matches(requested: str, returned: Any) -> bool:
    """Accept a gateway's decorated alias without claiming origin proof.

    DMXAPI currently returns internal route names such as
    ``xopdeepseekv4flash0731`` for ``deepseek-v4-flash-0731``.  Removing
    punctuation and requiring the returned value to end in the complete
    requested identifier tolerates that prefix while still rejecting a
    different model or version.
    """
    if not isinstance(returned, str):
        return False
    def normalize(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", value.lower())

    return normalize(returned).endswith(normalize(requested))


async def _one_call(client: httpx.AsyncClient, args: argparse.Namespace, index: int) -> dict[str, Any]:
    started = time.perf_counter()
    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": args.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
            ],
            "thinking": {"type": "disabled"},
            "temperature": 0.2,
            "max_tokens": 256,
            "response_format": {"type": "json_object"},
        },
    )
    response.raise_for_status()
    body = response.json()
    content = body["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    score = parsed.get("score")
    if not isinstance(score, (int, float)) or isinstance(score, bool) or not 0 <= score <= 1:
        raise ValueError(f"call {index}: invalid score in JSON response")
    if not isinstance(parsed.get("reason"), str):
        raise ValueError(f"call {index}: missing string reason in JSON response")

    usage = body.get("usage") or {}
    returned_model = body.get("model")
    if not _model_alias_matches(args.model, returned_model):
        raise ValueError(
            f"call {index}: model provenance mismatch: requested {args.model!r}, "
            f"returned {returned_model!r}"
        )
    return {
        "index": index,
        "model": returned_model,
        "model_is_gateway_alias": returned_model != args.model,
        "latency_seconds": round(time.perf_counter() - started, 3),
        "score": float(score),
        "reason": parsed["reason"],
        "usage": usage,
        "cost_cny": round(_usage_cost(usage, args), 8),
    }


async def _run(args: argparse.Namespace, key: str) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"), headers=headers, timeout=args.timeout
    ) as client:
        warm = await _one_call(client, args, 1)
        projected = warm["cost_cny"] * 3
        if projected > args.max_cost_cny:
            raise RuntimeError(
                f"Warm-up cost projects to CNY {projected:.6f}, above cap "
                f"CNY {args.max_cost_cny:.6f}; remaining calls were not sent."
            )
        rest = await asyncio.gather(
            _one_call(client, args, 2),
            _one_call(client, args, 3),
        )

    calls = [warm, *rest]
    total = sum(item["cost_cny"] for item in calls)
    if total > args.max_cost_cny:
        raise RuntimeError(
            f"Actual cost CNY {total:.6f} exceeded cap CNY {args.max_cost_cny:.6f}"
        )
    return {
        "probe": "dmxapi_judge_qualification_v1",
        "base_url": args.base_url,
        "requested_model": args.model,
        "thinking": "disabled",
        "pattern": "warm_then_fan_out",
        "price_cny_per_million": {
            "input_uncached": args.input_price,
            "input_cached": args.cached_input_price,
            "output": args.output_price,
        },
        "max_cost_cny": args.max_cost_cny,
        "total_cost_cny": round(total, 8),
        "calls": calls,
    }


def main() -> int:
    args = _parse_args()
    key = _validate_args(args)
    try:
        result = asyncio.run(_run(args, key))
    except (httpx.HTTPError, KeyError, json.JSONDecodeError, ValueError, RuntimeError) as exc:
        print(f"DMXAPI judge probe FAILED: {exc}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"dmxapi_judge_{int(time.time())}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cached = [
        int((call["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0)
        for call in result["calls"]
    ]
    print(f"PASS model={result['requested_model']} scores={[c['score'] for c in result['calls']]}")
    print(f"cached_input_tokens={cached} total_cost_cny={result['total_cost_cny']}")
    print(f"artifact={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
