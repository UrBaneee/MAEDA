"""
Shared retry-on-rate-limit helper.

Three call sites independently needed "retry with backoff on a transient
429, but not on anything else, and not forever": src/eval/metrics.py's
judge, src/agents/analysis_agent.py's planner, and
src/agents/guardrail_agent.py's guardrail judge. All three used to share a
different, worse pattern instead -- a bare `except Exception` that treated
a rate limit identically to a permanently broken response. Found via
docs/noise_floor.md's incident writeup: under real rate-limit contention,
the planner silently produced an empty plan and the guardrail silently
defaulted every check to "pass", exactly the load condition where a
guardrail is least safe to fail open.
"""
from __future__ import annotations

import asyncio
import random
from typing import Awaitable, Callable, Optional, TypeVar

T = TypeVar("T")


def is_rate_limit_error(exc: BaseException) -> bool:
    """Duck-typed rate-limit detection across OpenAI's and Anthropic's
    client exception classes (both expose a 429 one way or another)
    without hard-importing either SDK's exception hierarchy here."""
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status == 429:
        return True
    return "429" in str(exc) or "rate_limit" in str(exc).lower()


async def call_with_rate_limit_retry(
    fn: Callable[[], Awaitable[T]],
    max_retries: int = 5,
    logger: Optional[object] = None,
    label: str = "call",
) -> T:
    """
    Await fn() -- a zero-arg async callable; bind other arguments with a
    closure or functools.partial at the call site -- retrying with
    exponential backoff + jitter whenever the raised exception looks like a
    rate limit. Any other exception, or a rate limit still failing after
    max_retries, propagates to the caller unchanged: this helper only
    decides whether to retry, never what to do when retries are exhausted
    or the failure is something else -- that fallback stays the caller's
    decision.
    """
    attempt = 0
    while True:
        try:
            return await fn()
        except Exception as exc:
            if not is_rate_limit_error(exc) or attempt >= max_retries:
                raise
            wait = min(2 ** attempt + random.random(), 30)
            if logger is not None:
                logger.warning(
                    "%s rate-limited (attempt %d/%d) — retrying in %.1fs",
                    label, attempt + 1, max_retries, wait,
                )
            await asyncio.sleep(wait)
            attempt += 1
