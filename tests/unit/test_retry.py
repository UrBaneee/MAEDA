"""
Tests for src/utils/retry.py -- the shared rate-limit retry helper used by
src/eval/metrics.py's judge, src/agents/analysis_agent.py's planner, and
src/agents/guardrail_agent.py's guardrail judge (see docs/noise_floor.md's
incident writeup for why this exists).
"""
from unittest.mock import AsyncMock

import pytest

from src.utils.retry import call_with_rate_limit_retry, is_rate_limit_error


class _FakeRateLimitError(Exception):
    status_code = 429


def test_is_rate_limit_error_detects_status_code():
    assert is_rate_limit_error(_FakeRateLimitError("boom"))


def test_is_rate_limit_error_detects_message_text():
    assert is_rate_limit_error(RuntimeError("Error code: 429 rate_limit_exceeded"))


def test_is_rate_limit_error_rejects_unrelated_errors():
    assert not is_rate_limit_error(RuntimeError("malformed response"))
    assert not is_rate_limit_error(ValueError("no JSON found"))


@pytest.mark.asyncio
async def test_retries_on_rate_limit_then_succeeds(monkeypatch):
    monkeypatch.setattr("src.utils.retry.asyncio.sleep", AsyncMock())
    calls = AsyncMock(side_effect=[_FakeRateLimitError("429"), "ok"])
    result = await call_with_rate_limit_retry(calls)
    assert result == "ok"
    assert calls.await_count == 2


@pytest.mark.asyncio
async def test_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr("src.utils.retry.asyncio.sleep", AsyncMock())
    calls = AsyncMock(side_effect=_FakeRateLimitError("429"))
    with pytest.raises(_FakeRateLimitError):
        await call_with_rate_limit_retry(calls, max_retries=3)
    assert calls.await_count == 4  # 1 initial + 3 retries


@pytest.mark.asyncio
async def test_non_rate_limit_error_is_not_retried(monkeypatch):
    sleep_mock = AsyncMock()
    monkeypatch.setattr("src.utils.retry.asyncio.sleep", sleep_mock)
    calls = AsyncMock(side_effect=RuntimeError("malformed"))
    with pytest.raises(RuntimeError):
        await call_with_rate_limit_retry(calls)
    assert calls.await_count == 1
    sleep_mock.assert_not_awaited()
