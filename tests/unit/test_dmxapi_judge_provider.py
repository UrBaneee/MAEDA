import asyncio
from datetime import datetime, timezone
from unittest.mock import patch

from src.config.settings import MAEDASettings
from src.eval.metrics import _build_eval_llm, _judge_usage_summary, _run_judge_samples
from scripts.score_with_judge import _dmx_peak_window


def test_dmxapi_provider_resolves_default_model():
    cfg = MAEDASettings(
        _env_file=None,
        EVAL_LLM_PROVIDER="dmxapi",
        DMX_API_KEY="sk-test",
    )
    assert cfg.resolved_eval_provider == "dmxapi"
    assert cfg.resolved_eval_model == "deepseek-v4-flash-0731"


def test_dmxapi_factory_uses_dedicated_endpoint_key_and_disables_thinking():
    cfg = MAEDASettings(
        _env_file=None,
        EVAL_LLM_PROVIDER="dmxapi",
        EVAL_MODEL="deepseek-v4-flash-0731",
        DMX_API_KEY="sk-test",
        DMX_API_BASE_URL="https://www.dmxapi.cn/v1",
    )
    with patch("src.eval.metrics.settings", cfg):
        llm = _build_eval_llm(max_tokens=321)
    assert llm.model_name == "deepseek-v4-flash-0731"
    assert str(llm.openai_api_base).rstrip("/") == "https://www.dmxapi.cn/v1"
    assert llm.max_tokens == 321
    assert llm.extra_body == {"thinking": {"type": "disabled"}}


def test_dmxapi_usage_summary_records_cached_tokens_and_cny_cost():
    cfg = MAEDASettings(
        _env_file=None,
        EVAL_LLM_PROVIDER="dmxapi",
        EVAL_MODEL="deepseek-v4-flash-0731",
        DMX_INPUT_PRICE_CNY_PER_MILLION=0.79,
        DMX_CACHED_INPUT_PRICE_CNY_PER_MILLION=0.158,
        DMX_OUTPUT_PRICE_CNY_PER_MILLION=1.58,
    )
    samples = [
        {"usage": {"input_tokens": 1000, "cached_tokens": 600, "output_tokens": 200}},
        {"usage": {"input_tokens": 500, "cached_tokens": 0, "output_tokens": 100}},
    ]
    with patch("src.eval.metrics.settings", cfg):
        summary = _judge_usage_summary(samples, "answer_relevance")
    assert summary["provider"] == "dmxapi"
    assert summary["calls"] == 2
    assert summary["cached_input_tokens"] == 600
    expected = (900 * 0.79 + 600 * 0.158 + 300 * 1.58) / 1_000_000
    assert summary["cost_cny"] == round(expected, 8)


def test_groundedness_can_use_openai_sol_while_relevance_stays_dmxapi():
    cfg = MAEDASettings(
        _env_file=None,
        EVAL_LLM_PROVIDER="dmxapi",
        EVAL_MODEL="qwen3.8-max",
        EVAL_GROUNDEDNESS_PROVIDER="openai",
        EVAL_GROUNDEDNESS_MODEL="gpt-5.6-sol",
        EVAL_GROUNDEDNESS_REASONING_EFFORT="xhigh",
        OPENAI_API_KEY="sk-test",
    )
    with patch("src.eval.metrics.settings", cfg):
        llm = _build_eval_llm(max_tokens=2048, metric="groundedness")
    assert cfg.resolved_eval_model == "qwen3.8-max"
    assert cfg.resolved_groundedness_model == "gpt-5.6-sol"
    assert llm.model_name == "gpt-5.6-sol"
    assert llm.reasoning_effort == "xhigh"
    assert llm.max_tokens == 2048
    assert llm.max_completion_tokens == 16384
    assert not hasattr(llm, "temperature")


def test_judge_samples_warm_once_before_fanning_out():
    active = 0
    calls = 0
    first_finished = False

    async def sample():
        nonlocal active, calls, first_finished
        calls += 1
        call_number = calls
        if call_number > 1:
            assert first_finished is True
        active += 1
        await asyncio.sleep(0)
        active -= 1
        if call_number == 1:
            first_finished = True
        return {"score": 1.0}

    samples, errors = asyncio.run(_run_judge_samples(sample, 3))
    assert len(samples) == 3
    assert errors == []
    assert active == 0


def test_dmx_peak_window_uses_frozen_utc_weekday_schedule():
    assert _dmx_peak_window(datetime(2026, 8, 25, 1, 0, tzinfo=timezone.utc)) == "01:00-04:00 UTC"
    assert _dmx_peak_window(datetime(2026, 8, 25, 9, 59, tzinfo=timezone.utc)) == "06:00-10:00 UTC"
    assert _dmx_peak_window(datetime(2026, 8, 25, 4, 0, tzinfo=timezone.utc)) is None
    assert _dmx_peak_window(datetime(2026, 8, 29, 2, 0, tzinfo=timezone.utc)) is None
