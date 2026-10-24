from argparse import Namespace

import pytest

from scripts.check_dmxapi_judge import (
    MODEL,
    _model_alias_matches,
    _usage_cost,
    _validate_args,
)


def _args(**overrides):
    values = {
        "execute": True,
        "model": MODEL,
        "input_price": 0.79,
        "cached_input_price": 0.158,
        "output_price": 1.58,
        "max_cost_cny": 0.05,
    }
    values.update(overrides)
    return Namespace(**values)


def test_paid_probe_requires_both_execute_and_explicit_env(monkeypatch):
    monkeypatch.setenv("DMX_API_KEY", "sk-real-test-key")
    monkeypatch.delenv("MAEDA_ALLOW_PAID_TESTS", raising=False)
    with pytest.raises(SystemExit, match="MAEDA_ALLOW_PAID_TESTS"):
        _validate_args(_args())

    monkeypatch.setenv("MAEDA_ALLOW_PAID_TESTS", "1")
    with pytest.raises(SystemExit, match="--execute"):
        _validate_args(_args(execute=False))


def test_paid_probe_rejects_missing_or_placeholder_key(monkeypatch):
    monkeypatch.setenv("MAEDA_ALLOW_PAID_TESTS", "1")
    monkeypatch.setenv("DMX_API_KEY", "sk-...")
    with pytest.raises(SystemExit, match="DMX_API_KEY"):
        _validate_args(_args())


def test_usage_cost_separates_cached_and_uncached_tokens():
    usage = {
        "prompt_tokens": 1000,
        "prompt_tokens_details": {"cached_tokens": 600},
        "completion_tokens": 200,
    }
    cost = _usage_cost(usage, _args())
    expected = (400 * 0.79 + 600 * 0.158 + 200 * 1.58) / 1_000_000
    assert cost == pytest.approx(expected)


def test_paid_probe_rejects_negative_prices(monkeypatch):
    monkeypatch.setenv("MAEDA_ALLOW_PAID_TESTS", "1")
    monkeypatch.setenv("DMX_API_KEY", "sk-real-test-key")
    with pytest.raises(SystemExit, match="input-price"):
        _validate_args(_args(input_price=-1))


@pytest.mark.parametrize(
    ("returned", "expected"),
    [
        ("deepseek-v4-flash-0731", True),
        ("xopdeepseekv4flash0731", True),
        ("deepseek-v4-flash", False),
        ("xopdeepseekv4pro0731", False),
        (None, False),
    ],
)
def test_model_alias_matching_is_version_strict(returned, expected):
    assert _model_alias_matches(MODEL, returned) is expected
