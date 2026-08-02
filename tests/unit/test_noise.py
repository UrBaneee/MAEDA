"""
Tests for src/eval/noise.py — the noise-floor utilities used by
scripts/measure_noise.py (eval v2 Step 1, see docs/eval_v2_plan.md).
"""
import pytest

from src.eval.noise import paired_bootstrap_ci, summarize


def test_summarize_returns_none_below_two_samples():
    assert summarize("x", []) is None
    assert summarize("x", [0.5]) is None


def test_summarize_basic_stats():
    s = summarize("x", [0.6, 0.7, 0.8])
    assert s.n == 3
    assert s.mean == pytest.approx(0.7)
    assert s.min == 0.6
    assert s.max == 0.8
    assert s.two_sigma == pytest.approx(2 * s.std)


def test_paired_bootstrap_ci_requires_at_least_two_deltas():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([0.1])


def test_paired_bootstrap_ci_no_real_change_is_not_significant():
    """Deltas centered on 0 with real spread -- the CI should straddle 0,
    i.e. the method correctly reports 'not significant' when nothing
    actually changed. This is the same sanity check
    scripts/measure_noise.py runs on real trial data."""
    deltas = [0.02, -0.03, 0.01, -0.01, 0.04, -0.04, 0.0, 0.02, -0.02, 0.01]
    result = paired_bootstrap_ci(deltas, seed=0)
    assert result.n == 10
    assert result.ci_low <= 0.0 <= result.ci_high
    assert result.significant is False


def test_paired_bootstrap_ci_detects_a_real_shift():
    """Deltas consistently positive and far from 0 relative to their spread
    -- the CI should exclude 0, i.e. the method reports 'significant' when
    there's a real, consistent effect."""
    deltas = [0.20, 0.22, 0.19, 0.21, 0.23, 0.20, 0.18, 0.21, 0.22, 0.20]
    result = paired_bootstrap_ci(deltas, seed=0)
    assert result.ci_low > 0.0
    assert result.significant is True


def test_paired_bootstrap_ci_is_deterministic_given_a_seed():
    deltas = [0.1, -0.05, 0.15, 0.02, -0.08, 0.11, 0.03, -0.02]
    a = paired_bootstrap_ci(deltas, seed=42)
    b = paired_bootstrap_ci(deltas, seed=42)
    assert a.ci_low == b.ci_low
    assert a.ci_high == b.ci_high
