"""FE.13 E1 and G4: checked against data whose true parameters are known."""
import numpy as np
import pytest

from src.eval.noise import (
    average_by_event,
    decompose_paired_variance,
    event_sign_flip_test,
    required_n,
    sigma_separates_n,
)


def _synthetic(sigma_hetero, sigma_noise, n=400, k=5, seed=1):
    rng = np.random.default_rng(seed)
    effect = rng.normal(0.05, sigma_hetero, n)
    a = {f"c{i}": list(effect[i] + rng.normal(0, sigma_noise, k)) for i in range(n)}
    b = {f"c{i}": list(rng.normal(0, sigma_noise, k)) for i in range(n)}
    return a, b


@pytest.mark.parametrize("truth", [0.05, 0.10])
def test_decomposition_recovers_known_sigma_hetero(truth):
    d = decompose_paired_variance(*_synthetic(truth, 0.06), n_resamples=2000)
    assert d.sigma_hetero == pytest.approx(truth, abs=0.01)
    assert d.repeat_var == pytest.approx(2 * 0.06 ** 2, rel=0.1)
    assert d.mean_delta == pytest.approx(0.05, abs=0.01)


def test_interval_covers_the_truth_at_about_its_nominal_rate():
    hits = 0
    for seed in range(60):
        low, high = decompose_paired_variance(
            *_synthetic(0.10, 0.06, n=100, seed=seed), n_resamples=1000).sigma_hetero_ci
        hits += low < 0.10 < high
    assert hits / 60 >= 0.85


def test_no_heterogeneity_floors_near_zero():
    d = decompose_paired_variance(*_synthetic(0.0, 0.06), n_resamples=2000)
    assert d.sigma_hetero < 0.02


def test_single_repeat_cases_are_dropped_and_counted():
    a, b = _synthetic(0.05, 0.06, n=5)
    a["c0"] = a["c0"][:1]
    assert decompose_paired_variance(a, b, n_resamples=100).n_dropped == 1


def test_too_few_cases_is_refused():
    with pytest.raises(ValueError):
        decompose_paired_variance({"c": [1.0, 2.0]}, {"c": [1.0, 2.0]})


def test_required_n_matches_hand_calculation():
    assert required_n(0.01) == int(np.ceil(1.2 * 2.241 ** 2 * 0.01 / 0.028 ** 2))


def test_wide_interval_fails_ej4_criterion():
    wide = decompose_paired_variance(*_synthetic(0.1, 0.06, n=6, seed=3), n_resamples=2000)
    tight = decompose_paired_variance(*_synthetic(0.05, 0.06, n=2000), n_resamples=500)
    assert sigma_separates_n(wide) is False
    assert sigma_separates_n(tight) is True


def test_sign_flip_smallest_possible_p():
    assert event_sign_flip_test([0.5] * 5).p_value == pytest.approx(2 / 2 ** 5)
    assert event_sign_flip_test([0.5] * 7).p_value == pytest.approx(2 / 2 ** 7)


def test_sign_flip_is_exact_for_a_hand_case():
    # Sums of ±1 ±2 ±3 are ±6, ±4, ±2, ±0: |sum| >= 6 in 2 of 8, >= 2 in 6 of 8.
    assert event_sign_flip_test([1.0, 2.0, 3.0]).p_value == pytest.approx(0.25)
    assert event_sign_flip_test([1.0, -2.0, 3.0]).p_value == pytest.approx(0.75)


def test_sign_flip_random_branch_has_no_zero_p():
    result = event_sign_flip_test([1.0] * 20, n_random=2000)
    assert result.exact is False and result.p_value > 0


def test_sign_flip_keeps_nominal_size_under_the_null():
    rng = np.random.default_rng(0)
    rejections = [event_sign_flip_test(list(rng.normal(0, 1, 8))).p_value <= 0.05 for _ in range(600)]
    assert np.mean(rejections) <= 0.08


def test_cases_on_one_event_average_to_one_observation():
    out = average_by_event({"a": 1.0, "b": 0.0, "c": 1.0}, {"a": "E1", "b": "E1", "c": "E2"})
    assert out == {"E1": 0.5, "E2": 1.0}
