"""
Tests for src/eval/agreement.py -- QWK/Spearman/confusion-matrix stats
used by scripts/compute_agreement.py (eval v2 Step 3, docs/eval_v2_plan.md).
"""
import random

import pytest

from src.eval.agreement import compute_agreement, discretize


def test_discretize_boundaries():
    assert discretize(0.0) == 0
    assert discretize(1.0) == 4          # top score lands in the top band, not a phantom 5th
    assert discretize(0.5) == 2
    assert discretize(0.99) == 4


def test_discretize_clamps_out_of_range_input():
    assert discretize(-0.5) == 0
    assert discretize(1.5) == 4


def test_discretize_respects_custom_band_count():
    assert discretize(1.0, n_bands=3) == 2
    assert discretize(0.0, n_bands=3) == 0


def test_perfect_agreement_gives_qwk_near_one():
    scores = [0.0, 0.25, 0.5, 0.75, 1.0] * 4
    report = compute_agreement(scores, scores, metric="answer_relevance")
    assert report.qwk == pytest.approx(1.0, abs=1e-6)
    assert report.spearman == pytest.approx(1.0, abs=1e-6)
    assert report.mae == pytest.approx(0.0, abs=1e-9)
    assert report.exact_match_rate == 1.0


def test_constant_offset_still_ranks_perfectly_by_spearman():
    """A systematic +0.05 offset preserves rank order (Spearman=1.0) but
    can push some pairs into different discrete bands, so QWK isn't
    necessarily 1.0 -- this is exactly why both statistics are reported,
    not just one."""
    human = [0.1, 0.3, 0.5, 0.7, 0.9]
    judge = [h + 0.05 for h in human]
    report = compute_agreement(human, judge, metric="groundedness")
    assert report.spearman == pytest.approx(1.0, abs=1e-6)


def test_independent_random_scores_give_low_qwk():
    random.seed(0)
    human = [random.random() for _ in range(200)]
    judge = [random.random() for _ in range(200)]
    report = compute_agreement(human, judge, metric="answer_relevance")
    assert report.qwk < 0.2, f"expected near-zero agreement for independent random scores, got {report.qwk}"


def test_systematic_disagreement_in_one_band_shows_up_in_confusion_matrix():
    """Human consistently rates band-2 samples as band-0 (e.g. a rubric
    misunderstanding); the confusion matrix should show that concentration
    at [row=0][col=2] (human band 0, judge band 2) rather than spread
    evenly across the table."""
    human = [0.0] * 10  # human always says "band 0"
    judge = [0.5] * 10  # judge always says "band 2"
    report = compute_agreement(human, judge, metric="groundedness")
    assert report.confusion[0][2] == 10
    assert sum(sum(row) for row in report.confusion) == 10


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        compute_agreement([0.5, 0.5], [0.5], metric="x")


def test_too_few_samples_raise():
    with pytest.raises(ValueError):
        compute_agreement([0.5], [0.5], metric="x")


def test_report_to_dict_and_summary_line_are_well_formed():
    report = compute_agreement([0.0, 0.5, 1.0], [0.0, 0.5, 1.0], metric="answer_relevance")
    d = report.to_dict()
    assert d["metric"] == "answer_relevance"
    assert d["n"] == 3
    assert "QWK=" in report.summary_line()
