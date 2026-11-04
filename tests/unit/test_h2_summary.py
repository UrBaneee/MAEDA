"""H2 summary: refused runs score 0 in the primary analysis, count by their
text label in the sensitivity analysis, and are never dropped (附录 FM)."""
from __future__ import annotations

import pytest

from src.eval.h2_summary import summarise_h2


def _run(job, arm, case, event, etype="normal", refused=False):
    return {"job_id": job, "arm": arm, "case_id": case, "event_id": event,
            "event_type": etype, "terminal_state": "safe_refusal" if refused else "success"}


def test_refused_correct_scores_zero_in_primary_one_in_sensitivity():
    runs = [_run("b1", "B", "c1", "e1"), _run("r1", "R", "c1", "e1", refused=True)]
    s = summarise_h2(runs, {"b1": "none", "r1": "correct"})
    assert s.primary.per_event["e1"] == {"B": 0.0, "R": 0.0, "delta": 0.0}
    assert s.sensitivity.per_event["e1"]["delta"] == 1.0
    assert s.refusals == {"B": {"normal": [0, 1]}, "R": {"normal": [1, 1]}}
    assert s.refusal_rate["R"]["normal"] == 1.0


def test_trials_average_within_case_then_cases_within_event():
    runs = [
        _run("r1", "R", "c1", "e1"), _run("r2", "R", "c1", "e1"), _run("r3", "R", "c1", "e1"),
        _run("r4", "R", "c2", "e1"),
        _run("b1", "B", "c1", "e1"), _run("b2", "B", "c2", "e1"),
    ]
    labels = {"r1": "correct", "r2": "correct", "r3": "wrong", "r4": "none", "b1": "none", "b2": "none"}
    s = summarise_h2(runs, labels)
    # case c1 = 2/3, case c2 = 0, event = mean of cases = 1/3 (not 2/4 over trials)
    assert s.primary.per_event["e1"]["R"] == pytest.approx(1 / 3)


def test_sign_flip_over_events_and_floor():
    runs, labels = [], {}
    for i in range(8):
        runs += [_run(f"b{i}", "B", f"c{i}", f"e{i}"), _run(f"r{i}", "R", f"c{i}", f"e{i}")]
        labels[f"b{i}"], labels[f"r{i}"] = "none", "correct"
    s = summarise_h2(runs, labels)
    assert s.primary.n_events == 8
    assert s.primary.p_value == pytest.approx(2 / 2 ** 8)
    assert s.primary.significant and s.primary.above_floor and s.primary.exact
    assert s.notes == []


def test_fewer_than_seven_events_noted():
    runs = [_run("b", "B", "c", "e"), _run("r", "R", "c", "e")]
    s = summarise_h2(runs, {"b": "none", "r": "correct"})
    assert s.primary.n_events == 1 and not s.primary.significant
    assert "fewer than 7" in s.notes[0]


def test_descriptive_rates_and_unlabelled_count():
    runs = [
        _run("d1", "R", "cd", "ed", "distractor"), _run("d2", "R", "cd", "ed", "distractor", refused=True),
        _run("d3", "B", "cd", "ed", "distractor"),
        _run("u1", "B", "cu", "eu", "unexplained"),
        _run("x", "B", "c", "e"),
    ]
    s = summarise_h2(runs, {"d1": "wrong", "d2": "none", "d3": "none", "u1": "wrong"})
    assert s.descriptive_wrong_rate == {"distractor": {"B": 0.0, "R": 0.5}, "unexplained": {"B": 1.0}}
    assert s.n_unlabelled == 1
    assert s.primary.n_events == 0 and s.primary.p_value is None
