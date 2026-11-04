"""
H2 summary (预注册 §7.2.1, 附录 FM). Pure computation on saved runs and labels;
no API call.

Primary analysis: a trial scores 1 only if its label is `correct` AND the run
was not refused (`safe_refusal` delivers no report, so it scores 0). The
sensitivity analysis drops that condition and scores the text label alone.
Trials are averaged within a case, cases within an event (the event is the
unit, 附录 FI), and the per-event R − B differences go to the exact sign-flip
test. Refused runs are never dropped.

Distractor and unexplained events are descriptive only: the rate of `wrong`
per arm, from text labels, with the refusal count beside it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from src.eval.noise import average_by_event, event_sign_flip_test

REFUSED = "safe_refusal"
ALPHA = 0.0167        # §7.6
PRACTICAL_FLOOR = 0.028  # §7.6


@dataclass
class H2Test:
    n_events: int
    per_event: dict[str, dict[str, float]]   # event -> {"B": .., "R": .., "delta": ..}
    mean_delta: Optional[float]
    p_value: Optional[float]
    exact: Optional[bool]
    significant: Optional[bool]
    above_floor: Optional[bool]


@dataclass
class H2Summary:
    n_runs: int
    n_unlabelled: int
    refusal_rate: dict[str, dict[str, float]]          # arm -> event_type -> rate
    refusals: dict[str, dict[str, list[int]]]          # arm -> event_type -> [refused, total]
    primary: H2Test
    sensitivity: H2Test
    descriptive_wrong_rate: dict[str, dict[str, float]]  # event_type -> arm -> rate
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _event_scores(runs: list[dict], labels: dict[str, str], arm: str, primary: bool) -> dict[str, float]:
    by_case: dict[str, list[float]] = {}
    case_event: dict[str, str] = {}
    for r in runs:
        if r["arm"] != arm or r["event_type"] != "normal" or r["job_id"] not in labels:
            continue
        ok = labels[r["job_id"]] == "correct"
        if primary and r.get("terminal_state") == REFUSED:
            ok = False
        by_case.setdefault(r["case_id"], []).append(1.0 if ok else 0.0)
        case_event[r["case_id"]] = r["event_id"]
    case_means = {c: float(np.mean(v)) for c, v in by_case.items()}
    return average_by_event(case_means, case_event)


def _test(runs, labels, primary: bool, alpha: float, floor: float) -> H2Test:
    b = _event_scores(runs, labels, "B", primary)
    r = _event_scores(runs, labels, "R", primary)
    events = sorted(set(b) & set(r))
    per_event = {e: {"B": b[e], "R": r[e], "delta": r[e] - b[e]} for e in events}
    if not events:
        return H2Test(0, {}, None, None, None, None, None)
    res = event_sign_flip_test([per_event[e]["delta"] for e in events])
    return H2Test(
        n_events=res.n, per_event=per_event, mean_delta=res.mean_delta,
        p_value=res.p_value, exact=res.exact,
        significant=res.p_value < alpha, above_floor=res.mean_delta >= floor,
    )


def summarise_h2(runs: list[dict], labels: dict[str, str],
                 alpha: float = ALPHA, floor: float = PRACTICAL_FLOOR) -> H2Summary:
    """runs: dicts with job_id, arm (B/R), case_id, event_id, event_type,
    terminal_state. labels: job_id -> correct/wrong/none."""
    refusals: dict[str, dict[str, list[int]]] = {}
    for r in runs:
        cell = refusals.setdefault(r["arm"], {}).setdefault(r["event_type"], [0, 0])
        cell[0] += r.get("terminal_state") == REFUSED
        cell[1] += 1
    refusal_rate = {a: {t: c[0] / c[1] for t, c in d.items()} for a, d in refusals.items()}

    wrong: dict[str, dict[str, float]] = {}
    for etype in ("distractor", "unexplained"):
        for arm in sorted(refusals):
            rows = [r for r in runs if r["arm"] == arm and r["event_type"] == etype and r["job_id"] in labels]
            if rows:
                wrong.setdefault(etype, {})[arm] = sum(labels[r["job_id"]] == "wrong" for r in rows) / len(rows)

    notes = []
    primary = _test(runs, labels, True, alpha, floor)
    if primary.n_events and primary.n_events < 7:
        notes.append(f"{primary.n_events} events: p cannot fall below α={alpha} with fewer than 7 "
                     "(2/2^n), so no result here can be significant (§7.2.1).")
    return H2Summary(
        n_runs=len(runs),
        n_unlabelled=sum(1 for r in runs if r["job_id"] not in labels),
        refusal_rate=refusal_rate, refusals=refusals,
        primary=primary,
        sensitivity=_test(runs, labels, False, alpha, floor),
        descriptive_wrong_rate=wrong, notes=notes,
    )
