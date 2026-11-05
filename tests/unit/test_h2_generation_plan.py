"""附录 FL: the job plan is what the user approved, and no test-split event leaks in."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def _cases():
    return json.loads(Path("tests/eval/h2_dev_attribution_cases.json").read_text())


def _v1_cases():
    # 附录 GB: the case file now holds 13 dev events (7 v1 + 6 v2, 26 cases).
    # The user approved 60 reports (附录 FL) for the 7 v1 dev events; that
    # approval is pinned on this subset. The v2 slots are named EVT-V2-NNN.
    return [c for c in _cases() if not c["event_id"].startswith("EVT-V2-")]


def test_full_dev_pool_plan_exceeds_the_approved_sixty():
    """Recorded, not hidden: running the formal plan over all 13 dev events
    is 132 reports, not the approved 60. It needs a user decision (附录 GB)."""
    from generate_h2_reports import jobs
    assert len(_cases()) == 26 and len(jobs("formal", _cases())) == 132


def test_formal_plan_is_sixty_reports_split_as_approved():
    from generate_h2_reports import jobs
    formal = jobs("formal", _v1_cases())
    assert len(formal) == 60
    types = {c["id"]: c["event_type"] for c in _cases()}
    assert sum(types[j["case_id"]] == "normal" for j in formal) == 48
    assert {j["arm"] for j in formal} == {"B", "R"}
    assert len({j["job_id"] for j in formal}) == 60


def test_dry_run_is_six_reports_outside_the_sixty():
    from generate_h2_reports import jobs
    dry, formal = jobs("dry", _cases()), jobs("formal", _cases())
    assert len(dry) == 6
    assert not {j["job_id"] for j in dry} & {j["job_id"] for j in formal}


def test_only_dev_events_and_distractors_have_no_cause():
    for case in _cases():
        assert case["split"] == "dev"
        assert (case["event_cause"] is None) == (case["event_type"] != "normal")
