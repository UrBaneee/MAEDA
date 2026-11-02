"""附录 FL: the job plan is what the user approved, and no test-split event leaks in."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def _cases():
    return json.loads(Path("tests/eval/h2_dev_attribution_cases.json").read_text())


def test_formal_plan_is_sixty_reports_split_as_approved():
    from generate_h2_reports import jobs
    formal = jobs("formal", _cases())
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
