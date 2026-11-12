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


# 附录 HP/HQ/HR: reveal plan. Synthetic cases only; no test-split content, nothing run.
import pytest  # noqa: E402

from src.config.settings import settings  # noqa: E402
from src.eval.reveal_guard import REVEAL_ENV, RevealNotAuthorized  # noqa: E402


def _synthetic_test_cases(n_events=12, probes=2):
    return [{"id": f"H2-SYN-{e:02d}-Q{q}", "event_id": f"EVT-SYN-{e:02d}", "event_type": "normal",
             "split": "test", "query": "synthetic"} for e in range(n_events) for q in range(1, probes + 1)]


def test_reveal_plan_is_b_r_three_trials_plus_arm_a_two():
    from generate_h2_reports import jobs
    plan = jobs("reveal", _synthetic_test_cases())
    by_arm = {a: [j for j in plan if j["arm"] == a] for a in ("B", "R", "A")}
    assert len(by_arm["B"]) == len(by_arm["R"]) == 24 * 3
    assert len(by_arm["A"]) == 24 * 2
    assert len({j["job_id"] for j in plan}) == len(plan) == 192


def test_formal_and_dry_plans_never_include_arm_a():
    from generate_h2_reports import jobs
    assert {j["arm"] for j in jobs("formal", _cases()) + jobs("dry", _cases())} == {"B", "R"}


def test_arm_a_is_auto_auto_and_b_r_keep_cleaner_off():
    from generate_h2_reports import ARMS
    assert ARMS == {"B": ("force_off", "force_off"), "R": ("force_off", "force_on"), "A": ("auto", "auto")}


@pytest.fixture
def reveal_ok(monkeypatch):
    monkeypatch.setattr(settings, "rag_db_path", "/x/tb5_full.db")
    monkeypatch.setattr(settings, "intent_refine_trigger", "always")
    monkeypatch.setenv(REVEAL_ENV, "1")
    return monkeypatch


def test_reveal_preflight_passes_when_all_hold(reveal_ok):
    from generate_h2_reports import preflight
    assert preflight("reveal") is None and preflight("formal") is None


@pytest.mark.parametrize("breaker,needle", [
    (lambda m: m.delenv(REVEAL_ENV), REVEAL_ENV),
    (lambda m: m.setattr(settings, "intent_refine_trigger", "if_unresolved"), "always"),
    (lambda m: m.setattr(settings, "rag_db_path", "/x/tb5_dev.db"), "tb5_full.db"),
])
def test_reveal_preflight_refuses(reveal_ok, breaker, needle):
    from generate_h2_reports import preflight
    breaker(reveal_ok)
    assert needle in preflight("reveal")


def test_test_split_cases_refused_without_flag_in_any_plan(tmp_path, monkeypatch):
    from generate_h2_reports import load_cases
    monkeypatch.delenv(REVEAL_ENV, raising=False)
    p = tmp_path / "cases.json"
    p.write_text(json.dumps(_synthetic_test_cases(1, 1)))
    with pytest.raises(RevealNotAuthorized):
        load_cases(p)
    assert len(load_cases(Path("tests/eval/h2_dev_attribution_cases.json"))) == 26
