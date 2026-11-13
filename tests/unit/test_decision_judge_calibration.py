"""Offline tests for scripts/decision_judge_calibration.py (no API call)."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("djc", ROOT / "scripts" / "decision_judge_calibration.py")
djc = importlib.util.module_from_spec(_spec)
sys.modules["djc"] = djc
_spec.loader.exec_module(djc)


def _answer(probs, conf=None):
    n = len(probs)
    return {"type": "score", "probabilities": {str(i): p for i, p in enumerate(probs)},
            "score": sum(i * p for i, p in enumerate(probs)),
            "legend": {str(i): f"l{i}" for i in range(n)},
            "confidence": conf if conf is not None else max(probs)}


class FakeClient:
    def __init__(self, fail_first=0):
        self.calls = []
        self.fail_first = fail_first
        self.decisions = self

    def create(self, **req):
        self.calls.append(req)
        if self.fail_first:
            self.fail_first -= 1
            raise RuntimeError("429 rate limited")
        return {"answers": {"groundedness": _answer([0, 0, 0.2, 0.7, 0.1]),
                            "answer_relevance": _answer([0, 0, 0, 0.1, 0.9])},
                "usage": {"total_input_tokens": 1234, "total_output_tokens": 0}}


def test_questions_match_api_contract():
    assert set(djc.QUESTIONS) == {"groundedness", "answer_relevance"}
    for key, q in djc.QUESTIONS.items():
        assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", key)
        assert q["type"] == "score"
        assert isinstance(q["instructions"], str) and q["instructions"]
        assert len(q["criteria"]) == 5  # matches human grid 0/.25/.5/.75/1
    assert "Executive Summary" in djc.QUESTIONS["groundedness"]["instructions"]


def test_build_request_shape():
    r = djc.build_request("hello")
    assert r == {"model": "gemini-decision-1", "input": "hello", "questions": djc.QUESTIONS}


def test_parse_answer_argmax_and_expected():
    a = djc.parse_answer(_answer([0, 0, 0.2, 0.7, 0.1]))
    assert a["argmax"] == 0.75
    assert a["expected"] == pytest.approx((0.4 + 2.1 + 0.4) / 4)
    tie = djc.parse_answer(_answer([0, 0.5, 0.5, 0, 0]))
    assert tie["argmax"] == 0.25  # tie resolves to the lower level


def test_argmax_lands_in_same_band_as_human_label():
    from src.eval.agreement import discretize
    for level in range(5):
        probs = [0.0] * 5
        probs[level] = 1.0
        assert discretize(djc.parse_answer(_answer(probs))["argmax"]) == discretize(level / 4)


def test_estimate_tokens_monotone_and_has_overhead():
    small, big = djc.build_request("x"), djc.build_request("x" * 7000)
    assert djc.estimate_tokens(small) > djc.REQUEST_OVERHEAD_TOKENS
    assert djc.estimate_tokens(big) - djc.estimate_tokens(small) == pytest.approx(2000, abs=2)


def test_execute_refuses_without_env(monkeypatch, tmp_path):
    monkeypatch.delenv(djc.ALLOW_ENV, raising=False)
    with pytest.raises(SystemExit, match="Refusing"):
        djc.execute("cal20", tmp_path / "o.json", client=FakeClient())


def test_execute_rejects_non_json_out(monkeypatch, tmp_path):
    monkeypatch.setenv(djc.ALLOW_ENV, "1")
    with pytest.raises(SystemExit, match=".json"):
        djc.execute("cal20", tmp_path, client=FakeClient())


@pytest.fixture
def two_cases(monkeypatch):
    reqs = {"A1": djc.build_request("in a"), "B2": djc.build_request("in b")}
    monkeypatch.setattr(djc, "load_requests", lambda which: reqs)
    monkeypatch.setenv(djc.ALLOW_ENV, "1")
    return reqs


def test_execute_writes_resumes_and_retries(two_cases, tmp_path):
    out = tmp_path / "o.json"
    slept = []
    c = FakeClient(fail_first=1)
    res = djc.execute("cal20", out, client=c, sleep=slept.append)
    assert set(res) == {"A1", "B2"} and slept == [15] and len(c.calls) == 3
    saved = json.loads(out.read_text())
    assert saved["metadata"]["primary"] == "argmax"
    assert saved["results"]["A1"]["answers"]["groundedness"]["argmax"] == 0.75
    assert saved["results"]["A1"]["input_tokens"] == 1234
    c2 = FakeClient()
    djc.execute("cal20", out, client=c2)
    assert c2.calls == []  # resume: nothing re-sent


def test_execute_gives_up_after_retries(two_cases, tmp_path):
    with pytest.raises(RuntimeError):
        djc.execute("cal20", tmp_path / "o.json", client=FakeClient(fail_first=99), sleep=lambda s: None)


def test_analyze_perfect_and_bar(monkeypatch):
    labels = {f"c{i}": {"groundedness": lv, "answer_relevance": lv}
              for i, lv in enumerate([0.25, 0.5, 0.75, 1.0] * 3)}
    monkeypatch.setattr(djc, "human_labels", lambda which: {"strict": labels})
    res = {c: {"answers": {m: {"argmax": v[m], "expected": v[m]} for m in djc.METRICS}} for c, v in labels.items()}
    an = djc.analyze(res, "cal20")
    r = an["groundedness/strict/argmax"]
    assert r["n"] == 12 and r["qwk"] == pytest.approx(1.0) and r["meets_bar"]
    assert an["answer_relevance/strict/argmax"]["bar"] == 0.4032


def test_dry_run_makes_no_network_call(monkeypatch, capsys):
    monkeypatch.setattr(djc, "load_requests", lambda which: {"A1": djc.build_request("x" * 350)})
    monkeypatch.setitem(sys.modules, "google.genai", None)  # import would fail if attempted
    out = djc.dry_run("cal20")
    assert out["n"] == 1 and out["cost_usd"] < 0.001
    assert "no API call" in capsys.readouterr().out


_CACHE_OK = (ROOT / "data").exists()


@pytest.mark.skipif(not _CACHE_OK, reason="needs local replay cache")
def test_real_cal20_inputs_match_current_judge_prompt():
    from src.eval.metrics import _build_judge_prompt
    from src.eval.replay_cache import ReplayCache
    try:
        reqs = djc.load_requests("cal20")
    except SystemExit as e:
        pytest.skip(str(e))
    assert len(reqs) == 20
    st = ReplayCache()._entries["C02"].state
    assert reqs["C02"]["input"] == _build_judge_prompt(
        st.get("user_query", ""), st.get("report") or "", st.get("analysis_results") or [],
        st.get("rag_context") or [], data_quality_report=st.get("data_quality_report"))
    refs = djc.human_labels("cal20")
    assert set(reqs) == set(refs["strict"]) == set(refs["lenient"])
