"""Offline tests for scripts/decision_router_trial.py (no API call)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("drt", ROOT / "scripts" / "decision_router_trial.py")
drt = importlib.util.module_from_spec(_spec)
sys.modules["drt"] = drt
_spec.loader.exec_module(drt)

G = {"A": {"query_type": "descriptive", "query": "a"}, "B": {"query_type": "diagnostic", "query": "b"},
     "C": {"query_type": "comparative", "query": "c"}, "D": {"query_type": "exploratory", "query": "d"}}


class FakeClient:
    def __init__(self, answer="diagnostic", fail_first=0):
        self.calls, self.answer, self.fail_first, self.decisions = [], answer, fail_first, self

    def create(self, **req):
        self.calls.append(req)
        if self.fail_first:
            self.fail_first -= 1
            raise RuntimeError("429")
        probs = {t: (0.8 if t == self.answer else 0.05) for t in drt.TYPES}
        return {"answers": {"query_type": {"type": "choice", "choice": self.answer,
                                           "probabilities": probs, "confidence": 0.8}},
                "usage": {"total_input_tokens": 70}}


def test_question_contract_and_same_information_as_parser():
    q = drt.QUESTIONS["query_type"]
    assert q["type"] == "choice"
    assert set(q["criteria"]) == {"descriptive", "diagnostic", "predictive", "comparative", "exploratory"}
    assert all(v == "" for v in q["criteria"].values())  # no definitions, like the parser prompt
    from src.config.agent_prompts import INTENT_PARSER_SYSTEM
    for t in q["criteria"]:
        assert f'"{t}"' in INTENT_PARSER_SYSTEM


def test_parser_from_reasoning():
    assert drt.parser_from_reasoning("query_type correct: descriptive; metrics overlap: 100%") == "descriptive"
    assert drt.parser_from_reasoning("query_type mismatch: got comparative expected descriptive; x") == "comparative"
    assert drt.parser_from_reasoning("nothing here") is None


def test_score_accuracy_and_gate():
    pred = {"A": "diagnostic", "B": "diagnostic", "C": "comparative", "D": "descriptive"}
    r = drt.score(pred, G)
    assert r["n"] == 4 and r["accuracy"] == 0.5
    assert r["gate_fp"] == 1 and r["gate_fn"] == 1 and r["gate_accuracy"] == 0.5
    assert r["confusion"] == {"descriptive->diagnostic": 1, "exploratory->descriptive": 1}


def test_analyze_compares_on_common_cases():
    res = {c: {"choice": v["query_type"]} for c, v in G.items()}
    an = drt.analyze(res, G, {"parser_sigma": {"A": "comparative", "B": "diagnostic"}})
    assert an["decision_all"]["accuracy"] == 1.0
    assert an["parser_sigma"]["n"] == 2 and an["parser_sigma"]["accuracy"] == 0.5
    assert an["decision_on_parser_sigma_cases"]["n"] == 2


def test_execute_refuses_without_env(monkeypatch, tmp_path):
    monkeypatch.delenv(drt.ALLOW_ENV, raising=False)
    with pytest.raises(SystemExit, match="Refusing"):
        drt.execute(tmp_path / "o.json", client=FakeClient())


def test_execute_writes_resumes_retries(monkeypatch, tmp_path):
    monkeypatch.setenv(drt.ALLOW_ENV, "1")
    monkeypatch.setattr(drt, "gold", lambda: G)
    out, slept = tmp_path / "o.json", []
    c = FakeClient(fail_first=1)
    res = drt.execute(out, client=c, sleep=slept.append)
    assert len(res) == 4 and slept == [15] and len(c.calls) == 5
    assert c.calls[0]["input"] == "User question: a" and c.calls[0]["model"] == "gemini-decision-1"
    assert json.loads(out.read_text())["results"]["A"]["choice"] == "diagnostic"
    c2 = FakeClient()
    drt.execute(out, client=c2)
    assert c2.calls == []


def test_dry_run_no_network(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "google.genai", None)
    out = drt.dry_run()
    assert out["n"] == 130
    assert "no API call" in capsys.readouterr().out
