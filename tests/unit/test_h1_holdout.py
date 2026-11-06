"""附录 HA.5/HB: the H1 holdout is generated, reproducible, disjoint from dev, and guarded."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_h1_holdout as g  # noqa: E402
from src.eval import reveal_guard as rg  # noqa: E402


def _cases():
    return json.loads((ROOT / "tests/eval/h1_holdout_test.json").read_text())


def test_quotas_scale_the_40_case_mix_to_119():
    q = g.quotas()
    assert sum(q.values()) == 119
    assert q["收入合计"] == 68 and q["销量合计"] == 18 and q["收入均值"] == 12
    assert g.quotas(40) == g.TYPE_COUNTS_40


def test_holdout_is_119_test_split_sales_cases_with_clean_truth():
    cs = _cases()
    assert len(cs) == 119 and len({c["id"] for c in cs}) == 119
    assert {c["split"] for c in cs} == {"test"}
    assert {c["data_source"]["path"] for c in cs} == {"data/demo/sales_data.csv"}
    for c in cs:
        vals = list(c["ground_truth"].values())
        assert vals and len(set(vals)) == len(vals) and all(isinstance(v, (int, float)) for v in vals)


def test_generation_is_reproducible_from_the_seed():
    cases, meta = g.generate()
    public = [{k: v for k, v in c.items() if not k.startswith("_")} for c in cases]
    on_disk = hashlib.sha256((ROOT / "tests/eval/h1_holdout_test.json").read_bytes()).hexdigest()
    manifest = json.loads((ROOT / "tests/eval/h1_holdout_manifest.json").read_text())
    assert manifest["seed"] == g.SEED and manifest["cases_sha256"] == on_disk
    assert public == _cases()


def test_no_overlap_with_dev_cases():
    dev = json.loads((ROOT / "tests/eval/test_suite.json").read_text())
    dev_vals = {tuple(sorted(round(float(v), 2) for v in d["ground_truth"].values() if isinstance(v, (int, float)))) for d in dev}
    dev_qs = {" ".join(d["query"].lower().split()) for d in dev}
    for c in _cases():
        assert tuple(sorted(round(float(v), 2) for v in c["ground_truth"].values())) not in dev_vals
        assert " ".join(c["query"].lower().split()) not in dev_qs


def test_holdout_is_not_in_the_dev_suite_or_sample():
    dev_ids = {c["id"] for c in json.loads((ROOT / "tests/eval/test_suite.json").read_text())}
    assert not dev_ids & {c["id"] for c in _cases()}
    assert not {c["id"] for c in _cases()} & set(json.loads((ROOT / "tests/eval/h1_sample.json").read_text())["case_ids"])


def test_every_exclusion_has_a_reason():
    m = json.loads((ROOT / "tests/eval/h1_holdout_manifest.json").read_text())
    assert m["final_count"] == 119 and all(e.get("reason") for e in m["exclusions"])


def test_reveal_guard_refuses_test_cases_without_the_flag(monkeypatch):
    monkeypatch.delenv(rg.REVEAL_ENV, raising=False)
    with pytest.raises(rg.RevealNotAuthorized):
        rg.require_reveal_authorization([{"id": "X", "split": "test"}])
    rg.require_reveal_authorization([{"id": "D", "split": "dev"}])
    monkeypatch.setenv(rg.REVEAL_ENV, "1")
    rg.require_reveal_authorization([{"id": "X", "split": "test"}])


def test_loading_the_holdout_file_is_refused_before_the_reveal(monkeypatch):
    from src.eval.runner import load_golden_suite
    monkeypatch.delenv(rg.REVEAL_ENV, raising=False)
    with pytest.raises(rg.RevealNotAuthorized):
        load_golden_suite(str(ROOT / "tests/eval/h1_holdout_test.json"))


def test_run_one_case_refuses_a_test_case_before_touching_the_graph(monkeypatch):
    import asyncio
    import run_eval
    from src.eval.runner import GoldenTestCase
    monkeypatch.delenv(rg.REVEAL_ENV, raising=False)
    tc = GoldenTestCase(id="T1", query="q", query_type="descriptive", expected_metrics=[],
                        expected_dimensions=[], ground_truth={}, split="test")

    class Boom:
        async def ainvoke(self, state):
            raise AssertionError("graph must not be invoked")

    with pytest.raises(rg.RevealNotAuthorized):
        asyncio.run(run_eval.run_one_case(tc, Boom(), None))
