"""附录 GS: H1 sales ground truth comes from the clean table; the pipeline input stays dirty."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

CLEAN = "6c95334422ccfefd4a3a4815944418dd995c3bcdd9513b3aa4db0be76f7d771d"
DIRTY = "b863b60173f3c0bf30c4582738b85ba3304a7a212794d66ab14aa35dce422719"


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_clean_table_and_dirty_input_hashes_are_pinned():
    assert _sha(ROOT / "data/truth/sales_data_clean.csv") == CLEAN
    assert _sha(ROOT / "data/demo/sales_data.csv") == DIRTY
    assert not (ROOT / "data/demo/sales_data_clean.csv").exists()  # never an input


def test_clean_is_the_dirty_table_without_quality_noise():
    d = pd.read_csv(ROOT / "data/demo/sales_data.csv")
    c = pd.read_csv(ROOT / "data/truth/sales_data_clean.csv")
    assert len(c) == 48_000 and c.isna().sum().sum() == 0 and not c.duplicated().any()
    head = d.iloc[:48_000]
    other = [x for x in c.columns if x != "revenue"]
    assert (head[other].reset_index(drop=True) == c[other]).all().all()
    diff = head["revenue"].reset_index(drop=True)
    bad = diff.notna() & (diff != c["revenue"]) & ((diff - c["revenue"] * 10).abs() > 0.011)
    assert not bad.any()  # only missing or x10


def test_receive_refuses_wrong_hash_and_rows(tmp_path):
    import receive_clean_truth as r
    src = tmp_path / "a.csv"
    src.write_text("date,revenue\n2022-01-01,1\n")
    with pytest.raises(RuntimeError, match="sha256"):
        r.receive(src, tmp_path / "out.csv", sha256="0" * 64, rows=1)
    assert not (tmp_path / "out.csv").exists()
    with pytest.raises(RuntimeError, match="rows"):
        r.receive(src, tmp_path / "out.csv", sha256=_sha(src), rows=5)
    r.receive(src, tmp_path / "out.csv", sha256=_sha(src), rows=1)
    assert (tmp_path / "out.csv").read_bytes() == src.read_bytes()


def test_truth_loader_refuses_a_different_table(tmp_path, monkeypatch):
    import compute_ground_truth as cg
    (tmp_path / "sales_data_clean.csv").write_text("date,region,revenue,units\n2022-01-01,N,1,1\n")
    monkeypatch.setattr(cg, "TRUTH_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="sha256"):
        cg._Data().sales


def test_sales_truth_is_computed_on_the_clean_table():
    import compute_ground_truth as cg
    c = pd.read_csv(ROOT / "data/truth/sales_data_clean.csv")
    assert cg.COMPUTERS["E01"](cg._Data())["n_rows"] == len(c) == 48_000
    d01 = cg.COMPUTERS["D01"](cg._Data())
    assert d01["north_region_revenue"] == round(c[c.region == "North"].revenue.sum(), 2)


def test_30_new_dev_sales_cases():
    from h1_sales_cases import SPEC
    suite = {c["id"]: c for c in json.loads((ROOT / "tests/eval/test_suite.json").read_text())}
    assert len(SPEC) == 30 and len({s[0] for s in SPEC}) == 30
    for s in SPEC:
        c = suite[s[0]]
        assert c["split"] == "dev"
        assert c["data_source"]["path"] == "data/demo/sales_data.csv"
        assert c["ground_truth"] and all(isinstance(v, (int, float)) for v in c["ground_truth"].values())
        assert len(set(c["ground_truth"].values())) == len(c["ground_truth"])  # no two fields share a number


def test_suite_has_no_holdout_case_added():
    assert {c["split"] for c in json.loads((ROOT / "tests/eval/test_suite.json").read_text())} == {"dev"}


def test_h1_sample_is_40_dev_cases_without_e01_e02():
    sample = json.loads((ROOT / "tests/eval/h1_sample.json").read_text())
    ids = sample["case_ids"]
    suite = {c["id"]: c for c in json.loads((ROOT / "tests/eval/test_suite.json").read_text())}
    assert len(ids) == 40 and len(set(ids)) == 40
    assert "E01" not in ids and "E02" not in ids and set(sample["excluded"]) == {"E01", "E02"}
    for i in ids:
        c = suite[i]
        assert c["split"] == "dev" and c["data_source"]["path"] == "data/demo/sales_data.csv"
        assert any(isinstance(v, (int, float)) for k, v in c["ground_truth"].items() if k != "_note")
