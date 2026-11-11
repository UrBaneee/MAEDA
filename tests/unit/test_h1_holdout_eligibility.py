"""附录 HO: 预注册 §7.4 eligibility fields for the H1 holdout. No case is run."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import h1_holdout_eligibility as el  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def test_parse_list_and_scalar_params():
    assert el.parse('T_rev_entity_year[["region", "North"], 2022]') == ("T_rev_entity_year", [["region", "North"], 2022])
    assert el.parse('T_avgunits_dim"region"') == ("T_avgunits_dim", "region")


def test_differ_uses_one_percent_tolerance_and_exact_ids():
    assert not el.truths_differ({"x": 100.0}, {"x": 100.9})
    assert el.truths_differ({"x": 100.0}, {"x": 101.5})
    assert el.truths_differ({"top_rep_id": 1011}, {"top_rep_id": 1027})
    assert el.truths_differ({"x": 1.0}, None)


def test_repair_drops_null_and_negative_and_clips():
    df = pd.DataFrame({"revenue": [None, -5.0, 50.0, 5000.0], "units": [1, 2, 3, 99]})
    f = {"revenue": (10.0, 1000.0), "units": (0.0, 20.0)}
    t = el.repaired(df, f)
    assert list(t["revenue"]) == [50.0, 1000.0] and list(t["units"]) == [3, 20]
    assert list(el.defect_mask(df, f)) == [True, True, False, True]


def test_committed_output_covers_every_holdout_case_and_holdout_unchanged():
    out = json.loads((ROOT / "tests/eval/h1_holdout_eligibility.json").read_text())
    ids = {c["id"] for c in json.loads((ROOT / "tests/eval/h1_holdout_test.json").read_text())}
    assert set(out["cases"]) == ids and out["n_cases"] == 119
    manifest = json.loads((ROOT / "tests/eval/h1_holdout_manifest.json").read_text())
    assert out["holdout_sha256"] == manifest["cases_sha256"]
    assert all(c["corpus_backed"] is False for c in out["cases"].values())
