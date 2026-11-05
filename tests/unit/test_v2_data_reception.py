"""附录 GB / FZ.6 / GA.2: receiving the v2 sales CSV and recording data versions."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from src.eval.data_provenance import data_version, tb5_content_hash  # noqa: E402


def _csv(tmp_path, rows=3):
    p = tmp_path / "src.csv"
    p.write_text("date,region,revenue\n" + "".join(f"2022-01-0{i},N,{i}\n" for i in range(1, rows + 1)))
    return p


def test_generate_sales_copies_verified_csv(tmp_path, monkeypatch):
    import generate_demo_data as g
    src = _csv(tmp_path)
    out = tmp_path / "out"
    monkeypatch.setattr(g, "OUT", out)
    df = g.generate_sales(source=src, expected_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                          expected_rows=3)
    assert len(df) == 3
    assert (out / "sales_data.csv").read_bytes() == src.read_bytes()


def test_generate_sales_refuses_hash_mismatch_and_writes_nothing(tmp_path, monkeypatch):
    import generate_demo_data as g
    out = tmp_path / "out"
    monkeypatch.setattr(g, "OUT", out)
    with pytest.raises(RuntimeError, match="sha256"):
        g.generate_sales(source=_csv(tmp_path), expected_sha256="0" * 64, expected_rows=3)
    assert not (out / "sales_data.csv").exists()


def test_generate_sales_refuses_row_count_mismatch(tmp_path, monkeypatch):
    import generate_demo_data as g
    src = _csv(tmp_path)
    monkeypatch.setattr(g, "OUT", tmp_path / "out")
    with pytest.raises(RuntimeError, match="rows"):
        g.generate_sales(source=src, expected_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                         expected_rows=99)


def test_generate_sales_constants_match_fz4():
    import generate_demo_data as g
    assert g._RENDERED_SALES_SHA256.startswith("b863b60173f3c0bf")
    assert g._RENDERED_SALES_ROWS == 48_960


def test_h2_dev_cases_read_h2_probes_and_dev_only():
    import build_h2_dev_cases as b
    events = [
        {"event_id": "EVT-X-1", "event_type": "normal", "split": "dev", "business_description": "d",
         "valid_from": "2023-01-01", "valid_to": "2023-01-02", "probe_queries": ["old"],
         "h2_probes": ["new1", "new2"], "data_signal": {"metric": "revenue", "segment": {},
                                                        "direction": "up", "magnitude": 0.2}},
        {"event_id": "EVT-X-2", "event_type": "normal", "split": "test", "business_description": "t",
         "valid_from": "2023-01-01", "valid_to": "2023-01-02", "probe_queries": ["t"],
         "h2_probes": ["t1"]},
    ]
    cases = b.build(events)
    assert [c["query"] for c in cases] == ["new1", "new2"]
    assert {c["split"] for c in cases} == {"dev"}


def test_committed_h2_dev_cases_cover_13_dev_events():
    cases = json.loads((ROOT / "tests/eval/h2_dev_attribution_cases.json").read_text())
    events = {c["event_id"] for c in cases}
    assert len(events) == 13 and len(cases) == 26
    assert sum(1 for e in events if e.startswith("EVT-V2-")) == 6
    assert {c["split"] for c in cases} == {"dev"}


def _db(path, chunks):
    c = sqlite3.connect(path)
    c.executescript("""CREATE TABLE documents(doc_id TEXT PRIMARY KEY, source_path TEXT, created_at TEXT);
    CREATE TABLE chunks(chunk_id TEXT PRIMARY KEY, doc_id TEXT, stable_text TEXT, created_at TEXT);""")
    for i, (cid, doc, text, ts) in enumerate(chunks):
        c.execute("INSERT OR IGNORE INTO documents VALUES (?,?,?)", (doc, f"/c/{doc}.txt", ts))
        c.execute("INSERT INTO chunks VALUES (?,?,?,?)", (cid, doc, text, ts))
    c.commit(); c.close()


def test_tb5_content_hash_ignores_timestamps_and_insert_order(tmp_path):
    a, b = tmp_path / "a.db", tmp_path / "b.db"
    _db(a, [("c1", "d1", "alpha", "2026-01-01"), ("c2", "d2", "beta", "2026-01-01")])
    _db(b, [("c2", "d2", "beta", "2030-05-05"), ("c1", "d1", "alpha", "2030-05-05")])
    assert tb5_content_hash(a)["content_sha256"] == tb5_content_hash(b)["content_sha256"]


def test_tb5_content_hash_changes_with_text_and_model(tmp_path):
    a, b = tmp_path / "a.db", tmp_path / "b.db"
    _db(a, [("c1", "d1", "alpha", "t")])
    _db(b, [("c1", "d1", "alpha!", "t")])
    assert tb5_content_hash(a)["content_sha256"] != tb5_content_hash(b)["content_sha256"]
    assert tb5_content_hash(a)["content_sha256"] != tb5_content_hash(a, "other-model")["content_sha256"]


def test_tb5_content_hash_is_read_only(tmp_path):
    a = tmp_path / "a.db"
    _db(a, [("c1", "d1", "alpha", "t")])
    before = a.read_bytes()
    tb5_content_hash(a)
    assert a.read_bytes() == before


def test_data_version_without_db_and_in_snapshot(tmp_path):
    v = data_version(ROOT, None)
    assert v["tb5_db"] is None
    assert v["sales_data_csv_sha256"] == hashlib.sha256((ROOT / "data/demo/sales_data.csv").read_bytes()).hexdigest()
    import run_eval
    assert "data_version" in run_eval._settings_snapshot()
