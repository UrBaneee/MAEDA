"""
Golden-hash guard over data/demo/* (附录 BC.1 / BF).

Nothing before this test would have caught the original problem: TB5
changed generate_sales()'s internals, which silently changed
churn_data.csv/marketing_campaigns.csv/ecommerce_orders.db too (shared
global random stream -- see scripts/generate_demo_data.py's module-level
comment), invalidating 109 of tests/eval/test_suite.json's ground_truth
fixtures with no visible symptom: row counts, column counts, and file
sizes all stayed the same (see 附录 BC.1's ecommerce_orders.db evidence).
Nothing short of a real cross-process eval run surfaced it, months later.

This test pins the CURRENT (post-52fd014-fixture-correction) content of
all four demo files. If it ever fails, that means data/demo/* changed --
whoever re-ran scripts/generate_demo_data.py (deliberately or by
running it as a side effect of something else) must, before updating
the hash below:
  1. Recompute tests/eval/test_suite.json's ground_truth fixtures
     against the new files (see 52fd014's derivation approach --
     pandas/sqlite3 directly over the data, never from a model report).
  2. Re-run rag-framework/eval/fixtures/event_script/verify.py
     --data-path against the new sales_data.csv (the TB5 9/9
     detectability gate) if sales_data.csv specifically changed.
  3. Re-check the 6 known false-premise cases (DG07/DG08/DG11/DG13/
     DG15/DG18 -- 附录 BB.1/BC.4) against the new ecommerce_orders.db,
     since all six are ecommerce-derived.

The three CSVs are hashed as raw file bytes (deterministic byte-for-byte
across identical-content renders -- confirmed live, 附录 BF). The
SQLite database is NOT hashed as raw file bytes: identical row content
can still produce different file bytes (page layout / internal
metadata, not logical content) -- confirmed live, 附录 BF. It's hashed
as a canonical JSON serialization of each table's rows instead, in a
stable row order.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

_DEMO_DIR = Path(__file__).resolve().parents[2] / "data" / "demo"

# 附录 BF: captured 2026-08-17 against the current data/demo/* (matching
# 52fd014's corrected fixtures). Update ONLY after completing the three
# steps in this file's own docstring above -- not just to make this test
# pass.
_EXPECTED_CSV_SHA256 = {
    "sales_data.csv": "a0da755ec4c046de058280e4642f950697963baa8ec2f8b27d36f90021815ed2",
    "churn_data.csv": "230efca61993bc7ab2cec21d8b641b6336448c025bf096bc88145070b2231801",
    "marketing_campaigns.csv": "1cabbb4593b92f62f9171e88e6394c92509cb17f1b5313477286eb62fa169617",
}
_EXPECTED_ECOMMERCE_DB_CANONICAL_SHA256 = (
    "d10c11879913a5a02b7a9b8dc6615fb16363eb72d686280bc6acf39283bae7ac"
)


def _canonical_ecommerce_hash(db_path: Path) -> str:
    conn = sqlite3.connect(str(db_path))
    try:
        c = conn.cursor()
        parts = []
        for table in ["customers", "products", "orders"]:
            rows = c.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
            parts.append(f"{table}:" + json.dumps(rows, default=str, separators=(",", ":")))
        canonical = "|".join(parts).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()
    finally:
        conn.close()


def _require_demo_dir():
    if not _DEMO_DIR.is_dir():
        pytest.skip(f"{_DEMO_DIR} not present -- nothing to check")


@pytest.mark.parametrize("filename", sorted(_EXPECTED_CSV_SHA256))
def test_demo_csv_unchanged(filename):
    _require_demo_dir()
    path = _DEMO_DIR / filename
    if not path.is_file():
        pytest.skip(f"{path} not present -- nothing to check")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == _EXPECTED_CSV_SHA256[filename], (
        f"{filename} content changed (sha256 {actual} != expected "
        f"{_EXPECTED_CSV_SHA256[filename]}) -- see this file's module "
        f"docstring for what must happen before updating the expected hash."
    )


# ─── Regression test for the guard itself: does it actually fail on drift? ────

def test_guard_actually_fails_on_a_csv_that_changed(tmp_path, monkeypatch):
    """Not just 'the hashes would differ' -- drives the real test function
    against deliberately-different content under the expected filename and
    confirms IT actually raises. A guard that always passes regardless of
    content would not be caught by a weaker check."""
    import tests.unit.test_demo_data_integrity as this_module
    monkeypatch.setattr(this_module, "_DEMO_DIR", tmp_path)
    (tmp_path / "sales_data.csv").write_text("region,revenue\nNorth,999\n")
    with pytest.raises(AssertionError, match="content changed"):
        test_demo_csv_unchanged("sales_data.csv")


def test_guard_actually_fails_on_ecommerce_db_that_changed(tmp_path, monkeypatch):
    """Same as above for the SQLite content hash -- real schema, different
    row content, drives the real test function and confirms it raises."""
    import tests.unit.test_demo_data_integrity as this_module
    monkeypatch.setattr(this_module, "_DEMO_DIR", tmp_path)
    conn = sqlite3.connect(str(tmp_path / "ecommerce_orders.db"))
    conn.execute("CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("CREATE TABLE products (product_id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("CREATE TABLE orders (order_id INTEGER PRIMARY KEY, revenue REAL)")
    conn.execute("INSERT INTO customers VALUES (1, 'Someone Different')")
    conn.execute("INSERT INTO products VALUES (1, 'Different Product')")
    conn.execute("INSERT INTO orders VALUES (1, 12345.0)")
    conn.commit()
    conn.close()
    with pytest.raises(AssertionError, match="table content changed"):
        test_demo_ecommerce_db_content_unchanged()


def test_demo_ecommerce_db_content_unchanged():
    _require_demo_dir()
    path = _DEMO_DIR / "ecommerce_orders.db"
    if not path.is_file():
        pytest.skip(f"{path} not present -- nothing to check")
    actual = _canonical_ecommerce_hash(path)
    assert actual == _EXPECTED_ECOMMERCE_DB_CANONICAL_SHA256, (
        f"ecommerce_orders.db table content changed (canonical sha256 "
        f"{actual} != expected {_EXPECTED_ECOMMERCE_DB_CANONICAL_SHA256}) "
        f"-- see this file's module docstring for what must happen before "
        f"updating the expected hash. Note: this hashes TABLE CONTENT, not "
        f"the raw file bytes (those aren't stable across identical-content "
        f"renders -- 附录 BF), so a genuine content change is what tripped "
        f"this, not SQLite file-layout noise."
    )
