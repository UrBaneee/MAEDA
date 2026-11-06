#!/usr/bin/env python3
"""
Receive the clean sales table that H1 ground truth is computed on (附录 GO.6,
GQ, GR.3). It is the v2 render with the quality noise (3% missing revenue, 2%
duplicate rows, 1% revenue x10) switched off and every event kept. It is a
TRUTH BASIS ONLY: the pipeline input stays data/demo/sales_data.csv
(b863b601...), and this file lives outside data/demo/ so nothing treats it as
an input. Verifies sha256 and row count, writes nothing on a mismatch. No API
call.

    python scripts/receive_clean_truth.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

CLEAN_SHA256 = "6c95334422ccfefd4a3a4815944418dd995c3bcdd9513b3aa4db0be76f7d771d"
CLEAN_ROWS = 48_000
SOURCE = Path.home() / "rag-framework" / "artifacts" / "g3" / "sales_data_clean.csv"
DEST = Path(__file__).resolve().parent.parent / "data" / "truth" / "sales_data_clean.csv"


def receive(source: Path = SOURCE, dest: Path = DEST, sha256: str = CLEAN_SHA256,
            rows: int = CLEAN_ROWS) -> str:
    raw = Path(source).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != sha256:
        raise RuntimeError(f"{source} sha256 {actual} != expected {sha256}")
    n = raw.count(b"\n") - 1  # header
    if n != rows:
        raise RuntimeError(f"{source} has {n} data rows, expected {rows}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)
    return actual


if __name__ == "__main__":
    print(f"received clean truth table {receive()[:12]}... -> {DEST}")
    sys.exit(0)
