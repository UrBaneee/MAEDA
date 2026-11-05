"""
Data-version records for the EJ.2 freeze list (附录 FQ.2 item 3, GA.2, GB).

A σ run and the reveal are comparable only if they saw the same data, so a
saved report must say which: the event-script tag and manifest hashes, the
hash of the rendered sales_data.csv, and the content hash of the R-arm
retrieval database. Every function here is read-only and makes no API call.

`tb5_dev.db` is hashed by CONTENT, not by file bytes: the file carries
`created_at` timestamps, so two ingests of the same corpus differ byte-wise
(附录 FZ.6 item 5). The hash covers every chunk (the retrieval unit), ordered
by chunk id, as (document source_path, chunk_id, stable_text), plus the
embedding model name. The model name is not stored in the database; it is the
one recorded when the database was built (附录 FZ.3) and is passed in.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from pathlib import Path
from typing import Optional

#: 附录 FZ.3: tb5_dev.db was embedded with OpenAI text-embedding-3-small.
TB5_DEV_EMBEDDING_MODEL = "text-embedding-3-small"

RAG_REPO = Path.home() / "rag-framework"
EVENT_SCRIPT_TAG = "event-script-v2"
_MANIFEST_PATH = "eval/fixtures/event_script/event_script.manifest.json"


def tb5_content_hash(db_path: Path | str, embedding_model: str = TB5_DEV_EMBEDDING_MODEL) -> dict:
    """Content hash of a rag SQLite store. Opens the file read-only."""
    uri = f"file:{Path(db_path).resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        rows = conn.execute(
            "SELECT d.source_path, c.chunk_id, c.stable_text "
            "FROM chunks c JOIN documents d ON d.doc_id = c.doc_id "
            "ORDER BY c.chunk_id"
        ).fetchall()
    finally:
        conn.close()
    payload = json.dumps({"embedding_model": embedding_model, "chunks": rows},
                         ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return {
        "content_sha256": hashlib.sha256(payload).hexdigest(),
        "n_chunks": len(rows),
        "n_documents": len({r[0] for r in rows}),
        "embedding_model": embedding_model,
    }


def event_script_manifest(repo: Path = RAG_REPO, tag: str = EVENT_SCRIPT_TAG) -> Optional[dict]:
    """Public manifest hashes at the frozen tag, or None if unreadable."""
    done = subprocess.run(["git", "-C", str(repo), "show", f"{tag}:{_MANIFEST_PATH}"],
                          capture_output=True, text=True, timeout=10)
    if done.returncode != 0:
        return None
    m = json.loads(done.stdout)
    keys = ("public_sha256", "sealed_sha256", "combined_sha256", "slot_plan_sha256", "frozen_at")
    return {"git_tag": tag, "manifest_sha256": m.get("sha256"), **{k: m.get(k) for k in keys}}


def data_version(root: Path, rag_db_path: Optional[str]) -> dict:
    """The data-version block that run_eval writes into settings_snapshot."""
    csv = root / "data" / "demo" / "sales_data.csv"
    db = Path(rag_db_path) if rag_db_path else None
    return {
        "event_script": event_script_manifest(),
        "sales_data_csv_sha256": hashlib.sha256(csv.read_bytes()).hexdigest() if csv.is_file() else None,
        "tb5_db": tb5_content_hash(db) if db and db.is_file() else None,
    }
