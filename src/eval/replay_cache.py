"""
Replay cache — decouples "generate a report" (slow, real pipeline calls,
~60-125s/case) from "score a report" (fast, just the judge, ~1-5s/case).

Step 2 of docs/eval_v2_plan.md is entirely about changing scoring logic in
src/eval/ — it never needs a different report. Regenerating all 100+ cases
on every metric tweak (the only option before this module existed) was the
single biggest cost/time sink in iterating on the eval harness itself.

`generation_fingerprint()` hashes everything that can change what a cached
report *means* -- source under src/ except src/eval/ itself (deliberately
excluded: that's exactly the code Step 2 iterates on, and a change there
must NOT invalidate cached reports), the demo datasets, and the resolved
settings that affect generation (model/provider/temperature). It does NOT
hash the EVAL_-prefixed prompt constants inside src/config/agent_prompts.py
even though that file also holds generation prompts -- see
_read_prompts_file_excluding_eval below.

Usage:
    from src.eval.replay_cache import ReplayCache
    cache = ReplayCache()
    cache.put("D01", full_state_dict)          # after a real graph.ainvoke()
    entry = cache.get("D01")                   # None if missing or stale
    if entry is None:
        ...regenerate...
"""
from __future__ import annotations

import glob
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CACHE_PATH = _REPO_ROOT / "logs" / "replay_cache" / "corpus.json"

# Anything under src/ EXCEPT src/eval/ is "generation-relevant" -- changing
# it can change what a cached report contains, so it must invalidate the
# cache. src/eval/ is exactly what Step 2 iterates on and must NOT
# invalidate cached reports, or replay mode loses its entire point.
_GENERATION_SRC_ROOT = _REPO_ROOT / "src"
_EXCLUDED_SRC_SUBTREE = _REPO_ROOT / "src" / "eval"

_DATA_GLOBS = ["data/demo/*.csv", "data/demo/*.db", "data/sample/**/*"]


def _read_prompts_file_excluding_eval(path: Path) -> str:
    """src/config/agent_prompts.py holds both generation prompts (planner,
    insight, guardrail, ...) and EVAL_RELEVANCE_SYSTEM, the judge's prompt.
    Hashing the whole file would invalidate every cached report the moment
    Step 2 edits the judge prompt -- exactly the churn replay mode exists to
    avoid. Splits on top-level triple-quoted-string constant definitions and
    drops any whose name starts with EVAL_."""
    text = path.read_text()
    blocks = re.split(r"(?=^[A-Z_]+ = (?:\"\"\"|'''))", text, flags=re.MULTILINE)
    kept = [b for b in blocks if not re.match(r"^EVAL_[A-Z_]* = ", b)]
    return "".join(kept)


def _iter_generation_source_files():
    for p in sorted(_GENERATION_SRC_ROOT.rglob("*.py")):
        if _EXCLUDED_SRC_SUBTREE in p.parents:
            continue
        if "__pycache__" in p.parts:
            continue
        yield p


def _resolved_generation_settings() -> dict:
    """Only the settings fields that can change what generation produces.
    Deliberately excludes eval_llm_model/eval_judge_samples/
    resolved_eval_provider/resolved_eval_model -- those are Step 2's
    territory and must not force a regeneration either."""
    from src.config.settings import settings
    return {
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "llm_temperature": settings.llm_temperature,
        "resolved_planner_model": settings.resolved_planner_model,
        "resolved_guardrail_model": settings.resolved_guardrail_model,
        "max_tokens_per_call": settings.max_tokens_per_call,
    }


def generation_fingerprint() -> str:
    """A single hash that changes iff something able to change a generated
    report's content has changed: generation-relevant source, the demo
    datasets, or generation-relevant resolved settings."""
    h = hashlib.sha256()

    for p in _iter_generation_source_files():
        rel = p.relative_to(_REPO_ROOT)
        if rel == Path("src/config/agent_prompts.py"):
            content = _read_prompts_file_excluding_eval(p)
        else:
            content = p.read_text()
        h.update(str(rel).encode())
        h.update(content.encode())

    data_files = set()
    for pattern in _DATA_GLOBS:
        data_files.update(glob.glob(str(_REPO_ROOT / pattern), recursive=True))
    for f in sorted(data_files):
        fp = Path(f)
        if fp.is_file():
            h.update(fp.name.encode())
            h.update(hashlib.sha256(fp.read_bytes()).hexdigest().encode())

    h.update(json.dumps(_resolved_generation_settings(), sort_keys=True).encode())
    return h.hexdigest()[:16]


@dataclass
class CacheEntry:
    case_id: str
    state: dict
    fingerprint: str
    generated_at: float
    elapsed_s: float
    cost_usd: float

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id, "state": self.state,
            "fingerprint": self.fingerprint, "generated_at": self.generated_at,
            "elapsed_s": self.elapsed_s, "cost_usd": self.cost_usd,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CacheEntry":
        return cls(
            case_id=d["case_id"], state=d["state"], fingerprint=d["fingerprint"],
            generated_at=d["generated_at"], elapsed_s=d.get("elapsed_s", 0.0),
            cost_usd=d.get("cost_usd", 0.0),
        )


class ReplayCache:
    """
    File-backed cache of full MAEDAState dicts, keyed by golden-case id.

    A cached entry is only returned by `get()` if its stored fingerprint
    matches the CURRENT generation_fingerprint() -- a stale entry (source
    changed, data changed, model config changed) is treated as absent
    rather than silently served, so replay mode can never score a report
    that doesn't reflect the pipeline as it exists right now.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = path or _DEFAULT_CACHE_PATH
        self._entries: dict[str, CacheEntry] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text())
        for cid, d in raw.items():
            self._entries[cid] = CacheEntry.from_dict(d)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            {cid: e.to_dict() for cid, e in self._entries.items()},
            indent=2, default=str,
        ))

    def get(self, case_id: str, current_fingerprint: Optional[str] = None) -> Optional[CacheEntry]:
        entry = self._entries.get(case_id)
        if entry is None:
            return None
        fp = current_fingerprint if current_fingerprint is not None else generation_fingerprint()
        if entry.fingerprint != fp:
            return None
        return entry

    def put(self, case_id: str, state: dict, elapsed_s: float = 0.0, cost_usd: float = 0.0,
            fingerprint: Optional[str] = None) -> None:
        self._entries[case_id] = CacheEntry(
            case_id=case_id, state=state,
            fingerprint=fingerprint if fingerprint is not None else generation_fingerprint(),
            generated_at=time.time(), elapsed_s=elapsed_s, cost_usd=cost_usd,
        )

    def stale_or_missing(self, case_ids: list[str]) -> list[str]:
        fp = generation_fingerprint()
        return [cid for cid in case_ids if self.get(cid, fp) is None]

    def __len__(self) -> int:
        return len(self._entries)
