"""
Business glossary (口径词表) — ECOSYSTEM_INTEGRATION_PLAN.md 阶段 3, 执行顺序表
轮次 4 (附录 CQ).

Curation-layer content: a YAML file of column definitions (config/glossary.yaml)
that is reconciled against the *live* schema before anything is injected, so a
definition for a column that does not exist in the data at hand is dropped
rather than handed to an LLM as fact. Deterministic — this layer never goes
through RAG retrieval (rag doc 5.1 / docs/handoff_maeda_to_subsystems.md:185).

THE SINGLE GATE
    `resolve_glossary()` is the only place coverage is judged and entries are
    filtered. It runs once per round in `connect_schema` (src/graph/nodes.py),
    the moment a schema exists, and its result is written to state. Both
    injection points and the alias tier read that stored result — same pattern
    as `cleaner_should_attempt_clean` / `rag_retrieval_decision`
    (src/graph/router.py, 附录 CC.2/CM.4): one judgment site, several
    projections, so the two injection points can never disagree about what the
    glossary says (附录 P.2's "过滤一次、两处投影").

    [2] `_build_intent_payload` → the TB3 intent payload's `glossary`
        (附录 U.2's frozen entry shape) + `glossary_coverage`
    [3] `AnalysisAgent.plan` → the plan_analysis prompt block
    (+) `_resolve_intent_columns` → the 附录 U.2 `glossary_alias` match tier,
        dead until now because the glossary had no content (附录 S.2)

THE THREE-STATE COVERAGE, AND WHY `absent` IS LOUD
    `full` / `partial` / `absent` describe how much of the *live schema* the
    glossary covers, which is also a drift check: a glossary written against
    an older version of a dataset shows up as `partial` (new columns) plus a
    non-empty `dropped_columns` (columns the glossary still describes but the
    data no longer has).

    `absent` is emitted explicitly everywhere — into state, into the decision
    trace, into the cleaner payload as `glossary_coverage`, and as a named
    instruction block in the planner prompt. It is never expressed by leaving
    the glossary out, because "no curated definition exists" and "the glossary
    was silently not injected" look identical downstream — the exact failure
    shape this project has hit repeatedly (附录 CH.2/CI.2/CK.3/CO). An unknown
    data source must say it is unknown; that is what stops a planner from
    inventing units, currencies and value semantics that nobody wrote down.

DATASET MATCHING
    By file basename, plus `table` for SQL sources. Deliberately blunt: a file
    with the same basename at a different path matches (same dataset, moved),
    a renamed copy does not (reports `absent`, the safe direction).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger("maeda.glossary")

COVERAGE_FULL = "full"
COVERAGE_PARTIAL = "partial"
COVERAGE_ABSENT = "absent"

# 附录 U.2's frozen glossary entry keys — the only ones that may cross to the
# cleaner. `aliases` is MAEDA-internal (the alias match tier) and is projected
# out at injection [2]: 附录 P.2 kept the entry shape to fields cleaner can act
# on, and adding a field cleaner cannot use would only invite it to be read as
# contract surface.
_PAYLOAD_KEYS = ("column", "meaning", "unit", "value_semantics", "notes")

_ANTI_FABRICATION_ABSENT = (
    "No curated business definition exists for ANY column of this data source. "
    "Do not invent meanings, units, currencies, value encodings or business "
    "rules for these columns, and do not assume a column means what its name "
    "suggests. Treat the column names as opaque labels: describe what the data "
    "shows, and state any assumption you have to make in the step's rationale "
    "instead of asserting it as an established definition."
)
_ANTI_FABRICATION_PARTIAL = (
    "The columns listed under \"No curated definition\" above have no written "
    "business definition. Do not invent meanings, units, currencies or value "
    "encodings for them; if a step depends on such a definition, say so in that "
    "step's rationale rather than asserting one."
)


@dataclass(frozen=True)
class GlossaryMatch:
    """Result of the single gate. `entries` are curation-layer form (they still
    carry `aliases`); use `payload_entries()` for the 附录 U.2 wire shape."""

    coverage: str                      # "full" | "partial" | "absent"
    entries: list[dict] = field(default_factory=list)
    covered_columns: list[str] = field(default_factory=list)
    uncovered_columns: list[str] = field(default_factory=list)
    # Glossary columns that the live schema does not have — dropped, never
    # injected (轮次 4: 词表条目要与真实列名对账后才注入). Non-empty means the
    # glossary and the data have drifted apart.
    dropped_columns: list[str] = field(default_factory=list)
    dataset_key: Optional[str] = None  # "sales_data.csv" / "orders.db::orders"
    reason: Optional[str] = None       # why coverage is absent, when it is


# ─── Loading ──────────────────────────────────────────────────────────────────

_cache: dict[str, tuple[tuple, dict]] = {}


def _resolved_path(path: Optional[str] = None) -> str:
    return str(Path(path or settings.glossary_path).expanduser())


def load_glossary(path: Optional[str] = None) -> dict:
    """
    Read and cache the glossary YAML. Cached on (mtime, size) so a test or an
    edit that rewrites the file is picked up without a process restart.

    A missing or malformed file is a warning, not an exception: it degrades to
    an empty glossary, which surfaces as coverage `absent` — loud downstream
    (see the module docstring), not a silently missing prompt section, and
    never a reason for the pipeline itself to fail.
    """
    resolved = _resolved_path(path)
    try:
        stat = os.stat(resolved)
        stamp = (stat.st_mtime_ns, stat.st_size)
    except OSError as exc:
        logger.warning("Glossary file unavailable (%s): %s — coverage will be 'absent'", resolved, exc)
        return {}

    cached = _cache.get(resolved)
    if cached is not None and cached[0] == stamp:
        return cached[1]

    try:
        with open(resolved, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except (yaml.YAMLError, OSError) as exc:
        logger.error("Glossary file could not be parsed (%s): %s — coverage will be 'absent'", resolved, exc)
        return {}

    if not isinstance(data, dict) or not isinstance(data.get("datasets"), list):
        logger.error("Glossary file %s has no `datasets:` list — coverage will be 'absent'", resolved)
        data = {}

    _cache[resolved] = (stamp, data)
    return data


# ─── The single gate ──────────────────────────────────────────────────────────

def _column_name(col: Any) -> str:
    """schema columns arrive as ColumnInfo objects (src/tools/data_connector.py);
    dicts and bare strings are accepted so callers/tests aren't forced to build
    one just to ask a coverage question."""
    name = getattr(col, "name", None)
    if name is None and isinstance(col, dict):
        name = col.get("name")
    return str(name if name is not None else col)


def _select_block(blocks: list, dataset_path: str, table_name: Optional[str]) -> Optional[dict]:
    basename = Path(dataset_path or "").name
    if not basename:
        return None
    candidates = [
        b for b in blocks
        if isinstance(b, dict) and b.get("dataset") == basename and isinstance(b.get("columns"), list)
    ]
    # A block that names a table only applies to that table. A block with no
    # `table` applies to the file as a whole (the CSV case). Never fall back
    # from one table's definitions to another's: `customers` must not inherit
    # `orders`'s 口径 just because they share a database file.
    for block in candidates:
        if block.get("table") and table_name and block["table"] == table_name:
            return block
    for block in candidates:
        if not block.get("table") and not table_name:
            return block
    return None


def resolve_glossary(
    dataset_path: str,
    schema_columns: Optional[list],
    table_name: Optional[str] = None,
    path: Optional[str] = None,
) -> GlossaryMatch:
    """
    THE gate (see module docstring). Filters the glossary down to entries whose
    `column` exists in this run's real schema and reports the three-state
    coverage. Never raises — every failure mode resolves to `absent` with a
    `reason`, because a caller that has to guard this call is a caller that
    will eventually forget to.
    """
    columns = [_column_name(c) for c in (schema_columns or [])]
    if not columns:
        return GlossaryMatch(coverage=COVERAGE_ABSENT, reason="no schema columns available")

    data = load_glossary(path)
    blocks = data.get("datasets") or []
    block = _select_block(blocks, dataset_path, table_name)
    if block is None:
        target = Path(dataset_path or "").name or "<unknown>"
        if table_name:
            target = f"{target}::{table_name}"
        return GlossaryMatch(
            coverage=COVERAGE_ABSENT,
            uncovered_columns=columns,
            reason=f"no glossary entry for data source {target}",
        )

    by_column: dict[str, dict] = {}
    dropped: list[str] = []
    for raw in block["columns"]:
        if not isinstance(raw, dict) or not raw.get("column"):
            continue
        name = str(raw["column"])
        if name not in columns:
            dropped.append(name)          # schema drift — described but not present
            continue
        if name in by_column:
            logger.warning("Glossary has duplicate entries for %s in %s; keeping the first",
                           name, block.get("dataset"))
            continue
        by_column[name] = raw

    covered = [c for c in columns if c in by_column]
    uncovered = [c for c in columns if c not in by_column]
    dataset_key = block.get("dataset")
    if block.get("table"):
        dataset_key = f"{dataset_key}::{block['table']}"

    if not covered:
        return GlossaryMatch(
            coverage=COVERAGE_ABSENT,
            uncovered_columns=uncovered,
            dropped_columns=sorted(dropped),
            dataset_key=dataset_key,
            reason=(
                f"glossary block {dataset_key!r} describes {len(dropped)} column(s), "
                "none of which exist in this schema"
            ),
        )

    return GlossaryMatch(
        coverage=COVERAGE_FULL if not uncovered else COVERAGE_PARTIAL,
        entries=[by_column[c] for c in covered],
        covered_columns=covered,
        uncovered_columns=uncovered,
        dropped_columns=sorted(dropped),
        dataset_key=dataset_key,
    )


# ─── Projections ──────────────────────────────────────────────────────────────

def payload_entries(entries: Optional[list[dict]]) -> list[dict]:
    """
    Injection [2]: project filtered entries onto 附录 U.2's frozen glossary
    entry shape. Optional keys are omitted when empty rather than sent as
    null/[] — U.2 makes every field except `column` optional, and an entry
    padded with empty fields reads to the cleaner's LLM planner (which dumps
    the whole intent into its prompt — agent/llm_planner.py:351) as "this was
    considered and is empty" rather than "this was never written".
    """
    projected: list[dict] = []
    for entry in entries or []:
        item = {"column": str(entry.get("column"))}
        for key in _PAYLOAD_KEYS[1:]:
            value = entry.get(key)
            if value in (None, "", [], {}):
                continue
            item[key] = value
        projected.append(item)
    return projected


def alias_index(entries: Optional[list[dict]]) -> dict[str, list[str]]:
    """
    Lower-cased alias → the schema column(s) it names, built only from entries
    that already survived the schema filter, so the 附录 U.2 `glossary_alias`
    tier can never resolve a mention to a column that isn't there.

    A list, not a single name: two entries claiming the same alias is a real
    glossary authoring mistake, and the resolver turns it into
    `unresolved_mentions[].reason = "ambiguous"` (附录 U.2's second, until now
    also unreachable, reason value) instead of silently picking one.
    """
    index: dict[str, list[str]] = {}
    for entry in entries or []:
        column = str(entry.get("column") or "")
        if not column:
            continue
        for alias in entry.get("aliases") or []:
            key = str(alias).strip().lower()
            if not key:
                continue
            names = index.setdefault(key, [])
            if column not in names:
                names.append(column)
    return index


def prompt_block(
    coverage: Optional[str],
    entries: Optional[list[dict]],
    uncovered_columns: Optional[list[str]] = None,
    source_label: Optional[str] = None,
) -> str:
    """
    Injection [3]: the plan_analysis prompt section. Returns a non-empty block
    in ALL THREE states — `absent` renders the loudest block of the three, not
    the shortest and never nothing (see the module docstring on why omission is
    the failure mode being designed against).
    """
    coverage = coverage or COVERAGE_ABSENT
    # A display label only (the data source as the user pointed at it) — the
    # gate's own matched-block key is not re-derived here.
    source = f" ({source_label})" if source_label else ""
    header = f"### Column Glossary (curated business definitions — coverage: {coverage})\n"

    if coverage == COVERAGE_ABSENT:
        return header + f"No curated glossary entry covers this data source{source}.\n" + _ANTI_FABRICATION_ABSENT + "\n"

    lines = []
    for entry in entries or []:
        parts = [f"- {entry.get('column')}: {_flat(entry.get('meaning')) or '(no meaning recorded)'}"]
        if entry.get("unit"):
            parts.append(f"  unit: {_flat(entry['unit'])}")
        for semantic in entry.get("value_semantics") or []:
            if not isinstance(semantic, dict):
                continue
            parts.append(
                f"  value {semantic.get('value')!r}: {_flat(semantic.get('means'))} "
                f"[action: {semantic.get('action', 'keep')}]"
            )
        if entry.get("notes"):
            parts.append(f"  note: {_flat(entry['notes'])}")
        lines.append("\n".join(parts))

    body = header + "\n".join(lines) + "\n"
    if coverage == COVERAGE_PARTIAL:
        body += (
            "\nNo curated definition: "
            + ", ".join(uncovered_columns or [])
            + "\n"
            + _ANTI_FABRICATION_PARTIAL
            + "\n"
        )
    else:
        body += (
            "\nEvery column of this data source has a curated definition above; "
            "use them as written rather than re-deriving what a column means.\n"
        )
    return body


def _flat(text: Any) -> str:
    """YAML block scalars keep trailing newlines; the prompt is line-oriented."""
    return " ".join(str(text or "").split())
