"""
Typed dataclasses for all MCP sub-system responses.
These form the type-safe boundary between MAEDA and external MCP servers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# ─── Data Cleaner types ───────────────────────────────────────────────────────

@dataclass
class ColumnProfile:
    name: str
    dtype: str
    null_pct: float
    unique_count: int
    sample_values: list[Any] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "ColumnProfile":
        return cls(
            name=d.get("name", ""),
            dtype=d.get("dtype", "unknown"),
            null_pct=float(d.get("null_pct", 0.0)),
            unique_count=int(d.get("unique_count", 0)),
            sample_values=d.get("sample_values") or [],
        )


@dataclass
class QualityIssue:
    """
    Normalized shape for one quality issue, regardless of where it came
    from. The real Data Cleaner (agentic-data-cleaner-v2 mcp_app.py
    profile_dataset) returns `quality_issues` as a flat list of issue-code
    strings (e.g. "schema_inconsistency"), not the list-of-dicts shape this
    boundary used to assume — that mismatch is what made the old
    `i.get("severity")` critical-issue derivation crash. The pandas
    fallback profiler (fallback.py) produces richer dicts with column/
    severity/detail. Both normalize into this one shape so the Insight
    Agent and eval renderer (src/agents/insight_agent.py,
    src/eval/metrics.py) have exactly one format to read, tagged with
    `source` so a caveat can say whether a finding came from the real
    cleaner or the degraded local profiler.
    """

    code: str
    severity: Optional[str] = None
    column: Optional[str] = None
    detail: Optional[str] = None
    source: str = "cleaner"  # "cleaner" | "fallback"

    @classmethod
    def from_cleaner_string(cls, code: str) -> "QualityIssue":
        return cls(code=code, source="cleaner")

    @classmethod
    def from_dict(cls, d: dict, source: str = "fallback") -> "QualityIssue":
        return cls(
            code=d.get("code") or d.get("issue") or "unknown",
            severity=d.get("severity"),
            column=d.get("column"),
            detail=d.get("detail"),
            source=d.get("source", source),
        )

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity,
            "column": self.column,
            "detail": self.detail,
            "source": self.source,
        }


def _normalize_quality_issues(raw_issues: list, source: str) -> list[QualityIssue]:
    """Accepts either the cleaner's `list[str]` shape or the fallback
    profiler's `list[dict]` shape and normalizes both to QualityIssue."""
    issues = []
    for i in raw_issues:
        if isinstance(i, str):
            issues.append(QualityIssue.from_cleaner_string(i))
        elif isinstance(i, dict):
            issues.append(QualityIssue.from_dict(i, source=source))
        # Anything else is a contract we don't understand yet; drop it
        # rather than guess — silently fabricating a code would be worse.
    return issues


@dataclass
class DataQualityReport:
    row_count: int
    columns: list[ColumnProfile]
    quality_issues: list[QualityIssue]
    has_critical_issues: bool = False
    quality_contract_version: Optional[str] = None
    # TB0.5 附录 B.3: no cleaner tool emits this today (verified against
    # mcp_app.py) -- forward-compat plumbing so M7's cleaning loop already
    # honors it the moment C4/TB4 adds it, rather than needing another pass.
    needs_review: bool = False
    # 附录 U.4.3/U.3 (quality_contract_version "2"): which scope(s)/
    # dimension(s) fired, and a hash of the intent's column_scope.columns
    # this response was computed against. None when the server predates
    # TB3+TB4 (v1-only) or when no intent was supplied for this call --
    # both cases mean "nothing to compare/disclose", not an error.
    critical_basis: Optional[dict] = None
    scope_fingerprint: Optional[str] = None

    @classmethod
    def from_mcp_response(cls, data: dict) -> "DataQualityReport":
        columns = [ColumnProfile.from_dict(c) for c in data.get("columns", [])]
        issues = _normalize_quality_issues(data.get("quality_issues") or [], source="cleaner")
        return cls(
            row_count=int(data.get("row_count", 0)),
            columns=columns,
            quality_issues=issues,
            # Read the cleaner's own verdict (附录 B.1, live since C1) rather
            # than re-deriving it from quality_issues — deriving it from a
            # `severity` key that string-shaped issues don't have is exactly
            # what used to crash this boundary.
            has_critical_issues=bool(data.get("has_critical_issues", False)),
            quality_contract_version=data.get("quality_contract_version"),
            needs_review=bool(data.get("needs_review", False)),
            critical_basis=data.get("critical_basis"),
            scope_fingerprint=data.get("scope_fingerprint"),
        )

    def to_dict(self) -> dict:
        return {
            "row_count": self.row_count,
            "columns": [
                {
                    "name": c.name,
                    "dtype": c.dtype,
                    "null_pct": c.null_pct,
                    "unique_count": c.unique_count,
                    "sample_values": c.sample_values,
                }
                for c in self.columns
            ],
            "quality_issues": [i.to_dict() for i in self.quality_issues],
            "has_critical_issues": self.has_critical_issues,
            "quality_contract_version": self.quality_contract_version,
            "needs_review": self.needs_review,
            "critical_basis": self.critical_basis,
            "scope_fingerprint": self.scope_fingerprint,
        }


@dataclass
class CleaningStep:
    operation: str
    target_column: str
    rationale: str
    estimated_impact: dict

    @classmethod
    def from_dict(cls, d: dict) -> "CleaningStep":
        return cls(
            operation=d.get("operation", ""),
            target_column=d.get("target_column", ""),
            rationale=d.get("rationale", ""),
            # Real shape (agentic-data-cleaner-v2 mcp_app.py get_cleaning_plan):
            # {"confidence": float, "risk_level": str, "produces_diff": bool},
            # not a string — verified against a live response (M3).
            estimated_impact=d.get("estimated_impact") or {},
        )


@dataclass
class CleaningPlan:
    steps: list[CleaningStep]

    @classmethod
    def from_mcp_response(cls, data: dict) -> "CleaningPlan":
        steps = [CleaningStep.from_dict(s) for s in data.get("steps", [])]
        return cls(steps=steps)

    def to_dict(self) -> dict:
        return {
            "steps": [
                {
                    "operation": s.operation,
                    "target_column": s.target_column,
                    "rationale": s.rationale,
                    "estimated_impact": s.estimated_impact,
                }
                for s in self.steps
            ]
        }


@dataclass
class CleaningResult:
    cleaned_path: str
    changes_summary: dict
    rows_affected: int
    needs_review: bool = False  # 附录 B.3 — not emitted today, see DataQualityReport.needs_review
    # M4 / 附录 D-I C3: {plan_id, planner_mode_requested, planner_mode_used,
    # planner_fallback_reason, steps[]} — #17's target-state field, separate
    # from changes_summary (results) and executed_steps (outcomes). Live
    # since cleaner's C3 (2026-08-15); empty dict if the server predates it.
    execution_plan: dict = field(default_factory=dict)
    # M8 / 定案 #16: cleaner now self-computes these (C3) from the same
    # ingested-CSV source for both input and output, so comparing them
    # directly is an apples-to-apples check MAEDA doesn't have to redo by
    # hashing the raw (possibly non-CSV) input itself. None if the server
    # predates C3 — callers must fall back to self-hashing, not assume "no
    # diff" from an absent field.
    input_sha256: Optional[str] = None
    output_sha256: Optional[str] = None
    output_bytes: Optional[int] = None
    # 附录 U.3: hash of the intent's column_scope.columns this clean_dataset
    # call was run against — must match the fingerprint from this round's
    # profile_dataset/validate_quality calls (including re-profile). None
    # when the server predates TB3+TB4 or no intent was supplied.
    scope_fingerprint: Optional[str] = None

    @classmethod
    def from_mcp_response(cls, data: dict) -> "CleaningResult":
        return cls(
            cleaned_path=data.get("cleaned_path", ""),
            # Real shape (agentic-data-cleaner-v2 mcp_app.py clean_dataset):
            # {"overall_score", "metrics", "data_versions", "diff_paths",
            # "total_rounds", "plan_steps" (an int count, not a step list —
            # 附录 D F3/#17), "needs_review_count"}. Not a string — verified
            # against a live response (M3). clean_dataset's response does
            # NOT carry quality_contract_version (only profile_dataset and
            # validate_quality do, per 附录 B.6) — nothing to read here.
            changes_summary=data.get("changes_summary") or {},
            rows_affected=int(data.get("rows_affected", 0)),
            needs_review=bool(data.get("needs_review", False)),
            execution_plan=data.get("execution_plan") or {},
            input_sha256=data.get("input_sha256"),
            output_sha256=data.get("output_sha256"),
            output_bytes=data.get("output_bytes"),
            scope_fingerprint=data.get("scope_fingerprint"),
        )


@dataclass
class QualityValidation:
    passed: bool
    score: float
    issues: list[str]
    quality_contract_version: Optional[str] = None
    # Real shape (agentic-data-cleaner-v2 mcp_app.py validate_quality):
    # {missing_score, duplicate_score, outlier_score, schema_score,
    # mean_null_ratio, duplicate_row_ratio, outlier_row_ratio}. M7 reads
    # mean_null_ratio/duplicate_row_ratio/schema_score to reconstruct the
    # same three-dimension verdict has_critical_issues is built from
    # (附录 B.0/B.1), so the cleaning loop can tell "no improvement between
    # rounds" apart from "still bad but for a different reason" — the
    # combined passed/score booleans alone can't make that distinction.
    details: dict = field(default_factory=dict)
    needs_review: bool = False  # 附录 B.3 — not emitted today, see DataQualityReport.needs_review
    # 附录 U.4.3/U.3 — same shape/meaning as DataQualityReport's fields;
    # `passed` is the dual-scope complement of `has_critical_issues` on the
    # current file (附录 U.4.1), critical_basis explains which scope(s)/
    # dimension(s) still fire.
    critical_basis: Optional[dict] = None
    scope_fingerprint: Optional[str] = None

    @classmethod
    def from_mcp_response(cls, data: dict) -> "QualityValidation":
        return cls(
            passed=bool(data.get("passed", True)),
            score=float(data.get("score", 1.0)),
            # Real shape (agentic-data-cleaner-v2 mcp_app.py validate_quality):
            # a flat list[str] of human-readable sentences, not list[dict] —
            # verified against a live response (M3).
            issues=data.get("issues") or [],
            quality_contract_version=data.get("quality_contract_version"),
            details=data.get("details") or {},
            needs_review=bool(data.get("needs_review", False)),
            critical_basis=data.get("critical_basis"),
            scope_fingerprint=data.get("scope_fingerprint"),
        )


# ─── RAG Server types ─────────────────────────────────────────────────────────

@dataclass
class RAGChunk:
    content: str
    score: float
    source_file: Optional[str] = None
    page: Optional[int] = None
    chunk_id: Optional[str] = None
    metadata: Optional[dict] = None

    @classmethod
    def from_mcp_response(cls, d: dict) -> "RAGChunk":
        return cls(
            content=d.get("content", ""),
            score=float(d.get("score", 0.0)),
            source_file=d.get("source_file"),
            page=d.get("page"),
            chunk_id=d.get("chunk_id"),
            metadata=d.get("metadata"),
        )

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "score": self.score,
            "source_file": self.source_file,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "metadata": self.metadata,
        }


@dataclass
class Collection:
    name: str
    doc_count: int
    description: str

    @classmethod
    def from_mcp_response(cls, d: dict) -> "Collection":
        return cls(
            name=d.get("name", ""),
            doc_count=int(d.get("doc_count", 0)),
            description=d.get("description", ""),
        )


# ─── Health status ────────────────────────────────────────────────────────────

@dataclass
class SubSystemHealth:
    data_cleaner_available: bool
    rag_server_available: bool
    data_cleaner_latency_ms: Optional[float] = None
    rag_server_latency_ms: Optional[float] = None

    @property
    def any_available(self) -> bool:
        return self.data_cleaner_available or self.rag_server_available

    def to_dict(self) -> dict:
        return {
            "data_cleaner_available": self.data_cleaner_available,
            "rag_server_available": self.rag_server_available,
            "data_cleaner_latency_ms": self.data_cleaner_latency_ms,
            "rag_server_latency_ms": self.rag_server_latency_ms,
        }
