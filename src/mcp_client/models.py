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
class DataQualityReport:
    row_count: int
    columns: list[ColumnProfile]
    quality_issues: list[dict]
    has_critical_issues: bool = False

    @classmethod
    def from_mcp_response(cls, data: dict) -> "DataQualityReport":
        columns = [ColumnProfile.from_dict(c) for c in data.get("columns", [])]
        issues = data.get("quality_issues") or []
        critical = any(i.get("severity") == "critical" for i in issues)
        return cls(
            row_count=int(data.get("row_count", 0)),
            columns=columns,
            quality_issues=issues,
            has_critical_issues=critical,
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
            "quality_issues": self.quality_issues,
            "has_critical_issues": self.has_critical_issues,
        }


@dataclass
class CleaningStep:
    operation: str
    target_column: str
    rationale: str
    estimated_impact: str

    @classmethod
    def from_dict(cls, d: dict) -> "CleaningStep":
        return cls(
            operation=d.get("operation", ""),
            target_column=d.get("target_column", ""),
            rationale=d.get("rationale", ""),
            estimated_impact=d.get("estimated_impact", ""),
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
    changes_summary: str
    rows_affected: int

    @classmethod
    def from_mcp_response(cls, data: dict) -> "CleaningResult":
        return cls(
            cleaned_path=data.get("cleaned_path", ""),
            changes_summary=data.get("changes_summary", ""),
            rows_affected=int(data.get("rows_affected", 0)),
        )


@dataclass
class QualityValidation:
    passed: bool
    score: float
    issues: list[dict]

    @classmethod
    def from_mcp_response(cls, data: dict) -> "QualityValidation":
        return cls(
            passed=bool(data.get("passed", True)),
            score=float(data.get("score", 1.0)),
            issues=data.get("issues") or [],
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
