"""
Unified data connector for MAEDA.

Supports CSV/TSV, SQLite/PostgreSQL, JSON/JSONL, and Excel.
Responsibilities:
  1. Load data into a pandas DataFrame
  2. Extract typed schema (column names, dtypes, null rates, samples)
  3. Generate a natural-language schema summary (LLM or rule-based fallback)

Data quality profiling and cleaning are DELEGATED to the Data Cleaner MCP
(see src/mcp_client/fallback.py). This module only handles connection + schema.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.config.agent_prompts import SCHEMA_SUMMARIZER_SYSTEM
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger("maeda.tools.data_connector")

_PREVIEW_ROWS = 5


# ─── Schema types ─────────────────────────────────────────────────────────────

@dataclass
class ColumnInfo:
    name: str
    dtype: str
    null_pct: float
    unique_count: int
    sample_values: list[Any] = field(default_factory=list)
    is_numeric: bool = False
    is_datetime: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "null_pct": round(self.null_pct, 4),
            "unique_count": self.unique_count,
            "sample_values": [str(v) for v in self.sample_values],
            "is_numeric": self.is_numeric,
            "is_datetime": self.is_datetime,
        }


@dataclass
class SchemaInfo:
    source_type: str           # csv | sql | json | excel
    path: str
    table_name: Optional[str]  # SQL table name or sheet name
    row_count: int
    columns: list[ColumnInfo]
    preview: list[dict]        # first N rows as records

    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def numeric_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.is_numeric]

    def to_dict(self) -> dict:
        return {
            "source_type": self.source_type,
            "path": self.path,
            "table_name": self.table_name,
            "row_count": self.row_count,
            "columns": [c.to_dict() for c in self.columns],
            "preview": self.preview,
        }

    def to_source_dict(self) -> dict:
        """Format for state["data_sources"] / state["active_source"]."""
        return {
            "type": self.source_type,
            "path": self.path,
            "table_name": self.table_name,
            "schema": self.to_dict(),
            "preview": self.preview,
        }

    def rule_based_summary(self) -> str:
        """
        Deterministic NL summary used when LLM is unavailable.
        Good enough for fallback; LLM version is richer.
        """
        col_list = ", ".join(self.column_names()[:8])
        suffix = f" and {len(self.columns) - 8} more" if len(self.columns) > 8 else ""
        numeric = self.numeric_columns()
        num_str = (
            f" Numeric columns: {', '.join(numeric[:5])}."
            if numeric
            else ""
        )
        table_str = f" (table: {self.table_name})" if self.table_name else ""
        return (
            f"Dataset{table_str} with {self.row_count:,} rows and "
            f"{len(self.columns)} columns: {col_list}{suffix}.{num_str}"
        )


# ─── Schema extractor ─────────────────────────────────────────────────────────

def extract_schema(df: pd.DataFrame, source_type: str, path: str,
                   table_name: Optional[str] = None) -> SchemaInfo:
    columns = []
    for col in df.columns:
        series = df[col]
        null_pct = float(series.isna().mean())
        unique_count = int(series.nunique())
        sample = series.dropna().head(3).tolist()
        is_numeric = pd.api.types.is_numeric_dtype(series)
        is_datetime = pd.api.types.is_datetime64_any_dtype(series)

        columns.append(ColumnInfo(
            name=col,
            dtype=str(series.dtype),
            null_pct=null_pct,
            unique_count=unique_count,
            sample_values=sample,
            is_numeric=is_numeric,
            is_datetime=is_datetime,
        ))

    preview = df.head(_PREVIEW_ROWS).fillna("").to_dict(orient="records")
    return SchemaInfo(
        source_type=source_type,
        path=path,
        table_name=table_name,
        row_count=len(df),
        columns=columns,
        preview=preview,
    )


# ─── Connector implementations ────────────────────────────────────────────────

def connect_csv(path: str) -> pd.DataFrame:
    """Load CSV or TSV, auto-detect delimiter and encoding."""
    p = Path(path)
    sep = "\t" if p.suffix.lower() in {".tsv", ".txt"} else ","
    try:
        return pd.read_csv(path, sep=sep, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=sep, encoding="latin-1", low_memory=False)


def connect_json(path: str) -> pd.DataFrame:
    """Load JSON or JSONL into a DataFrame."""
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        return pd.read_json(path, lines=True)
    return pd.read_json(path)


def connect_excel(path: str, sheet_name: Optional[str] = None) -> tuple[pd.DataFrame, str]:
    """Load Excel sheet; returns (df, sheet_name).
    Auto-detects the real header row by scanning the first 5 rows for the one
    with the most non-null string values (handles files with merged title rows).
    """
    xl = pd.ExcelFile(path)
    sheet = sheet_name or xl.sheet_names[0]

    # Try header=0 first; if >50% columns are "Unnamed" scan for a better header row
    df = xl.parse(sheet)
    unnamed_frac = sum(1 for c in df.columns if str(c).startswith("Unnamed")) / max(len(df.columns), 1)
    if unnamed_frac > 0.5:
        best_row, best_score = 0, 0
        for row in range(min(5, len(df))):
            vals = df.iloc[row]
            score = sum(1 for v in vals if isinstance(v, str) and v.strip())
            if score > best_score:
                best_score, best_row = score, row
        if best_row > 0:
            df = xl.parse(sheet, header=best_row)

    return df, sheet


def connect_sql(
    connection_string: str,
    table_name: Optional[str] = None,
    query: Optional[str] = None,
) -> tuple[pd.DataFrame, str]:
    """
    Load data from a SQL database via SQLAlchemy.
    Auto-detects the first table if neither table_name nor query is given.
    Returns (df, resolved_table_name).
    """
    from sqlalchemy import create_engine, inspect, text

    engine = create_engine(connection_string)
    inspector = inspect(engine)

    if query:
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        return df, "custom_query"

    if not table_name:
        tables = inspector.get_table_names()
        if not tables:
            raise ValueError(f"No tables found in database: {connection_string}")
        table_name = tables[0]
        logger.info("Auto-selected table: %s", table_name)

    with engine.connect() as conn:
        df = pd.read_sql_table(table_name, conn)
    return df, table_name


def list_related_tables(connection_string: str, exclude_table: Optional[str] = None) -> dict[str, list[str]]:
    """
    Enumerate every table in a SQL database and its column names/types —
    schema only, no rows loaded (unlike connect_sql, which pulls a full
    table into memory). Used to tell the Analysis Planner what else is
    available for a cross-table JOIN beyond the single "active" table it
    already has a preview of.
    """
    from sqlalchemy import create_engine, inspect

    engine = create_engine(connection_string)
    inspector = inspect(engine)
    tables: dict[str, list[str]] = {}
    for table_name in inspector.get_table_names():
        if table_name == exclude_table:
            continue
        tables[table_name] = [
            f"{col['name']} ({col['type']})" for col in inspector.get_columns(table_name)
        ]
    return tables


# ─── NL Schema Summarizer ─────────────────────────────────────────────────────

async def generate_nl_summary(schema: SchemaInfo, llm=None) -> str:
    """
    Generate a natural-language description of the dataset using an LLM.
    Falls back to rule_based_summary() if LLM is unavailable or fails.
    """
    if llm is None:
        llm = _build_llm()

    def _safe_default(obj):
        import datetime
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        return str(obj)

    schema_text = json.dumps(schema.to_dict(), indent=2, default=_safe_default)[:2000]
    prompt = (
        f"Schema JSON:\n{schema_text}\n\n"
        f"Preview (first {_PREVIEW_ROWS} rows):\n"
        + json.dumps(schema.preview[:3], indent=2, default=_safe_default)
    )

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        response = await llm.ainvoke([
            SystemMessage(content=SCHEMA_SUMMARIZER_SYSTEM),
            HumanMessage(content=prompt),
        ])
        return response.content.strip()
    except Exception as exc:
        logger.warning("LLM schema summary failed (%s); using rule-based fallback", exc)
        return schema.rule_based_summary()


def _build_llm():
    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.llm_model,
            temperature=0.0,
            max_tokens=256,
            api_key=settings.anthropic_api_key or "sk-no-key",
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=0.0,
        max_tokens=256,
        api_key=settings.openai_api_key or "sk-no-key",
    )


# ─── Unified DataConnector ────────────────────────────────────────────────────

class DataConnector:
    """
    Top-level connector: given a source descriptor dict, loads data, extracts
    schema, and (optionally) generates an NL summary.

    Source descriptor format (matches state["data_sources"] entries):
      {"type": "csv"|"sql"|"json"|"excel",
       "path": "...",            # file path OR SQL connection string
       "table_name": "...",      # SQL table / Excel sheet (optional)
       "query": "..."}           # raw SQL query (optional)
    """

    def __init__(self, llm=None):
        self._llm = llm  # injected for testing

    async def connect(self, source: dict) -> SchemaInfo:
        """
        Connect to a source, extract schema, return SchemaInfo.
        Does NOT do quality profiling (that's MCP's job).
        """
        source_type = source.get("type", "").lower()
        path = source.get("path", "")

        if source_type in {"csv", "tsv", ""}:
            df = connect_csv(path)
            schema = extract_schema(df, "csv", path)

        elif source_type == "json":
            df = connect_json(path)
            schema = extract_schema(df, "json", path)

        elif source_type == "excel":
            sheet = source.get("table_name")
            df, sheet_name = connect_excel(path, sheet)
            schema = extract_schema(df, "excel", path, table_name=sheet_name)

        elif source_type == "sql":
            table = source.get("table_name")
            query = source.get("query")
            df, table_name = connect_sql(path, table, query)
            schema = extract_schema(df, "sql", path, table_name=table_name)

        else:
            raise ValueError(f"Unsupported source type: {source_type!r}")

        logger.info(
            "Connected to %s | rows=%d cols=%d",
            path, schema.row_count, len(schema.columns)
        )
        return schema

    async def connect_with_summary(self, source: dict) -> tuple[SchemaInfo, str]:
        """Connect and generate NL summary. Returns (schema, nl_summary)."""
        schema = await self.connect(source)
        summary = await generate_nl_summary(schema, self._llm)
        return schema, summary
