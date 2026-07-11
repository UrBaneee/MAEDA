"""
SQL Query Tool — execute SQL against a database or a pandas DataFrame.

For file-based sources (CSV, JSON, Excel) the DataFrame is registered in an
in-memory SQLite database so the LLM can write SQL without knowing the source type.
For SQL sources, queries run directly against the configured engine.
"""
from __future__ import annotations

import re
import sqlite3
from typing import Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("maeda.tools.sql")

_BARE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SORT_DIRECTION_RE = re.compile(r"\s+(ASC|DESC)$", re.IGNORECASE)
_ALIAS_RE = re.compile(r"\bAS\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", re.IGNORECASE)


def _is_bare_identifier(value: object) -> bool:
    """True if value looks like a plain column name, not a SQL expression."""
    return isinstance(value, str) and bool(_BARE_IDENTIFIER_RE.match(value))


def _strip_sort_direction(order_by: str) -> str:
    return _SORT_DIRECTION_RE.sub("", order_by)


def _select_aliases(select_cols: Optional[list]) -> set[str]:
    """
    Names introduced by "<expr> AS alias" entries in select_columns — e.g.
    "SUM(revenue) AS total_revenue" defines "total_revenue". These are valid
    to reference in ORDER BY even though they're not columns on the original
    dataframe; column-existence validation must not reject them.
    """
    aliases = set()
    for c in select_cols or []:
        if isinstance(c, str):
            m = _ALIAS_RE.search(c.strip())
            if m:
                aliases.add(m.group(1))
    return aliases


def execute_sql(
    query: str,
    df: Optional[pd.DataFrame] = None,
    connection_string: Optional[str] = None,
    table_name: str = "data",
) -> dict:
    """
    Execute a SQL query and return a result dict.

    Priority:
      1. If connection_string is provided → run against that DB directly.
      2. If df is provided → register in in-memory SQLite, run query.

    Returns:
        {
          "rows": list[dict],
          "row_count": int,
          "columns": list[str],
          "result_df": pd.DataFrame,   # caller can pass to next step
        }
    """
    if connection_string:
        return _execute_on_db(query, connection_string)
    if df is not None:
        return _execute_on_dataframe(query, df, table_name)
    raise ValueError("Either df or connection_string must be provided")


def _execute_on_db(query: str, connection_string: str) -> dict:
    from sqlalchemy import create_engine, text
    engine = create_engine(connection_string)
    with engine.connect() as conn:
        result_df = pd.read_sql(text(query), conn)
    return _df_to_result(result_df)


def _execute_on_dataframe(
    query: str, df: pd.DataFrame, table_name: str = "data"
) -> dict:
    """Register df in a transient SQLite and run the query."""
    conn = sqlite3.connect(":memory:")
    df.to_sql(table_name, conn, index=False, if_exists="replace")
    try:
        result_df = pd.read_sql_query(query, conn)
    finally:
        conn.close()
    return _df_to_result(result_df)


def _df_to_result(df: pd.DataFrame) -> dict:
    return {
        "rows": df.to_dict(orient="records"),
        "row_count": len(df),
        "columns": list(df.columns),
        "result_df": df,
    }


# ─── LangGraph-compatible tool wrapper ────────────────────────────────────────

# Parameter keys the fallback query builder understands. Anything else present
# (e.g. a raw "aggregate" expression, or "table" instead of "table_name") means
# the planner intended something this builder can't construct — better to raise
# than to silently emit a "SELECT *" that drops that intent on the floor.
_SQL_BUILDER_KEYS = {
    "query", "table_name", "connection_string",
    "select_columns", "columns", "group_by", "order_by", "limit",
}


def sql_tool(df: pd.DataFrame, parameters: dict, prior_results: dict) -> dict:
    """
    Tool dispatcher called by the step executor.
    parameters keys: query, table_name (optional), connection_string (optional)
    """
    query = parameters.get("query", "")
    if not query:
        unrecognized = set(parameters.keys()) - _SQL_BUILDER_KEYS
        if unrecognized:
            raise ValueError(
                f"sql_query: no 'query' provided and parameters contain keys this "
                f"fallback builder doesn't understand: {sorted(unrecognized)}. "
                f"Provide a raw 'query' string, or use only {sorted(_SQL_BUILDER_KEYS)}."
            )

        # Build a basic query from other parameters the LLM may have provided
        table = parameters.get("table_name", "data")
        select_cols = parameters.get("select_columns") or parameters.get("columns")
        group_by = parameters.get("group_by")
        order_by = parameters.get("order_by")
        # A JSON "limit": null becomes Python None, and dict.get() only falls
        # back to the default when the key is *absent* — an explicit None
        # would otherwise reach the query as the literal text "LIMIT None".
        limit = parameters.get("limit")
        if limit is None:
            limit = 100

        # A GROUP BY with no explicit select_columns is meaningless — there's
        # no way to know what to aggregate, and defaulting to "SELECT *" lets
        # SQLite silently return one arbitrary row per group instead of an
        # actual aggregate. Require the caller to specify what to select.
        if group_by and not select_cols:
            raise ValueError(
                "sql_query: 'group_by' was given without 'select_columns' — "
                "specify the columns/aggregate expressions to select "
                "(e.g. ['region', 'SUM(revenue) AS total']) rather than "
                "relying on an implicit SELECT *."
            )

        # select_columns/order_by may legitimately be raw SQL expressions
        # (e.g. "AVG(unit_price - cost) AS margin"), not bare column names —
        # only validate entries that actually look like a plain identifier.
        # order_by may also legitimately reference an alias defined in
        # select_columns (e.g. "... AS total_revenue ... ORDER BY
        # total_revenue") rather than an original dataframe column.
        aliases = _select_aliases(select_cols)
        candidate_cols = [c for c in (select_cols or []) if _is_bare_identifier(c)]
        if group_by:
            candidate_cols += group_by if isinstance(group_by, list) else [group_by]
        if order_by:
            sort_col = _strip_sort_direction(order_by)
            if _is_bare_identifier(sort_col) and sort_col not in aliases:
                candidate_cols.append(sort_col)
        missing = [c for c in candidate_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"sql_query: column(s) {missing} not found in dataframe. "
                f"Available columns: {list(df.columns)}"
            )

        cols_str = ", ".join(select_cols) if select_cols else "*"
        query = f"SELECT {cols_str} FROM {table}"
        if group_by:
            gb = ", ".join(group_by) if isinstance(group_by, list) else group_by
            query += f" GROUP BY {gb}"
        if order_by:
            query += f" ORDER BY {order_by}"
        query += f" LIMIT {limit}"

    conn_str = parameters.get("connection_string")
    table = parameters.get("table_name", "data")
    result = execute_sql(query, df=df, connection_string=conn_str, table_name=table)
    # Build a human-readable summary that includes the actual top rows
    rows = result["rows"]
    cols = result["columns"]
    if rows:
        top = rows[:5]
        row_strs = [", ".join(f"{k}={v}" for k, v in r.items()) for r in top]
        data_preview = "; ".join(row_strs)
        summary = f"SQL returned {result['row_count']} rows. Top results: {data_preview}"
    else:
        summary = f"SQL returned 0 rows (no matching data)"
    return {
        "result": rows,
        "result_df": result["result_df"],
        "result_summary": summary,
        "columns": cols,
        "warnings": [],
    }
