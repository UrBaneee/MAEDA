"""
Analysis Agent — Phase 5.

Responsibilities:
  1. Plan: LLM generates an AnalysisPlan (list of typed AnalysisStep) from
     ParsedIntent + schema summary.
  2. Execute: Run each step via the tool registry, respecting step dependencies.
  3. Error recovery: Retry a failed step once with a simplified fallback; if
     still failing mark it as failed and continue rather than abort.
  4. Aggregate: Combine all step results into structured analysis_results.

State fields read:  parsed_intent, schema_summary, active_source, data_sources
State fields written: analysis_plan, analysis_results, intermediate_data,
                      decision_trace, token_usage
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal, Optional

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base_agent import BaseAgent
from src.config.agent_prompts import (
    ANALYSIS_EXECUTOR_SYSTEM,
    ANALYSIS_PLANNER_SYSTEM,
    STEP_REPAIR_SYSTEM,
)
from src.config.settings import settings
from src.state.graph_state import MAEDAState
from src.tools.sql_tool import sql_tool
from src.tools.stats_tool import (
    anomaly_tool,
    comparison_tool,
    pandas_tool,
    statistical_tool,
    timeseries_tool,
)
from src.utils.logger import get_logger

logger = get_logger("maeda.agent.analysis")

_MAX_RETRIES = 1  # retry a failed step once before marking as failed


# ─── Data model ───────────────────────────────────────────────────────────────

@dataclass
class AnalysisStep:
    step_number: int
    method: str
    tool: str       # one of the TOOL_REGISTRY keys
    parameters: dict
    depends_on: list[int]
    expected_output: str
    rationale: str

    @classmethod
    def from_dict(cls, d: dict) -> "AnalysisStep":
        return cls(
            step_number=int(d.get("step_number", 0)),
            method=d.get("method", ""),
            tool=d.get("tool", "pandas_transform"),
            parameters=d.get("parameters") or {},
            depends_on=[int(x) for x in (d.get("depends_on") or [])],
            expected_output=d.get("expected_output", ""),
            rationale=d.get("rationale", ""),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AnalysisPlan:
    steps: list[AnalysisStep]
    estimated_complexity: Literal["simple", "moderate", "complex"] = "moderate"
    rationale: str = ""

    @classmethod
    def from_llm_response(cls, data: dict) -> "AnalysisPlan":
        # LLM returns either a list directly or {"steps": [...], ...}
        if isinstance(data, list):
            steps_raw = data
            complexity = "moderate"
            rationale = ""
        else:
            steps_raw = data.get("steps") or []
            complexity = data.get("estimated_complexity", "moderate")
            rationale = data.get("rationale", "")
        steps = [AnalysisStep.from_dict(s) for s in steps_raw]
        return cls(steps=steps, estimated_complexity=complexity, rationale=rationale)

    def to_state_list(self) -> list[dict]:
        return [s.to_dict() for s in self.steps]


# ─── Tool registry ────────────────────────────────────────────────────────────

ToolFn = Callable[[pd.DataFrame, dict, dict], dict]

TOOL_REGISTRY: dict[str, ToolFn] = {
    "sql_query":       sql_tool,
    "pandas_transform": pandas_tool,
    "statistical_test": statistical_tool,
    "anomaly_detection": anomaly_tool,
    "time_series":     timeseries_tool,
    "comparison":      comparison_tool,
}


# ─── LLM factory ─────────────────────────────────────────────────────────────

def _build_llm():
    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.llm_model, temperature=0.0,
            max_tokens=settings.max_tokens_per_call, api_key=settings.anthropic_api_key or "sk-no-key",
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=settings.llm_model, temperature=0.0,
        max_tokens=settings.max_tokens_per_call, api_key=settings.openai_api_key or "sk-no-key",
    )


# ─── AnalysisAgent ────────────────────────────────────────────────────────────

class AnalysisAgent(BaseAgent):
    """
    Plans and executes multi-step data analysis.

    Two entry points (matching the graph nodes):
      - plan(state)    → populates state["analysis_plan"]
      - execute(state) → populates state["analysis_results"]
    """

    def __init__(self, llm=None):
        super().__init__("analysis_agent")
        self._llm = llm or _build_llm()

    # ── process() routes based on current_phase ───────────────────────────────

    async def process(self, state: MAEDAState) -> MAEDAState:
        if not state.get("analysis_plan"):
            return await self.plan(state)
        return await self.execute(state)

    # ── 5.1 Plan generator ────────────────────────────────────────────────────

    async def plan(self, state: MAEDAState) -> MAEDAState:
        intent = state.get("parsed_intent") or {}
        schema = state.get("schema_summary", "No schema available")
        active_source = state.get("active_source")
        columns = _column_manifest(active_source)
        related_tables = _related_tables_manifest(active_source)

        prompt = (
            f"### Parsed Intent\n{json.dumps(intent, indent=2)}\n\n"
            f"### Data Schema\n{schema}\n\n"
            f"### Available Columns (authoritative — use these exact names)\n{columns}\n\n"
        )
        if related_tables:
            prompt += f"### Related Tables\n{related_tables}\n\n"
        prompt += "Generate a step-by-step analysis plan."
        messages = [
            SystemMessage(content=ANALYSIS_PLANNER_SYSTEM),
            HumanMessage(content=prompt),
        ]

        try:
            response = await self._llm.ainvoke(messages)
            usage = getattr(response, "usage_metadata", None) or {}
            self._cost_tracker.record(
                agent_name=self.name, model=settings.llm_model,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                call_label="plan_analysis",
            )
            state["token_usage"] = self._cost_tracker.to_state_dict()
            data = _parse_json(response.content.strip())
            plan = AnalysisPlan.from_llm_response(data)
        except Exception as exc:
            logger.warning("Plan generation failed: %s — using empty plan", exc)
            plan = AnalysisPlan(steps=[], rationale=f"Planning failed: {exc}")

        state["analysis_plan"] = plan.to_state_list()
        state = self.log_decision(
            state,
            action="plan_analysis",
            reasoning=f"Generated {len(plan.steps)}-step plan; complexity={plan.estimated_complexity}",
            inputs={"intent_type": intent.get("query_type"), "schema_len": len(schema)},
            outputs={"n_steps": len(plan.steps), "complexity": plan.estimated_complexity},
            confidence=0.9 if plan.steps else 0.3,
        )
        return state

    # ── 5.6 / 5.7 Step executor with error recovery ───────────────────────────

    async def execute(self, state: MAEDAState) -> MAEDAState:
        plan_dicts = state.get("analysis_plan") or []
        if not plan_dicts:
            state["analysis_results"] = []
            return state

        steps = [AnalysisStep.from_dict(d) for d in plan_dicts]
        original_df = _load_dataframe(state)
        column_manifest = _column_manifest(state.get("active_source"))

        results: dict[int, dict] = {}  # step_number → result
        result_dfs: dict[int, pd.DataFrame] = {}  # step_number → its output DataFrame, if any
        analysis_results: list[dict] = []

        for step in _execution_order(steps):
            prior = {n: results[n] for n in step.depends_on if n in results}
            step_df = _select_input_dataframe(step, result_dfs, original_df)
            step_result = await self._execute_step(step, step_df, prior, column_manifest)
            results[step.step_number] = step_result
            analysis_results.append({
                "step": step.step_number,
                "method": step.method,
                "tool": step.tool,
                "result": step_result.get("result"),
                "result_summary": step_result.get("result_summary", ""),
                "confidence": step_result.get("confidence", 1.0),
                "warnings": step_result.get("warnings", []),
                "failed": step_result.get("failed", False),
            })

            if step_result.get("result_df") is not None:
                result_dfs[step.step_number] = step_result["result_df"]

        state["analysis_results"] = analysis_results
        # 5.8 Aggregate: store a compact intermediate_data for insight generation
        state["intermediate_data"] = _aggregate(analysis_results)
        state["token_usage"] = self._cost_tracker.to_state_dict()

        state = self.log_decision(
            state,
            action="execute_analysis",
            reasoning=f"Executed {len(analysis_results)} steps; "
                      f"{sum(1 for r in analysis_results if r['failed'])} failed",
            outputs={"n_results": len(analysis_results)},
            confidence=1.0,
        )
        return state

    async def _execute_step(
        self, step: AnalysisStep, df: pd.DataFrame, prior: dict, column_manifest: str = ""
    ) -> dict:
        """Execute one step with up to _MAX_RETRIES retries on failure.

        The retry tries an LLM-informed repair first — tool errors already
        include actionable detail (e.g. the exact list of available
        columns; see _require_columns), but that detail used to be
        discarded: the only retry path was _simplify_step, which blindly
        strips parameters down to a fixed whitelist with no awareness of
        *why* the step failed. _repair_step feeds the real error message
        back to the LLM for a targeted fix; _simplify_step remains the
        fallback when repair isn't available or doesn't return anything
        usable.
        """
        tool_fn = TOOL_REGISTRY.get(step.tool)
        if tool_fn is None:
            return {
                "result": None,
                "result_summary": f"Unknown tool: {step.tool}",
                "confidence": 0.0,
                "warnings": [f"Tool '{step.tool}' is not registered"],
                "failed": True,
            }

        last_exc: Optional[Exception] = None
        retry_note: Optional[str] = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                result = tool_fn(df, step.parameters, prior)
                result.setdefault("confidence", 1.0)
                result.setdefault("failed", False)
                if attempt > 0:
                    note = f"Succeeded on retry {attempt}"
                    if retry_note:
                        note += f" ({retry_note})"
                    result["warnings"] = result.get("warnings", []) + [note]
                return result
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Step %d (%s) attempt %d failed: %s",
                    step.step_number, step.tool, attempt + 1, exc
                )
                if attempt < _MAX_RETRIES:
                    repaired = await self._repair_step(step, exc, column_manifest)
                    if repaired is not None:
                        step = repaired
                        retry_note = "LLM-repaired"
                    else:
                        step = _simplify_step(step)
                        retry_note = "generic simplify"

        return {
            "result": None,
            "result_summary": f"Step failed after {_MAX_RETRIES + 1} attempts: {last_exc}",
            "confidence": 0.0,
            "warnings": [str(last_exc)],
            "failed": True,
        }

    async def _repair_step(
        self, step: AnalysisStep, error: Exception, column_manifest: str
    ) -> Optional[AnalysisStep]:
        """
        Ask the LLM to fix a failed step's parameters using the tool's own
        error message. Returns a corrected AnalysisStep (same tool/method,
        repaired parameters), or None if repair isn't confidently possible
        or the LLM call itself fails — callers must fall back to
        _simplify_step in that case.
        """
        prompt = (
            f"### Failed Step\n"
            f"tool: {step.tool}\n"
            f"method: {step.method}\n"
            f"parameters: {json.dumps(step.parameters)}\n"
            f"expected_output: {step.expected_output}\n"
            f"rationale: {step.rationale}\n\n"
            f"### Error\n{error}\n\n"
            f"### Available Columns (authoritative — use these exact names)\n"
            f"{column_manifest or '(not available)'}\n"
        )
        try:
            response = await self._llm.ainvoke([
                SystemMessage(content=STEP_REPAIR_SYSTEM),
                HumanMessage(content=prompt),
            ])
            usage = getattr(response, "usage_metadata", None) or {}
            self._cost_tracker.record(
                agent_name=self.name, model=settings.llm_model,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                call_label="repair_step",
            )
            data = _parse_json(response.content.strip())
            new_params = data.get("parameters")
            if not isinstance(new_params, dict) or not new_params:
                return None
            logger.info(
                "Step %d (%s) repair: %s",
                step.step_number, step.tool, data.get("reasoning", "")
            )
            from dataclasses import replace
            return replace(step, parameters=new_params)
        except Exception as exc:
            logger.warning("Step repair LLM call failed: %s", exc)
            return None


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _column_manifest(active_source: Optional[dict]) -> str:
    """
    Render the exact column list (name + dtype) from active_source's schema.
    This is the authoritative reference the Planner must copy column names
    from — schema_summary alone is free-text prose and gets paraphrased.
    """
    columns = ((active_source or {}).get("schema") or {}).get("columns") or []
    if not columns:
        return "(no column schema available)"
    return "\n".join(
        f"- {c.get('name')} ({c.get('dtype')})"
        for c in columns
    )


def _related_tables_manifest(active_source: Optional[dict]) -> str:
    """
    For a SQL-backed source, list every OTHER table in the database and its
    columns, so the Planner knows a cross-table JOIN is possible instead of
    only ever seeing the single "active" table's flat schema. Returns "" for
    non-SQL sources or if introspection fails for any reason — this is
    purely additive context; its absence must not block planning against
    the single active table the way it always has.
    """
    active = active_source or {}
    if active.get("type") != "sql":
        return ""
    connection_string = active.get("path")
    if not connection_string:
        return ""
    try:
        from src.tools.data_connector import list_related_tables
        tables = list_related_tables(connection_string, exclude_table=active.get("table_name"))
    except Exception as exc:
        logger.warning("Could not introspect related tables for %s: %s", connection_string, exc)
        return ""
    if not tables:
        return ""

    lines = [
        f'Other tables in this database (connection_string: "{connection_string}"). '
        f"To JOIN across tables, use tool \"sql_query\" with a raw \"query\" string "
        f"(real SQL, real table names below) and \"connection_string\": "
        f'"{connection_string}" in parameters — this runs directly against the '
        f"database. A \"pandas_transform\" step only ever sees the single active "
        f"table above; it cannot join.",
        "",
    ]
    for table_name, columns in tables.items():
        lines.append(f"Table: {table_name}")
        lines.extend(f"  - {c}" for c in columns)
        lines.append("")
    return "\n".join(lines).rstrip()


def _select_input_dataframe(
    step: "AnalysisStep", result_dfs: dict[int, pd.DataFrame], original_df: pd.DataFrame
) -> pd.DataFrame:
    """
    A step with no declared dependencies (depends_on=[]) always operates on
    the original, full dataset — never on whatever an unrelated preceding
    step happened to leave behind. Previously a single `df` variable was
    reassigned after every step regardless of dependencies, so an
    independent step could silently receive a different, already-aggregated
    (and column-reduced) DataFrame from whichever step ran immediately
    before it in plan order, causing spurious "column not found" failures
    on legitimately independent analysis facets (e.g. an "overview" query
    planning "revenue by region" and "units by product" as two unrelated
    steps).

    A step that does declare dependencies chains off the most recent
    dependency (in depends_on order) that actually produced a DataFrame.
    """
    for dep in reversed(step.depends_on):
        if dep in result_dfs:
            return result_dfs[dep]
    return original_df


def _load_dataframe(state: MAEDAState) -> pd.DataFrame:
    """Load the active DataFrame from state, preferring schema preview for speed."""
    active = state.get("active_source") or {}
    path = active.get("path", "")
    source_type = active.get("type", "csv")

    if not path:
        sources = state.get("data_sources") or []
        if sources:
            path = sources[0].get("path", "")
            source_type = sources[0].get("type", "csv")

    if not path:
        logger.warning("No data source path in state; returning empty DataFrame")
        return pd.DataFrame()

    from src.tools.data_connector import connect_csv, connect_excel, connect_json, connect_sql
    try:
        if source_type in {"csv", "tsv", ""}:
            return connect_csv(path)
        if source_type == "json":
            return connect_json(path)
        if source_type == "excel":
            df, _ = connect_excel(path, active.get("table_name"))
            return df
        if source_type == "sql":
            df, _ = connect_sql(path, active.get("table_name"))
            return df
    except Exception as exc:
        logger.error("Failed to load DataFrame from %s: %s", path, exc)
    return pd.DataFrame()


def _execution_order(steps: list[AnalysisStep]) -> list[AnalysisStep]:
    """
    Topological sort: steps with no dependencies first, then those whose
    dependencies are already resolved.  Falls back to sequential order.
    """
    resolved: set[int] = set()
    ordered: list[AnalysisStep] = []
    remaining = list(steps)
    max_iters = len(steps) * len(steps) + 1
    iters = 0
    while remaining and iters < max_iters:
        iters += 1
        for step in list(remaining):
            if all(d in resolved for d in step.depends_on):
                ordered.append(step)
                resolved.add(step.step_number)
                remaining.remove(step)
    # Any unresolved steps appended at the end (cycle safety)
    ordered.extend(remaining)
    return ordered


def _simplify_step(step: AnalysisStep) -> AnalysisStep:
    """
    Retry fallback: strip optional/advanced parameters to reduce failure surface.
    """
    simplified_params = {
        k: v for k, v in step.parameters.items()
        if k in {"query", "operation", "group_by", "agg_col", "column",
                 "test", "target", "features", "date_col", "value_col",
                 "segment_col", "method", "agg_func"}
    }
    # For pandas_transform always retry with count to handle non-numeric columns
    if step.tool == "pandas_transform":
        simplified_params["agg_func"] = "count"

    from dataclasses import replace
    return replace(step, parameters=simplified_params)


def _aggregate(results: list[dict]) -> dict:
    """
    5.8 Build a compact aggregated summary for downstream insight generation.
    """
    successful = [r for r in results if not r.get("failed")]
    return {
        "n_steps_total": len(results),
        "n_steps_successful": len(successful),
        "summaries": [r["result_summary"] for r in successful],
        "key_findings": [
            {"step": r["step"], "method": r["method"], "result": r["result"]}
            for r in successful
        ],
    }


def _parse_json(text: str) -> Any:
    """Extract JSON from an LLM response (handles markdown fences)."""
    if "```" in text:
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines)
    text = text.strip()

    # Fast path: the response is exactly JSON, as instructed. Try this
    # before any bracket-scanning heuristic -- scanning for the first "["
    # and last "]" breaks the moment the top-level shape is an object that
    # merely *contains* an array (e.g. {"parameters": {"group_by": [...]}}),
    # since the array's own brackets get mistaken for the outer ones.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: find the first opening bracket and match it to the outer
    # closing bracket, handling a JSON blob embedded in surrounding prose.
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"No valid JSON found in: {text[:200]!r}")
