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
from src.config.agent_prompts import ANALYSIS_EXECUTOR_SYSTEM, ANALYSIS_PLANNER_SYSTEM
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
        columns = _column_manifest(state.get("active_source"))

        prompt = (
            f"### Parsed Intent\n{json.dumps(intent, indent=2)}\n\n"
            f"### Data Schema\n{schema}\n\n"
            f"### Available Columns (authoritative — use these exact names)\n{columns}\n\n"
            "Generate a step-by-step analysis plan."
        )
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

        results: dict[int, dict] = {}  # step_number → result
        result_dfs: dict[int, pd.DataFrame] = {}  # step_number → its output DataFrame, if any
        analysis_results: list[dict] = []

        for step in _execution_order(steps):
            prior = {n: results[n] for n in step.depends_on if n in results}
            step_df = _select_input_dataframe(step, result_dfs, original_df)
            step_result = await self._execute_step(step, step_df, prior)
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
        self, step: AnalysisStep, df: pd.DataFrame, prior: dict
    ) -> dict:
        """Execute one step with up to _MAX_RETRIES retries on failure."""
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
        for attempt in range(_MAX_RETRIES + 1):
            try:
                result = tool_fn(df, step.parameters, prior)
                result.setdefault("confidence", 1.0)
                result.setdefault("failed", False)
                if attempt > 0:
                    result["warnings"] = result.get("warnings", []) + [
                        f"Succeeded on retry {attempt}"
                    ]
                return result
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Step %d (%s) attempt %d failed: %s",
                    step.step_number, step.tool, attempt + 1, exc
                )
                if attempt < _MAX_RETRIES:
                    # Simplify parameters for retry
                    step = _simplify_step(step)

        return {
            "result": None,
            "result_summary": f"Step failed after {_MAX_RETRIES + 1} attempts: {last_exc}",
            "confidence": 0.0,
            "warnings": [str(last_exc)],
            "failed": True,
        }


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

    # Try array first, then object
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"No valid JSON found in: {text[:200]!r}")
