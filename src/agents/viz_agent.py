"""
Visualization Agent — Phase 6.

Orchestrates: chart recommendation → static generation → interactive generation
              → LLM captioning → multi-chart dashboard.

State fields read:  analysis_results, parsed_intent, active_source, intermediate_data
State fields written: charts, decision_trace, token_usage
"""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base_agent import BaseAgent
from src.config.agent_prompts import VIZ_CAPTION_SYSTEM, VIZ_RECOMMENDER_SYSTEM
from src.config.settings import settings
from src.state.graph_state import MAEDAState
from src.tools.chart_tool import (
    ChartSpec,
    generate_dashboard,
    generate_interactive_chart,
    generate_static_chart,
    recommend_chart,
)
from src.utils.logger import get_logger

logger = get_logger("maeda.agent.viz")

_CHARTS_DIR = "./data/charts"


# ─── LLM factory ─────────────────────────────────────────────────────────────

def _build_llm():
    # Was hardcoded to temperature=0.2, bypassing settings.llm_temperature —
    # see insight_agent.py for why non-zero temperature caused a real
    # eval-visible failure. Kept consistent with the rest of the pipeline.
    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.llm_model, temperature=settings.llm_temperature,
            max_tokens=256, api_key=settings.anthropic_api_key or "sk-no-key",
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=settings.llm_model, temperature=settings.llm_temperature,
        max_tokens=256, api_key=settings.openai_api_key or "sk-no-key",
    )


# ─── VizAgent ────────────────────────────────────────────────────────────────

class VizAgent(BaseAgent):
    """
    Generates charts for all analysis results and stores them in state["charts"].

    For each analysis_result:
      1. recommend_chart() selects the chart type
      2. generate_static_chart() saves a PNG
      3. generate_interactive_chart() creates a Plotly JSON
      4. _caption_chart() asks the LLM for a one-sentence description

    Also generates a multi-chart dashboard PNG if ≥2 charts.
    """

    def __init__(self, llm=None, charts_dir: str = _CHARTS_DIR):
        super().__init__("viz_agent")
        self._llm = llm or _build_llm()
        self._charts_dir = charts_dir

    # ── Main entry point ─────────────────────────────────────────────────────

    async def process(self, state: MAEDAState) -> MAEDAState:
        results = state.get("analysis_results") or []
        intent_type = (state.get("parsed_intent") or {}).get("query_type", "descriptive")

        # Load the working DataFrame once
        df = _load_df(state)

        charts: list[dict] = []
        specs_for_dashboard: list[ChartSpec] = []

        for ar in results:
            if ar.get("failed"):
                continue
            result_data = ar.get("result")
            if result_data is None:
                continue

            spec = recommend_chart(result_data, method=ar.get("method", ""), intent_type=intent_type)
            if spec is None:
                continue

            # Determine the DataFrame for this chart
            chart_df = df if spec.data is None else None

            # 6.2 Static chart
            try:
                image_path = generate_static_chart(spec, df=chart_df, output_dir=self._charts_dir)
            except Exception as exc:
                logger.warning("Static chart generation failed: %s", exc)
                image_path = ""

            # 6.3 Interactive chart
            try:
                interactive = generate_interactive_chart(spec, df=chart_df)
                plotly_json = interactive.get("json", "")
            except Exception as exc:
                logger.warning("Interactive chart generation failed: %s", exc)
                plotly_json = ""

            # 6.5 Caption
            caption = await self._caption_chart(spec, ar)

            chart_record = {
                **spec.to_dict(),
                "image_path": image_path,
                "plotly_json": plotly_json,
                "caption": caption,
                "analysis_step": ar.get("step"),
            }
            charts.append(chart_record)
            specs_for_dashboard.append(spec)

        # 6.4 Dashboard
        if len(specs_for_dashboard) >= 2:
            try:
                dashboard_path = generate_dashboard(
                    specs_for_dashboard, output_dir=self._charts_dir
                )
                charts.append({
                    "chart_type": "dashboard",
                    "title": "MAEDA Analysis Dashboard",
                    "image_path": dashboard_path,
                    "plotly_json": "",
                    "caption": f"Dashboard combining {len(specs_for_dashboard)} charts.",
                    "analysis_step": None,
                })
            except Exception as exc:
                logger.warning("Dashboard generation failed: %s", exc)

        state["charts"] = charts
        # Sync token usage from _caption_chart()'s calls -- docstring has
        # always claimed this agent writes token_usage, but nothing ever
        # actually did; merge (not overwrite) so an earlier agent's entries
        # in the same state survive this write too.
        state["token_usage"] = {**state.get("token_usage", {}), **self._cost_tracker.to_state_dict()}
        state = self.log_decision(
            state,
            action="generate_viz",
            reasoning=f"Generated {len(charts)} charts from {len(results)} analysis results",
            inputs={"n_results": len(results)},
            outputs={"n_charts": len(charts)},
            confidence=1.0,
        )
        return state

    # ── 6.5 Chart captioning ─────────────────────────────────────────────────

    async def _caption_chart(self, spec: ChartSpec, analysis_result: dict) -> str:
        """Ask the LLM for a concise chart caption. Falls back to a rule-based caption."""
        context = (
            f"Chart type: {spec.chart_type}\n"
            f"Title: {spec.title}\n"
            f"Analysis method: {analysis_result.get('method', '')}\n"
            f"Result summary: {analysis_result.get('result_summary', '')}\n"
        )
        try:
            response = await self._llm.ainvoke([
                SystemMessage(content=VIZ_CAPTION_SYSTEM),
                HumanMessage(content=context),
            ])
            usage = getattr(response, "usage_metadata", None) or {}
            self._cost_tracker.record(
                agent_name=self.name, model=settings.llm_model,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                call_label="caption_chart",
            )
            return response.content.strip()
        except Exception as exc:
            logger.debug("Caption LLM failed (%s); using fallback", exc)
            return _rule_based_caption(spec, analysis_result)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _load_df(state: MAEDAState) -> Optional[pd.DataFrame]:
    active = state.get("active_source") or {}
    path = active.get("path", "")
    source_type = active.get("type", "csv")

    if not path:
        sources = state.get("data_sources") or []
        if sources:
            path = sources[0].get("path", "")
            source_type = sources[0].get("type", "csv")

    if not path:
        return None

    try:
        from src.tools.data_connector import (
            connect_csv, connect_excel, connect_json, connect_sql,
        )
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
        logger.warning("Could not load DataFrame for viz: %s", exc)
    return None


def _rule_based_caption(spec: ChartSpec, ar: dict) -> str:
    summary = ar.get("result_summary", "")
    chart_phrases = {
        "bar":            "Bar chart showing",
        "horizontal_bar": "Horizontal bar chart comparing",
        "line":           "Line chart tracking",
        "scatter":        "Scatter plot visualising",
        "heatmap":        "Heatmap of",
        "histogram":      "Distribution of",
        "box":            "Box plot of",
        "pie":            "Pie chart of",
        "grouped_bar":    "Grouped bar chart comparing",
    }
    prefix = chart_phrases.get(spec.chart_type, "Chart showing")
    return f"{prefix} {spec.title.lower()}. {summary}".strip()
