"""
Chart generation tools for MAEDA — static (matplotlib/seaborn) and
interactive (plotly).

6.6 Styling: all charts use the MAEDA theme — a clean professional palette
    with consistent fonts, grid, and axis labels.
6.1 Recommendation: rule-based engine maps data shape + intent to chart type.
6.2 Static charts: saved as PNG files; path stored in state["charts"].
6.3 Interactive charts: plotly figures serialised as JSON for Streamlit.
6.4 Multi-chart layout: matplotlib subplots for dashboard output.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — safe for server/test use
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns

from src.utils.logger import get_logger

logger = get_logger("maeda.tools.chart")

# ─── MAEDA theme ─────────────────────────────────────────────────────────────

MAEDA_PALETTE = ["#2563EB", "#16A34A", "#DC2626", "#D97706", "#7C3AED",
                 "#0891B2", "#BE185D", "#65A30D"]
MAEDA_PLOTLY_COLORS = ["#2563EB", "#16A34A", "#DC2626", "#D97706",
                        "#7C3AED", "#0891B2", "#BE185D", "#65A30D"]

def _apply_maeda_style() -> None:
    """Apply MAEDA matplotlib style globally."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#F8FAFC",
        "axes.edgecolor": "#CBD5E1",
        "axes.grid": True,
        "grid.color": "#E2E8F0",
        "grid.linewidth": 0.8,
        "axes.prop_cycle": plt.cycler(color=MAEDA_PALETTE),
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 100,
    })

_apply_maeda_style()


# ─── Chart spec ───────────────────────────────────────────────────────────────

@dataclass
class ChartSpec:
    chart_type: Literal[
        "bar", "horizontal_bar", "line", "scatter", "histogram",
        "box", "heatmap", "pie", "grouped_bar", "treemap"
    ]
    title: str
    x_axis: Optional[str] = None
    y_axis: Optional[str | list] = None
    color_by: Optional[str] = None
    data: Optional[list[dict]] = None     # rows to visualise
    matrix: Optional[dict] = None         # for heatmap
    rationale: str = ""

    def to_dict(self) -> dict:
        return {
            "chart_type": self.chart_type,
            "title": self.title,
            "x_axis": self.x_axis,
            "y_axis": self.y_axis,
            "color_by": self.color_by,
            "rationale": self.rationale,
        }


# ─── 6.1 Chart recommender ────────────────────────────────────────────────────

def recommend_chart(
    result: dict,
    method: str = "",
    intent_type: str = "descriptive",
) -> Optional[ChartSpec]:
    """
    Rule-based chart type selection driven by the analysis result structure.

    Priority rules:
      - time_series result  → line chart
      - correlation matrix  → heatmap
      - anomaly result      → box plot
      - comparison result   → bar / ANOVA bar
      - groupby rows        → bar (horizontal if > 6 categories)
      - regression result   → scatter
      - distribution data   → histogram
    """
    if not result:
        return None

    # Time-series
    if "trend" in result and "n_periods" in result:
        return ChartSpec(
            chart_type="line",
            title=f"Trend: {result.get('value_col', 'Value')} over time",
            x_axis=result.get("date_col"),
            y_axis=result.get("value_col"),
            rationale="Time-series data → line chart",
        )

    # Correlation matrix
    if "matrix" in result and "method" in result:
        return ChartSpec(
            chart_type="heatmap",
            title=f"Correlation Matrix ({result.get('method', 'pearson')})",
            matrix=result.get("matrix"),
            rationale="Correlation matrix → heatmap",
        )

    # Anomaly detection
    if "n_outliers" in result and "method" in result:
        return ChartSpec(
            chart_type="box",
            title=f"Outlier Detection: {result.get('column', 'value')}",
            y_axis=result.get("column"),
            rationale="Anomaly result → box plot",
        )

    # Segment comparison
    if "segments" in result and "top_segment" in result:
        segments = result.get("segments", [])
        n = len(segments)
        chart_type = "horizontal_bar" if n > 6 else "bar"
        return ChartSpec(
            chart_type=chart_type,
            title=f"{result.get('value_col', 'Value')} by {result.get('segment_col', 'Segment')}",
            x_axis=result.get("segment_col"),
            y_axis=result.get("value_col"),
            data=segments,
            rationale=f"Segment comparison → {'horizontal ' if n > 6 else ''}bar chart",
        )

    # Grouped list of dicts (groupby result)
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
        cols = list(result[0].keys())
        cat_cols = [c for c in cols if not _is_numeric_col(result, c)]
        num_cols = [c for c in cols if _is_numeric_col(result, c)]

        if cat_cols and num_cols:
            n = len(result)
            chart_type = "horizontal_bar" if n > 6 else "bar"
            return ChartSpec(
                chart_type=chart_type,
                title=f"{num_cols[0]} by {cat_cols[0]}",
                x_axis=cat_cols[0],
                y_axis=num_cols[0],
                data=result,
                rationale=f"Grouped data → {'horizontal ' if n > 6 else ''}bar chart",
            )
        if len(num_cols) >= 2:
            return ChartSpec(
                chart_type="scatter",
                title=f"{num_cols[0]} vs {num_cols[1]}",
                x_axis=num_cols[0],
                y_axis=num_cols[1],
                data=result,
                rationale="Two numeric columns → scatter plot",
            )

    # Regression
    if "r_squared" in result and "coefficients" in result:
        features = list(result.get("coefficients", {}).keys())
        return ChartSpec(
            chart_type="scatter",
            title=f"Regression: {result.get('target', 'y')} ~ {', '.join(features[:2])}",
            x_axis=features[0] if features else None,
            y_axis=result.get("target"),
            rationale="Regression result → scatter plot with fit line",
        )

    return None


def _is_numeric_col(rows: list[dict], col: str) -> bool:
    for row in rows:
        v = row.get(col)
        if v is not None:
            return isinstance(v, (int, float))
    return False


# ─── 6.2 Static chart generators (matplotlib/seaborn) ────────────────────────

def generate_static_chart(
    spec: ChartSpec,
    df: Optional[pd.DataFrame] = None,
    output_dir: str = "./data/charts",
    filename: Optional[str] = None,
) -> str:
    """
    Generate a static PNG chart and return the file path.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    safe_title = "".join(c if c.isalnum() or c in "-_ " else "_" for c in spec.title)[:50]
    fname = filename or f"{spec.chart_type}_{safe_title.replace(' ', '_')}.png"
    out_path = str(Path(output_dir) / fname)

    fig, ax = plt.subplots(figsize=(9, 5))

    try:
        if spec.chart_type == "bar":
            _draw_bar(ax, spec, df)
        elif spec.chart_type == "horizontal_bar":
            _draw_horizontal_bar(ax, spec, df)
        elif spec.chart_type == "line":
            _draw_line(ax, spec, df)
        elif spec.chart_type == "scatter":
            _draw_scatter(ax, spec, df)
        elif spec.chart_type == "histogram":
            _draw_histogram(ax, spec, df)
        elif spec.chart_type == "box":
            _draw_box(ax, spec, df)
        elif spec.chart_type == "heatmap":
            plt.close(fig)
            fig = _draw_heatmap(spec)
        elif spec.chart_type == "pie":
            _draw_pie(ax, spec, df)
        else:
            ax.text(0.5, 0.5, f"Chart type '{spec.chart_type}' not yet implemented",
                    ha="center", va="center", transform=ax.transAxes)
    except Exception as exc:
        logger.warning("Chart generation failed for %s: %s", spec.chart_type, exc)
        ax.text(0.5, 0.5, f"Error: {exc}", ha="center", va="center", transform=ax.transAxes)

    fig.suptitle(spec.title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    logger.debug("Saved chart: %s", out_path)
    return out_path


def _df_from_spec(spec: ChartSpec, df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if spec.data:
        return pd.DataFrame(spec.data)
    if df is not None:
        return df
    return pd.DataFrame()


def _draw_bar(ax, spec: ChartSpec, df: Optional[pd.DataFrame]) -> None:
    data = _df_from_spec(spec, df)
    if data.empty or not spec.x_axis or not spec.y_axis:
        return
    x, y = str(spec.x_axis), str(spec.y_axis)
    ax.bar(data[x].astype(str), data[y], color=MAEDA_PALETTE[0], edgecolor="white", linewidth=0.5)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.xticks(rotation=30, ha="right")


def _draw_horizontal_bar(ax, spec: ChartSpec, df: Optional[pd.DataFrame]) -> None:
    data = _df_from_spec(spec, df)
    if data.empty or not spec.x_axis or not spec.y_axis:
        return
    x, y = str(spec.x_axis), str(spec.y_axis)
    sorted_data = data.sort_values(y)
    ax.barh(sorted_data[x].astype(str), sorted_data[y],
            color=MAEDA_PALETTE[0], edgecolor="white", linewidth=0.5)
    ax.set_xlabel(y)
    ax.set_ylabel(x)


def _draw_line(ax, spec: ChartSpec, df: Optional[pd.DataFrame]) -> None:
    data = _df_from_spec(spec, df)
    if data.empty or not spec.x_axis or not spec.y_axis:
        return
    x, y = str(spec.x_axis), str(spec.y_axis)
    ax.plot(data[x].astype(str), data[y], color=MAEDA_PALETTE[0],
            marker="o", linewidth=2, markersize=5)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.xticks(rotation=30, ha="right")


def _draw_scatter(ax, spec: ChartSpec, df: Optional[pd.DataFrame]) -> None:
    data = _df_from_spec(spec, df)
    if data.empty or not spec.x_axis or not spec.y_axis:
        return
    x, y = str(spec.x_axis), str(spec.y_axis)
    ax.scatter(data[x], data[y], color=MAEDA_PALETTE[0], alpha=0.7, edgecolors="white", s=60)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    # Add trend line
    try:
        z = np.polyfit(data[x].astype(float), data[y].astype(float), 1)
        p = np.poly1d(z)
        ax.plot(sorted(data[x]), p(sorted(data[x])), "--", color=MAEDA_PALETTE[1], linewidth=1.5)
    except Exception:
        pass


def _draw_histogram(ax, spec: ChartSpec, df: Optional[pd.DataFrame]) -> None:
    data = _df_from_spec(spec, df)
    if data.empty:
        return
    col = spec.y_axis or (data.select_dtypes(include="number").columns[0] if not data.empty else None)
    if col and str(col) in data.columns:
        ax.hist(data[str(col)].dropna(), bins=20, color=MAEDA_PALETTE[0],
                edgecolor="white", linewidth=0.5)
        ax.set_xlabel(str(col))
        ax.set_ylabel("Frequency")


def _draw_box(ax, spec: ChartSpec, df: Optional[pd.DataFrame]) -> None:
    data = _df_from_spec(spec, df)
    col = str(spec.y_axis) if spec.y_axis else None
    if data.empty or not col or col not in data.columns:
        if df is not None and col and col in df.columns:
            data = df
        else:
            return
    ax.boxplot(data[col].dropna(), patch_artist=True,
               boxprops=dict(facecolor=MAEDA_PALETTE[0], alpha=0.6),
               medianprops=dict(color=MAEDA_PALETTE[1], linewidth=2))
    ax.set_ylabel(col)


def _draw_heatmap(spec: ChartSpec):
    matrix = spec.matrix or {}
    if not matrix:
        fig, ax = plt.subplots(figsize=(6, 5))
        return fig
    df_m = pd.DataFrame(matrix)
    fig, ax = plt.subplots(figsize=(max(6, len(df_m) + 1), max(5, len(df_m))))
    sns.heatmap(df_m, annot=True, fmt=".2f", cmap="Blues",
                linewidths=0.5, ax=ax, vmin=-1, vmax=1,
                annot_kws={"size": 9})
    return fig


def _draw_pie(ax, spec: ChartSpec, df: Optional[pd.DataFrame]) -> None:
    data = _df_from_spec(spec, df)
    if data.empty or not spec.x_axis or not spec.y_axis:
        return
    x, y = str(spec.x_axis), str(spec.y_axis)
    slices = data.groupby(x)[y].sum()
    ax.pie(slices.values, labels=slices.index, autopct="%1.1f%%",
           colors=MAEDA_PALETTE[:len(slices)], startangle=90)
    ax.axis("equal")


# ─── 6.3 Interactive charts (plotly) ─────────────────────────────────────────

def generate_interactive_chart(
    spec: ChartSpec,
    df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Generate a Plotly figure and return it as a dict with 'json' (serialised)
    and 'figure' (live Figure object) keys.
    """
    data = _df_from_spec(spec, df)
    fig = None

    try:
        if spec.chart_type in {"bar", "horizontal_bar"}:
            orientation = "h" if spec.chart_type == "horizontal_bar" else "v"
            fig = px.bar(
                data, x=spec.x_axis, y=spec.y_axis,
                color=spec.color_by,
                orientation=orientation,
                title=spec.title,
                color_discrete_sequence=MAEDA_PLOTLY_COLORS,
            )
        elif spec.chart_type == "line":
            fig = px.line(
                data, x=spec.x_axis, y=spec.y_axis,
                title=spec.title, markers=True,
                color_discrete_sequence=MAEDA_PLOTLY_COLORS,
            )
        elif spec.chart_type == "scatter":
            fig = px.scatter(
                data, x=spec.x_axis, y=spec.y_axis,
                color=spec.color_by, title=spec.title,
                trendline="ols" if spec.color_by is None else None,
                color_discrete_sequence=MAEDA_PLOTLY_COLORS,
            )
        elif spec.chart_type == "histogram":
            col = str(spec.y_axis) if spec.y_axis else data.select_dtypes("number").columns[0]
            fig = px.histogram(data, x=col, title=spec.title,
                                color_discrete_sequence=MAEDA_PLOTLY_COLORS)
        elif spec.chart_type == "box":
            col = str(spec.y_axis) if spec.y_axis else data.select_dtypes("number").columns[0]
            fig = px.box(data, y=col, title=spec.title,
                          color_discrete_sequence=MAEDA_PLOTLY_COLORS)
        elif spec.chart_type == "heatmap":
            matrix = spec.matrix or {}
            df_m = pd.DataFrame(matrix)
            fig = go.Figure(go.Heatmap(
                z=df_m.values,
                x=list(df_m.columns),
                y=list(df_m.index),
                colorscale="Blues",
                zmin=-1, zmax=1,
                text=df_m.round(2).values,
                texttemplate="%{text}",
            ))
            fig.update_layout(title=spec.title)
        elif spec.chart_type == "pie":
            if not data.empty and spec.x_axis and spec.y_axis:
                fig = px.pie(data, names=spec.x_axis, values=spec.y_axis,
                              title=spec.title,
                              color_discrete_sequence=MAEDA_PLOTLY_COLORS)
        else:
            fig = go.Figure()
            fig.update_layout(title=f"{spec.title} (not implemented)")

        if fig is None:
            fig = go.Figure()
            fig.update_layout(title=spec.title)

        _apply_plotly_theme(fig)
        fig_json = fig.to_json()

    except Exception as exc:
        logger.warning("Plotly chart failed for %s: %s", spec.chart_type, exc)
        fig = go.Figure()
        fig.update_layout(title=f"Error: {exc}")
        fig_json = fig.to_json()

    return {"figure": fig, "json": fig_json, "spec": spec.to_dict()}


def _apply_plotly_theme(fig: go.Figure) -> None:
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, system-ui, sans-serif", size=12),
        title_font_size=14,
        title_font_color="#1E293B",
        plot_bgcolor="#F8FAFC",
        paper_bgcolor="white",
        margin=dict(l=60, r=20, t=60, b=60),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E2E8F0")
    fig.update_yaxes(showgrid=True, gridcolor="#E2E8F0")


# ─── 6.4 Multi-chart layout ───────────────────────────────────────────────────

def generate_dashboard(
    specs: list[ChartSpec],
    dfs: Optional[list[Optional[pd.DataFrame]]] = None,
    output_dir: str = "./data/charts",
    filename: str = "dashboard.png",
) -> str:
    """
    Render multiple charts as a matplotlib subplot grid.
    Returns the path to the saved PNG.
    """
    n = len(specs)
    if n == 0:
        return ""

    dfs = dfs or [None] * n
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 4.5))

    # Normalise axes to always be a flat list
    if n == 1:
        axes_flat = [axes]
    elif rows == 1:
        axes_flat = list(axes)
    else:
        axes_flat = [ax for row in axes for ax in row]

    for i, (spec, df_i) in enumerate(zip(specs, dfs)):
        ax = axes_flat[i]
        try:
            if spec.chart_type == "bar":
                _draw_bar(ax, spec, df_i)
            elif spec.chart_type == "horizontal_bar":
                _draw_horizontal_bar(ax, spec, df_i)
            elif spec.chart_type == "line":
                _draw_line(ax, spec, df_i)
            elif spec.chart_type == "scatter":
                _draw_scatter(ax, spec, df_i)
            elif spec.chart_type == "histogram":
                _draw_histogram(ax, spec, df_i)
            elif spec.chart_type == "box":
                _draw_box(ax, spec, df_i)
            else:
                ax.text(0.5, 0.5, spec.chart_type, ha="center", va="center",
                        transform=ax.transAxes)
            ax.set_title(spec.title, fontsize=10, fontweight="bold")
        except Exception as exc:
            ax.text(0.5, 0.5, str(exc), ha="center", va="center", transform=ax.transAxes)

    # Hide unused subplots
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("MAEDA Analysis Dashboard", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(output_dir) / filename)
    fig.savefig(out_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    return out_path
