"""
Statistical, anomaly detection, time-series, and comparison tools.

All functions operate on pandas DataFrames and return structured result dicts
compatible with the step executor.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import get_logger

logger = get_logger("maeda.tools.stats")


# ─── Statistical tests ────────────────────────────────────────────────────────

def compute_correlation(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    method: str = "pearson",
) -> dict:
    """
    Compute a correlation matrix for numeric columns.
    method: 'pearson' | 'spearman' | 'kendall'
    """
    num_df = df.select_dtypes(include="number")
    if columns:
        num_df = num_df[[c for c in columns if c in num_df.columns]]
    corr = num_df.corr(method=method)
    # Find top pairs (abs corr > 0.5, excluding diagonal)
    strong_pairs = []
    for i, row in enumerate(corr.columns):
        for j, col in enumerate(corr.columns):
            if i < j:
                val = corr.iloc[i, j]
                if abs(val) > 0.5:
                    strong_pairs.append({
                        "col_a": row, "col_b": col,
                        "correlation": round(float(val), 4),
                        "strength": "strong" if abs(val) > 0.7 else "moderate",
                    })
    return {
        "matrix": corr.round(4).to_dict(),
        "strong_pairs": strong_pairs,
        "method": method,
        "columns_used": list(corr.columns),
    }


def run_linear_regression(
    df: pd.DataFrame,
    target: str,
    features: list[str],
) -> dict:
    """OLS linear regression via scipy. Returns coefficients, R², p-values."""
    from scipy.stats import pearsonr
    clean = df[[target] + features].dropna()
    y = clean[target].values
    X = clean[features].values

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Use numpy lstsq for multi-feature case
    X_with_const = np.column_stack([np.ones(len(X)), X])
    coeffs, _, _, _ = np.linalg.lstsq(X_with_const, y, rcond=None)
    y_pred = X_with_const @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    result = {
        "intercept": round(float(coeffs[0]), 6),
        "coefficients": {f: round(float(c), 6) for f, c in zip(features, coeffs[1:])},
        "r_squared": round(float(r2), 4),
        "n_observations": len(clean),
        "target": target,
        "features": features,
    }

    # For single-feature case, also include correlation and p-value
    if len(features) == 1:
        corr_val, p_val = pearsonr(clean[features[0]], clean[target])
        result["pearson_r"] = round(float(corr_val), 4)
        result["p_value"] = round(float(p_val), 6)

    return result


def run_ttest(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
) -> dict:
    """Independent-samples t-test between the first two groups in group_col."""
    groups = df[group_col].dropna().unique()
    if len(groups) < 2:
        return {"error": f"Need ≥2 groups in '{group_col}'; found {len(groups)}"}

    g1, g2 = groups[0], groups[1]
    a = df[df[group_col] == g1][value_col].dropna()
    b = df[df[group_col] == g2][value_col].dropna()
    t_stat, p_val = stats.ttest_ind(a, b)

    return {
        "group_a": str(g1), "mean_a": round(float(a.mean()), 4), "n_a": len(a),
        "group_b": str(g2), "mean_b": round(float(b.mean()), 4), "n_b": len(b),
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_val), 6),
        "significant": bool(p_val < 0.05),
        "interpretation": (
            f"Significant difference between {g1} and {g2} (p={p_val:.4f})"
            if p_val < 0.05
            else f"No significant difference between {g1} and {g2} (p={p_val:.4f})"
        ),
    }


def run_chi_square(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
) -> dict:
    """Chi-square test of independence between two categorical columns."""
    contingency = pd.crosstab(df[col_a], df[col_b])
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    return {
        "chi2_statistic": round(float(chi2), 4),
        "p_value": round(float(p_val), 6),
        "degrees_of_freedom": int(dof),
        "significant": bool(p_val < 0.05),
        "contingency_table": contingency.to_dict(),
    }


# ─── Anomaly detection ────────────────────────────────────────────────────────

def detect_anomalies_zscore(
    df: pd.DataFrame,
    column: str,
    threshold: float = 3.0,
) -> dict:
    """Flag outliers using Z-score (|z| > threshold)."""
    series = df[column].dropna()
    z_scores = np.abs(stats.zscore(series))
    outlier_mask = z_scores > threshold
    outlier_indices = series[outlier_mask].index.tolist()
    outlier_values = series[outlier_mask].tolist()
    return {
        "method": "zscore",
        "column": column,
        "threshold": threshold,
        "n_outliers": int(outlier_mask.sum()),
        "outlier_pct": round(float(outlier_mask.mean()), 4),
        "outlier_indices": outlier_indices[:20],  # cap for state size
        "outlier_values": [round(float(v), 4) for v in outlier_values[:20]],
        "mean": round(float(series.mean()), 4),
        "std": round(float(series.std()), 4),
    }


def detect_anomalies_iqr(
    df: pd.DataFrame,
    column: str,
    multiplier: float = 1.5,
) -> dict:
    """Flag outliers using the IQR fence method."""
    series = df[column].dropna()
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
    outlier_mask = (series < lower) | (series > upper)
    return {
        "method": "iqr",
        "column": column,
        "multiplier": multiplier,
        "q1": round(float(q1), 4),
        "q3": round(float(q3), 4),
        "iqr": round(float(iqr), 4),
        "lower_fence": round(float(lower), 4),
        "upper_fence": round(float(upper), 4),
        "n_outliers": int(outlier_mask.sum()),
        "outlier_pct": round(float(outlier_mask.mean()), 4),
        "outlier_values": [round(float(v), 4) for v in series[outlier_mask].tolist()[:20]],
    }


def detect_anomalies_isolation_forest(
    df: pd.DataFrame,
    columns: list[str],
    contamination: float = 0.05,
) -> dict:
    """Isolation Forest multi-variate anomaly detection."""
    from sklearn.ensemble import IsolationForest

    sub = df[columns].dropna()
    clf = IsolationForest(contamination=contamination, random_state=42)
    preds = clf.fit_predict(sub)
    anomaly_mask = preds == -1
    return {
        "method": "isolation_forest",
        "columns": columns,
        "contamination": contamination,
        "n_anomalies": int(anomaly_mask.sum()),
        "anomaly_pct": round(float(anomaly_mask.mean()), 4),
        "anomaly_indices": sub.index[anomaly_mask].tolist()[:20],
    }


# ─── Time-series analysis ─────────────────────────────────────────────────────

def analyze_time_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    freq: Optional[str] = None,
    forecast_periods: int = 0,
) -> dict:
    """
    Basic time-series analysis: trend (linear fit), rolling stats, and
    simple period-over-period growth.

    If forecast_periods > 0, also extrapolates the linear trend forward —
    this is the full extent of MAEDA's forecasting capability. There is no
    trained predictive model; callers/report writers must describe this as
    a simple trend projection, not a prediction.
    """
    ts = df[[date_col, value_col]].copy().dropna()
    ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
    ts = ts.dropna(subset=[date_col]).sort_values(date_col)

    if len(ts) < 3:
        return {"error": "Need at least 3 data points for time-series analysis"}

    values = ts[value_col].values.astype(float)
    n = len(values)
    x = np.arange(n)

    # Linear trend
    slope, intercept, r_val, p_val, _ = stats.linregress(x, values)
    trend_direction = "increasing" if slope > 0 else "decreasing"

    # Period-over-period growth (last vs first half)
    mid = n // 2
    first_half_mean = float(np.mean(values[:mid]))
    second_half_mean = float(np.mean(values[mid:]))
    pop_growth = (
        (second_half_mean - first_half_mean) / abs(first_half_mean)
        if first_half_mean != 0 else 0.0
    )

    # Rolling 3-period average (for smoothing)
    rolling_avg = ts[value_col].rolling(min(3, n)).mean().tolist()

    result = {
        "date_col": date_col,
        "value_col": value_col,
        "n_periods": n,
        "trend": {
            "direction": trend_direction,
            "slope": round(float(slope), 6),
            "r_squared": round(float(r_val ** 2), 4),
            "p_value": round(float(p_val), 6),
            "significant": bool(p_val < 0.05),
        },
        "summary_stats": {
            "min": round(float(values.min()), 4),
            "max": round(float(values.max()), 4),
            "mean": round(float(values.mean()), 4),
            "std": round(float(values.std()), 4),
        },
        "period_over_period_growth_pct": round(float(pop_growth * 100), 2),
        "rolling_avg_3": [round(v, 4) if not np.isnan(v) else None for v in rolling_avg],
    }

    if forecast_periods > 0:
        last_date = ts[date_col].iloc[-1]
        step = ts[date_col].diff().median()  # typical spacing between observations
        forecast = []
        for i in range(1, forecast_periods + 1):
            forecast.append({
                "period_index": n + i - 1,
                "date": (last_date + step * i).isoformat() if pd.notna(step) else None,
                "predicted_value": round(float(intercept + slope * (n + i - 1)), 4),
            })
        result["forecast"] = {
            "method": "linear_trend_extrapolation",
            "periods": forecast,
            "caveat": (
                "This is a naive linear extrapolation of the historical trend, "
                "not a trained predictive model. Treat as a rough directional "
                "estimate, not a precise forecast."
            ),
        }

    return result


# ─── Segment comparison ───────────────────────────────────────────────────────

def compare_segments(
    df: pd.DataFrame,
    segment_col: str,
    value_col: str,
    agg: str = "mean",
) -> dict:
    """
    Compare metric across segments. Returns per-segment stats plus
    ANOVA test if ≥3 groups, t-test if 2 groups.
    """
    agg_fn = {"mean": "mean", "sum": "sum", "median": "median", "count": "count"}.get(agg, "mean")
    grouped = df.groupby(segment_col)[value_col].agg(["mean", "sum", "median", "count", "std"])
    segments = grouped.reset_index().to_dict(orient="records")

    groups = [g[value_col].dropna().values for _, g in df.groupby(segment_col)]
    significance = {}
    if len(groups) == 2:
        t, p = stats.ttest_ind(*groups)
        significance = {"test": "t-test", "statistic": round(float(t), 4),
                        "p_value": round(float(p), 6), "significant": bool(p < 0.05)}
    elif len(groups) >= 3:
        f, p = stats.f_oneway(*groups)
        significance = {"test": "one-way ANOVA", "statistic": round(float(f), 4),
                        "p_value": round(float(p), 6), "significant": bool(p < 0.05)}

    top_segment = grouped["mean"].idxmax()
    return {
        "segment_col": segment_col,
        "value_col": value_col,
        "aggregation": agg,
        "segments": segments,
        "top_segment": str(top_segment),
        "significance_test": significance,
    }


# ─── Pandas transforms ────────────────────────────────────────────────────────

def pandas_groupby(
    df: pd.DataFrame,
    group_by: list[str],
    agg_col: str,
    agg_func: str = "sum",
    sort_desc: bool = True,
) -> dict:
    """GroupBy aggregation."""
    valid_funcs = {"sum", "mean", "count", "median", "min", "max", "std"}
    func = agg_func if agg_func in valid_funcs else "sum"
    result = df.groupby(group_by)[agg_col].agg(func).reset_index()
    if sort_desc:
        result = result.sort_values(agg_col, ascending=False)
    return {
        "result": result.to_dict(orient="records"),
        "result_df": result,
        "n_groups": len(result),
        "agg_func": func,
    }


def pandas_pivot(
    df: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    agg_func: str = "sum",
) -> dict:
    """Pivot table."""
    pivot = df.pivot_table(index=index, columns=columns, values=values, aggfunc=agg_func)
    return {
        "result": pivot.to_dict(),
        "result_df": pivot.reset_index(),
        "shape": list(pivot.shape),
    }


_DERIVE_OPS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
}


def pandas_derive(
    df: pd.DataFrame,
    new_column: str,
    left: str,
    op: str,
    right: "str | float | int",
) -> dict:
    """
    Compute a new column as a simple binary arithmetic expression over an
    existing column and either another column or a constant, e.g.
    new_column="margin", left="unit_price", op="-", right="cost".

    Deliberately restricted to +,-,*,/ between two operands (no eval() of
    arbitrary expressions) — this is the only derived-column capability
    the planner may rely on.
    """
    if op not in _DERIVE_OPS:
        raise ValueError(f"derive: unsupported op {op!r}, must be one of {sorted(_DERIVE_OPS)}")
    left_series = df[left]
    right_val = df[right] if isinstance(right, str) and right in df.columns else right
    result = df.copy()
    result[new_column] = _DERIVE_OPS[op](left_series, right_val)
    return {
        "result": result.to_dict(orient="records"),
        "result_df": result,
        "new_column": new_column,
    }


_FILTER_OPS = {
    "==": lambda s, v: s == v,
    "!=": lambda s, v: s != v,
    ">": lambda s, v: s > v,
    ">=": lambda s, v: s >= v,
    "<": lambda s, v: s < v,
    "<=": lambda s, v: s <= v,
    "in": lambda s, v: s.isin(v),
    "not_in": lambda s, v: ~s.isin(v),
    "contains": lambda s, v: s.astype(str).str.contains(str(v), na=False),
}


def pandas_filter(
    df: pd.DataFrame,
    filters: list[dict],
) -> dict:
    """
    Apply a list of filter conditions.
    Each filter: {"column": str, "op": str, "value": any}
    ops: ==, !=, >, >=, <, <=, in, not_in, contains

    Column-existence is validated by the caller (pandas_tool); an
    unrecognized "op" (e.g. "=" instead of "==") raises here rather than
    silently applying no condition — a single-equals filter that never
    fires is easy to miss and leaves downstream steps operating on
    unfiltered data.
    """
    result = df.copy()
    for f in filters:
        col, op, val = f.get("column"), f.get("op"), f.get("value")
        if col not in result.columns:
            continue
        if op not in _FILTER_OPS:
            raise ValueError(
                f"filter: unsupported op {op!r} for column {col!r}. "
                f"Must be one of {sorted(_FILTER_OPS)}."
            )
        result = result[_FILTER_OPS[op](result[col], val)]
    return {"result": result.to_dict(orient="records"), "result_df": result, "row_count": len(result)}


# ─── Tool dispatcher (used by step executor) ─────────────────────────────────

# Parameter keys recognized by each operation. Used both to infer the intended
# operation when the planner omits "operation", and to reject parameter sets
# that don't actually belong to any operation this tool understands.
_GROUPBY_KEYS = {"group_by", "agg_col", "sort_desc"}
_PIVOT_KEYS = {"index", "columns", "values"}
_FILTER_KEYS = {"filters"}
_DERIVE_KEYS = {"new_column", "left", "op", "right"}


def _infer_operation(parameters: dict) -> Optional[str]:
    keys = set(parameters.keys())
    if keys & _FILTER_KEYS:
        return "filter"
    if keys & _DERIVE_KEYS:
        return "derive"
    if keys & _PIVOT_KEYS:
        return "pivot"
    if keys & _GROUPBY_KEYS:
        return "groupby"
    return None


def _require_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{context}: column(s) {missing} not found in dataframe. "
            f"Available columns: {list(df.columns)}"
        )


def pandas_tool(df: pd.DataFrame, parameters: dict, prior_results: dict) -> dict:
    op = parameters.get("operation") or _infer_operation(parameters)
    if not op:
        raise ValueError(
            f"pandas_transform: cannot determine an operation from parameters {parameters!r}. "
            f"Provide an 'operation' key ('groupby'|'pivot'|'filter'|'derive') or one of the "
            f"recognized keys for that operation "
            f"(groupby: {_GROUPBY_KEYS}, pivot: {_PIVOT_KEYS}, filter: {_FILTER_KEYS}, "
            f"derive: {_DERIVE_KEYS})."
        )

    if op == "derive":
        missing_keys = [k for k in ("new_column", "left", "op", "right") if k not in parameters]
        if missing_keys:
            raise ValueError(f"derive: missing required parameter(s) {missing_keys}")
        left, right = parameters["left"], parameters["right"]
        cols_to_check = [left] + ([right] if isinstance(right, str) else [])
        _require_columns(df, cols_to_check, "derive")
        r = pandas_derive(df, parameters["new_column"], left, parameters["op"], right)
    elif op == "groupby":
        group_by = parameters.get("group_by")
        agg_col = parameters.get("agg_col")
        if group_by is not None:
            group_by = group_by if isinstance(group_by, list) else [group_by]
            _require_columns(df, group_by, "groupby: group_by")
        else:
            # Infer: first categorical column
            cats = [c for c in df.columns if df[c].dtype == object]
            group_by = cats[:1] if cats else [df.columns[0]]
        if agg_col is not None:
            _require_columns(df, [agg_col], "groupby: agg_col")
        else:
            # Infer: first numeric column not in group_by
            nums = [c for c in df.select_dtypes(include="number").columns if c not in group_by]
            agg_col = nums[0] if nums else group_by[0]
        r = pandas_groupby(df, group_by, agg_col,
                           parameters.get("agg_func", "sum"),
                           parameters.get("sort_desc", True))
    elif op == "pivot":
        missing_keys = [k for k in ("index", "columns", "values") if k not in parameters]
        if missing_keys:
            raise ValueError(f"pivot: missing required parameter(s) {missing_keys}")
        _require_columns(
            df, [parameters["index"], parameters["columns"], parameters["values"]], "pivot"
        )
        r = pandas_pivot(df, parameters["index"], parameters["columns"],
                         parameters["values"], parameters.get("agg_func", "sum"))
    elif op == "filter":
        filters = parameters.get("filters", [])
        _require_columns(df, [f.get("column") for f in filters if f.get("column")], "filter")
        r = pandas_filter(df, filters)
    else:
        raise ValueError(f"Unknown pandas operation: {op!r}")

    rows = r["result"] if isinstance(r["result"], list) else []
    if rows:
        top = rows[:5]
        row_strs = [", ".join(f"{k}={v}" for k, v in rec.items()) for rec in top]
        data_preview = "; ".join(row_strs)
        summary = f"pandas/{op} → {len(rows)} rows. Top results: {data_preview}"
    else:
        summary = f"pandas/{op} → {len(rows)} rows"
    return {
        "result": r["result"],
        "result_df": r.get("result_df"),
        "result_summary": summary,
        "warnings": [],
    }


def statistical_tool(df: pd.DataFrame, parameters: dict, prior_results: dict) -> dict:
    test = parameters.get("test", "correlation")
    if test == "correlation":
        r = compute_correlation(df, parameters.get("columns"), parameters.get("method", "pearson"))
    elif test == "regression":
        r = run_linear_regression(df, parameters["target"], parameters["features"])
    elif test == "ttest":
        r = run_ttest(df, parameters["group_col"], parameters["value_col"])
    elif test == "chi_square":
        r = run_chi_square(df, parameters["col_a"], parameters["col_b"])
    else:
        raise ValueError(f"Unknown statistical test: {test!r}")
    return {"result": r, "result_df": None, "result_summary": f"stats/{test} complete", "warnings": []}


def anomaly_tool(df: pd.DataFrame, parameters: dict, prior_results: dict) -> dict:
    method = parameters.get("method", "iqr")
    col = parameters.get("column")
    if method == "zscore":
        r = detect_anomalies_zscore(df, col, parameters.get("threshold", 3.0))
    elif method == "iqr":
        r = detect_anomalies_iqr(df, col, parameters.get("multiplier", 1.5))
    elif method == "isolation_forest":
        r = detect_anomalies_isolation_forest(df, parameters.get("columns", [col]),
                                              parameters.get("contamination", 0.05))
    else:
        raise ValueError(f"Unknown anomaly method: {method!r}")
    return {"result": r, "result_df": None,
            "result_summary": f"anomaly/{method}: {r.get('n_outliers', '?')} outliers", "warnings": []}


def timeseries_tool(df: pd.DataFrame, parameters: dict, prior_results: dict) -> dict:
    r = analyze_time_series(df, parameters["date_col"], parameters["value_col"],
                            parameters.get("freq"),
                            parameters.get("forecast_periods", 0))
    summary = f"timeseries: {r.get('trend', {}).get('direction', '?')} trend"
    if r.get("forecast"):
        summary += f" | naive linear forecast for {len(r['forecast']['periods'])} period(s)"
    return {"result": r, "result_df": None, "result_summary": summary, "warnings": []}


def comparison_tool(df: pd.DataFrame, parameters: dict, prior_results: dict) -> dict:
    r = compare_segments(df, parameters["segment_col"], parameters["value_col"],
                         parameters.get("agg", "mean"))
    return {"result": r, "result_df": None,
            "result_summary": f"comparison: top={r.get('top_segment', '?')}", "warnings": []}
