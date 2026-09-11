"""
eda_engine.py – Plan validation, KPI computation, and chart building.

This is the **trust boundary**: every column name and aggregation that comes
from Grok is validated against the real DataFrame before any computation.
No LLM-generated number is ever displayed — every value is computed here.

Enhancements over the base version:
  - Smart date handling with auto-resample for line/bar charts
  - Auto-aggregation for grouped bar charts
  - Treemap, sunburst, and funnel chart types
  - KPI delta indicators (mean vs median comparison)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# Allowed chart types (Plotly Express mapping)
ALLOWED_CHART_TYPES = {
    "bar", "line", "scatter", "histogram", "box", "pie",
    "heatmap", "treemap", "sunburst", "funnel",
}

# Allowed aggregation functions
ALLOWED_AGGS = {"mean", "sum", "count", "max", "min", "median", "nunique", "std"}

# Display-format helpers
FORMAT_FUNCS = {
    "number": lambda v: f"{v:,.2f}" if v is not None else "N/A",
    "currency": lambda v: f"${v:,.2f}" if v is not None else "N/A",
    "percent": lambda v: f"{v:.2f}%" if v is not None else "N/A",
    "integer": lambda v: f"{int(v):,}" if v is not None else "N/A",
}

# Max unique values before auto-aggregation kicks in
_HIGH_CARDINALITY_THRESHOLD = 30


# ── Plan sanitisation ───────────────────────────────────────────────────────

def sanitize_plan(plan: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    """
    Validate and clean an analysis plan returned by Grok.

    - Drops KPIs referencing non-existent columns or unsupported aggs.
    - Drops charts with invalid types or non-existent columns.
    - Ensures the plan always has the three required keys.
    - Returns a *new* dict (never mutates the input).
    """
    valid_cols = set(df.columns)

    # ── KPIs ─────────────────────────────────────────────────────────────
    clean_kpis: list[dict[str, Any]] = []
    for kpi in plan.get("kpis", []):
        col = kpi.get("column")
        agg = kpi.get("agg", "").lower()
        if col not in valid_cols:
            logger.warning("KPI dropped – column '%s' not in dataframe", col)
            continue
        if agg not in ALLOWED_AGGS:
            logger.warning("KPI dropped – unsupported agg '%s'", agg)
            continue
        clean_kpis.append(kpi)

    # ── Charts ───────────────────────────────────────────────────────────
    clean_charts: list[dict[str, Any]] = []
    for chart in plan.get("charts", []):
        ctype = (chart.get("type") or "").lower()
        if ctype not in ALLOWED_CHART_TYPES:
            logger.warning("Chart dropped – unsupported type '%s'", ctype)
            continue

        # Validate referenced columns
        cols_to_check = ["x", "y", "color"]
        skip = False
        for key in cols_to_check:
            val = chart.get(key)
            if val is not None and val not in valid_cols:
                logger.warning(
                    "Chart dropped – column '%s' (key=%s) not in dataframe", val, key
                )
                skip = True
                break
        if skip:
            continue

        # Ensure 'x' is present for chart types that need it
        needs_x = {"bar", "line", "scatter", "histogram", "box", "pie", "funnel"}
        if ctype in needs_x and chart.get("x") is None:
            logger.warning("Chart dropped – 'x' is required for type '%s'", ctype)
            continue

        chart["type"] = ctype  # normalise to lowercase
        clean_charts.append(chart)

    # ── Insights ─────────────────────────────────────────────────────────
    raw_insights = plan.get("insights", [])
    clean_insights = [s for s in raw_insights if isinstance(s, str) and s.strip()]

    sanitized = {
        "kpis": clean_kpis,
        "charts": clean_charts,
        "insights": clean_insights,
    }
    logger.info(
        "Plan sanitised: %d KPIs, %d charts, %d insights",
        len(clean_kpis),
        len(clean_charts),
        len(clean_insights),
    )
    return sanitized


# ── KPI computation ──────────────────────────────────────────────────────────

def compute_kpis(
    kpis: list[dict[str, Any]],
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Compute every KPI value from the **real** DataFrame.

    Returns a list of dicts, each with:
      name, value (raw float/int), formatted (display string),
      delta, delta_label (for comparison indicators).
    """
    results: list[dict[str, Any]] = []
    for kpi in kpis:
        col = kpi["column"]
        agg = kpi["agg"].lower()
        fmt_key = kpi.get("format", "number")
        fmt_func = FORMAT_FUNCS.get(fmt_key, FORMAT_FUNCS["number"])

        try:
            series = df[col]
            value = _apply_agg(series, agg)
        except Exception as exc:
            logger.error("KPI computation failed for %s/%s: %s", col, agg, exc)
            value = None

        # Compute a delta indicator (compare to a reference metric)
        delta, delta_label = _compute_kpi_delta(df, col, agg, value)

        results.append(
            {
                "name": kpi.get("name", f"{agg}({col})"),
                "column": col,
                "agg": agg,
                "value": value,
                "formatted": fmt_func(value),
                "delta": delta,
                "delta_label": delta_label,
            }
        )
    return results


def _compute_kpi_delta(
    df: pd.DataFrame, col: str, agg: str, value: float | None
) -> tuple[float | None, str]:
    """
    Compute a meaningful delta for a KPI.
    - For mean: delta = mean - median (shows skewness)
    - For sum: delta = % of non-null rows
    - For others: no delta
    """
    if value is None or not pd.api.types.is_numeric_dtype(df.get(col, pd.Series())):
        return None, ""

    try:
        series = df[col].dropna()
        if agg == "mean":
            median_val = float(series.median())
            if median_val != 0:
                pct_diff = round(((value - median_val) / abs(median_val)) * 100, 1)
                return pct_diff, "vs median"
        elif agg == "sum":
            fill_rate = round(series.count() / max(len(df), 1) * 100, 1)
            return fill_rate, "% non-null"
        elif agg in ("max", "min"):
            mean_val = float(series.mean())
            if mean_val != 0:
                pct_diff = round(((value - mean_val) / abs(mean_val)) * 100, 1)
                return pct_diff, "vs mean"
    except Exception:
        pass

    return None, ""


def _apply_agg(series: pd.Series, agg: str) -> float | int | None:
    """Apply a validated aggregation to a pandas Series."""
    func_map = {
        "mean": lambda s: s.mean(),
        "sum": lambda s: s.sum(),
        "count": lambda s: s.count(),
        "max": lambda s: s.max(),
        "min": lambda s: s.min(),
        "median": lambda s: s.median(),
        "nunique": lambda s: s.nunique(),
        "std": lambda s: s.std(),
    }
    result = func_map[agg](series)
    # Convert numpy types to native Python
    try:
        return float(result) if result is not None else None
    except (TypeError, ValueError):
        return None


# ── Chart building ───────────────────────────────────────────────────────────

def build_charts(
    charts: list[dict[str, Any]],
    df: pd.DataFrame,
) -> list[tuple[go.Figure, str]]:
    """
    Build Plotly figures from the validated chart specs using the **real** data.

    Returns a list of (Figure, description) tuples.
    """
    figures: list[tuple[go.Figure, str]] = []
    for spec in charts:
        try:
            fig = _build_single_chart(spec, df)
            if fig is not None:
                desc = spec.get("description", "")
                figures.append((fig, desc))
        except Exception as exc:
            logger.error("Chart build failed for '%s': %s", spec.get("title"), exc)
    return figures


def _build_single_chart(spec: dict[str, Any], df: pd.DataFrame) -> go.Figure | None:
    """Dispatch a single chart spec to the appropriate Plotly Express call."""
    ctype = spec["type"]
    x = spec.get("x")
    y = spec.get("y")
    color = spec.get("color")
    title = spec.get("title", "Chart")

    # Common kwargs
    kwargs: dict[str, Any] = {"title": title}
    if color:
        kwargs["color"] = color

    if ctype == "bar":
        fig = _build_bar_chart(df, x, y, color, title, kwargs)

    elif ctype == "line":
        fig = _build_line_chart(df, x, y, color, title, kwargs)

    elif ctype == "scatter":
        fig = px.scatter(df, x=x, y=y, **kwargs)

    elif ctype == "histogram":
        fig = px.histogram(df, x=x, **kwargs)

    elif ctype == "box":
        fig = px.box(df, x=x, y=y, **kwargs)

    elif ctype == "pie":
        fig = _build_pie_chart(df, x, title)

    elif ctype == "heatmap":
        fig = _build_heatmap(df, title)

    elif ctype == "treemap":
        fig = _build_treemap(df, x, y, color, title)

    elif ctype == "sunburst":
        fig = _build_sunburst(df, x, y, color, title)

    elif ctype == "funnel":
        fig = _build_funnel(df, x, y, title)

    else:
        return None

    if fig is None:
        return None

    # Polish layout
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=40, r=40, t=60, b=40),
        height=450,
        font=dict(family="Inter, sans-serif"),
    )
    return fig


# ── Specialised chart builders ───────────────────────────────────────────────

def _build_bar_chart(
    df: pd.DataFrame, x: str, y: str | None, color: str | None,
    title: str, kwargs: dict[str, Any]
) -> go.Figure:
    """
    Build a bar chart with smart aggregation:
    - If x is high-cardinality, aggregate to top-N
    - If color is set, group-by before plotting
    """
    plot_df = df.copy()

    if y and x:
        # Check if aggregation is needed
        nunique_x = plot_df[x].nunique()
        if nunique_x > _HIGH_CARDINALITY_THRESHOLD:
            # Auto-aggregate: top 20 by sum of y
            if pd.api.types.is_numeric_dtype(plot_df.get(y, pd.Series())):
                group_cols = [x] + ([color] if color else [])
                plot_df = (
                    plot_df.groupby(group_cols, as_index=False)[y]
                    .sum()
                    .nlargest(_HIGH_CARDINALITY_THRESHOLD, y)
                )
                logger.info("Bar chart auto-aggregated: %d → %d groups", nunique_x, len(plot_df))
        elif color and pd.api.types.is_numeric_dtype(plot_df.get(y, pd.Series())):
            # Aggregate grouped bar
            group_cols = [x, color]
            plot_df = plot_df.groupby(group_cols, as_index=False)[y].sum()

    return px.bar(plot_df, x=x, y=y, **kwargs)


def _build_line_chart(
    df: pd.DataFrame, x: str, y: str | None, color: str | None,
    title: str, kwargs: dict[str, Any]
) -> go.Figure:
    """
    Build a line chart with smart date resampling:
    - If x is a datetime column, resample to an appropriate frequency.
    """
    plot_df = df.copy()

    if pd.api.types.is_datetime64_any_dtype(plot_df.get(x, pd.Series())):
        plot_df = plot_df.sort_values(x)

        if y and pd.api.types.is_numeric_dtype(plot_df.get(y, pd.Series())):
            # Determine resample frequency based on date span
            span = (plot_df[x].max() - plot_df[x].min()).days
            if span > 365 * 2:
                freq, freq_label = "MS", "Monthly"
            elif span > 90:
                freq, freq_label = "W", "Weekly"
            else:
                freq, freq_label = "D", "Daily"

            if color:
                # Group by color + resample
                groups = []
                for name, group in plot_df.groupby(color):
                    resampled = (
                        group.set_index(x)[[y]]
                        .resample(freq)
                        .mean()
                        .reset_index()
                    )
                    resampled[color] = name
                    groups.append(resampled)
                plot_df = pd.concat(groups, ignore_index=True) if groups else plot_df
            else:
                plot_df = (
                    plot_df.set_index(x)[[y]]
                    .resample(freq)
                    .mean()
                    .reset_index()
                )
            logger.info("Line chart resampled to %s (%d points)", freq_label, len(plot_df))

    return px.line(plot_df, x=x, y=y, **kwargs)


def _build_pie_chart(df: pd.DataFrame, x: str, title: str) -> go.Figure:
    """Build a pie chart from value counts, with top-15 + Other bucketing."""
    value_counts = df[x].value_counts().reset_index()
    value_counts.columns = [x, "count"]

    if len(value_counts) > 15:
        top = value_counts.head(14)
        other_count = value_counts.iloc[14:]["count"].sum()
        other_row = pd.DataFrame([{x: "Other", "count": other_count}])
        value_counts = pd.concat([top, other_row], ignore_index=True)

    return px.pie(value_counts, names=x, values="count", title=title)


def _build_heatmap(df: pd.DataFrame, title: str) -> go.Figure | None:
    """Build a correlation heatmap from numeric columns."""
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        logger.warning("Heatmap skipped – fewer than 2 numeric columns")
        return None

    corr = numeric_df.corr()
    return px.imshow(
        corr,
        text_auto=".2f",
        title=title,
        color_continuous_scale="RdBu_r",
        aspect="auto",
    )


def _build_treemap(
    df: pd.DataFrame, x: str, y: str | None, color: str | None, title: str
) -> go.Figure | None:
    """Build a treemap with category hierarchy."""
    path_cols = [x]
    if color and color != x:
        path_cols = [x, color]

    values_col = y if y and pd.api.types.is_numeric_dtype(df.get(y, pd.Series())) else None

    try:
        if values_col:
            agg_df = df.groupby(path_cols, as_index=False)[values_col].sum()
            fig = px.treemap(agg_df, path=path_cols, values=values_col, title=title)
        else:
            # Count-based treemap
            count_df = df.groupby(path_cols).size().reset_index(name="count")
            fig = px.treemap(count_df, path=path_cols, values="count", title=title)
        return fig
    except Exception as exc:
        logger.warning("Treemap failed: %s", exc)
        return None


def _build_sunburst(
    df: pd.DataFrame, x: str, y: str | None, color: str | None, title: str
) -> go.Figure | None:
    """Build a sunburst chart with category hierarchy."""
    path_cols = [x]
    if color and color != x:
        path_cols = [x, color]

    values_col = y if y and pd.api.types.is_numeric_dtype(df.get(y, pd.Series())) else None

    try:
        if values_col:
            agg_df = df.groupby(path_cols, as_index=False)[values_col].sum()
            fig = px.sunburst(agg_df, path=path_cols, values=values_col, title=title)
        else:
            count_df = df.groupby(path_cols).size().reset_index(name="count")
            fig = px.sunburst(count_df, path=path_cols, values="count", title=title)
        return fig
    except Exception as exc:
        logger.warning("Sunburst failed: %s", exc)
        return None


def _build_funnel(
    df: pd.DataFrame, x: str, y: str | None, title: str
) -> go.Figure | None:
    """Build a funnel chart from categorical counts or numeric values."""
    try:
        if y and pd.api.types.is_numeric_dtype(df.get(y, pd.Series())):
            agg_df = df.groupby(x, as_index=False)[y].sum().sort_values(y, ascending=False)
            fig = px.funnel(agg_df, x=y, y=x, title=title)
        else:
            counts = df[x].value_counts().reset_index()
            counts.columns = [x, "count"]
            fig = px.funnel(counts, x="count", y=x, title=title)
        return fig
    except Exception as exc:
        logger.warning("Funnel failed: %s", exc)
        return None
