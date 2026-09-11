"""
filters.py – Dynamic sidebar filter builder for Streamlit.

Inspects the DataFrame and creates the appropriate Streamlit widget
for each column type: multi-select for categoricals, range sliders for
numerics, and date-range pickers for datetime columns.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

# Maximum unique values for a categorical column before we skip it
_MAX_CATEGORIES = 50


def render_filters(df: pd.DataFrame, key_prefix: str = "filter") -> pd.DataFrame:
    """
    Render dynamic filter widgets in the current Streamlit context and
    return a filtered copy of *df*.

    Call this inside a ``st.sidebar`` or ``st.expander`` block.
    """
    filtered = df.copy()
    active_filters = 0

    st.markdown("##### 🔎 Data Filters")

    # ── Categorical filters ──────────────────────────────────────────────
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    for col in cat_cols:
        nunique = df[col].nunique()
        if nunique > _MAX_CATEGORIES or nunique < 2:
            continue  # Skip very high cardinality or single-value columns

        unique_vals = sorted(df[col].dropna().unique().tolist(), key=str)
        selected = st.multiselect(
            f"🏷️ {col}",
            options=unique_vals,
            default=[],
            key=f"{key_prefix}_cat_{col}",
        )
        if selected:
            filtered = filtered[filtered[col].isin(selected)]
            active_filters += 1

    # ── Numeric filters ──────────────────────────────────────────────────
    num_cols = df.select_dtypes(include="number").columns.tolist()
    for col in num_cols:
        col_min = float(df[col].min())
        col_max = float(df[col].max())
        if col_min == col_max:
            continue  # Skip constant columns

        # Round to sensible step
        span = col_max - col_min
        if span > 1000:
            step = 10.0
        elif span > 100:
            step = 1.0
        elif span > 1:
            step = 0.1
        else:
            step = 0.01

        range_val = st.slider(
            f"📊 {col}",
            min_value=col_min,
            max_value=col_max,
            value=(col_min, col_max),
            step=step,
            key=f"{key_prefix}_num_{col}",
        )
        if range_val != (col_min, col_max):
            filtered = filtered[
                (filtered[col] >= range_val[0]) & (filtered[col] <= range_val[1])
            ]
            active_filters += 1

    # ── Date filters ─────────────────────────────────────────────────────
    date_cols = df.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()
    for col in date_cols:
        col_min = df[col].min().date()
        col_max = df[col].max().date()
        if col_min == col_max:
            continue

        date_range = st.date_input(
            f"📅 {col}",
            value=(col_min, col_max),
            min_value=col_min,
            max_value=col_max,
            key=f"{key_prefix}_date_{col}",
        )
        # date_input may return a single date or a tuple
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start, end = date_range
            mask = (filtered[col].dt.date >= start) & (filtered[col].dt.date <= end)
            if not mask.all():
                filtered = filtered[mask]
                active_filters += 1

    # ── Summary ──────────────────────────────────────────────────────────
    if active_filters > 0:
        st.info(f"🔍 **{len(filtered):,}** / {len(df):,} rows after {active_filters} filter(s)")
    else:
        st.caption(f"{len(df):,} rows (no filters active)")

    return filtered
