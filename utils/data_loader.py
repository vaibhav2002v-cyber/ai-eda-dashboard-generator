"""
data_loader.py – File ingestion and data-profiling utilities.

All heavy pandas work lives here so the main app stays thin.
Includes auto-date parsing, correlation extraction, and outlier detection.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

import pandas as pd
import numpy as np
import streamlit as st

logger = logging.getLogger(__name__)

# Regex patterns for common date-like strings
_DATE_PATTERNS = [
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}",          # 2024-01-15, 2024/1/5
    r"\d{1,2}[-/]\d{1,2}[-/]\d{4}",          # 15-01-2024, 1/5/2024
    r"\d{1,2}\s+\w{3,9}\s+\d{4}",            # 15 Jan 2024
    r"\w{3,9}\s+\d{1,2},?\s+\d{4}",          # January 15, 2024
]
_DATE_RE = re.compile("|".join(f"(?:{p})" for p in _DATE_PATTERNS))


# ── File loading ─────────────────────────────────────────────────────────────

def load_data(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    """
    Read an uploaded file (.csv, .xlsx, .xls) into a pandas DataFrame.
    Automatically attempts to parse date columns.

    Raises
    ------
    ValueError
        If the file type is not supported.
    """
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file type: {name}")

    # Auto-detect and convert date columns
    df = _auto_parse_dates(df)

    logger.info("Loaded '%s' → %d rows × %d columns", uploaded_file.name, *df.shape)
    return df


# ── Auto date parsing ────────────────────────────────────────────────────────

def _auto_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scan object columns for date-like strings and convert them to datetime.
    Only converts if ≥ 80% of non-null values parse successfully.
    """
    for col in df.select_dtypes(include=["object"]).columns:
        sample = df[col].dropna().head(20)
        if len(sample) == 0:
            continue

        # Quick regex check on the sample
        date_matches = sample.astype(str).apply(lambda v: bool(_DATE_RE.search(v)))
        if date_matches.mean() < 0.8:
            continue

        # Attempt full conversion
        try:
            converted = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            success_rate = converted.notna().sum() / max(df[col].notna().sum(), 1)
            if success_rate >= 0.8:
                df[col] = converted
                logger.info("Auto-parsed column '%s' as datetime (%.0f%% success)", col, success_rate * 100)
        except Exception:
            pass  # Leave as-is

    return df


# ── Data profiling ───────────────────────────────────────────────────────────

def build_data_profile(df: pd.DataFrame, sample_rows: int = 5) -> dict[str, Any]:
    """
    Build a compact JSON-serialisable profile of *df* suitable for sending
    to the LLM.  The profile contains:

    - shape (rows, columns)
    - column names, dtypes, and semantic types
    - missing-value percentages
    - numeric summaries (min/max/mean/median/std)
    - top categorical values (up to 10 per column)
    - date ranges for datetime columns
    - top-5 correlations (positive & negative)
    - outlier percentages per numeric column (IQR method)
    - a small row sample

    The full dataset is **never** sent – only this digest.
    """
    profile: dict[str, Any] = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_pct": {
            col: round(df[col].isna().mean() * 100, 2) for col in df.columns
        },
    }

    # ── Semantic column types (for the UI badges) ────────────────────────
    col_types: dict[str, str] = {}
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            col_types[col] = "date"
        elif pd.api.types.is_numeric_dtype(df[col]):
            col_types[col] = "numeric"
        elif pd.api.types.is_bool_dtype(df[col]):
            col_types[col] = "boolean"
        else:
            col_types[col] = "categorical"
    profile["column_types"] = col_types

    # ── Numeric summaries ────────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_summary: dict[str, dict[str, float]] = {}
    for col in numeric_cols:
        numeric_summary[col] = {
            "min": _safe_float(df[col].min()),
            "max": _safe_float(df[col].max()),
            "mean": _safe_float(df[col].mean()),
            "median": _safe_float(df[col].median()),
            "std": _safe_float(df[col].std()),
        }
    profile["numeric_summary"] = numeric_summary

    # ── Categorical top-values ───────────────────────────────────────────
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    cat_summary: dict[str, dict[str, int]] = {}
    for col in cat_cols:
        top = df[col].value_counts().head(10)
        cat_summary[col] = {str(k): int(v) for k, v in top.items()}
    profile["categorical_top_values"] = cat_summary

    # ── Date ranges ──────────────────────────────────────────────────────
    date_cols = df.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()
    date_ranges: dict[str, dict[str, str]] = {}
    for col in date_cols:
        try:
            date_ranges[col] = {
                "min": str(df[col].min()),
                "max": str(df[col].max()),
                "span_days": int((df[col].max() - df[col].min()).days),
            }
        except Exception:
            pass
    profile["date_ranges"] = date_ranges

    # ── Top correlations ─────────────────────────────────────────────────
    profile["top_correlations"] = _get_top_correlations(df, n=5)

    # ── Outlier percentages (IQR method) ─────────────────────────────────
    outlier_pct: dict[str, float] = {}
    for col in numeric_cols:
        outlier_pct[col] = _iqr_outlier_pct(df[col])
    profile["outlier_pct"] = outlier_pct

    # ── Small sample (head) ──────────────────────────────────────────────
    sample_df = df.head(sample_rows)
    profile["sample_rows"] = _df_to_serialisable(sample_df)

    return profile


# ── Correlation extraction ───────────────────────────────────────────────────

def _get_top_correlations(
    df: pd.DataFrame, n: int = 5
) -> list[dict[str, Any]]:
    """
    Return the top-N strongest correlations (positive and negative),
    excluding self-correlations.
    """
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return []

    corr = numeric_df.corr()
    # Mask the upper triangle + diagonal
    mask = np.triu(np.ones_like(corr, dtype=bool))
    corr_masked = corr.where(~mask)

    # Stack and sort by absolute value
    pairs = (
        corr_masked.stack()
        .reset_index()
        .rename(columns={"level_0": "col_a", "level_1": "col_b", 0: "correlation"})
    )
    pairs["abs_corr"] = pairs["correlation"].abs()
    top = pairs.nlargest(n, "abs_corr")

    return [
        {
            "col_a": row["col_a"],
            "col_b": row["col_b"],
            "correlation": round(float(row["correlation"]), 4),
        }
        for _, row in top.iterrows()
    ]


# ── Outlier detection ────────────────────────────────────────────────────────

def _iqr_outlier_pct(series: pd.Series) -> float:
    """Return the percentage of values outside 1.5 × IQR."""
    s = series.dropna()
    if len(s) < 4:
        return 0.0
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = ((s < lower) | (s > upper)).sum()
    return round(outliers / len(s) * 100, 2)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_float(val: Any) -> float | None:
    """Convert a numpy/pandas scalar to a plain Python float, or None."""
    try:
        f = float(val)
        return None if math.isnan(f) or math.isinf(f) else round(f, 4)
    except (TypeError, ValueError):
        return None


def _df_to_serialisable(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a small DataFrame to a list-of-dicts with JSON-safe types."""
    records = df.head(20).to_dict(orient="records")
    clean: list[dict[str, Any]] = []
    for row in records:
        clean.append(
            {
                k: (
                    None
                    if pd.isna(v)
                    else int(v) if isinstance(v, (int,)) and not isinstance(v, bool)
                    else float(v) if isinstance(v, float)
                    else str(v)
                )
                for k, v in row.items()
            }
        )
    return clean
