"""
app.py – Streamlit UI & orchestration for the AI-Powered EDA Dashboard Generator.

State management:
  All intermediate results (dataframe, profile, plan, KPIs, charts, generated
  code/image) are stored in ``st.session_state`` so that button clicks for
  code generation and image generation do NOT re-run the analysis pipeline.
  Uploading a new file resets all downstream state automatically.

Enhancements (v2):
  - Multi-step progress bar during analysis
  - Tabbed layout for results (KPIs / Charts / Insights / Code / Image)
  - Data filtering sidebar with dynamic widgets
  - KPI delta indicators with coloured arrows
  - Chart description captions
  - Polished data preview with column-type badges
  - Sidebar dataset stats widget
  - Analysis caching by prompt hash
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# ── App-level logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Local imports (must come after logging setup) ────────────────────────────
from config import XAI_API_KEY, XAI_BASE_URL, XAI_IMAGE_MODEL, XAI_TEXT_MODEL
from utils.data_loader import build_data_profile, load_data
from utils.eda_engine import build_charts, compute_kpis, sanitize_plan
from utils.filters import render_filters
from utils.grok_client import GrokClient

# ── Output directories ──────────────────────────────────────────────────────
CODE_DIR = Path("outputs/generated_code")
IMAGE_DIR = Path("outputs/generated_images")
CODE_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


# ── Cached Grok client ──────────────────────────────────────────────────────

def _get_grok_client() -> GrokClient:
    """Singleton Grok client – instantiated once per Streamlit server process."""
    if not XAI_API_KEY:
        st.error(
            "🔑 **XAI_API_KEY not found.** Set it in `.env` (local) or "
            "Streamlit Secrets (cloud). See README for details."
        )
        st.stop()
    return GrokClient(
        api_key=XAI_API_KEY,
        base_url=XAI_BASE_URL,
        text_model=XAI_TEXT_MODEL,
        image_model=XAI_IMAGE_MODEL,
    )


# ── Session-state helpers ───────────────────────────────────────────────────

_DOWNSTREAM_KEYS = [
    "df", "profile", "plan", "kpi_results", "chart_figures",
    "generated_code", "generated_image", "filename", "analysis_cache",
]


def _reset_downstream():
    """Clear all analysis results when a new file is uploaded."""
    for key in _DOWNSTREAM_KEYS:
        st.session_state.pop(key, None)


def _cache_key(profile: dict, user_request: str) -> str:
    """Compute a deterministic hash for (profile + request) to enable caching."""
    blob = json.dumps(profile, sort_keys=True, default=str) + "|" + user_request.strip().lower()
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI-Powered EDA & Dashboard Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* KPI card styling */
    .kpi-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.4rem 1.2rem;
        text-align: center;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 0.8rem;
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.25);
    }
    .kpi-name {
        font-size: 0.78rem;
        color: #a0a0b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #818cf8, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .kpi-meta {
        font-size: 0.7rem;
        color: #6b6b80;
        margin-top: 0.25rem;
    }
    .kpi-delta {
        font-size: 0.72rem;
        font-weight: 600;
        margin-top: 0.3rem;
    }
    .kpi-delta.positive { color: #34d399; }
    .kpi-delta.negative { color: #f87171; }
    .kpi-delta.neutral  { color: #9ca3af; }

    /* Insight list */
    .insight-item {
        background: rgba(99, 102, 241, 0.06);
        border-left: 3px solid #6366f1;
        padding: 0.7rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.92rem;
        color: #e0e0e8;
    }

    /* Section headers */
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #e0e0e8;
        margin: 1.2rem 0 0.8rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Chart description caption */
    .chart-desc {
        font-size: 0.8rem;
        color: #9ca3af;
        font-style: italic;
        margin-top: -0.4rem;
        margin-bottom: 1rem;
        padding-left: 0.3rem;
    }

    /* Column type badges */
    .col-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-right: 4px;
    }
    .col-badge.numeric  { background: #312e81; color: #a5b4fc; }
    .col-badge.categorical { background: #1e3a5f; color: #93c5fd; }
    .col-badge.date     { background: #064e3b; color: #6ee7b7; }
    .col-badge.boolean  { background: #4a1d96; color: #c4b5fd; }

    /* Progress stepper */
    .progress-step {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0;
        font-size: 0.88rem;
    }
    .progress-step .icon { font-size: 1.1rem; }
    .progress-step.active { color: #818cf8; font-weight: 600; }
    .progress-step.done   { color: #34d399; }
    .progress-step.pending { color: #6b6b80; }

    /* Sidebar stats */
    .sidebar-stat {
        display: flex;
        justify-content: space-between;
        padding: 0.2rem 0;
        font-size: 0.82rem;
        color: #a0a0b8;
    }
    .sidebar-stat .label { font-weight: 500; }
    .sidebar-stat .value { font-weight: 700; color: #e0e0e8; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📊 AI-Powered EDA")
    st.markdown("Upload a dataset and let Grok decide what to analyse.")
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload your data",
        type=["csv", "xlsx", "xls"],
        help="Accepted formats: CSV, Excel (.xlsx / .xls)",
    )

    if uploaded_file is not None:
        # Reset state when a *different* file is uploaded
        if st.session_state.get("filename") != uploaded_file.name:
            _reset_downstream()
            st.session_state["filename"] = uploaded_file.name

        # Load data (only once per file)
        if "df" not in st.session_state:
            try:
                df = load_data(uploaded_file)
                st.session_state["df"] = df
                st.session_state["profile"] = build_data_profile(df)
            except Exception as exc:
                st.error(f"⚠️ Could not load file: {exc}")
                logger.error("File load error:\n%s", traceback.format_exc())
                st.stop()

        df: pd.DataFrame = st.session_state["df"]
        st.success(f"✅ **{uploaded_file.name}** loaded")

        # ── Sidebar dataset stats ────────────────────────────────────────
        profile = st.session_state["profile"]
        col_types = profile.get("column_types", {})

        n_numeric = sum(1 for v in col_types.values() if v == "numeric")
        n_cat = sum(1 for v in col_types.values() if v == "categorical")
        n_date = sum(1 for v in col_types.values() if v == "date")
        n_bool = sum(1 for v in col_types.values() if v == "boolean")
        avg_missing = sum(profile.get("missing_pct", {}).values()) / max(len(profile.get("missing_pct", {})), 1)

        st.markdown(
            f"""
            <div class="sidebar-stat"><span class="label">Rows</span><span class="value">{df.shape[0]:,}</span></div>
            <div class="sidebar-stat"><span class="label">Columns</span><span class="value">{df.shape[1]}</span></div>
            <div class="sidebar-stat"><span class="label">📊 Numeric</span><span class="value">{n_numeric}</span></div>
            <div class="sidebar-stat"><span class="label">🏷️ Categorical</span><span class="value">{n_cat}</span></div>
            <div class="sidebar-stat"><span class="label">📅 Date</span><span class="value">{n_date}</span></div>
            <div class="sidebar-stat"><span class="label">⚠️ Avg Missing</span><span class="value">{avg_missing:.1f}%</span></div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Data filters ─────────────────────────────────────────────────
        with st.expander("🔎 Filter Data", expanded=False):
            filtered_df = render_filters(df)
            st.session_state["filtered_df"] = filtered_df

    st.divider()
    st.caption("Powered by [xAI Grok](https://x.ai/) · Built with Streamlit")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("# 📊 AI-Powered EDA & Dashboard Generator")

if "df" not in st.session_state:
    st.info("👈 Upload a CSV or Excel file in the sidebar to get started.")
    st.stop()

df_raw = st.session_state["df"]
df = st.session_state.get("filtered_df", df_raw)
profile = st.session_state["profile"]

# ── Data preview ─────────────────────────────────────────────────────────────
with st.expander("🔍 Data Preview", expanded=False):
    # Column type badges
    col_types = profile.get("column_types", {})
    badge_map = {
        "numeric": ("📊", "numeric"),
        "categorical": ("🏷️", "categorical"),
        "date": ("📅", "date"),
        "boolean": ("🔘", "boolean"),
    }
    badges_html = " ".join(
        f'<span class="col-badge {badge_map.get(ctype, ("", ""))[1]}">'
        f'{badge_map.get(ctype, ("❓", ""))[0]} {col}</span>'
        for col, ctype in col_types.items()
    )
    st.markdown(badges_html, unsafe_allow_html=True)
    st.dataframe(df.head(50), use_container_width=True)

    # Missing value summary
    missing = profile.get("missing_pct", {})
    cols_with_missing = {k: v for k, v in missing.items() if v > 0}
    if cols_with_missing:
        st.caption(f"⚠️ Columns with missing data: " +
                   ", ".join(f"**{k}** ({v}%)" for k, v in cols_with_missing.items()))
    st.caption(f"{df.shape[0]:,} rows × {df.shape[1]} columns")

# ── User prompt ──────────────────────────────────────────────────────────────
st.markdown("---")
user_request = st.text_area(
    "What would you like to know about this data?",
    placeholder=(
        "e.g. 'Analyse the data', 'Show me the most important KPIs', "
        "'What are the main trends?', 'Compare sales by region' …"
    ),
    height=100,
)

analyze_clicked = st.button("🚀 Analyze", type="primary", use_container_width=True)

# ── Run the analysis pipeline ────────────────────────────────────────────────
if analyze_clicked:
    if not user_request.strip():
        st.warning("Please describe what you'd like to analyse.")
        st.stop()

    grok = _get_grok_client()

    # Check analysis cache
    cache = st.session_state.get("analysis_cache", {})
    ck = _cache_key(profile, user_request)

    if ck in cache:
        # Restore from cache — no API call needed
        cached = cache[ck]
        st.session_state["plan"] = cached["plan"]
        st.session_state["kpi_results"] = cached["kpi_results"]
        st.session_state["chart_figures"] = cached["chart_figures"]
        # Clear generated artifacts (they depend on the plan)
        st.session_state.pop("generated_code", None)
        st.session_state.pop("generated_image", None)
        st.toast("⚡ Loaded from cache!", icon="⚡")
    else:
        # Clear previous results (keep df & profile)
        for key in ["plan", "kpi_results", "chart_figures", "generated_code", "generated_image"]:
            st.session_state.pop(key, None)

        # ── Progress stepper ─────────────────────────────────────────────
        progress_container = st.container()
        progress_bar = st.progress(0)

        # ── Step 1: Get analysis plan from Grok ──────────────────────────
        with progress_container:
            st.markdown(
                '<div class="progress-step active">'
                '<span class="icon">🤖</span> Step 1/3 — Grok is designing the analysis plan…</div>',
                unsafe_allow_html=True,
            )
        progress_bar.progress(10)

        try:
            # Build profile from filtered data for the API call
            filtered_profile = build_data_profile(df)
            raw_plan = grok.get_analysis_plan(filtered_profile, user_request)
        except json.JSONDecodeError:
            st.error(
                "⚠️ Grok returned malformed JSON. This can happen occasionally — "
                "please click **Analyze** again to retry."
            )
            logger.error("JSONDecodeError:\n%s", traceback.format_exc())
            st.stop()
        except Exception as exc:
            st.error(f"⚠️ Grok API error: {exc}")
            logger.error("Grok API error:\n%s", traceback.format_exc())
            st.stop()

        progress_bar.progress(40)

        # ── Step 2: Validate the plan ────────────────────────────────────
        plan = sanitize_plan(raw_plan, df)
        st.session_state["plan"] = plan

        if not plan["kpis"] and not plan["charts"]:
            st.warning(
                "Grok's analysis plan had no valid KPIs or charts after validation. "
                "Try rephrasing your request."
            )
            st.stop()

        # ── Step 3: Compute KPIs from real data ─────────────────────────
        with progress_container:
            st.markdown(
                '<div class="progress-step active">'
                '<span class="icon">📐</span> Step 2/3 — Computing KPIs from real data…</div>',
                unsafe_allow_html=True,
            )
        progress_bar.progress(55)

        kpi_results = compute_kpis(plan["kpis"], df)
        st.session_state["kpi_results"] = kpi_results

        # ── Step 4: Build charts from real data ──────────────────────────
        with progress_container:
            st.markdown(
                '<div class="progress-step active">'
                '<span class="icon">📊</span> Step 3/3 — Building interactive charts…</div>',
                unsafe_allow_html=True,
            )
        progress_bar.progress(75)

        chart_figures = build_charts(plan["charts"], df)
        st.session_state["chart_figures"] = chart_figures

        progress_bar.progress(100)

        # ── Cache the result ─────────────────────────────────────────────
        if "analysis_cache" not in st.session_state:
            st.session_state["analysis_cache"] = {}
        st.session_state["analysis_cache"][ck] = {
            "plan": plan,
            "kpi_results": kpi_results,
            "chart_figures": chart_figures,
        }

        st.toast("✅ Analysis complete!", icon="🎉")


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS DISPLAY  (reads from session_state — survives reruns)
# ══════════════════════════════════════════════════════════════════════════════

if "kpi_results" in st.session_state:
    kpi_results = st.session_state["kpi_results"]
    plan = st.session_state["plan"]
    chart_figures = st.session_state.get("chart_figures", [])

    # ── Tabbed layout ────────────────────────────────────────────────────
    tab_kpis, tab_charts, tab_insights, tab_code, tab_image = st.tabs(
        ["📌 KPIs", "📊 Charts", "💡 Insights", "🧑‍💻 Code", "🖼️ Image"]
    )

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1: KPI Metric Cards
    # ══════════════════════════════════════════════════════════════════════
    with tab_kpis:
        st.markdown('<div class="section-header">📌 Key Performance Indicators</div>', unsafe_allow_html=True)

        cols = st.columns(min(len(kpi_results), 4)) if kpi_results else []
        for idx, kpi in enumerate(kpi_results):
            col = cols[idx % len(cols)]
            with col:
                # Delta indicator
                delta = kpi.get("delta")
                delta_label = kpi.get("delta_label", "")
                if delta is not None:
                    if delta > 0:
                        delta_html = f'<div class="kpi-delta positive">▲ {delta:+.1f}% {delta_label}</div>'
                    elif delta < 0:
                        delta_html = f'<div class="kpi-delta negative">▼ {delta:+.1f}% {delta_label}</div>'
                    else:
                        delta_html = f'<div class="kpi-delta neutral">● 0% {delta_label}</div>'
                else:
                    delta_html = ""

                st.markdown(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-name">{kpi['name']}</div>
                        <div class="kpi-value">{kpi['formatted']}</div>
                        <div class="kpi-meta">{kpi['agg'].upper()}({kpi['column']})</div>
                        {delta_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2: Interactive Charts
    # ══════════════════════════════════════════════════════════════════════
    with tab_charts:
        if chart_figures:
            st.markdown('<div class="section-header">📊 Interactive Charts</div>', unsafe_allow_html=True)

            for i in range(0, len(chart_figures), 2):
                chart_cols = st.columns(2)
                for j, c in enumerate(chart_cols):
                    if i + j < len(chart_figures):
                        fig, desc = chart_figures[i + j]
                        with c:
                            st.plotly_chart(
                                fig,
                                use_container_width=True,
                                key=f"chart_{i+j}",
                            )
                            if desc:
                                st.markdown(
                                    f'<div class="chart-desc">💬 {desc}</div>',
                                    unsafe_allow_html=True,
                                )
        else:
            st.info("No charts were generated for this analysis.")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3: AI Insights
    # ══════════════════════════════════════════════════════════════════════
    with tab_insights:
        insights = plan.get("insights", [])
        if insights:
            st.markdown('<div class="section-header">💡 AI-Generated Insights</div>', unsafe_allow_html=True)
            st.caption("_These observations are AI-generated commentary grounded in the computed data profile._")
            for insight in insights:
                st.markdown(
                    f'<div class="insight-item">{insight}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No insights were generated for this analysis.")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 4: Code Generation
    # ══════════════════════════════════════════════════════════════════════
    with tab_code:
        st.markdown('<div class="section-header">🧑‍💻 EDA Code Generator</div>', unsafe_allow_html=True)
        st.caption("Generate a standalone Python script that reproduces this analysis.")

        generate_code_clicked = st.button(
            "⚡ Generate EDA Code",
            use_container_width=True,
            help="Generate a standalone Python script (pandas + matplotlib) for this analysis.",
            key="btn_generate_code",
        )

        if generate_code_clicked:
            grok = _get_grok_client()
            with st.spinner("🧑‍💻 Generating EDA script…"):
                try:
                    code = grok.generate_eda_code(
                        plan,
                        profile,
                        st.session_state.get("filename", "data.csv"),
                    )
                    st.session_state["generated_code"] = code

                    # Save a timestamped copy
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = CODE_DIR / f"eda_script_{ts}.py"
                    save_path.write_text(code, encoding="utf-8")
                    logger.info("EDA script saved to %s", save_path)

                except Exception as exc:
                    st.error(f"⚠️ Code generation failed: {exc}")
                    logger.error("Code gen error:\n%s", traceback.format_exc())

        if "generated_code" in st.session_state:
            code = st.session_state["generated_code"]
            st.code(code, language="python")
            st.download_button(
                "⬇️ Download Python Script",
                data=code,
                file_name="eda_analysis.py",
                mime="text/x-python",
                key="dl_code",
            )

    # ══════════════════════════════════════════════════════════════════════
    # TAB 5: Dashboard Image
    # ══════════════════════════════════════════════════════════════════════
    with tab_image:
        st.markdown('<div class="section-header">🖼️ Dashboard Mockup Generator</div>', unsafe_allow_html=True)
        st.caption("Generate a stylised dashboard mockup image (visual concept, not a literal chart re-render).")

        generate_image_clicked = st.button(
            "🎨 Generate Dashboard Image",
            use_container_width=True,
            help="Generate a stylised dashboard mockup image.",
            key="btn_generate_image",
        )

        if generate_image_clicked:
            grok = _get_grok_client()
            with st.spinner("🖼️ Generating dashboard mockup image…"):
                try:
                    image_bytes = grok.generate_dashboard_image(
                        kpi_results,
                        plan.get("insights", []),
                    )
                    st.session_state["generated_image"] = image_bytes

                    # Save a timestamped copy
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = IMAGE_DIR / f"dashboard_{ts}.png"
                    save_path.write_bytes(image_bytes)
                    logger.info("Dashboard image saved to %s", save_path)

                except Exception as exc:
                    st.error(f"⚠️ Image generation failed: {exc}")
                    logger.error("Image gen error:\n%s", traceback.format_exc())

        if "generated_image" in st.session_state:
            image_bytes = st.session_state["generated_image"]
            st.caption(
                "⚠️ _This is an AI-generated **visual mockup** — not a literal "
                "re-render of the charts above. It illustrates a possible dashboard style._"
            )
            st.image(image_bytes, use_container_width=True)
            st.download_button(
                "⬇️ Download Dashboard Image",
                data=image_bytes,
                file_name="dashboard_mockup.png",
                mime="image/png",
                key="dl_image",
            )
