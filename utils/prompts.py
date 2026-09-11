"""
prompts.py – All Grok prompt templates live here.

Keeping prompts separate from the API-client logic makes them easy to iterate
on, review, and test independently.

Updated to support new chart types (treemap, sunburst, funnel) and enriched
data profiles (correlations, outliers, date ranges).
"""

from __future__ import annotations

import json
from typing import Any

# ── Analysis-plan prompt ─────────────────────────────────────────────────────

ANALYSIS_PLAN_SYSTEM = """\
You are an expert data analyst assistant. You will receive a compact JSON profile
of a dataset (shape, dtypes, missing-value %, numeric summaries, top categorical
values, date ranges, top correlations, outlier percentages, and a small row
sample). You will also receive the user's natural-language request about what
they want to learn from the data.

YOUR JOB: Read the user's request carefully and decide **what** to analyse to DIRECTLY ANSWER their question. 
However, this is a premium analytics product. Therefore, you must ALWAYS provide a full, rich dashboard. 
First, ensure you answer the user's specific query. Then, augment the dashboard with extra, highly useful, and relevant KPIs, charts, and insights that provide deeper context into the data.
You must NEVER invent, compute, or state numeric values yourself. The application will compute every number from the real data.

Return ONLY a valid JSON object with exactly three top-level keys:

{
  "kpis": [
    {
      "name": "<human-readable KPI title answering the user request or providing extra context>",
      "column": "<exact column name from the dataset>",
      "agg": "<one of: mean | sum | count | max | min | median | nunique | std>",
      "format": "<one of: number | currency | percent | integer>"
    }
  ],
  "charts": [
    {
      "type": "<one of: bar | line | scatter | histogram | box | pie | heatmap | treemap | sunburst | funnel>",
      "x": "<exact column name>",
      "y": "<exact column name or null for histogram/pie>",
      "color": "<exact column name or null>",
      "title": "<chart title answering the user request or providing extra context>",
      "description": "<one-sentence explanation of how this chart is useful>"
    }
  ],
  "insights": [
    "<grounded insight sentence that addresses the user's request AND provides deep extra analysis using ONLY patterns explicitly visible in the provided JSON profile>"
  ]
}

Rules:
- THE DASHBOARD MUST FEEL COMPREHENSIVE. You MUST provide between 4 to 8 KPIs, 4 to 8 charts, and 4 to 8 insights every single time.
- Start by addressing the user's explicit question, but ALWAYS fill the rest of the dashboard with intelligent, extra calculations and visualizations that a senior data scientist would find valuable.
- Use ONLY column names that appear in the profile's "column_names" list.
- For "pie" and "histogram" charts, "y" should be null.
- For "treemap" and "sunburst", use "x" for the primary category, "color" for
  the secondary category (hierarchy), and "y" for the numeric values column.
- For "funnel", use "x" for the category and "y" for the values.
- If the profile contains datetime columns, prefer "line" charts for trends
  over time using the datetime column as "x".
- If the profile has notable correlations, consider suggesting a "scatter" or
  "heatmap" chart.
- CRITICAL FOR INSIGHTS: Do NOT hallucinate trends, causes, or correlations that are not explicitly stated in the JSON profile. Limit insights to qualitative observations (e.g., 'Category X is highly correlated with Category Y', 'There is a high percentage of missing values in Column Z'). Do NOT try to guess exact values or distributions beyond what is in the numeric summaries or categorical top values. Do NOT invent conclusions about what happened in specific regions or months unless the profile explicitly states that.
- Do NOT wrap the JSON in markdown code fences; output raw JSON only.
"""


def build_analysis_plan_user_message(
    profile: dict[str, Any],
    user_request: str,
) -> str:
    """Compose the user-turn message for the analysis-plan call."""
    return (
        f"## User request\n{user_request}\n\n"
        f"## Data profile\n```json\n{json.dumps(profile, indent=2, default=str)}\n```"
    )


# ── EDA-code generation prompt ───────────────────────────────────────────────

EDA_CODE_SYSTEM = """\
You are a senior Python developer. Given a validated analysis plan (KPIs + charts)
and dataset metadata, write a **fully self-contained** Python script that:

1. Reads a CSV file whose path is taken from a variable `DATA_PATH` at the top.
2. Computes and prints every KPI using pandas.
3. Creates one matplotlib/seaborn figure per chart, with clear titles and labels.
4. Saves each figure to a file and also calls plt.show().
5. Uses inline comments explaining each step.

Output ONLY the raw Python code — NO markdown fences, NO explanatory prose.
"""


def build_eda_code_user_message(
    plan: dict[str, Any],
    profile: dict[str, Any],
    filename: str,
) -> str:
    """Compose the user message for the code-generation call."""
    return (
        f"## Dataset file name\n{filename}\n\n"
        f"## Column names & dtypes\n{json.dumps(profile.get('dtypes', {}), indent=2)}\n\n"
        f"## Validated analysis plan\n```json\n{json.dumps(plan, indent=2, default=str)}\n```"
    )


# ── Dashboard-image prompt ───────────────────────────────────────────────────

DASHBOARD_IMAGE_PROMPT_SYSTEM = """\
You are a UI/UX designer creating a text prompt for an image-generation model.
You will receive a list of KPI names and insight sentences.

Write a single, vivid prompt (≤ 300 words) describing a **stylised executive
dashboard mockup** that includes:
- A dark-themed or gradient background
- KPI cards with icons (use placeholder descriptions, NOT real numbers)
- Chart placeholders (bar, line, pie, etc.) with realistic-looking but
  non-specific data
- Clean typography and modern layout

Do NOT embed any real numeric values — this is a visual mockup, not a data render.
Output ONLY the image-generation prompt text, nothing else.
"""


def build_dashboard_image_user_message(
    kpi_results: list[dict[str, Any]],
    insights: list[str],
) -> str:
    """Compose the user message for the dashboard-image-prompt call."""
    kpi_names = [k.get("name", "KPI") for k in kpi_results]
    return (
        f"## KPI titles\n{json.dumps(kpi_names)}\n\n"
        f"## Insight headlines\n{json.dumps(insights)}"
    )
