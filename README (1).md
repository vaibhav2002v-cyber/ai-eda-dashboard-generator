# 📊 AI-Powered EDA & Dashboard Generator

Upload a CSV or Excel file, describe what you want to know in plain English, and get an interactive analytics dashboard — KPIs, charts, and AI-generated insights — built from your real data.

This project pairs an LLM (xAI's Grok) with a validated computation layer: **the AI decides *what* to analyze, but every number displayed is computed directly from your data, never generated or guessed by the model.**

---

## ✨ Features

- **Natural-language analysis** — describe your question ("compare sales by region", "what are the main trends?") and get a tailored dashboard
- **Automated data profiling** — auto date-detection, correlation mining, IQR-based outlier detection, missing-value analysis
- **LLM-as-planner, not calculator** — Grok proposes KPIs/charts/insights as a JSON plan; a validation layer checks every column name and aggregation against the real schema before anything is computed or rendered
- **10 chart types** — bar, line, scatter, histogram, box, pie, heatmap, treemap, sunburst, funnel (via Plotly)
- **Dynamic filters** — sidebar widgets (multi-select, range sliders, date pickers) auto-generated based on your dataset's columns
- **One-click code export** — generates a standalone, reproducible Python (pandas + matplotlib) script for the analysis
- **AI dashboard mockup image** — separate image-generation pipeline for a stylized executive-dashboard visual
- **Resilience & performance** — retry logic with exponential backoff on API calls, token-budget guarding to avoid context overflows, and session-based caching to avoid redundant LLM calls

---

## 🏗️ Architecture

```
├── app.py              # Streamlit UI & orchestration, session-state management
├── config.py           # Centralized config (env vars / Streamlit secrets / defaults)
└── utils/
    ├── data_loader.py   # File ingestion, auto date-parsing, data profiling
    ├── filters.py       # Dynamic sidebar filter widgets
    ├── grok_client.py   # xAI Grok API wrapper (retries, token budgeting)
    ├── prompts.py       # All LLM prompt templates
    └── eda_engine.py    # Plan validation ("trust boundary"), KPI computation, chart building
```

**The core design principle:** `eda_engine.py` acts as a trust boundary. Every column name, aggregation, and chart type suggested by the LLM is validated against the real DataFrame in `sanitize_plan()` before anything runs — invalid suggestions are silently dropped rather than crashing the app or fabricating a result. No LLM-generated number is ever displayed; every value shown is computed by pandas from the actual uploaded data.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- An [xAI API key](https://x.ai/) (Grok)

### Installation

```bash
git clone https://github.com/<your-username>/ai-eda-dashboard-generator.git
cd ai-eda-dashboard-generator
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
XAI_API_KEY=your_api_key_here
```

(Alternatively, set `XAI_API_KEY` in Streamlit Secrets if deploying to Streamlit Cloud.)

### Run

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`), upload a CSV or Excel file, and describe what you'd like to learn from it.

---

## 🧰 Tech Stack

| Layer | Tools |
|---|---|
| UI / App framework | Streamlit |
| Data processing | Pandas, NumPy |
| Visualization | Plotly |
| LLM | xAI Grok (OpenAI-compatible API) |
| File support | OpenPyXL (Excel) |
| Config | python-dotenv |

---

## 📸 How It Works

1. **Upload** a CSV/Excel file — the app profiles it (types, missing values, correlations, outliers, date ranges)
2. **Ask a question** in plain English about what you want to learn
3. **Grok proposes a plan** — a JSON object naming KPIs, charts, and insight angles (never actual data values)
4. **The plan is validated** against your real dataset's columns and types
5. **KPIs and charts are computed and rendered** entirely from your real data
6. **Optionally**, export a standalone Python script that reproduces the analysis, or generate a stylized dashboard mockup image

---

## 🛠️ Development Notes

This project was built using an AI-assisted ("vibe coding") development workflow — architecture, prompt design, and the validation/trust-boundary logic were directed by me, with AI pair-programming used to accelerate implementation.

---

## 📄 License

Add your preferred license here (e.g., MIT).
