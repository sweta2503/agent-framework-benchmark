# agent-framework-benchmark
Benchmark: LangGraph vs CrewAI vs AutoGen on 107 real data engineering tasks. Success rate, token cost, latency, boilerplate — all measured. LangGraph wins.
# Agent Framework Benchmark
### LangGraph vs CrewAI vs AutoGen — 107 Real Data Engineering Tasks

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Full video breakdown:** [YouTube — Agentic Data Lab](https://www.youtube.com/@agenticdatalab)  
> **Live results leaderboard:** [Google Sheets](https://docs.google.com/spreadsheets/d/1Bl05tp99AZOwkHsHTmRj8Q87jwVWLq_CroGwXZQWjr8/edit?gid=277558238#gid=277558238)

---

## What This Is

A benchmark harness that runs identical data engineering tasks through LangGraph, CrewAI, and AutoGen using the same LLM (Groq Llama 3.3 70B), same prompts, and same timeout conditions — then measures success rate, token cost, latency, and boilerplate lines.

**Not a tutorial. Not a toy demo. Real tasks, real numbers.**

---

## Key Results

| Framework | SQL Gen | Pipeline Debug | Transformation | Avg Tokens | Avg Latency |
|---|---|---|---|---|---|
| **LangGraph** | **87.5%** | **79.0%** | **75.0%** | **~2,700** | **~12.7s** |
| CrewAI | 82.6% | 73.7% | 68.8% | ~5,005 | ~20.0s |
| AutoGen | 82.6% | 79.0% | **56.3%** | ~5,678 | ~17.9s |

**LangGraph wins on accuracy, token cost, and latency simultaneously.**  
AutoGen transformation at 56.3% is the worst result in the benchmark.

Full results: [Google Sheets leaderboard](https://docs.google.com/spreadsheets/d/1Bl05tp99AZOwkHsHTmRj8Q87jwVWLq_CroGwXZQWjr8/edit?gid=277558238#gid=277558238)

---

## Task Categories

107 tasks across 6 real DE categories:

- **SQL Generation** (24 tasks) — window functions, CTEs, complex aggregations
- **Pipeline Debugging** (19 tasks) — broken PySpark, schema mismatches, dropped rows  
- **Data Quality** (17 tasks) — null checks, range validation, schema enforcement
- **ETL Orchestration** (16 tasks) — Airflow DAG design, incremental loads
- **Transformation** (16 tasks) — JSON flattening, type casting, reshaping
- **Metadata Generation** (16 tasks) — dbt descriptions, data dictionaries

---

## Setup

**Requirements:** Python 3.11+, Groq API key (free at console.groq.com), optional Google Sheets service account

```bash
git clone https://github.com/YOUR_USERNAME/de-agent-framework-benchmark
cd de-agent-framework-benchmark
python3.11 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env           # add your Groq API key
python agent.py
```

Results stream to your terminal + write to MLflow. Google Sheets integration is optional — see `.env.example` for setup.

---

## Run Your Own Tasks

Add tasks to the `BENCHMARK_TASKS` list in `agent.py`:

```python
{
    "id": "sql_window_001",
    "category": "sql_generation",
    "prompt": "Write a SQL window function to...",
    "expected_output_schema": {"sql": str, "explanation": str}
}
```

Works with any OpenAI-compatible endpoint — swap Groq for Ollama to run fully local.

---

## Stack

- **Frameworks:** LangGraph, CrewAI, AutoGen
- **LLM:** Groq Llama 3.3 70B (free tier)
- **Tracking:** MLflow, Google Sheets
- **Storage:** SQLite
- **Free stack — no paid cloud required**

---

## Video

Full breakdown of these results, what they mean for production DE work, and which framework to use for each task type:

📺 **[Watch on YouTube — Agentic Data Lab](https://www.youtube.com/@agenticdatalab)**

---

## License

MIT — use it, fork it, run it against your own tasks.

---

*Built by [@agenticdatalab](https://youtube.com/@agenticdatalab)*
