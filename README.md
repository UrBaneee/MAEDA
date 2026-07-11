# MAEDA — Multi-Agent Enterprise Data Analyst

> A production-grade multi-agent system that takes a natural language question about enterprise data, autonomously connects to data sources, performs analysis, generates visualizations, and delivers an insight report — with built-in evaluation, guardrails, and full decision tracing.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    MAEDA (This Project)                      │
│  LangGraph Orchestrator · Eval · Guardrails · Viz · MCP      │
│                                                              │
│  Intent Parser → Data Connector → Analysis Agent             │
│       ↓                                                      │
│  Viz Agent → Insight Agent → Guardrail Agent → Eval Module   │
│                                                              │
│  ┌──────────────────┐    ┌──────────────────────────────┐    │
│  │  Agentic Data    │    │     RAG-MCP-Server            │    │
│  │  Cleaner (MCP)   │    │     (MCP)                     │    │
│  │  Data quality    │    │     Domain knowledge          │    │
│  └──────────────────┘    └──────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

**Key Design Decisions:**
- **LangGraph state graph** — all agent flow via conditional edges, typed `MAEDAState` as single source of truth
- **MCP protocol** for sub-system communication — MAEDA delegates, never reimplements
- **Graceful degradation** — runs standalone if sub-systems are offline
- **Every LLM call tracked** — cost tracker, decision trace on every node, eval scores on every run

---

## Quick Start

### 1. Install
```bash
git clone https://github.com/yourhandle/MAEDA.git
cd MAEDA
pip install poetry && poetry install
cp .env.example .env   # add your API key
```

### 2. Generate demo data
```bash
python scripts/generate_demo_data.py
```

### 3. Run the Streamlit UI
```bash
streamlit run ui/app.py
```

### 4. Full ecosystem (Docker)
```bash
docker-compose up maeda                          # MAEDA standalone
docker-compose --profile full-ecosystem up       # with Data Cleaner + RAG Server
```

### 5. Use as an MCP server (Claude Desktop)
Add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "maeda": {
      "command": "python",
      "args": ["-m", "src.mcp_server.server"],
      "cwd": "/path/to/MAEDA"
    }
  }
}
```

---

## Demo Scenarios

| # | Scenario | Dataset | Showcases |
|---|----------|---------|-----------|
| 1 | The Full Stack | `sales_data.csv` | End-to-end pipeline, Data Cleaner integration |
| 2 | Q3 Revenue Drop | `sales_data.csv` | Diagnostic multi-step analysis |
| 3 | Churn Spike | `churn_data.csv` | Root cause analysis + RAG enrichment |
| 4 | Marketing ROI | `marketing_campaigns.csv` | Comparative analysis + charts |
| 5 | Graceful Degradation | `ecommerce_orders.db` | MAEDA without sub-systems |

```bash
python scripts/demo_scenarios.py --scenario 1
python scripts/demo_scenarios.py --all
```

---

## Project Structure

```
src/
├── agents/         Intent parser, Analysis, Viz, Insight, Guardrail
├── eval/           EvalRunner, metrics, 20-case golden suite, regression detection
├── graph/          LangGraph builder, nodes, router
├── mcp_client/     Data Cleaner + RAG Server MCP clients with fallbacks
├── mcp_server/     MAEDA-as-MCP-server (FastMCP)
├── state/          MAEDAState TypedDict
├── tools/          Chart tool, data connector, SQL, stats, anomaly, time-series
├── config/         Settings (Pydantic), all agent prompts
└── utils/          Logger, cost tracker

ui/                 Streamlit app (Phase 11)
scripts/            Demo data generator, demo scenarios
tests/
├── unit/           217+ tests, all phases covered
└── eval/           Golden test suite JSON
data/demo/          Sales, churn, marketing, ecommerce datasets
```

---

## Agent Pipeline

```
User Query
    │
    ▼
Intent Parser      (LLM) → parsed_intent, clarification if needed
    │
    ▼
Data Connector     → schema, NL summary  ──→  Data Cleaner MCP (quality)
    │
    ▼
Analysis Planner   (LLM) → AnalysisPlan (topological execution order)
    │
    ▼
Analysis Executor  → pandas / SQL / stats / anomaly / time-series / comparison tools
    │
    ▼
Viz Agent          → recommend chart → static PNG + Plotly JSON → LLM caption
    │
    ▼
Retrieve Knowledge → RAG-MCP-Server (focused retrieval query)
    │
    ▼
Insight Agent      (LLM) → grounded insights + markdown report
    │
    ▼
Guardrail Agent    → PII check, SQL safety, hallucination (LLM-as-judge), completeness
    │
    ▼
Eval Module        → answer_relevance, groundedness, factual_accuracy, agent metrics
    │
    ▼
Final Report + Charts + Eval Scores
```

---

## Evaluation

Every run is automatically scored:

| Metric | Method |
|--------|--------|
| answer_relevance | LLM-as-judge |
| groundedness | LLM-as-judge |
| factual_accuracy | Number overlap with results |
| intent_accuracy | Confidence + type match |
| tool_selection | Step success rate |
| plan_efficiency | Step count heuristic |
| chart_appropriateness | Chart validity |
| token_cost | Cost tracker |
| retry_count | Guardrail retries |
| error_rate | Run error flag |

Regression detection alerts on any metric drop > 5% vs baseline.

---

## Tests

```bash
pytest tests/unit/ -v          # 251 tests, all phases
pytest tests/unit/test_phase9.py -v   # eval module only
```

---

## Sub-System MCP Interfaces

**Agentic Data Cleaner** (`http://localhost:8001`):
- `profile_dataset(path)` → quality report
- `get_cleaning_plan(path)` → cleaning steps
- `clean_dataset(path, plan)` → cleaned file
- `validate_quality(path)` → pass/fail score

**RAG-MCP-Server** (`http://localhost:8002`):
- `retrieve(query, top_k)` → chunks with scores
- `retrieve_with_metadata(query, top_k)` → chunks with source attribution
- `list_collections()` → available knowledge bases
