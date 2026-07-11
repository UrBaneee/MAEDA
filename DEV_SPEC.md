# DEV_SPEC.md — Multi-Agent Enterprise Data Analyst (MAEDA)

## Project Overview

**Project Name:** MAEDA (Multi-Agent Enterprise Data Analyst)
**Version:** 2.0
**Target Role:** NVIDIA Solutions Architect, Agentic AI
**Tech Stack:** Python 3.11+ · LangGraph · LangChain · OpenAI/Anthropic API · Streamlit · MCP Protocol
**Architecture Pattern:** Meta-Orchestrator with Sub-System Delegation (LangGraph State Graph)

### What Is This?

A production-grade multi-agent system that takes a natural language question about enterprise data, autonomously connects to data sources, performs analysis, generates visualizations, and delivers an insight report — with built-in evaluation, guardrails, and full decision tracing.

**Key differentiator:** MAEDA is the meta-orchestrator of a 3-project AI agent ecosystem. It delegates data quality tasks to **Agentic Data Cleaner** and domain knowledge retrieval to **RAG-MCP-Server** via MCP protocol, while adding multi-agent coordination, evaluation, guardrails, and visualization on top.

### The Portfolio Ecosystem

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│   MAEDA (Multi-Agent Enterprise Data Analyst)                │
│   ═══════════════════════════════════════════                │
│   LangGraph Orchestrator · Eval · Guardrails · Viz           │
│                                                              │
│   ┌──────────────┐  ┌──────────────┐  ┌────────────────┐    │
│   │ Intent Agent  │  │ Analysis     │  │ Viz Agent      │    │
│   │ (NEW)        │  │ Agent (NEW)  │  │ (NEW)          │    │
│   └──────────────┘  └──────────────┘  └────────────────┘    │
│          │                │                    │             │
│          │     ┌──────────┴──────────┐         │             │
│          ▼     ▼                     ▼         │             │
│   ┌──────────────┐          ┌──────────────┐   │             │
│   │  Agentic     │          │  RAG-MCP     │   │             │
│   │  Data        │◄── MCP ──│  Server      │   │             │
│   │  Cleaner     │          │              │   │             │
│   │  (Project 1) │          │ (Project 2)  │   │             │
│   └──────────────┘          └──────────────┘   │             │
│          │                         │           │             │
│          ▼                         ▼           ▼             │
│   ┌──────────────────────────────────────────────────────┐   │
│   │              Guardrail Agent (NEW)                    │   │
│   └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│                          ▼                                   │
│   ┌──────────────────────────────────────────────────────┐   │
│   │               Eval Module (NEW)                       │   │
│   └──────────────────────────────────────────────────────┘   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### NVIDIA JD Alignment

| NVIDIA JD Requirement | MAEDA Coverage |
|---|---|
| Agentic frameworks for enterprise data retrieval & insight generation | LangGraph orchestrating specialized agents |
| Deep research assistants | Analysis Agent + RAG-powered domain knowledge |
| Multi-modal dialogue systems | Text + CSV + SQL + chart inputs/outputs |
| Evaluation harnesses & success metrics | Built-in eval module (accuracy, groundedness, relevance) |
| Guardrail frameworks for safe, reliable, production-ready workflows | Guardrail Agent + output validation layer |
| MCP or A2A communication protocols | MCP client (calls sub-systems) + MCP server (exposes MAEDA) |
| Fine-tuning and prompt engineering | Optimized system prompts per agent role |
| Production-grade deployment patterns | Docker Compose + graceful degradation |

### What Each Project Contributes (Zero Overlap)

| Capability | Project 1: Data Cleaner | Project 2: RAG-MCP | Project 3: MAEDA |
|---|:---:|:---:|:---:|
| Single Agent Orchestration | ✅ built from scratch | | |
| Plan-Execute Pattern | ✅ | | |
| Decision Tracing | ✅ | | ✅ cross-system |
| Guards / Validation | ✅ | | |
| RAG (Hybrid Retrieval) | | ✅ core | via MCP → Project 2 |
| Vector Search (FAISS) | | ✅ | via MCP → Project 2 |
| MCP Server | | ✅ core | ✅ exposes MAEDA |
| MCP Client | | | ✅ calls Projects 1 & 2 |
| **Multi-Agent Coordination** | | | **✅ LangGraph** |
| **Intent Parsing + Clarification** | | | **✅** |
| **Auto Visualization** | | | **✅** |
| **Guardrail Framework** | | | **✅** |
| **Eval Harness** | | | **✅** |
| **Token Budget Management** | | | **✅** |
| **Graceful Degradation** | | | **✅** |

---

## Architecture

### Core Design Principles

1. **Graph-Based Orchestration:** LangGraph state graph defines agent flow with conditional edges, cycles (for retry/refinement), and human-in-the-loop breakpoints
2. **Sub-System Delegation:** Data quality → Agentic Data Cleaner via MCP; Knowledge retrieval → RAG-MCP-Server via MCP. MAEDA never rebuilds what sub-systems already do.
3. **Typed State:** All inter-agent communication uses TypedDict state objects — no unstructured message passing
4. **Decision Tracing:** Every agent decision is logged with reasoning, inputs, outputs, and confidence scores — unified trace across all 3 systems
5. **Fail-Safe by Default:** Guardrail Agent validates all outputs before user delivery; fallback paths for every failure mode
6. **Graceful Degradation:** If a sub-system is offline, MAEDA falls back to basic built-in alternatives
7. **Eval-First:** Evaluation is not an afterthought — it's a first-class module that runs on every execution

---

## Project Structure

```
maeda/
├── pyproject.toml
├── .env.example
├── docker-compose.yml           # Launches MAEDA + Data Cleaner + RAG Server
├── Dockerfile
├── CLAUDE.md
├── DEV_SPEC.md
├── README.md
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py          # Pydantic Settings (env vars, model config)
│   │   └── agent_prompts.py     # All system prompts centralized
│   ├── state/
│   │   ├── __init__.py
│   │   └── graph_state.py       # TypedDict state definitions
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── builder.py           # LangGraph graph construction
│   │   ├── router.py            # Conditional edge logic
│   │   └── nodes.py             # Node function registry
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py        # Abstract base class
│   │   ├── intent_parser.py     # Parse NL queries
│   │   ├── analysis_agent.py    # Plan + execute analysis
│   │   ├── viz_agent.py         # Auto visualization
│   │   ├── insight_agent.py     # Generate insights (uses RAG via MCP)
│   │   └── guardrail_agent.py   # Validate outputs
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── sql_tool.py
│   │   ├── csv_tool.py
│   │   ├── stats_tool.py
│   │   └── chart_tool.py
│   ├── mcp_client/
│   │   ├── __init__.py
│   │   ├── client.py            # Unified MCP client for sub-systems
│   │   ├── data_cleaner.py      # Data Cleaner MCP integration
│   │   ├── rag_server.py        # RAG-MCP-Server integration
│   │   └── fallback.py          # Graceful degradation handlers
│   ├── mcp_server/
│   │   ├── __init__.py
│   │   └── server.py            # Expose MAEDA as MCP server
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── evaluator.py
│   │   ├── metrics.py
│   │   └── test_cases.py
│   ├── guardrails/
│   │   ├── __init__.py
│   │   ├── validators.py
│   │   ├── hallucination_check.py
│   │   └── pii_filter.py
│   ├── memory/
│   │   ├── __init__.py
│   │   └── conversation.py      # Conversation memory for multi-turn
│   └── utils/
│       ├── __init__.py
│       ├── logger.py            # Structured logging + decision trace
│       └── cost_tracker.py      # Token usage tracking
├── ui/
│   ├── app.py                   # Streamlit main app
│   ├── components/
│   │   ├── chat.py
│   │   ├── data_preview.py
│   │   ├── chart_display.py
│   │   ├── trace_viewer.py      # Unified decision trace across 3 systems
│   │   └── eval_dashboard.py
│   └── assets/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── eval/
│       ├── test_suite.json      # Golden test cases
│       └── run_eval.py
├── data/
│   ├── sample/                  # Sample datasets for demo
│   └── knowledge_base/          # Domain docs for RAG (indexed by RAG-MCP-Server)
└── docs/
    ├── architecture.md
    ├── agent_design.md
    └── eval_report.md
```

---

## LangGraph State Definition

```python
# src/state/graph_state.py
from typing import TypedDict, Optional, Literal

class MAEDAState(TypedDict):
    # === User Input ===
    user_query: str
    conversation_history: list[dict]

    # === Intent Parsing ===
    parsed_intent: dict       # {type, entities, constraints, ambiguity_score}
    clarification_needed: bool
    clarification_question: Optional[str]

    # === Data Connection ===
    data_sources: list[dict]  # [{type, path/uri, schema, preview}]
    active_source: Optional[dict]
    schema_summary: str

    # === Data Quality (DELEGATED to Data Cleaner via MCP) ===
    data_quality_report: Optional[dict]   # From Data Cleaner MCP
    cleaning_applied: bool
    cleaning_summary: Optional[str]

    # === Analysis ===
    analysis_plan: list[dict]    # [{step, method, rationale}]
    analysis_results: list[dict] # [{step, result, confidence}]
    intermediate_data: Optional[dict]

    # === Visualization ===
    charts: list[dict]  # [{type, config, image_path}]

    # === Insight Generation (RAG via MCP) ===
    rag_context: list[dict]    # From RAG-MCP-Server
    rag_sources: list[dict]    # Source attribution from RAG
    insights: list[dict]       # [{finding, evidence, confidence, recommendation}]
    report: Optional[str]      # Final markdown report

    # === Guardrails ===
    guardrail_checks: list[dict]
    guardrail_passed: bool

    # === Eval ===
    eval_scores: Optional[dict]  # {accuracy, groundedness, relevance}

    # === Meta ===
    decision_trace: list[dict]   # Unified trace across all 3 systems
    mcp_call_log: list[dict]     # All MCP calls to sub-systems
    token_usage: dict            # {agent_name: {input, output, cost}}
    current_phase: Literal["plan", "execute", "synthesize", "guardrail", "complete", "error"]
    error: Optional[str]
    iteration_count: int         # For retry loops
```

---

## LangGraph Graph Construction

```python
# src/graph/builder.py
from langgraph.graph import StateGraph, END
from src.state.graph_state import MAEDAState

def build_graph() -> StateGraph:
    graph = StateGraph(MAEDAState)

    # === MAEDA's Own Agents (all NEW) ===
    graph.add_node("parse_intent", parse_intent_node)
    graph.add_node("plan_analysis", plan_analysis_node)
    graph.add_node("execute_analysis", execute_analysis_node)
    graph.add_node("generate_viz", generate_viz_node)
    graph.add_node("generate_insights", generate_insights_node)
    graph.add_node("run_guardrails", run_guardrails_node)
    graph.add_node("run_eval", run_eval_node)
    graph.add_node("ask_clarification", ask_clarification_node)
    graph.add_node("handle_error", handle_error_node)

    # === Delegated to Sub-Systems via MCP ===
    graph.add_node("connect_and_profile_data", connect_and_profile_node)
    #   → Calls Data Cleaner MCP: profile_dataset, clean_dataset
    graph.add_node("retrieve_domain_knowledge", retrieve_knowledge_node)
    #   → Calls RAG-MCP-Server: retrieve_with_metadata

    # === Entry Point ===
    graph.set_entry_point("parse_intent")

    # === Flow ===

    # Intent → clarify or proceed
    graph.add_conditional_edges("parse_intent", route_after_intent, {
        "proceed": "connect_and_profile_data",
        "clarify": "ask_clarification",
    })
    graph.add_edge("ask_clarification", "parse_intent")  # Clarification loop

    # Data profiling (delegates to Data Cleaner via MCP)
    graph.add_conditional_edges("connect_and_profile_data", route_after_profiling, {
        "clean": "connect_and_profile_data",  # Re-profile after cleaning
        "ready": "plan_analysis",
    })

    # Analysis pipeline
    graph.add_edge("plan_analysis", "execute_analysis")
    graph.add_edge("execute_analysis", "generate_viz")

    # Insight generation (delegates RAG to RAG-MCP-Server via MCP)
    graph.add_edge("generate_viz", "retrieve_domain_knowledge")
    graph.add_edge("retrieve_domain_knowledge", "generate_insights")

    # Guardrails + Eval
    graph.add_edge("generate_insights", "run_guardrails")
    graph.add_conditional_edges("run_guardrails", route_after_guardrails, {
        "passed": "run_eval",
        "retry": "execute_analysis",   # Feedback loop!
        "fail": "handle_error",
    })
    graph.add_edge("run_eval", END)
    graph.add_edge("handle_error", END)

    return graph.compile()
```

---

## Phase Breakdown

### Phase 1: Project Scaffold & LangGraph Foundation (8 tasks)

**Goal:** Set up project structure, LangGraph state graph skeleton, and basic orchestration loop.

| # | Task | Description | Acceptance Criteria |
|---|------|-------------|---------------------|
| 1.1 | Init project | pyproject.toml with all deps, .env.example | `poetry install` succeeds |
| 1.2 | State definition | Implement MAEDAState TypedDict | All fields typed, importable |
| 1.3 | Graph skeleton | Build LangGraph with placeholder nodes | Graph compiles, visualizable |
| 1.4 | Router logic | Implement conditional edge functions | Routes correctly on mock state |
| 1.5 | Logger setup | Structured JSON logging + decision trace | Logs with trace_id, agent_name, action |
| 1.6 | Cost tracker | Token counting per agent per call | Accurate per-call tracking |
| 1.7 | Base agent class | Abstract base with common interface | `process()`, `log_decision()`, `track_cost()` |
| 1.8 | Config system | Pydantic Settings + prompt management | Env vars loaded, prompts centralized |

---

### Phase 2: Intent Parser Agent (6 tasks)

**Goal:** Parse natural language queries into structured analysis intents.

The Intent Parser classifies query type, extracts entities, detects ambiguity, and handles multi-turn clarification.

#### Intent Schema

```python
@dataclass
class ParsedIntent:
    query_type: Literal["descriptive", "diagnostic", "predictive", "comparative", "exploratory"]
    target_metrics: list[str]       # e.g., ["revenue", "growth_rate"]
    dimensions: list[str]           # e.g., ["region", "quarter"]
    filters: list[dict]             # e.g., [{"column": "year", "op": ">=", "value": 2023}]
    time_range: Optional[dict]
    aggregation: Optional[str]
    sort_by: Optional[str]
    limit: Optional[int]
    confidence: float               # 0.0 - 1.0, triggers clarification if < 0.7
    ambiguities: list[str]
    raw_query: str
```

| # | Task | Description | Acceptance Criteria |
|---|------|-------------|---------------------|
| 2.1 | Intent prompt | Design system prompt for intent parsing | Parses 5 sample queries correctly |
| 2.2 | Parser implementation | LLM-based parsing with structured output | Returns ParsedIntent dataclass |
| 2.3 | Ambiguity detection | Confidence scoring + ambiguity list | Low-confidence triggers clarification |
| 2.4 | Clarification flow | Generate clarification question, re-parse on answer | Multi-turn loop works |
| 2.5 | Schema-aware parsing | Inject data source schema to improve entity extraction | Column names correctly identified |
| 2.6 | Unit tests | Test suite for intent parser | 90%+ accuracy on test cases |

---

### Phase 3: MCP Integration Layer (8 tasks)

**Goal:** Build the MCP client that calls Data Cleaner and RAG-MCP-Server as sub-systems.

#### MCP Client Architecture

```python
# src/mcp_client/client.py
class SubSystemClient:
    """Unified MCP client for calling sub-system projects."""

    def __init__(self, config: MCPConfig):
        self.data_cleaner = MCPClient(config.data_cleaner_url)
        self.rag_server = MCPClient(config.rag_server_url)

    # === Data Cleaner Delegation ===
    async def profile_dataset(self, source_path: str) -> DataQualityReport:
        result = await self.data_cleaner.call_tool("profile_dataset", {"path": source_path})
        return DataQualityReport.from_mcp_response(result)

    async def clean_dataset(self, source_path: str, plan=None) -> CleaningResult:
        result = await self.data_cleaner.call_tool("clean_dataset", {"path": source_path, "plan": plan})
        return CleaningResult.from_mcp_response(result)

    async def get_cleaning_plan(self, source_path: str) -> CleaningPlan:
        result = await self.data_cleaner.call_tool("get_cleaning_plan", {"path": source_path})
        return CleaningPlan.from_mcp_response(result)

    async def validate_quality(self, source_path: str) -> QualityValidation:
        result = await self.data_cleaner.call_tool("validate_quality", {"path": source_path})
        return QualityValidation.from_mcp_response(result)

    # === RAG Server Delegation ===
    async def retrieve_knowledge(self, query: str, top_k: int = 5) -> list[RAGChunk]:
        result = await self.rag_server.call_tool("retrieve_with_metadata", {"query": query, "top_k": top_k})
        return [RAGChunk.from_mcp_response(r) for r in result]

    async def list_collections(self) -> list[Collection]:
        result = await self.rag_server.call_tool("list_collections", {})
        return [Collection.from_mcp_response(c) for c in result]
```

#### Fallback Behavior

```python
# src/mcp_client/fallback.py
class SubSystemWithFallback:
    """Graceful degradation when sub-systems are offline."""

    async def profile_dataset(self, source_path: str) -> DataQualityReport:
        try:
            return await self.client.profile_dataset(source_path)
        except MCPConnectionError:
            logger.warning("Data Cleaner unavailable, using basic profiling")
            return self._basic_pandas_profile(source_path)

    async def retrieve_knowledge(self, query: str) -> list[RAGChunk]:
        try:
            return await self.client.retrieve_knowledge(query)
        except MCPConnectionError:
            logger.warning("RAG Server unavailable, skipping domain enrichment")
            return []
```

#### Sub-System MCP Tools (Already Implemented)

**Data Cleaner exposes:**
- `profile_dataset`: input {path} → {row_count, columns, quality_issues}
- `get_cleaning_plan`: input {path} → {steps: [{operation, target_column, rationale, estimated_impact}]}
- `clean_dataset`: input {path, plan?} → {cleaned_path, changes_summary, rows_affected}
- `validate_quality`: input {path} → {passed, score, issues}

**RAG-MCP-Server exposes:**
- `retrieve`: input {query, top_k} → {chunks: [{content, score, metadata}]}
- `retrieve_with_metadata`: input {query, top_k} → {chunks: [{content, score, source_file, page, chunk_id}]}
- `list_collections`: input {} → {collections: [{name, doc_count, description}]}

| # | Task | Description | Acceptance Criteria |
|---|------|-------------|---------------------|
| 3.1 | MCP client base | Async MCP client with connection management | Connects to running MCP servers |
| 3.2 | Data Cleaner integration | profile + clean + validate calls | Gets quality report from Data Cleaner |
| 3.3 | RAG Server integration | retrieve + retrieve_with_metadata calls | Gets relevant chunks from RAG |
| 3.4 | Fallback layer | Graceful degradation when sub-systems offline | MAEDA works standalone (basic mode) |
| 3.5 | MCP call logging | Log all MCP calls with timing and results | Full audit trail of sub-system calls |
| 3.6 | Response parsing | Parse MCP responses into typed dataclasses | Type-safe data across system boundary |
| 3.7 | Health check | Periodic health check of sub-system availability | Status displayed in UI |
| 3.8 | Integration test | Test MAEDA ↔ Data Cleaner ↔ RAG full chain | Three services communicate correctly |

---

### Phase 4: Data Source Connection (6 tasks)

**Goal:** Connect to data sources, get schema info, delegate quality checks to Data Cleaner.

MAEDA handles the connection and schema understanding. Data quality profiling and cleaning are delegated to Data Cleaner via MCP.

#### Supported Data Sources

| Source Type | Implementation | Priority |
|---|---|---|
| CSV/TSV | pandas read_csv | P0 |
| SQLite | sqlite3 + sqlalchemy | P0 |
| PostgreSQL | psycopg2 + sqlalchemy | P1 |
| JSON/JSONL | pandas read_json | P1 |
| Excel (.xlsx) | openpyxl | P1 |

#### Node Implementation

```python
async def connect_and_profile_node(state: MAEDAState) -> MAEDAState:
    # Step 1: MAEDA connects to data source and reads schema
    source = connect_source(state["data_sources"][0])
    schema = extract_schema(source)  # column names, types, row count

    # Step 2: Delegate quality profiling to Data Cleaner via MCP
    quality_report = await mcp_client.profile_dataset(source["path"])

    # Step 3: If quality issues found, delegate cleaning
    if quality_report.has_critical_issues:
        plan = await mcp_client.get_cleaning_plan(source["path"])
        result = await mcp_client.clean_dataset(source["path"], plan)
        state["cleaning_applied"] = True
        state["cleaning_summary"] = result.changes_summary
        # Re-read schema from cleaned data
        schema = extract_schema({"path": result.cleaned_path})

    state["schema_summary"] = schema.to_natural_language()
    state["data_quality_report"] = quality_report.to_dict()
    return state
```

| # | Task | Description | Acceptance Criteria |
|---|------|-------------|---------------------|
| 4.1 | CSV connector | Load CSV, extract schema | Works with messy CSVs |
| 4.2 | SQL connector | SQLAlchemy connection + schema introspection | Auto-detects tables/columns |
| 4.3 | NL schema summary | LLM generates human-readable data description | Accurate for 5 test datasets |
| 4.4 | MCP quality delegation | Call Data Cleaner for profiling + cleaning | Quality report returned, cleaning executed |
| 4.5 | JSON/Excel connectors | Support additional formats | Load and schema extract correctly |
| 4.6 | Source registry | Register and manage multiple data sources | Multi-source queries possible |

---

### Phase 5: Analysis Agent (8 tasks)

**Goal:** Plan and execute data analysis based on parsed intent, using tool-calling pattern.

#### Analysis Tools (LangChain Tools)

```python
tools = [
    SQLQueryTool,          # Execute SQL queries
    PandasTransformTool,   # DataFrame operations (groupby, pivot, merge)
    StatisticalTestTool,   # t-test, chi-square, correlation, regression
    AnomalyDetectionTool,  # IQR, Z-score, isolation forest
    TimeSeriesAnalysisTool, # Trend, seasonality, basic forecast
    ComparisonTool,        # Cross-segment comparison with significance testing
]
```

#### Analysis Plan Pattern

```python
@dataclass
class AnalysisPlan:
    steps: list[AnalysisStep]
    estimated_complexity: Literal["simple", "moderate", "complex"]
    rationale: str

@dataclass
class AnalysisStep:
    step_number: int
    method: str             # e.g., "groupby_aggregate", "correlation_matrix"
    tool: str               # Which tool to invoke
    parameters: dict
    depends_on: list[int]   # Step dependencies
    expected_output: str
    rationale: str
```

| # | Task | Description | Acceptance Criteria |
|---|------|-------------|---------------------|
| 5.1 | Plan generator | LLM generates AnalysisPlan from intent + schema | Logical plan for 5 query types |
| 5.2 | SQL query tool | Generate and execute SQL | Correct results on test DB |
| 5.3 | Pandas transform tool | GroupBy, pivot, merge, filter | Handles common transforms |
| 5.4 | Statistical tool | Correlation, regression, hypothesis tests | Matches scipy results |
| 5.5 | Anomaly detection | Z-score, IQR-based outlier detection | Detects known anomalies in test data |
| 5.6 | Step executor | Execute plan steps with dependency tracking | Handles step dependencies correctly |
| 5.7 | Error recovery | Retry failed steps, adjust plan if step fails | Recovers from 1 failed step |
| 5.8 | Result aggregator | Combine multi-step results into coherent output | Results structured and traceable |

---

### Phase 6: Visualization Agent (6 tasks)

**Goal:** Automatically generate appropriate charts based on data type and analysis results.

#### Chart Selection Logic

```
Metric over time         → Line chart
Category comparison      → Bar chart (horizontal if >6 categories)
Distribution             → Histogram / box plot
Correlation              → Scatter plot / heatmap
Part-of-whole            → Pie chart (≤6 segments) / treemap
Multi-metric comparison  → Grouped bar / radar chart
```

Implementation: matplotlib + seaborn (static), plotly (interactive for Streamlit)

| # | Task | Description | Acceptance Criteria |
|---|------|-------------|---------------------|
| 6.1 | Chart recommender | Select chart type from data shape + intent | Correct recommendation 90%+ |
| 6.2 | Chart generator | Generate charts with matplotlib/seaborn | Clean, readable charts |
| 6.3 | Interactive charts | Plotly versions for Streamlit | Hover, zoom, filter work |
| 6.4 | Multi-chart layout | Generate dashboard-style multi-chart output | Layout is logical and readable |
| 6.5 | Chart captioning | LLM generates natural language chart descriptions | Captions match visual accurately |
| 6.6 | Styling system | Consistent theme, labels, formatting | Professional look, all charts consistent |

---

### Phase 7: Insight Agent (6 tasks)

**Goal:** Generate actionable insights by combining analysis results with domain knowledge from RAG-MCP-Server.

MAEDA does NOT build its own RAG. It calls RAG-MCP-Server via MCP for domain knowledge retrieval.

#### Insight Structure

```python
@dataclass
class Insight:
    finding: str              # What was found
    evidence: list[str]       # Data points supporting the finding
    confidence: float
    domain_context: str       # From RAG-MCP-Server
    impact: Literal["high", "medium", "low"]
    recommendation: str       # Actionable next step
    sources: list[str]        # RAG source references
```

#### Node Implementation

```python
async def retrieve_knowledge_node(state: MAEDAState) -> MAEDAState:
    # Build retrieval query from analysis results
    query = synthesize_retrieval_query(state["parsed_intent"], state["analysis_results"])

    # Delegate to RAG-MCP-Server
    chunks = await mcp_client.retrieve_knowledge(query, top_k=5)

    state["rag_context"] = [c.content for c in chunks]
    state["rag_sources"] = [{"source_file": c.source_file, "page": c.page, "chunk_id": c.chunk_id} for c in chunks]
    return state
```

| # | Task | Description | Acceptance Criteria |
|---|------|-------------|---------------------|
| 7.1 | Retrieval query builder | Build RAG query from analysis results + intent | Relevant queries generated |
| 7.2 | MCP retrieval call | Call RAG-MCP-Server for domain knowledge | Chunks returned with metadata |
| 7.3 | Insight generator | Combine analysis + RAG context → insights | Insights are grounded, not hallucinated |
| 7.4 | Confidence scoring | Score each insight's confidence | Scores correlate with evidence quality |
| 7.5 | Report generator | Produce markdown report from insights | Report is clear, structured, complete |
| 7.6 | Source attribution | Track which RAG sources informed each insight | Every domain claim has a source |

---

### Phase 8: Guardrail Agent (8 tasks)

**Goal:** Validate all outputs for safety, accuracy, and reliability before user delivery.

#### Guardrail Checks

```python
GUARDRAIL_PIPELINE = [
    # Accuracy
    "numerical_consistency",    # Numbers in report match analysis results?
    "sql_injection_check",      # Generated SQL is safe?
    "statistical_validity",     # Statistical claims valid?

    # Hallucination
    "claim_grounding",          # Every claim traceable to data or RAG source?
    "fabrication_detection",    # Made-up statistics or entities?

    # Safety
    "pii_detection",            # PII leakage in outputs?
    "bias_check",               # Biased conclusions?

    # Quality
    "completeness_check",       # Report addresses original query?
    "readability_check",        # Output clear and well-structured?
]
```

#### Failure Handling

- **Critical fail** (hallucination): Block output, retry with stricter constraints (max 2 retries)
- **Warning** (readability): Attach caveat, deliver
- **Info**: Log only, deliver normally

| # | Task | Description | Acceptance Criteria |
|---|------|-------------|---------------------|
| 8.1 | Numerical consistency | Cross-check report numbers vs raw results | Catches mismatched numbers |
| 8.2 | Claim grounding | Verify every claim has data/source evidence | Detects ungrounded claims |
| 8.3 | SQL safety | Validate generated SQL | Blocks dangerous SQL |
| 8.4 | PII filter | Detect and redact PII in outputs | Catches email, SSN, phone |
| 8.5 | Hallucination detector | LLM-as-judge for fabricated content | Catches fake stats |
| 8.6 | Completeness check | Does output address the query? | Detects incomplete answers |
| 8.7 | Guardrail orchestrator | Run all checks, aggregate, route decision | Correct routing on pass/fail/retry |
| 8.8 | Retry mechanism | On critical fail, retry with constraints | Recovers on retry |

---

### Phase 9: Evaluation Module (8 tasks)

**Goal:** Automated evaluation of the entire pipeline. THE differentiator for NVIDIA interviews.

#### Eval Metrics

```python
EVAL_METRICS = {
    # Output Quality
    "answer_relevance": "Does the output answer the user's question?",
    "groundedness": "Is every claim supported by data or retrieved context?",
    "factual_accuracy": "Are numerical values and statistical claims correct?",
    "completeness": "Does the analysis cover all aspects of the query?",

    # Agent Performance
    "intent_accuracy": "Did the Intent Parser correctly understand the query?",
    "tool_selection": "Did the Analysis Agent choose appropriate tools?",
    "plan_efficiency": "Was the analysis plan efficient?",
    "chart_appropriateness": "Was the chart type appropriate?",

    # System Performance
    "total_latency": "End-to-end execution time",
    "token_cost": "Total tokens consumed across all agents",
    "retry_count": "Number of guardrail-triggered retries",
    "error_rate": "Percentage of runs that ended in error",
}
```

| # | Task | Description | Acceptance Criteria |
|---|------|-------------|---------------------|
| 9.1 | Eval framework | Implement evaluation runner | Runs test suite, produces scores |
| 9.2 | Answer relevance metric | LLM-as-judge for relevance | Correlates with human judgment |
| 9.3 | Groundedness metric | Claim-to-source traceability | Catches ungrounded claims |
| 9.4 | Factual accuracy metric | Validate numbers against ground truth | Catches numerical errors |
| 9.5 | Agent performance metrics | Per-agent scoring | Identifies weakest agent |
| 9.6 | Golden test suite | 20+ test cases with ground truth | Covers all 5 query types |
| 9.7 | Eval dashboard | Streamlit page for results viz | Interactive, filterable |
| 9.8 | Regression detection | Compare eval runs, alert on degradation | Detects score drops > 5% |

---

### Phase 10: MAEDA MCP Server (6 tasks)

**Goal:** Expose MAEDA itself as an MCP server, so it can be called by Claude Desktop or other MCP clients.

#### MCP Tools Exposed

```python
MCP_TOOLS = [
    {
        "name": "analyze_data",
        "description": "Analyze a dataset using multi-agent pipeline.",
        "parameters": {
            "query": "string — Natural language analysis question",
            "data_source": "string — Path to CSV/DB or connection string",
        }
    },
    {
        "name": "connect_data_source",
        "description": "Register a new data source for analysis.",
        "parameters": {
            "source_type": "string — csv|sqlite|postgres|json|excel",
            "path_or_uri": "string — File path or connection URI",
        }
    },
    {
        "name": "get_eval_report",
        "description": "Get the latest evaluation report.",
        "parameters": {}
    }
]
```

| # | Task | Description | Acceptance Criteria |
|---|------|-------------|---------------------|
| 10.1 | MCP server skeleton | FastMCP server setup | Server starts, lists tools |
| 10.2 | analyze_data tool | Full pipeline as MCP tool | Claude can invoke analysis |
| 10.3 | connect_data_source tool | Register sources via MCP | Source registered and usable |
| 10.4 | get_eval_report tool | Return eval info via MCP | Correct report returned |
| 10.5 | Streaming support | Stream progress updates during analysis | Client receives progress |
| 10.6 | Integration test | Test with Claude Desktop / MCP Inspector | Full round-trip works |

---

### Phase 11: Streamlit UI (8 tasks)

**Goal:** Production-quality UI that showcases the full ecosystem.

#### UI Layout

```
┌─────────────────────────────────────────────────┐
│  MAEDA — Multi-Agent Enterprise Data Analyst    │
├─────────────────┬───────────────────────────────┤
│                 │                               │
│  SIDEBAR        │   MAIN AREA                   │
│                 │                               │
│  Data Sources   │   Chat Interface              │
│  ├ Upload CSV   │   ┌───────────────────────┐   │
│  ├ Connect DB   │   │ User: Why did revenue │   │
│  └ Source List  │   │ drop in Q3?           │   │
│                 │   │                       │   │
│  Sub-Systems    │   │ 🔌 Data Cleaner: ✅   │   │
│  ├ Data Cleaner │   │ 🔌 RAG Server: ✅     │   │
│  └ RAG Server   │   │                       │   │
│  (health status)│   │ 🤖 Parsing intent...  │   │
│                 │   │ 🤖 Profiling data...  │   │
│  Agent Status   │   │ 🤖 Analyzing...       │   │
│  ├ Intent ✅    │   │ 🤖 Generating viz...  │   │
│  ├ Profile ✅   │   │                       │   │
│  ├ Analysis ⏳  │   │ [CHART EMBEDDED]      │   │
│  ├ Viz ⬜       │   │                       │   │
│  ├ Insight ⬜   │   │ Key Findings:         │   │
│  ├ Guard ⬜     │   │ 1. Revenue dropped... │   │
│  └ Eval ⬜      │   └───────────────────────┘   │
│                 │                               │
│  Token Cost     │   [Report] [Charts] [Trace]   │
│  $0.023         │                               │
└─────────────────┴───────────────────────────────┘
```

| # | Task | Description | Acceptance Criteria |
|---|------|-------------|---------------------|
| 11.1 | Chat interface | Message-based UI with streaming | Messages stream in real-time |
| 11.2 | Data source panel | Upload/connect/preview data | CSV upload and DB connect work |
| 11.3 | Sub-system health | Show Data Cleaner + RAG Server status | Live health indicators |
| 11.4 | Agent status tracker | Real-time agent progress display | Shows which agent is active |
| 11.5 | Chart embedding | Display charts inline in chat | Charts render correctly |
| 11.6 | Report view | Formatted markdown report display | Report is readable and complete |
| 11.7 | Unified trace viewer | Decision trace across all 3 systems | Full cross-system trace viewable |
| 11.8 | Eval dashboard tab | Show eval metrics and test results | Dashboard renders with data |

---

### Phase 12: Ecosystem Demo & Datasets (6 tasks)

**Goal:** Killer demo showing all 3 projects as one system.

#### Demo Datasets

| Dataset | Description | Showcases |
|---|---|---|
| `sales_data.csv` | 3 years of product sales across regions (with intentional quality issues) | Data Cleaner cleans → Analysis → Viz |
| `churn_data.csv` | Customer churn with behavioral features | Diagnostic analysis + RAG enrichment |
| `marketing_campaigns.csv` | Campaign spend vs conversions | Comparative analysis |
| `ecommerce_orders.db` | SQLite DB with orders, products, customers | Multi-table SQL joins |

#### Demo Scenarios

1. **The Full Stack:** Upload messy CSV → Data Cleaner auto-cleans → Analysis → RAG enriches → Report. Shows entire ecosystem.
2. **Diagnostic:** "Why did churn spike in March?" → Multi-step diagnostic with root cause analysis.
3. **Clarification Loop:** "Tell me everything" (vague) → Intent Parser asks for clarification → User refines → Analysis.
4. **Guardrail Demo:** Feed inconsistent data → Guardrail catches issues → Retry with adjusted constraints.
5. **Graceful Degradation:** Kill RAG Server → MAEDA still works, just without domain enrichment.

| # | Task | Description | Acceptance Criteria |
|---|------|-------------|---------------------|
| 12.1 | Sales dataset | 10K+ rows, realistic patterns, intentional quality issues | Data Cleaner finds and fixes issues |
| 12.2 | Churn dataset | Clear discoverable root causes | Analysis Agent identifies them |
| 12.3 | Marketing dataset | Clear spend-conversion relationships | Comparative analysis produces insights |
| 12.4 | Docker Compose | All 3 projects start with one command | `docker-compose up` launches ecosystem |
| 12.5 | Demo script | 5 scripted scenarios for interviews | Each highlights different capability |
| 12.6 | Screen recording / GIF | Record demo for README | Shows full ecosystem in action |

---

### Phase 13: Documentation & Interview Prep (4 tasks)

| # | Task | Description | Acceptance Criteria |
|---|------|-------------|---------------------|
| 13.1 | README.md | Comprehensive README with demo GIF, architecture diagram, quick start | Clear, professional, with visuals |
| 13.2 | Architecture doc | Detailed architecture + design decisions | Covers why LangGraph, why MCP, why eval |
| 13.3 | Eval report doc | Published eval results and analysis | Quantitative proof of quality |
| 13.4 | Interview talking points | 5 stories mapped to NVIDIA JD | Ready to tell in behavioral interviews |

---

## Task Summary

| Phase | Name | Tasks |
|---|---|---|
| 1 | Project Scaffold & LangGraph Foundation | 8 |
| 2 | Intent Parser Agent | 6 |
| 3 | MCP Integration Layer | 8 |
| 4 | Data Source Connection | 6 |
| 5 | Analysis Agent | 8 |
| 6 | Visualization Agent | 6 |
| 7 | Insight Agent (RAG via MCP) | 6 |
| 8 | Guardrail Agent | 8 |
| 9 | Evaluation Module | 8 |
| 10 | MAEDA MCP Server | 6 |
| 11 | Streamlit UI | 8 |
| 12 | Ecosystem Demo & Datasets | 6 |
| 13 | Documentation & Interview Prep | 4 |
| **TOTAL** | | **88** |

---

## Technical Decisions & Rationale

### Why LangGraph over CrewAI/AutoGen?

State graph model gives precise control over agent flow. Conditional edges enable retry loops, clarification cycles, and error recovery. Most recognized framework in NVIDIA and Google JDs. Complements existing portfolio: Data Cleaner shows pure orchestration; MAEDA shows framework mastery.

### Why MCP for Inter-Project Communication?

MCP is the emerging standard for agent-tool communication. NVIDIA's JD explicitly mentions it. Each sub-system is independently deployable and testable. If you swap out the RAG implementation, MAEDA doesn't know the difference. Interview story: "I designed microservices for AI agents."

### Why Eval as a First-Class Module?

NVIDIA JD explicitly asks for "evaluation harnesses, success metrics." This is the #1 thing most portfolio projects lack. Interview line: "Most people build demo agents. I build agents I can measure."

### Why Guardrails?

NVIDIA JD explicitly asks for "guardrail frameworks to ensure agentic AI workflows are safe, reliable, and production-ready." Interview line: "In production, the agent that checks the agent is more important than the agent itself."

### Why Graceful Degradation?

Shows production mindset. In real deployments, services go down. MAEDA works at three levels: full ecosystem, partial (one sub-system down), and standalone. Interview line: "I don't build demos. I build systems that handle failure."

---

## Dependencies

```toml
[tool.poetry.dependencies]
python = "^3.11"
langgraph = ">=0.2"
langchain = ">=0.3"
langchain-openai = ">=0.2"
langchain-anthropic = ">=0.3"
pandas = ">=2.0"
numpy = ">=1.24"
matplotlib = ">=3.8"
seaborn = ">=0.13"
plotly = ">=5.18"
sqlalchemy = ">=2.0"
streamlit = ">=1.30"
pydantic = ">=2.0"
pydantic-settings = ">=2.0"
httpx = ">=0.27"
python-dotenv = ">=1.0"
tiktoken = ">=0.7"
mcp = ">=1.0"

[tool.poetry.group.dev.dependencies]
pytest = ">=8.0"
pytest-asyncio = ">=0.23"
ruff = ">=0.5"
mypy = ">=1.10"
```

## Environment Variables

```bash
# .env.example
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DEFAULT_MODEL=gpt-4o
EVAL_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small

# Sub-system MCP connections
DATA_CLEANER_MCP_COMMAND=python
DATA_CLEANER_MCP_ARGS=["-m", "mcp_server.mcp_app"]
DATA_CLEANER_MCP_CWD=/path/to/agentic-data-cleaner-v2

RAG_SERVER_MCP_COMMAND=python
RAG_SERVER_MCP_ARGS=["-m", "mcp_server.mcp_app"]
RAG_SERVER_MCP_CWD=/path/to/RAG-MCP-Server

LOG_LEVEL=INFO
TRACE_ENABLED=true
MAX_RETRIES=2
TOKEN_BUDGET_PER_QUERY=50000
```
