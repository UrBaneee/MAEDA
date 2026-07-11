# MAEDA — Interview Talking Points (NVIDIA Solutions Architect, Agentic AI)

Five stories mapped to the NVIDIA JD competencies.

---

## Story 1: "Why we built a meta-orchestrator instead of one big agent"

**Context:** Early prototype had a single agent calling tools sequentially. It was brittle — one failed tool call broke everything, and there was no way to retry specific steps.

**Decision:** Restructured as a LangGraph state graph with 8 specialised agents. Each agent owns one concern. Guardrails are a first-class graph cycle, not an afterthought.

**Result:** The graph handles partial failures gracefully — a failed analysis step is retried with simplified parameters while other steps proceed. The guardrail loop catches hallucinations before they reach the user.

**NVIDIA relevance:** This is the exact pattern at scale — NIM microservices for inference, orchestration layer on top. MAEDA demonstrates understanding of when to decompose vs. when to keep things together.

---

## Story 2: "Building for graceful degradation from day one"

**Context:** The project integrates two external sub-systems (Data Cleaner, RAG Server) via MCP. During development, those systems aren't always running.

**Decision:** Every MCP call is wrapped in `SubSystemWithFallback`. If the Data Cleaner is offline, MAEDA profiles the dataset with pandas. If RAG Server is down, the Insight Agent generates insights from analysis results alone. `MAEDA can run a full demo with no sub-systems and no API key.`

**Result:** Demo reliability went from ~60% (dependent on sub-systems being up) to ~100%. The graceful degradation also became a demo scenario itself — showing the system degrade and recover.

**NVIDIA relevance:** Production AI systems at NVIDIA customers run in constrained environments. Designing for failure modes first is a core Solutions Architect skill.

---

## Story 3: "Eval as a first-class concern, not a post-hoc check"

**Context:** After building the pipeline, I realised I had no way to quantitatively measure quality. Without metrics, I couldn't tell if a change to the Insight Agent prompt improved or degraded output.

**Decision:** Built an EvalModule that runs on every pipeline execution. LLM-as-judge scores answer_relevance and groundedness. Rule-based checks score factual_accuracy and plan_efficiency. Regression detection alerts on any metric drop > 5%.

**Result:** Every run produces a score card. I can compare prompt changes quantitatively. The 20-case golden test suite covers all 5 query types and catches regressions in CI.

**NVIDIA relevance:** NVIDIA customers deploying LLM pipelines ask "how do I know it's working?" This is the answer — automated eval infrastructure, not just manual spot-checks.

---

## Story 4: "Exposing MAEDA as an MCP server so Claude can use it"

**Context:** The project already consumed two MCP servers. The obvious next step was to expose MAEDA itself as an MCP server, making it callable from Claude Desktop.

**Decision:** Implemented a FastMCP server with three tools: `analyze_data`, `connect_data_source`, `get_eval_report`. A user can now ask Claude "analyse sales by region using my CSV" and Claude invokes MAEDA's full pipeline transparently.

**Result:** MAEDA is now both an MCP client and an MCP server — demonstrating the composability of the protocol. The `analyze_data` tool returns progress events, a full report, and eval scores in a single JSON response.

**NVIDIA relevance:** MCP is becoming the integration standard for enterprise AI. Being able to explain and implement the protocol at both ends is directly relevant to NVIDIA's Solutions Architect role.

---

## Story 5: "Decision tracing as the answer to 'why did the AI do that?'"

**Context:** Enterprise customers always ask "why did the system reach this conclusion?" without a good answer, AI adoption stalls.

**Decision:** Every agent logs a structured decision trace record: `{agent_name, action, reasoning, inputs, outputs, confidence, timestamp}`. The Streamlit UI exposes this as a full audit trail. The MCP call log shows exactly which sub-systems were invoked, with latency and success/failure.

**Result:** For any output, you can trace: which intent was parsed → which analysis steps ran → which RAG chunks informed insights → which guardrail checks ran → what the eval score was. Full end-to-end explainability.

**NVIDIA relevance:** This directly addresses enterprise AI governance requirements. A Solutions Architect who can articulate and demonstrate auditability wins deals that a pure ML engineer cannot.
