"""
MAEDA Streamlit UI — Phase 11.

Production-quality interface exposing the full multi-agent ecosystem.

Layout:
  Sidebar  — data source upload/connect, sub-system health, agent status, token cost
  Main     — chat interface with streaming progress, inline charts, report, trace viewer
  Tabs     — Chat | Report | Charts | Eval | Trace
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

# ─── Page config (must be first Streamlit call) ──────────────────────────────

st.set_page_config(
    page_title="MAEDA — Multi-Agent Enterprise Data Analyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Imports (lazy where possible to avoid import-time LLM construction) ─────


# ─── Session state init ───────────────────────────────────────────────────────

def _init_session():
    defaults = {
        "messages": [],          # list of {role, content}
        "last_result": None,     # latest pipeline result dict
        "data_source_path": None,
        "run_count": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ MAEDA")
    st.caption("Multi-Agent Enterprise Data Analyst")
    st.divider()

    # 11.2 Data source panel
    st.subheader("📂 Data Source")
    upload = st.file_uploader("Upload CSV / Excel / JSON", type=["csv", "xlsx", "json"])
    if upload:
        save_dir = Path("./data/uploads")
        save_dir.mkdir(parents=True, exist_ok=True)
        dest = save_dir / upload.name
        dest.write_bytes(upload.read())
        st.session_state["data_source_path"] = str(dest)
        st.success(f"Loaded: {upload.name}")

    manual_path = st.text_input("Or enter file path / DB URI", placeholder="./data/demo/sales.csv")
    if manual_path:
        st.session_state["data_source_path"] = manual_path

    if st.session_state["data_source_path"]:
        st.info(f"Active: `{Path(st.session_state['data_source_path']).name}`")
        if st.button("Clear source"):
            st.session_state["data_source_path"] = None
            st.rerun()

    st.divider()

    # 11.3 Sub-system health
    st.subheader("🔌 Sub-Systems")
    _col1, _col2 = st.columns(2)
    try:
        import httpx
        dc_ok = httpx.get(
            os.getenv("DATA_CLEANER_MCP_URL", "http://localhost:8001") + "/health",
            timeout=1.0,
        ).status_code == 200
    except Exception:
        dc_ok = False

    try:
        rag_ok = httpx.get(
            os.getenv("RAG_SERVER_MCP_URL", "http://localhost:8002") + "/health",
            timeout=1.0,
        ).status_code == 200
    except Exception:
        rag_ok = False

    _col1.metric("Data Cleaner", "✅ Online" if dc_ok else "⚫ Offline")
    _col2.metric("RAG Server", "✅ Online" if rag_ok else "⚫ Offline")

    st.divider()

    # 11.4 Agent status (populated after a run)
    st.subheader("🤖 Agent Status")
    result = st.session_state.get("last_result")
    if result:
        trace = result.get("decision_trace") or []
        agents_seen = {t["agent_name"] for t in trace}
        for agent in ["intent_parser", "data_connector", "analysis_agent",
                      "viz_agent", "insight_agent", "guardrail_agent", "eval_module"]:
            icon = "✅" if agent in agents_seen else "⬜"
            st.write(f"{icon} {agent.replace('_', ' ').title()}")
    else:
        st.caption("Run a query to see agent status.")

    st.divider()

    # Token cost
    st.subheader("💰 Token Cost")
    if result:
        token_usage = result.get("token_usage") or {}
        total_cost = sum(
            v.get("total_cost", 0) for v in token_usage.values() if isinstance(v, dict)
        )
        total_tokens = sum(
            v.get("total_tokens", 0) for v in token_usage.values() if isinstance(v, dict)
        )
        st.metric("Session Cost", f"${total_cost:.4f}")
        st.metric("Total Tokens", f"{total_tokens:,}")
    else:
        st.caption("No runs yet.")

    st.divider()
    if st.button("🗑️ Clear chat"):
        st.session_state["messages"] = []
        st.session_state["last_result"] = None
        st.rerun()


# ─── Main area ────────────────────────────────────────────────────────────────

st.title("🤖 MAEDA — Multi-Agent Enterprise Data Analyst")

tab_chat, tab_report, tab_charts, tab_eval, tab_trace = st.tabs(
    ["💬 Chat", "📄 Report", "📊 Charts", "🎯 Eval", "🔍 Trace"]
)

# ── 11.1 Chat tab ─────────────────────────────────────────────────────────────
with tab_chat:
    # Display message history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 11.5 Inline charts from last run
    result = st.session_state.get("last_result")
    if result:
        charts = [c for c in (result.get("charts") or [])
                  if c.get("chart_type") != "dashboard" and c.get("image_path")]
        if charts:
            st.markdown("**Generated Charts:**")
            cols = st.columns(min(len(charts), 3))
            for i, chart in enumerate(charts[:3]):
                with cols[i]:
                    path = chart.get("image_path", "")
                    if path and Path(path).exists():
                        st.image(path, caption=chart.get("title", ""), use_container_width=True)
                    # Interactive Plotly chart
                    if chart.get("plotly_json"):
                        try:
                            import plotly.io as pio
                            fig = pio.from_json(chart["plotly_json"])
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            pass

    # Chat input
    if prompt := st.chat_input("Ask a question about your data…"):
        from src.graph.streaming import NODE_LABELS, run_pipeline_streaming

        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if not st.session_state.get("data_source_path"):
            err = "⚠️ No data source loaded. Upload a file or enter a file path in the sidebar first."
            st.session_state["messages"].append({"role": "assistant", "content": err})
            with st.chat_message("assistant"):
                st.warning(err)
            st.stop()

        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            result_placeholder = st.empty()

            status_placeholder.info(NODE_LABELS.get("parse_intent", "Starting..."))

            try:
                _data_source_path = st.session_state.get("data_source_path")

                def _on_node(node_name: str, _state: dict) -> None:
                    # Called synchronously as each graph node actually
                    # completes (via graph.astream()) -- st.empty().info()
                    # pushes an incremental update to the browser
                    # immediately, so this is real per-node progress, not a
                    # fixed-timer animation guessing how far along we are.
                    status_placeholder.info(NODE_LABELS.get(node_name, f"Running {node_name}..."))

                pipeline_result = run_pipeline_streaming(prompt, _data_source_path, on_node=_on_node)

                if pipeline_result.get("current_phase") == "error":
                    err_msg = pipeline_result.get("error") or "Analysis could not complete"
                    status_placeholder.error(f"Error: {err_msg}")
                    response = f"⚠️ {err_msg}"
                else:
                    status_placeholder.success("✅ Analysis complete!")
                    st.session_state["last_result"] = pipeline_result

                    # Show key insights in chat
                    insights = pipeline_result.get("insights") or []
                    if insights:
                        lines = ["**Key Insights:**"]
                        for ins in insights[:3]:
                            lines.append(f"- {ins.get('finding', '')}")
                        response = "\n".join(lines)
                    else:
                        response = pipeline_result.get("report", "Analysis complete. See the Report tab.")[:500]

                    # Guardrail status
                    if not pipeline_result.get("guardrail_passed", True):
                        response += "\n\n⚠️ *Some guardrail checks did not pass — see Trace tab for details.*"

            except Exception as exc:
                status_placeholder.error(f"Pipeline error: {exc}")
                response = f"Error: {exc}"

            result_placeholder.markdown(response)
            st.session_state["messages"].append({"role": "assistant", "content": response})
            st.rerun()


# ── 11.6 Report tab ───────────────────────────────────────────────────────────
with tab_report:
    result = st.session_state.get("last_result")
    if result and result.get("report"):
        st.markdown(result["report"])
        st.download_button(
            "⬇️ Download Report",
            data=result["report"],
            file_name="maeda_report.md",
            mime="text/markdown",
        )
    else:
        st.info("Run a query to generate a report.")


# ── Charts tab ────────────────────────────────────────────────────────────────
with tab_charts:
    result = st.session_state.get("last_result")
    if result:
        charts = result.get("charts") or []
        individual = [c for c in charts if c.get("chart_type") != "dashboard"]
        dashboard = next((c for c in charts if c.get("chart_type") == "dashboard"), None)

        if dashboard and dashboard.get("image_path") and Path(dashboard["image_path"]).exists():
            st.subheader("Dashboard")
            st.image(dashboard["image_path"], use_container_width=True)

        if individual:
            st.subheader("Individual Charts")
            for chart in individual:
                with st.expander(chart.get("title", "Chart"), expanded=True):
                    caption = chart.get("caption", "")
                    if caption:
                        st.caption(caption)

                    # Prefer interactive Plotly
                    if chart.get("plotly_json"):
                        try:
                            import plotly.io as pio
                            fig = pio.from_json(chart["plotly_json"])
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            pass
                    elif chart.get("image_path") and Path(chart["image_path"]).exists():
                        st.image(chart["image_path"], use_container_width=True)
        else:
            st.info("No charts generated.")
    else:
        st.info("Run a query to generate charts.")


# ── 11.8 Eval tab ─────────────────────────────────────────────────────────────
with tab_eval:
    result = st.session_state.get("last_result")
    if result and result.get("eval_scores"):
        scores = result["eval_scores"]
        aggregate = scores.pop("_aggregate", None)

        if aggregate is not None:
            color = "green" if aggregate >= 0.7 else ("orange" if aggregate >= 0.4 else "red")
            st.metric("Aggregate Score", f"{aggregate:.0%}")

        # Show metric scores as a table
        import pandas as pd
        rows = []
        for metric, data in scores.items():
            if isinstance(data, dict):
                rows.append({
                    "Metric": metric.replace("_", " ").title(),
                    "Score": f"{data['score']:.0%}",
                    "Label": data["label"].upper(),
                    "Notes": data.get("reasoning", "")[:80],
                })
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Run a query to see evaluation scores.")


# ── 11.7 Trace tab ────────────────────────────────────────────────────────────
with tab_trace:
    result = st.session_state.get("last_result")
    if result:
        st.subheader("Decision Trace")
        trace = result.get("decision_trace") or []
        for i, entry in enumerate(trace):
            with st.expander(
                f"{i + 1}. [{entry.get('agent_name', '?')}] {entry.get('action', '')}",
                expanded=False
            ):
                col1, col2 = st.columns(2)
                col1.write(f"**Timestamp:** {entry.get('timestamp', '')}")
                col2.write(f"**Confidence:** {entry.get('confidence', '')}")
                st.write(f"**Reasoning:** {entry.get('reasoning', '')}")
                if entry.get("inputs"):
                    st.json(entry["inputs"])

        st.divider()
        st.subheader("MCP Call Log")
        mcp_log = result.get("mcp_call_log") or []
        if mcp_log:
            for log in mcp_log:
                status_icon = "✅" if log.get("success") else "❌"
                st.write(f"{status_icon} `{log.get('tool', '?')}` — {log.get('source', '?')}"
                         f" ({log.get('latency_ms', '?')}ms)")
        else:
            st.caption("No MCP calls logged.")

        st.divider()
        st.subheader("Raw State")
        with st.expander("View full state (JSON)"):
            # Exclude large binary fields
            safe = {k: v for k, v in result.items()
                    if k not in ("charts",) and not isinstance(v, bytes)}
            st.json(safe)
    else:
        st.info("Run a query to see the execution trace.")
