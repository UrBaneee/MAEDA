"""
Pydantic Settings — all configuration loaded from environment variables.
Import `settings` wherever config is needed; never read os.environ directly.
"""
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class MAEDASettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM Provider ─────────────────────────────────────────────────────────
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")

    llm_provider: Literal["openai", "anthropic"] = Field(
        default="openai", alias="MAEDA_LLM_PROVIDER"
    )
    llm_model: str = Field(default="gpt-4o-mini", alias="MAEDA_LLM_MODEL")
    llm_temperature: float = Field(default=0.0, alias="MAEDA_LLM_TEMPERATURE")

    # ── Token / Cost Limits ───────────────────────────────────────────────────
    max_tokens_per_call: int = Field(default=4096, alias="MAEDA_MAX_TOKENS_PER_CALL")
    max_cost_per_session: float = Field(default=1.00, alias="MAEDA_MAX_COST_PER_SESSION")

    # ── MCP Sub-System URLs ───────────────────────────────────────────────────
    data_cleaner_mcp_url: str = Field(
        default="http://localhost:8001", alias="DATA_CLEANER_MCP_URL"
    )
    rag_server_mcp_url: str = Field(
        default="http://localhost:8002", alias="RAG_SERVER_MCP_URL"
    )
    rag_collection: Optional[str] = Field(
        default=None, alias="MAEDA_RAG_COLLECTION"
    )
    """Scope RAG retrieval to a single collection (e.g. "wake_apparel").
    Left unset, retrieval searches the entire knowledge base, which risks
    surfacing chunks from unrelated documents sharing the same RAG-MCP-Server
    instance (see eval_report.md #23)."""

    mcp_strict_mode: Literal["strict", "degraded"] = Field(
        default="degraded", alias="MCP_STRICT_MODE"
    )
    """ECOSYSTEM_INTEGRATION_PLAN.md 定案 #14. "strict" (联调/CI) fails on any
    sub-system error per the 错误处理矩阵, never fabricating a result.
    "degraded" (demo, the default so MAEDA still runs standalone out of the
    box) falls back to built-in alternatives, but only for the matrix rows
    that say so — connection failures and unclassified internal errors, NOT
    param/contract/auth/data-input errors, which fail in both modes."""

    data_cleaner_quality_contract_version: str = Field(
        default="2", alias="DATA_CLEANER_QUALITY_CONTRACT_VERSION"
    )
    """ECOSYSTEM_INTEGRATION_PLAN.md 定案 #4b / 附录 B / 附录 U.0. Expected
    quality_contract_version on profile_dataset/validate_quality responses.
    A mismatch is a contract-version error — fails in both strict and
    degraded, never treated as a transient/fallback-able error. "2" is the
    TB3+TB4 dual-scope contract (附录 U.4); it is defined to be byte-for-byte
    equivalent to "1" whenever no intent is supplied (附录 U.4.2), so there
    is no dual-version transition period to support (附录 U.0)."""

    data_cleaner_planner_mode: Literal["rule", "llm"] = Field(
        default="llm", alias="DATA_CLEANER_PLANNER_MODE"
    )
    """ECOSYSTEM_INTEGRATION_PLAN.md 定案 #6 / M4 / 附录 R.4. Which planner
    clean_dataset asks the cleaner to use. Defaults to "llm" -- it's the
    hard dependency for the integration path (定案 #6), "rule" is only for
    smoke tests, CLI, and offline scenarios (附录 R.4). Requires the
    cleaner process to have a working LLM key configured
    (agentic-data-cleaner-v2 附录 D C5's `health` tool reports this via
    `planner_modes_available`) — the cleaner silently falls back to rule
    internally if it isn't, and reports that in
    execution_plan.planner_mode_used/planner_fallback_reason.
    In strict mode a requested "llm" that came back as "rule" is a hard
    failure (集成计划: "不得静默接受降级"); in degraded mode it's allowed,
    but must still be logged, not swallowed."""

    maeda_cleaner_mode: Literal["auto", "force_on", "force_off"] = Field(
        default="auto", alias="MAEDA_CLEANER_MODE"
    )
    """ECOSYSTEM_INTEGRATION_PLAN.md 定案 #15 / 阶段 3 收尾执行计划轮次 1 /
    附录 CB/CC. Whether the Data Cleaner subsystem's clean_dataset action is
    attempted at all, independent of quality-driven routing:
    "auto" (default) -- unchanged from pre-定案-#15 behavior:
        route_after_profiling/profile_and_clean gate clean_dataset purely
        on profile_dataset's `has_critical_issues` verdict (附录 B.1), same
        as before this setting existed.
    "force_on" -- 附录 CC.2: bypasses the has_critical_issues judgement
        ITSELF, not merely "don't let anything block the call". 附录 CC.2
        found (full `logs/runs.db` scan) that has_critical_issues has never
        once fired on real demo data (12/13 non-synthetic rows are "none");
        a force_on that only removed a gate nothing was ever tripping would
        be behaviorally identical to "auto", and Experiment 1's cleaner arm
        (阶段 4) would measure nothing. profile_dataset is still called
        unconditionally in every mode (it always was) -- only the decision
        to invoke clean_dataset is forced.
    "force_off" -- 附录 CC.3: clean_dataset is never invoked this run,
        REGARDLESS of has_critical_issues -- not "call it and discard the
        result" (that alternative was explicitly rejected: it would double
        real API call volume across a 4-arm x k-trial run, run straight
        into AP.4's rate-limit incident, and report a total_latency number
        no real deployment would ever have). profile_dataset still runs
        (diagnostic, not "cleaning"); route_after_profiling/profile_and_clean
        record an explicit `mode="skipped"` mcp_call_log entry and a
        `cleaning_stop_reason="force_off"` the moment this is decided, so
        "deliberately turned off" is distinguishable in the eval report
        from an ordinary auto-mode round where has_critical_issues just
        happened to be false (附录 CB.1.3 -- an empty mcp_call_log for this
        tool is otherwise ambiguous between the two).
    Experiment 1's cleaner arm (阶段 4) is therefore "force_on 强制清洗 vs
    force_off 不清洗" (附录 CC.2) -- whether has_critical_issues judges
    *correctly* is a separate question, answered by Experiment 2 (routing
    accuracy) using the "auto" mode instead."""

    maeda_rag_mode: Literal["auto", "force_on", "force_off"] = Field(
        default="auto", alias="MAEDA_RAG_MODE"
    )
    """ECOSYSTEM_INTEGRATION_PLAN.md 定案 #15 / 阶段 3 收尾执行计划轮次 1 /
    附录 CB.3.1/CC.7. Whether retrieve_knowledge_node calls the RAG server
    at all:
    "auto" -- 附录 CB.3.1/CC.7 裁定 4: STILL DEGRADES to "always retrieve",
        i.e. byte-identical to pre-定案-#15 behavior. 轮次 3 has since
        replaced `src/graph/builder.py`'s unconditional
        "generate_viz" -> "retrieve_domain_knowledge" `add_edge` with a
        real conditional route (route_after_viz, src/graph/router.py), so
        the *edge* now exists and both destinations are reachable and
        tested -- but the judgement that would make `auto` choose to skip
        (purely-computational query_types) is deliberately NOT
        implemented, per 裁定 4. Today only force_off takes the skip
        branch. This degeneration is adjudicated and temporary, not an
        oversight: do not read "auto exists" or "the conditional edge
        exists" as "auto's routing judgement exists" -- it doesn't yet.
        `rag_retrieval_decision`'s "auto" branch is the single place that
        changes when it does.
    "force_on" -- always retrieves; identical routing to "auto" today for
        the reason just given. Additionally the ONLY mode under which a
        retrieval that didn't run at settings.
        maeda_rag_expected_retrieval_mode (or degraded to fallback, or
        hard-failed) invalidates the trial for aggregation -- 附录 CK.3,
        implemented in src/graph/nodes.py::retrieve_knowledge_node and
        enforced in src/eval/trials.py::is_applicable.
    "force_off" -- retrieve_knowledge is never called; records an explicit
        `mode="skipped"` mcp_call_log entry (see maeda_cleaner_mode's
        docstring for why "skipped" must be distinct from an empty log)."""

    maeda_rag_expected_retrieval_mode: Literal["hybrid", "bm25_only"] = Field(
        default="hybrid", alias="MAEDA_RAG_EXPECTED_RETRIEVAL_MODE"
    )
    """ECOSYSTEM_INTEGRATION_PLAN.md 附录 CH.2/CI.3/**CK.3**. Which
    retrieval tier a `MAEDA_RAG_MODE=force_on` arm is asserting it ran at.

    rag-framework resolves its embedding/reranker providers from its OWN
    environment (rag/app/mcp_server/retrieval_config.py) — nothing MAEDA
    sends selects a tier — and it echoes what it actually used back on
    every retrieve response. 附录 AL.3 measured the gap that echo covers:
    dev Recall@5 **0.7083 BM25-only vs 0.9444 hybrid**, with 2 of 12
    queries returning nothing at all under BM25. So an on-arm that
    silently landed on BM25-only produces a perfectly plausible,
    merely-lower RAG number — indistinguishable from "RAG genuinely
    isn't worth it", which is precisely the one conclusion 阶段 4's TB6
    is pre-committed to accepting (附录 CI.2).

    "hybrid" (default) — the vector index must actually have been used
        (rag `retrieval_mode` ∈ {hybrid, hybrid_rerank}) and rag must not
        have set `degraded_reason`. Anything else invalidates the trial
        under force_on (src/graph/nodes.py::retrieve_knowledge_node),
        rather than scoring it as a normal on-arm run.
    "bm25_only" — a deliberately declared BM25-only deployment. Chosen so
        that a legitimate no-embedding-key environment can still run a
        coherent on-arm without every trial being thrown away; the guard
        it removes is only the *tier* check, `degraded_reason`, an
        unreachable server and a response with no echo fields at all
        still invalidate. The point of the guard was never "hybrid or
        bust" — it was that the tier must be **declared, not discovered
        after the fact**. Setting this to bm25_only is a pre-registration
        statement and must be reported alongside the results.

    Only consulted when maeda_rag_mode == "force_on"; "auto"/"force_off"
    runs record the tier in mcp_call_log but never invalidate on it,
    because they aren't making a claim about RAG's contribution."""

    intent_refine_trigger: Literal["always", "if_unresolved"] = Field(
        default="if_unresolved", alias="MAEDA_INTENT_REFINE_TRIGGER"
    )
    """ECOSYSTEM_INTEGRATION_PLAN.md E2 附录 BQ. When the schema-aware
    `refine_intent` node (src/graph/nodes.py) actually calls the LLM a
    second time, now that connect_schema has a real schema to inject:
    "always" (every run, whether or not the first parse's mentions already
    matched the schema) or "if_unresolved" (only when a deterministic
    pre-check — the same `_resolve_intent_columns` reconciliation
    `profile_and_clean` runs for real afterward — finds at least one
    mention that doesn't match any column). Defaults to "if_unresolved":
    refine only costs ~$0.0003/run (~1.4% of a full pipeline run), so this
    isn't about the dollar amount, it's about not spending an extra LLM
    round-trip's *latency* on the common case where the first parse
    already got every column name right. Eval runs should force "always"
    (set this env var, or monkeypatch settings.intent_refine_trigger)
    instead of relying on the default: an eval comparing on/off arms needs
    every case to go through the same code path, not have some cases
    silently skip refine because their first parse happened to already be
    schema-compatible while others didn't."""

    maeda_artifact_root: str = Field(
        default="./artifacts/cleaner_runs", alias="MAEDA_ARTIFACT_ROOT"
    )
    """M8 / 定案 #16. Root directory MAEDA asks the cleaner to place run_id
    subdirectories under. Code that sends this MUST resolve it to an
    absolute path first (`Path(settings.maeda_artifact_root).resolve()`) —
    sending the relative string as-is would have each process resolve it
    against its own CWD, which is exactly the F1 class of bug (cleaner
    returning a path only its own process could interpret) this whole
    run_id/artifact_root contract exists to prevent. 定案 #11's same-machine
    shared filesystem assumption is what makes a single resolved path valid
    for both sides."""

    # ── Data Sources ─────────────────────────────────────────────────────────
    default_data_dir: str = Field(default="./data/sample", alias="MAEDA_DEFAULT_DATA_DIR")

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: Literal["json", "pretty"] = Field(default="pretty", alias="LOG_FORMAT")
    log_file: Optional[str] = Field(default=None, alias="LOG_FILE")

    # ── Run persistence ──────────────────────────────────────────────────────
    # SQLite store for decision_trace/mcp_call_log (see src/persistence/run_store.py)
    # -- previously discarded when the process exited, leaving nothing
    # auditable after the fact.
    runs_db_path: str = Field(default="logs/runs.db", alias="MAEDA_RUNS_DB_PATH")

    # ── Business glossary (口径词表) ──────────────────────────────────────────
    glossary_path: str = Field(default="config/glossary.yaml", alias="MAEDA_GLOSSARY_PATH")
    """ECOSYSTEM_INTEGRATION_PLAN.md 阶段 3 轮次 4 (附录 CQ). Curated column
    definitions, filtered against the live schema by src/tools/glossary.py and
    injected at two points: the TB3 intent payload sent to the cleaner (附录
    U.2's `glossary`) and the plan_analysis prompt. A missing or unreadable
    file is NOT an error — it degrades to coverage "absent", which is stated
    explicitly at both injection points rather than silently omitted."""

    # ── Streamlit UI ─────────────────────────────────────────────────────────
    streamlit_port: int = Field(default=8501, alias="STREAMLIT_PORT")

    # ── Eval ─────────────────────────────────────────────────────────────────
    eval_enabled: bool = Field(default=True, alias="EVAL_ENABLED")
    eval_golden_suite: str = Field(
        default="./tests/eval/test_suite.json", alias="EVAL_GOLDEN_SUITE"
    )
    # Explicit overrides for the eval judge model. Left unset by default so
    # the judge can be resolved automatically (see resolved_eval_provider/
    # resolved_eval_model below) — a judge sharing weights/training with the
    # agent it's scoring is a self-preference risk, and DEV_SPEC originally
    # called for a distinct EVAL_MODEL that was never actually wired up.
    eval_llm_provider: Optional[Literal["openai", "anthropic"]] = Field(
        default=None, alias="EVAL_LLM_PROVIDER"
    )
    eval_llm_model: Optional[str] = Field(default=None, alias="EVAL_MODEL")
    # How many independent judge calls to make per relevance/groundedness
    # check, aggregated by median. Mitigates single-sample judge variance
    # (the same case scoring 0.5 one run and 1.0 the next).
    eval_judge_samples: int = Field(default=3, alias="EVAL_JUDGE_SAMPLES")

    # ── Per-agent model tiering ──────────────────────────────────────────────
    # Everything defaulted to llm_model (cost-optimized, gpt-4o-mini) until
    # this. The Planner (structured multi-step reasoning, including the
    # roadmap #11 step-repair path) and the Guardrail's live pass/fail judge
    # (catching hallucination/fabrication in a report a sibling model wrote)
    # are the two spots where the eval harness's own judge-tiering precedent
    # (resolved_eval_model above) already showed a stronger model earns its
    # keep. Left unset by default so both resolve automatically; explicit
    # overrides only needed to pin a specific model.
    planner_llm_model: Optional[str] = Field(default=None, alias="MAEDA_PLANNER_MODEL")
    guardrail_llm_model: Optional[str] = Field(default=None, alias="MAEDA_GUARDRAIL_MODEL")

    @field_validator("llm_temperature")
    @classmethod
    def clamp_temperature(cls, v: float) -> float:
        return max(0.0, min(2.0, v))

    @property
    def has_llm_key(self) -> bool:
        if self.llm_provider == "openai":
            return bool(self.openai_api_key)
        return bool(self.anthropic_api_key)

    @property
    def resolved_eval_provider(self) -> str:
        """
        Prefer a provider *different* from the agent's own (llm_provider) so
        the judge isn't scoring output from a sibling of itself — falls back
        to the agent's provider if no usable key exists for the other one.
        """
        if self.eval_llm_provider:
            return self.eval_llm_provider
        other = "anthropic" if self.llm_provider == "openai" else "openai"
        other_key = self.anthropic_api_key if other == "anthropic" else self.openai_api_key
        if _looks_like_real_key(other_key):
            return other
        return self.llm_provider

    @property
    def resolved_eval_model(self) -> str:
        """Default to a stronger model than the (typically cost-optimized) agent model."""
        if self.eval_llm_model:
            return self.eval_llm_model
        return "claude-sonnet-5" if self.resolved_eval_provider == "anthropic" else "gpt-4o"

    @property
    def resolved_planner_model(self) -> str:
        """
        Model for AnalysisAgent (plan() and the roadmap #11 step-repair path).
        Same provider as llm_provider (unlike the eval judge, there's no
        self-preference concern here — this is generating a plan, not
        scoring one), just a stronger model in the same family.
        """
        if self.planner_llm_model:
            return self.planner_llm_model
        return _stronger_model_for(self.llm_provider)

    @property
    def resolved_guardrail_model(self) -> str:
        """Model for the Guardrail Agent's live pass/fail judge (hallucination/
        fabrication detection). Same tiering rationale as resolved_planner_model."""
        if self.guardrail_llm_model:
            return self.guardrail_llm_model
        return _stronger_model_for(self.llm_provider)


def _looks_like_real_key(key: Optional[str]) -> bool:
    """
    Rejects unset keys and this project's own .env.example placeholder
    convention (e.g. "sk-ant-...") — good enough to avoid silently trying to
    call a provider with a key that was never actually filled in.
    """
    return bool(key) and not key.endswith("...")


def _stronger_model_for(provider: str) -> str:
    """The step-up model within a provider's own family — same tier used by
    resolved_eval_model when it stays on the agent's own provider."""
    return "claude-sonnet-5" if provider == "anthropic" else "gpt-4o"


# Singleton — import this everywhere
settings = MAEDASettings()
