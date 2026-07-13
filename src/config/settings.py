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
        return "claude-3-5-sonnet-20241022" if self.resolved_eval_provider == "anthropic" else "gpt-4o"

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
    return "claude-3-5-sonnet-20241022" if provider == "anthropic" else "gpt-4o"


# Singleton — import this everywhere
settings = MAEDASettings()
