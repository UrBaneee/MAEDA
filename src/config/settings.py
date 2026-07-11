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

    # ── Data Sources ─────────────────────────────────────────────────────────
    default_data_dir: str = Field(default="./data/sample", alias="MAEDA_DEFAULT_DATA_DIR")

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: Literal["json", "pretty"] = Field(default="pretty", alias="LOG_FORMAT")
    log_file: Optional[str] = Field(default=None, alias="LOG_FILE")

    # ── Streamlit UI ─────────────────────────────────────────────────────────
    streamlit_port: int = Field(default=8501, alias="STREAMLIT_PORT")

    # ── Eval ─────────────────────────────────────────────────────────────────
    eval_enabled: bool = Field(default=True, alias="EVAL_ENABLED")
    eval_golden_suite: str = Field(
        default="./tests/eval/test_suite.json", alias="EVAL_GOLDEN_SUITE"
    )

    @field_validator("llm_temperature")
    @classmethod
    def clamp_temperature(cls, v: float) -> float:
        return max(0.0, min(2.0, v))

    @property
    def has_llm_key(self) -> bool:
        if self.llm_provider == "openai":
            return bool(self.openai_api_key)
        return bool(self.anthropic_api_key)


# Singleton — import this everywhere
settings = MAEDASettings()
