"""
Token-usage and cost tracking per agent per call.
Supports OpenAI and Anthropic pricing models.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ─── Pricing tables (USD per 1 000 tokens) ────────────────────────────────────

_PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o":                  {"input": 0.005,   "output": 0.015},
    "gpt-4o-mini":             {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo":             {"input": 0.01,    "output": 0.03},
    # Anthropic
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-3-haiku-20240307":    {"input": 0.00025, "output": 0.00125},
    "claude-opus-4-6":            {"input": 0.015,   "output": 0.075},
    "claude-sonnet-4-6":          {"input": 0.003,   "output": 0.015},
    "claude-haiku-4-5-20251001":  {"input": 0.0008,  "output": 0.004},
}
_DEFAULT_PRICING = {"input": 0.005, "output": 0.015}


def _price_for(model: str) -> dict[str, float]:
    for key, pricing in _PRICING.items():
        if key in model.lower():
            return pricing
    return _DEFAULT_PRICING


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class CallRecord:
    agent_name: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    call_label: str = ""    # e.g., "parse_intent", "plan_analysis"

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class AgentUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0

    def add(self, record: CallRecord) -> None:
        self.input_tokens += record.input_tokens
        self.output_tokens += record.output_tokens
        self.cost_usd += record.cost_usd
        self.calls += 1

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "calls": self.calls,
        }


@dataclass
class CostTracker:
    """
    Tracks token usage and cost across all agents in a session.

    Usage (in an agent node):
        tracker = CostTracker()
        record = tracker.record(
            agent_name="intent_parser",
            model="gpt-4o-mini",
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )
        state["token_usage"] = tracker.to_state_dict()
    """
    max_cost_usd: float = 1.0
    _per_agent: dict[str, AgentUsage] = field(default_factory=dict, repr=False)
    _call_log: list[CallRecord] = field(default_factory=list, repr=False)

    def record(
        self,
        agent_name: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        call_label: str = "",
    ) -> CallRecord:
        """Record one LLM call. Raises BudgetExceededError if over limit."""
        pricing = _price_for(model)
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1000

        rec = CallRecord(
            agent_name=agent_name,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            call_label=call_label,
        )
        self._call_log.append(rec)
        if agent_name not in self._per_agent:
            self._per_agent[agent_name] = AgentUsage()
        self._per_agent[agent_name].add(rec)

        if self.total_cost > self.max_cost_usd:
            raise BudgetExceededError(
                f"Session cost ${self.total_cost:.4f} exceeds limit ${self.max_cost_usd:.4f}"
            )
        return rec

    @property
    def total_cost(self) -> float:
        return sum(a.cost_usd for a in self._per_agent.values())

    @property
    def total_tokens(self) -> int:
        return sum(a.input_tokens + a.output_tokens for a in self._per_agent.values())

    def to_state_dict(self) -> dict:
        """Return a dict suitable for MAEDAState["token_usage"]."""
        return {
            agent: usage.to_dict()
            for agent, usage in self._per_agent.items()
        }

    def summary(self) -> dict:
        return {
            "total_cost_usd": round(self.total_cost, 6),
            "total_tokens": self.total_tokens,
            "per_agent": self.to_state_dict(),
        }


class BudgetExceededError(RuntimeError):
    """Raised when session cost exceeds the configured maximum."""
