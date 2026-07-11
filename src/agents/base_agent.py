"""
Abstract base class for all MAEDA agents.
All agents inherit from BaseAgent and get: decision tracing, cost tracking,
and a standard process() interface.
"""
from __future__ import annotations

import abc
from typing import Any, Optional

from src.state.graph_state import MAEDAState
from src.utils.cost_tracker import CostTracker
from src.utils.logger import DecisionTracer, get_logger


class BaseAgent(abc.ABC):
    """
    Every MAEDA agent extends this class.

    Subclasses must implement `process(state) → state`.

    Provided helpers:
      - self.log_decision(...)  → appends to state["decision_trace"]
      - self.track_cost(...)    → records token usage to state["token_usage"]
      - self.logger             → pre-configured logger
    """

    def __init__(self, name: str, cost_tracker: Optional[CostTracker] = None):
        self.name = name
        self._tracer = DecisionTracer(name)
        self._cost_tracker = cost_tracker or CostTracker()
        self.logger = get_logger(f"maeda.agent.{name}")

    # ── Core interface ──────────────────────────────────────────────────────

    @abc.abstractmethod
    async def process(self, state: MAEDAState) -> MAEDAState:
        """
        Execute the agent's logic against the current state.
        Returns the updated state. Must not raise — catch and set state["error"].
        """
        ...

    # ── Decision Tracing ────────────────────────────────────────────────────

    def log_decision(
        self,
        state: MAEDAState,
        action: str,
        reasoning: str,
        inputs: Any = None,
        outputs: Any = None,
        confidence: float = 1.0,
        extra: Optional[dict] = None,
    ) -> MAEDAState:
        """Append a decision record to state["decision_trace"] and return state."""
        record = self._tracer.log(
            action=action,
            reasoning=reasoning,
            inputs=inputs,
            outputs=outputs,
            confidence=confidence,
            extra=extra,
        )
        state["decision_trace"] = [*state.get("decision_trace", []), record]
        return state

    # ── Cost Tracking ───────────────────────────────────────────────────────

    def track_cost(
        self,
        state: MAEDAState,
        model: str,
        input_tokens: int,
        output_tokens: int,
        call_label: str = "",
    ) -> MAEDAState:
        """Record one LLM call's token usage and propagate to state."""
        self._cost_tracker.record(
            agent_name=self.name,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            call_label=call_label,
        )
        state["token_usage"] = self._cost_tracker.to_state_dict()
        return state

    # ── Error Handling Helper ───────────────────────────────────────────────

    def set_error(self, state: MAEDAState, message: str) -> MAEDAState:
        """Set error state and update phase."""
        self.logger.error("Agent error: %s", message)
        state["error"] = message
        state["current_phase"] = "error"
        return state
