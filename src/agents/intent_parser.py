"""
Intent Parser Agent — Phase 2.

Converts a natural language query into a structured ParsedIntent using an LLM.
Handles ambiguity detection, confidence scoring, and multi-turn clarification.
Schema injection (task 2.5) enriches parsing when a data source is already loaded.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base_agent import BaseAgent
from src.config.agent_prompts import INTENT_CLARIFICATION_SYSTEM, INTENT_PARSER_SYSTEM
from src.config.settings import settings
from src.state.graph_state import MAEDAState
from src.utils.logger import get_logger

logger = get_logger("maeda.agent.intent_parser")

# Confidence threshold below which clarification is requested
_CLARIFICATION_THRESHOLD = 0.7


# ─── ParsedIntent dataclass ───────────────────────────────────────────────────

@dataclass
class ParsedIntent:
    query_type: Literal["descriptive", "diagnostic", "predictive", "comparative", "exploratory"]
    target_metrics: list[str]
    dimensions: list[str]
    filters: list[dict]
    time_range: Optional[dict]
    aggregation: Optional[str]
    sort_by: Optional[str]
    limit: Optional[int]
    confidence: float          # 0.0–1.0; < 0.7 triggers clarification
    ambiguities: list[str]
    raw_query: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_llm_response(cls, data: dict, raw_query: str) -> "ParsedIntent":
        return cls(
            query_type=data.get("query_type", "exploratory"),
            target_metrics=data.get("target_metrics") or [],
            dimensions=data.get("dimensions") or [],
            filters=data.get("filters") or [],
            time_range=data.get("time_range"),
            aggregation=data.get("aggregation"),
            sort_by=data.get("sort_by"),
            limit=data.get("limit"),
            confidence=float(data.get("confidence", 0.5)),
            ambiguities=data.get("ambiguities") or [],
            raw_query=raw_query,
        )


# ─── LLM factory ─────────────────────────────────────────────────────────────

def _build_llm():
    """Instantiate the configured LLM (OpenAI or Anthropic)."""
    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.max_tokens_per_call,
            api_key=settings.anthropic_api_key or "sk-no-key",
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.max_tokens_per_call,
        api_key=settings.openai_api_key or "sk-no-key",
    )


# ─── IntentParserAgent ────────────────────────────────────────────────────────

class IntentParserAgent(BaseAgent):
    """
    Parses a natural language query into a structured ParsedIntent.

    State fields read:   user_query, conversation_history, schema_summary
    State fields written: parsed_intent, clarification_needed, clarification_question,
                          decision_trace, token_usage
    """

    def __init__(self, llm=None):
        super().__init__("intent_parser")
        self._llm = llm or _build_llm()

    # ── Public entry point ────────────────────────────────────────────────────

    async def process(self, state: MAEDAState) -> MAEDAState:
        query = state.get("user_query", "").strip()
        if not query:
            return self.set_error(state, "user_query is empty")

        try:
            intent = await self._parse(query, state.get("schema_summary", ""))
            # Sync token usage accumulated inside _parse back to state
            state["token_usage"] = self._cost_tracker.to_state_dict()
        except Exception as exc:
            logger.exception("Intent parsing failed")
            return self.set_error(state, f"Intent parsing failed: {exc}")

        state["parsed_intent"] = intent.to_dict()

        # Ambiguity detection — triggers clarification flow
        needs_clarification = (
            intent.confidence < _CLARIFICATION_THRESHOLD
            or bool(intent.ambiguities)
        )
        state["clarification_needed"] = needs_clarification

        if needs_clarification:
            try:
                question = await self._generate_clarification(intent)
                state["clarification_question"] = question
            except Exception as exc:
                logger.warning("Could not generate clarification: %s", exc)
                state["clarification_question"] = (
                    "Could you please clarify your question with more detail?"
                )

        state = self.log_decision(
            state,
            action="parse_intent",
            reasoning=(
                f"Query type: {intent.query_type}, "
                f"confidence: {intent.confidence:.2f}, "
                f"clarification_needed: {needs_clarification}"
            ),
            inputs={"query": query, "has_schema": bool(state.get("schema_summary"))},
            outputs=intent.to_dict(),
            confidence=intent.confidence,
        )

        # Append the user turn to conversation history
        state["conversation_history"] = [
            *state.get("conversation_history", []),
            {"role": "user", "content": query},
        ]
        return state

    async def generate_clarification_question(self, state: MAEDAState) -> MAEDAState:
        """
        Node handler for ask_clarification.
        Surfaces the stored question and appends it to conversation history.
        """
        question = state.get("clarification_question") or (
            "Could you provide more details about what you're looking for?"
        )
        state["conversation_history"] = [
            *state.get("conversation_history", []),
            {"role": "assistant", "content": question},
        ]
        # Clear the flag — will be re-evaluated after user answers
        state["clarification_needed"] = False
        state["clarification_question"] = None
        state = self.log_decision(
            state,
            action="ask_clarification",
            reasoning="Surfacing clarification question to user",
            outputs={"question": question},
            confidence=1.0,
        )
        return state

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _parse(self, query: str, schema_summary: str) -> ParsedIntent:
        """Call the LLM to produce a ParsedIntent."""
        system = INTENT_PARSER_SYSTEM
        if schema_summary:
            system = (
                system
                + f"\n\n### Available Data Schema\n{schema_summary}\n"
                + "Use the schema to identify valid column names for dimensions, "
                + "metrics, and filters."
            )

        messages = [
            SystemMessage(content=system),
            HumanMessage(content=query),
        ]

        response = await self._llm.ainvoke(messages)
        raw_text = response.content.strip()

        # Track token usage
        usage = getattr(response, "usage_metadata", None) or {}
        self._cost_tracker.record(
            agent_name=self.name,
            model=settings.llm_model,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            call_label="parse_intent",
        )

        data = _parse_json(raw_text)
        return ParsedIntent.from_llm_response(data, raw_query=query)

    async def _generate_clarification(self, intent: ParsedIntent) -> str:
        """Ask the LLM to formulate a clarification question."""
        context = (
            f"Original query: {intent.raw_query}\n"
            f"Ambiguities: {', '.join(intent.ambiguities)}"
        )
        messages = [
            SystemMessage(content=INTENT_CLARIFICATION_SYSTEM),
            HumanMessage(content=context),
        ]
        response = await self._llm.ainvoke(messages)

        usage = getattr(response, "usage_metadata", None) or {}
        self._cost_tracker.record(
            agent_name=self.name,
            model=settings.llm_model,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            call_label="generate_clarification",
        )
        return response.content.strip()


# ─── JSON extraction helper ───────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    """
    Extract the first JSON object from an LLM response.
    Handles triple-backtick fences gracefully.
    """
    # Strip markdown fences if present
    if "```" in text:
        lines = text.split("\n")
        inner = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(inner)

    # Find the outermost { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in LLM response: {text[:200]!r}")
    return json.loads(text[start : end + 1])
