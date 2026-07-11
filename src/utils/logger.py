"""
Structured JSON logger with decision trace support.
Every agent decision is recorded with: agent_name, action, reasoning,
inputs, outputs, confidence, and timestamp.
"""
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Optional


class _JsonFormatter(logging.Formatter):
    """Format log records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Attach any extra fields set on the record
        for key, value in record.__dict__.items():
            if key not in {
                "args", "asctime", "created", "exc_info", "exc_text",
                "filename", "funcName", "id", "levelname", "levelno",
                "lineno", "message", "module", "msecs", "msg", "name",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "thread", "threadName",
            }:
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _PrettyFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        base = f"{color}[{ts}] {record.levelname:<8}{self.RESET} {record.name} — {record.getMessage()}"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def get_logger(
    name: str,
    level: str = "INFO",
    fmt: str = "pretty",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Return a configured logger. Call once per module."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter: logging.Formatter = (
        _JsonFormatter() if fmt.lower() == "json" else _PrettyFormatter()
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(_JsonFormatter())  # Always JSON in files
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


# ─── Decision Trace ───────────────────────────────────────────────────────────

class DecisionTracer:
    """
    Appends decision records to MAEDAState["decision_trace"].
    Each record captures what an agent decided and why.
    """

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self._logger = get_logger(f"maeda.trace.{agent_name}")

    def log(
        self,
        action: str,
        reasoning: str,
        inputs: Any = None,
        outputs: Any = None,
        confidence: float = 1.0,
        extra: Optional[dict] = None,
    ) -> dict:
        """Build a trace record and return it (caller appends to state)."""
        record: dict[str, Any] = {
            "trace_id": str(uuid.uuid4()),
            "agent_name": self.agent_name,
            "action": action,
            "reasoning": reasoning,
            "inputs": inputs,
            "outputs": outputs,
            "confidence": confidence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            record.update(extra)
        self._logger.debug(
            "decision",
            extra={
                "agent": self.agent_name,
                "action": action,
                "confidence": confidence,
            },
        )
        return record
