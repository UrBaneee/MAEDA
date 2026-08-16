"""
Data Cleaner MCP integration.

Wraps MCPClient to provide typed, high-level calls to the Agentic Data Cleaner
sub-system. All raw dict responses are parsed into typed dataclasses.

Tools exposed by the Data Cleaner MCP server:
  profile_dataset   {dataset_path} → DataQualityReport
  get_cleaning_plan {dataset_path} → CleaningPlan
  clean_dataset     {dataset_path, plan?} → CleaningResult
  validate_quality  {dataset_path} → QualityValidation

Argument key is `dataset_path`, not `path` — this is the cleaner's real
server-side parameter name (ECOSYSTEM_INTEGRATION_PLAN.md 定案 #2: MAEDA
conforms to the cleaner's actual schema rather than the other way around).
"""
from __future__ import annotations

import json
from typing import Optional

from src.config.settings import settings
from src.mcp_client.client import MCPClient, MCPContractError, MCPToolError
from src.mcp_client.models import (
    CleaningPlan,
    CleaningResult,
    DataQualityReport,
    QualityValidation,
)
from src.utils.logger import get_logger

logger = get_logger("maeda.mcp.data_cleaner")


def _raise_if_error_envelope(raw: dict, tool: str) -> None:
    """
    cleaner wraps internal failures as `{"error": true, "error_type": ...,
    "message": ...}` inside an otherwise protocol-successful MCP response
    (定案 #13 migration-era envelope; mcp_app.py::_error_response) rather
    than a protocol-level tool error. Before this check existed, every one
    of this client's methods silently parsed that envelope as if it were a
    real DataQualityReport/CleaningResult/QualityValidation (row_count=0,
    columns=[], etc.) instead of surfacing the failure — confirmed live via
    scripts/check_ecosystem.py before this fix landed. Raising here is what
    lets fallback.py's error-matrix classification (M1) see it at all.
    """
    if isinstance(raw, dict) and raw.get("error") is True:
        raise MCPToolError(
            f"{tool} reported an internal error: {raw.get('message', '(no message)')}",
            error_type=raw.get("error_type"),
            raw=raw,
        )


def _check_field_present(raw: dict, tool: str, field_name: str) -> None:
    """
    附录 B.3: a response missing a field this contract guarantees is a
    contract violation, not license to assume the safest-looking default.
    Before this check, `DataQualityReport.from_mcp_response`'s
    `data.get("has_critical_issues", False)` would silently treat a
    genuinely dirty dataset as clean if the server ever failed to send the
    field — a false "nothing to clean" is worse than a loud failure. strict
    mode fails outright; degraded mode is allowed to proceed with the
    caller's default, but must not do so *silently* — the warning here is
    the difference (附录 B.3: "不得静默当 false").
    """
    if field_name in raw:
        return
    message = f"{tool}: response is missing required field {field_name!r}"
    if settings.mcp_strict_mode == "strict":
        raise MCPContractError(message, error_type="ContractViolation", raw=raw)
    logger.warning("%s (degraded mode: proceeding with the field's default value)", message)


def _check_contract_version(raw: dict, tool: str) -> None:
    """
    定案 #4b: quality_contract_version must match what Settings expects
    before the response is trusted. A mismatch is a contract error — fails
    in both strict and degraded per the 错误处理矩阵, never a fallback
    candidate. Only profile_dataset and validate_quality carry this field
    (附录 B.6); clean_dataset's response doesn't, so callers must not call
    this for clean_dataset.
    """
    version = raw.get("quality_contract_version")
    expected = settings.data_cleaner_quality_contract_version
    if version != expected:
        raise MCPContractError(
            f"{tool}: quality_contract_version mismatch "
            f"(expected {expected!r}, got {version!r})",
            error_type="ContractVersionMismatch",
            raw=raw,
        )


def _check_planner_mode_not_silently_degraded(result: CleaningResult, requested: str) -> None:
    """
    定案 #6 / M4: the cleaner's own LLMPlanner already degrades gracefully
    from "llm" to "rule" internally (agentic-data-cleaner-v2 附录 D F2 fix)
    and reports what really happened via execution_plan.planner_mode_used/
    planner_fallback_reason (live since cleaner's C3). strict mode must not
    silently accept that degradation — that is exactly the kind of "ran,
    but not the way you asked" failure strict mode exists to surface.
    degraded mode is allowed to accept it (the result is still usable data),
    but must not do so silently — hence the warning either way.
    """
    if requested != "llm":
        return  # "rule" has no external dependency; nothing can degrade
    used = result.execution_plan.get("planner_mode_used")
    if used == "llm":
        return
    reason = result.execution_plan.get("planner_fallback_reason")
    message = (
        f"clean_dataset: requested planner_mode='llm' but the server used "
        f"{used!r} instead ({reason or 'no reason reported'})"
    )
    if settings.mcp_strict_mode == "strict":
        raise MCPContractError(message, error_type="PlannerModeDegraded")
    logger.warning("%s (degraded mode: proceeding with the actual result)", message)


class DataCleanerClient:
    """High-level client for the Agentic Data Cleaner MCP server."""

    def __init__(self, transport: MCPClient):
        self._transport = transport

    async def profile_dataset(
        self, path: str, run_id: str = "", artifact_root: str = "", intent: Optional[dict] = None,
    ) -> DataQualityReport:
        """Profile a dataset and return a DataQualityReport."""
        logger.debug("profile_dataset | path=%s run_id=%s", path, run_id)
        args: dict = {"dataset_path": path}
        if run_id:
            args["run_id"] = run_id
        if artifact_root:
            args["artifact_root"] = artifact_root
        if intent is not None:
            args["intent"] = intent
        raw = await self._transport.call_tool(
            "profile_dataset", args, timeout=15.0, deadline=30.0,
        )
        _raise_if_error_envelope(raw, "profile_dataset")
        _check_field_present(raw, "profile_dataset", "has_critical_issues")
        _check_contract_version(raw, "profile_dataset")
        return DataQualityReport.from_mcp_response(raw)

    async def get_cleaning_plan(self, path: str) -> CleaningPlan:
        """Get a recommended cleaning plan for a dataset."""
        logger.debug("get_cleaning_plan | path=%s", path)
        raw = await self._transport.call_tool(
            "get_cleaning_plan", {"dataset_path": path}, timeout=30.0, deadline=60.0,
        )
        _raise_if_error_envelope(raw, "get_cleaning_plan")
        return CleaningPlan.from_mcp_response(raw)

    async def clean_dataset(
        self,
        path: str,
        plan: CleaningPlan | None = None,
        planner_mode: str = "rule",
        max_rounds: int = 1,
        run_id: str = "",
        artifact_root: str = "",
        intent: Optional[dict] = None,
    ) -> CleaningResult:
        """Execute cleaning (optionally with a pre-built plan) and return results."""
        logger.debug(
            "clean_dataset | path=%s planner_mode=%s max_rounds=%s run_id=%s",
            path, planner_mode, max_rounds, run_id,
        )
        args: dict = {"dataset_path": path, "planner_mode": planner_mode, "max_rounds": max_rounds}
        if run_id:
            args["run_id"] = run_id
        if artifact_root:
            args["artifact_root"] = artifact_root
        if intent is not None:
            args["intent"] = intent
        if plan is not None:
            # cleaner's real signature is `plan: str = ""` — a JSON string
            # it json.loads()s itself (mcp_app.py::clean_dataset), not a
            # structured object. Sending the dict directly fails the
            # server's own pydantic validation (`Input should be a valid
            # string [type=string_type]`) -- confirmed live against a real
            # cleaner process, where it silently degraded to the "Data
            # Cleaner unavailable" fallback (internal_unknown, degraded
            # mode) instead of actually cleaning anything.
            args["plan"] = json.dumps(plan.to_dict())
        raw = await self._transport.call_tool(
            "clean_dataset", args, timeout=90.0, deadline=180.0,
        )
        _raise_if_error_envelope(raw, "clean_dataset")
        result = CleaningResult.from_mcp_response(raw)
        _check_planner_mode_not_silently_degraded(result, planner_mode)
        return result

    async def validate_quality(
        self, path: str, run_id: str = "", artifact_root: str = "", intent: Optional[dict] = None,
    ) -> QualityValidation:
        """Validate final data quality after cleaning."""
        logger.debug("validate_quality | path=%s run_id=%s", path, run_id)
        args: dict = {"dataset_path": path}
        if run_id:
            args["run_id"] = run_id
        if artifact_root:
            args["artifact_root"] = artifact_root
        if intent is not None:
            args["intent"] = intent
        raw = await self._transport.call_tool(
            "validate_quality", args, timeout=15.0, deadline=30.0,
        )
        _raise_if_error_envelope(raw, "validate_quality")
        _check_field_present(raw, "validate_quality", "passed")
        _check_contract_version(raw, "validate_quality")
        return QualityValidation.from_mcp_response(raw)
