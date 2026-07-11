"""
Data Cleaner MCP integration.

Wraps MCPClient to provide typed, high-level calls to the Agentic Data Cleaner
sub-system. All raw dict responses are parsed into typed dataclasses.

Tools exposed by the Data Cleaner MCP server:
  profile_dataset   {path} → DataQualityReport
  get_cleaning_plan {path} → CleaningPlan
  clean_dataset     {path, plan?} → CleaningResult
  validate_quality  {path} → QualityValidation
"""
from __future__ import annotations

from src.mcp_client.client import MCPClient
from src.mcp_client.models import (
    CleaningPlan,
    CleaningResult,
    DataQualityReport,
    QualityValidation,
)
from src.utils.logger import get_logger

logger = get_logger("maeda.mcp.data_cleaner")


class DataCleanerClient:
    """High-level client for the Agentic Data Cleaner MCP server."""

    def __init__(self, transport: MCPClient):
        self._transport = transport

    async def profile_dataset(self, path: str) -> DataQualityReport:
        """Profile a dataset and return a DataQualityReport."""
        logger.debug("profile_dataset | path=%s", path)
        raw = await self._transport.call_tool("profile_dataset", {"path": path})
        return DataQualityReport.from_mcp_response(raw)

    async def get_cleaning_plan(self, path: str) -> CleaningPlan:
        """Get a recommended cleaning plan for a dataset."""
        logger.debug("get_cleaning_plan | path=%s", path)
        raw = await self._transport.call_tool("get_cleaning_plan", {"path": path})
        return CleaningPlan.from_mcp_response(raw)

    async def clean_dataset(
        self, path: str, plan: CleaningPlan | None = None
    ) -> CleaningResult:
        """Execute cleaning (optionally with a pre-built plan) and return results."""
        logger.debug("clean_dataset | path=%s", path)
        args: dict = {"path": path}
        if plan is not None:
            args["plan"] = plan.to_dict()
        raw = await self._transport.call_tool("clean_dataset", args)
        return CleaningResult.from_mcp_response(raw)

    async def validate_quality(self, path: str) -> QualityValidation:
        """Validate final data quality after cleaning."""
        logger.debug("validate_quality | path=%s", path)
        raw = await self._transport.call_tool("validate_quality", {"path": path})
        return QualityValidation.from_mcp_response(raw)
