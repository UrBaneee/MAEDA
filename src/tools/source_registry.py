"""
Source Registry — manages registered data sources for a MAEDA session.

Allows multi-source queries by keeping a named registry of source descriptors.
The active source is tracked separately and used by the graph nodes.
"""
from __future__ import annotations

from typing import Optional

from src.utils.logger import get_logger

logger = get_logger("maeda.tools.source_registry")


class SourceRegistry:
    """
    Maintains a named registry of data source descriptors.
    Thread-safe for single-session use (no locking needed for sync access).

    A source descriptor:
        {
          "name": str,          # unique identifier
          "type": str,          # csv | sql | json | excel
          "path": str,          # file path or connection string
          "table_name": str,    # optional
          "query": str,         # optional SQL
          "schema": dict,       # populated after connect()
          "preview": list,      # populated after connect()
        }
    """

    def __init__(self):
        self._sources: dict[str, dict] = {}
        self._active_name: Optional[str] = None

    def register(self, source: dict) -> str:
        """
        Register a data source descriptor. Returns the source name.
        If no "name" key, auto-generates one from the path.
        """
        name = source.get("name") or _infer_name(source.get("path", ""))
        source = {**source, "name": name}
        self._sources[name] = source
        if self._active_name is None:
            self._active_name = name
        logger.debug("Registered source: %s (%s)", name, source.get("type", "?"))
        return name

    def register_many(self, sources: list[dict]) -> list[str]:
        return [self.register(s) for s in sources]

    def get(self, name: str) -> Optional[dict]:
        return self._sources.get(name)

    def set_active(self, name: str) -> None:
        if name not in self._sources:
            raise KeyError(f"Source not registered: {name!r}")
        self._active_name = name

    @property
    def active(self) -> Optional[dict]:
        if self._active_name:
            return self._sources.get(self._active_name)
        return None

    def update(self, name: str, updates: dict) -> None:
        """Merge updates into an existing source descriptor (e.g., after profiling)."""
        if name not in self._sources:
            raise KeyError(f"Source not registered: {name!r}")
        self._sources[name] = {**self._sources[name], **updates}

    def all_sources(self) -> list[dict]:
        return list(self._sources.values())

    def to_state_list(self) -> list[dict]:
        """Return all sources in the format expected by state["data_sources"]."""
        return self.all_sources()

    def __len__(self) -> int:
        return len(self._sources)

    def __contains__(self, name: str) -> bool:
        return name in self._sources


def _infer_name(path: str) -> str:
    """Derive a short readable name from a file path or connection string."""
    if not path:
        return "source"
    # For SQL connection strings, use the DB name portion
    if "://" in path:
        return path.split("/")[-1].split("?")[0] or "db"
    import os
    return os.path.splitext(os.path.basename(path))[0] or "source"
