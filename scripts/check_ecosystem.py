"""
TB0 smoke test: transport + parameter-contract check against the real
Data Cleaner and RAG-MCP-Server sub-systems.

See ECOSYSTEM_INTEGRATION_PLAN.md, 阶段 0 / TB0 (v4收紧) for the full spec
this implements. Summary of what "pass" means here, since TB0 was
explicitly tightened to stop treating parameter-contract errors as
acceptable "business layer" noise (that gap is exactly how the ecosystem
integration sat at 0% exercised for as long as it did):

  1. initialize + list_tools succeed against both servers.
  2. Every required tool is present:
       cleaner: profile_dataset, clean_dataset, validate_quality
       rag:     retrieve, retrieve_with_metadata, list_collections
     (get_cleaning_plan is intentionally excluded — 定案 #3: MAEDA never
     calls it.)
  3. For each required tool, the exact argument shape MAEDA's own client
     code sends (src/mcp_client/data_cleaner.py: flat {"dataset_path":
     ...}; src/mcp_client/rag_server.py: wrapped {"input": {...}}) is
     validated *locally* against the tool's declared inputSchema before
     any network call. This is the check that would have caught
     {"path": ...} vs {"dataset_path": ...} before it ever shipped.
  4. Each required tool is then actually invoked once with minimal legal
     args, live.
  5. Response checking follows the TB0 three-layer split — do not mix them:
       - inputSchema validates the request only, never the response
       - outputSchema validates the response only if the tool declares one
         (MCP does not require it; neither sub-system's tools do today) —
         absent that, we only check protocol status (isError) and the
         migration-era error envelope (定案 #13)
       - full response-parsing compatibility against MAEDA's own boundary
         parsers (src/mcp_client/models.py) is explicitly out of scope —
         that's TB1a's job, because those parsers have a known bug
         (M2: `.get("severity")` on what may be a string list) and using
         them as today's baseline would be circular (see 附录 A.4)
  6. Feeding cleaner a nonexistent file path is expected to come back as a
     *stable-shaped* error, not a crash — today that shape is
     `{"error": true, "error_type": "FileNotFoundError"}` with no
     `error_code` yet (C4 not landed, 附录 D.3). That is recorded as a
     pass, not a violation.
  7. If rag's corpus has no usable collection yet, the "no index" case is
     recorded as an informational item rather than a hard failure —
     "无索引错误契约检查" and "成功检索检查" are two separate things per TB0.

Exit code is non-zero iff any required tool is missing, or any local
inputSchema validation fails, or any minimal-legal-call comes back
error-shaped. Everything else various sub-checks record as pass/info and
do not by themselves fail TB0.

Usage:
    poetry run python scripts/check_ecosystem.py
    poetry run python scripts/check_ecosystem.py --rag-collection wake_apparel --top-k 3
    poetry run python scripts/check_ecosystem.py --skip-rag   # cleaner only
    poetry run python scripts/check_ecosystem.py --skip-cleaner

Expects both sub-system servers already running (阶段0 "启动命令"):
    cd ~/rag-framework && MCP_TRANSPORT=streamable-http MCP_HOST=127.0.0.1 MCP_PORT=8002 \\
        python -m rag.app.mcp_server.server
    cd ~/agentic-data-cleaner-v2 && python -m mcp_server.mcp_app --transport http \\
        --host 127.0.0.1 --port 8001
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from jsonschema import Draft7Validator

from src.config.settings import settings
from src.mcp_client.client import MCPClient, MCPConnectionError, MCPToolError

REQUIRED_CLEANER_TOOLS = ["profile_dataset", "clean_dataset", "validate_quality"]
REQUIRED_RAG_TOOLS = ["retrieve", "retrieve_with_metadata", "list_collections"]

DEFAULT_TEST_CSV = "data/demo/sales_data.csv"
NONEXISTENT_PATH = f"/tmp/check_ecosystem_does_not_exist_{uuid.uuid4().hex[:8]}.csv"

VERDICT_ICON = {"pass": "PASS", "fail": "FAIL", "info": "INFO"}


# ─── Report ─────────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    system: str
    tool: str
    check: str  # "presence" | "request_schema" | "call" | "response_schema" | "nonexistent_file"
    verdict: str  # "pass" | "fail" | "info"
    detail: str = ""


@dataclass
class Report:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, system: str, tool: str, check: str, verdict: str, detail: str = "") -> None:
        self.results.append(CheckResult(system, tool, check, verdict, detail))

    def any_fail(self, system: Optional[str] = None, check: Optional[str] = None) -> bool:
        return any(
            r.verdict == "fail"
            and (system is None or r.system == system)
            and (check is None or r.check == check)
            for r in self.results
        )

    @property
    def failed(self) -> list[CheckResult]:
        return [r for r in self.results if r.verdict == "fail"]

    def print_table(self) -> None:
        col_w = (10, 24, 16, 6, 90)
        header = ("system", "tool", "check", "ok", "detail")
        print(
            "  ".join(h.ljust(w) for h, w in zip(header, col_w))
        )
        print("  ".join("-" * w for w in col_w))
        for r in self.results:
            row = (
                r.system,
                r.tool,
                r.check,
                VERDICT_ICON.get(r.verdict, r.verdict),
                r.detail[:col_w[4]],
            )
            print("  ".join(str(c).ljust(w) for c, w in zip(row, col_w)))


# ─── Local request-schema validation ───────────────────────────────────────

def validate_request_args(schema: dict, args: dict) -> Optional[str]:
    """
    Validate `args` against a tool's inputSchema *before* making the call.
    Returns None if valid, else a short human-readable error. This is the
    layer that catches "MAEDA's client code sends a shape the server's own
    declared schema rejects" without needing a network round trip — the
    exact class of bug 定案 #2 (path -> dataset_path) was.
    """
    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(args), key=lambda e: list(e.absolute_path))
    if not errors:
        return None
    first = errors[0]
    where = "/".join(str(p) for p in first.absolute_path) or "<root>"
    return f"{first.message} (at {where})"


# ─── Cleaner checks ─────────────────────────────────────────────────────────

async def check_cleaner(url: str, test_csv: str, report: Report) -> None:
    # 定案 #11: same-machine shared filesystem. The cleaner is a separate
    # process with its own CWD, so a relative path here resolves against
    # *its* working directory, not MAEDA's -- resolve to an absolute path
    # so the minimal-legal-call checks don't spuriously fail as
    # DATASET_NOT_FOUND regardless of how/where the cleaner was launched.
    test_csv = str(Path(test_csv).resolve())
    if not Path(test_csv).is_file():
        report.add(
            "cleaner", "*", "presence", "fail",
            f"--test-csv resolves to {test_csv}, which doesn't exist locally; "
            f"pass an existing CSV path with --test-csv",
        )
        return

    transport = MCPClient(url)
    try:
        tools = await transport.list_tools()
    except MCPConnectionError as exc:
        report.add("cleaner", "*", "presence", "fail", f"cannot reach cleaner at {url}: {exc}")
        return

    by_name = {t["name"]: t for t in tools}
    for name in REQUIRED_CLEANER_TOOLS:
        present = name in by_name
        report.add(
            "cleaner", name, "presence", "pass" if present else "fail",
            "" if present else "tool missing from list_tools",
        )
    if report.any_fail("cleaner", "presence"):
        return  # can't meaningfully call tools that don't exist

    calls = {
        "profile_dataset": {"dataset_path": test_csv},
        "clean_dataset": {"dataset_path": test_csv},
        "validate_quality": {"dataset_path": test_csv},
    }
    for name, args in calls.items():
        schema = by_name[name]["inputSchema"]
        err = validate_request_args(schema, args)
        report.add("cleaner", name, "request_schema", "fail" if err else "pass", err or "")
        if err:
            continue

        try:
            raw = await transport.call_tool(name, args)
        except MCPToolError as exc:
            report.add("cleaner", name, "call", "fail", f"protocol-level tool error: {exc}")
            continue
        except MCPConnectionError as exc:
            report.add("cleaner", name, "call", "fail", f"connection error: {exc}")
            continue

        out_schema = by_name[name].get("outputSchema")
        if out_schema:
            oerr = validate_request_args(out_schema, raw)
            report.add(
                "cleaner", name, "response_schema", "fail" if oerr else "pass",
                oerr or "response conforms to declared outputSchema",
            )
        elif isinstance(raw, dict) and raw.get("error") is True:
            report.add(
                "cleaner", name, "call", "fail",
                f"error envelope on a minimal legal call: "
                f"error_type={raw.get('error_type')} message={str(raw.get('message'))[:80]}",
            )
        else:
            report.add(
                "cleaner", name, "call", "pass",
                "no outputSchema declared; protocol ok, no error envelope",
            )

    # Acceptable-error contract: nonexistent file -> stable error shape, not a crash.
    for name in REQUIRED_CLEANER_TOOLS:
        args = {"dataset_path": NONEXISTENT_PATH}
        try:
            raw = await transport.call_tool(name, args)
        except MCPToolError as exc:
            report.add(
                "cleaner", name, "nonexistent_file", "info",
                f"surfaced as protocol-level tool error (also acceptable): {exc}",
            )
            continue
        except MCPConnectionError as exc:
            report.add("cleaner", name, "nonexistent_file", "fail", f"connection error: {exc}")
            continue

        if isinstance(raw, dict) and raw.get("error") is True:
            report.add(
                "cleaner", name, "nonexistent_file", "pass",
                f"stable-shaped error envelope, error_type={raw.get('error_type')} "
                f"(no error_code yet -- expected, C4 not landed)",
            )
        else:
            report.add(
                "cleaner", name, "nonexistent_file", "fail",
                f"expected an error envelope for a nonexistent file, got: {str(raw)[:150]}",
            )


# ─── RAG checks ─────────────────────────────────────────────────────────────

_NO_INDEX_HINTS = ("index", "no such collection", "not found", "empty", "no documents")


async def check_rag(url: str, collection: Optional[str], top_k: int, report: Report) -> None:
    transport = MCPClient(url)
    try:
        tools = await transport.list_tools()
    except MCPConnectionError as exc:
        report.add("rag", "*", "presence", "fail", f"cannot reach rag server at {url}: {exc}")
        return

    by_name = {t["name"]: t for t in tools}
    for name in REQUIRED_RAG_TOOLS:
        present = name in by_name
        report.add(
            "rag", name, "presence", "pass" if present else "fail",
            "" if present else "tool missing from list_tools",
        )
    if report.any_fail("rag", "presence"):
        return

    # list_collections first, to find a known-existing collection for retrieve.
    schema = by_name["list_collections"]["inputSchema"]
    err = validate_request_args(schema, {})
    report.add("rag", "list_collections", "request_schema", "fail" if err else "pass", err or "")

    collections_seen: list[str] = []
    if not err:
        try:
            raw = await transport.call_tool("list_collections", {})
            collections_seen = [
                c.get("name") for c in raw.get("collections", []) if isinstance(c, dict)
            ]
            report.add("rag", "list_collections", "call", "pass", f"collections={collections_seen}")
        except (MCPToolError, MCPConnectionError) as exc:
            report.add("rag", "list_collections", "call", "fail", str(exc))

    test_collection = collection
    if test_collection is None and collections_seen:
        non_default = [c for c in collections_seen if c and c != "default"]
        test_collection = non_default[0] if non_default else None
        # Leaving it None is also fine: rag_server.py searches the whole
        # knowledge base when collection is unset.

    for name in ("retrieve", "retrieve_with_metadata"):
        input_args: dict[str, Any] = {"query": "revenue trend", "top_k": top_k}
        if test_collection:
            input_args["collection"] = test_collection
        wrapped = {"input": input_args}

        schema = by_name[name]["inputSchema"]
        err = validate_request_args(schema, wrapped)
        report.add("rag", name, "request_schema", "fail" if err else "pass", err or "")
        if err:
            continue

        try:
            raw = await transport.call_tool(name, wrapped)
        except MCPToolError as exc:
            report.add("rag", name, "call", "fail", f"protocol-level tool error: {exc}")
            continue
        except MCPConnectionError as exc:
            report.add("rag", name, "call", "fail", f"connection error: {exc}")
            continue

        out_schema = by_name[name].get("outputSchema")
        if out_schema:
            oerr = validate_request_args(out_schema, raw)
            report.add(
                "rag", name, "response_schema", "fail" if oerr else "pass",
                oerr or "response conforms to declared outputSchema",
            )

        error_field = raw.get("error") if isinstance(raw, dict) else None
        chunks = raw.get("chunks") if isinstance(raw, dict) else None
        if error_field:
            lowered = str(error_field).lower()
            looks_like_no_index = any(hint in lowered for hint in _NO_INDEX_HINTS)
            explanation = (
                "looks like no-index-yet, not a contract violation -- record and move on per TB0"
                if looks_like_no_index else "unclear cause, inspect manually"
            )
            report.add(
                "rag", name, "call", "info" if looks_like_no_index else "fail",
                f"error field present: {str(error_field)[:120]!r} ({explanation})",
            )
        elif chunks is not None:
            report.add("rag", name, "call", "pass", f"{len(chunks)} chunk(s) returned, no error field")
        else:
            report.add("rag", name, "call", "fail", f"unexpected response shape: {str(raw)[:150]}")


# ─── Entry point ─────────────────────────────────────────────────────────────

async def _run(args: argparse.Namespace) -> int:
    report = Report()

    if not args.skip_cleaner:
        await check_cleaner(args.cleaner_url, args.test_csv, report)
    if not args.skip_rag:
        await check_rag(args.rag_url, args.rag_collection, args.top_k, report)

    report.print_table()

    fails = report.failed
    print()
    if fails:
        print(f"TB0: FAIL ({len(fails)} check(s) failed)")
        for r in fails:
            print(f"  - [{r.system}/{r.tool}/{r.check}] {r.detail}")
        return 1

    print("TB0: PASS")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cleaner-url", default=settings.data_cleaner_mcp_url)
    parser.add_argument("--rag-url", default=settings.rag_server_mcp_url)
    parser.add_argument("--rag-collection", default=None, help="Force a specific collection instead of auto-discovering one via list_collections.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--test-csv", default=DEFAULT_TEST_CSV, help="Existing CSV used for the minimal-legal-call checks against the cleaner.")
    parser.add_argument("--skip-cleaner", action="store_true")
    parser.add_argument("--skip-rag", action="store_true")
    args = parser.parse_args()

    exit_code = asyncio.run(_run(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
