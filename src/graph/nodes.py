"""
Node function registry for the MAEDA LangGraph graph.

Nodes for completed phases use real agent implementations.
Nodes for future phases remain as labeled stubs.

All I/O-bound nodes are `async def` and run under a single event loop —
the graph must be invoked via `graph.ainvoke(state)` (see src/graph/builder.py
and every call site: scripts/run_eval.py, scripts/demo_scenarios.py,
ui/app.py, src/mcp_server/server.py). Previously each node individually
wrapped its work in `asyncio.run()`, spinning up and tearing down a fresh
event loop per node — harmless in isolation, but async clients created in
one node (e.g. the MCP transport's httpx.AsyncClient) don't survive being
used from a *different* loop in a later node, producing the "Event loop is
closed" errors visible throughout this project's logs. handle_error_node
does no I/O and is left as a plain sync function — LangGraph runs sync and
async nodes together transparently under ainvoke().
"""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.config.settings import settings
from src.graph.router import _MAX_CLEAN_ITERATIONS
from src.mcp_client.fallback import SubSystemHardFailure
from src.state.graph_state import MAEDAState
from src.utils.logger import get_logger

logger = get_logger("maeda.nodes")

# ─── Agent / client singletons (lazy-init to avoid import-time construction) ──
_intent_parser = None
_subsystem_client = None
_data_connector = None
_analysis_agent = None
_viz_agent = None
_insight_agent = None
_guardrail_agent = None
_eval_runner = None
_run_store = None

def _get_intent_parser():
    global _intent_parser
    if _intent_parser is None:
        from src.agents.intent_parser import IntentParserAgent
        _intent_parser = IntentParserAgent()
    return _intent_parser

def _get_subsystem_client():
    global _subsystem_client
    if _subsystem_client is None:
        from src.mcp_client.fallback import build_subsystem_client
        _subsystem_client = build_subsystem_client()
    return _subsystem_client

def _get_data_connector():
    global _data_connector
    if _data_connector is None:
        from src.tools.data_connector import DataConnector
        _data_connector = DataConnector()
    return _data_connector

def _get_analysis_agent():
    global _analysis_agent
    if _analysis_agent is None:
        from src.agents.analysis_agent import AnalysisAgent
        _analysis_agent = AnalysisAgent()
    return _analysis_agent

def _get_viz_agent():
    global _viz_agent
    if _viz_agent is None:
        from src.agents.viz_agent import VizAgent
        _viz_agent = VizAgent()
    return _viz_agent

def _get_insight_agent():
    global _insight_agent
    if _insight_agent is None:
        from src.agents.insight_agent import InsightAgent
        _insight_agent = InsightAgent()
    return _insight_agent

def _get_guardrail_agent():
    global _guardrail_agent
    if _guardrail_agent is None:
        from src.agents.guardrail_agent import GuardrailAgent
        _guardrail_agent = GuardrailAgent()
    return _guardrail_agent

def _get_eval_runner():
    global _eval_runner
    if _eval_runner is None:
        from src.eval.runner import EvalRunner
        _eval_runner = EvalRunner()
    return _eval_runner

def _get_run_store():
    global _run_store
    if _run_store is None:
        from src.persistence.run_store import RunStore
        _run_store = RunStore()
    return _run_store


# ─── Cleaning-loop helpers (M7, TB0.5 v1 — 附录 B) ─────────────────────────────

# 附录 B.1's thresholds, frozen under quality_contract_version "1" (附录 B.7:
# "禁止在'1'版本号下改变上述任何判据"). Mirrors cleaner's own
# _HCI_MISSING_THRESHOLD/_HCI_DUP_THRESHOLD (mcp_app.py) so the signature
# computed here matches has_critical_issues's own logic exactly — if TB0.5
# ever bumps to v2 these must move in lockstep with the server-side values.
_HCI_MISSING_THRESHOLD = 0.10
_HCI_DUP_THRESHOLD = 0.05


def _file_is_readable(path: str) -> bool:
    """Existence + readability check for 附录 B's "输出文件存在、可读、被接管
    才置 cleaning_applied=true" requirement — a plain Path.is_file() can be
    true for a file you don't actually have permission to read."""
    try:
        with open(path, "rb"):
            return True
    except OSError:
        return False


def _sha256_file(path: str) -> Optional[str]:
    """
    Self-computed content hash, used for 附录 B.4 stop condition #2 ("no
    actual change") as a fallback when the server doesn't supply its own
    hashes. cleaner has returned input_sha256/output_sha256 since its C3
    (M8); 定案 #11's same-machine shared filesystem assumption is what makes
    self-hashing valid at all when an older server doesn't. None means
    "could not read the file", which callers must treat as inconclusive,
    not as proof of "no diff".
    """
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


# ─── Path/hash protection (M8, 定案 #16) ────────────────────────────────────

def _resolved_artifact_root() -> str:
    """
    Always resolved to absolute before being sent anywhere (定案 #16):
    sending the setting's raw (possibly relative) string across processes
    would have MAEDA and the cleaner each resolve it against their own
    CWD, silently landing on two different directories — exactly the F1
    class of bug this whole run_id/artifact_root contract exists to
    prevent. 定案 #11's same-machine shared filesystem assumption is what
    makes one resolved path valid for both sides to agree on.
    """
    return str(Path(settings.maeda_artifact_root).resolve())


# Matches DataConnector.connect's own type dispatch (data_connector.py:
# `source_type in {"csv", "tsv", ""}` / `"json"` / `"excel"` / `"sql"`) --
# every source type whose "path" is actually a filesystem path, as opposed
# to "sql"'s SQLAlchemy connection string.
_FILE_BACKED_SOURCE_TYPES = {"csv", "tsv", "excel", "json", ""}


def _resolved_dataset_path(path: str, source_type: str) -> str:
    """
    定案 #16: every cross-process file path must be a normalized absolute
    path, for exactly the reason _resolved_artifact_root() above exists --
    MAEDA and the cleaner are separate processes with separate CWDs, and a
    relative path resolves against whichever process happens to read it.
    This is the SAME class of bug the plan's own history already
    documents being caught and fixed once for `artifact_root`/
    `cleaned_path` (附录 D's F1, 附录 F.4's G1, 附录 H.4) -- what slipped
    through all three of those rounds is the dataset path itself: `state
    ["data_sources"][0]["path"]` (e.g. "data/demo/sales_data.csv", the
    literal value scripts/run_eval.py's GoldenTestCase.data_source
    entries and CLI/eval callers use) was sent to `profile_dataset`/
    `clean_dataset`/`validate_quality` and embedded in `intent_payload
    ["dataset_path"]` completely unresolved. TB0's smoke test
    (scripts/check_ecosystem.py) never caught this because IT resolves
    its own test path before calling the cleaner
    (`str(Path(test_csv).resolve())`) -- passing was never evidence the
    production graph honored #16 here, only that the smoke script did on
    the graph's behalf. Unit tests couldn't catch it either: the cleaner
    is mocked in every one of them, so nothing ever actually resolved a
    relative path against a different CWD. It took a real cross-process
    run to surface it.

    Only resolves file-backed source types (see _FILE_BACKED_SOURCE_TYPES,
    mirroring data_connector.py's own dispatch) -- a "sql" source's
    "path" is a SQLAlchemy connection string (e.g. "sqlite:///data/demo/
    ecommerce_orders.db"), not a filesystem path. Running it through
    Path(...).resolve() would silently produce garbage (the "sqlite://"
    scheme mangled into a bogus path segment) instead of a real path, not
    raise anything -- worse than doing nothing. Left byte-for-byte
    unchanged for those; whether/how a SQL connection string should cross
    this same boundary is a separate, pre-existing design question this
    fix does not take on.
    """
    if source_type not in _FILE_BACKED_SOURCE_TYPES or not path:
        return path
    try:
        return str(Path(path).resolve())
    except (OSError, ValueError):
        # A malformed path shouldn't crash the graph here -- let the
        # cleaner's own error handling surface it downstream with a
        # legible MCP error instead of MAEDA failing earlier with a raw
        # pathlib traceback.
        return path


def _validate_cleaned_path(
    cleaned_path: str, effective_path: str, artifact_root: str, run_id: str,
) -> Optional[str]:
    """
    M8: re-validate clean_dataset's output rather than trusting it blindly
    — absolute, normalized (no ".." segments survive a round-trip through
    os.path.normpath), inside *this call's own* run_id directory under
    artifact_root (not just "somewhere under artifact_root" — a stable
    per-call boundary is what makes concurrent-trial isolation and "which
    round produced this" auditability hold), genuinely distinct from the
    input path, readable, and in a supported format (cleaner's pipeline
    always emits CSV, regardless of input format — 定案 #16).

    Returns None if all checks pass, else a short human-readable reason.
    Uses os.path.commonpath rather than a plain string prefix check so a
    sibling directory that merely shares a prefix (e.g. ".../run_1x" vs
    ".../run_1") can't be mistaken for "inside" the expected run directory.
    """
    if not cleaned_path:
        return "returned an empty cleaned_path"
    if not os.path.isabs(cleaned_path):
        return f"cleaned_path {cleaned_path!r} is not an absolute path"
    if os.path.normpath(cleaned_path) != cleaned_path:
        return f"cleaned_path {cleaned_path!r} is not a normalized path"

    expected_root = str(Path(artifact_root) / run_id)
    try:
        inside_expected_root = os.path.commonpath([cleaned_path, expected_root]) == expected_root
    except ValueError:
        inside_expected_root = False  # e.g. different drives on Windows
    if not inside_expected_root:
        return f"cleaned_path {cleaned_path!r} is outside this run's artifact directory ({expected_root!r})"

    if cleaned_path == effective_path:
        return "cleaned_path is identical to the input path (no distinct output produced)"
    if not _file_is_readable(cleaned_path):
        return f"output {cleaned_path!r} doesn't exist or isn't readable"
    if Path(cleaned_path).suffix.lower() != ".csv":
        return f"output {cleaned_path!r} is not in a supported format (expected .csv)"
    return None


def _cleaning_signature(validation) -> dict:
    """
    Recomputes the same three-dimension verdict has_critical_issues is
    built from (附录 B.0/B.1) out of validate_quality.details, so
    round-over-round comparison (附录 B.4 #3) can tell "identical failure
    two rounds running" apart from "still bad, but for a different/fewer
    reasons, still making progress". Comparing the single OR'd
    has_critical_issues boolean alone can't make that distinction.
    """
    details = validation.details or {}
    return {
        "missing": details.get("mean_null_ratio", 0.0) > _HCI_MISSING_THRESHOLD,
        "duplicate": details.get("duplicate_row_ratio", 0.0) > _HCI_DUP_THRESHOLD,
        "schema": details.get("schema_score", 1.0) < 1.0,
    }


# ─── TB3 intent payload construction (P.1/P.2, 附录 U.1/U.2/U.3) ──────────────

def _scope_fingerprint(columns: list[str]) -> str:
    """
    附录 U.3 frozen algorithm — must byte-for-byte match cleaner's
    `_scope_fingerprint()` (mcp_app.py): sha256 of the sorted column list,
    first 16 hex chars. Computed independently here (not read back from any
    single response) so the check below verifies cleaner's *cross-call*
    consistency for the round, not just "did this one response echo back
    whatever intent it happened to receive" — see the module-level note on
    same-round intent consistency where this is used.
    """
    payload = json.dumps(sorted(columns), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _resolve_intent_columns(
    parsed_intent: dict, schema_columns: list,
) -> tuple[list[dict], list[dict], str]:
    """
    P.1 (附录 U.2's solution (d)): deterministic column-name reconciliation
    between the LLM-parsed intent's free-text mentions and the dataset's
    real schema. Runs after schema extraction, before intent is sent to
    cleaner — cleaner treats `resolved_columns` as pre-validated and does
    no second matching pass of its own (附录 U.2).

    Only two match tiers are implemented: `exact` and `case_insensitive`.
    `glossary_alias` is reserved in the frozen schema but never produced
    here — 附录 S.2 confirmed the glossary has no content yet (no file, no
    loader in this repo), so alias matching would always be a dead branch.
    It activates automatically once 阶段 3's glossary work lands; nothing
    here needs to change for that.

    Mentions this function knows how to extract from `ParsedIntent`
    (src/agents/intent_parser.py): `target_metrics` → role "metric",
    `dimensions`/`sort_by` → role "dimension", `filters[].column` → role
    "filter". `role` is a MAEDA-side implementation choice (the frozen
    contract only fixes the four allowed values, not how MAEDA derives
    them) — a resolved column gets upgraded to role "time" whenever its own
    schema dtype says datetime, since ParsedIntent has no separate list of
    "time" mentions to read from.

    This cannot see the failure mode 附录 P.9.1 describes (a column the LLM
    never mentioned at all, e.g. `discount_amount` when only `revenue` and
    `quarter` were asked about) — that happens upstream in parse_intent,
    before any schema exists to reconcile against. What this function does
    catch is the other half: a mention that *was* extracted but doesn't
    match any real column, or only matches after normalizing case.
    """
    by_exact = {c.name: c for c in schema_columns}
    by_lower = {c.name.lower(): c for c in schema_columns}

    def _role_for(col, requested_role: str) -> str:
        return "time" if getattr(col, "is_datetime", False) else requested_role

    def _resolve_one(mention: str, role: str) -> Optional[dict]:
        col = by_exact.get(mention)
        if col is not None:
            return {"name": col.name, "role": _role_for(col, role), "matched_from": mention, "match": "exact"}
        col = by_lower.get(mention.lower())
        if col is not None:
            return {"name": col.name, "role": _role_for(col, role), "matched_from": mention, "match": "case_insensitive"}
        return None

    mentions: list[tuple[str, str]] = []
    for m in parsed_intent.get("target_metrics") or []:
        if m:
            mentions.append((m, "metric"))
    for m in parsed_intent.get("dimensions") or []:
        if m:
            mentions.append((m, "dimension"))
    for f in parsed_intent.get("filters") or []:
        col_name = (f or {}).get("column")
        if col_name:
            mentions.append((col_name, "filter"))
    sort_by = parsed_intent.get("sort_by")
    if sort_by:
        mentions.append((sort_by, "dimension"))

    resolved: list[dict] = []
    unresolved: list[dict] = []
    seen_names: set[str] = set()
    for mention, role in mentions:
        match = _resolve_one(mention, role)
        if match is None:
            unresolved.append({"text": mention, "role": role, "reason": "no_match", "candidates": []})
        elif match["name"] not in seen_names:
            # Same column mentioned twice under different roles (e.g. both
            # a metric and the sort_by column) — one resolved_columns entry
            # is enough; the first role assigned wins.
            resolved.append(match)
            seen_names.add(match["name"])

    # 附录 U.4.2: cleaner's own _resolve_scope() treats an empty
    # column_scope.columns as "absent" regardless of what resolution_status
    # says, so "absent" here whenever nothing resolved keeps this field
    # honest about what will actually happen, not just internally consistent.
    if not resolved:
        status = "absent"
    elif unresolved:
        status = "partial"
    else:
        status = "full"

    return resolved, unresolved, status


def _build_intent_payload(
    dataset_path: str, parsed_intent: dict,
    resolved_columns: list[dict], unresolved_mentions: list[dict], resolution_status: str,
) -> dict:
    """
    P.2 (附录 U.2 frozen schema). `column_scope.profiling`/`mutation` are
    frozen literal values (附录 U.2/U.9) — not configurable here: `mutation`
    stays "advisory" until a column-resolution recall benchmark exists to
    justify "restrict" (附录 P.9.1's dilution-risk argument), which is not
    this round's work. `glossary` is always empty — 附录 S.2/U.9: no
    glossary content or loader exists in this repo yet (阶段 3).
    """
    columns = [r["name"] for r in resolved_columns]
    return {
        "intent_contract_version": "1",
        "dataset_path": dataset_path,
        "detected_issues": [],
        "analysis": {
            "query_type": parsed_intent.get("query_type", "exploratory"),
            "resolution_status": resolution_status,
            "resolved_columns": resolved_columns,
            "unresolved_mentions": unresolved_mentions,
        },
        "column_scope": {
            "columns": columns,
            "profiling": "restrict",
            "mutation": "advisory",
        },
        "glossary": [],
    }


def _check_scope_fingerprint(
    state: MAEDAState, tool: str, expected: str, actual: Optional[str], call_mode: str,
) -> Optional[str]:
    """
    附录 U.3: every call in one cleaning round (含 re-profile — 附录 T.2's
    correction to R.1, not just "one call per tool name") must be verified
    against the same expected fingerprint, computed once from the intent
    this round actually built (see `_scope_fingerprint` above) — not
    re-derived from whatever `intent` each individual call happened to
    receive, which would not catch a call that silently got a different (or
    no) intent by mistake.

    `call_mode` is the mcp_call_log "mode" for this call ("mcp" |
    "fallback") — a degraded-mode fallback response never went near cleaner
    and carries no real scope_fingerprint, so there is nothing to check.
    Returns an error message on mismatch, None otherwise.
    """
    if call_mode != "mcp":
        return None
    if actual is None:
        # Server predates TB3+TB4 (v1-only) — nothing to check against.
        return None
    if actual != expected:
        return (
            f"{tool} scope_fingerprint mismatch this round: expected {expected!r}, got {actual!r} "
            "(附录 U.3: every call in a cleaning round must be verified against the same intent)"
        )
    return None


def _cleaning_applied_level(state: MAEDAState) -> str:
    """
    附录 R.3: a derived (not separately stored) trial-recording field —
    "full" (cleaning ran and completed a round), "blocked_needs_review"
    (cleaning was triggered but a step's risk tier blocked it, or the
    server flagged the whole round for manual review), or "none" (cleaning
    never triggered this run). Kept as a pure function of `cleaning_applied`
    /`cleaning_stop_reason` rather than a third state field so it can never
    drift out of sync with the two fields it is entirely determined by —
    call this wherever the value is actually needed (decision trace now;
    D0's trial records later, per R.3's calibration: "D0 落地时必须已经在记").
    """
    if state.get("cleaning_stop_reason") == "needs_review":
        return "blocked_needs_review"
    if state.get("cleaning_applied"):
        return "full"
    return "none"


def _trace(state: MAEDAState, agent_name: str, action: str, reasoning: str) -> MAEDAState:
    """Append a minimal decision trace record (no LLM, no cost)."""
    record = {
        "agent_name": agent_name,
        "action": action,
        "reasoning": reasoning,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "inputs": None,
        "outputs": None,
        "confidence": 1.0,
    }
    state["decision_trace"] = [*state.get("decision_trace", []), record]
    return state


# ─── Nodes ────────────────────────────────────────────────────────────────────

async def parse_intent_node(state: MAEDAState) -> MAEDAState:
    """Phase 2: real LLM-based intent parsing."""
    logger.info("Node: parse_intent | query=%s", state.get("user_query", "")[:80])
    state["current_phase"] = "plan"
    return await _get_intent_parser().process(state)


async def ask_clarification_node(state: MAEDAState) -> MAEDAState:
    """Phase 2: surface clarification question to user."""
    logger.info("Node: ask_clarification")
    state["clarification_count"] = state.get("clarification_count", 0) + 1
    return await _get_intent_parser().generate_clarification_question(state)


async def connect_schema(state: MAEDAState) -> MAEDAState:
    """
    Phase 4 step 1 (E2 BO.1 split, submission 1 — ECOSYSTEM_INTEGRATION_PLAN.md
    附录 BQ): connect to the data source and extract schema + NL summary,
    including E1's intent-aware table selection for multi-table SQL
    sources. First half of what used to be one 460+-line
    `connect_and_profile_node` — split into two real LangGraph nodes so a
    later `refine_intent` node (E2 submission 2) can sit between "schema
    known" and "cleaner invoked" on its own conditional edge, instead of
    being a branch buried inside a monolithic function.

    Hands off to `profile_and_clean` through two state fields, not local
    variables — LangGraph nodes only communicate via state, they don't
    share a Python call stack. `effective_dataset_path` carries 定案 #16's
    already-resolved path; `schema_columns` carries the real `ColumnInfo`
    objects `_resolve_intent_columns` needs (`data_sources[0]['schema']
    ['columns']`'s dict form doesn't support the attribute access that
    function does). Both are recomputed fresh on every entry to this node
    and are not meant to be read by anything outside this connect_schema /
    profile_and_clean pair.
    """
    logger.info("Node: connect_schema")
    state["current_phase"] = "plan"

    sources = state.get("data_sources", [])
    if not sources:
        state["error"] = "No data source provided. Please upload a file or specify a data path."
        state["current_phase"] = "error"
        return state

    source = sources[0]
    source_path = source.get("path", "")
    connector = _get_data_connector()

    # Step 1: Connect and extract schema + NL summary
    # E1: parsed_intent is already in state by this point (parse_intent runs
    # before this node) and gets forwarded so a multi-table SQL source
    # without an explicit table_name is matched against what the user
    # actually asked about, instead of DataConnector always picking
    # whichever table SQLAlchemy's inspector happens to list first.
    schema = None
    try:
        schema, nl_summary = await connector.connect_with_summary(
            source, intent=state.get("parsed_intent"),
        )
        state["active_source"] = schema.to_source_dict()
        state["schema_summary"] = nl_summary
        # Merge schema back into the source descriptor in state
        state["data_sources"] = [
            {**source, "schema": schema.to_dict(), "preview": schema.preview},
            *sources[1:],
        ]
        # 定案 #16 (see _resolved_dataset_path's docstring): resolve BEFORE
        # this crosses to the cleaner, not after -- source_path itself
        # (e.g. "data/demo/sales_data.csv") is relative to MAEDA's own
        # CWD, and the cleaner is a separate process with its own.
        effective_path = _resolved_dataset_path(source_path, schema.source_type)
        if schema.source_type == "sql":
            _trace(
                state, "data_connector", "table_selection",
                f"resolved table={schema.table_name!r} for {source_path}",
            )
    except Exception as exc:
        logger.warning("DataConnector failed for %s: %s", source_path, exc)
        # Schema extraction failed — still run MCP profiling on original
        # path. source.get("type") is the same fallback convention
        # viz_agent.py's _load_df uses (defaults to "csv", the most
        # common case) since there's no successfully-extracted schema to
        # read source_type off of here.
        state["schema_summary"] = f"Schema unavailable: {exc}"
        effective_path = _resolved_dataset_path(source_path, source.get("type", "csv"))

    state["effective_dataset_path"] = effective_path
    state["schema_columns"] = schema.columns if schema is not None else None
    return state


async def profile_and_clean(state: MAEDAState) -> MAEDAState:
    """
    Phase 4 step 2 (E2 BO.1 split, submission 1 — ECOSYSTEM_INTEGRATION_PLAN.md
    附录 BQ): reconcile the parsed intent against the schema `connect_schema`
    just extracted, delegate quality profiling to Data Cleaner MCP, and run
    at most one clean+validate+re-profile round if the data has critical
    issues (TB0.5 v1, 附录 B). route_after_profiling decides whether another
    LangGraph tick through this node is needed — this function's job each
    tick is to do exactly one round and leave state honest about what
    happened. Second half of the pre-split `connect_and_profile_node` (see
    `connect_schema`'s docstring for why the function was split).

    Guards on `current_phase == "error"` at the top so connect_schema's
    "no data source" exit — the only error connect_schema can raise before
    `data_sources` is ever touched — propagates through unchanged instead
    of this function dereferencing a `data_sources[0]` that was never
    populated. This is plain early-return plumbing to preserve identical
    behavior across the split, not a new routing decision: the actual
    "error" -> handle_error edge is still owned entirely by
    route_after_profiling downstream, unchanged from before the split.

    Fixes a confirmed real bug (M7): the cleaned file used to never get
    consumed by later rounds or by analysis — `data_sources[0].path` was
    never updated, so every loop iteration re-profiled and re-cleaned the
    *original* dirty file, and `data_quality_report` stayed frozen at the
    pre-clean state forever. This version writes `cleaned_path` back into
    `data_sources[0]`/`active_source`, re-profiles the cleaned file so the
    router sees fresh `has_critical_issues`, and calls `validate_quality`
    as the actual exit check (previously wired up but never invoked).
    """
    logger.info("Node: profile_and_clean")
    if state.get("current_phase") == "error":
        return state

    mcp_client = _get_subsystem_client()
    connector = _get_data_connector()

    source = state.get("data_sources", [{}])[0]
    source_path = source.get("path", "")
    effective_path = state.get("effective_dataset_path")
    schema_columns = state.get("schema_columns")

    # M4 / 定案 #16: per-call run_id derived from the pipeline's own run_id,
    # now that cleaner's C3 actually honors it (附录 E P4 is resolved).
    # Each call gets a DISTINCT id, not one shared id for the whole pipeline
    # run: run_two_stage's stage1/stage2 directories are fixed names under
    # run_root, so reusing the same run_id across two separate clean_dataset
    # calls would silently overwrite round 1's artifacts with round 2's.
    # The shared prefix still lets every artifact from this pipeline run be
    # found/grouped by directory-name prefix (附录 E.2 P1).
    pipeline_run_id = state.get("run_id", "")
    round_index = state.get("iteration_count", 0)
    artifact_root = _resolved_artifact_root()

    def _round_run_id(label: str, index: int) -> str:
        return f"{pipeline_run_id}_{label}{index}" if pipeline_run_id else ""

    # TB3 (附录 U.1/U.2/P.1): resolve the parsed intent's free-text column
    # mentions against the real schema (empty list -> resolution_status
    # "absent" if schema extraction failed above), and build the one intent
    # payload every MCP call this round will carry. Built once and reused
    # as the same object for every call below, so 附录 U.3's "same intent
    # every call" requirement holds by construction, not just convention —
    # the fingerprint check further down verifies cleaner's honesty about
    # that intent, not MAEDA's own internal consistency.
    resolved_columns, unresolved_mentions, resolution_status = _resolve_intent_columns(
        state.get("parsed_intent") or {}, schema_columns if schema_columns is not None else [],
    )
    intent_payload = _build_intent_payload(
        effective_path, state.get("parsed_intent") or {},
        resolved_columns, unresolved_mentions, resolution_status,
    )
    state["cleaning_intent"] = intent_payload
    expected_fingerprint = _scope_fingerprint([r["name"] for r in resolved_columns])
    _trace(
        state, "data_connector", "intent_column_resolution",
        f"resolution_status={resolution_status!r}, "
        f"resolved={[r['name'] for r in resolved_columns]}, "
        f"unresolved={[u['text'] for u in unresolved_mentions]}",
    )

    # Step 2: Delegate quality profiling to Data Cleaner MCP
    try:
        report, prof_log = await mcp_client.profile_dataset(
            effective_path, run_id=_round_run_id("profile", round_index), artifact_root=artifact_root,
            intent=intent_payload,
        )
    except SubSystemHardFailure as exc:
        # 错误处理矩阵 (M1): param/contract/data-input errors must not be
        # papered over in either mode -- surface as a real pipeline error
        # rather than proceeding as if profiling succeeded.
        state["mcp_call_log"] = [*state.get("mcp_call_log", []), exc.log]
        state["error"] = f"Data quality profiling failed ({exc.error_class}): {exc}"
        state["current_phase"] = "error"
        return _trace(state, "data_connector", "connect_and_profile",
                      f"profile_dataset hard failure: {exc}")
    state["mcp_call_log"] = [*state.get("mcp_call_log", []), prof_log]
    state["data_quality_report"] = report.to_dict()

    fingerprint_error = _check_scope_fingerprint(
        state, "profile_dataset", expected_fingerprint, report.scope_fingerprint,
        prof_log.get("mode", "mcp"),
    )
    if fingerprint_error:
        state["error"] = fingerprint_error
        state["current_phase"] = "error"
        return _trace(state, "data_connector", "connect_and_profile", fingerprint_error)
    if report.critical_basis:
        _trace(
            state, "data_connector", "critical_basis",
            f"profile_dataset: has_critical_issues={report.has_critical_issues}, "
            f"critical_basis={report.critical_basis}",
        )

    if report.needs_review:  # 附录 B.3 — no cleaner tool emits this today; forward-compat
        state["cleaning_stop_reason"] = "needs_review"
        state["cleaning_caveat"] = (
            "Data Cleaner flagged this dataset for manual review before any "
            "automatic cleaning; proceeding to analysis on the original data."
        )
        state["cleaning_applied_level"] = _cleaning_applied_level(state)
        return _trace(state, "data_connector", "connect_and_profile",
                      "profile_dataset returned needs_review; skipping auto-clean")

    if not report.has_critical_issues:
        # Nothing (more) to clean this round. If we already completed at
        # least one clean round earlier in this run, this is genuine
        # convergence, not "never needed cleaning" — record it either way
        # so a stale stop_reason from an earlier round can't linger.
        if state.get("iteration_count", 0) > 0:
            state["cleaning_stop_reason"] = "passed"
        state["cleaning_applied_level"] = _cleaning_applied_level(state)
        return _trace(state, "data_connector", "connect_and_profile",
                      f"Connected to {source_path}; critical_issues=False")

    # has_critical_issues=True from here on — run one clean+validate+re-profile round.
    if state.get("iteration_count", 0) >= _MAX_CLEAN_ITERATIONS:
        # route_after_profiling should have stopped looping before this
        # node ran again; this is a defensive backstop, not the primary
        # enforcement point, so it doesn't silently fall through to "ready"
        # without a caveat if it's ever reached some other way.
        state["cleaning_stop_reason"] = "max_rounds"
        state["cleaning_caveat"] = (
            f"Reached the maximum of {_MAX_CLEAN_ITERATIONS} automatic cleaning "
            "rounds without the data passing quality validation; proceeding "
            "to analysis on the best available (partially cleaned) version."
        )
        state["cleaning_applied_level"] = _cleaning_applied_level(state)
        return _trace(state, "data_connector", "connect_and_profile",
                      "max_rounds already reached; not attempting another clean")

    pre_clean_hash = _sha256_file(effective_path)

    try:
        # 定案 #3: call clean_dataset directly, without a get_cleaning_plan
        # round-trip. Confirmed live (not hypothetical) that the round-trip
        # is actually broken, not just an avoidable double-plan: cleaner's
        # get_cleaning_plan response is a MAEDA-facing reshaping of its
        # internal plan (steps filtered/renamed for display), not the full
        # internal "Plan Contract" document clean_dataset's `plan` argument
        # expects back — feeding it back gets rejected server-side with
        # "Plan missing top-level field: 'intent'". Passing no plan lets
        # clean_dataset generate and use its own internally-valid one.
        # M4: planner_mode read from Settings (定案 #6) and max_rounds
        # pinned to 1 (定案 #5) — MAEDA's graph loop is the outer round
        # controller, cleaner's own internal multi-round feedback loop
        # would be a second, uncoordinated one if left at its default.
        clean_run_id = _round_run_id("clean", round_index + 1)
        result, clean_log = await mcp_client.clean_dataset(
            effective_path,
            planner_mode=settings.data_cleaner_planner_mode,
            max_rounds=1,
            run_id=clean_run_id,
            artifact_root=artifact_root,
            intent=intent_payload,
        )
        state["mcp_call_log"] = [*state.get("mcp_call_log", []), clean_log]
    except SubSystemHardFailure as exc:
        state["mcp_call_log"] = [*state.get("mcp_call_log", []), exc.log]
        state["error"] = f"Data cleaning failed ({exc.error_class}): {exc}"
        state["current_phase"] = "error"
        return _trace(state, "data_connector", "connect_and_profile",
                      f"cleaning hard failure: {exc}")

    fingerprint_error = _check_scope_fingerprint(
        state, "clean_dataset", expected_fingerprint, result.scope_fingerprint,
        clean_log.get("mode", "mcp"),
    )
    if fingerprint_error:
        state["error"] = fingerprint_error
        state["current_phase"] = "error"
        return _trace(state, "data_connector", "connect_and_profile", fingerprint_error)

    # 附录 B.5: counts *completed* clean_dataset calls, incremented right
    # after the call returns successfully — not on every node entry (that
    # was the pre-M7 off-by-one that capped the loop at 2 actual clean
    # attempts despite _MAX_CLEAN_ITERATIONS = 3), and not gated on whether
    # the output turns out to be usable below ("调用成功返回后自增").
    state["iteration_count"] = state.get("iteration_count", 0) + 1

    # M4: record the execution plan cleaner actually used (live since its
    # C3) as its own decision-trace entry — separate from the round-summary
    # trace below because this is worth recording even on the branches that
    # return early afterward (needs_review, unusable output, no_diff, ...).
    # U.7: also kept in state (not just the trace string above) so the
    # Insight Agent can read U.5's lineage (risk_tier/escalated_by/impact
    # per step) at report-generation time, long after this node has
    # returned — the trace string is for human/log readability, this is
    # the structured source of truth the disclosure chain reads from.
    plan = result.execution_plan
    state["cleaning_execution_plan"] = plan
    _trace(
        state, "data_connector", "clean_dataset_execution_plan",
        f"planner_mode_requested={plan.get('planner_mode_requested')!r}, "
        f"planner_mode_used={plan.get('planner_mode_used')!r}, "
        f"planner_fallback_reason={plan.get('planner_fallback_reason')!r}, "
        f"steps={len(plan.get('steps', []))}",
    )

    if result.needs_review:
        state["cleaning_stop_reason"] = "needs_review"
        state["cleaning_caveat"] = (
            "Data Cleaner flagged this cleaning run for manual review; "
            "stopping automatic cleaning and proceeding with the "
            "pre-cleaning data."
        )
        state["cleaning_applied_level"] = _cleaning_applied_level(state)
        blocked = [
            {"step_id": s.get("step_id"), "mcp_tool": s.get("mcp_tool"),
             "risk_tier": s.get("risk_tier"), "confidence": s.get("confidence")}
            for s in plan.get("steps", []) if s.get("risk_tier") == "high_risk"
        ]
        if blocked:  # 附录 R.3: which step(s) actually triggered the block, not just that one did
            _trace(state, "data_connector", "cleaning_blocked_steps", f"blocked_steps={blocked}")
        return _trace(state, "data_connector", "connect_and_profile",
                      "clean_dataset returned needs_review")

    cleaned_path = result.cleaned_path

    # M8 / 定案 #16: re-validate the output rather than trusting it blindly
    # — absolute, normalized, inside *this call's own* run_id directory
    # under artifact_root, distinct from the input, readable, supported
    # format. Superset of the old "does it exist and can we read it" check.
    path_validation_error = _validate_cleaned_path(cleaned_path, effective_path, artifact_root, clean_run_id)
    _trace(
        state, "data_connector", "clean_dataset_path_validation",
        f"cleaned_path={cleaned_path!r}, valid={path_validation_error is None}, "
        f"reason={path_validation_error!r}, input_sha256={result.input_sha256!r}, "
        f"output_sha256={result.output_sha256!r}, output_bytes={result.output_bytes!r}",
    )
    if path_validation_error:
        # 附录 B's "仅当输出文件存在、可读、被接管时才置 cleaning_applied=true"
        # means a failed validation must not be treated as if cleaning happened.
        state["cleaning_stop_reason"] = "no_diff"
        state["cleaning_caveat"] = (
            f"Data Cleaner's output failed validation ({path_validation_error}); "
            "proceeding to analysis on the pre-cleaning data."
        )
        state["cleaning_applied_level"] = _cleaning_applied_level(state)
        return _trace(state, "data_connector", "connect_and_profile",
                      f"clean_dataset output failed path validation: {path_validation_error}")

    # M8: prefer the server's own content hashes (live since cleaner's C3)
    # over self-hashing -- both are computed from the same ingested-CSV
    # source on the server side, so comparing them directly is more correct
    # than comparing MAEDA's hash of the raw (possibly non-CSV) input
    # against a hash of the CSV output. Fall back to self-hashing only if
    # an older server didn't send them.
    if result.input_sha256 is not None and result.output_sha256 is not None:
        no_diff = result.input_sha256 == result.output_sha256
    else:
        post_clean_hash = _sha256_file(cleaned_path)
        no_diff = (
            pre_clean_hash is not None and post_clean_hash is not None
            and pre_clean_hash == post_clean_hash
        )

    # Adopt the cleaned file — this is the fix for the confirmed bug: later
    # rounds (and downstream analysis) must read the cleaned data, not the
    # original.
    state["cleaning_applied"] = True
    state["cleaning_summary"] = result.changes_summary
    state["data_sources"] = [
        {**state["data_sources"][0], "path": cleaned_path},
        *state["data_sources"][1:],
    ]
    try:
        cleaned_source = {**source, "path": cleaned_path}
        schema2, nl2 = await connector.connect_with_summary(cleaned_source)
        state["active_source"] = schema2.to_source_dict()
        state["schema_summary"] = nl2
    except Exception:
        pass  # Keep prior schema if re-extraction fails

    if no_diff:
        state["cleaning_stop_reason"] = "no_diff"
        state["cleaning_caveat"] = (
            "Data Cleaner ran but produced byte-identical output; stopping "
            "the cleaning loop rather than repeating a no-op round."
        )
        state["cleaning_applied_level"] = _cleaning_applied_level(state)
        return _trace(state, "data_connector", "connect_and_profile",
                      "clean_dataset produced no actual change (hash match)")

    # Re-profile the cleaned file so the router sees a fresh verdict —
    # this replaces the stale pre-clean report that used to linger for the
    # rest of the run.
    try:
        report2, prof2_log = await mcp_client.profile_dataset(
            cleaned_path, run_id=_round_run_id("reprofile", state["iteration_count"]), artifact_root=artifact_root,
            intent=intent_payload,
        )
    except SubSystemHardFailure as exc:
        state["mcp_call_log"] = [*state.get("mcp_call_log", []), exc.log]
        state["error"] = f"Re-profiling cleaned data failed ({exc.error_class}): {exc}"
        state["current_phase"] = "error"
        return _trace(state, "data_connector", "connect_and_profile",
                      f"re-profile hard failure: {exc}")
    state["mcp_call_log"] = [*state.get("mcp_call_log", []), prof2_log]
    state["data_quality_report"] = report2.to_dict()

    # 附录 T.2's correction to R.1: re-profile is the same tool called a
    # second time this round, and is exactly the call most likely to be
    # forgotten when this code is next edited — it must be checked too, not
    # just the three distinctly-named tools.
    fingerprint_error = _check_scope_fingerprint(
        state, "profile_dataset(re-profile)", expected_fingerprint, report2.scope_fingerprint,
        prof2_log.get("mode", "mcp"),
    )
    if fingerprint_error:
        state["error"] = fingerprint_error
        state["current_phase"] = "error"
        return _trace(state, "data_connector", "connect_and_profile", fingerprint_error)
    if report2.critical_basis:
        _trace(
            state, "data_connector", "critical_basis",
            f"re-profile: has_critical_issues={report2.has_critical_issues}, "
            f"critical_basis={report2.critical_basis}",
        )

    if report2.needs_review:
        state["cleaning_stop_reason"] = "needs_review"
        state["cleaning_caveat"] = (
            "Data Cleaner flagged the cleaned data for manual review; "
            "stopping automatic cleaning."
        )
        state["cleaning_applied_level"] = _cleaning_applied_level(state)
        return _trace(state, "data_connector", "connect_and_profile",
                      "re-profile returned needs_review")

    # validate_quality as the actual exit check (附录 B.4 #1) — previously
    # wired up in mcp_client but never called from the graph at all.
    try:
        validation, val_log = await mcp_client.validate_quality(
            cleaned_path, run_id=_round_run_id("validate", state["iteration_count"]), artifact_root=artifact_root,
            intent=intent_payload,
        )
    except SubSystemHardFailure as exc:
        state["mcp_call_log"] = [*state.get("mcp_call_log", []), exc.log]
        state["error"] = f"validate_quality failed ({exc.error_class}): {exc}"
        state["current_phase"] = "error"
        return _trace(state, "data_connector", "connect_and_profile",
                      f"validate_quality hard failure: {exc}")
    state["mcp_call_log"] = [*state.get("mcp_call_log", []), val_log]

    fingerprint_error = _check_scope_fingerprint(
        state, "validate_quality", expected_fingerprint, validation.scope_fingerprint,
        val_log.get("mode", "mcp"),
    )
    if fingerprint_error:
        state["error"] = fingerprint_error
        state["current_phase"] = "error"
        return _trace(state, "data_connector", "connect_and_profile", fingerprint_error)
    if validation.critical_basis:
        _trace(
            state, "data_connector", "critical_basis",
            f"validate_quality: passed={validation.passed}, critical_basis={validation.critical_basis}",
        )

    if validation.needs_review:
        state["cleaning_stop_reason"] = "needs_review"
        state["cleaning_caveat"] = (
            "Data Cleaner flagged the cleaned data for manual review during "
            "validation; stopping automatic cleaning."
        )
        state["cleaning_applied_level"] = _cleaning_applied_level(state)
        return _trace(state, "data_connector", "connect_and_profile",
                      "validate_quality returned needs_review")

    if validation.passed:
        state["cleaning_stop_reason"] = "passed"
        state["cleaning_applied_level"] = _cleaning_applied_level(state)
        return _trace(state, "data_connector", "connect_and_profile",
                      f"validate_quality passed after {state['iteration_count']} round(s)")

    # Not passed — check for stall (附录 B.4 #3) or regression (#4) against
    # the previous round before letting the loop continue.
    signature = _cleaning_signature(validation)
    prev_signature = state.get("cleaning_last_signature")
    prev_score = state.get("cleaning_last_score")

    if prev_signature is not None and signature == prev_signature:
        state["cleaning_stop_reason"] = "no_improvement"
        state["cleaning_caveat"] = (
            "Cleaning round produced the same quality verdict as the "
            "previous round (missing/duplicate/schema dimensions unchanged); "
            "stopping to avoid an unproductive loop. Proceeding to analysis "
            "with the best available cleaned data."
        )
    elif prev_score is not None and validation.score < prev_score:
        state["cleaning_stop_reason"] = "regressed"
        state["cleaning_caveat"] = (
            f"Cleaning round made quality worse (score {prev_score:.2f} -> "
            f"{validation.score:.2f}); stopping and flagging for manual "
            "review. Proceeding to analysis with the best available "
            "cleaned data."
        )
    elif state["iteration_count"] >= _MAX_CLEAN_ITERATIONS:
        state["cleaning_stop_reason"] = "max_rounds"
        state["cleaning_caveat"] = (
            f"Reached the maximum of {_MAX_CLEAN_ITERATIONS} automatic cleaning "
            "rounds without the data passing quality validation; proceeding "
            "to analysis on the best available (partially cleaned) version."
        )
    # else: still making progress and rounds remain -- leave
    # cleaning_stop_reason unset so route_after_profiling loops back.

    state["cleaning_last_signature"] = signature
    state["cleaning_last_score"] = validation.score
    state["cleaning_applied_level"] = _cleaning_applied_level(state)

    return _trace(
        state, "data_connector", "connect_and_profile",
        f"round {state['iteration_count']} complete; validate_quality.passed=False, "
        f"score={validation.score}, stop_reason={state.get('cleaning_stop_reason')}",
    )


async def connect_and_profile_node(state: MAEDAState) -> MAEDAState:
    """
    Back-compat single-call entry point (E2 BO.1 split, submission 1 —
    ECOSYSTEM_INTEGRATION_PLAN.md 附录 BQ): runs `connect_schema` then
    `profile_and_clean` in sequence, reproducing the pre-split monolithic
    node's exact behavior for callers that drive one full round (schema
    extraction through profile/clean/validate) with a single call —
    notably the existing direct-node unit tests
    (tests/unit/test_tb3_tb4.py, tests/unit/test_phase4.py's
    TestCleaningLoop) that thread `state` through repeated calls to
    simulate the graph's multi-round self-loop by hand, and never went
    through `src.graph.builder`'s actual node wiring in the first place.

    The compiled graph itself (src/graph/builder.py) does NOT use this
    function — it registers `connect_schema` and `profile_and_clean` as
    two separate nodes joined by a plain edge, so a `refine_intent` node
    (E2 submission 2) can be inserted as a real conditional edge between
    them without another rewrite of this wrapper.
    """
    state = await connect_schema(state)
    return await profile_and_clean(state)


async def plan_analysis_node(state: MAEDAState) -> MAEDAState:
    """Phase 5: LLM generates AnalysisPlan from parsed intent + schema."""
    logger.info("Node: plan_analysis")
    state["current_phase"] = "plan"
    return await _get_analysis_agent().plan(state)


async def execute_analysis_node(state: MAEDAState) -> MAEDAState:
    """Phase 5: execute plan steps with dependency tracking and error recovery."""
    logger.info("Node: execute_analysis")
    state["current_phase"] = "execute"
    return await _get_analysis_agent().execute(state)


async def generate_viz_node(state: MAEDAState) -> MAEDAState:
    """Phase 6: recommend charts, generate static/interactive, caption, dashboard."""
    logger.info("Node: generate_viz")
    state["current_phase"] = "synthesize"
    return await _get_viz_agent().process(state)


async def retrieve_knowledge_node(state: MAEDAState) -> MAEDAState:
    """Phase 7: build focused retrieval query, delegate to RAG-MCP-Server."""
    logger.info("Node: retrieve_domain_knowledge")

    client = _get_subsystem_client()
    insight_agent = _get_insight_agent()
    # 7.1 Build focused retrieval query from analysis results + intent
    query = insight_agent.build_retrieval_query(state)

    try:
        chunks, log = await client.retrieve_knowledge(
            query, top_k=5, collection=settings.rag_collection
        )
    except SubSystemHardFailure as exc:
        # RAG is enrichment-only (CLAUDE.md: "MAEDA must be able to run
        # standalone"), unlike the cleaner's profiling call which is on the
        # critical path -- degrade to no domain context instead of aborting
        # analysis that has already been computed, even in strict mode.
        # The failure is still fully logged, just not treated as terminal.
        state["mcp_call_log"] = [*state.get("mcp_call_log", []), exc.log]
        state["rag_context"] = []
        state["rag_sources"] = []
        return _trace(state, "insight_agent", "retrieve_knowledge",
                      f"RAG hard failure ({exc.error_class}), continuing without domain context: {exc}")
    state["mcp_call_log"] = [*state.get("mcp_call_log", []), log]
    state["rag_context"] = [c.to_dict() for c in chunks]
    state["rag_sources"] = [
        {"source_file": c.source_file, "page": c.page, "chunk_id": c.chunk_id}
        for c in chunks if c.source_file
    ]
    return _trace(state, "insight_agent", "retrieve_knowledge",
                  f"Query: {query[:80]!r} → {len(state['rag_context'])} chunks")


async def generate_insights_node(state: MAEDAState) -> MAEDAState:
    """Phase 7: combine analysis results + RAG context → insights + report."""
    logger.info("Node: generate_insights")
    state["current_phase"] = "synthesize"
    return await _get_insight_agent().generate(state)


async def run_guardrails_node(state: MAEDAState) -> MAEDAState:
    """Phase 8: run all guardrail checks on outputs before user delivery."""
    logger.info("Node: run_guardrails")
    state["current_phase"] = "guardrail"
    # Increment guardrail_retry_count so route_after_guardrails can cap retry loops
    state["guardrail_retry_count"] = state.get("guardrail_retry_count", 0) + 1
    return await _get_guardrail_agent().process(state)


async def run_eval_node(state: MAEDAState) -> MAEDAState:
    """Phase 9: score the completed pipeline run against all eval metrics."""
    logger.info("Node: run_eval")
    state["current_phase"] = "complete"

    runner = _get_eval_runner()
    result = await runner.score(state, run_id=state.get("run_id"))
    state["eval_scores"] = {
        s.metric: {"score": s.score, "label": s.label, "reasoning": s.reasoning, "valid": s.valid}
        for s in result.scores
    }
    state["eval_scores"]["_aggregate"] = result.aggregate_score
    return state


def handle_error_node(state: MAEDAState) -> MAEDAState:
    checks = state.get("guardrail_checks") or []
    # Reaching handle_error via route_after_guardrails' "fail" branch means the
    # guardrail correctly blocked an unsafe/ungrounded output after exhausting
    # retries — a safe refusal, not a system crash. Any other path here (e.g.
    # no data source, connection failure) is a genuine pipeline error. This
    # distinction is what eval's error_rate/safe_refusal metrics key off of.
    is_safe_refusal = bool(checks) and checks[-1].get("overall_verdict") == "fail"
    state["error_type"] = "safe_refusal" if is_safe_refusal else "pipeline_error"

    if not state.get("error"):
        if is_safe_refusal:
            reason = checks[-1].get("retry_reason") or "Guardrail checks failed after maximum retries"
            state["error"] = reason
        else:
            state["error"] = "Pipeline terminated due to unrecoverable error"

    logger.error("Node: handle_error | error_type=%s | error=%s", state["error_type"], state.get("error"))
    state["current_phase"] = "error"
    state = _trace(state, "orchestrator", "handle_error",
                    f"{state['error_type']}: {state.get('error')}")
    return state


def persist_run_node(state: MAEDAState) -> MAEDAState:
    """
    Terminal node on every path (both run_eval and handle_error route here
    before END) — persists decision_trace/mcp_call_log to SQLite so they
    survive past this process, instead of vanishing the moment the graph
    finishes. See src/persistence/run_store.py.

    Persistence failures must never break the pipeline the user is
    actually waiting on: caught and logged, not raised.
    """
    try:
        _get_run_store().save_run(state)
    except Exception as exc:
        logger.warning("Failed to persist run %s: %s", state.get("run_id"), exc)
    return state
