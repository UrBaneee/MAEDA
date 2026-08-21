"""
Business glossary (口径词表) — ECOSYSTEM_INTEGRATION_PLAN.md 阶段 3, 执行顺序表
轮次 4 (附录 CQ).

Three things are being pinned here, in rough order of how expensive they'd be
to get wrong:

1. `absent` is never expressed by omission. It reaches state, the cleaner
   payload and the planner prompt as a stated fact with an explicit
   anti-fabrication instruction. "Nothing was injected" and "we checked, and
   nobody has written a definition" must not look the same downstream — the
   recurring failure shape behind 附录 CH.2/CI.2/CK.3/CO.
2. Entries are reconciled against the live schema before injection, and
   coverage is judged in exactly ONE place (`resolve_glossary`), whose stored
   result every consumer reads. No second gate.
3. The shipped config/glossary.yaml describes columns that actually exist in
   data/demo/*, so the content can't rot into fiction unnoticed.
"""
from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.state.graph_state import initial_state
from src.tools import glossary as g

REPO = Path(__file__).resolve().parents[2]
SHIPPED = REPO / "config" / "glossary.yaml"


def _col(name: str, is_datetime: bool = False):
    class _C:
        pass
    c = _C()
    c.name = name
    c.is_datetime = is_datetime
    return c


@pytest.fixture
def tmp_glossary(tmp_path):
    """Write a small glossary and return its path. Deliberately not the shipped
    file: behaviour tests must not move when curated content is edited."""
    def _write(doc: dict) -> str:
        path = tmp_path / "glossary.yaml"
        path.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
        return str(path)
    return _write


_BASIC = {
    "glossary_version": "1",
    "datasets": [
        {
            "dataset": "sales.csv",
            "columns": [
                {
                    "column": "revenue",
                    "aliases": ["营收", "Turnover"],
                    "meaning": "Net revenue on the line.",
                    "unit": "currency units",
                    "notes": "No currency declared.",
                    "value_semantics": [{"value": 0, "means": "no sale", "action": "keep"}],
                },
                {"column": "region", "aliases": ["地区"], "meaning": "Sales region."},
            ],
        },
        {
            "dataset": "orders.db",
            "table": "orders",
            "columns": [{"column": "order_id", "meaning": "Order line id."}],
        },
    ],
}


# ─── The single gate: schema reconciliation + three-state coverage ────────────

class TestCoverageStates:
    def test_full_when_every_schema_column_is_curated(self, tmp_glossary):
        m = g.resolve_glossary("/data/sales.csv", [_col("revenue"), _col("region")],
                               path=tmp_glossary(_BASIC))
        assert m.coverage == "full"
        assert m.covered_columns == ["revenue", "region"]
        assert m.uncovered_columns == []

    def test_partial_is_a_real_state_not_rounded_to_full_or_absent(self, tmp_glossary):
        """轮次 4 explicitly keeps three states. A schema with one curated and
        one uncurated column is neither "covered" nor "unknown", and collapsing
        it either way loses the only information the planner can act on."""
        m = g.resolve_glossary("/data/sales.csv", [_col("revenue"), _col("margin")],
                               path=tmp_glossary(_BASIC))
        assert m.coverage == "partial"
        assert m.covered_columns == ["revenue"]
        assert m.uncovered_columns == ["margin"]

    def test_absent_for_an_unknown_data_source(self, tmp_glossary):
        m = g.resolve_glossary("/data/mystery.csv", [_col("a"), _col("b")],
                               path=tmp_glossary(_BASIC))
        assert m.coverage == "absent"
        assert m.entries == []
        assert m.uncovered_columns == ["a", "b"]
        assert "mystery.csv" in (m.reason or "")

    def test_absent_when_schema_extraction_failed(self, tmp_glossary):
        """connect_schema runs the gate even when the connector raised. No
        schema means no verified definitions -- that has to read as 'absent',
        not as an unset field a consumer can interpret as 'not applicable'."""
        m = g.resolve_glossary("/data/sales.csv", None, path=tmp_glossary(_BASIC))
        assert m.coverage == "absent"
        assert "no schema columns" in (m.reason or "")

    def test_missing_glossary_file_degrades_to_absent_and_does_not_raise(self, tmp_path):
        m = g.resolve_glossary("/data/sales.csv", [_col("revenue")],
                               path=str(tmp_path / "not_written.yaml"))
        assert m.coverage == "absent"

    def test_malformed_glossary_file_degrades_to_absent_and_does_not_raise(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("datasets: [oops\n", encoding="utf-8")
        m = g.resolve_glossary("/data/sales.csv", [_col("revenue")], path=str(bad))
        assert m.coverage == "absent"


class TestSchemaReconciliation:
    def test_entry_for_a_column_the_data_does_not_have_is_dropped_not_injected(self, tmp_glossary):
        """轮次 4: 词表条目要与真实列名对账后才注入. A stale definition handed to
        an LLM is worse than no definition -- it reads as authoritative."""
        m = g.resolve_glossary("/data/sales.csv", [_col("revenue")], path=tmp_glossary(_BASIC))
        assert [e["column"] for e in m.entries] == ["revenue"]
        assert m.dropped_columns == ["region"]      # described, but not in this schema
        assert m.coverage == "full"                 # every column PRESENT is curated

    def test_glossary_that_matches_nothing_in_the_schema_is_absent_not_full(self, tmp_glossary):
        m = g.resolve_glossary("/data/sales.csv", [_col("qty")], path=tmp_glossary(_BASIC))
        assert m.coverage == "absent"
        assert m.entries == []
        assert m.dropped_columns == ["region", "revenue"]

    def test_sql_block_applies_only_to_its_own_table(self, tmp_glossary):
        path = tmp_glossary(_BASIC)
        assert g.resolve_glossary("/d/orders.db", [_col("order_id")], "orders", path=path).coverage == "full"
        # Same file, different table: `customers` must not inherit `orders`'s
        # 口径 just because they live in one database.
        other = g.resolve_glossary("/d/orders.db", [_col("order_id")], "customers", path=path)
        assert other.coverage == "absent"

    def test_dataset_matching_is_by_basename_so_a_moved_file_still_matches(self, tmp_glossary):
        path = tmp_glossary(_BASIC)
        assert g.resolve_glossary("/elsewhere/sales.csv", [_col("revenue")], path=path).coverage == "full"
        assert g.resolve_glossary("/d/sales_copy.csv", [_col("revenue")], path=path).coverage == "absent"


# ─── Injection [2]: the cleaner payload projection ────────────────────────────

class TestPayloadProjection:
    def test_projects_onto_frozen_u2_keys_only_and_drops_aliases(self, tmp_glossary):
        m = g.resolve_glossary("/data/sales.csv", [_col("revenue"), _col("region")],
                               path=tmp_glossary(_BASIC))
        entries = g.payload_entries(m.entries)
        assert entries[0] == {
            "column": "revenue",
            "meaning": "Net revenue on the line.",
            "unit": "currency units",
            "value_semantics": [{"value": 0, "means": "no sale", "action": "keep"}],
            "notes": "No currency declared.",
        }
        # `aliases` is MAEDA-internal (附录 P.2: cleaner only gets fields it can
        # act on) and must never reach the wire.
        assert all("aliases" not in e for e in entries)

    def test_empty_optional_fields_are_omitted_not_sent_as_blanks(self, tmp_glossary):
        m = g.resolve_glossary("/d/orders.db", [_col("order_id")], "orders",
                               path=tmp_glossary(_BASIC))
        assert g.payload_entries(m.entries) == [
            {"column": "order_id", "meaning": "Order line id."},
        ]

    def test_payload_carries_glossary_and_coverage_together(self, tmp_glossary):
        from src.graph.nodes import _build_intent_payload
        m = g.resolve_glossary("/data/sales.csv", [_col("revenue"), _col("margin")],
                               path=tmp_glossary(_BASIC))
        payload = _build_intent_payload("/data/sales.csv", {}, [], [], "absent",
                                        glossary_entries=m.entries, glossary_coverage=m.coverage)
        assert payload["glossary_coverage"] == "partial"
        assert [e["column"] for e in payload["glossary"]] == ["revenue"]

    def test_coverage_key_is_present_even_when_nothing_is_curated(self):
        """The whole point of the additive key: an empty `glossary` alone can't
        distinguish "no definitions exist" from "not attached"."""
        from src.graph.nodes import _build_intent_payload
        payload = _build_intent_payload("/x.csv", {}, [], [], "absent")
        assert payload["glossary"] == []
        assert payload["glossary_coverage"] == "absent"


# ─── Injection [3]: the plan_analysis prompt block ────────────────────────────

class TestPromptBlock:
    def test_absent_block_is_present_and_carries_the_anti_fabrication_rule(self):
        block = g.prompt_block("absent", [], [], source_label="/data/mystery.csv")
        assert block.strip()                       # never empty -- omission IS the bug
        assert "coverage: absent" in block
        assert "/data/mystery.csv" in block
        assert "Do not invent" in block

    def test_partial_block_names_the_uncurated_columns(self, tmp_glossary):
        m = g.resolve_glossary("/data/sales.csv", [_col("revenue"), _col("margin")],
                               path=tmp_glossary(_BASIC))
        block = g.prompt_block(m.coverage, m.entries, m.uncovered_columns)
        assert "coverage: partial" in block
        assert "No curated definition: margin" in block
        assert "Do not invent" in block
        assert "Net revenue on the line." in block

    def test_full_block_renders_units_notes_and_value_semantics(self, tmp_glossary):
        m = g.resolve_glossary("/data/sales.csv", [_col("revenue"), _col("region")],
                               path=tmp_glossary(_BASIC))
        block = g.prompt_block(m.coverage, m.entries, m.uncovered_columns)
        assert "coverage: full" in block
        assert "unit: currency units" in block
        assert "note: No currency declared." in block
        assert "action: keep" in block

    def test_none_coverage_is_treated_as_absent_not_as_a_blank_section(self):
        """A consumer that runs before connect_schema (or a state built by hand)
        must still get the anti-fabrication instruction, not silence."""
        block = g.prompt_block(None, None, None)
        assert "coverage: absent" in block
        assert "Do not invent" in block


# ─── The 附录 U.2 `glossary_alias` match tier ─────────────────────────────────

class TestAliasTier:
    def test_alias_resolves_a_mention_that_no_column_name_matches(self, tmp_glossary):
        from src.graph.nodes import _resolve_intent_columns
        m = g.resolve_glossary("/data/sales.csv", [_col("revenue")], path=tmp_glossary(_BASIC))
        resolved, unresolved, status = _resolve_intent_columns(
            {"target_metrics": ["营收"]}, [_col("revenue")], alias_index=g.alias_index(m.entries),
        )
        assert resolved == [{
            "name": "revenue", "role": "metric", "matched_from": "营收", "match": "glossary_alias",
        }]
        assert unresolved == [] and status == "full"

    def test_alias_match_is_case_insensitive(self, tmp_glossary):
        from src.graph.nodes import _resolve_intent_columns
        m = g.resolve_glossary("/data/sales.csv", [_col("revenue")], path=tmp_glossary(_BASIC))
        resolved, _, _ = _resolve_intent_columns(
            {"target_metrics": ["turnover"]}, [_col("revenue")], alias_index=g.alias_index(m.entries),
        )
        assert resolved[0]["match"] == "glossary_alias"

    def test_a_real_column_name_always_wins_over_an_alias(self):
        """Tier order (附录 U.2): a glossary may extend reach, never redirect a
        mention that already names a real column."""
        from src.graph.nodes import _resolve_intent_columns
        resolved, _, _ = _resolve_intent_columns(
            {"target_metrics": ["sales"]}, [_col("sales"), _col("revenue")],
            alias_index={"sales": ["revenue"]},
        )
        assert resolved == [{
            "name": "sales", "role": "metric", "matched_from": "sales", "match": "exact",
        }]

    def test_alias_claimed_by_two_columns_is_ambiguous_not_an_arbitrary_pick(self):
        from src.graph.nodes import _resolve_intent_columns
        resolved, unresolved, status = _resolve_intent_columns(
            {"target_metrics": ["收入"]}, [_col("revenue"), _col("net_revenue")],
            alias_index={"收入": ["net_revenue", "revenue"]},
        )
        assert resolved == []
        assert unresolved == [{
            "text": "收入", "role": "metric", "reason": "ambiguous",
            "candidates": ["net_revenue", "revenue"],
        }]
        assert status == "absent"

    def test_without_a_glossary_the_tier_stays_dead_exactly_as_before(self):
        """附录 S.2's pre-轮次-4 behaviour must be unchanged when no glossary is
        supplied -- the alias tier is additive, not a change to the two tiers
        the frozen contract already had."""
        from src.graph.nodes import _resolve_intent_columns
        _, unresolved, status = _resolve_intent_columns({"target_metrics": ["营收"]}, [_col("revenue")])
        assert unresolved == [{"text": "营收", "role": "metric", "reason": "no_match", "candidates": []}]
        assert status == "absent"

    def test_alias_index_only_covers_schema_filtered_entries(self, tmp_glossary):
        """`region` is in the glossary but not in this schema, so its alias must
        not be resolvable -- otherwise the tier could name a missing column."""
        m = g.resolve_glossary("/data/sales.csv", [_col("revenue")], path=tmp_glossary(_BASIC))
        assert "地区" not in g.alias_index(m.entries)


# ─── One gate, several consumers (no second judgment site) ────────────────────

class TestSingleGateWiring:
    @staticmethod
    def _offline_connector(monkeypatch):
        """connect_schema builds its DataConnector through a module singleton,
        and DataConnector's default LLM is a REAL client (src/tools/
        data_connector.py::_build_llm) -- an un-injected connector in a unit
        test makes a real, billed schema-summary call. Inject a stub instead."""
        from unittest.mock import AsyncMock, MagicMock

        import src.graph.nodes as nodes
        from src.tools.data_connector import DataConnector

        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(content="stub summary"))
        monkeypatch.setattr(nodes, "_data_connector", DataConnector(llm=llm), raising=False)

    def test_connect_schema_writes_the_three_state_fields_and_traces_them(
        self, monkeypatch, tmp_glossary, tmp_path,
    ):
        import src.graph.nodes as nodes

        csv = tmp_path / "sales.csv"
        pd.DataFrame({"revenue": [1.0, 2.0], "margin": [0.1, 0.2]}).to_csv(csv, index=False)
        monkeypatch.setattr(g.settings, "glossary_path", tmp_glossary(_BASIC), raising=False)
        self._offline_connector(monkeypatch)

        state = initial_state("q", data_sources=[{"type": "csv", "path": str(csv)}])
        state = asyncio.run(nodes.connect_schema(state))

        assert state["glossary_coverage"] == "partial"
        assert [e["column"] for e in state["glossary_entries"]] == ["revenue"]
        assert state["glossary_uncovered_columns"] == ["margin"]
        traced = [t for t in state["decision_trace"] if t.get("action") == "glossary_coverage"]
        assert len(traced) == 1 and "partial" in str(traced[0])

    def test_unknown_data_source_reaches_connect_schema_state_as_absent(
        self, monkeypatch, tmp_glossary, tmp_path,
    ):
        import src.graph.nodes as nodes

        csv = tmp_path / "unheard_of.csv"
        pd.DataFrame({"a": [1], "b": [2]}).to_csv(csv, index=False)
        monkeypatch.setattr(g.settings, "glossary_path", tmp_glossary(_BASIC), raising=False)
        self._offline_connector(monkeypatch)

        state = initial_state("q", data_sources=[{"type": "csv", "path": str(csv)}])
        state = asyncio.run(nodes.connect_schema(state))
        assert state["glossary_coverage"] == "absent"
        assert state["glossary_entries"] == []

    def test_plan_prompt_uses_the_state_the_gate_wrote(self, monkeypatch, tmp_glossary):
        """Injection [3] reads connect_schema's stored result. If it re-filtered
        on its own, [2] and [3] could disagree about what is curated."""
        from src.agents.analysis_agent import AnalysisAgent

        captured = {}

        class _LLM:
            async def ainvoke(self, messages):
                captured["prompt"] = messages[-1].content
                class _R:
                    content = '{"steps": []}'
                    usage_metadata = {}
                return _R()

        m = g.resolve_glossary("/data/sales.csv", [_col("revenue"), _col("margin")],
                               path=tmp_glossary(_BASIC))
        state = initial_state("q")
        state["glossary_coverage"] = m.coverage
        state["glossary_entries"] = m.entries
        state["glossary_uncovered_columns"] = m.uncovered_columns
        asyncio.run(AnalysisAgent(llm=_LLM()).plan(state))

        assert "### Column Glossary" in captured["prompt"]
        assert "coverage: partial" in captured["prompt"]
        assert "No curated definition: margin" in captured["prompt"]

    def test_plan_prompt_states_absent_when_nothing_is_curated(self):
        from src.agents.analysis_agent import AnalysisAgent

        captured = {}

        class _LLM:
            async def ainvoke(self, messages):
                captured["prompt"] = messages[-1].content
                class _R:
                    content = '{"steps": []}'
                    usage_metadata = {}
                return _R()

        asyncio.run(AnalysisAgent(llm=_LLM()).plan(initial_state("q")))
        assert "coverage: absent" in captured["prompt"]
        assert "Do not invent" in captured["prompt"]

    def test_refine_router_precheck_sees_the_same_aliases_as_the_real_resolution(
        self, monkeypatch, tmp_glossary,
    ):
        """route_after_schema's `if_unresolved` pre-check must use the same alias
        index profile_and_clean will use; a stricter pre-check would send runs
        to refine_intent over mentions the glossary already resolves."""
        from src.config.settings import settings as s
        from src.graph.router import route_after_schema

        monkeypatch.setattr(s, "intent_refine_trigger", "if_unresolved", raising=False)
        m = g.resolve_glossary("/data/sales.csv", [_col("revenue")], path=tmp_glossary(_BASIC))
        state = initial_state("q")
        state["schema_columns"] = [_col("revenue")]
        state["parsed_intent"] = {"target_metrics": ["营收"]}

        state["glossary_entries"] = []
        assert route_after_schema(state) == "refine"     # no glossary: unresolved
        state["glossary_entries"] = m.entries
        assert route_after_schema(state) == "profile"    # alias resolves it


# ─── The shipped content is real ──────────────────────────────────────────────

class TestShippedGlossaryContent:
    """Content guard. 轮次 4's rule was 宁可少而准: every curated column must
    exist in the dataset it claims to describe. Without this, the glossary can
    drift into confident fiction the moment data/demo/* is re-rendered -- the
    same class of silent staleness as 附录 BC.1's fixtures."""

    def _blocks(self):
        return yaml.safe_load(SHIPPED.read_text(encoding="utf-8"))["datasets"]

    def test_file_parses_and_declares_a_version(self):
        doc = yaml.safe_load(SHIPPED.read_text(encoding="utf-8"))
        assert doc["glossary_version"] == "1"
        assert doc["datasets"]

    @pytest.mark.parametrize("dataset", [
        "sales_data.csv", "churn_data.csv", "marketing_campaigns.csv", "ecommerce_orders.db",
    ])
    def test_every_curated_column_exists_in_the_real_dataset(self, dataset):
        block = next(b for b in self._blocks() if b["dataset"] == dataset)
        path = REPO / "data" / "demo" / dataset
        if not path.exists():                       # demo data not rendered in this checkout
            pytest.skip(f"{path} not present")
        if dataset.endswith(".csv"):
            real = set(pd.read_csv(path, nrows=1).columns)
        else:
            conn = sqlite3.connect(path)
            real = {r[1] for r in conn.execute(f"PRAGMA table_info({block['table']})")}
            conn.close()
        curated = {e["column"] for e in block["columns"]}
        assert curated <= real, f"glossary describes columns {curated - real} that {dataset} does not have"

    def test_no_alias_collides_within_a_dataset(self):
        for block in self._blocks():
            seen: dict[str, str] = {}
            for entry in block["columns"]:
                for alias in entry.get("aliases") or []:
                    key = alias.strip().lower()
                    assert key not in seen, (
                        f"{block['dataset']}: alias {alias!r} claimed by both "
                        f"{seen.get(key)} and {entry['column']}"
                    )
                    seen[key] = entry["column"]

    def test_value_semantics_actions_are_frozen_values(self):
        """附录 U.2: action ∈ {exclude, keep}. Only `exclude` has behaviour for
        the cleaner, so a typo here would silently mean nothing."""
        for block in self._blocks():
            for entry in block["columns"]:
                for semantic in entry.get("value_semantics") or []:
                    assert semantic["action"] in ("exclude", "keep")
                    assert semantic.get("means")

    def test_shipped_file_resolves_to_full_coverage_on_the_real_sales_data(self):
        path = REPO / "data" / "demo" / "sales_data.csv"
        if not path.exists():
            pytest.skip(f"{path} not present")
        columns = [_col(c) for c in pd.read_csv(path, nrows=1).columns]
        m = g.resolve_glossary(str(path), columns, path=str(SHIPPED))
        assert m.coverage == "full"
        assert m.dropped_columns == []
