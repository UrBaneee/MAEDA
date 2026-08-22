"""附录 CR.2 — the intent gate on tests that make real, billable API calls.

The incident (附录 CQ.0) was not "someone was careless": `pytest
tests/integration` was, on any machine with a key configured, a command
that charged money by default. The gate these tests lock down is
tests/paid_calls.py; the one that matters most is
`test_a_real_key_without_opt_in_still_skips`, because that is the exact
configuration that billed.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.paid_calls import (
    ALLOW_PAID_TESTS_ENV,
    paid_test_skip_reason,
    paid_tests_opted_in,
    requires_paid_calls,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INTEGRATION_DIR = _REPO_ROOT / "tests" / "integration"

# A key that passes _looks_like_real_key (it does not end in "...") but is
# not a credential. If the gate ever regresses, the suite reaches a real
# provider with this and gets a 401 -- a loud failure, not a charge.
_FAKE_REAL_LOOKING_KEY = "sk-ant-api03-NOT-A-REAL-KEY-0000000000"


# ── the truth table ──────────────────────────────────────────────────────

@pytest.mark.parametrize("has_real_key,opted_in,should_run", [
    (True, True, True),
    (True, False, False),   # 附录 CQ.0's configuration
    (False, True, False),
    (False, False, False),
])
def test_both_conditions_are_required(has_real_key, opted_in, should_run):
    reason = paid_test_skip_reason(has_real_key=has_real_key, opted_in=opted_in)
    assert (reason is None) is should_run


def test_the_skip_reason_names_what_is_missing():
    """A skip nobody can act on gets ignored; name the variable and the 附录."""
    reason = paid_test_skip_reason(has_real_key=True, opted_in=False)
    assert ALLOW_PAID_TESTS_ENV in reason
    assert "CR.2" in reason

    both = paid_test_skip_reason(has_real_key=False, opted_in=False)
    assert ALLOW_PAID_TESTS_ENV in both and "API key" in both


@pytest.mark.parametrize("raw,expected", [
    ("1", True), ("true", True), ("TRUE", True), ("yes", True), ("on", True),
    (" 1 ", True),
    ("0", False), ("false", False), ("no", False), ("", False),
    ("maybe", False), ("2", False),
])
def test_only_affirmative_values_opt_in(raw, expected):
    assert paid_tests_opted_in({ALLOW_PAID_TESTS_ENV: raw}) is expected


def test_an_absent_variable_is_not_an_opt_in():
    assert paid_tests_opted_in({}) is False


def test_the_opt_in_is_read_from_the_environment_not_dotenv():
    """附录 DB.3's discipline, applied locally.

    src/config/settings.py loads `.env` for its own fields. The opt-in
    deliberately does not go through Settings: a standing line in a file
    nobody re-reads is a permission that outlives the run it was meant for.
    """
    src = (_REPO_ROOT / "tests" / "paid_calls.py").read_text()
    tree = ast.parse(src)
    reads = {
        node.value.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute)
    }
    assert "settings" not in reads
    # The variable is looked up in os.environ, and nowhere else.
    assert "os.environ" in src
    assert "env_file" not in src and "load_dotenv" not in src


def test_the_mark_is_a_skipif_that_fires_when_the_key_is_missing(monkeypatch):
    monkeypatch.delenv(ALLOW_PAID_TESTS_ENV, raising=False)
    mark = requires_paid_calls(has_real_key=False)
    assert mark.name == "skipif"
    assert mark.args == (True,)


# ── the structural lock: no second, ungated billable suite ───────────────

def test_every_integration_module_is_gated_by_the_shared_helper():
    """A future paid suite must not re-invent its own weaker gate.

    This is the same shape as the split guard's "no second direct reader"
    test on the rag side: it does not stop anyone writing a billable test,
    it makes an ungated one a red test rather than a habit.
    """
    modules = [p for p in _INTEGRATION_DIR.glob("*.py") if p.name != "__init__.py"]
    assert modules, "expected at least one integration module to guard"
    for path in modules:
        tree = ast.parse(path.read_text())
        gates = [
            node.value for node in tree.body
            if isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == "pytestmark" for t in node.targets)
        ]
        assert len(gates) == 1, f"{path.name}: expected exactly one module-level pytestmark"
        call = gates[0]
        assert isinstance(call, ast.Call) and getattr(call.func, "id", None) == "requires_paid_calls", (
            f"{path.name}: pytestmark must come from tests.paid_calls.requires_paid_calls, "
            "not a locally written skipif"
        )


# ── the end-to-end proof ─────────────────────────────────────────────────

def _run_integration_suite(env_overrides: dict[str, str]) -> subprocess.CompletedProcess:
    env = {**os.environ, **env_overrides}
    return subprocess.run(
        [sys.executable, "-m", "pytest", "tests/integration", "-q",
         "-p", "no:cacheprovider", "--no-header"],
        cwd=_REPO_ROOT, env=env, capture_output=True, text=True, timeout=300,
    )


def test_a_real_key_without_opt_in_still_skips():
    """The exact 附录 CQ.0 configuration, run for real in a subprocess.

    Keys are overridden in the child's environment so a real credential on
    the developer's machine (or in `.env`, which Settings loads) can never
    reach a provider from here: env vars win over the dotenv file.
    """
    result = _run_integration_suite({
        "ANTHROPIC_API_KEY": _FAKE_REAL_LOOKING_KEY,
        "OPENAI_API_KEY": _FAKE_REAL_LOOKING_KEY,
        ALLOW_PAID_TESTS_ENV: "",
    })
    out = result.stdout + result.stderr
    assert result.returncode == 0, out
    assert "skipped" in out, out
    assert " passed" not in out, out
    assert "failed" not in out and "error" not in out.lower(), out


def test_opting_in_without_a_key_also_skips():
    """The opt-in is a declaration of intent, not a substitute for a key."""
    result = _run_integration_suite({
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "OPENAI_API_KEY": "sk-...",
        ALLOW_PAID_TESTS_ENV: "1",
    })
    out = result.stdout + result.stderr
    assert result.returncode == 0, out
    assert "skipped" in out, out
    assert " passed" not in out, out
