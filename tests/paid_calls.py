"""The intent gate for tests that make real, billable API calls (附录 CR.2).

附录 CQ.0 recorded the incident this exists to prevent: a `pytest
tests/integration` run made ~12 real judge calls because the suite's only
protection was ``skipif(no real API key)``. On a machine with a key
configured -- which is every machine the suite is actually useful on --
that is not a gate at all: the safe state depended on something happening
to be *absent*, not on anyone having *declared an intent*.

CI already had the other half of this discipline (附录 BV.4/BW made the
`integration-tests` job `workflow_dispatch`-only, so nothing bills without
someone pressing a button). This module is the local half, and the rule is
the same one 附录 CK.6 states for the split guard: **safe by default;
spending money requires saying so.**

Usage -- one line at the top of any suite that can bill::

    from tests.paid_calls import requires_paid_calls
    pytestmark = requires_paid_calls(eval_judge_key_looks_real())

and to actually run it::

    MAEDA_ALLOW_PAID_TESTS=1 poetry run pytest tests/integration -v

Two deliberate properties:

* **The opt-in is read from the process environment only, never from
  ``.env``.** ``src/config/settings.py`` loads ``.env`` for its own fields,
  and a standing ``MAEDA_ALLOW_PAID_TESTS=1`` line in a file nobody
  re-reads is an ambient, invisible permission that outlives the run it was
  meant for -- the failure mode 附录 CZ.6(2) names on the rag side and 附录
  DB.3 forbids for `RAG_TEST_SPLIT_REVEAL`. Same reasoning, same answer:
  per-invocation only.
* **The opt-in alone is not sufficient**, and neither is the key alone.
  Both are required, so opting in on a keyless machine still skips rather
  than failing with an auth error.
"""
from __future__ import annotations

import os
from typing import Mapping, Optional

import pytest

from src.config.settings import _looks_like_real_key, settings

#: Environment variable that says "I know this run costs money; do it."
ALLOW_PAID_TESTS_ENV = "MAEDA_ALLOW_PAID_TESTS"

#: Accepted affirmative values. Anything else -- including unset, "", "0"
#: and "false" -- means not opted in.
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def paid_tests_opted_in(env: Optional[Mapping[str, str]] = None) -> bool:
    """True only if the process environment carries an affirmative opt-in."""
    raw = (os.environ if env is None else env).get(ALLOW_PAID_TESTS_ENV, "")
    return raw.strip().casefold() in _TRUTHY


def eval_judge_key_looks_real() -> bool:
    """Whether a real key exists for the provider the eval judge resolves to."""
    key = (
        settings.anthropic_api_key
        if settings.resolved_eval_provider == "anthropic"
        else settings.openai_api_key
    )
    return _looks_like_real_key(key)


def paid_test_skip_reason(*, has_real_key: bool, opted_in: bool) -> Optional[str]:
    """The reason to skip a billable suite, or ``None`` if it may run.

    Pure function of the two conditions so the truth table is testable
    without an API key, an environment, or a subprocess.
    """
    unmet: list[str] = []
    if not opted_in:
        unmet.append(f"{ALLOW_PAID_TESTS_ENV} is not set to an affirmative value")
    if not has_real_key:
        unmet.append("no real API key is configured for the resolved eval judge provider")
    if not unmet:
        return None
    return (
        "this suite makes real, billable API calls and "
        + " and ".join(unmet)
        + f" (附录 CR.2 — run it with {ALLOW_PAID_TESTS_ENV}=1)"
    )


def requires_paid_calls(has_real_key: bool) -> pytest.MarkDecorator:
    """Module-level mark for a suite that makes real, billable API calls."""
    reason = paid_test_skip_reason(
        has_real_key=has_real_key, opted_in=paid_tests_opted_in(),
    )
    return pytest.mark.skipif(reason is not None, reason=reason or "")
