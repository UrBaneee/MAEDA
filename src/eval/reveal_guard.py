"""
Reveal guard (附录 HA.5 item 4, HB). `split: "test"` cases are the H1 holdout: no pipeline,
judge or smoke run may touch them before the one reveal. Every path that runs a case
through the graph calls `require_reveal_authorization`; it refuses a test-split case unless
MAEDA_REVEAL_AUTHORIZED=1 is set in the process environment by the reveal command itself
(NOT in .env or .env.example, same discipline as the existing test-split opt-ins, 附录 DB.3).
"""
from __future__ import annotations

import os
from typing import Iterable

REVEAL_ENV = "MAEDA_REVEAL_AUTHORIZED"


class RevealNotAuthorized(RuntimeError):
    pass


def reveal_authorized() -> bool:
    return os.environ.get(REVEAL_ENV) == "1"


def require_reveal_authorization(cases: Iterable) -> None:
    ids = sorted(getattr(c, "id", None) or c.get("id") for c in cases
                 if (getattr(c, "split", None) or (c.get("split") if isinstance(c, dict) else None)) == "test")
    if ids and not reveal_authorized():
        raise RevealNotAuthorized(
            f"REFUSING to run {len(ids)} test-split case(s) (first: {ids[:3]}): the holdout is revealed once, "
            f"by the predetermined reveal command, which sets {REVEAL_ENV}=1 in the process environment. "
            "Nothing was run and nothing was billed. No smoke test on test cases (附录 HA.5)."
        )
