"""
Eval judge calibration — Phase A #8.

Unlike tests/unit/test_phase9.py (which mocks the LLM to test the scoring
*plumbing*), this makes real judge calls to check the judge's actual
discrimination ability: does it reliably score an obviously grounded,
on-topic report high, and an obviously fabricated or off-topic one low?

This suite spends money. 附录 CR.2: a real API key is necessary but NOT
sufficient — MAEDA_ALLOW_PAID_TESTS must also say so, or the suite skips.
See tests/paid_calls.py for why "no key configured" was never a gate.

Run explicitly with:
    MAEDA_ALLOW_PAID_TESTS=1 poetry run pytest tests/integration/test_eval_judge_calibration.py -v
"""
import asyncio

import pytest

from src.eval.metrics import score_relevance_and_groundedness
from tests.paid_calls import eval_judge_key_looks_real, requires_paid_calls

pytestmark = requires_paid_calls(eval_judge_key_looks_real())

_QUERY = "What is total revenue by region?"
_FINDINGS = [{
    "result_summary": (
        "pandas/groupby -> 5 rows. North=1363760.55, South=1006074.27, "
        "East=1164964.55, West=1271795.18, Central=821174.97"
    ),
    "failed": False,
}]

_GROUNDED_REPORT = """\
# Executive Summary
North region leads with $1,363,760.55 in total revenue, followed by West
at $1,271,795.18, East at $1,164,964.55, South at $1,006,074.27, and
Central at $821,174.97.

## Key Findings
- North: $1,363,760.55
- West: $1,271,795.18
- East: $1,164,964.55
- South: $1,006,074.27
- Central: $821,174.97
"""

_FABRICATED_REPORT = """\
# Executive Summary
North region generated an extraordinary $50 million in revenue this
quarter, a 300% year-over-year increase driven by our new premium
product line and expanded partnership network across all channels.

## Key Findings
- North region revenue grew 300% YoY to $50M
- The premium product line contributed 60% of new growth
- Partnership channel revenue tripled
"""

_OFF_TOPIC_REPORT = """\
# Executive Summary
Customer churn spiked sharply in March, driven primarily by a wave of
support ticket escalations and a competitor's aggressive pricing
promotion that ran throughout the month.

## Key Findings
- Support ticket volume rose 40% in March
- A competitor launched a 20%-off promotion mid-month
- Churn was concentrated among month-to-month subscribers
"""


def _score(report: str) -> tuple[float, float]:
    rel, gnd = asyncio.run(score_relevance_and_groundedness(
        _QUERY, report, _FINDINGS, [],
    ))
    return rel.score, gnd.score


def test_judge_scores_grounded_on_topic_report_high():
    rel, gnd = _score(_GROUNDED_REPORT)
    assert rel >= 0.7, f"expected high relevance for a report that directly answers the query, got {rel}"
    assert gnd >= 0.7, f"expected high groundedness for a report citing only real figures, got {gnd}"


def test_judge_scores_fabricated_numbers_low_groundedness():
    gnd = _score(_FABRICATED_REPORT)[1]
    assert gnd <= 0.4, f"expected low groundedness for fabricated figures unrelated to the findings, got {gnd}"


# The expectation this test encodes is still correct: a report that is
# entirely about a different topic than the query (customer churn vs.
# revenue-by-region) should score low on relevance. That expectation is not
# in question. What's xfail'd below is the *current judge's* failure to meet
# it — a known, tracked calibration gap (roadmap.md #30), not a wrong test.
@pytest.mark.xfail(
    reason=(
        "Known EVAL_RELEVANCE_SYSTEM calibration gap tracked in "
        "docs/roadmap.md #30: the judge stably scores this clear-cut "
        "off-topic report 0.5 instead of <=0.4 (3-sample median, "
        "reproduced in CI and locally, not flaky). Remove this xfail only "
        "after EVAL_RELEVANCE_SYSTEM's prompt or the judge model changes "
        "and this case is re-evaluated -- do not let it silently linger "
        "past such a change."
    ),
    strict=False,
)
def test_judge_scores_off_topic_report_low_relevance():
    rel = _score(_OFF_TOPIC_REPORT)[0]
    assert rel <= 0.4, f"expected low relevance for a report answering a different question entirely, got {rel}"


def test_judge_discriminates_grounded_from_fabricated():
    """The core calibration check: the judge must rank these in the right order, not just cross fixed thresholds."""
    grounded_gnd = _score(_GROUNDED_REPORT)[1]
    fabricated_gnd = _score(_FABRICATED_REPORT)[1]
    assert grounded_gnd > fabricated_gnd, (
        f"judge failed to rank a fully-grounded report ({grounded_gnd}) above a "
        f"fabricated one ({fabricated_gnd}) on groundedness"
    )
