"""
Recompute golden-suite ground truth directly from data/demo/* -- eval v2
Step 4 (docs/eval_v2_plan.md): "ground truth values only have a comment
pointing at 'the pandas computation in test_suite.json' -- not reproducible
or auditable. A script means ground truth can be recomputed whenever the
underlying data changes."

One function per case id, registered in COMPUTERS below, each a small,
readable pandas/sqlite computation mirroring exactly how that case's
numbers were derived. Cases with no checkable numeric ground truth
(predictive, data_mismatch, ambiguous -- see _KNOWN_UNANSWERABLE_CASES in
tests/unit/test_phase9.py) have no function here; they're skipped, not an
error.

Usage:
    poetry run python scripts/compute_ground_truth.py            # check mode (default): report mismatches, write nothing
    poetry run python scripts/compute_ground_truth.py --write     # overwrite ground_truth in test_suite.json with freshly computed values
    poetry run python scripts/compute_ground_truth.py --case D01 DG02  # only these cases

IMPORTANT -- this recompute must also audit `_note` prose, not just numeric
fields (附录 BI.2/BJ, 2026-08-17): a prior recompute (`52fd014`) updated
numeric ground_truth in place but left the hand-written `_note` strings
sitting next to them untouched, so three self-consistent records (DG08,
DG13, DG19) silently went self-contradictory -- the note kept citing the
number the field used to hold. This script now runs a `NOTE-AUDIT` pass
automatically on every case that has a `_note` (see `_audit_note` below);
read its output, not just the MISMATCH lines, before treating a `--write`
run as complete. It also proved DG09's five CTR fields were stale from the
very first render and had been sitting in the "matches" bucket the whole
time -- see the note on `_numbers_match` below for why the *comparison*
tolerance used by an earlier ad-hoc recompute pass (not this script) let
that slide.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

SUITE_PATH = Path("tests/eval/test_suite.json")
DEMO_DIR = Path("data/demo")

COMPUTERS: dict[str, Callable[["_Data"], dict]] = {}


def register(case_id: str):
    def deco(fn: Callable[["_Data"], dict]):
        COMPUTERS[case_id] = fn
        return fn
    return deco


class _Data:
    """Lazily loads and caches each demo source exactly once per run."""

    def __init__(self):
        self._sales = None
        self._churn = None
        self._mkt = None
        self._orders = None
        self._customers = None
        self._products = None

    @property
    def sales(self) -> pd.DataFrame:
        if self._sales is None:
            df = pd.read_csv(DEMO_DIR / "sales_data.csv")
            df["date"] = pd.to_datetime(df["date"])
            df["quarter"] = df["date"].dt.quarter
            df["year"] = df["date"].dt.year
            self._sales = df
        return self._sales

    @property
    def churn(self) -> pd.DataFrame:
        if self._churn is None:
            self._churn = pd.read_csv(DEMO_DIR / "churn_data.csv")
        return self._churn

    @property
    def mkt(self) -> pd.DataFrame:
        if self._mkt is None:
            df = pd.read_csv(DEMO_DIR / "marketing_campaigns.csv")
            df["conversion_rate"] = df["conversions"] / df["clicks"]
            df["ctr"] = df["clicks"] / df["impressions"]
            self._mkt = df
        return self._mkt

    def _db(self):
        return sqlite3.connect(DEMO_DIR / "ecommerce_orders.db")

    @property
    def orders(self) -> pd.DataFrame:
        if self._orders is None:
            df = pd.read_sql("SELECT * FROM orders", self._db())
            df["order_date"] = pd.to_datetime(df["order_date"])
            self._orders = df
        return self._orders

    @property
    def customers(self) -> pd.DataFrame:
        if self._customers is None:
            df = pd.read_sql("SELECT * FROM customers", self._db())
            df["joined_date"] = pd.to_datetime(df["joined_date"])
            self._customers = df
        return self._customers

    @property
    def products(self) -> pd.DataFrame:
        if self._products is None:
            df = pd.read_sql("SELECT * FROM products", self._db())
            df["margin"] = df["unit_price"] - df["cost"]
            self._products = df
        return self._products

    @property
    def orders_x_customers(self) -> pd.DataFrame:
        return self.orders.merge(self.customers, on="customer_id")

    @property
    def orders_x_products(self) -> pd.DataFrame:
        return self.orders.merge(self.products, on="product_id")


def _iqr_outlier_count(series: pd.Series) -> int:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((series < lo) | (series > hi)).sum())


# ─── _note prose auditing (附录 BJ) ─────────────────────────────────────────
#
# `52fd014` proved that comparing only the *numeric* ground_truth fields
# isn't enough: a hand-written `_note` string sitting next to those fields
# can cite one of them (or a comparison between two of them) in prose, and
# nothing checks whether that citation is still accurate after a
# recompute. A few cases (DG03, DG07, DG08, DG13, DG15) have fully
# hand-authored, multi-sentence narrative notes making a specific editorial
# judgment (e.g. DG13's "half the premise is true, half isn't") -- those
# can't be regenerated by this script without writing a natural-language
# generator, which would be over-engineering for what's fundamentally a
# content decision. Rejected approach: teaching the script to *rewrite*
# those notes. What IS tractable and is done here instead: extract the
# numeric literals a note cites and check whether each one is still
# consistent with this case's own freshly computed fields -- a "does the
# footnote still agree with the numbers" check, not a "is the prose still
# well-written" check. It's a check, not a comment: `main()` prints a
# NOTE-AUDIT line whenever a citation doesn't resolve, on every run, so it
# can't be silently skipped the way the `_note` step was silently skipped
# in `52fd014`.
#
# Known, accepted false positives, both left as-is deliberately rather than
# special-cased, since either fix would mean writing real NL parsing:
#   1. Notes that cite a *domain* bound rather than a *derived* one (E10
#      "min 0, max 0.20", E14 "1-10 units", E23 "1-60 months") -- those
#      numbers describe the column's fixed schema, not something computed
#      from the current data, so they'll never appear among the candidates.
#   2. Small integers that are actually IDs/labels, not values (DG07/DG15
#      note "Product 8"/"Product 1" -- the regex has no notion of "this
#      digit is part of a name", so it extracts 8 and 1 as citations).
# Both are expected, not bugs; a human still has to read the NOTE-AUDIT
# line and use judgment, exactly as the task that added this asked for
# ("a check... is strictly better than prose, but do not over-engineer a
# general natural-language number extractor").

# Cases whose `_note` is hand-authored narrative prose making an editorial
# judgment call, not a value mechanically derivable from this case's own
# fields. Excluded from the exact-string MISMATCH comparison in main()
# (comparing hand-written prose byte-for-byte is the wrong tool -- see
# module docstring); still covered by the NOTE-AUDIT number check below.
_AUTHORED_NOTE_CASES = {"DG03", "DG07", "DG08", "DG13", "DG15"}

_NUM_RE = re.compile(r"(?<![\d%])-?\$?\d[\d,]*\.?\d*%?")
# The negative lookbehind keeps a range like "2.70%-2.82%" from being
# misread as "2.70" followed by "-2.82" (a spurious negative) -- a "-"
# right after a digit or "%" is a range separator, not a sign. Found while
# fixing DG09's own note, which cites exactly this shape (附录 BJ).


def _extract_note_numbers(note: str) -> list[tuple[float, ...]]:
    """Pull numeric literals out of a hand-written `_note` string.

    Each literal becomes a tuple of alternative readings that all count as
    "the same citation verified" -- a plain decimal (`0.0083`) or currency
    amount (`$284.70`, `$4,047,382.46`) is a 1-tuple, but a percentage
    (`5.6%`, `2.70%`) becomes `(5.6, 0.056)`: a note might state "5.6%
    more" (a relative difference) while the closest matching candidate is
    more naturally expressed as a fraction, and we'd rather accept either
    reading than flag a citation that's actually fine just because we
    guessed the wrong form.
    """
    out: list[tuple[float, ...]] = []
    for tok in _NUM_RE.findall(note):
        is_pct = tok.endswith("%")
        raw = tok.rstrip("%").replace("$", "").replace(",", "")
        if raw in ("", "-", "."):
            continue
        try:
            val = float(raw)
        except ValueError:
            continue
        out.append((val, val / 100) if is_pct else (val,))
    return out


def _note_candidates(fresh: dict) -> set[float]:
    """Numbers a note could legitimately cite, derived from this case's own
    freshly computed ground_truth fields: the raw values themselves, plus
    the simple pairwise relationships (absolute difference, % difference in
    both directions, ratio) a short comparison sentence is likely to quote.
    Deliberately not exhaustive -- see the module note on rejecting a
    general NL-number extractor.
    """
    nums = [v for v in fresh.values() if isinstance(v, (int, float))]
    cand: set[float] = {round(v, 6) for v in nums}
    for a in nums:
        for b in nums:
            if a is b:
                continue
            cand.add(round(abs(a - b), 6))
            if b != 0:
                cand.add(round(abs(a - b) / abs(b) * 100, 2))
                cand.add(round(abs(a - b) / abs(b) * 100, 1))
                cand.add(round(a / b * 100, 2))
            if a != 0:
                cand.add(round(abs(a - b) / abs(a) * 100, 2))
    return cand


def _audit_note(note: str, fresh: dict, tol: float = 0.01) -> list[tuple[float, ...]]:
    """Citations in `note` where NONE of the alternative readings (see
    `_extract_note_numbers`) match anything derivable from `fresh` within a
    small tolerance. A non-empty result is a signal for a human to check,
    not an automatic verdict of staleness -- see the "known, accepted
    false positives" comment above.
    """
    candidates = _note_candidates(fresh)
    return [alts for alts in _extract_note_numbers(note)
            if not any(abs(n - c) <= max(tol, abs(c) * 0.02) for n in alts for c in candidates)]


# ─── Original 20 (D01-E04) ───────────────────────────────────────────────

@register("D01")
def _d01(d: _Data) -> dict:
    by_region = d.sales.groupby("region")["revenue"].sum()
    return {"north_region_revenue": round(by_region["North"], 2),
            "central_region_revenue": round(by_region["Central"], 2)}


@register("D02")
def _d02(d: _Data) -> dict:
    by_cat = d.orders_x_products.groupby("category")["revenue"].mean()
    return {"books_avg_order_value": round(by_cat["Books"], 2),
            "office_supplies_avg_order_value": round(by_cat["Office Supplies"], 2)}


@register("D03")
def _d03(d: _Data) -> dict:
    counts = d.customers["country"].value_counts()
    return {"canada_customers": int(counts["Canada"]), "france_customers": int(counts["France"]),
            "total_customers": len(d.customers)}


@register("D04")
def _d04(d: _Data) -> dict:
    by_product = d.sales.groupby("product")["revenue"].sum().sort_values(ascending=False)
    return {"top_product_revenue": round(by_product.iloc[0], 2), "n_products": len(by_product)}


@register("D05")
def _d05(d: _Data) -> dict:
    monthly = d.orders.groupby(d.orders["order_date"].dt.to_period("M")).size()
    return {"dec_2024_orders": int(monthly.get("2024-12", 0)), "jan_2024_orders": int(monthly.get("2024-01", 0))}


@register("DG01")
def _dg01(d: _Data) -> dict:
    qrev = d.sales[d.sales["year"] == 2023].groupby("quarter")["revenue"].sum()
    return {"q3_2023_revenue": round(qrev[3], 2), "q2_2023_revenue": round(qrev[2], 2),
            "q4_2023_revenue": round(qrev[4], 2)}


@register("DG02")
def _dg02(d: _Data) -> dict:
    march = d.churn[d.churn["cohort_month"].astype(str) == "2024-03"]
    churned, total = int(march["churned"].sum()), len(march)
    return {"march_2024_churned": churned, "march_2024_total": total,
            "march_2024_churn_rate_pct": round(churned / total * 100, 2)}


@register("DG03")
def _dg03(d: _Data) -> dict:
    # `_note` kept here (unlike DG07/08/13/15) since it's a short,
    # single-fact claim with no number to go stale -- still in
    # _AUTHORED_NOTE_CASES so main() doesn't exact-match it, since "is the
    # highest" is itself a claim that could in principle flip.
    by_region = d.sales.groupby("region")["revenue"].sum()
    return {"north_region_revenue": round(by_region["North"], 2),
            "_note": "North is actually the HIGHEST-revenue region, not underperforming"}


@register("C01")
def _c01(d: _Data) -> dict:
    qrev = d.sales[d.sales["year"] == 2023].groupby("quarter")["revenue"].sum()
    return {"q1_2023_revenue": round(qrev[1], 2), "q2_2023_revenue": round(qrev[2], 2),
            "q3_2023_revenue": round(qrev[3], 2), "q4_2023_revenue": round(qrev[4], 2)}


@register("C02")
def _c02(d: _Data) -> dict:
    # NOTE: fresh computation (both mean-of-per-campaign-rate and
    # aggregate conversions/clicks) gives Search=6.49%, Display=1.20% --
    # the stored 6.41%/1.22% doesn't match either method. Small, real
    # drift from however this was originally derived pre-Step-4; flagged
    # by --check rather than silently overwritten.
    by_channel = d.mkt.groupby("channel")["conversion_rate"].mean() * 100
    return {"search_conversion_rate_pct": round(by_channel["Search"], 2),
            "display_conversion_rate_pct": round(by_channel["Display"], 2)}


@register("C04")
def _c04(d: _Data) -> dict:
    by_cat = d.products.groupby("category")["margin"].mean()
    return {"books_avg_margin": round(by_cat["Books"], 2),
            "office_supplies_avg_margin": round(by_cat["Office Supplies"], 2)}


@register("E01")
def _e01(d: _Data) -> dict:
    return {"n_rows": len(d.sales), "n_columns": d.sales.shape[1] - 2}  # -2 for derived quarter/year


@register("E02")
def _e02(d: _Data) -> dict:
    return {"revenue_outliers_iqr": _iqr_outlier_count(d.sales["revenue"]),
            "units_outliers_iqr": _iqr_outlier_count(d.sales["units"])}


@register("E03")
def _e03(d: _Data) -> dict:
    return {"n_orders": len(d.orders), "n_customers": len(d.customers)}


@register("E04")
def _e04(d: _Data) -> dict:
    return {"spend_revenue_correlation": round(d.mkt["spend"].corr(d.mkt["revenue"]), 4)}


# ─── Batch 2 (D06-E08) ────────────────────────────────────────────────────

@register("D06")
def _d06(d: _Data) -> dict:
    oc = d.orders_x_customers
    return {"usa_order_count": int((oc["country"] == "USA").sum()), "total_orders": len(d.orders)}


@register("D07")
def _d07(d: _Data) -> dict:
    by_channel = d.mkt.groupby("channel")[["spend", "revenue"]].sum()
    return {"email_spend": round(by_channel.loc["Email", "spend"], 2),
            "email_revenue": round(by_channel.loc["Email", "revenue"], 2),
            "search_revenue": round(by_channel.loc["Search", "revenue"], 2)}


@register("D08")
def _d08(d: _Data) -> dict:
    by_churned = d.churn.groupby("churned")["tenure_months"].mean()
    return {"churned_avg_tenure_months": round(by_churned[1], 2),
            "retained_avg_tenure_months": round(by_churned[0], 2)}


@register("D09")
def _d09(d: _Data) -> dict:
    # There is a genuine 2-way tie for max revenue (order_id 6553 and 5810,
    # both $4992.20, both from UK customers) -- which row idxmax() returns
    # depends on merge/row order, not a real answer. top_order_id was
    # dropped from ground_truth for exactly this reason (found via
    # scripts/compute_ground_truth.py's --check on 2026-08-01); revenue and
    # country are both stable regardless of which tied row is picked.
    oc = d.orders_x_customers
    top_revenue = oc["revenue"].max()
    top_rows = oc[oc["revenue"] == top_revenue]
    countries = set(top_rows["country"])
    assert len(countries) == 1, f"tied top orders disagree on country: {countries}"
    return {"top_order_revenue": round(float(top_revenue), 2), "top_order_customer_country": countries.pop()}


def _login_bucket_churn(d: _Data) -> pd.Series:
    b = pd.cut(d.churn["login_days_last_30d"], bins=[-1, 0, 9, 30], labels=["0", "1-9", "10+"])
    return d.churn.groupby(b, observed=True)["churned"].mean()


@register("DG05")
def _dg05(d: _Data) -> dict:
    rates = _login_bucket_churn(d)
    return {"churn_rate_0_login_days": round(rates["0"], 4), "churn_rate_10plus_login_days": round(rates["10+"], 4)}


@register("DG06")
def _dg06(d: _Data) -> dict:
    by_channel = d.mkt.groupby("channel")["roi"].mean()
    return {"email_avg_roi": round(by_channel["Email"], 2), "display_avg_roi": round(by_channel["Display"], 2)}


@register("DG07")
def _dg07(d: _Data) -> dict:
    # No `_note` here: DG07's stored note is hand-authored narrative
    # (see _AUTHORED_NOTE_CASES) -- this function only tracks the numeric
    # fields the note quotes, checked against it by _audit_note in main().
    p8 = d.products[d.products["product_id"] == 8].iloc[0]
    return {"product_8_unit_price": round(float(p8["unit_price"]), 2), "product_8_cost": round(float(p8["cost"]), 2),
            "product_8_margin": round(float(p8["margin"]), 2)}


@register("DG08")
def _dg08(d: _Data) -> dict:
    # No `_note` here -- see DG07's comment; DG08's stored note is
    # hand-authored (see _AUTHORED_NOTE_CASES).
    oc = d.orders_x_customers
    by_segment = oc.groupby("segment")["revenue"].sum()
    return {"consumer_total_revenue": round(by_segment["Consumer"], 2),
            "corporate_total_revenue": round(by_segment["Corporate"], 2)}


@register("C05")
def _c05(d: _Data) -> dict:
    rates = _login_bucket_churn(d)
    return {"churn_rate_0_days": round(rates["0"], 4), "churn_rate_10plus_days": round(rates["10+"], 4)}


@register("C06")
def _c06(d: _Data) -> dict:
    oc = d.orders_x_customers
    by_segment = oc.groupby("segment")["revenue"].mean()
    return {"consumer_avg_order_revenue": round(by_segment["Consumer"], 2),
            "corporate_avg_order_revenue": round(by_segment["Corporate"], 2),
            "home_office_avg_order_revenue": round(by_segment["Home Office"], 2)}


@register("C07")
def _c07(d: _Data) -> dict:
    reg_year = d.sales[d.sales["region"].isin(["North", "South"])].groupby(["region", "year"])["revenue"].sum()
    return {"north_2022_revenue": round(reg_year["North", 2022], 2), "north_2024_revenue": round(reg_year["North", 2024], 2),
            "south_2022_revenue": round(reg_year["South", 2022], 2), "south_2024_revenue": round(reg_year["South", 2024], 2)}


@register("C08")
def _c08(d: _Data) -> dict:
    q1_2024 = d.orders[(d.orders["order_date"] >= "2024-01-01") & (d.orders["order_date"] <= "2024-03-31")]
    q4_2024 = d.orders[(d.orders["order_date"] >= "2024-10-01") & (d.orders["order_date"] <= "2024-12-31")]
    return {"q1_2024_orders": len(q1_2024), "q4_2024_orders": len(q4_2024)}


@register("E06")
def _e06(d: _Data) -> dict:
    return {"n_rows": len(d.churn), "n_columns": d.churn.shape[1]}


@register("E07")
def _e07(d: _Data) -> dict:
    return {"roi_outliers_iqr": _iqr_outlier_count(d.mkt["roi"]), "spend_outliers_iqr": _iqr_outlier_count(d.mkt["spend"])}


@register("E08")
def _e08(d: _Data) -> dict:
    return {"support_calls_churn_correlation": round(d.churn["support_calls_last_90d"].corr(d.churn["churned"]), 4)}


# ─── Batch 3 (D10-E13) ────────────────────────────────────────────────────

@register("D10")
def _d10(d: _Data) -> dict:
    by_rep = d.sales.groupby("rep_id")["revenue"].sum().sort_values(ascending=False)
    return {"top_rep_id": int(by_rep.index[0]), "top_rep_revenue": round(by_rep.iloc[0], 2)}


@register("D11")
def _d11(d: _Data) -> dict:
    by_product = d.sales.groupby("product")["units"].sum().sort_values(ascending=False)
    return {"widget_b_units": int(by_product["Widget B"]), "widget_a_units": int(by_product["Widget A"])}


@register("D12")
def _d12(d: _Data) -> dict:
    by_year = d.customers["joined_date"].dt.year.value_counts()
    return {"customers_joined_2023": int(by_year[2023]), "customers_joined_2020": int(by_year[2020])}


@register("D13")
def _d13(d: _Data) -> dict:
    return {"avg_discount": round(d.orders["discount"].mean(), 4), "orders_with_discount": int((d.orders["discount"] > 0).sum()),
            "total_orders": len(d.orders)}


@register("DG09")
def _dg09(d: _Data) -> dict:
    # CTR per channel = mean of each campaign's own clicks/impressions
    # ratio (unweighted across campaigns), matching how the pipeline's own
    # step primitives would actually produce this: pandas_derive() adds a
    # per-row `ctr` column, then pandas_groupby(agg_col="ctr", func="mean")
    # aggregates it -- there's no single-step primitive for the alternative
    # sum(clicks)/sum(impressions) reading, which would need groupby-sum on
    # two columns followed by a manual cross-aggregate division the step
    # executor doesn't expose. This also matches C09's existing use of the
    # same d.mkt["ctr"] column, so it isn't a new convention introduced
    # just for this case (附录 BJ).
    #
    # The note is an f-string tied to this same computation (like DG19's),
    # not a hardcoded string, specifically so it self-heals on the next
    # recompute instead of quietly going stale next to the numbers it
    # cites -- the exact failure mode this case itself was found in.
    by_channel = d.mkt.groupby("channel")["ctr"].mean()
    lowest = by_channel.idxmin()
    return {"display_ctr": round(by_channel["Display"], 4), "email_ctr": round(by_channel["Email"], 4),
            "search_ctr": round(by_channel["Search"], 4), "social_ctr": round(by_channel["Social"], 4),
            "affiliate_ctr": round(by_channel["Affiliate"], 4),
            "_note": (f"Display's CTR ({by_channel['Display']:.2%}) is NOT the lowest -- {lowest} is "
                      f"({by_channel[lowest]:.2%}), and the full spread across channels "
                      f"({by_channel.min():.2%}-{by_channel.max():.2%}) shows no meaningful "
                      f"channel-level underperformance for Display specifically to explain")}


def _monthly_charges_by_plan(d: _Data) -> pd.Series:
    return d.churn.groupby("plan")["monthly_charges"].mean()


@register("DG10")
def _dg10(d: _Data) -> dict:
    by_plan = _monthly_charges_by_plan(d)
    return {"premium_avg_monthly_charge": round(by_plan["Premium"], 2), "basic_avg_monthly_charge": round(by_plan["Basic"], 2)}


@register("DG11")
def _dg11(d: _Data) -> dict:
    by_cat = d.products.groupby("category")["margin"].mean()
    return {"office_supplies_avg_margin": round(by_cat["Office Supplies"], 2), "books_avg_margin": round(by_cat["Books"], 2)}


@register("C09")
def _c09(d: _Data) -> dict:
    by_channel = d.mkt.groupby("channel")["ctr"].mean()
    return {"search_ctr": round(by_channel["Search"], 4), "social_ctr": round(by_channel["Social"], 4)}


@register("C10")
def _c10(d: _Data) -> dict:
    by_plan = _monthly_charges_by_plan(d)
    return {"basic_avg_monthly_charge": round(by_plan["Basic"], 2), "premium_avg_monthly_charge": round(by_plan["Premium"], 2)}


@register("C11")
def _c11(d: _Data) -> dict:
    by_channel = d.sales.groupby("channel")["units"].sum()
    return {"online_units": int(by_channel["Online"]), "retail_units": int(by_channel["Retail"])}


@register("C12")
def _c12(d: _Data) -> dict:
    cutoff = pd.Timestamp("2022-01-01")
    before = int((d.customers["joined_date"] < cutoff).sum())
    return {"joined_before_2022": before, "joined_2022_or_later": len(d.customers) - before}


@register("E09")
def _e09(d: _Data) -> dict:
    pct = (d.customers["segment"].value_counts(normalize=True) * 100).round(1)
    return {"consumer_pct": float(pct["Consumer"]), "corporate_pct": float(pct["Corporate"]),
            "home_office_pct": float(pct["Home Office"])}


@register("E10")
def _e10(d: _Data) -> dict:
    return {"discount_outliers_iqr": _iqr_outlier_count(d.orders["discount"]),
            "_note": "discount values are capped/quantized (min 0, max 0.20); the IQR-based outlier count is 0 -- "
                     "there are no statistical outliers in this field"}


@register("E11")
def _e11(d: _Data) -> dict:
    return {"spend_clicks_correlation": round(d.mkt["spend"].corr(d.mkt["clicks"]), 4)}


@register("E12")
def _e12(d: _Data) -> dict:
    return {"n_rows": len(d.orders), "n_columns": 7}  # 7 stored columns, excluding derived order_date dtype conversion


# ─── Batch 4 (D14-E19) ─────────────────────────────────────────────────────

@register("D14")
def _d14(d: _Data) -> dict:
    by_cat = d.orders_x_products.groupby("category")["quantity"].sum()
    return {"furniture_quantity": int(by_cat["Furniture"]), "books_quantity": int(by_cat["Books"])}


@register("D15")
def _d15(d: _Data) -> dict:
    counts = d.mkt["channel"].value_counts()
    return {"email_campaign_count": int(counts["Email"]), "search_campaign_count": int(counts["Search"])}


@register("D16")
def _d16(d: _Data) -> dict:
    by_product = d.orders_x_products.groupby("name")["revenue"].mean().sort_values(ascending=False)
    return {"top_product_by_avg_order_revenue": by_product.index[0],
            "top_product_avg_order_revenue": round(by_product.iloc[0], 2)}


@register("D17")
def _d17(d: _Data) -> dict:
    counts = d.churn["plan"].value_counts()
    return {"premium_plan_customers": int(counts["Premium"]), "basic_plan_customers": int(counts["Basic"]),
            "standard_plan_customers": int(counts["Standard"])}


@register("DG13")
def _dg13(d: _Data) -> dict:
    # No `_note` here -- see DG07's comment; DG13's stored note is
    # hand-authored and deliberately split ("half the premise is true,
    # half isn't" -- 附录 BH.6), not a generic template (see
    # _AUTHORED_NOTE_CASES).
    by_cat_rev = d.orders_x_products.groupby("category")["revenue"].sum()
    by_cat_margin = d.products.groupby("category")["margin"].mean()
    return {"furniture_revenue": round(by_cat_rev["Furniture"], 2), "books_revenue": round(by_cat_rev["Books"], 2),
            "books_avg_margin": round(by_cat_margin["Books"], 2), "furniture_avg_margin": round(by_cat_margin["Furniture"], 2)}


@register("DG14")
def _dg14(d: _Data) -> dict:
    b = pd.cut(d.churn["support_calls_last_90d"], bins=[-1, 1, 3, 100], labels=["0-1", "2-3", "4+"])
    rates = d.churn.groupby(b, observed=True)["churned"].mean()
    return {"churn_rate_0_1_calls": round(rates["0-1"], 4), "churn_rate_4plus_calls": round(rates["4+"], 4)}


@register("DG15")
def _dg15(d: _Data) -> dict:
    # No `_note` here -- see DG07's comment; DG15's stored note is
    # hand-authored (see _AUTHORED_NOTE_CASES).
    p1 = d.products[d.products["product_id"] == 1].iloc[0]
    return {"product_1_unit_price": round(float(p1["unit_price"]), 2), "product_1_cost": round(float(p1["cost"]), 2),
            "product_1_margin": round(float(p1["margin"]), 2)}


@register("C13")
def _c13(d: _Data) -> dict:
    return _d14(d)


@register("C14")
def _c14(d: _Data) -> dict:
    return _d15(d)


@register("C15")
def _c15(d: _Data) -> dict:
    rates = d.churn.groupby("plan")["churned"].mean()
    return {"basic_churn_rate": round(rates["Basic"], 4), "premium_churn_rate": round(rates["Premium"], 4)}


@register("C16")
def _c16(d: _Data) -> dict:
    rev = d.sales[d.sales["year"] == 2023].groupby("region")["revenue"].sum()
    return {"west_2023_revenue": round(rev["West"], 2), "east_2023_revenue": round(rev["East"], 2)}


@register("E14")
def _e14(d: _Data) -> dict:
    return {"quantity_outliers_iqr": _iqr_outlier_count(d.orders["quantity"]),
            "_note": "order quantity is capped/quantized (1-10 units); the IQR-based outlier count is 0 -- "
                     "there are no statistical outliers in this field"}


@register("E15")
def _e15(d: _Data) -> dict:
    return {"unit_price_margin_correlation": round(d.products["unit_price"].corr(d.products["margin"]), 4)}


@register("E17")
def _e17(d: _Data) -> dict:
    return {"tenure_monthly_charges_correlation": round(d.churn["tenure_months"].corr(d.churn["monthly_charges"]), 4)}


@register("E18")
def _e18(d: _Data) -> dict:
    return {"impressions_outliers_iqr": _iqr_outlier_count(d.mkt["impressions"])}


# ─── Batch 5 (D18-E24) ─────────────────────────────────────────────────────

@register("D18")
def _d18(d: _Data) -> dict:
    oc = d.orders_x_customers
    return {"germany_revenue": round(oc[oc["country"] == "Germany"]["revenue"].sum(), 2)}


def _order_count_by_category(d: _Data) -> pd.Series:
    return d.orders_x_products["category"].value_counts()


@register("D19")
def _d19(d: _Data) -> dict:
    counts = _order_count_by_category(d)
    return {"furniture_order_count": int(counts["Furniture"]), "books_order_count": int(counts["Books"])}


def _discount_by_category(d: _Data) -> pd.Series:
    return d.orders_x_products.groupby("category")["discount"].mean()


@register("D20")
def _d20(d: _Data) -> dict:
    by_cat = _discount_by_category(d)
    return {"books_avg_discount": round(by_cat["Books"], 4), "furniture_avg_discount": round(by_cat["Furniture"], 4)}


@register("D21")
def _d21(d: _Data) -> dict:
    charges = d.churn[d.churn["churned"] == 1]["monthly_charges"]
    return {"churned_monthly_charges_mean": round(charges.mean(), 2), "churned_monthly_charges_median": round(charges.median(), 2)}


@register("DG17")
def _dg17(d: _Data) -> dict:
    by_churned = d.churn.groupby("churned")["monthly_charges"].mean()
    return {"monthly_charges_churn_correlation": round(d.churn["monthly_charges"].corr(d.churn["churned"]), 4),
            "churned_avg_monthly_charge": round(by_churned[1], 2), "retained_avg_monthly_charge": round(by_churned[0], 2)}


@register("DG18")
def _dg18(d: _Data) -> dict:
    counts = _order_count_by_category(d)
    return {"books_order_count": int(counts["Books"]), "furniture_order_count": int(counts["Furniture"])}


@register("DG19")
def _dg19(d: _Data) -> dict:
    corr = round(d.orders["discount"].corr(d.orders["quantity"]), 4)
    return {"discount_quantity_correlation": corr,
            "_note": f"The correlation between order quantity and discount is essentially zero ({corr}) -- "
                     "there is no real relationship to explain here"}


@register("C17")
def _c17(d: _Data) -> dict:
    by_country = d.orders_x_customers.groupby("country")["revenue"].sum()
    return {"germany_revenue": round(by_country["Germany"], 2), "uk_revenue": round(by_country["UK"], 2)}


@register("C18")
def _c18(d: _Data) -> dict:
    counts = _order_count_by_category(d)
    return {"books_order_count": int(counts["Books"]), "electronics_order_count": int(counts["Electronics"])}


@register("C19")
def _c19(d: _Data) -> dict:
    by_cat = _discount_by_category(d)
    return {"electronics_avg_discount": round(by_cat["Electronics"], 4), "clothing_avg_discount": round(by_cat["Clothing"], 4)}


@register("C20")
def _c20(d: _Data) -> dict:
    by_churned = d.churn.groupby("churned")["monthly_charges"].mean()
    return {"churned_avg_monthly_charge": round(by_churned[1], 2), "retained_avg_monthly_charge": round(by_churned[0], 2)}


@register("E20")
def _e20(d: _Data) -> dict:
    return {"conversions_outliers_iqr": _iqr_outlier_count(d.mkt["conversions"])}


@register("E21")
def _e21(d: _Data) -> dict:
    return {"discount_quantity_correlation": round(d.orders["discount"].corr(d.orders["quantity"]), 4)}


@register("E22")
def _e22(d: _Data) -> dict:
    return {"n_products": len(d.products), "n_columns": d.products.shape[1] - 1}  # -1 for derived margin


@register("E23")
def _e23(d: _Data) -> dict:
    return {"tenure_outliers_iqr": _iqr_outlier_count(d.churn["tenure_months"]),
            "_note": "customer tenure ranges 1-60 months with no statistical outliers (IQR-based outlier count is 0)"}


@register("E24")
def _e24(d: _Data) -> dict:
    return {"order_revenue_outliers_iqr": _iqr_outlier_count(d.orders["revenue"])}


# ─── main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="Overwrite ground_truth in test_suite.json with freshly computed values")
    parser.add_argument("--case", nargs="+", default=None, help="Only check/write these case ids")
    args = parser.parse_args()

    suite = json.loads(SUITE_PATH.read_text())
    by_id = {c["id"]: c for c in suite}
    d = _Data()

    case_ids = args.case if args.case else sorted(by_id.keys())
    checked, mismatches, skipped, errors = 0, [], [], []

    for cid in case_ids:
        if cid not in by_id:
            print(f"  [{cid}] not in suite, skipping")
            continue
        if cid not in COMPUTERS:
            skipped.append(cid)
            continue
        checked += 1
        try:
            fresh = COMPUTERS[cid](d)
        except Exception as exc:
            errors.append(cid)
            print(f"  [{cid}] ERROR computing ground truth: {exc}")
            continue

        stored = by_id[cid]["ground_truth"]
        # Compare numeric keys with a small tolerance (rounding drift is
        # not a real mismatch); string keys must match exactly -- except
        # `_note` on _AUTHORED_NOTE_CASES, where exact-string comparison is
        # the wrong tool (hand-authored prose this script doesn't try to
        # regenerate; see that set's docstring). Those are covered by the
        # NOTE-AUDIT check below instead.
        diffs = {}
        for k in set(fresh) | set(stored):
            if k == "_note" and cid in _AUTHORED_NOTE_CASES:
                continue
            fv, sv = fresh.get(k), stored.get(k)
            if isinstance(fv, (int, float)) and isinstance(sv, (int, float)):
                if abs(fv - sv) > max(1e-6, abs(sv) * 1e-4):
                    diffs[k] = (sv, fv)
            elif fv != sv:
                diffs[k] = (sv, fv)

        if diffs:
            mismatches.append(cid)
            print(f"  [{cid}] MISMATCH:")
            for k, (sv, fv) in diffs.items():
                print(f"      {k}: stored={sv!r}  fresh={fv!r}")
            if args.write:
                if cid in _AUTHORED_NOTE_CASES and "_note" in stored:
                    fresh = {**fresh, "_note": stored["_note"]}  # preserve hand-authored prose
                by_id[cid]["ground_truth"] = fresh
                print("      -> overwritten")
        else:
            print(f"  [{cid}] OK ({len(fresh)} field(s) match)")

        # NOTE-AUDIT: independent of the MISMATCH check above -- runs on
        # every case with a `_note`, whether or not fresh/stored numeric
        # fields agree, and whether or not the case is in
        # _AUTHORED_NOTE_CASES (附录 BI.2/BJ: this is the check that would
        # have caught 52fd014's stale-note regression on the day it
        # happened).
        note = stored.get("_note")
        if isinstance(note, str):
            unverified = _audit_note(note, fresh)
            if unverified:
                cited = [alts[0] for alts in unverified]
                print(f"  [{cid}] NOTE-AUDIT: note cites {cited} -- not found among this case's "
                      f"current computed values or simple pairwise differences/ratios; verify by hand "
                      f"(known false positives: column min/max caps like E10/E14/E23, and small integers "
                      f"that are actually IDs/labels embedded in prose like DG07/DG15's 'Product 8'/'Product 1')")

    print(f"\n{checked} case(s) checked, {len(mismatches)} mismatch(es), {len(skipped)} skipped "
          f"(no checkable ground truth), {len(errors)} error(s).")
    if skipped:
        print(f"Skipped (predictive/data_mismatch/ambiguous, expected): {skipped}")

    if args.write and mismatches:
        SUITE_PATH.write_text(json.dumps(suite, indent=2) + "\n")
        print(f"\nWrote {len(mismatches)} updated ground_truth value(s) to {SUITE_PATH}.")
    elif not args.write and mismatches:
        print("\nRun with --write to overwrite test_suite.json with these freshly computed values.")


if __name__ == "__main__":
    main()
