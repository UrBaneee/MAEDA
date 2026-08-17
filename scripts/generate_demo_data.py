"""
Generate demo datasets for MAEDA — Phase 12.

Produces four datasets:
  data/demo/sales_data.csv          — 3 years, regional sales, intentional quality issues
  data/demo/churn_data.csv          — customer churn with behavioural features
  data/demo/marketing_campaigns.csv — spend vs conversions
  data/demo/ecommerce_orders.db     — SQLite with orders, products, customers tables

Run: python scripts/generate_demo_data.py
"""
from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import subprocess
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SEED = 42

# 附录 BC.1/BF: previously a single module-level `rng`/`np.random.seed(SEED)`
# pair shared across all four generate_*() calls, consumed sequentially in
# __main__'s fixed order (sales -> churn -> marketing -> ecommerce). Any
# change to an EARLIER generator's random-draw COUNT silently shifted every
# LATER generator's output, even with its own code and parameters completely
# unchanged -- this is exactly what happened when TB5 changed generate_sales()
# (阶段 3, 附录 AZ/BB/BC): churn/marketing/ecommerce data changed too, purely
# from stream-position drift, invalidating downstream eval fixtures with no
# visible cause (row counts and file sizes looked untouched -- see 附录 BC.1's
# "ecommerce_orders.db" evidence for why that made it hard to spot).
#
# Each generate_*() function now builds its OWN local RNG state (see
# `_seeded_rngs()` below) instead of reading this module-level pair --
# independent of what any other generator call did or didn't consume.
#
# 附录 BF: `np.random.RandomState(seed)` here, deliberately NOT
# `np.random.default_rng(seed)` (numpy's modern recommendation for new code).
# The two use DIFFERENT underlying algorithms (RandomState: legacy MT19937,
# the same one `np.random.seed()`'s old global state used; default_rng:
# PCG64) -- same seed integer, different number sequence. Using RandomState
# is what lets `generate_sales()` specifically reproduce the exact same
# output as before this change (see that function's own docstring for why),
# and using the same RNG class in all four generators (rather than the
# "legacy" one only where it happens to matter) keeps the isolation pattern
# uniform and auditable instead of mixing two different numpy RNG APIs
# across this one file.
def _seeded_rngs(seed: int) -> tuple[random.Random, np.random.RandomState]:
    return random.Random(seed), np.random.RandomState(seed)

OUT = Path("data/demo")
OUT.mkdir(parents=True, exist_ok=True)


# ─── TB5 event script loading (ECOSYSTEM_INTEGRATION_PLAN.md 附录 X/AG/AI) ────
#
# 定案 #10: MAEDA fixes the rag-framework commit/tag rather than trusting
# whatever's currently checked out there, and recomputes the sha256 against
# the sidecar manifest before using the content -- a `git show` against the
# frozen tag, not a plain file read, so a later unrelated commit on that
# repo's `main` can't silently change what gets rendered here.

_RAG_FRAMEWORK_REPO = Path.home() / "rag-framework"
_EVENT_SCRIPT_TAG = "event-script-v1"
_EVENT_SCRIPT_YAML_PATH = "eval/fixtures/event_script/event_script.yaml"
_EVENT_SCRIPT_MANIFEST_PATH = "eval/fixtures/event_script/event_script.manifest.json"


def _git_show(path: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(_RAG_FRAMEWORK_REPO), "show", f"{_EVENT_SCRIPT_TAG}:{path}"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not read {path!r} at tag {_EVENT_SCRIPT_TAG!r} from "
            f"{_RAG_FRAMEWORK_REPO} (定案 #11: same-machine shared filesystem "
            "assumption) -- is the repo cloned there and has the tag been "
            f"pushed? git stderr: {result.stderr.strip()}"
        )
    return result.stdout


def load_frozen_events() -> list[dict]:
    """Load the 12 TB5 events, pinned to the frozen tag and hash-verified
    against event_script.manifest.json (附录 X.5 canonicalization, 附录
    AI.2's recorded sha256) before anything here trusts its content."""
    yaml_text = _git_show(_EVENT_SCRIPT_YAML_PATH)
    manifest = json.loads(_git_show(_EVENT_SCRIPT_MANIFEST_PATH))
    data = yaml.safe_load(yaml_text)
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    actual_sha = hashlib.sha256(canonical).hexdigest()
    if actual_sha != manifest["sha256"]:
        raise RuntimeError(
            f"event_script.yaml at tag {_EVENT_SCRIPT_TAG!r} does not match "
            f"its manifest sha256 (manifest={manifest['sha256']}, "
            f"recomputed={actual_sha}) -- refusing to render off a script "
            "whose integrity can't be confirmed."
        )
    return data["events"]


# 附录 AF.1/AG.2: shape -> time-weight function over the event's window,
# where frac is the row's position in [valid_from, valid_to] as [0, 1].
# `ramp`/`spike` both average to 0.5 across the window by construction,
# matching verify.py's `_SHAPE_MEAN_WEIGHT` exactly -- the detectability gate
# and this generator must agree on that mean or a render that looks fine
# here could still fail `verify.py --data-path`.
_SHAPE_WEIGHT = {
    "step": lambda frac: 1.0,
    "ramp": lambda frac: frac,                    # linear 0 -> 1, mean 0.5
    "spike": lambda frac: 1.0 - abs(2 * frac - 1),  # triangular peak at midpoint, mean 0.5
}

# AD.2 (non-blocking reminder, resolved here): three events had their
# `segment` widened to region-only (附录 AC.5/AF) even though
# `business_description` still names one specific product. Applying the
# same uplift to every product in the region would read as "the corpus
# says single-product promo, the data shows the whole region moving in
# lockstep" -- AD.2's flagged inconsistency. Weighting the named product
# higher and the rest lower keeps the REGION-LEVEL AVERAGE unchanged
# (lead*3.0 + other*4*0.5, divided by 5, == 1.0 exactly), so
# `verify.py --data-path`'s region-level detectability check -- which
# doesn't know about this per-product split -- sees the same aggregate
# signal AC.5 verified; only the per-product breakdown differs.
_LEAD_PRODUCT = {
    "EVT-2022-11-NORTH-PROMO": "Widget A",
    "EVT-2022-06-WEST-WIDGETB-LAUNCH": "Widget B",
    "EVT-2024-05-NORTH-GADGETPRO-EOL": "Gadget Pro",
}
_LEAD_PRODUCT_MULT = 3.0
_OTHER_PRODUCT_MULT = 0.5


def _segment_matches(segment: dict, region: str, product: str, channel: str) -> bool:
    if segment.get("region") not in (None, region):
        return False
    if segment.get("product") not in (None, product):
        return False
    if segment.get("channel") not in (None, channel):
        return False
    return True


def _event_multiplier(event: dict, d: _date, region: str, product: str, channel: str) -> float:
    """Multiplicative effect (1.0 = no effect) this event applies to its
    `data_signal.metric` on this row, given shape/magnitude/direction and
    (for the three AD.2 events) whether `product` is the named lead."""
    signal = event.get("data_signal")
    if not signal:
        return 1.0  # distractor / not this row's metric -- caller filters by metric first
    segment = signal.get("segment") or {}
    if not _segment_matches(segment, region, product, channel):
        return 1.0
    vf = _date.fromisoformat(event["valid_from"])
    vt = _date.fromisoformat(event["valid_to"])
    if not (vf <= d <= vt):
        return 1.0
    span_days = (vt - vf).days
    frac = (d - vf).days / span_days if span_days > 0 else 1.0
    weight = _SHAPE_WEIGHT[signal["shape"]](frac)

    lead_product = _LEAD_PRODUCT.get(event["event_id"])
    if lead_product is not None:
        weight *= _LEAD_PRODUCT_MULT if product == lead_product else _OTHER_PRODUCT_MULT

    sign = 1.0 if signal["direction"] == "up" else -1.0
    return 1.0 + sign * signal["magnitude"] * weight


# ─── 12.1 Sales data ──────────────────────────────────────────────────────────

# 附录 AC.5's noise coefficient (0.12, down from the original 0.20) holds --
# but its row count (24,000) turned out too low once actually rendered and
# checked against verify.py's real gate. AC.5's z-values were computed as if
# every event applied full-strength across its whole window; verify.py's
# `_SHAPE_MEAN_WEIGHT` (附录 AG.2/AH) correctly discounts `ramp`/`spike`
# events to a mean weight of 0.5, which AC.5 never re-applied to the two
# events that stayed `spike` (EVT-2022-11-NORTH-PROMO,
# EVT-2022-06-WEST-WIDGETB-LAUNCH) or to the `ramp` events -- three of them
# came up short (z 2.3-2.8, need >=3.0) against a real render at n=24,000.
# 48,000 clears all nine with real margin across multiple seeds; see the
# render/check script this was validated with (not committed -- ad hoc).
_NOISE_COEFF = 0.12

# AC.4: a fixed per-product implied unit price (± small row-level jitter)
# replaces the old fully-independent `rng.uniform(20, 80)` divisor, which
# added noise unrelated to any business signal and made `units`-metric
# events structurally harder to detect than `revenue`-metric ones.
_PRODUCT_PRICE = {
    "Widget A": 45.0, "Widget B": 38.0, "Gadget Pro": 62.0,
    "Gadget Lite": 30.0, "Service Pack": 55.0,
}


def generate_sales(n: int = 48_000, seed: int = SEED) -> pd.DataFrame:
    """
    附录 BF: `generate_sales()` runs first in `__main__` (nothing precedes
    it), so under the OLD shared-stream design it was already, in effect,
    drawing from a virgin seed=SEED state -- no other generator's calls came
    before it to shift its position. Isolating it (own `random.Random(seed)`
    + own `np.random.RandomState(seed)`, both algorithm-compatible with what
    the old global state used -- see the module-level comment above) is
    therefore output-PRESERVING for this one function specifically:
    `generate_sales(seed=SEED)` under this isolated version reproduces
    `data/demo/sales_data.csv` byte-for-byte (verified -- see 附录 BF).
    churn/marketing/ecommerce are NOT preserved by the same argument (each
    used to start mid-stream, at a position only the old coupled call order
    could reproduce) -- isolating them necessarily changes their output.
    """
    rng, np_state = _seeded_rngs(seed)
    regions = ["North", "South", "East", "West", "Central"]
    products = ["Widget A", "Widget B", "Gadget Pro", "Gadget Lite", "Service Pack"]
    channels = ["Direct", "Partner", "Online", "Retail"]

    dates = pd.date_range("2022-01-01", "2024-12-31", periods=n)
    region_base = {"North": 500, "South": 380, "East": 420, "West": 460, "Central": 310}

    events = load_frozen_events()
    # X-V3/AC.3: business signal and quality noise are two independent
    # paths. Every event here is applied on top of the base
    # region/seasonal/noise model; the 3%/2%/1% quality-noise injection
    # below is unrelated and untouched by this loop.
    revenue_events = [e for e in events if (e.get("data_signal") or {}).get("metric") == "revenue"]
    units_events = [e for e in events if (e.get("data_signal") or {}).get("metric") == "units"]

    rows = []
    for d in dates:
        region = rng.choice(regions)
        product = rng.choice(products)
        channel = rng.choice(channels)
        d_date = d.date()
        base = region_base[region]
        # Q4 seasonality boost. 1.15, down from the original 1.3 -- not a
        # TB5 event, but its old strength stacked with EVT-2022-11-NORTH-
        # PROMO (North, November, +28% on top of an already-boosted Q4
        # baseline) to push that event's window mean past the whole-table
        # IQR ceiling that handle_outliers would clip at (附录 AA.2/AE.1's
        # predicted noise-coefficient/IQR tension, confirmed on a real
        # render). Lowering the row-noise coefficient further to compensate
        # would have tightened that same ceiling instead of loosening it
        # (AE.1); this lever doesn't, since it's independent of both.
        seasonal = 1.15 if d.month in (10, 11, 12) else 1.0
        # X-V1: the old hardcoded "Q3 2023 dip" is gone. It's now
        # EVT-2023-Q3-GLOBAL-UNEXPLAINED-DIP in the frozen event script
        # (segment={}, direction=down, magnitude=0.30, shape=step, same
        # window and magnitude as the multiplier this replaces) and is
        # applied below through the same machinery as every other event.
        revenue = max(0.0, np_state.normal(base * seasonal, base * _NOISE_COEFF))
        for event in revenue_events:
            revenue *= _event_multiplier(event, d_date, region, product, channel)
        revenue = round(revenue, 2)

        implied_price = _PRODUCT_PRICE[product] * rng.uniform(0.9, 1.1)
        units_f = revenue / implied_price
        for event in units_events:
            units_f *= _event_multiplier(event, d_date, region, product, channel)
        units = max(1, round(units_f))

        rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "region": region,
            "product": product,
            "channel": channel,
            "revenue": revenue,
            "units": units,
            "rep_id": rng.randint(1001, 1050),
        })

    df = pd.DataFrame(rows)

    # Intentional quality issues for Data Cleaner demo -- unchanged path,
    # independent of the business-signal path above (X-V3/AC.3).
    n_rows = len(df)
    # 3% missing revenue
    df.loc[df.sample(frac=0.03, random_state=1).index, "revenue"] = np.nan
    # 2% duplicate rows
    dup_idx = df.sample(frac=0.02, random_state=2).index
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)
    # 1% outlier revenues
    outlier_idx = df.sample(frac=0.01, random_state=3).index
    df.loc[outlier_idx, "revenue"] = df.loc[outlier_idx, "revenue"] * 10

    df.to_csv(OUT / "sales_data.csv", index=False)
    print(f"✅ sales_data.csv  ({len(df):,} rows)")
    return df


# ─── 12.2 Churn data ──────────────────────────────────────────────────────────

def generate_churn(n: int = 5_000, seed: int = SEED) -> pd.DataFrame:
    rng, np_state = _seeded_rngs(seed)
    rows = []
    for i in range(n):
        tenure = rng.randint(1, 60)
        plan = rng.choice(["Basic", "Standard", "Premium"])
        monthly_charges = {"Basic": 29, "Standard": 59, "Premium": 99}[plan]
        monthly_charges += np_state.normal(0, 5)
        support_calls = max(0, int(np_state.poisson(1.5)))
        login_days = max(0, int(np_state.normal(15, 8)))
        # March 2024 spike: new competitor launched
        cohort_month = rng.choice(pd.date_range("2023-01-01", "2024-06-01", freq="MS"))
        is_march_2024 = cohort_month.year == 2024 and cohort_month.month == 3

        # Churn probability model
        p_churn = 0.05
        p_churn += 0.15 if tenure < 6 else 0
        p_churn += 0.20 if support_calls >= 3 else 0
        p_churn += 0.10 if login_days < 5 else 0
        p_churn += 0.25 if is_march_2024 else 0  # competitor spike
        p_churn += 0.10 if plan == "Basic" else 0
        churned = 1 if rng.random() < min(p_churn, 0.95) else 0

        rows.append({
            "customer_id": f"C{i + 1:05d}",
            "cohort_month": cohort_month.strftime("%Y-%m"),
            "tenure_months": tenure,
            "plan": plan,
            "monthly_charges": round(monthly_charges, 2),
            "support_calls_last_90d": support_calls,
            "login_days_last_30d": login_days,
            "churned": churned,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "churn_data.csv", index=False)
    print(f"✅ churn_data.csv  ({len(df):,} rows, churn rate={df['churned'].mean():.1%})")
    return df


# ─── 12.3 Marketing campaigns ─────────────────────────────────────────────────

def generate_marketing(n: int = 2_000, seed: int = SEED) -> pd.DataFrame:
    rng, np_state = _seeded_rngs(seed)
    channels = ["Email", "Social", "Search", "Display", "Affiliate"]
    rows = []
    for i in range(n):
        channel = rng.choice(channels)
        spend = round(abs(np_state.normal(5000, 2000)), 2)
        # Channel-specific efficiency
        cvr = {"Email": 0.045, "Social": 0.022, "Search": 0.065,
               "Display": 0.012, "Affiliate": 0.038}[channel]
        impressions = max(100, int(spend * rng.uniform(50, 200)))
        clicks = max(1, int(impressions * rng.uniform(0.005, 0.05)))
        conversions = max(0, int(clicks * (cvr + np_state.normal(0, cvr * 0.3))))
        revenue = round(conversions * rng.uniform(80, 300), 2)
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=rng.randint(0, 364))

        rows.append({
            "campaign_id": f"CAM{i + 1:04d}",
            "date": date.strftime("%Y-%m-%d"),
            "channel": channel,
            "spend": spend,
            "impressions": impressions,
            "clicks": clicks,
            "conversions": conversions,
            "revenue": revenue,
            "roi": round((revenue - spend) / spend * 100, 1) if spend > 0 else 0,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "marketing_campaigns.csv", index=False)
    print(f"✅ marketing_campaigns.csv  ({len(df):,} rows)")
    return df


# ─── 12.4 eCommerce SQLite DB ─────────────────────────────────────────────────

def generate_ecommerce_db(seed: int = SEED) -> None:
    rng, _np_state = _seeded_rngs(seed)  # this generator doesn't use np.random.* itself
    db_path = OUT / "ecommerce_orders.db"
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Customers
    c.execute("DROP TABLE IF EXISTS customers")
    c.execute("""
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            name        TEXT,
            email       TEXT,
            country     TEXT,
            segment     TEXT,
            joined_date TEXT
        )
    """)
    countries = ["USA", "UK", "Germany", "France", "Canada", "Australia"]
    segments = ["Consumer", "Corporate", "Home Office"]
    customer_rows = [
        (i, f"Customer {i}", f"cust{i}@example.com",
         rng.choice(countries), rng.choice(segments),
         (pd.Timestamp("2020-01-01") + pd.Timedelta(days=rng.randint(0, 1460))).strftime("%Y-%m-%d"))
        for i in range(1, 1001)
    ]
    c.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?)", customer_rows)

    # Products
    c.execute("DROP TABLE IF EXISTS products")
    c.execute("""
        CREATE TABLE products (
            product_id   INTEGER PRIMARY KEY,
            name         TEXT,
            category     TEXT,
            unit_price   REAL,
            cost         REAL
        )
    """)
    categories = ["Electronics", "Furniture", "Office Supplies", "Clothing", "Books"]
    product_rows = [
        (i, f"Product {i}", rng.choice(categories),
         round(rng.uniform(10, 500), 2), round(rng.uniform(5, 200), 2))
        for i in range(1, 101)
    ]
    c.executemany("INSERT INTO products VALUES (?,?,?,?,?)", product_rows)

    # Orders
    c.execute("DROP TABLE IF EXISTS orders")
    c.execute("""
        CREATE TABLE orders (
            order_id    INTEGER PRIMARY KEY,
            customer_id INTEGER,
            product_id  INTEGER,
            order_date  TEXT,
            quantity    INTEGER,
            discount    REAL,
            revenue     REAL,
            FOREIGN KEY(customer_id) REFERENCES customers(customer_id),
            FOREIGN KEY(product_id)  REFERENCES products(product_id)
        )
    """)
    order_rows = []
    for i in range(1, 10_001):
        cid = rng.randint(1, 1000)
        pid = rng.randint(1, 100)
        _, _, _, unit_price, _ = product_rows[pid - 1]
        qty = rng.randint(1, 10)
        discount = rng.choice([0, 0.05, 0.10, 0.15, 0.20])
        revenue = round(unit_price * qty * (1 - discount), 2)
        date = (pd.Timestamp("2022-01-01") + pd.Timedelta(days=rng.randint(0, 1095))).strftime("%Y-%m-%d")
        order_rows.append((i, cid, pid, date, qty, discount, revenue))
    c.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?)", order_rows)

    conn.commit()
    conn.close()
    print(f"✅ ecommerce_orders.db  (1000 customers, 100 products, 10000 orders)")


if __name__ == "__main__":
    print("Generating MAEDA demo datasets…")
    generate_sales()
    generate_churn()
    generate_marketing()
    generate_ecommerce_db()
    print("\nAll datasets written to data/demo/")
