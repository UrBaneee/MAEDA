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

import random
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
rng = random.Random(SEED)
np.random.seed(SEED)

OUT = Path("data/demo")
OUT.mkdir(parents=True, exist_ok=True)


# ─── 12.1 Sales data ──────────────────────────────────────────────────────────

def generate_sales(n: int = 12_000) -> pd.DataFrame:
    regions = ["North", "South", "East", "West", "Central"]
    products = ["Widget A", "Widget B", "Gadget Pro", "Gadget Lite", "Service Pack"]
    channels = ["Direct", "Partner", "Online", "Retail"]

    dates = pd.date_range("2022-01-01", "2024-12-31", periods=n)
    region_base = {"North": 500, "South": 380, "East": 420, "West": 460, "Central": 310}

    rows = []
    for d in dates:
        region = rng.choice(regions)
        product = rng.choice(products)
        channel = rng.choice(channels)
        base = region_base[region]
        # Q4 seasonality boost
        seasonal = 1.3 if d.month in (10, 11, 12) else 1.0
        # Q3 2023 dip (for diagnostic demo)
        if d.year == 2023 and d.month in (7, 8, 9):
            seasonal *= 0.7
        revenue = round(max(0, np.random.normal(base * seasonal, base * 0.2)), 2)
        units = max(1, int(revenue / rng.uniform(20, 80)))
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

    # Intentional quality issues for Data Cleaner demo
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

def generate_churn(n: int = 5_000) -> pd.DataFrame:
    rows = []
    for i in range(n):
        tenure = rng.randint(1, 60)
        plan = rng.choice(["Basic", "Standard", "Premium"])
        monthly_charges = {"Basic": 29, "Standard": 59, "Premium": 99}[plan]
        monthly_charges += np.random.normal(0, 5)
        support_calls = max(0, int(np.random.poisson(1.5)))
        login_days = max(0, int(np.random.normal(15, 8)))
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

def generate_marketing(n: int = 2_000) -> pd.DataFrame:
    channels = ["Email", "Social", "Search", "Display", "Affiliate"]
    rows = []
    for i in range(n):
        channel = rng.choice(channels)
        spend = round(abs(np.random.normal(5000, 2000)), 2)
        # Channel-specific efficiency
        cvr = {"Email": 0.045, "Social": 0.022, "Search": 0.065,
               "Display": 0.012, "Affiliate": 0.038}[channel]
        impressions = max(100, int(spend * rng.uniform(50, 200)))
        clicks = max(1, int(impressions * rng.uniform(0.005, 0.05)))
        conversions = max(0, int(clicks * (cvr + np.random.normal(0, cvr * 0.3))))
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

def generate_ecommerce_db() -> None:
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
