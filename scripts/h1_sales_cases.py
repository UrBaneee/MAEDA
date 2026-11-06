"""
The 28 new H1 dev sales cases (附录 GR.3 item 5, GS). Each entry is the case
metadata plus a `compute(sales)` that derives its numeric ground truth from the
CLEAN sales table (data/truth/sales_data_clean.csv, 附录 GO.6/GQ). The pipeline
input for every case stays data/demo/sales_data.csv. `compute_ground_truth.py`
registers every entry here so a normal check run audits them like the older ones.

    python scripts/h1_sales_cases.py        # append missing cases to test_suite.json (split: dev)

All cases are `split: "dev"`. No holdout is touched. No API call.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

R = lambda x: round(float(x), 2)  # noqa: E731


def _rev(s, **f):
    for k, v in f.items():
        s = s[s[k] == v]
    return s["revenue"].sum()


def _y(s, y):
    return s[s["year"] == y]


def _q(s, y, q):
    return s[(s["year"] == y) & (s["quarter"] == q)]


def _s01(s):
    g = s.groupby("channel")["revenue"].sum()
    return {f"{k.lower()}_revenue": R(g[k]) for k in ["Direct", "Partner", "Online", "Retail"]}


def _s02(s):
    g = s.groupby("product")["revenue"].sum()
    return {f"{k.lower().replace(' ', '_')}_revenue": R(g[k]) for k in sorted(g.index)}


def _s03(s):
    g = s.groupby("region")["revenue"].mean()
    return {f"{k.lower()}_avg_revenue": R(g[k]) for k in sorted(g.index)}


def _s04(s):
    g = s.groupby("region")["units"].sum()
    return {f"{k.lower()}_units": int(g[k]) for k in sorted(g.index)}


def _s05(s):
    g = s.groupby("year")["revenue"].sum()
    return {f"revenue_{y}": R(g[y]) for y in (2022, 2023, 2024)}


def _s06(s):
    g = _y(s, 2024).groupby("quarter")["revenue"].sum()
    return {f"q{q}_2024_revenue": R(g[q]) for q in (1, 2, 3, 4)}


def _s07(s):
    g = s.groupby("region").size()
    return {f"{k.lower()}_transactions": int(g[k]) for k in sorted(g.index)}


def _s08(s):
    g = s.groupby("channel")["units"].mean()
    return {f"{k.lower()}_avg_units": R(g[k]) for k in ["Direct", "Partner", "Online", "Retail"]}


def _s09(s):
    return {"q4_2022_revenue": R(_q(s, 2022, 4)["revenue"].sum()),
            "q4_2024_revenue": R(_q(s, 2024, 4)["revenue"].sum())}


def _s10(s):
    t = _y(s, 2024)
    return {"online_2024_revenue": R(_rev(t, channel="Online")), "retail_2024_revenue": R(_rev(t, channel="Retail"))}


def _s11(s):
    t = _y(s, 2023)
    return {"widget_a_2023_units": int(t[t["product"] == "Widget A"]["units"].sum()),
            "gadget_pro_2023_units": int(t[t["product"] == "Gadget Pro"]["units"].sum())}


def _s12(s):
    g = s.groupby("channel")["revenue"].mean()
    return {"direct_avg_revenue": R(g["Direct"]), "partner_avg_revenue": R(g["Partner"])}


def _s13(s):
    t = _y(s, 2022)
    return {"south_2022_revenue": R(_rev(t, region="South")), "east_2022_revenue": R(_rev(t, region="East"))}


def _s14(s):
    return {"q1_2022_revenue": R(_q(s, 2022, 1)["revenue"].sum()),
            "q1_2024_revenue": R(_q(s, 2024, 1)["revenue"].sum())}


def _s15(s):
    g = s.groupby("region")["revenue"].sum()
    return {"lowest_region_revenue": R(g.min())}


def _s16(s):
    g = s.groupby("product")["revenue"].sum().sort_values(ascending=False).head(3)
    return {f"top{i}_product_revenue": R(v) for i, v in enumerate(g.values, 1)}


def _s17(s):
    g = s.groupby("channel")["units"].sum()
    return {"top_channel_units": int(g.max())}


def _s18(s):
    g = s.groupby("rep_id")["units"].sum().sort_values(ascending=False).head(3)
    out = {}
    for i, (rep, v) in enumerate(g.items(), 1):
        out[f"top{i}_rep_id"] = int(rep)
        out[f"top{i}_rep_units"] = int(v)
    return out


def _s19(s):
    t = _y(s, 2023)
    g = t.groupby(t["date"].dt.month)["revenue"].sum()
    return {"best_month_2023": int(g.idxmax()), "best_month_2023_revenue": R(g.max())}


def _s20(s):
    g = s.groupby("quarter")["revenue"].sum()
    return {"best_quarter": int(g.idxmax()), "best_quarter_revenue": R(g.max())}


def _s21(s):
    return {"west_gadget_pro_revenue": R(_rev(s, region="West", product="Gadget Pro"))}


def _s22(s):
    t = s[(s["product"] == "Service Pack") & (s["channel"] == "Partner")]
    return {"service_pack_partner_units": int(t["units"].sum())}


def _s23(s):
    return {"central_q4_2023_revenue": R(_q(s[s["region"] == "Central"], 2023, 4)["revenue"].sum())}


def _s24(s):
    t = _y(s, 2024)
    return {"gadget_lite_2024_avg_revenue": R(t[t["product"] == "Gadget Lite"]["revenue"].mean())}


def _s25(s):
    n = s[s["region"] == "North"]
    return {"north_transactions_2022": int(len(_y(n, 2022))), "north_transactions_2024": int(len(_y(n, 2024)))}


def _s26(s):
    tot = s["revenue"].sum()
    n = _rev(s, region="North")
    return {"north_revenue": R(n), "north_share_pct": round(float(n / tot * 100), 1)}


def _s27(s):
    a, b = _rev(_y(s, 2022), ), _rev(_y(s, 2024), )
    return {"revenue_2022": R(a), "revenue_2024": R(b), "growth_pct_2022_2024": round(float((b - a) / a * 100), 1)}


def _s28(s):
    return {"avg_monthly_revenue_2023": R(_y(s, 2023)["revenue"].sum() / 12),
            "avg_monthly_revenue_2024": R(_y(s, 2024)["revenue"].sum() / 12)}


# (id, query, query_type, metrics, dimensions, tags, tools, charts, compute)
_B = ["pandas_transform"]
_BC = ["pandas_transform", "comparison"]
SPEC = [
    ("S01", "What is total revenue for each sales channel?", "descriptive", ["revenue"], ["channel"], ["descriptive"], _B, ["bar"], _s01),
    ("S02", "Show total revenue by product", "descriptive", ["revenue"], ["product"], ["descriptive"], _B, ["bar"], _s02),
    ("S03", "What is the average revenue per transaction in each region?", "descriptive", ["revenue"], ["region"], ["descriptive"], _B, ["bar"], _s03),
    ("S04", "How many units were sold in each region?", "descriptive", ["units"], ["region"], ["descriptive"], _B, ["bar"], _s04),
    ("S05", "What was total revenue in each year from 2022 to 2024?", "descriptive", ["revenue"], ["year"], ["descriptive", "time_series"], _B, ["bar", "line"], _s05),
    ("S06", "Give me quarterly revenue for 2024", "descriptive", ["revenue"], ["quarter"], ["descriptive", "time_series"], _B, ["bar", "line"], _s06),
    ("S07", "How many sales transactions are there in each region?", "descriptive", ["transactions"], ["region"], ["descriptive"], _B, ["bar"], _s07),
    ("S08", "What is the average number of units per transaction for each channel?", "descriptive", ["units"], ["channel"], ["descriptive"], _B, ["bar"], _s08),
    ("S09", "What was revenue in Q4 2022 and in Q4 2024?", "descriptive", ["revenue"], ["quarter", "year"], ["descriptive", "time_series"], _B, ["bar"], _s09),
    ("S10", "Compare total revenue of the Online and Retail channels in 2024", "comparative", ["revenue"], ["channel"], ["comparative"], _BC, ["bar"], _s10),
    ("S11", "Compare units sold of Widget A and Gadget Pro in 2023", "comparative", ["units"], ["product"], ["comparative"], _BC, ["bar"], _s11),
    ("S12", "Compare average revenue per transaction between the Direct and Partner channels", "comparative", ["revenue"], ["channel"], ["comparative"], _BC, ["bar"], _s12),
    ("S13", "Compare total revenue of the South and East regions in 2022", "comparative", ["revenue"], ["region"], ["comparative"], _BC, ["bar"], _s13),
    ("S14", "Compare revenue in Q1 2022 with Q1 2024", "comparative", ["revenue"], ["quarter", "year"], ["comparative", "time_series"], _BC, ["bar"], _s14),
    ("S15", "Which region has the lowest total revenue, and what is it?", "descriptive", ["revenue"], ["region"], ["descriptive"], _B, ["bar", "horizontal_bar"], _s15),
    ("S16", "What are the top 3 products by total revenue?", "descriptive", ["revenue"], ["product"], ["descriptive"], _B, ["bar", "horizontal_bar"], _s16),
    ("S17", "Which channel sold the most units overall, and how many?", "descriptive", ["units"], ["channel"], ["descriptive"], _B, ["bar"], _s17),
    ("S18", "Who are the top 3 sales reps by units sold?", "descriptive", ["units"], ["rep_id"], ["descriptive"], _B, ["bar", "horizontal_bar"], _s18),
    ("S19", "Which month of 2023 had the highest revenue? Give the month number and the revenue", "descriptive", ["revenue"], ["month"], ["descriptive", "time_series"], _B, ["bar", "line"], _s19),
    ("S20", "Which calendar quarter has the highest total revenue across all years, and how much?", "descriptive", ["revenue"], ["quarter"], ["descriptive", "time_series"], _B, ["bar"], _s20),
    ("S21", "What is the total revenue of Gadget Pro in the West region?", "descriptive", ["revenue"], ["product", "region"], ["descriptive"], _B, [], _s21),
    ("S22", "How many Service Pack units were sold through the Partner channel?", "descriptive", ["units"], ["product", "channel"], ["descriptive"], _B, [], _s22),
    ("S23", "What was the Central region's revenue in Q4 2023?", "descriptive", ["revenue"], ["region", "quarter"], ["descriptive", "time_series"], _B, [], _s23),
    ("S24", "What is the average revenue per transaction for Gadget Lite in 2024?", "descriptive", ["revenue"], ["product", "year"], ["descriptive"], _B, [], _s24),
    ("S25", "How many transactions did the North region record in 2022 versus 2024?", "comparative", ["transactions"], ["region", "year"], ["comparative"], _BC, ["bar"], _s25),
    ("S26", "What share of total revenue comes from the North region?", "descriptive", ["revenue"], ["region"], ["descriptive"], _B, ["bar"], _s26),
    ("S27", "How much did total revenue grow from 2022 to 2024, in percent?", "comparative", ["revenue"], ["year"], ["comparative", "time_series"], _BC, ["line", "bar"], _s27),
    ("S28", "What was the average monthly revenue in 2023 and in 2024?", "comparative", ["revenue"], ["year"], ["comparative", "time_series"], _BC, ["bar"], _s28),
]
assert len(SPEC) == 28


def entry(spec: tuple, sales: pd.DataFrame) -> dict:
    cid, query, qtype, metrics, dims, tags, tools, charts, fn = spec
    return {
        "id": cid, "query": query, "query_type": qtype,
        "expected_metrics": metrics, "expected_dimensions": dims,
        "ground_truth": fn(sales), "tags": tags,
        "data_source": {"type": "csv", "path": "data/demo/sales_data.csv"},
        "expected_tools": tools, "expected_chart_types": charts, "split": "dev",
    }


def main() -> int:
    import compute_ground_truth as cg
    sales = cg._Data().sales
    path = Path("tests/eval/test_suite.json")
    suite = json.loads(path.read_text())
    have = {c["id"] for c in suite}
    new = [entry(sp, sales) for sp in SPEC if sp[0] not in have]
    path.write_text(json.dumps(suite + new, indent=2) + "\n")
    print(f"appended {len(new)} case(s) (split: dev)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
