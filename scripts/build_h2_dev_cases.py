#!/usr/bin/env python3
"""
Write the H2 dev attribution cases (预注册 §7.2.1, 附录 FL) from the PUBLIC frozen
event script (git tag event-script-v2, 附录 FZ): every dev-split event, one case
per `h2_probes` entry (附录 GB). Test-split events are
never read into the output. Makes no API call.

    python scripts/build_h2_dev_cases.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_demo_data import load_frozen_events  # noqa: E402

OUT = Path("tests/eval/h2_dev_attribution_cases.json")


def anomaly_text(event: dict) -> str:
    signal = event.get("data_signal")
    if not signal:
        return ("None in the data: the event below left no measurable effect. "
                f"Event: {event['business_description']}")
    segment = ", ".join(f"{k}={v}" for k, v in (signal.get("segment") or {}).items()) or "all regions"
    return (f"{signal['metric']} {signal['direction']} by about {signal['magnitude']:.0%} "
            f"in {segment}, {event['valid_from']} to {event['valid_to']}")


def build(events: list[dict]) -> list[dict]:
    cases = []
    for event in sorted(events, key=lambda e: e["event_id"]):
        if event["split"] != "dev":
            continue
        cause = event["business_description"] if event["event_type"] == "normal" else None
        metric = (event.get("data_signal") or {}).get("metric", "revenue")
        for n, query in enumerate(event["h2_probes"], 1):
            cases.append({
                "id": f"H2-{event['event_id'].removeprefix('EVT-')}-Q{n}",
                "query": query,
                "query_type": "diagnostic",
                "expected_metrics": [metric],
                "expected_dimensions": [],
                "ground_truth": {},
                "tags": ["h2_attribution", event["event_type"]],
                "data_source": {"type": "csv", "path": "data/demo/sales_data.csv"},
                "split": "dev",
                "event_id": event["event_id"],
                "event_type": event["event_type"],
                "event_anomaly": anomaly_text(event),
                "event_cause": cause,
            })
    return cases


def main() -> int:
    cases = build(load_frozen_events())
    OUT.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {len(cases)} cases to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
