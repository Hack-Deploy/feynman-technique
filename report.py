#!/usr/bin/env python3
"""Discovery Market – results dashboard.

Reads output/{events,ledger,summary}.json (running the sweeps first if they
are missing) and writes output/report.html: one self-contained page with
agent balances over time, the prize each world needs, profits per prize and
the public ledger. Open it by double-clicking.

Run: uv run python report.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from runner import PROB_SOURCES

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
AGENTS = ["opus-4.7", "gpt-5.5", "sonnet-4.6", "qwen3.5-397b"]
PRIZES = [5, 20, 50, 100, 200]
SEEDS = [0, 1, 2, 3, 4]
# Worlds the DiscoverPhysics paper describes as needing hidden structure.
HIDDEN_STRUCTURE = ["circle", "extra_dimensions", "dark_matter", "three_species"]


def load_outputs() -> tuple[list, list, dict]:
    if not (OUTPUT_DIR / "summary.json").exists():
        import run
        run.main()
    events = json.loads((OUTPUT_DIR / "events.json").read_text())
    ledger = json.loads((OUTPUT_DIR / "ledger.json").read_text())
    summary = json.loads((OUTPUT_DIR / "summary.json").read_text())
    return events, ledger, summary


def balance_series(events: list[dict]) -> dict[str, dict]:
    """Per Track A run: each agent's balance after every payment it makes or gets.

    Points are [step, balance], where step counts agent payments in order,
    so many payments inside one tick stay visible. `ticks` holds the first
    step of each tick, for gridlines.
    """
    by_run: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        if e["run_id"].startswith("track_a_"):
            by_run[e["run_id"]].append(e)

    series: dict[str, dict] = {}
    for run_id, evs in by_run.items():
        evs.sort(key=lambda e: (e["tick"], e["seq"]))
        bal = {a: 0.0 for a in AGENTS}
        pts: dict[str, list] = {a: [] for a in AGENTS}
        ticks: list[list[int]] = []
        step = 0
        for e in evs:
            amt = e.get("amount")
            if amt is None:
                continue
            touched = False
            for side, sign in (("from", -1), ("to", 1)):
                who = (e.get(side) or "").removeprefix("agent:")
                if who in bal:
                    bal[who] += sign * amt
                    touched = True
            if not touched:
                continue
            if e["type"] != "account_funded":
                step += 1
            if not ticks or ticks[-1][1] != e["tick"]:
                ticks.append([step, e["tick"]])
            for a in AGENTS:
                pts[a].append([step, round(bal[a], 2)])
        series[run_id] = {"points": pts, "ticks": ticks, "steps": step}
    return series


def ledger_rows(ledger: list[dict]) -> dict[str, list]:
    """Per Track A run: [tick disclosed, world, agent, passed, credits]."""
    rows: dict[str, list] = defaultdict(list)
    for r in ledger:
        run_id = r["provenance"]["run_id"]
        if run_id.startswith("track_a_"):
            rows[run_id].append([
                r["disclosed_at_tick"], r["question"], r["solver"],
                r["outcome"]["passed"], r["effort"]["credits"],
            ])
    for v in rows.values():
        v.sort(key=lambda x: (x[0] is None, x[0]))
    return rows


def build_data(events: list, ledger: list, summary: dict) -> dict:
    h1 = {
        src: {
            prize: {a: d["profits"][a]["mean_profit"] for a in AGENTS}
            for prize, d in summary["h1_analysis"][src].items()
        }
        for src in PROB_SOURCES
    }
    return {
        "sources": PROB_SOURCES,
        "agents": AGENTS,
        "prizes": PRIZES,
        "seeds": SEEDS,
        "hidden": HIDDEN_STRUCTURE,
        "balances": balance_series(events),
        "ledger": ledger_rows(ledger),
        "clearing": {s: summary["h2_clearing_prizes"][s] for s in PROB_SOURCES},
        "profits": h1,
        "verdicts": summary["verdicts"],
    }


def main() -> None:
    events, ledger, summary = load_outputs()
    data = build_data(events, ledger, summary)
    template = (Path(__file__).resolve().parent / "report_template.html").read_text()
    html = template.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    out = OUTPUT_DIR / "report.html"
    out.write_text(html)
    print(f"Wrote {out} ({out.stat().st_size // 1024} KB). Open it in a browser.")


if __name__ == "__main__":
    main()
