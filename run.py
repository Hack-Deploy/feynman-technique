#!/usr/bin/env python3
"""Discovery Market – main entry point.

Runs all simulations (Track A and Track C sweeps), produces:
  output/events.json   – all market events
  output/ledger.json   – all ledger rows
  output/summary.json  – analysis with H1, H2, H3 verdicts

NOTE: All probabilities used are stand-in values derived from published
benchmark scores, not measured market outcomes.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from runner import run_all_track_a, run_all_track_c, save_results
from analysis import build_full_summary, save_summary

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def main():
    print("=" * 60)
    print("Discovery Market – Simulation Runner")
    print("=" * 60)
    print()
    print("CAVEAT: All probabilities used are stand-in values derived")
    print("from published benchmark scores, not measured market outcomes.")
    print()

    # Phase 2: Track A runs
    print("Phase 2: Running Track A sweeps...")
    print("  Agents: opus-4.7, gpt-5.5, sonnet-4.6, qwen3.5-397b")
    print("  Prizes: 5, 20, 50, 100, 200")
    print("  Sources: raw, calibrated, power")
    print("  Seeds: 0-4")
    results_a = run_all_track_a()
    print(f"  → {len(results_a)} Track A runs completed.")

    # Phase 3: Track C runs
    print("\nPhase 3: Running Track C sweeps...")
    print("  Agents: mda, llm_opus, llm_opus_unthrottled")
    print("  Prices: 0.25, 0.5, 1, 2")
    print("  Prizes: 20, 50, 100")
    print("  Seeds: 0-4")
    results_c = run_all_track_c()
    print(f"  → {len(results_c)} Track C runs completed.")

    # Merge all results
    all_results = {}
    all_results.update(results_a)
    all_results.update(results_c)

    # Save events and ledger
    print("\nSaving events and ledger...")
    save_results(all_results)

    # Phase 4: Analysis
    print("\nPhase 4: Computing analysis and summary...")

    # Reload from saved files for consistency
    with open(OUTPUT_DIR / "events.json") as f:
        all_events = json.load(f)
    with open(OUTPUT_DIR / "ledger.json") as f:
        all_ledger = json.load(f)

    summary = build_full_summary(all_events, all_ledger)
    save_summary(summary, OUTPUT_DIR)

    # Print verdicts
    print("\n" + "=" * 60)
    print("VERDICTS")
    print("=" * 60)
    for key, verdict in summary["verdicts"].items():
        print(f"\n{verdict}")

    print("\n" + "=" * 60)
    print("Done. Outputs in output/")
    print("=" * 60)


if __name__ == "__main__":
    main()
