"""Discovery Market – run orchestrator.

Sets up Track A and Track C runs, writes events and ledger to output/.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_loader import load_table1, load_table2, load_table3, load_config
from market import (
    MarketRun, run_market, TrackACostModel, TrackCCostModel, Event, LedgerRow,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def build_track_a_probs(
    table1: pd.DataFrame, table2: pd.DataFrame, source: str
) -> dict[str, dict[str, float]]:
    """Build true probability tables for Track A agents.

    Sources:
      - raw: p = explanation score (Table 1).
      - calibrated: p = min(1, score * pass_at_1 / mean_score).
    """
    agents = list(table1.columns)
    worlds = list(table1.index)
    probs: dict[str, dict[str, float]] = {}

    for agent in agents:
        probs[agent] = {}
        for world in worlds:
            score = table1.loc[world, agent]
            if source == "raw":
                probs[agent][world] = float(score)
            elif source == "calibrated":
                mean_score = float(table2.loc[agent, "mean_score"])
                pass_at_1 = float(table2.loc[agent, "pass_at_1"])
                p = min(1.0, score * pass_at_1 / mean_score)
                probs[agent][world] = float(p)
            else:
                raise ValueError(f"Unknown probability source: {source}")

    return probs


def build_track_a_beliefs(
    table2: pd.DataFrame, worlds: list[str]
) -> dict[str, dict[str, float]]:
    """Initial belief mean = agent's pass_at_1 for all worlds (Track A)."""
    beliefs: dict[str, dict[str, float]] = {}
    for agent in table2.index:
        beliefs[agent] = {}
        for world in worlds:
            beliefs[agent][world] = float(table2.loc[agent, "pass_at_1"])
    return beliefs


def build_track_c_probs(
    table3: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Build true probability tables for Track C agents.

    mda: from Table 3 'mda' column.
    llm_opus: from Table 3 'llm_opus' column.
    llm_opus_unthrottled: uses Table 3 'mda' column (assumption stated).
    """
    worlds = list(table3.index)
    probs: dict[str, dict[str, float]] = {
        "mda": {},
        "llm_opus": {},
        "llm_opus_unthrottled": {},
    }
    for world in worlds:
        probs["mda"][world] = float(table3.loc[world, "mda"])
        probs["llm_opus"][world] = float(table3.loc[world, "llm_opus"])
        # ASSUMPTION: llm_opus_unthrottled uses mda column values
        probs["llm_opus_unthrottled"][world] = float(table3.loc[world, "mda"])
    return probs


def build_track_c_beliefs(
    table3: pd.DataFrame, agents: list[str]
) -> dict[str, dict[str, float]]:
    """Initial belief mean = agent's mean pass rate over Table 3 worlds (Track C)."""
    worlds = list(table3.index)
    probs = build_track_c_probs(table3)
    beliefs: dict[str, dict[str, float]] = {}
    for agent in agents:
        mean_rate = np.mean([probs[agent][w] for w in worlds])
        beliefs[agent] = {w: float(mean_rate) for w in worlds}
    return beliefs


def run_track_a(
    prize: float,
    probability_source: str,
    seed: int,
    table1: pd.DataFrame | None = None,
    table2: pd.DataFrame | None = None,
    ticks: int = 200,
    starting_credits: float = 100,
    belief_weight: float = 2,
) -> tuple[list[Event], list[LedgerRow]]:
    """Run a single Track A simulation."""
    if table1 is None:
        table1 = load_table1()
    if table2 is None:
        table2 = load_table2()

    worlds = list(table1.index)
    agents = list(table1.columns)
    prizes = {w: prize for w in worlds}
    true_probs = build_track_a_probs(table1, table2, probability_source)
    initial_beliefs = build_track_a_beliefs(table2, worlds)

    cost_model = TrackACostModel()

    run_id = f"track_a_{probability_source}_prize{int(prize)}_seed{seed}"

    cfg = MarketRun(
        run_id=run_id,
        seed=seed,
        ticks=ticks,
        worlds=worlds,
        agents=agents,
        prizes=prizes,
        starting_credits=starting_credits,
        cost_model=cost_model,
        true_probs=true_probs,
        initial_beliefs=initial_beliefs,
        belief_weight=belief_weight,
        track="A",
        probability_source=probability_source,
    )

    return run_market(cfg)


def run_track_c(
    prize: float,
    price_per_experiment: float,
    seed: int,
    table3: pd.DataFrame | None = None,
    ticks: int = 200,
    starting_credits: float = 100,
    belief_weight: float = 2,
) -> tuple[list[Event], list[LedgerRow]]:
    """Run a single Track C simulation."""
    if table3 is None:
        table3 = load_table3()

    worlds = list(table3.index)
    agents = ["mda", "llm_opus", "llm_opus_unthrottled"]
    prizes = {w: prize for w in worlds}
    true_probs = build_track_c_probs(table3)
    initial_beliefs = build_track_c_beliefs(table3, agents)

    experiments_per_attempt = {
        "mda": 8,
        "llm_opus": 8,
        "llm_opus_unthrottled": 41,
    }
    cost_model = TrackCCostModel(experiments_per_attempt, price_per_experiment)

    run_id = f"track_c_price{price_per_experiment}_prize{int(prize)}_seed{seed}"

    cfg = MarketRun(
        run_id=run_id,
        seed=seed,
        ticks=ticks,
        worlds=worlds,
        agents=agents,
        prizes=prizes,
        starting_credits=starting_credits,
        cost_model=cost_model,
        true_probs=true_probs,
        initial_beliefs=initial_beliefs,
        belief_weight=belief_weight,
        track="C",
        probability_source="table3_mda",
    )

    return run_market(cfg)


def run_all_track_a(
    seeds: list[int] | None = None,
    prizes: list[float] | None = None,
    prob_sources: list[str] | None = None,
) -> dict[str, tuple[list[Event], list[LedgerRow]]]:
    """Run all Track A sweep configurations."""
    table1 = load_table1()
    table2 = load_table2()

    if seeds is None:
        seeds = [0, 1, 2, 3, 4]
    if prizes is None:
        prizes = [5, 20, 50, 100, 200]
    if prob_sources is None:
        prob_sources = ["raw", "calibrated"]

    results: dict[str, tuple[list[Event], list[LedgerRow]]] = {}

    for source in prob_sources:
        for prize in prizes:
            for seed in seeds:
                run_id = f"track_a_{source}_prize{int(prize)}_seed{seed}"
                events, ledger = run_track_a(
                    prize=prize,
                    probability_source=source,
                    seed=seed,
                    table1=table1,
                    table2=table2,
                )
                results[run_id] = (events, ledger)

    return results


def run_all_track_c(
    seeds: list[int] | None = None,
    prizes: list[float] | None = None,
    prices: list[float] | None = None,
) -> dict[str, tuple[list[Event], list[LedgerRow]]]:
    """Run all Track C sweep configurations."""
    table3 = load_table3()

    if seeds is None:
        seeds = [0, 1, 2, 3, 4]
    if prizes is None:
        prizes = [20, 50, 100]
    if prices is None:
        prices = [0.25, 0.5, 1, 2]

    results: dict[str, tuple[list[Event], list[LedgerRow]]] = {}

    for price in prices:
        for prize in prizes:
            for seed in seeds:
                run_id = f"track_c_price{price}_prize{int(prize)}_seed{seed}"
                events, ledger = run_track_c(
                    prize=prize,
                    price_per_experiment=price,
                    seed=seed,
                    table3=table3,
                )
                results[run_id] = (events, ledger)

    return results


def save_results(
    results: dict[str, tuple[list[Event], list[LedgerRow]]],
    output_dir: Path | None = None,
) -> None:
    """Save all results to JSON files."""
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    all_events = []
    all_ledger = []

    for run_id, (events, ledger) in results.items():
        for e in events:
            all_events.append(e.to_dict())
        for r in ledger:
            all_ledger.append(r.to_dict())

    with open(out / "events.json", "w") as f:
        json.dump(all_events, f, indent=1)

    with open(out / "ledger.json", "w") as f:
        json.dump(all_ledger, f, indent=1)

    print(f"Saved {len(all_events)} events and {len(all_ledger)} ledger rows to {out}")
