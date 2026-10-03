"""Discovery Market – analysis module.

Computes summary statistics from events and ledger rows.
Produces summary.json with verdicts on H1, H2, H3.

NOTE: All probabilities used in this simulation are stand-in values
derived from published benchmark scores, not measured market outcomes.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from data_loader import load_table2
from runner import PROB_SOURCES


def compute_final_balances(events: list[dict]) -> dict[str, float]:
    """Compute final balance for every account from events."""
    balances: dict[str, float] = defaultdict(float)

    for e in events:
        if "from" in e and "to" in e and "amount" in e:
            balances[e["from"]] -= e["amount"]
            balances[e["to"]] += e["amount"]

    return dict(balances)


def compute_balances_over_time(events: list[dict]) -> dict[str, list[tuple[int, float]]]:
    """Compute running balance for each account at each tick."""
    balances: dict[str, float] = defaultdict(float)
    history: dict[str, list[tuple[int, float]]] = defaultdict(list)

    # Group events by tick
    max_tick = max(e["tick"] for e in events) if events else 0
    events_by_tick: dict[int, list[dict]] = defaultdict(list)
    for e in events:
        events_by_tick[e["tick"]].append(e)

    for tick in range(max_tick + 1):
        for e in events_by_tick.get(tick, []):
            if "from" in e and "to" in e and "amount" in e:
                balances[e["from"]] -= e["amount"]
                balances[e["to"]] += e["amount"]
        for acct in list(balances.keys()):
            history[acct].append((tick, balances[acct]))

    return dict(history)


def find_worlds_solved(events: list[dict]) -> dict[str, dict]:
    """Find which worlds were solved, by whom, and at what tick."""
    solved = {}
    for e in events:
        if e["type"] == "attempt_passed":
            solved[e["world"]] = {
                "agent": e["agent"],
                "tick": e["tick"],
            }
    return solved


def compute_run_summary(events: list[dict], ledger: list[dict]) -> dict[str, Any]:
    """Compute summary for a single run."""
    balances = compute_final_balances(events)
    solved = find_worlds_solved(events)
    worlds_refunded = [e["world"] for e in events if e["type"] == "prize_refunded"]

    # Lab revenue
    lab_revenue = balances.get("lab", 0)

    # All worlds
    all_worlds = set()
    for e in events:
        if e.get("world"):
            all_worlds.add(e["world"])

    return {
        "final_balances": balances,
        "worlds_solved": solved,
        "worlds_unsolved": sorted(all_worlds - set(solved.keys())),
        "worlds_refunded": worlds_refunded,
        "lab_revenue": lab_revenue,
        "refunded_share": len(worlds_refunded) / len(all_worlds) if all_worlds else 0,
        "ledger_size": len(ledger),
        "num_events": len(events),
    }


def group_events_by_run(all_events: list[dict]) -> dict[str, list[dict]]:
    """Group events by run_id."""
    by_run: dict[str, list[dict]] = defaultdict(list)
    for e in all_events:
        by_run[e["run_id"]].append(e)
    return dict(by_run)


def group_ledger_by_run(all_ledger: list[dict]) -> dict[str, list[dict]]:
    """Group ledger rows by run_id (from provenance)."""
    by_run: dict[str, list[dict]] = defaultdict(list)
    for r in all_ledger:
        run_id = r["provenance"]["run_id"]
        by_run[run_id].append(r)
    return dict(by_run)


def parse_run_id(run_id: str) -> dict[str, Any]:
    """Parse a run_id into its components."""
    parts = run_id.split("_")
    result: dict[str, Any] = {"track": parts[1].upper()}

    if result["track"] == "A":
        # track_a_{source}_prize{N}_seed{N}
        result["probability_source"] = parts[2]
        result["prize"] = int(parts[3].replace("prize", ""))
        result["seed"] = int(parts[4].replace("seed", ""))
    elif result["track"] == "C":
        # track_c_price{N}_prize{N}_seed{N}
        result["price"] = float(parts[2].replace("price", ""))
        result["prize"] = int(parts[3].replace("prize", ""))
        result["seed"] = int(parts[4].replace("seed", ""))

    return result


def compute_clearing_prizes(
    all_events: list[dict], worlds: list[str],
    prize_levels: list[float], seeds: list[int],
    track: str = "A", probability_source: str | None = None,
    price: float | None = None,
) -> dict[str, float | None]:
    """Find clearing prize: lowest prize at which a world is solved in ≥3 of 5 seeds.

    Returns dict: world -> clearing_prize (or None if never clears).
    """
    by_run = group_events_by_run(all_events)
    clearing: dict[str, float | None] = {}

    for world in worlds:
        clearing[world] = None
        for prize in sorted(prize_levels):
            solved_count = 0
            for seed in seeds:
                if track == "A":
                    run_id = f"track_a_{probability_source}_prize{int(prize)}_seed{seed}"
                else:
                    run_id = f"track_c_price{price}_prize{int(prize)}_seed{seed}"

                run_events = by_run.get(run_id, [])
                solved = find_worlds_solved(run_events)
                if world in solved:
                    solved_count += 1

            if solved_count >= 3:
                clearing[world] = prize
                break

    return clearing


def compute_funding(events: list[dict]) -> dict[str, float]:
    """Opening balance per account, from tick-0 account_funded events."""
    return {
        e["to"]: e["amount"] for e in events if e["type"] == "account_funded"
    }


def compute_agent_profits(
    all_events: list[dict], seeds: list[int],
    run_id_template: str,
) -> dict[str, dict[str, float]]:
    """Compute mean and std of final profit for each agent across seeds.

    Profit = final balance - opening balance. Agents that never bid are
    included with profit 0.

    run_id_template: template with {seed} placeholder.
    """
    by_run = group_events_by_run(all_events)
    agent_profits: dict[str, list[float]] = defaultdict(list)

    for seed in seeds:
        run_id = run_id_template.format(seed=seed)
        run_events = by_run.get(run_id, [])
        if not run_events:
            continue
        balances = compute_final_balances(run_events)
        for acct, funded in compute_funding(run_events).items():
            if acct.startswith("agent:"):
                agent = acct.replace("agent:", "")
                agent_profits[agent].append(balances[acct] - funded)

    result: dict[str, dict[str, float]] = {}
    for agent, profits in agent_profits.items():
        result[agent] = {
            "mean_profit": float(np.mean(profits)),
            "std_profit": float(np.std(profits)),
            "min_profit": float(np.min(profits)),
            "max_profit": float(np.max(profits)),
            "num_seeds": len(profits),
        }
    return result


def compute_agent_bid_counts(
    all_events: list[dict], seeds: list[int],
    run_id_template: str,
) -> dict[str, float]:
    """Mean bid count per agent across seeds."""
    by_run = group_events_by_run(all_events)
    agent_bids: dict[str, list[int]] = defaultdict(list)

    for seed in seeds:
        run_id = run_id_template.format(seed=seed)
        run_events = by_run.get(run_id, [])
        if not run_events:
            continue
        # Start every funded agent at 0 so non-bidders count as 0 bids
        counts: dict[str, int] = {
            acct.replace("agent:", ""): 0
            for acct in compute_funding(run_events) if acct.startswith("agent:")
        }
        for e in run_events:
            if e["type"] == "bid_placed":
                counts[e["agent"]] += 1
        for agent, count in counts.items():
            agent_bids[agent].append(count)

    return {a: float(np.mean(bids)) for a, bids in agent_bids.items()}


def build_full_summary(
    all_events: list[dict], all_ledger: list[dict]
) -> dict[str, Any]:
    """Build the complete summary.json."""
    by_run_events = group_events_by_run(all_events)
    by_run_ledger = group_ledger_by_run(all_ledger)

    seeds = [0, 1, 2, 3, 4]

    # Per-run summaries
    run_summaries = {}
    for run_id, events in by_run_events.items():
        ledger = by_run_ledger.get(run_id, [])
        run_summaries[run_id] = compute_run_summary(events, ledger)

    # ===== TRACK A ANALYSIS =====
    track_a_worlds = [
        "gravity", "yukawa", "hubble", "ether", "oscillator",
        "coulomb", "circle", "extra_dimensions", "fractional",
        "dark_matter", "three_species",
    ]
    prize_sweep_a = [5, 20, 50, 100, 200]

    # H1: Do stronger agents end in profit while weaker agents stop bidding?
    h1_results = {}
    for source in PROB_SOURCES:
        h1_results[source] = {}
        for prize in prize_sweep_a:
            template = f"track_a_{source}_prize{prize}_seed{{seed}}"
            profits = compute_agent_profits(all_events, seeds, template)
            bids = compute_agent_bid_counts(all_events, seeds, template)
            h1_results[source][str(prize)] = {
                "profits": profits,
                "mean_bids": bids,
            }

    # H2: Clearing prizes
    h2_results = {}
    for source in PROB_SOURCES:
        clearing = compute_clearing_prizes(
            all_events, track_a_worlds, prize_sweep_a, seeds,
            track="A", probability_source=source,
        )
        h2_results[source] = {
            w: (cp if cp is not None else "never") for w, cp in clearing.items()
        }

    # ===== TRACK C ANALYSIS =====
    track_c_worlds = [
        "gravity", "yukawa", "coulomb", "oscillator",
        "fractional", "extra_dimensions",
    ]
    prize_sweep_c = [20, 50, 100]
    price_sweep_c = [0.25, 0.5, 1, 2]

    # H3: At equal accuracy, does the agent needing fewer experiments earn more?
    h3_results = {}
    for price in price_sweep_c:
        h3_results[str(price)] = {}
        for prize in prize_sweep_c:
            template = f"track_c_price{price}_prize{prize}_seed{{seed}}"
            profits = compute_agent_profits(all_events, seeds, template)
            bids = compute_agent_bid_counts(all_events, seeds, template)
            h3_results[str(price)][str(prize)] = {
                "profits": profits,
                "mean_bids": bids,
            }

    # Build verdicts
    table2 = load_table2()
    strength_order = list(
        table2["pass_at_1"].sort_values(ascending=False).index
    )
    verdicts = _build_verdicts(h1_results, h2_results, h3_results,
                               strength_order)

    summary = {
        "caveat": (
            "All probabilities used are stand-in values derived from published "
            "benchmark scores, not measured market outcomes."
        ),
        "run_summaries": run_summaries,
        "h1_analysis": h1_results,
        "h2_clearing_prizes": h2_results,
        "h3_analysis": h3_results,
        "verdicts": verdicts,
    }

    return summary


def _h1_verdict(source: str, h1_data: dict, strength_order: list[str]) -> str:
    """H1: do stronger agents end in profit while weaker agents stop bidding?

    Agents are ranked by Table 2 pass@1; the top half are "stronger", the
    bottom half "weaker". Per prize level:
      SUPPORTED      every stronger agent has mean profit > 0 and no weaker
                     agent places a bid
      PARTIAL        stronger agents profit, but weaker agents bid and lose
      NOT SUPPORTED  a stronger agent does not profit, or a weaker agent profits
    """
    half = len(strength_order) // 2
    strong, weak = strength_order[:half], strength_order[half:]
    lines = [
        f"H1 verdict ({source} odds):",
        f"  stronger (by pass@1): {', '.join(strong)}; "
        f"weaker: {', '.join(weak)}",
    ]

    supported, partial = [], []
    weak_ever_bid = False
    for prize_str in sorted(h1_data.keys(), key=int):
        profits = h1_data[prize_str].get("profits", {})
        bids = h1_data[prize_str].get("mean_bids", {})

        def profit(a: str) -> float:
            return profits.get(a, {}).get("mean_profit", 0.0)

        def std(a: str) -> float:
            return profits.get(a, {}).get("std_profit", 0.0)

        agent_strs = [
            f"{a}={profit(a):.1f}(±{std(a):.1f}, bids={bids.get(a, 0):.1f})"
            for a in strength_order
        ]

        if not any(bids.get(a, 0) > 0 for a in strength_order):
            lines.append(f"  prize={prize_str}: no agents bid → no market")
            continue

        weak_bid = any(bids.get(a, 0) > 0 for a in weak)
        weak_ever_bid |= weak_bid
        if not all(profit(a) > 0 for a in strong) or any(profit(a) > 0 for a in weak):
            status = "NOT SUPPORTED"
        elif weak_bid:
            status = "PARTIAL"
            partial.append(prize_str)
        else:
            status = "SUPPORTED"
            supported.append(prize_str)
        lines.append(f"  prize={prize_str}: {', '.join(agent_strs)} → {status}")

    if supported:
        lines.append(f"  → SUPPORTED at prize={', '.join(supported)}")
    elif partial:
        lines.append(f"  → PARTIAL at prize={', '.join(partial)}")
    else:
        lines.append("  → NOT SUPPORTED at any prize level")
    if not weak_ever_bid:
        lines.append(
            f"  Note: weaker agents ({', '.join(weak)}) never placed a bid at any "
            "prize; their initial belief prices them out from tick 1, so "
            "\"stop bidding\" reflects the prior, not learning from losses."
        )
    return "\n".join(lines)


def _build_verdicts(h1_results, h2_results, h3_results,
                    strength_order: list[str]) -> dict[str, Any]:
    """Build verdict strings from analysis results."""
    verdicts = {}

    # H1 verdict (per probability source)
    for source in PROB_SOURCES:
        verdicts[f"h1_{source}"] = _h1_verdict(
            source, h1_results.get(source, {}), strength_order
        )

    # H2 verdict
    for source in PROB_SOURCES:
        key = f"h2_{source}"
        clearing = h2_results.get(source, {})
        lines = [f"H2 verdict ({source} odds):"]
        for world, cp in clearing.items():
            lines.append(f"  {world}: clearing_prize={cp}")

        unsolved = [w for w, cp in clearing.items() if cp == "never"]
        if unsolved:
            lines.append(f"  Unsolved at any prize: {', '.join(unsolved)}")
        verdicts[key] = "\n".join(lines)

    # H3 verdict
    lines = ["H3 verdict (Track C):"]
    # Compare mda (8 exp) vs llm_opus_unthrottled (41 exp) at same accuracy
    for price in h3_results:
        for prize in h3_results[price]:
            profits = h3_results[price][prize].get("profits", {})
            mda_profit = profits.get("mda", {}).get("mean_profit", None)
            unthrottled_profit = profits.get("llm_opus_unthrottled", {}).get("mean_profit", None)
            llm_profit = profits.get("llm_opus", {}).get("mean_profit", None)

            if mda_profit is not None and unthrottled_profit is not None:
                llm_str = f"{llm_profit:.1f}" if llm_profit is not None else "N/A"
                lines.append(
                    f"  price={price}, prize={prize}: "
                    f"mda(8exp)={mda_profit:.1f}, "
                    f"llm_opus(8exp)={llm_str}, "
                    f"unthrottled(41exp)={unthrottled_profit:.1f}"
                )

    # Overall H3 verdict
    # Look at price=1, prize=50 as representative
    rep = h3_results.get("1", {}).get("50", {}).get("profits", {})
    mda_p = rep.get("mda", {}).get("mean_profit", 0)
    unthrottled_p = rep.get("llm_opus_unthrottled", {}).get("mean_profit", 0)
    if mda_p > unthrottled_p:
        lines.append(
            f"  → SUPPORTED: mda (8 experiments, profit={mda_p:.1f}) "
            f"earns more than unthrottled (41 experiments, profit={unthrottled_p:.1f}) "
            "at equal accuracy (ASSUMPTION: same pass rates)."
        )
    else:
        lines.append(
            f"  → NOT SUPPORTED at price=1, prize=50: "
            f"mda={mda_p:.1f}, unthrottled={unthrottled_p:.1f}"
        )

    verdicts["h3"] = "\n".join(lines)

    return verdicts


def save_summary(summary: dict, output_dir: Path) -> None:
    """Save summary.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.json"
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp, path)
    print(f"Saved summary to {output_dir / 'summary.json'}")
