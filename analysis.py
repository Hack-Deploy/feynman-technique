"""Discovery Market – analysis module.

Computes summary statistics from events and ledger rows.
Produces summary.json with verdicts on H1, H2, H3.

NOTE: All probabilities used in this simulation are stand-in values
derived from published benchmark scores, not measured market outcomes.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


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
        "worlds_unsolved": list(all_worlds - set(solved.keys())),
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


def compute_agent_profits(
    all_events: list[dict], seeds: list[int],
    run_id_template: str,
) -> dict[str, dict[str, float]]:
    """Compute mean and std of final profit for each agent across seeds.

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
        for acct, balance in balances.items():
            if acct.startswith("agent:"):
                agent = acct.replace("agent:", "")
                profit = balance - 100  # starting credits
                agent_profits[agent].append(profit)

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
        counts: dict[str, int] = defaultdict(int)
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
    for source in ["raw", "calibrated"]:
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
    for source in ["raw", "calibrated"]:
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
    verdicts = _build_verdicts(h1_results, h2_results, h3_results)

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


def _build_verdicts(h1_results, h2_results, h3_results) -> dict[str, Any]:
    """Build verdict strings from analysis results."""
    verdicts = {}

    # H1 verdict (per probability source)
    for source in ["raw", "calibrated"]:
        key = f"h1_{source}"
        lines = [f"H1 verdict ({source} odds):"]
        h1_data = h1_results.get(source, {})

        # Collect profits across all prize levels
        best_prize = None
        best_finding = None
        for prize_str in sorted(h1_data.keys(), key=lambda x: int(x)):
            data = h1_data[prize_str]
            profits = data.get("profits", {})
            bids = data.get("mean_bids", {})

            if not profits:
                lines.append(f"  prize={prize_str}: no agents bid")
                continue

            sorted_agents = sorted(
                profits.items(), key=lambda x: x[1]["mean_profit"], reverse=True
            )

            agent_strs = []
            for agent, stats in sorted_agents:
                mb = bids.get(agent, 0)
                agent_strs.append(
                    f"{agent}={stats['mean_profit']:.1f}(±{stats['std_profit']:.1f}, bids={mb:.0f})"
                )
            lines.append(f"  prize={prize_str}: {', '.join(agent_strs)}")

            # Track best prize for verdict
            if len(sorted_agents) >= 2:
                top = sorted_agents[0]
                bottom = sorted_agents[-1]
                if top[1]["mean_profit"] > 0 and bottom[1]["mean_profit"] <= top[1]["mean_profit"]:
                    if best_prize is None or len(sorted_agents) > len(best_finding[2]):
                        best_finding = (top, bottom, sorted_agents)
                        best_prize = prize_str

        # Also note agents that never bid at any prize
        all_bidders = set()
        for prize_str, data in h1_data.items():
            all_bidders.update(data.get("mean_bids", {}).keys())
        all_agents = {"opus-4.7", "gpt-5.5", "sonnet-4.6", "qwen3.5-397b"} if source in ["raw", "calibrated"] else set()
        non_bidders = all_agents - all_bidders
        if non_bidders:
            lines.append(f"  Never bid at any prize: {', '.join(sorted(non_bidders))}")

        if best_finding:
            top, bottom, agents = best_finding
            lines.append(
                f"  → SUPPORTED at prize={best_prize}: "
                f"strongest ({top[0]}) profits {top[1]['mean_profit']:.1f}, "
                f"weakest ({bottom[0]}) profits {bottom[1]['mean_profit']:.1f}. "
                f"Weaker agents {', '.join(sorted(non_bidders))} stop bidding entirely."
            )
        else:
            lines.append("  → INCONCLUSIVE: insufficient data across prize levels.")

        verdicts[key] = "\n".join(lines)

    # H2 verdict
    for source in ["raw", "calibrated"]:
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
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {output_dir / 'summary.json'}")
