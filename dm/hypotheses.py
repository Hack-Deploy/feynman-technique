"""H1–H4 analyses derived from replay event logs."""

from __future__ import annotations

import dataclasses
from collections import Counter, defaultdict
from typing import Any

from dm.calibration import calibration_summary
from dm.types import AttemptRecord

QUESTIONS = {
    "H1": "Do stronger agents end in profit while weaker agents stop bidding?",
    "H2": "What prize clears each world?",
    "H3": "At equal accuracy, does needing fewer experiments earn more?",
    "H4": "Are the AI scientists' stated chances calibrated?",
}


def strength_split(records: list[AttemptRecord]) -> dict[str, Any]:
    outcomes: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        outcomes[record.solver].append(record.passed)
    pass_rates = {
        solver: sum(passed) / len(passed)
        for solver, passed in outcomes.items()
    }
    ordered = sorted(pass_rates, key=lambda solver: (-pass_rates[solver], solver))
    if len(ordered) < 2:
        return {
            "pass_rates": pass_rates,
            "ordered": ordered,
            "strong": [],
            "weak": [],
            "gap": None,
        }
    gaps = [pass_rates[ordered[i]] - pass_rates[ordered[i + 1]]
            for i in range(len(ordered) - 1)]
    split_at = max(range(len(gaps)), key=gaps.__getitem__)
    return {
        "pass_rates": pass_rates,
        "ordered": ordered,
        "strong": ordered[:split_at + 1],
        "weak": ordered[split_at + 1:],
        "gap": gaps[split_at],
    }


def _unavailable(question: str, reason: str, evidence: dict | None = None) -> dict:
    return {
        "question": question,
        "verdict": "UNAVAILABLE",
        "reason": reason,
        "evidence": evidence or {},
    }


def _h1(events: list[dict], records: list[AttemptRecord], label: str,
        prizes: list[float], seeds: list[int]) -> dict:
    from dm.replay import profit_and_bids

    split = strength_split(records)
    rates = split["pass_rates"]
    strength_evidence = {
        "strength": "each solver's pass rate over its records in this pool",
        "pass_rates": rates,
        "strong": split["strong"],
        "weak": split["weak"],
        "gap": split["gap"],
    }
    if len(split["ordered"]) < 2 or split["gap"] < 0.10:
        reason = (f"solvers are about equally strong in this pool: pass rates {rates}")
        return _unavailable(QUESTIONS["H1"], reason, strength_evidence)

    measures = profit_and_bids(events, label, prizes, seeds)
    statuses: list[str] = []
    per_prize: dict[int, dict] = {}
    for prize in prizes:
        agents = measures.get(int(prize), {})
        profits = {agent: values["profit"]["mean"] for agent, values in agents.items()}
        bids = {agent: values["bids"]["mean"] for agent, values in agents.items()}
        if not any(value > 0 for value in bids.values()):
            status = "no market"
        elif any(profits.get(agent, 0.0) <= 0 for agent in split["strong"]) or any(
                profits.get(agent, 0.0) > 0 for agent in split["weak"]):
            status = "NOT_SUPPORTED"
        elif any(bids.get(agent, 0.0) > 0 for agent in split["weak"]):
            status = "PARTIAL"
        else:
            status = "SUPPORTED"
        per_prize[int(prize)] = {"status": status, "profit": profits, "bids": bids}
        if status != "no market":
            statuses.append(status)
    evidence = {**strength_evidence, "per_prize": per_prize}
    if not statuses:
        return _unavailable(QUESTIONS["H1"], "no prize had any bids, so there was no market",
                            evidence)
    verdict = ("SUPPORTED" if "SUPPORTED" in statuses else
               "PARTIAL" if "PARTIAL" in statuses else "NOT_SUPPORTED")
    reason = f"{verdict}: per-prize outcomes are shown in evidence."
    return {"question": QUESTIONS["H1"], "verdict": verdict,
            "reason": reason, "evidence": evidence}


def _h2(events: list[dict], records: list[AttemptRecord], label: str,
        prizes: list[float], seeds: list[int]) -> dict:
    from dm.replay import clearing_prizes, solved_counts

    runs = {event.get("run_id") for event in events if event.get("run_id")}
    if not runs:
        return _unavailable(QUESTIONS["H2"], "event log contains no replay runs")
    worlds = sorted({record.world for record in records})
    clearing = clearing_prizes(events, worlds, label, prizes, seeds)
    never = [world for world, prize in clearing.items() if prize == "never"]
    counts = solved_counts(events, worlds, label, prizes, seeds)
    seed_dependent = {
        world: {prize: count for prize, count in cells.items()
                if 0 < count < len(seeds)}
        for world, cells in counts.items()
    }
    seed_dependent = {world: values for world, values in seed_dependent.items() if values}
    cells: dict[tuple[str, str], int] = Counter((r.solver, r.world) for r in records)
    one_attempt = bool(cells) and all(count == 1 for count in cells.values())
    reason = ("Clearing prizes use the >= 3 of 5 seeds rule; it is near-deterministic "
              "(outcomes identical across seeds, only bid order varies)."
              if one_attempt else
              "Clearing prizes use the >= 3 of 5 seeds rule; outcomes may vary across seeds.")
    evidence = {
        "clearing_prizes": clearing,
        "never": never,
        "seed_dependent_cells": seed_dependent,
        "one_attempt_per_cell": one_attempt,
    }
    return {"question": QUESTIONS["H2"], "verdict": "MEASURED",
            "reason": reason, "evidence": evidence}


def _h3(events: list[dict], label: str, prizes: list[float],
        seeds: list[int], charge_event: str) -> dict:
    if charge_event != "experiment_charged":
        return _unavailable(
            QUESTIONS["H3"],
            "pool charges rounds, not experiments; ARA experiment counts are not exact (PLAN C7)",
            {"charge_event": charge_event},
        )
    charges = [event for event in events if event.get("type") == charge_event]
    if not charges:
        return _unavailable(QUESTIONS["H3"], "event log has no experiment charge events")
    if any(isinstance(event.get("count"), bool) or
           not isinstance(event.get("count"), int) for event in charges):
        return _unavailable(
            QUESTIONS["H3"],
            "experiment charge events do not all carry an integer count",
            {"n_charge_events": len(charges)},
        )

    starts: dict[str, dict[str, None]] = defaultdict(dict)
    passed: dict[tuple[str, str], bool] = {}
    experiment_counts: dict[tuple[str, str], int] = {}
    replay_attempt_events: Counter[str] = Counter()
    for event in events:
        agent = event.get("agent")
        if event.get("type") == "attempt_started" and agent is not None:
            attempt_id = event.get("attempt_id")
            if attempt_id is None:
                return _unavailable(
                    QUESTIONS["H3"], "attempt_started events lack source attempt ids")
            starts[agent].setdefault(attempt_id, None)
            replay_attempt_events[agent] += 1
        elif event.get("type") == "verdict_issued" and agent is not None:
            attempt_id = event.get("attempt_id")
            if attempt_id is not None:
                passed.setdefault((agent, attempt_id),
                                  (event.get("detail") or {}).get("passed") is True)
        elif event.get("type") == charge_event and agent is not None:
            attempt_id = event.get("attempt_id")
            if attempt_id is not None:
                experiment_counts.setdefault((agent, attempt_id), event["count"])
    if any((agent, attempt_id) not in passed or
           (agent, attempt_id) not in experiment_counts
           for agent, attempt_ids in starts.items() for attempt_id in attempt_ids):
        return _unavailable(
            QUESTIONS["H3"],
            "started source attempts are missing a verdict or experiment charge",
        )
    per_agent = {
        agent: {
            "pass_rate": (sum(passed[(agent, attempt_id)] for attempt_id in attempt_ids) /
                          len(attempt_ids)),
            "experiments_per_attempt": (
                sum(experiment_counts[(agent, attempt_id)] for attempt_id in attempt_ids) /
                len(attempt_ids)),
            "attempts": len(attempt_ids),
            "attempt_started_events": replay_attempt_events[agent],
        }
        for agent, attempt_ids in sorted(starts.items()) if attempt_ids
    }
    if not per_agent:
        return _unavailable(QUESTIONS["H3"], "event log contains no started attempts")

    eligible = []
    agents = sorted(per_agent)
    for index, first in enumerate(agents):
        for second in agents[index + 1:]:
            a, b = per_agent[first], per_agent[second]
            if (abs(a["pass_rate"] - b["pass_rate"]) <= 0.10 and
                    abs(a["experiments_per_attempt"] -
                        b["experiments_per_attempt"]) >= 0.5):
                cheaper, dearer = (
                    (first, second) if a["experiments_per_attempt"] <
                    b["experiments_per_attempt"] else (second, first)
                )
                eligible.append((cheaper, dearer))
    if not eligible:
        return _unavailable(QUESTIONS["H3"], "no agent pair meets the accuracy and effort criteria",
                            {"per_agent": per_agent, "pairs": []})

    from dm.replay import profit_and_bids

    profit_data = profit_and_bids(events, label, prizes, seeds)
    pair_evidence = []
    results: list[bool] = []
    for cheaper, dearer in eligible:
        per_prize = {}
        for prize in prizes:
            metrics = profit_data.get(int(prize), {})
            cheaper_values = metrics.get(cheaper)
            dearer_values = metrics.get(dearer)
            if (cheaper_values is None or dearer_values is None or
                    cheaper_values["bids"]["mean"] <= 0 or
                    dearer_values["bids"]["mean"] <= 0):
                continue
            cheaper_profit = cheaper_values["profit"]["mean"]
            dearer_profit = dearer_values["profit"]["mean"]
            fewer_wins = cheaper_profit > dearer_profit
            per_prize[int(prize)] = {
                "fewer_experiments_profit": cheaper_profit,
                "more_experiments_profit": dearer_profit,
                "fewer_experiments_higher": fewer_wins,
            }
            results.append(fewer_wins)
        pair_evidence.append({
            "fewer_experiments": cheaper,
            "more_experiments": dearer,
            "per_prize": per_prize,
        })
    if not results:
        return _unavailable(QUESTIONS["H3"], "eligible agents had no compared prize with bids",
                            {"per_agent": per_agent, "pairs": pair_evidence})
    verdict = ("SUPPORTED" if all(results) else
               "NOT_SUPPORTED" if not any(results) else "PARTIAL")
    return {
        "question": QUESTIONS["H3"],
        "verdict": verdict,
        "reason": f"{verdict}: compared profits are shown for eligible pairs.",
        "evidence": {"per_agent": per_agent, "pairs": pair_evidence},
    }


def _h4(events: list[dict], records: list[AttemptRecord]) -> dict:
    belief_count = sum(
        (event.get("detail") or {}).get("p_source") == "belief"
        for event in events if event.get("type") == "confidence_stated"
    )
    confidences: dict[str, dict] = {}
    for event in events:
        detail = event.get("detail") or {}
        if (event.get("type") == "confidence_stated" and
                detail.get("p_source") == "solver" and
                detail.get("market_attempt_id")):
            confidences.setdefault(detail["market_attempt_id"], event)
    verdicts = {
        (event.get("detail") or {}).get("market_attempt_id"): event
        for event in events if event.get("type") == "verdict_issued"
        if (event.get("detail") or {}).get("market_attempt_id")
    }
    records_by_id = {record.attempt_id: record for record in records}
    selected: dict[str, AttemptRecord] = {}
    for market_id, confidence in confidences.items():
        source_id = confidence.get("attempt_id")
        verdict_event = verdicts.get(market_id)
        record = records_by_id.get(source_id)
        if record is None or verdict_event is None or confidence.get("p") is None:
            continue
        verdict_detail = verdict_event.get("detail") or {}
        selected.setdefault(source_id, dataclasses.replace(
            record,
            stated_p_success=confidence["p"],
            verdict={**record.verdict, "passed": verdict_detail.get("passed") is True},
        ))
    if not selected:
        return _unavailable(
            QUESTIONS["H4"],
            "no stated probabilities in this pool (ARA records carry none); models state their chances on /live",
            {"n_attempts_p_from_belief": belief_count, "n_excluded": 0},
        )

    summary = calibration_summary(selected.values())
    citl = summary["calibration_in_the_large"]
    verdict = "SUPPORTED" if abs(citl) <= 0.10 else "NOT_SUPPORTED"
    metrics = ("n", "brier", "calibration_in_the_large", "mean_stated_p",
               "pass_rate", "reliability", "n_excluded_no_stated_p")
    pooled = {key: summary[key] for key in metrics}
    pooled["n_excluded"] = belief_count
    per_agent = {
        solver: {
            **{key: values[key] for key in metrics},
            "n_excluded": sum(
                event.get("agent") == solver and
                (event.get("detail") or {}).get("p_source") == "belief"
                for event in events if event.get("type") == "confidence_stated"
            ),
        }
        for solver, values in summary["by_solver"].items()
    }
    return {
        "question": QUESTIONS["H4"],
        "verdict": verdict,
        "reason": (f"{verdict}: pooled calibration-in-the-large is {citl:.3f} "
                   f"(threshold ±0.10)."),
        "evidence": {
            "pooled": pooled,
            "per_agent": per_agent,
            "n_excluded": belief_count,
            "n_attempts_p_from_belief": belief_count,
        },
    }


def evaluate_hypotheses(events: list[dict], records: list[AttemptRecord],
                        label: str, prizes: list[float], seeds: list[int],
                        charge_event: str) -> dict[str, dict]:
    """Evaluate the four replay hypotheses from the event log and record pool."""
    return {
        "H1": _h1(events, records, label, prizes, seeds),
        "H2": _h2(events, records, label, prizes, seeds),
        "H3": _h3(events, label, prizes, seeds, charge_event),
        "H4": _h4(events, records),
    }
