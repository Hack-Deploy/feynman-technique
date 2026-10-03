"""Replay markets: the Track A prize sweep on a pool of real attempts.

    uv run python -m dm.replay ara      # attempts/ara.jsonl → output/replay_ara/

Each run replays ``AttemptRecord``s through ``market.run_market`` with a
``ReplayPool`` as both cost model and outcome source (rounds × 1 credit, like
Track A). Starting beliefs are each solver's leave-one-out pass rate in the pool.
All summaries are recomputed from the event log.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from analysis import compute_final_balances, compute_funding, find_worlds_solved
from dm.outcomes import ReplayPool, rounds_cost
from dm.store import ATTEMPTS_DIR, AttemptStore
from dm.types import AttemptRecord
from market import MarketRun, run_market

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"

PRIZES = [5, 20, 50, 100, 200]
SEEDS = list(range(5))
TICKS = 200
STARTING_CREDITS = 100.0
BELIEF_WEIGHT = 2.0  # same as Track A
MIN_SEEDS_SOLVED = 3

POOLS = {"ara": ATTEMPTS_DIR / "ara.jsonl"}


VERDICTS = ("numeric", "ara")


def with_verdict(records: list[AttemptRecord], verdict: str) -> list[AttemptRecord]:
    """``numeric``: records as imported (nMSE < 0.1). ``ara``: sensitivity run that
    swaps in ARA's own pass flag (nMSE < 0.1 AND explanation >= 0.75)."""
    if verdict == "numeric":
        return records
    if verdict != "ara":
        raise ValueError(f"unknown verdict rule {verdict!r}")
    return [dataclasses.replace(r, verdict={**r.verdict,
                                            "passed": bool(r.extra.get("ara_passed"))})
            for r in records]


def run_id(label: str, prize: float, seed: int) -> str:
    return f"replay_{label}_prize{int(prize)}_seed{seed}"


def make_pool(records: list[AttemptRecord]) -> ReplayPool:
    return ReplayPool(records, rounds_cost(1.0), charge_event="round_charged")


def loo_beliefs(pool: ReplayPool, fallback: float = 0.5) -> dict[str, dict[str, float]]:
    """Leave-one-out prior: a solver's belief about world w is its pass rate on
    the other worlds in the pool (0.5 if it has none)."""
    out: dict[str, dict[str, float]] = {}
    for s in pool.solvers():
        out[s] = {}
        for w in pool.worlds():
            p = pool.pass_rate(s, exclude_world=w)
            out[s][w] = fallback if p is None else p
    return out


def run_sweep(records: list[AttemptRecord], label: str,
              prizes: list[float] = PRIZES, seeds: list[int] = SEEDS,
              ticks: int = TICKS, starting_credits: float = STARTING_CREDITS,
              ) -> tuple[list[dict], list[dict]]:
    pool = make_pool(records)
    worlds, agents = pool.worlds(), pool.solvers()
    beliefs = loo_beliefs(pool)
    events: list[dict] = []
    ledger: list[dict] = []
    for prize in prizes:
        for seed in seeds:
            cfg = MarketRun(
                run_id=run_id(label, prize, seed), seed=seed, ticks=ticks,
                worlds=worlds, agents=agents, prizes={w: float(prize) for w in worlds},
                starting_credits=starting_credits,
                # The same pool must drive the bid rule and the charges, so an
                # agent's affordability check matches what it is charged.
                cost_model=pool, true_probs=None,
                initial_beliefs=beliefs, belief_weight=BELIEF_WEIGHT,
                track=f"replay_{label}", probability_source=f"replay:{label}",
                outcome_source=pool, state_confidence=True,
            )
            ev, led = run_market(cfg)
            events.extend(e.to_dict() for e in ev)
            ledger.extend(r.to_dict() for r in led)
    return events, ledger


# ------------------------------------------------------------- summaries

def _by_run(events: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        out[e["run_id"]].append(e)
    return out


def clearing_prizes(events: list[dict], worlds: list[str], label: str,
                    prizes: list[float] = PRIZES, seeds: list[int] = SEEDS,
                    min_seeds: int = MIN_SEEDS_SOLVED) -> dict[str, Any]:
    """Lowest prize at which a world is solved in ≥ min_seeds seeds, else "never"."""
    by_run = _by_run(events)
    out: dict[str, Any] = {}
    for w in worlds:
        out[w] = "never"
        for p in sorted(prizes):
            n = sum(w in find_worlds_solved(by_run.get(run_id(label, p, s), []))
                    for s in seeds)
            if n >= min_seeds:
                out[w] = int(p)
                break
    return out


def solved_counts(events: list[dict], worlds: list[str], label: str,
                  prizes: list[float] = PRIZES, seeds: list[int] = SEEDS
                  ) -> dict[str, dict[int, int]]:
    """world → prize → number of seeds in which it was solved."""
    by_run = _by_run(events)
    return {w: {int(p): sum(w in find_worlds_solved(by_run.get(run_id(label, p, s), []))
                            for s in seeds) for p in prizes} for w in worlds}


def _mean_sd(xs: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(xs)), "sd": float(np.std(xs)),
            "min": float(np.min(xs)), "max": float(np.max(xs)), "n": len(xs)}


def profit_and_bids(events: list[dict], label: str, prizes: list[float] = PRIZES,
                    seeds: list[int] = SEEDS) -> dict[int, dict[str, dict]]:
    """prize → solver → {profit: mean/sd/…, bids: mean/sd/…, wins: mean}."""
    by_run = _by_run(events)
    out: dict[int, dict[str, dict]] = {}
    for p in prizes:
        profits: dict[str, list[float]] = defaultdict(list)
        bids: dict[str, list[float]] = defaultdict(list)
        wins: dict[str, list[float]] = defaultdict(list)
        for s in seeds:
            ev = by_run.get(run_id(label, p, s), [])
            if not ev:
                continue
            bal = compute_final_balances(ev)
            nb = Counter(e["agent"] for e in ev if e["type"] == "bid_placed")
            nw = Counter(e["agent"] for e in ev if e["type"] == "prize_paid")
            for acct, funded in compute_funding(ev).items():
                if acct.startswith("agent:"):
                    a = acct[len("agent:"):]
                    profits[a].append(bal[acct] - funded)
                    bids[a].append(nb.get(a, 0))
                    wins[a].append(nw.get(a, 0))
        out[int(p)] = {a: {"profit": _mean_sd(profits[a]), "bids": _mean_sd(bids[a]),
                           "wins": float(np.mean(wins[a]))} for a in sorted(profits)}
    return out


def lab_revenue(events: list[dict], label: str, prizes: list[float] = PRIZES,
                seeds: list[int] = SEEDS) -> dict[int, dict[str, float]]:
    by_run = _by_run(events)
    return {int(p): _mean_sd([compute_final_balances(by_run[run_id(label, p, s)])
                              .get("lab", 0.0) for s in seeds
                              if run_id(label, p, s) in by_run])
            for p in prizes}


def max_draws_per_record(events: list[dict]) -> int:
    """Largest number of times any record was drawn within one run (must be 1)."""
    per_run: Counter = Counter((e["run_id"], e["attempt_id"]) for e in events
                               if e["type"] == "attempt_started")
    return max(per_run.values(), default=0)


def max_attempts_per_solver_world(events: list[dict]) -> int:
    per: Counter = Counter((e["run_id"], e["agent"], e["world"]) for e in events
                           if e["type"] == "attempt_started")
    return max(per.values(), default=0)


def summarise(events: list[dict], ledger: list[dict], records: list[AttemptRecord],
              label: str, prizes: list[float] = PRIZES,
              seeds: list[int] = SEEDS) -> dict[str, Any]:
    pool = make_pool(records)
    worlds = pool.worlds()
    counts = solved_counts(events, worlds, label, prizes, seeds)
    seed_dependent = {w: {p: n for p, n in c.items() if 0 < n < len(seeds)}
                      for w, c in counts.items()}
    by_run = _by_run(events)
    return {
        "pool": label,
        "caveat": (
            "One attempt per (solver, world): within a run each solver can try each "
            "world at most once, and a world's outcome for a given solver is identical "
            "in every seed (only bid order changes). The '>= 3 of 5 seeds' clearing "
            "rule is therefore close to deterministic."),
        "config": {"prizes": prizes, "seeds": seeds, "ticks": TICKS,
                   "starting_credits": STARTING_CREDITS, "belief_weight": BELIEF_WEIGHT,
                   "cost": "rounds x 1 credit (round_charged)",
                   "beliefs": "leave-one-out pass rate per solver, 0.5 fallback",
                   "clearing_rule": f"lowest prize solved in >= {MIN_SEEDS_SOLVED} of "
                                    f"{len(seeds)} seeds, else 'never'"},
        "n_records": len(records),
        "solvers": pool.solvers(),
        "worlds": worlds,
        "initial_beliefs": loo_beliefs(pool),
        "record_outcomes": {f"{r.solver}/{r.world}": {"passed": r.passed,
                                                     "rounds": r.rounds}
                            for r in sorted(records, key=lambda r: (r.solver, r.world))},
        "clearing_prizes": clearing_prizes(events, worlds, label, prizes, seeds),
        "solved_seed_counts": counts,
        "seed_dependent_cells": {w: c for w, c in seed_dependent.items() if c},
        "profit_and_bids": profit_and_bids(events, label, prizes, seeds),
        "lab_revenue": lab_revenue(events, label, prizes, seeds),
        "checks": {
            "credit_sum_per_run_max_abs": max(
                abs(sum(compute_final_balances(ev).values())) for ev in by_run.values()),
            "max_draws_per_record_per_run": max_draws_per_record(events),
            "max_attempts_per_solver_world_per_run": max_attempts_per_solver_world(events),
            "n_runs": len(by_run), "n_events": len(events), "n_ledger_rows": len(ledger),
        },
    }


def run_and_save(label: str, out_dir: Path | None = None,
                 store_path: Path | None = None, verdict: str = "numeric") -> dict[str, Any]:
    path = store_path or POOLS[label]
    records = with_verdict(AttemptStore(path).load(), verdict)
    if not records:
        raise SystemExit(f"no records in {path}; run the importer first "
                         f"(uv run python -m dm.importers.{label})")
    run_label = label if verdict == "numeric" else f"{label}_{verdict}verdict"
    events, ledger = run_sweep(records, run_label)
    summary = summarise(events, ledger, records, run_label)
    summary["verdict_rule"] = verdict
    out = out_dir or OUTPUT_DIR / f"replay_{run_label}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "events.json").write_text(json.dumps(events, indent=1))
    (out / "ledger.json").write_text(json.dumps(ledger, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Track A prize sweep on a replay pool")
    ap.add_argument("pool", choices=sorted(POOLS))
    ap.add_argument("--store", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--verdict", choices=VERDICTS, default="numeric",
                    help="pass rule: numeric-only (default) or ARA's own (sensitivity)")
    args = ap.parse_args(argv)
    s = run_and_save(args.pool, args.out, args.store, args.verdict)
    print(f"{s['n_records']} records, {len(s['solvers'])} solvers, "
          f"{len(s['worlds'])} worlds, verdict rule {args.verdict} → "
          f"{args.out or OUTPUT_DIR / ('replay_' + s['pool'])}")
    print("NOTE:", s["caveat"])
    print("clearing prizes:", s["clearing_prizes"])
    for p, rev in s["lab_revenue"].items():
        print(f"  prize {p:>3}: lab revenue {rev['mean']:.1f} ± {rev['sd']:.1f}")
    print("checks:", s["checks"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
