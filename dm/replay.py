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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from analysis import compute_final_balances, compute_funding, find_worlds_solved
from dm.calibration import calibration_summary
from dm.hypotheses import evaluate_hypotheses
from dm.importers.forcebench import default_settle_path, records_from_settle
from dm.outcomes import ReplayPool, experiments_cost, rounds_cost
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


@dataclass(frozen=True)
class ReplaySpec:
    venue: str
    cost_fn: Any
    charge_event: str
    prizes: list[float]
    cost_description: str
    caveat: str


ARA_CAVEAT = (
    "One attempt per (solver, world): within a run each solver can try each "
    "world at most once, and a world's outcome for a given solver is identical "
    "in every seed (only bid order changes). The '>= 3 of 5 seeds' clearing "
    "rule is therefore close to deterministic.")
FORCEBENCH_CAVEAT = (
    "5 settled attempts per (solver, world), drawn without replacement within "
    "a run; first launch free, each paid launch 1 credit (experiment_charged)")
REPLAY_SPECS = {
    "ara": ReplaySpec(
        venue="discoverphysics",
        cost_fn=rounds_cost(1.0),
        charge_event="round_charged",
        prizes=PRIZES,
        cost_description="rounds x 1 credit (round_charged)",
        caveat=ARA_CAVEAT,
    ),
    "forcebench": ReplaySpec(
        venue="forcebench",
        cost_fn=experiments_cost(1.0),
        charge_event="experiment_charged",
        prizes=[2, 5, 10, 20, 50],
        cost_description=FORCEBENCH_CAVEAT,
        caveat=FORCEBENCH_CAVEAT,
    ),
}
POOLS = {"ara": ATTEMPTS_DIR / "ara.jsonl", "forcebench": default_settle_path()}


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


def _spec(pool: str) -> ReplaySpec:
    try:
        return REPLAY_SPECS[pool]
    except KeyError:
        raise ValueError(f"unknown replay pool {pool!r}") from None


def _validate_venues(records: list[AttemptRecord], pool: str) -> None:
    venues = {record.venue for record in records}
    spec = _spec(pool)
    if len(venues) > 1:
        raise ValueError(f"replay pool {pool!r} cannot mix venues: {sorted(venues)}")
    if venues and venues != {spec.venue}:
        raise ValueError(
            f"replay pool {pool!r} requires venue {spec.venue!r}, "
            f"got {sorted(venues)}")


def make_pool(records: list[AttemptRecord], pool: str = "ara") -> ReplayPool:
    _validate_venues(records, pool)
    spec = _spec(pool)
    return ReplayPool(records, spec.cost_fn, charge_event=spec.charge_event,
                      venue=spec.venue)


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
              prizes: list[float] | None = None, seeds: list[int] | None = None,
              ticks: int = TICKS, starting_credits: float = STARTING_CREDITS,
              pool: str = "ara",
              ) -> tuple[list[dict], list[dict]]:
    _validate_venues(records, pool)
    spec = _spec(pool)
    chosen_prizes = spec.prizes if prizes is None else prizes
    chosen_seeds = SEEDS if seeds is None else seeds
    replay_pool = make_pool(records, pool)
    worlds, agents = replay_pool.worlds(), replay_pool.solvers()
    beliefs = loo_beliefs(replay_pool)
    events: list[dict] = []
    ledger: list[dict] = []
    for prize in chosen_prizes:
        for seed in chosen_seeds:
            cfg = MarketRun(
                run_id=run_id(label, prize, seed), seed=seed, ticks=ticks,
                worlds=worlds, agents=agents, prizes={w: float(prize) for w in worlds},
                starting_credits=starting_credits,
                # The same pool must drive the bid rule and the charges, so an
                # agent's affordability check matches what it is charged.
                cost_model=replay_pool, true_probs=None,
                initial_beliefs=beliefs, belief_weight=BELIEF_WEIGHT,
                track=f"replay_{label}", probability_source=f"replay:{label}",
                outcome_source=replay_pool, state_confidence=True,
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
              label: str, prizes: list[float] | None = None,
              seeds: list[int] | None = None, pool: str = "ara",
              source_file: str | None = None) -> dict[str, Any]:
    _validate_venues(records, pool)
    spec = _spec(pool)
    prizes = spec.prizes if prizes is None else prizes
    seeds = SEEDS if seeds is None else seeds
    replay_pool = make_pool(records, pool)
    worlds = replay_pool.worlds()
    counts = solved_counts(events, worlds, label, prizes, seeds)
    seed_dependent = {w: {p: n for p, n in c.items() if 0 < n < len(seeds)}
                      for w, c in counts.items()}
    by_run = _by_run(events)
    return {
        "pool": label,
        "caveat": spec.caveat,
        "config": {"prizes": prizes, "seeds": seeds, "ticks": TICKS,
                   "starting_credits": STARTING_CREDITS, "belief_weight": BELIEF_WEIGHT,
                   "cost": spec.cost_description,
                   "beliefs": "leave-one-out pass rate per solver, 0.5 fallback",
                   "clearing_rule": f"lowest prize solved in >= {MIN_SEEDS_SOLVED} of "
                                    f"{len(seeds)} seeds, else 'never'"},
        "n_records": len(records),
        "source_file": source_file or next(
            (r.extra.get("settle_file") for r in records if r.extra.get("settle_file")),
            str(POOLS[pool].resolve().relative_to(ROOT))
            if POOLS[pool].resolve().is_relative_to(ROOT) else str(POOLS[pool])),
        "solvers": replay_pool.solvers(),
        "worlds": worlds,
        "initial_beliefs": loo_beliefs(replay_pool),
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
        "calibration": calibration_summary(records),
        "hypotheses": evaluate_hypotheses(
            events, records, label, prizes, seeds, spec.charge_event),
    }


def run_and_save(label: str, out_dir: Path | None = None,
                 store_path: Path | None = None, verdict: str = "numeric") -> dict[str, Any]:
    if label not in REPLAY_SPECS:
        raise ValueError(f"unknown replay pool {label!r}")
    if verdict != "numeric" and label != "ara":
        raise ValueError("ARA verdict sensitivity is only available for the ARA pool")
    path = store_path or (default_settle_path() if label == "forcebench" else POOLS[label])
    if path.suffix == ".json":
        if label != "forcebench":
            raise ValueError("settle .json inputs are only supported for ForceBench")
        records = records_from_settle(path)
    else:
        records = AttemptStore(path).load()
    records = with_verdict(records, verdict)
    _validate_venues(records, label)
    if not records:
        raise SystemExit(f"no records in {path}; run the importer first "
                         f"(uv run python -m dm.importers.{label})")
    run_label = label if verdict == "numeric" else f"{label}_{verdict}verdict"
    events, ledger = run_sweep(records, run_label, pool=label)
    try:
        source_file = str(path.resolve().relative_to(ROOT))
    except ValueError:
        source_file = str(path.resolve())
    summary = summarise(events, ledger, records, run_label, pool=label,
                        source_file=source_file)
    summary["verdict_rule"] = verdict
    if summary["checks"]["credit_sum_per_run_max_abs"] > 1e-9:
        raise ValueError(
            "replay failed credit conservation: "
            f"{summary['checks']['credit_sum_per_run_max_abs']}")
    out = out_dir or OUTPUT_DIR / f"replay_{run_label}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "events.json").write_text(json.dumps(events, indent=1))
    (out / "ledger.json").write_text(json.dumps(ledger, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Track A prize sweep on a replay pool")
    ap.add_argument("pool", choices=sorted(REPLAY_SPECS))
    ap.add_argument("--store", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--verdict", choices=VERDICTS, default="numeric",
                    help="pass rule: numeric-only (default) or ARA's own (sensitivity)")
    args = ap.parse_args(argv)
    s = run_and_save(args.pool, args.out, args.store, args.verdict)
    if args.pool == "forcebench":
        print(f"settle file: {s['source_file']}")
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
