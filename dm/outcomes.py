"""Outcome sources for the market engine (see ``market.OutcomeSource``).

``BernoulliTable`` (in ``market.py``) is the legacy coin flip. ``ReplayPool``
replays real attempts: each bid draws one ``AttemptRecord`` for that solver and
world, without replacement within a run and with replacement across runs (each
run calls ``reset``). A solver with no attempts on a world cannot bid on it.

A ReplayPool is also the run's cost model: the bid rule uses the expected and
maximum cost of the attempts still in the pool.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np

from dm.types import AttemptRecord

# A cost function maps a record to (credits charged, detail for the charge event).
CostFn = Callable[[AttemptRecord], tuple[float, dict]]


def rounds_cost(price_per_round: float = 1.0) -> CostFn:
    """Charge per round (ARA replays: rounds × 1 credit, comparable with Track A)."""
    def fn(r: AttemptRecord) -> tuple[float, dict]:
        return r.rounds * price_per_round, {"rounds": r.rounds, "experiments": r.experiments}
    return fn


def experiments_cost(price_per_experiment: float) -> CostFn:
    """Charge per experiment (exact counts: live and ForceBench attempts)."""
    def fn(r: AttemptRecord) -> tuple[float, dict]:
        return (r.experiments * price_per_experiment,
                {"rounds": r.rounds, "experiments": r.experiments, "count": r.experiments})
    return fn


def recorded_cost() -> CostFn:
    """Charge what the attempt actually paid the lab when it ran."""
    def fn(r: AttemptRecord) -> tuple[float, dict]:
        return r.lab_cost, {"rounds": r.rounds, "experiments": r.experiments,
                            "count": r.experiments}
    return fn


class ReplayPool:
    def __init__(self, records: list[AttemptRecord], cost_fn: CostFn,
                 charge_event: str = "experiment_charged"):
        self.cost_fn = cost_fn
        self.charge_event = charge_event
        pool: dict[tuple[str, str], list[AttemptRecord]] = defaultdict(list)
        for r in sorted(records, key=lambda r: r.attempt_id):  # order-independent
            pool[(r.solver, r.world)].append(r)
        self.pool = dict(pool)
        self.remaining: dict[tuple[str, str], list[AttemptRecord]] = {}
        self.reset()

    # --- OutcomeSource ---------------------------------------------------
    def reset(self) -> None:
        self.remaining = {k: list(v) for k, v in self.pool.items()}

    def available(self, agent: str, world: str) -> bool:
        return bool(self.remaining.get((agent, world)))

    def draw(self, agent: str, world: str,
             rng: np.random.Generator) -> tuple[bool, dict, str | None]:
        left = self.remaining[(agent, world)]
        rec = left.pop(int(rng.integers(len(left))))
        credits, detail = self.cost_fn(rec)
        detail = {
            **detail,
            "credits": credits,
            "stated_p": rec.stated_p_success,
            "normalised_mse": rec.verdict.get("normalised_mse"),
            "commitment": rec.verdict.get("prereg_commitment"),
            "attempt_source": rec.source,
            "protocol": rec.protocol,
        }
        return rec.passed, detail, rec.attempt_id

    # --- CostModel -------------------------------------------------------
    def _costs(self, agent: str, world: str) -> list[float]:
        return [self.cost_fn(r)[0] for r in self.remaining.get((agent, world), [])]

    def max_cost(self, agent: str, world: str) -> float:
        costs = self._costs(agent, world)
        return max(costs) if costs else float("inf")

    def expected_cost(self, agent: str, world: str) -> float:
        costs = self._costs(agent, world)
        return float(np.mean(costs)) if costs else float("inf")

    def charge_event_type(self) -> str:
        return self.charge_event

    def draw_cost(self, agent: str, world: str, rng: np.random.Generator):
        raise NotImplementedError("ReplayPool charges through draw()")

    # --- Helpers ---------------------------------------------------------
    def solvers(self) -> list[str]:
        return sorted({s for s, _ in self.pool})

    def worlds(self) -> list[str]:
        return sorted({w for _, w in self.pool})

    def pass_rate(self, solver: str, exclude_world: str | None = None) -> float | None:
        """Share of this solver's pooled attempts that passed (optionally leaving one
        world out, so a prior about a world never uses that world's own outcome)."""
        recs = [r for (s, w), rs in self.pool.items() for r in rs
                if s == solver and w != exclude_world]
        return sum(r.passed for r in recs) / len(recs) if recs else None
