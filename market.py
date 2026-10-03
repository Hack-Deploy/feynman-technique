"""Discovery Market – simulation engine.

Pure logic, no file I/O. Returns a list of events and ledger rows.
The cost model and success table are pluggable so Track A and Track C
share one engine.

NOTE: All probabilities used in this simulation are stand-in values
derived from published benchmark scores, not measured market outcomes.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Event:
    """A single market event."""
    run_id: str
    seed: int
    tick: int
    seq: int
    type: str
    world: str | None = None
    agent: str | None = None
    from_account: str | None = None
    to_account: str | None = None
    amount: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "run_id": self.run_id,
            "seed": self.seed,
            "tick": self.tick,
            "seq": self.seq,
            "type": self.type,
        }
        if self.world is not None:
            d["world"] = self.world
        if self.agent is not None:
            d["agent"] = self.agent
        if self.from_account is not None:
            d["from"] = self.from_account
        if self.to_account is not None:
            d["to"] = self.to_account
        if self.amount is not None:
            d["amount"] = float(self.amount)
        return d


@dataclass
class LedgerRow:
    """A ledger row recording an attempt."""
    attempt_id: str
    source: str
    question: str  # world
    solver: str  # agent
    design: str  # track description
    context: dict[str, Any]
    outcome: dict[str, Any]
    effort: dict[str, Any]
    provenance: dict[str, Any]
    disclosed_at_tick: int | None  # None until disclosed

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "source": self.source,
            "question": self.question,
            "solver": self.solver,
            "design": self.design,
            "context": self.context,
            "outcome": self.outcome,
            "effort": self.effort,
            "provenance": self.provenance,
            "disclosed_at_tick": self.disclosed_at_tick,
        }


# ---------------------------------------------------------------------------
# Cost model protocol
# ---------------------------------------------------------------------------

class CostModel(Protocol):
    """Protocol for drawing the cost of one attempt."""

    def draw_cost(self, agent: str, world: str, rng: np.random.Generator) -> tuple[float, dict]:
        """Return (total_cost, detail_dict).

        detail_dict must contain keys that the engine uses to emit
        the right charge events (e.g. rounds or experiments).
        """
        ...

    def charge_event_type(self) -> str:
        """Return the event type for charging: 'round_charged' or 'experiment_charged'."""
        ...

    def max_cost(self, agent: str, world: str) -> float:
        """Return the maximum possible cost for one attempt."""
        ...

    def expected_cost(self, agent: str, world: str) -> float:
        """Return the expected cost of one attempt (used by the bid rule)."""
        ...


# ---------------------------------------------------------------------------
# Concrete cost models
# ---------------------------------------------------------------------------

class TrackACostModel:
    """Track A: rounds uniform in [4, 16], 1 credit per round."""

    def __init__(self, cost_per_round: float = 1.0,
                 rounds_min: int = 4, rounds_max: int = 16):
        self.cost_per_round = cost_per_round
        self.rounds_min = rounds_min
        self.rounds_max = rounds_max

    def draw_cost(self, agent: str, world: str,
                  rng: np.random.Generator) -> tuple[float, dict]:
        rounds = int(rng.integers(self.rounds_min, self.rounds_max + 1))
        cost = rounds * self.cost_per_round
        return cost, {"rounds": rounds}

    def charge_event_type(self) -> str:
        return "round_charged"

    def max_cost(self, agent: str, world: str) -> float:
        return self.rounds_max * self.cost_per_round

    def expected_cost(self, agent: str, world: str) -> float:
        return (self.rounds_min + self.rounds_max) / 2 * self.cost_per_round


class TrackCCostModel:
    """Track C: fixed experiments per attempt × price per experiment."""

    def __init__(self, experiments_per_attempt: dict[str, int],
                 price_per_experiment: float):
        self.experiments_per_attempt = experiments_per_attempt
        self.price_per_experiment = price_per_experiment

    def draw_cost(self, agent: str, world: str,
                  rng: np.random.Generator) -> tuple[float, dict]:
        exps = self.experiments_per_attempt[agent]
        cost = exps * self.price_per_experiment
        return cost, {"experiments": exps}

    def charge_event_type(self) -> str:
        return "experiment_charged"

    def max_cost(self, agent: str, world: str) -> float:
        exps = self.experiments_per_attempt[agent]
        return exps * self.price_per_experiment

    def expected_cost(self, agent: str, world: str) -> float:
        return self.experiments_per_attempt[agent] * self.price_per_experiment


# ---------------------------------------------------------------------------
# Agent belief model
# ---------------------------------------------------------------------------

@dataclass
class AgentBelief:
    """Beta-distribution belief about pass probability for a world."""
    alpha: float
    beta_param: float

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta_param)

    def update(self, passed: bool) -> None:
        if passed:
            self.alpha += 1
        else:
            self.beta_param += 1


# ---------------------------------------------------------------------------
# Market engine
# ---------------------------------------------------------------------------

@dataclass
class MarketRun:
    """Configuration for a single market run."""
    run_id: str
    seed: int
    ticks: int
    worlds: list[str]
    agents: list[str]
    prizes: dict[str, float]   # world -> prize
    starting_credits: float
    cost_model: CostModel      # pluggable
    true_probs: dict[str, dict[str, float]]  # agent -> world -> p
    initial_beliefs: dict[str, dict[str, float]]  # agent -> world -> mean
    belief_weight: float  # pseudo-attempt weight for prior
    track: str  # "A" or "C"
    probability_source: str  # "raw", "calibrated", etc.
    code_version: str = "1.0.0"


def run_market(cfg: MarketRun) -> tuple[list[Event], list[LedgerRow]]:
    """Execute a full market simulation. Pure function, no I/O."""
    rng = np.random.default_rng(cfg.seed)
    events: list[Event] = []
    ledger: list[LedgerRow] = []
    seq = 0

    # --- Accounts ---
    # Total researcher funding = sum of all prizes
    total_prizes = sum(cfg.prizes.values())
    accounts: dict[str, float] = {
        "researcher": total_prizes,
        "escrow": 0.0,
        "lab": 0.0,
    }
    for agent in cfg.agents:
        accounts[f"agent:{agent}"] = cfg.starting_credits

    initial_total = sum(accounts.values())

    # --- State ---
    open_worlds = set(cfg.worlds)
    # Track which worlds closed during current tick (for bid cancellation)
    closed_this_tick: set[str] = set()

    # --- Beliefs ---
    beliefs: dict[str, dict[str, AgentBelief]] = {}
    for agent in cfg.agents:
        beliefs[agent] = {}
        for world in cfg.worlds:
            mean = cfg.initial_beliefs[agent].get(world, 0.5)
            alpha = mean * cfg.belief_weight
            beta = (1 - mean) * cfg.belief_weight
            beliefs[agent][world] = AgentBelief(alpha=alpha, beta_param=beta)

    # --- Hidden ledger rows (not yet disclosed) ---
    hidden_rows: list[LedgerRow] = []

    # --- Tick 0: fund accounts (so balances are recomputable from events) ---
    for acct in ["researcher"] + [f"agent:{a}" for a in cfg.agents]:
        events.append(Event(
            run_id=cfg.run_id, seed=cfg.seed, tick=0, seq=seq,
            type="account_funded",
            from_account="external", to_account=acct,
            amount=accounts[acct]
        ))
        seq += 1

    # --- Tick 0: post prizes ---
    for world in cfg.worlds:
        prize = cfg.prizes[world]
        accounts["researcher"] -= prize
        accounts["escrow"] += prize
        events.append(Event(
            run_id=cfg.run_id, seed=cfg.seed, tick=0, seq=seq,
            type="prize_posted", world=world,
            from_account="researcher", to_account="escrow",
            amount=prize
        ))
        seq += 1

    _check_conservation(accounts, initial_total, 0)

    # --- Ticks 1..ticks ---
    for tick in range(1, cfg.ticks + 1):
        closed_this_tick.clear()

        # 1. Collect bids: every open world × every agent with credits
        bids: list[tuple[str, str]] = []  # (agent, world)
        for world in [w for w in cfg.worlds if w in open_worlds]:
            prize = cfg.prizes[world]
            for agent in cfg.agents:
                acct = f"agent:{agent}"
                max_c = cfg.cost_model.max_cost(agent, world)
                if accounts[acct] < max_c:
                    continue
                belief_mean = beliefs[agent][world].mean
                expected_cost = cfg.cost_model.expected_cost(agent, world)
                if belief_mean * prize > expected_cost:
                    bids.append((agent, world))

        # 2. Shuffle bids
        if bids:
            indices = rng.permutation(len(bids))
            bids = [bids[i] for i in indices]

        # 3-6. Process bids
        for agent, world in bids:
            acct = f"agent:{agent}"

            # 3. Check if world closed this tick
            if world in closed_this_tick or world not in open_worlds:
                events.append(Event(
                    run_id=cfg.run_id, seed=cfg.seed, tick=tick, seq=seq,
                    type="bid_cancelled", world=world, agent=agent
                ))
                seq += 1
                continue

            # Re-check affordability (credits may have changed)
            max_c = cfg.cost_model.max_cost(agent, world)
            if accounts[acct] < max_c:
                continue

            # Emit bid_placed
            events.append(Event(
                run_id=cfg.run_id, seed=cfg.seed, tick=tick, seq=seq,
                type="bid_placed", world=world, agent=agent
            ))
            seq += 1

            # 4. Draw cost and charge
            cost, cost_detail = cfg.cost_model.draw_cost(agent, world, rng)
            accounts[acct] -= cost
            accounts["lab"] += cost

            charge_type = cfg.cost_model.charge_event_type()
            events.append(Event(
                run_id=cfg.run_id, seed=cfg.seed, tick=tick, seq=seq,
                type=charge_type, world=world, agent=agent,
                from_account=acct, to_account="lab",
                amount=cost
            ))
            seq += 1

            # 5. Draw pass/fail
            true_p = cfg.true_probs[agent].get(world, 0.0)
            passed = bool(rng.random() < true_p)

            attempt_id = str(uuid.uuid5(
                uuid.NAMESPACE_DNS,
                f"{cfg.run_id}:{cfg.seed}:{tick}:{agent}:{world}"
            ))

            if passed:
                prize = cfg.prizes[world]
                accounts["escrow"] -= prize
                accounts[acct] += prize
                open_worlds.discard(world)
                closed_this_tick.add(world)

                events.append(Event(
                    run_id=cfg.run_id, seed=cfg.seed, tick=tick, seq=seq,
                    type="attempt_passed", world=world, agent=agent
                ))
                seq += 1

                events.append(Event(
                    run_id=cfg.run_id, seed=cfg.seed, tick=tick, seq=seq,
                    type="prize_paid", world=world, agent=agent,
                    from_account="escrow", to_account=acct,
                    amount=prize
                ))
                seq += 1

                # Disclose all hidden rows for this world
                for row in hidden_rows:
                    if row.question == world and row.disclosed_at_tick is None:
                        row.disclosed_at_tick = tick

                # Create disclosed ledger row for the passing attempt
                row = LedgerRow(
                    attempt_id=attempt_id,
                    source="market_sim",
                    question=world,
                    solver=agent,
                    design=f"track_{cfg.track}",
                    context={
                        "prize": cfg.prizes[world],
                        "probability_source": cfg.probability_source,
                        "track": cfg.track,
                    },
                    outcome={"passed": True, "metric": None},
                    effort={
                        "rounds": cost_detail.get("rounds"),
                        "experiments": cost_detail.get("experiments"),
                        "credits": cost,
                    },
                    provenance={
                        "run_id": cfg.run_id,
                        "seed": cfg.seed,
                        "code_version": cfg.code_version,
                    },
                    disclosed_at_tick=tick,
                )
                ledger.append(row)
            else:
                events.append(Event(
                    run_id=cfg.run_id, seed=cfg.seed, tick=tick, seq=seq,
                    type="attempt_failed", world=world, agent=agent
                ))
                seq += 1

                # Hidden ledger row
                row = LedgerRow(
                    attempt_id=attempt_id,
                    source="market_sim",
                    question=world,
                    solver=agent,
                    design=f"track_{cfg.track}",
                    context={
                        "prize": cfg.prizes[world],
                        "probability_source": cfg.probability_source,
                        "track": cfg.track,
                    },
                    outcome={"passed": False, "metric": None},
                    effort={
                        "rounds": cost_detail.get("rounds"),
                        "experiments": cost_detail.get("experiments"),
                        "credits": cost,
                    },
                    provenance={
                        "run_id": cfg.run_id,
                        "seed": cfg.seed,
                        "code_version": cfg.code_version,
                    },
                    disclosed_at_tick=None,  # hidden until world closes or run ends
                )
                hidden_rows.append(row)
                ledger.append(row)

            # 6. Update belief from own outcome
            beliefs[agent][world].update(passed)

            # Emit estimate_updated
            events.append(Event(
                run_id=cfg.run_id, seed=cfg.seed, tick=tick, seq=seq,
                type="estimate_updated", world=world, agent=agent
            ))
            seq += 1

        _check_conservation(accounts, initial_total, tick)

    # --- End of run: refund open prizes ---
    for world in [w for w in cfg.worlds if w in open_worlds]:
        prize = cfg.prizes[world]
        accounts["escrow"] -= prize
        accounts["researcher"] += prize
        events.append(Event(
            run_id=cfg.run_id, seed=cfg.seed, tick=cfg.ticks, seq=seq,
            type="prize_refunded", world=world,
            from_account="escrow", to_account="researcher",
            amount=prize
        ))
        seq += 1

        # Disclose hidden rows for unsolved worlds at run end
        for row in hidden_rows:
            if row.question == world and row.disclosed_at_tick is None:
                row.disclosed_at_tick = cfg.ticks

    _check_conservation(accounts, initial_total, cfg.ticks)

    # Verify escrow is zero
    assert abs(accounts["escrow"]) < 1e-9, \
        f"Escrow not zero at end: {accounts['escrow']}"

    return events, ledger


def _check_conservation(accounts: dict[str, float], expected: float,
                        tick: int) -> None:
    """Assert total credits are conserved."""
    total = sum(accounts.values())
    assert abs(total - expected) < 1e-9, (
        f"Credit conservation violated at tick {tick}: "
        f"total={total}, expected={expected}, diff={total - expected}"
    )
