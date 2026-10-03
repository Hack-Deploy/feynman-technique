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
    # Real-attempt fields; serialised only when set, so legacy outputs keep their bytes.
    count: int | None = None          # experiments charged
    p: float | None = None            # stated probability of success
    commitment: str | None = None     # preregistration sha256
    attempt_id: str | None = None     # AttemptRecord being replayed / run
    detail: dict[str, Any] | None = None

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
        if self.count is not None:
            d["count"] = int(self.count)
        if self.p is not None:
            d["p"] = float(self.p)
        if self.commitment is not None:
            d["commitment"] = self.commitment
        if self.attempt_id is not None:
            d["attempt_id"] = self.attempt_id
        if self.detail is not None:
            d["detail"] = self.detail
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
# Outcome sources
# ---------------------------------------------------------------------------

class OutcomeSource(Protocol):
    """Decides what one attempt costs and whether it passes.

    ``draw`` returns ``(passed, cost_detail, attempt_id)``. ``cost_detail`` must hold
    ``credits`` (the amount charged to the lab) and may hold ``rounds``,
    ``experiments``, ``count`` (experiments, put on the charge event),
    ``stated_p``, ``normalised_mse`` and ``commitment``. ``attempt_id`` is the
    AttemptRecord replayed, or None for a simulated draw.
    """

    def available(self, agent: str, world: str) -> bool:
        """False if the agent cannot attempt this world (e.g. empty pool)."""
        ...

    def draw(self, agent: str, world: str,
             rng: np.random.Generator) -> tuple[bool, dict, str | None]:
        ...


class BernoulliTable:
    """Today's behaviour: cost from the cost model, pass with probability p.

    Makes exactly the same rng calls, in the same order (cost, then pass), as the
    engine did before outcome sources existed, so results are byte-identical.
    """

    def __init__(self, cost_model: CostModel,
                 true_probs: dict[str, dict[str, float]]):
        self.cost_model = cost_model
        self.true_probs = true_probs

    def reset(self) -> None:
        pass

    def available(self, agent: str, world: str) -> bool:
        return True

    def draw(self, agent: str, world: str,
             rng: np.random.Generator) -> tuple[bool, dict, str | None]:
        cost, detail = self.cost_model.draw_cost(agent, world, rng)
        passed = bool(rng.random() < self.true_probs[agent].get(world, 0.0))
        return passed, {**detail, "credits": cost}, None


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
    cost_model: CostModel      # pluggable; drives the bid rule (max / expected cost)
    true_probs: dict[str, dict[str, float]] | None  # agent -> world -> p (BernoulliTable)
    initial_beliefs: dict[str, dict[str, float]]  # agent -> world -> mean
    belief_weight: float  # pseudo-attempt weight for prior
    track: str  # "A", "C", or a replay label
    probability_source: str  # "raw", "calibrated", "replay:ara", etc.
    code_version: str = "1.0.0"
    # None → BernoulliTable(cost_model, true_probs), i.e. the legacy behaviour.
    outcome_source: OutcomeSource | None = None
    # Real-attempt mode: emit confidence_stated and attempt lifecycle events.
    state_confidence: bool = False
    # world -> Preregistration; commitment published at posting, revealed at close.
    preregs: dict[str, Any] | None = None


def run_market(cfg: MarketRun) -> tuple[list[Event], list[LedgerRow]]:
    """Execute a full market simulation. Pure function, no I/O."""
    rng = np.random.default_rng(cfg.seed)
    events: list[Event] = []
    ledger: list[LedgerRow] = []
    seq = 0

    source: OutcomeSource = cfg.outcome_source or BernoulliTable(
        cfg.cost_model, cfg.true_probs or {})
    if hasattr(source, "reset"):
        source.reset()
    real = cfg.state_confidence
    preregs = cfg.preregs or {}

    def emit(**kw) -> None:
        nonlocal seq
        events.append(Event(run_id=cfg.run_id, seed=cfg.seed, seq=seq, **kw))
        seq += 1

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
    # (agent, world) pairs already told they cannot afford an attempt
    broke_noted: set[tuple[str, str]] = set()

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
        emit(tick=0, type="account_funded",
             from_account="external", to_account=acct, amount=accounts[acct])

    # --- Tick 0: post prizes (with the preregistration commitment, if any) ---
    for world in cfg.worlds:
        prize = cfg.prizes[world]
        accounts["researcher"] -= prize
        accounts["escrow"] += prize
        commitment = preregs[world].commitment() if world in preregs else None
        emit(tick=0, type="prize_posted", world=world,
             from_account="researcher", to_account="escrow", amount=prize,
             commitment=commitment)
        if commitment is not None:
            emit(tick=0, type="prereg_committed", world=world,
                 commitment=commitment)

    _check_conservation(accounts, initial_total, 0)

    def reveal(world: str, tick: int) -> None:
        if world in preregs:
            emit(tick=tick, type="prereg_revealed", world=world,
                 commitment=preregs[world].commitment(),
                 detail=preregs[world].to_dict())

    # --- Ticks 1..ticks ---
    for tick in range(1, cfg.ticks + 1):
        closed_this_tick.clear()

        # 1. Collect bids: every open world × every agent with credits
        bids: list[tuple[str, str]] = []  # (agent, world)
        for world in [w for w in cfg.worlds if w in open_worlds]:
            prize = cfg.prizes[world]
            for agent in cfg.agents:
                if not source.available(agent, world):
                    continue
                acct = f"agent:{agent}"
                belief_mean = beliefs[agent][world].mean
                expected_cost = cfg.cost_model.expected_cost(agent, world)
                wants = belief_mean * prize > expected_cost
                if accounts[acct] < cfg.cost_model.max_cost(agent, world):
                    if real and wants and (agent, world) not in broke_noted:
                        broke_noted.add((agent, world))
                        emit(tick=tick, type="insufficient_credits",
                             world=world, agent=agent,
                             amount=cfg.cost_model.max_cost(agent, world))
                    continue
                if wants:
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
                emit(tick=tick, type="bid_cancelled", world=world, agent=agent)
                continue

            # Re-check affordability (credits may have changed)
            max_c = cfg.cost_model.max_cost(agent, world)
            if accounts[acct] < max_c:
                continue

            emit(tick=tick, type="bid_placed", world=world, agent=agent)

            # 4. Draw the attempt (cost and outcome) from the outcome source
            passed, cost_detail, source_attempt_id = source.draw(agent, world, rng)
            cost = cost_detail["credits"]
            if (world in preregs and source_attempt_id is not None
                    and cost_detail.get("commitment") != preregs[world].commitment()):
                raise ValueError(
                    f"attempt {source_attempt_id} on {world} was not scored against the "
                    f"preregistration posted for this prize")

            if real:
                emit(tick=tick, type="attempt_started", world=world, agent=agent,
                     attempt_id=source_attempt_id)
                stated = cost_detail.get("stated_p")
                emit(tick=tick, type="confidence_stated", world=world, agent=agent,
                     attempt_id=source_attempt_id,
                     p=stated if stated is not None else beliefs[agent][world].mean)

            accounts[acct] -= cost
            accounts["lab"] += cost
            emit(tick=tick, type=cfg.cost_model.charge_event_type(),
                 world=world, agent=agent,
                 from_account=acct, to_account="lab", amount=cost,
                 count=cost_detail.get("count"), attempt_id=source_attempt_id)

            if real:
                emit(tick=tick, type="attempt_submitted", world=world, agent=agent,
                     attempt_id=source_attempt_id)
                emit(tick=tick, type="verdict_issued", world=world, agent=agent,
                     attempt_id=source_attempt_id,
                     commitment=cost_detail.get("commitment"),
                     detail={"passed": passed,
                             "normalised_mse": cost_detail.get("normalised_mse")})

            attempt_id = str(uuid.uuid5(
                uuid.NAMESPACE_DNS,
                f"{cfg.run_id}:{cfg.seed}:{tick}:{agent}:{world}"
            ))
            context = {
                "prize": cfg.prizes[world],
                "probability_source": cfg.probability_source,
                "track": cfg.track,
            }
            if source_attempt_id is not None:
                context["replayed_attempt_id"] = source_attempt_id
            row = LedgerRow(
                attempt_id=attempt_id,
                source="market_sim",
                question=world,
                solver=agent,
                design=f"track_{cfg.track}",
                context=context,
                # Simulated rows record only pass/fail (never the hidden true p);
                # replayed real attempts also record their measured normalised MSE.
                outcome=({"passed": passed}
                         if source_attempt_id is None else
                         {"passed": passed,
                          "metric": cost_detail.get("normalised_mse")}),
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

            if passed:
                prize = cfg.prizes[world]
                accounts["escrow"] -= prize
                accounts[acct] += prize
                open_worlds.discard(world)
                closed_this_tick.add(world)

                emit(tick=tick, type="attempt_passed", world=world, agent=agent)
                emit(tick=tick, type="prize_paid", world=world, agent=agent,
                     from_account="escrow", to_account=acct, amount=prize)
                reveal(world, tick)

                # Disclose all hidden rows for this world
                for hidden in hidden_rows:
                    if hidden.question == world and hidden.disclosed_at_tick is None:
                        hidden.disclosed_at_tick = tick

                # The passing attempt is disclosed immediately
                row.disclosed_at_tick = tick
            else:
                emit(tick=tick, type="attempt_failed", world=world, agent=agent)
                hidden_rows.append(row)
            ledger.append(row)

            # 6. Update belief from own outcome
            beliefs[agent][world].update(passed)
            emit(tick=tick, type="estimate_updated", world=world, agent=agent)

        _check_conservation(accounts, initial_total, tick)

    # --- End of run: refund open prizes ---
    for world in [w for w in cfg.worlds if w in open_worlds]:
        prize = cfg.prizes[world]
        accounts["escrow"] -= prize
        accounts["researcher"] += prize
        emit(tick=cfg.ticks, type="prize_refunded", world=world,
             from_account="escrow", to_account="researcher", amount=prize)
        reveal(world, cfg.ticks)

        # Disclose hidden rows for unsolved worlds at run end
        for hidden in hidden_rows:
            if hidden.question == world and hidden.disclosed_at_tick is None:
                hidden.disclosed_at_tick = cfg.ticks

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
