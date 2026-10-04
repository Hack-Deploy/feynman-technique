"""Settings for the DiscoverPhysics market benchmark (PLAN.md §6).

Everything a human may want to change lives in ``config.yaml`` next to this file;
this module loads and validates it and holds the fixed paths and protocol constants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
VENDOR_ROOT = ROOT / "vendor" / "discovery-agents"
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
ENV_PATH = Path(__file__).resolve().parent / ".env"  # API keys; git-ignored

# Fixed protocol (not human-tunable: changing these changes what a pass means).
MIN_ROUNDS = 1  # every round is paid, so the agent may submit straight away
ENGINE = "nbody"
MAX_TOKENS = 8192  # vendor default for non-reasoning models
VENUE = "discoverphysics"
PROTOCOL = "discoverphysics_market"
VERDICTS = ("supported", "refuted", "inconclusive")
ANSWERS = ("supported", "refuted")
RULES = ("market", "naive")

ATTEMPTS_PATH = ROOT / "attempts" / "poc_dp_bench.jsonl"
TRANSCRIPTS_DIR = ROOT / "attempts" / "transcripts" / "poc_dp"
TRAJECTORIES_DIR = ROOT / "attempts" / "poc_trajectories"
REPORT_PATH = ROOT / "output" / "poc_report.json"

COST_KEYS = ("per_experiment", "per_measurement", "per_time_unit", "per_particle",
             "per_custom_property")


@dataclass(frozen=True)
class Quantity:
    name: str
    meaning: str
    tolerance: float  # a claim is confirmed only if |estimate − truth| <= tolerance


@dataclass(frozen=True)
class Rule:
    """Turns estimates into a verdict: supported if ``quantity`` is above / below / between /
    outside the given bound(s), refuted otherwise. Public: it is part of the posting."""

    quantity: str
    kind: str  # above | below | between | outside
    bounds: tuple[float, ...]

    def verdict(self, estimates: dict[str, float]) -> str | None:
        x = estimates.get(self.quantity)
        if x is None or not math.isfinite(x):
            return None
        if self.kind == "above":
            ok = x > self.bounds[0]
        elif self.kind == "below":
            ok = x < self.bounds[0]
        elif self.kind == "between":
            ok = self.bounds[0] <= x <= self.bounds[1]
        else:
            ok = not (self.bounds[0] <= x <= self.bounds[1])
        return "supported" if ok else "refuted"

    def margin(self, x: float) -> float:
        """Distance from x to the nearest decision boundary."""
        return min(abs(x - b) for b in self.bounds)


@dataclass(frozen=True)
class Hypothesis:
    id: str
    world: str
    hypothesis: str
    resolution_criteria: str
    answer: str  # hidden from the agent
    prize: float
    quantities: tuple[Quantity, ...] = ()
    supported_if: Rule | None = None
    particles: int | None = None  # override when the agent does not place particles itself

    def quantity(self, name: str) -> Quantity:
        for q in self.quantities:
            if q.name == name:
                return q
        raise KeyError(name)


@dataclass(frozen=True)
class Config:
    max_rounds: int
    noise_std: float
    budget: float | None
    market_payout_rule: str
    naive_payout_rule: str
    claim_bond: float
    calibration_bonus: float
    round_fee: float
    experiment_costs: dict[str, float]
    hypotheses: tuple[Hypothesis, ...]
    ledger_max_entries: int
    ledger_max_data_chars: int
    ledger_read_fee: float

    def hypothesis(self, hid: str) -> Hypothesis:
        for h in self.hypotheses:
            if h.id == hid:
                return h
        raise KeyError(f"no hypothesis {hid!r} in config.yaml")


def _amount(value, name: str) -> float:
    x = float(value)
    if not math.isfinite(x) or x < 0:
        raise ValueError(f"config.yaml: {name} must be a finite number >= 0")
    return x


def _rule(d: dict, hid: str, names: set[str]) -> Rule:
    kinds = [k for k in ("above", "below", "between", "outside") if k in d]
    if len(kinds) != 1 or d.get("quantity") not in names:
        raise ValueError(f"config.yaml: {hid}: supported_if needs a quantity and one of "
                         "above/below/between/outside")
    raw = d[kinds[0]]
    bounds = tuple(float(b) for b in (raw if isinstance(raw, list) else [raw]))
    if len(bounds) != (2 if kinds[0] in ("between", "outside") else 1):
        raise ValueError(f"config.yaml: {hid}: wrong number of bounds for {kinds[0]}")
    return Rule(quantity=d["quantity"], kind=kinds[0], bounds=bounds)


def load(path: Path = CONFIG_PATH) -> Config:
    d = yaml.safe_load(Path(path).read_text())
    default_prize = _amount(d.get("default_prize", 100), "default_prize")
    costs = d.get("experiment_costs") or {}
    unknown = sorted(set(costs) - set(COST_KEYS))
    if unknown:
        raise ValueError(f"config.yaml: unknown experiment_costs keys: {', '.join(unknown)}")
    hypotheses = []
    for h in d["hypotheses"]:
        answer = str(h["answer"]).strip().lower()
        if answer not in ANSWERS:
            raise ValueError(f"config.yaml: {h['id']}: answer must be supported or refuted")
        criteria = str(h.get("resolution_criteria") or "").strip()
        if not criteria:
            raise ValueError(f"config.yaml: {h['id']}: resolution_criteria is required")
        quantities = tuple(Quantity(name=str(q["name"]), meaning=str(q["meaning"]),
                                    tolerance=_amount(q["tolerance"], f"{h['id']}.tolerance"))
                           for q in h.get("quantities") or [])
        if not quantities:
            raise ValueError(f"config.yaml: {h['id']}: at least one quantity is required")
        names = {q.name for q in quantities}
        hypotheses.append(Hypothesis(
            id=str(h["id"]),
            world=str(h["world"]),
            hypothesis=str(h["hypothesis"]).strip(),
            resolution_criteria=criteria,
            answer=answer,
            prize=_amount(h.get("prize", default_prize), f"{h['id']}.prize"),
            quantities=quantities,
            supported_if=_rule(h.get("supported_if") or {}, h["id"], names),
            particles=int(h["particles"]) if h.get("particles") is not None else None,
        ))
    ids = [h.id for h in hypotheses]
    if len(set(ids)) != len(ids):
        raise ValueError("config.yaml: hypothesis ids must be unique")
    max_rounds = int(d.get("max_rounds", 16))
    if max_rounds < 1:
        raise ValueError("config.yaml: max_rounds must be >= 1")
    ledger = d.get("ledger") or {}
    budget = d.get("budget")
    return Config(
        max_rounds=max_rounds,
        noise_std=_amount(d.get("noise_std", 0.075), "noise_std"),
        budget=None if budget is None else _amount(budget, "budget"),
        market_payout_rule=str(d.get("market_payout_rule", "")).strip(),
        naive_payout_rule=str(d.get("naive_payout_rule", "")).strip(),
        claim_bond=_amount(d.get("claim_bond", 0), "claim_bond"),
        calibration_bonus=_amount(d.get("calibration_bonus", 0), "calibration_bonus"),
        round_fee=_amount(d.get("round_fee", 0), "round_fee"),
        experiment_costs={k: _amount(costs.get(k, 0), k) for k in COST_KEYS},
        hypotheses=tuple(hypotheses),
        ledger_max_entries=int(ledger.get("max_entries", 10)),
        ledger_max_data_chars=int(ledger.get("max_data_chars", 6000)),
        ledger_read_fee=_amount(ledger.get("read_fee", 30), "ledger.read_fee"),
    )
