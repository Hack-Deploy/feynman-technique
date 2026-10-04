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

ATTEMPTS_PATH = ROOT / "attempts" / "poc_dp_bench.jsonl"
TRANSCRIPTS_DIR = ROOT / "attempts" / "transcripts" / "poc_dp"
TRAJECTORIES_DIR = ROOT / "attempts" / "poc_trajectories"
REPORT_PATH = ROOT / "output" / "poc_report.json"

COST_KEYS = ("per_experiment", "per_measurement", "per_time_unit", "per_particle",
             "per_custom_property")


@dataclass(frozen=True)
class Hypothesis:
    id: str
    world: str
    hypothesis: str
    resolution_criteria: str
    answer: str  # hidden from the agent
    prize: float
    particles: int | None = None  # override when the agent does not place particles itself


@dataclass(frozen=True)
class Config:
    max_rounds: int
    noise_std: float
    velocity_noise_std: float
    budget: float | None
    payout_rule: str
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


def load(path: Path = CONFIG_PATH) -> Config:
    d = yaml.safe_load(Path(path).read_text())
    default_prize = _amount(d.get("default_prize", 100), "default_prize")
    payout_rule = str(d.get("payout_rule", "")).strip()
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
        hypotheses.append(Hypothesis(
            id=str(h["id"]),
            world=str(h["world"]),
            hypothesis=str(h["hypothesis"]).strip(),
            resolution_criteria=criteria,
            answer=answer,
            prize=_amount(h.get("prize", default_prize), f"{h['id']}.prize"),
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
        velocity_noise_std=_amount(d.get("velocity_noise_std", 0.0), "velocity_noise_std"),
        budget=None if budget is None else _amount(budget, "budget"),
        payout_rule=payout_rule,
        round_fee=_amount(d.get("round_fee", 0), "round_fee"),
        experiment_costs={k: _amount(costs.get(k, 0), k) for k in COST_KEYS},
        hypotheses=tuple(hypotheses),
        ledger_max_entries=int(ledger.get("max_entries", 10)),
        ledger_max_data_chars=int(ledger.get("max_data_chars", 6000)),
        ledger_read_fee=_amount(ledger.get("read_fee", 30), "ledger.read_fee"),
    )
