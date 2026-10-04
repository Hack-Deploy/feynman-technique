"""Lab prices and the per-attempt account. Pure: no vendor imports."""

from __future__ import annotations

from dataclasses import dataclass, field

from poc.config import Config, Hypothesis


def _is_default(value) -> bool:
    try:
        return float(value) == 1.0
    except (TypeError, ValueError):
        return False


def particles_placed(inp: dict, hyp: Hypothesis) -> int:
    for key in ("probe_positions", "positions"):
        if isinstance(inp.get(key), list):
            return len(inp[key])
    if "pos2" in inp:
        return 1
    return hyp.particles if hyp.particles is not None else 1


def custom_properties(inp: dict) -> int:
    n = sum(1 for key in ("p1", "p2") if key in inp and not _is_default(inp[key]))
    masses = inp.get("probe_masses")
    if isinstance(masses, list):
        n += sum(1 for m in masses if not _is_default(m))
    return n


def experiment_price(inp: dict, cfg: Config, hyp: Hypothesis) -> tuple[float, dict]:
    """Price of one experiment and its itemised breakdown."""
    if not isinstance(inp, dict):
        raise ValueError("each experiment must be a JSON object")
    times = inp.get("measurement_times") or []
    if not isinstance(times, list):
        raise ValueError("measurement_times must be a list")
    duration = inp.get("duration")
    if duration is None:
        duration = max((float(t) for t in times), default=0.0)
    units = {
        "per_experiment": 1,
        "per_measurement": len(times),
        "per_time_unit": max(float(duration), 0.0),
        "per_particle": particles_placed(inp, hyp),
        "per_custom_property": custom_properties(inp),
    }
    items = {k: round(units[k] * cfg.experiment_costs[k], 6) for k in units}
    return round(sum(items.values()), 6), {"units": units, "items": items}


@dataclass
class Account:
    """What one agent has spent on one hypothesis, with each charge recorded as an event."""

    agent: str
    hypothesis: str
    budget: float | None
    spent: float = 0.0
    events: list[dict] = field(default_factory=list)

    @property
    def remaining(self) -> float | None:
        return None if self.budget is None else round(self.budget - self.spent, 6)

    def can_afford(self, amount: float) -> bool:
        return self.budget is None or self.spent + amount <= self.budget + 1e-9

    def charge(self, kind: str, amount: float, round_num: int, detail: dict | None = None) -> None:
        if amount < 0:
            raise ValueError("charges must be >= 0")
        self.spent = round(self.spent + amount, 6)
        recipient = "market" if kind == "record_charged" else "lab"
        self.events.append({
            "type": kind, "from": f"agent:{self.agent}", "to": recipient, "amount": amount,
            "hypothesis": self.hypothesis, "round": round_num, **({"detail": detail} if detail else {}),
        })

    def refuse(self, amount: float, round_num: int, count: int) -> None:
        self.events.append({
            "type": "insufficient_budget", "from": f"agent:{self.agent}", "to": "lab",
            "amount": amount, "count": count, "remaining": self.remaining,
            "hypothesis": self.hypothesis, "round": round_num,
        })

    def lab_revenue(self) -> float:
        return round(
            sum(e["amount"] for e in self.events
                if e.get("to") == "lab" and e["type"].endswith("_charged")),
            6,
        )


class OverBudget(RuntimeError):
    """Shown to the agent as the experiment output when a batch would exceed its budget."""

    def __init__(self, price: float, remaining: float):
        super().__init__(
            f"refused: these experiments cost {price:g} credits but only {remaining:g} remain "
            "in your budget. Nothing was run or charged. Run cheaper experiments, submit your "
            "<verdict>, or <withdraw>reason</withdraw> if you cannot afford what you would need.")
