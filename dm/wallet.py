"""Credit wallet shared by Discovery Market venues."""

from __future__ import annotations

import math

from dm.types import InsufficientCredits


def _milli_credits(value: float, name: str) -> int:
    amount = float(value)
    if not math.isfinite(amount) or amount < 0:
        raise ValueError(f"{name} must be a finite non-negative amount")
    scaled = amount * 1000
    rounded = round(scaled)
    if abs(scaled - rounded) > 1e-6:
        raise ValueError(f"{name} must be a whole number of milli-credits")
    return int(rounded)


class Wallet:
    """Integer-backed wallet that records every transfer to the lab."""

    def __init__(self, owner: str, balance: float):
        self.owner = owner
        self._balance_m = _milli_credits(balance, "balance")
        self._lab_m = 0
        self.events: list[dict] = []

    @property
    def balance(self) -> float:
        return self._balance_m / 1000

    @property
    def lab_revenue(self) -> float:
        return self._lab_m / 1000

    def charge(self, count: int, price: float, *, world: str, round_num: int) -> None:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("count must be a non-negative integer")
        price_m = _milli_credits(price, "price")
        cost_m = count * price_m
        cost = cost_m / 1000
        if self._balance_m < cost_m:
            self.events.append(
                {
                    "type": "insufficient_credits",
                    "from": f"agent:{self.owner}",
                    "to": "lab",
                    "amount": cost,
                    "count": count,
                    "balance": self.balance,
                    "world": world,
                    "round": round_num,
                }
            )
            raise InsufficientCredits(
                needed=cost, balance=self.balance, count=count, price=float(price)
            )
        self._balance_m -= cost_m
        self._lab_m += cost_m
        self.events.append(
            {
                "type": "experiment_charged",
                "from": f"agent:{self.owner}",
                "to": "lab",
                "amount": cost,
                "count": count,
                "world": world,
                "round": round_num,
            }
        )

    def refund(self, count: int, price: float, *, world: str, round_num: int,
               reason: str) -> None:
        """Return a charge for experiments the lab could not run (lab → agent)."""
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("count must be a non-negative integer")
        cost_m = count * _milli_credits(price, "price")
        if cost_m > self._lab_m:
            raise ValueError("refund exceeds what the lab was paid")
        self._lab_m -= cost_m
        self._balance_m += cost_m
        self.events.append(
            {
                "type": "experiment_refunded",
                "from": "lab",
                "to": f"agent:{self.owner}",
                "amount": cost_m / 1000,
                "count": count,
                "world": world,
                "round": round_num,
                "reason": reason,
            }
        )
