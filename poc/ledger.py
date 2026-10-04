"""Credits for one run: researcher, escrow, agent and lab, in integer milli-credits.

Every movement is an event. The total over all accounts never changes; ``Ledger.close``
raises if it did or if anything is left in escrow. The same run is settled under both reward
rules, each on its own ledger:

- market: prize_posted → charges → bond_posted (clear claim) → prize_paid + bond_returned
  (confirmed) or prize_refunded + bond_forfeited (otherwise) → calibration_bonus (bid accepted);
- naive: prize_posted → charges → prize_paid for any clear verdict, else prize_refunded.
"""

from __future__ import annotations

import math

from poc import config as C

ACCOUNTS = ("researcher", "escrow", "agent", "lab")
CHARGE_TYPES = ("round_charged", "experiments_charged")


class ConservationError(RuntimeError):
    pass


def _m(x: float) -> int:
    if not math.isfinite(x):
        raise ValueError("amounts must be finite")
    return int(round(x * 1000))


class Ledger:
    def __init__(self, opening: dict[str, float]):
        self.balances = {a: _m(opening.get(a, 0.0)) for a in ACCOUNTS}
        self._total = sum(self.balances.values())
        self.events: list[dict] = []

    def transfer(self, kind: str, frm: str, to: str, amount: float, **meta) -> None:
        m = _m(amount)
        if m < 0:
            frm, to, m = to, frm, -m
        self.balances[frm] -= m
        self.balances[to] += m
        self.events.append({"type": kind, "from": frm, "to": to, "amount": m / 1000, **meta})

    def balance(self, account: str) -> float:
        return self.balances[account] / 1000

    def close(self) -> None:
        if sum(self.balances.values()) != self._total:
            raise ConservationError(f"credits not conserved: {self.balances}")
        if self.balances["escrow"] != 0:
            raise ConservationError(f"escrow not empty: {self.balance('escrow'):g}")


def calibration_bonus(cfg: C.Config, prize: float, p: float | None, confirmed: bool) -> float:
    """cfg.calibration_bonus × prize × (1 − 4(p − y)²): a Brier-based proper score; 0 at p = 0.5,
    positive for well-placed confidence, negative for overconfidence."""
    if p is None:
        return 0.0
    return cfg.calibration_bonus * prize * (1.0 - 4.0 * (p - float(confirmed)) ** 2)


def settle(rule: str, cfg: C.Config, prize: float, charges: list[dict], outcome: str,
           agent_verdict: str | None, confirmed: bool, bid_p: float | None) -> dict:
    """Settle one run under ``rule``. ``charges`` are the run's account events."""
    ledger = Ledger({"researcher": prize})
    ledger.transfer("prize_posted", "researcher", "escrow", prize)
    for e in charges:
        if e["type"] in CHARGE_TYPES and e["amount"]:
            ledger.transfer(e["type"], "agent", "lab", e["amount"], round=e.get("round"))
    clear = outcome == "verdict" and agent_verdict in C.ANSWERS
    bond = bonus = 0.0
    if rule == "naive":
        paid = clear
        ledger.transfer("prize_paid" if paid else "prize_refunded", "escrow",
                        "agent" if paid else "researcher", prize)
    elif rule == "market":
        paid = clear and confirmed
        if clear:
            bond = cfg.claim_bond * prize
            ledger.transfer("bond_posted", "agent", "escrow", bond)
        ledger.transfer("prize_paid" if paid else "prize_refunded", "escrow",
                        "agent" if paid else "researcher", prize)
        if clear:
            ledger.transfer("bond_returned" if paid else "bond_forfeited", "escrow",
                            "agent" if paid else "researcher", bond)
        if outcome not in ("walked_away", "declined"):
            bonus = calibration_bonus(cfg, prize, bid_p, confirmed)
            if bonus:
                ledger.transfer("calibration_bonus", "researcher", "agent", bonus)
    else:
        raise ValueError(f"unknown rule {rule!r}")
    ledger.close()
    return {"rule": rule, "paid": paid, "prize_paid": prize if paid else 0.0,
            "bond_lost": bond if clear and not paid and rule == "market" else 0.0,
            "calibration_bonus": round(bonus, 3), "profit": ledger.balance("agent"),
            "lab_revenue": ledger.balance("lab"), "events": ledger.events}
