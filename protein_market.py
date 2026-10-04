"""Discovery Market – peptide-HLA venue.

The same market rules as the physics venue, on a question answered by a real
measurement instead of a simulation: does one peptide bind one HLA allele for
longer than an hour?

What the solver buys is evidence about *other* peptides. The row it is being
judged on is removed from the lab before the first round, so the answer cannot
be looked up - only predicted. The judge compares the solver's verdict with the
withheld measurement.

Data: Track 3 peptide-HLA stability set (data/protein/track3_raw.csv),
28,165 measured 9-mers. Claims and their hidden answers: data/protein/claims.json.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "protein"
THRESHOLD_H = 1.0
MAX_ROUNDS = 6
BUDGET = 100.0
PRIZE = 100.0
ROUND_FEE = 5.0
PER_LOOKUP = 2.0
MAX_LOOKUPS_PER_ROUND = 25
MODEL_DEFAULT = "claude-opus-5-5"
VERDICTS = ("supported", "refuted")

_jobs: dict[str, dict] = {}
_lock = threading.Lock()


# --------------------------------------------------------------------------- data

def claims() -> list[dict]:
    return json.loads((DATA / "claims.json").read_text())


def claim(claim_id: str) -> dict:
    for c in claims():
        if c["id"] == claim_id:
            return c
    raise KeyError(claim_id)


def public_claim(c: dict) -> dict:
    """Everything the solver may see: never the measurement or the answer."""
    return {k: c[k] for k in ("id", "allele", "peptide", "claim", "criteria", "prize")}


@dataclass
class Lab:
    """Measured half-lives for one allele, with the judged peptide withheld."""
    allele: str
    withheld: str
    rows: dict[str, float]

    @classmethod
    def for_claim(cls, c: dict) -> "Lab":
        import pandas as pd

        d = pd.read_csv(DATA / "track3_raw.csv")
        d = d[(d.allele == c["allele"]) & (d.peptide != c["peptide"])]
        return cls(allele=c["allele"], withheld=c["peptide"],
                   rows=dict(zip(d.peptide, d.thalf_hours.astype(float))))

    def catalogue(self, n: int = 40) -> list[str]:
        """A deterministic sample of peptide names the solver may buy."""
        return sorted(self.rows)[:: max(1, len(self.rows) // n)][:n]

    def lookup(self, peptides: list[str]) -> tuple[dict, list[str]]:
        found, missing = {}, []
        for p in peptides[:MAX_LOOKUPS_PER_ROUND]:
            key = str(p).strip().upper()
            if key == self.withheld:
                missing.append(f"{key} (withheld: this is the peptide under test)")
            elif key in self.rows:
                found[key] = self.rows[key]
            else:
                missing.append(key)
        return found, missing


# --------------------------------------------------------------------------- market

@dataclass
class Attempt:
    claim: dict
    model: str
    scripted: bool
    balance: float = BUDGET
    spent: float = 0.0
    rounds: list[dict] = field(default_factory=list)
    bought: dict[str, float] = field(default_factory=dict)
    verdict: str | None = None
    stated_p: float | None = None
    outcome: str = "running"

    def charge(self, amount: float, reason: str) -> bool:
        if amount > self.balance:
            return False
        self.balance -= amount
        self.spent += amount
        return True

    def settle(self) -> dict:
        correct = self.verdict == self.claim["answer"] if self.verdict else None
        payout = PRIZE if correct else 0.0
        self.balance += payout
        return {
            "verdict": self.verdict, "answer": self.claim["answer"], "correct": correct,
            "measured_thalf_hours": self.claim["measured_thalf_hours"],
            "prize": payout, "spent": round(self.spent, 2),
            "net": round(payout - self.spent, 2), "balance": round(self.balance, 2),
            "stated_p": self.stated_p, "outcome": self.outcome,
        }


SYSTEM = """You are an AI scientist bidding in a research market.

A researcher has posted a claim about one peptide-HLA pair and a prize. You pay for every
round and for every measurement you look up, out of your own credits. You win the prize only
if your final verdict matches the withheld measurement.

Each round, reply with exactly these tags:

<assessment>what the evidence so far tells you</assessment>
<confidence>a number from 0 to 1: your probability that the verdict you would give now is correct</confidence>
then EITHER
<lookup>PEPTIDE1, PEPTIDE2, ...</lookup>   to buy measured half-lives for other peptides on this allele
OR
<verdict>supported</verdict>  or  <verdict>refuted</verdict>   to answer and end the attempt
OR
<walk_away>why</walk_away>   to stop and keep your remaining credits

Be calibrated. Saying you are confident costs nothing, but being wrong costs you the prize and
everything you spent. The peptide under test is withheld from the lab, so you must predict it
from the others."""


def _prompt(att: Attempt, lab: Lab, first: bool) -> str:
    c = public_claim(att.claim)
    lines = [
        f"CLAIM: {c['claim']}",
        f"HOW IT WILL BE JUDGED: {c['criteria']}",
        f"PRIZE: {c['prize']:g} credits",
        "",
        f"Round {len(att.rounds) + 1} of {MAX_ROUNDS}. "
        f"Balance {att.balance:g} credits. Round fee {ROUND_FEE:g}, "
        f"each lookup {PER_LOOKUP:g} (max {MAX_LOOKUPS_PER_ROUND} per round).",
    ]
    if first:
        lines += ["", f"Peptides available on {lab.allele} include: "
                      + ", ".join(lab.catalogue()) + ". You may name any 9-mer on this allele."]
    if att.bought:
        lines += ["", "Measurements you have bought (peptide: half-life in hours):"]
        lines += [f"  {p}: {v:g}" for p, v in sorted(att.bought.items())]
    return "\n".join(lines)


def _tag(reply: str, name: str) -> str | None:
    m = re.search(rf"<{name}>(.*?)</{name}>", reply, re.S | re.I)
    return m.group(1).strip() if m else None


def _parse(reply: str) -> dict:
    p = _tag(reply, "confidence")
    try:
        conf = min(max(float(re.search(r"[\d.]+", p).group()), 0.0), 1.0) if p else None
    except (AttributeError, ValueError):
        conf = None
    verdict = (_tag(reply, "verdict") or "").strip().lower() or None
    lookup = _tag(reply, "lookup")
    return {
        "assessment": _tag(reply, "assessment") or "",
        "confidence": conf,
        "verdict": verdict if verdict in VERDICTS else None,
        "lookup": [x.strip() for x in re.split(r"[,\s]+", lookup) if x.strip()] if lookup else [],
        "walk_away": _tag(reply, "walk_away"),
    }


def _scripted(att: Attempt, lab: Lab) -> str:
    """Free demo solver: buy once, then answer from the sample mean. No API call."""
    if not att.bought:
        return ("<assessment>No data yet. Buying a spread of measured peptides on this "
                "allele to see the baseline.</assessment><confidence>0.5</confidence>"
                f"<lookup>{', '.join(lab.catalogue(12))}</lookup>")
    vals = list(att.bought.values())
    frac = sum(v > THRESHOLD_H for v in vals) / len(vals)
    verdict = "supported" if frac > 0.5 else "refuted"
    return (f"<assessment>{frac:.0%} of the {len(vals)} peptides I bought are above "
            f"{THRESHOLD_H:g} h, so I go with the majority for this allele.</assessment>"
            f"<confidence>{max(frac, 1 - frac):.2f}</confidence><verdict>{verdict}</verdict>")


def _ask_claude(att: Attempt, lab: Lab, first: bool, api_key: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    messages = [{"role": "user", "content": _prompt(att, lab, first)}]
    response = client.beta.messages.create(
        model=att.model, max_tokens=4096,
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        output_config={"effort": "medium"},
        system=SYSTEM, messages=messages,
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("the model declined this request")
    return "".join(b.text for b in response.content if b.type == "text")


def api_key() -> str | None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    for env in (ROOT / "poc" / ".env", ROOT / ".env"):
        if env.exists():
            for line in env.read_text().splitlines():
                k, _, v = line.partition("=")
                if k.strip() == "ANTHROPIC_API_KEY" and v.strip():
                    return v.strip().strip("\"'")
    return None


def live_enabled() -> bool:
    try:
        return (api_key() is not None and os.environ.get("ENABLE_LIVE") == "1"
                and float(os.environ.get("DM_MAX_USD", "0")) > 0)
    except ValueError:
        return False


def run(claim_id: str, model: str = MODEL_DEFAULT, scripted: bool = True,
        on_round=None) -> dict:
    c = claim(claim_id)
    lab = Lab.for_claim(c)
    att = Attempt(claim=c, model="scripted" if scripted else model, scripted=scripted)
    key = None if scripted else api_key()
    if not scripted and not live_enabled():
        raise RuntimeError("live runs need ANTHROPIC_API_KEY, ENABLE_LIVE=1 and DM_MAX_USD")

    for i in range(MAX_ROUNDS):
        if not att.charge(ROUND_FEE, "round"):
            att.outcome = "out_of_credits"
            break
        reply = _scripted(att, lab) if scripted else _ask_claude(att, lab, i == 0, key)
        step = _parse(reply)
        entry = {"round": i + 1, "assessment": step["assessment"],
                 "confidence": step["confidence"], "action": None,
                 "cost": ROUND_FEE, "balance": round(att.balance, 2)}
        if step["confidence"] is not None:
            att.stated_p = step["confidence"]

        if step["verdict"]:
            att.verdict = step["verdict"]
            att.outcome = "verdict"
            entry["action"] = "verdict"
            entry["verdict"] = step["verdict"]
        elif step["walk_away"]:
            att.outcome = "walk_away"
            entry["action"] = "walk_away"
            entry["reason"] = step["walk_away"]
        elif step["lookup"]:
            names = step["lookup"][:MAX_LOOKUPS_PER_ROUND]
            cost = PER_LOOKUP * len(names)
            if not att.charge(cost, "lookups"):
                entry["action"] = "lookup_refused"
                entry["note"] = "not enough credits for that many lookups"
            else:
                found, missing = lab.lookup(names)
                att.bought.update(found)
                entry.update(action="lookup", bought=found, missing=missing,
                             cost=ROUND_FEE + cost, balance=round(att.balance, 2))
        else:
            entry["action"] = "no_action"
            entry["note"] = "reply had no lookup, verdict or walk_away tag"

        att.rounds.append(entry)
        if on_round:
            on_round(entry)
        if att.outcome in ("verdict", "walk_away"):
            break
    else:
        att.outcome = att.outcome if att.outcome != "running" else "out_of_rounds"

    return {
        "claim": public_claim(c), "difficulty": c["difficulty"],
        "allele_frac_above": c["allele_frac_above"],
        "base_rate_answer": c["base_rate_answer"],
        "model": att.model, "scripted": scripted,
        "rounds": att.rounds, "result": att.settle(),
        "lab_size": len(lab.rows),
    }


def distribution(claim_id: str) -> dict:
    """Every measured half-life on this allele except the judged one.

    No leak: these are exactly the rows a solver may buy, so showing them
    reveals nothing it could not pay for. The hidden value is not included.
    """
    c = claim(claim_id)
    lab = Lab.for_claim(c)
    values = sorted(round(v, 2) for v in lab.rows.values())
    above = sum(v > THRESHOLD_H for v in values)
    return {
        "claim": public_claim(c), "threshold": THRESHOLD_H,
        "values": values, "n": len(values),
        "frac_above": round(above / len(values), 3) if values else 0.0,
    }


# --------------------------------------------------------------------------- app API

def info() -> dict:
    return {
        "claims": [{**public_claim(c), "difficulty": c["difficulty"]} for c in claims()],
        "live": live_enabled(), "model": MODEL_DEFAULT,
        "prices": {"round_fee": ROUND_FEE, "per_lookup": PER_LOOKUP,
                   "budget": BUDGET, "prize": PRIZE, "max_rounds": MAX_ROUNDS},
    }


def start(claim_id: str, model: str, scripted: bool) -> dict:
    claim(claim_id)  # raises for an unknown id
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[job_id] = {"state": "running", "rounds": [], "result": None, "error": None}

    def worker():
        try:
            def on_round(entry):
                with _lock:
                    _jobs[job_id]["rounds"].append(entry)
            out = run(claim_id, model=model, scripted=scripted, on_round=on_round)
            with _lock:
                _jobs[job_id].update(state="done", **{k: out[k] for k in ("rounds", "result")},
                                     run=out)
        except Exception as exc:
            with _lock:
                _jobs[job_id].update(state="error", error=str(exc)[:300])

    threading.Thread(target=worker, daemon=True).start()
    return {"job": job_id}


def job(job_id: str) -> dict | None:
    with _lock:
        j = _jobs.get(job_id)
        return dict(j) if j else None
