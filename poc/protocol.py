"""What the agent is told and how its replies are read. Pure: no vendor imports.

Flow (slides.html, "How it works"): a hypothesis with resolution criteria and a prize is posted;
the AI scientist designs experiments, states its chance of a clear answer, bids or walks away,
pays the lab, and is paid only for a clear, correct verdict. Failed runs go into the public record.
"""

from __future__ import annotations

import json
import math
import re

from poc.config import VERDICTS, Config, Hypothesis

_FENCES = re.compile(r"```(?:xml|python|json)?\s*\n?|```\s*")
_WITHDRAW = re.compile(r"<withdraw\s*/>|<withdraw>(.*?)</withdraw>", re.DOTALL)
_NUMBER = re.compile(r"^\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*(%?)\s*$")


def extract_tag(text: str | None, tag: str) -> str | None:
    """Content of <tag>...</tag>; also tried with markdown code fences removed (as the vendor does)."""
    if not text:
        return None
    pattern = rf"<{tag}>(.*?)</{tag}>"
    for candidate in (text, _FENCES.sub("", text)):
        m = re.search(pattern, candidate, re.DOTALL)
        if m:
            return m.group(1)
    return None


def _number(raw: str | None) -> tuple[float, bool] | None:
    if raw is None:
        return None
    m = _NUMBER.match(raw)
    if not m:
        return None
    value = float(m.group(1))
    return (value, bool(m.group(2))) if math.isfinite(value) else None


def parse_p(text: str | None) -> float | None:
    """<p_success> as a probability in [0, 1]; accepts '0.7' or '70%'. Anything else is None."""
    n = _number(extract_tag(text, "p_success"))
    if n is None:
        return None
    value = n[0] / 100 if n[1] else n[0]
    return value if 0 <= value <= 1 else None


def parse_text(text: str | None, tag: str) -> str | None:
    raw = extract_tag(text, tag)
    return (raw.strip() or None) if raw is not None else None


def parse_verdict(text: str | None) -> str | None:
    """'supported' | 'refuted' | 'inconclusive' if a <verdict> was given (anything else counts as
    inconclusive), or None if there is no <verdict> tag."""
    raw = extract_tag(text, "verdict")
    if raw is None:
        return None
    v = raw.strip().lower()
    return v if v in VERDICTS else "inconclusive"


def parse_withdraw(text: str | None) -> str | None:
    """Withdrawal reason ('' for a bare <withdraw/>), or None if the agent did not withdraw."""
    if not text:
        return None
    for candidate in (text, _FENCES.sub("", text)):
        m = _WITHDRAW.search(candidate)
        if m:
            return (m.group(1) or "").strip()
    return None


def _fmt(x: float) -> str:
    return f"{x:g}"


_ESTIMATE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*[=:]\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
    r"(?:\s*(?:±|\+/-|\+-)\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?))?\s*$")


def parse_estimates(text: str | None) -> dict[str, dict]:
    """Every ``<estimate>name = value ± sigma</estimate>`` (σ optional; several lines per tag
    allowed) as {name: {"value": v, "sigma": s or None}}. Later tags win."""
    out: dict[str, dict] = {}
    if not text:
        return out
    for candidate in (text, _FENCES.sub("", text)):
        for body in re.findall(r"<estimate>(.*?)</estimate>", candidate, re.DOTALL):
            for line in re.split(r"[\n;,]+(?=\s*[A-Za-z_])", body):
                m = _ESTIMATE.match(line)
                if not m:
                    continue
                value = float(m.group(2))
                sigma = float(m.group(3)) if m.group(3) else None
                if math.isfinite(value):
                    out[m.group(1)] = {"value": value, "sigma": sigma}
        if out:
            break
    return out


def parse_planned_cost(text: str | None) -> float | None:
    n = _number(extract_tag(text, "planned_cost"))
    return None if n is None or n[1] else n[0]


def parse_plan(text: str | None) -> dict | None:
    """The preregistered plan: JSON with "experiments", optional "controls" (indices into
    experiments) and "analysis". Returns {"raw": text} if it is not valid JSON."""
    raw = extract_tag(text, "plan")
    if raw is None:
        return None
    try:
        plan = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {"raw": raw.strip()}
    return plan if isinstance(plan, dict) else {"raw": raw.strip()}


def bids(p: float | None, planned_cost: float | None, prize: float) -> bool:
    """The bid rule, applied in code with the agent's own numbers: bid only if p × prize > cost."""
    return p is not None and planned_cost is not None and p * prize > planned_cost


def _rule_text(h: Hypothesis) -> str:
    r = h.supported_if
    bounds = {"above": f"above {_fmt(r.bounds[0])}", "below": f"below {_fmt(r.bounds[0])}",
              "between": f"between {_fmt(r.bounds[0])} and {_fmt(r.bounds[-1])}",
              "outside": f"below {_fmt(r.bounds[0])} or above {_fmt(r.bounds[-1])}"}[r.kind]
    return f"supported if your estimate of {r.quantity} is {bounds}; refuted otherwise"


def quantities_block(h: Hypothesis) -> str:
    lines = [f"- `{q.name}`: {q.meaning}. Tolerance ±{_fmt(q.tolerance)}." for q in h.quantities]
    example = "\n".join(f"{q.name} = <value> ± <your 1σ uncertainty>" for q in h.quantities)
    return ("\n".join(lines) + f"\n\n**Decision rule:** {_rule_text(h)}.\n\n"
            f"Claim format, one line per quantity:\n<estimate>\n{example}\n</estimate>")


def payout_text(cfg: Config, h: Hypothesis, rule: str) -> str:
    if rule == "naive":
        return cfg.naive_payout_rule
    return cfg.market_payout_rule.replace("{bond}", _fmt(cfg.claim_bond * h.prize))


def market_block(cfg: Config, h: Hypothesis, rule: str = "market",
                 experiments: bool = True) -> str:
    c = cfg.experiment_costs
    budget = "none" if cfg.budget is None else f"{_fmt(cfg.budget)} credits for this hypothesis"
    if not experiments:
        return f"""## THE BOUNTY (this overrides any conflicting instructions above)

You are not asked to submit a `discovered_law`, and you cannot run experiments or MSE fits.
A researcher has posted a hypothesis about this world. Answer from what you already know, in
ONE reply.

**Hypothesis:** "{h.hypothesis}"

**Resolution criteria:** {h.resolution_criteria}

**Quantities to estimate:**
{quantities_block(h)}

**Prize:** {_fmt(h.prize)} credits. {payout_text(cfg, h, rule)}

Reply with <assessment>1–2 sentences</assessment>, <p_success>probability between 0 and 1 that
your claim will be confirmed</p_success>, <planned_cost>0</planned_cost>, your <estimate>
block, and <verdict>supported</verdict>, <verdict>refuted</verdict> or
<verdict>inconclusive</verdict> with <evidence>your reasoning in 2–4 sentences</evidence>.
Or walk away with <withdraw>reason</withdraw>."""

    pay = ("**Profit** = prize if your claim is confirmed − everything you spent − the bond if "
           "your claim is false." if rule == "market" else
           "**Profit** = prize for any clear verdict − everything you spent.")
    return f"""## THE BOUNTY (this overrides any conflicting instructions above)

You are not asked to submit a `discovered_law`. A researcher has posted a hypothesis about this
world. Your job is to settle it with experiments, by estimating the quantities below.
Use the world description above only to learn what you can measure and how to request
experiments.

**Hypothesis:** "{h.hypothesis}"

**Resolution criteria:** {h.resolution_criteria}

**Quantities to estimate:**
{quantities_block(h)}

**Prize:** {_fmt(h.prize)} credits. {payout_text(cfg, h, rule)}

**Costs** (paid to the lab, never refunded):
- every round: {_fmt(cfg.round_fee)} credits, whatever you do in it (experiments, MSE fit, verdict);
  if your bid is not accepted, or you walk away in your first reply, you pay nothing;
- each experiment the simulator runs: {_fmt(c['per_experiment'])}
  + {_fmt(c['per_measurement'])} per measurement time
  + {_fmt(c['per_time_unit'])} per time unit of duration (the last measurement time if you give no duration)
  + {_fmt(c['per_particle'])} per particle you place
  + {_fmt(c['per_custom_property'])} per property set away from the default 1.0 (p1, p2, probe masses);
- a batch of experiments is priced in full before it runs; experiments that fail to run are not charged.

**Budget:** {budget}. At most {cfg.max_rounds} rounds.
{pay}
Design experiments that are cheap and decisive.

**Bid, or walk away.** Your FIRST reply is your bid. Before spending anything, preregister the
cheapest design that could settle the hypothesis and state:
<p_success>probability between 0 and 1 that your claim will be {"confirmed" if rule == "market" else "a clear verdict"}</p_success>
<planned_cost>total credits you expect to spend, round fees included</planned_cost>
<plan>{{"experiments": [the experiments you plan to run, in the lab's JSON format],
"controls": [indices of the planned experiments that are controls], "analysis": "how you will
compute each quantity from the data"}}</plan>
The market accepts your bid only if p_success × prize > planned_cost; otherwise the run ends
and you pay nothing. You may also walk away with <withdraw>reason</withdraw>. If your first
reply also contains an action, it runs only if the bid is accepted.
At any later point, if the experiments you would still need cost more than they are worth,
give <verdict>inconclusive</verdict> or withdraw.

**Every reply must contain**, besides its action:
<assessment>what the evidence so far says about the hypothesis, in 1–2 sentences</assessment>
<p_success>your current probability that your claim will be {"confirmed" if rule == "market" else "a clear verdict"}</p_success>

**Actions** (one per round):
- <run_experiment>[...]</run_experiment>, as described above;
- <run_mse_fit>...</run_mse_fit>, to test a candidate law against your data, as described above;
- <verdict>supported</verdict> or <verdict>refuted</verdict> with your <estimate> block, or
  <verdict>inconclusive</verdict> ("I don't know, stopping"), plus <evidence>which experiments
  decide it and what they showed, in 2–4 sentences</evidence>. This ends the bounty;
- <withdraw>reason</withdraw>. This ends the bounty; you keep everything you have not spent.

When you give your verdict, ignore any instruction above to submit only <final_law> and <explanation>."""


def status_line(round_num: int, cfg: Config, round_fee: float, experiments_cost: float,
                spent: float, remaining: float | None) -> str:
    left = cfg.max_rounds - round_num
    budget = "" if remaining is None else f" Budget left: {_fmt(remaining)}."
    return (f"[Lab] Round {round_num} of {cfg.max_rounds} done. Charged this round: "
            f"{_fmt(round_fee + experiments_cost)} (round fee {_fmt(round_fee)} + experiments "
            f"{_fmt(experiments_cost)}). Total spent: {_fmt(spent)}. Rounds left: {left}.{budget}")


CONFIDENCE_REPROMPT = (
    "Your reply is missing <assessment> or a valid <p_success> (a number between 0 and 1). "
    "Reply NOW with ONLY:\n<assessment>...</assessment>\n<p_success>...</p_success>")

BID_REPROMPT = (
    "Your first reply is your bid, and it is missing a valid <p_success> (between 0 and 1) or "
    "<planned_cost> (credits). Reply NOW with ONLY:\n<assessment>...</assessment>\n"
    "<p_success>...</p_success>\n<planned_cost>...</planned_cost>\n<plan>...</plan>\n"
    "plus the action you want to take if the bid is accepted, or <withdraw>reason</withdraw>.")


def estimate_reprompt(names: list[str]) -> str:
    lines = "\n".join(f"{n} = <value> ± <1σ>" for n in names)
    return ("Your verdict is recorded, but your <estimate> block is missing or incomplete "
            f"(needed: {', '.join(names)}). Reply NOW with ONLY:\n<estimate>\n{lines}\n</estimate>")

EVIDENCE_REPROMPT = (
    "Your verdict is recorded, but you did not include <evidence>. Reply NOW with ONLY "
    "<evidence>...</evidence>: which experiments decide it and what they showed, in 2–4 sentences.")


# ------------------------------------------------------------------ public record of failed runs

def is_failure(record: dict) -> bool:
    return not (record.get("verdict") or {}).get("passed", False)


def ledger_entry(record: dict, max_data_chars: int) -> dict:
    """The public part of a failed or withdrawn run: what was done and what it cost, plus the raw
    experiment data. Never the verdict, evidence or assessment: for a wrong verdict they would
    reveal the answer."""
    extra = record.get("extra") or {}
    data = json.dumps(extra.get("runs") or [], separators=(",", ":"))
    if len(data) > max_data_chars:
        data = data[:max_data_chars] + " ... (cut)"
    return {
        "id": record["attempt_id"],
        "model": record["solver"],
        "outcome": extra.get("outcome"),
        "rounds": record["rounds"],
        "experiments": record["experiments"],
        "spent": record["lab_cost"],
        "p_success": extra.get("final_p"),
        "withdraw_reason": extra.get("withdraw_reason"),
        "data": data,
    }


_OUTCOME_TEXT = {
    "verdict": "no clear result",
    "walked_away": "walked away before running anything",
    "declined": "bid not accepted (p_success × prize ≤ planned cost)",
    "withdrawn": "withdrew",
    "out_of_rounds": "ran out of rounds",
}


def ledger_block(entries: list[dict]) -> str:
    head = ("## PUBLIC RECORD: earlier runs on this hypothesis that did not succeed\n"
            "(Other AI scientists' runs. Their experiments and data are shown; their conclusions "
            "are not.)\n")
    if not entries:
        return head + "\nNo earlier failed runs."
    lines = [head]
    for i, e in enumerate(entries, 1):
        p = "n/a" if e["p_success"] is None else _fmt(e["p_success"])
        line = (f"{i}. {e['model']}: {_OUTCOME_TEXT.get(e['outcome'], e['outcome'])} after "
                f"{e['rounds']} round(s), {e['experiments']} experiment(s), spent {_fmt(e['spent'])}, "
                f"final stated p {p}.")
        if e["withdraw_reason"]:
            line += f" Reason given: {e['withdraw_reason']}"
        lines.append(line)
        if e["experiments"]:
            lines.append(f"   experiments and data: {e['data']}")
    return "\n".join(lines)
