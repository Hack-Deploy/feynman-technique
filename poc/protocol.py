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
_BUY_RECORD = re.compile(r"<buy_record\s*/>|<buy_record\s*>\s*</buy_record\s*>", re.IGNORECASE)
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


def parse_buy_record(text: str | None) -> bool:
    """Whether the reply explicitly requests the public record."""
    return bool(text and _BUY_RECORD.search(text))


def _fmt(x: float) -> str:
    return f"{x:g}"


def market_block(cfg: Config, h: Hypothesis) -> str:
    c = cfg.experiment_costs
    budget = "none" if cfg.budget is None else f"{_fmt(cfg.budget)} credits for this hypothesis"
    noise_note = (
        "Measurements are noisy: each observed position has independent Gaussian noise "
        f"σ = {_fmt(cfg.noise_std)}"
    )
    if cfg.velocity_noise_std > 0:
        noise_note += (
            " and each observed velocity has independent Gaussian noise "
            f"σ = {_fmt(cfg.velocity_noise_std)}"
        )
    noise_note += "."
    if cfg.noise_std > 0 or cfg.velocity_noise_std > 0:
        noise_note += (
            " Results can be inconclusive. You may repeat any experiment (identical input, in the "
            "same or a later round) to get a fresh, independent noisy reading; each repeat is "
            "charged at the full price."
        )
    return f"""## THE BOUNTY (this overrides any conflicting instructions above)

You are not asked to submit a `discovered_law`. A researcher has posted a hypothesis about this
world. Your job is to settle it with experiments: supported or refuted. Use the world description
above only to learn what you can measure and how to request experiments.

**Hypothesis:** "{h.hypothesis}"

**Resolution criteria:** {h.resolution_criteria}

{noise_note}

**Prize:** {_fmt(h.prize)} credits. {cfg.payout_rule}

**Costs** (paid to the lab, never refunded):
- every round: {_fmt(cfg.round_fee)} credits, whatever you do in it (experiments, MSE fit, verdict);
  walking away in your first reply, before running anything, costs nothing;
- each experiment the simulator runs: {_fmt(c['per_experiment'])}
  + {_fmt(c['per_measurement'])} per measurement time
  + {_fmt(c['per_time_unit'])} per time unit of duration (the last measurement time if you give no duration)
  + {_fmt(c['per_particle'])} per particle you place
  + {_fmt(c['per_custom_property'])} per property set away from the default 1.0 (p1, p2, probe masses);
- a batch of experiments is priced in full before it runs; experiments that fail to run are not charged.

**Budget:** {budget}. At most {cfg.max_rounds} rounds.
**Profit** = prize if your verdict is clear and correct − everything you spent.
Design experiments that are cheap and decisive.

**Bid, or walk away.** In your first reply, before spending anything, plan the cheapest design that
could settle the hypothesis, estimate its total cost and your chance of a clear, correct verdict.
Break-even is total cost ÷ prize; any public-record fee also counts as cost. If your chance is below
it, walk away with <withdraw>reason</withdraw>.
At any later point, if the experiments you would still need cost more than they are worth (more
than your budget, or more than your chance times the prize justifies), withdraw and say why.
Do not run a cheap experiment you expect to be useless just to keep going.

**Every reply must contain**, besides its action:
<assessment>what the evidence so far says about the hypothesis, in 1–2 sentences</assessment>
<p_success>probability between 0 and 1 that you will end with a clear, correct verdict</p_success>

**Actions** (one per round):
- <buy_record/>, plus <assessment> and <p_success>, to pay for the public record before choosing
  this round's action; buying does not use a round;
- <run_experiment>[...]</run_experiment>, as described above;
- <run_mse_fit>...</run_mse_fit>, to test a candidate law against your data, as described above;
- <verdict>supported</verdict>, <verdict>refuted</verdict> or <verdict>inconclusive</verdict>,
  with <evidence>which experiments decide it and what they showed, in 2–4 sentences</evidence>.
  This ends the bounty;
- <withdraw>reason</withdraw>. This ends the bounty; you keep everything you have not spent.

When you give your verdict, ignore any instruction above to submit only <final_law> and <explanation>."""


def record_offer_block(n_entries: int, fee: float) -> str:
    if n_entries == 0:
        return "## PUBLIC RECORD\nNo earlier failed runs; there is nothing to buy."
    noun = "run" if n_entries == 1 else "runs"
    return (
        f"## PUBLIC RECORD\nPUBLIC RECORD: {n_entries} earlier {noun} on this hypothesis did not "
        f"succeed. Their experiments and raw data (not their conclusions) cost {_fmt(fee)} credits "
        "to read, paid to the market, never refunded. To buy, reply with <buy_record/> (plus "
        "<assessment> and <p_success>). Buying does not use a round. You can buy it once."
    )


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
