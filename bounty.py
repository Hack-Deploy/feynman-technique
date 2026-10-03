"""Discovery Market – post a bounty, get a bid.

A researcher submits a hypothesis, a pre-registered success criterion and a
prize. Claude, acting as the AI scientist, proposes a protocol priced from a
fixed lab price list, states its probability of a clear answer, and flags
biosafety. The market rule (bid only if p × prize > cost, and never above
BSL-2) is applied here in code, so the model cannot argue its way into a bid.

Without ANTHROPIC_API_KEY (environment or .env), a saved example is returned.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = "claude-opus-5-5"
MAX_BSL = 2

# Illustrative cloud-lab prices in GBP. The model may only use these items.
PRICE_LIST = {
    "96-well plate with media and culture": 6.00,
    "Plate reader time, per hour": 1.50,
    "Technician time, per 15 minutes": 6.00,
    "Compound handling and dilution series, per compound": 4.00,
    "qPCR run, per 96-well plate": 40.00,
    "Fluorescence microscopy, per hour": 25.00,
    "Cell culture flask, mammalian line, per flask": 15.00,
    "Short-read sequencing, per sample": 60.00,
    "Western blot, per gel": 30.00,
}

SYSTEM = f"""You are an AI scientist bidding in a research market. A researcher has posted a
hypothesis, a pre-registered success criterion and a prize. You pay for the lab run yourself
and receive the prize only if the result meets the criterion in either direction (confirmed or
refuted). An inconclusive result earns nothing.

Design the cheapest experiment that would give a clear answer. Price it using ONLY items from
this lab price list (GBP), with whole-number quantities:
{json.dumps(PRICE_LIST, indent=2)}

Then estimate p_clear_answer: the probability that this exact protocol produces a result that
meets the success criterion, in either direction. Be calibrated. Overconfidence loses you money,
so account for failure modes such as contamination, assay noise, too few replicates and a
criterion the assay cannot measure. If the hypothesis cannot be tested with the price list, say
so and give a low probability.

State the biosafety level the work needs. Anything above BSL-{MAX_BSL}, human subjects or animal
work is out of scope for this lab."""

SCHEMA = {
    "type": "object",
    "properties": {
        "protocol_steps": {"type": "array", "items": {"type": "string"}},
        "controls": {"type": "array", "items": {"type": "string"}},
        "replicates": {"type": "integer"},
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item": {"type": "string", "enum": list(PRICE_LIST)},
                    "quantity": {"type": "integer"},
                },
                "required": ["item", "quantity"],
                "additionalProperties": False,
            },
        },
        "p_clear_answer": {"type": "number"},
        "why_this_probability": {"type": "string"},
        "main_risks": {"type": "array", "items": {"type": "string"}},
        "result_that_would_refute": {"type": "string"},
        "biosafety_level": {"type": "integer"},
        "safety_notes": {"type": "string"},
    },
    "required": [
        "protocol_steps", "controls", "replicates", "line_items", "p_clear_answer",
        "why_this_probability", "main_risks", "result_that_would_refute",
        "biosafety_level", "safety_notes",
    ],
    "additionalProperties": False,
}

EXAMPLE_REQUEST = {
    "hypothesis": "Compound DM-17 inhibits growth of E. coli K-12 at 10 µg/mL.",
    "criterion": "OD600 at 12 h at least 50% below vehicle control (p < 0.05), with ampicillin "
                 "and vehicle controls behaving as expected. A clear failure to reach 50% also counts.",
    "prize": 500,
}

EXAMPLE_DESIGN = {
    "protocol_steps": [
        "Prepare DM-17 at 0, 1, 10 and 100 µg/mL in LB.",
        "Inoculate E. coli K-12 at OD600 0.05 into a 96-well plate, three biological replicates per condition.",
        "Add ampicillin (100 µg/mL) as positive control and vehicle alone as negative control.",
        "Read OD600 every hour for 12 h at 37 °C with shaking.",
        "Compare 12 h OD600 at 10 µg/mL against vehicle with a two-sided t-test.",
    ],
    "controls": ["Ampicillin 100 µg/mL (should block growth)", "Vehicle only (should grow normally)", "Media-only blank wells"],
    "replicates": 3,
    "line_items": [
        {"item": "96-well plate with media and culture", "quantity": 1},
        {"item": "Plate reader time, per hour", "quantity": 12},
        {"item": "Technician time, per 15 minutes", "quantity": 2},
    ],
    "p_clear_answer": 0.85,
    "why_this_probability": "A standard growth assay with both controls. Most failure risk is "
                            "contamination or edge effects, which the blanks and replicates catch.",
    "main_risks": ["Plate contamination", "Compound precipitating at 100 µg/mL", "Evaporation in edge wells"],
    "result_that_would_refute": "12 h OD600 at 10 µg/mL within 50% of vehicle, with controls working.",
    "biosafety_level": 1,
    "safety_notes": "E. coli K-12 is a BSL-1 lab strain.",
}


def api_key() -> str | None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "ANTHROPIC_API_KEY" and value.strip():
                return value.strip().strip('"').strip("'")
    return None


def ask_claude(hypothesis: str, criterion: str, prize: float, api_key: str) -> dict:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        system=SYSTEM,
        messages=[{
            "role": "user",
            "content": f"Hypothesis: {hypothesis}\n\nSuccess criterion: {criterion}\n\nPrize: £{prize:g}",
        }],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("The model declined to design this experiment.")
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def decide(design: dict, prize: float) -> dict:
    """Apply the market rule to a design. Pure, so it is easy to test."""
    items = [
        {**li, "unit_cost": PRICE_LIST[li["item"]], "cost": PRICE_LIST[li["item"]] * li["quantity"]}
        for li in design["line_items"]
    ]
    cost = round(sum(li["cost"] for li in items), 2)
    p = min(max(float(design["p_clear_answer"]), 0.0), 1.0)
    break_even = cost / prize if prize > 0 else 1.0
    safe = design["biosafety_level"] <= MAX_BSL
    bids = safe and p * prize > cost
    if not safe:
        reason = f"No bid: the work needs BSL-{design['biosafety_level']}, beyond this lab's BSL-{MAX_BSL} limit."
    elif bids:
        reason = f"Bid: {p:.0%} odds on £{prize:g} are worth £{p * prize:.0f}, more than the £{cost:.0f} run."
    else:
        reason = f"No bid: {p:.0%} odds on £{prize:g} are worth £{p * prize:.0f}, less than the £{cost:.0f} run."
    return {
        "line_items": items, "cost": cost, "p": p, "break_even": round(break_even, 4),
        "expected_value": round(p * prize, 2), "safe": safe, "bids": bids, "reason": reason,
    }


def post_bounty(hypothesis: str, criterion: str, prize: float) -> dict:
    hypothesis, criterion = hypothesis.strip(), criterion.strip()
    if not hypothesis or not criterion or prize <= 0:
        return {"ok": False, "error": "A hypothesis, a success criterion and a positive prize are all required."}
    key = api_key()
    if key:
        try:
            design = ask_claude(hypothesis, criterion, prize, key)
            source = "live"
        except Exception as e:  # surface API trouble in the UI instead of crashing the demo
            return {"ok": False, "error": f"Claude API call failed: {e}"}
    else:
        # No key: show the saved DM-17 example as itself, never as an answer to the user's text.
        design = EXAMPLE_DESIGN
        source = "example"
        hypothesis, criterion, prize = (
            EXAMPLE_REQUEST["hypothesis"], EXAMPLE_REQUEST["criterion"], EXAMPLE_REQUEST["prize"],
        )
    return {
        "ok": True, "source": source, "model": MODEL if source == "live" else None,
        "request": {"hypothesis": hypothesis, "criterion": criterion, "prize": prize},
        "design": design, "decision": decide(design, prize),
    }
