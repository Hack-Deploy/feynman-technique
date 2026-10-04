"""Careful Lab: run a full experiment design once, fit a force law to everything it paid for,
and claim the verdict its own estimates imply."""

import json
import re
import sys
from types import SimpleNamespace

from poc import reference
from poc.pricing import experiment_price

req = json.load(sys.stdin)
bounty = req["bounty"]
world = bounty["world"]


def tag(text, name):
    m = re.search(rf"<{name}>(.*?)</{name}>", text, re.S)
    return m.group(1).strip() if m else None


def price(experiments):
    costs = SimpleNamespace(experiment_costs=req["prices"]["experiment_costs"])
    hyp = SimpleNamespace(particles=bounty.get("particles"))
    # experiment_price reads only these two fields.
    return sum(experiment_price(e, costs, hyp)[0] for e in experiments)  # pyright: ignore[reportArgumentType]


def verdict(values):
    rule = bounty["supported_if"]
    x, lo, hi = values[rule["quantity"]], rule["bounds"][0], rule["bounds"][-1]
    ok = {"above": x > lo, "below": x < lo, "between": lo <= x <= hi,
          "outside": not lo <= x <= hi}[rule["kind"]]
    return "supported" if ok else "refuted"


design = reference.design(world)
outputs = None
for m in req["messages"]:
    raw = tag(m["content"], "experiment_output") if m["role"] == "user" else None
    if raw and raw.startswith("["):
        outputs = json.loads(raw)

if outputs is None:
    cost = price(design) + 2 * req["prices"]["round_fee"]
    plan = {"experiments": design, "controls": [], "analysis": "least-squares fit of a force law"}
    print(f"<assessment>Running a full design: several launches covering the range the bounty "
          f"asks about.</assessment><p_success>0.9</p_success><planned_cost>{cost:g}</planned_cost>"
          f"<plan>{json.dumps(plan)}</plan><run_experiment>{json.dumps(design)}</run_experiment>")
else:
    try:
        est = reference.analyse(world, design, outputs)
        values = {q["name"]: est.values[q["name"]] for q in bounty["quantities"]}
    except (ValueError, KeyError):
        print("<assessment>The fit did not converge.</assessment><p_success>0</p_success>"
              "<verdict>inconclusive</verdict><evidence>The fit did not converge.</evidence>")
        sys.exit(0)
    def line(k, v):
        s = est.sigmas.get(k)
        return f"{k} = {v:.6g}" + (f" ± {s:.3g}" if s is not None and s == s else "")
    lines = "\n".join(line(k, v) for k, v in values.items())
    print(f"<assessment>Fitted every launch I paid for.</assessment><p_success>0.9</p_success>"
          f"<verdict>{verdict(values)}</verdict><estimate>\n{lines}\n</estimate>"
          f"<evidence>A least-squares fit of a force law to all launches gives these values.</evidence>")
