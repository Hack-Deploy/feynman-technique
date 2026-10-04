"""Yes Man: buys no experiments and says "supported" at once, with numbers that fit the rule.
Uses only the standard library: the smallest possible agent."""

import json
import sys

req = json.load(sys.stdin)
bounty = req["bounty"]
rule = bounty["supported_if"]
lo, hi = rule["bounds"][0], rule["bounds"][-1]
guess = {"above": lo * 1.5 if lo else 1.0, "below": lo / 2, "between": (lo + hi) / 2,
         "outside": hi * 1.5}[rule["kind"]]
values = {q["name"]: guess if q["name"] == rule["quantity"] else 1.0 for q in bounty["quantities"]}
lines = "\n".join(f"{k} = {v:.6g}" for k, v in values.items())
fee = req["prices"]["round_fee"]

print(f"<assessment>Sounds right to me.</assessment><p_success>0.95</p_success>"
      f"<planned_cost>{fee}</planned_cost><verdict>supported</verdict>"
      f"<estimate>\n{lines}\n</estimate><evidence>The hypothesis is plausible.</evidence>")
