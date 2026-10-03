"""One real ForceBench attempt, end to end, offline (no LLM, no API spend).

A solver pays per simulator launch, submits a law, and the oracle scores it on
hidden held-out cases fixed before the attempt. Run from the repo root:
    uv run python forcebench_demo.py [world] [solver] [seed]
"""
import sys

from dm.settle import prereg_for, settle
from dm.venues.forcebench import run_attempt
from dm.wallet import Wallet

world = sys.argv[1] if len(sys.argv) > 1 else "gravity"
solver = sys.argv[2] if len(sys.argv) > 2 else "bayes_lite"
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0

prereg = prereg_for("forcebench", world, test_seed=0)
print(f"posted prize for forcebench/{world}: commitment {prereg.commitment()[:16]}… (hidden cases sealed)")

wallet = Wallet(solver, balance=10.0)
start = wallet.balance + wallet.lab_revenue
attempt = run_attempt(solver, world, seed, wallet, price=1.0)
for ev in wallet.events:
    print(f"  {ev['type']}: round {ev.get('round')} amount {ev.get('amount')}")
print(f"{solver} ran {attempt.experiments} paid launches, lab earned {wallet.lab_revenue}, "
      f"stated p(success) = {attempt.stated_p_success:.2f}")
assert wallet.balance + wallet.lab_revenue == start, "credits not conserved"

record = settle(prereg, attempt)
v = record.verdict
print(f"oracle verdict on hidden cases: passed={v.get('passed')} nMSE={v.get('normalised_mse')}")
print(f"commitment matches posted prereg: {v.get('prereg_commitment') == prereg.commitment()}")
print(f"transcript: {record.transcript_path}")
print("submitted law (first lines):")
print("\n".join((record.submitted_law or "").splitlines()[:12]))
