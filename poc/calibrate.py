"""Calibrate tolerances and prizes from the reference design (Phase 2).

For each hypothesis: run the reference design on CALIBRATION_SEEDS (never used by the
benchmark), measure its RMS error per quantity against the truth, and derive

- tolerance = 3 × RMS error, rounded up to 2 significant figures;
- prize = reference cost (experiments + 2 round fees) / TARGET_COST_SHARE, rounded up to 10;
- solvable = the reference lands within tolerance on >= 90% of seeds, and every truth is
  further from the decision boundary than its tolerance.

    uv run python -m poc.calibrate            # print the table, write output/poc_calibration.json
    uv run python -m poc.calibrate --write    # also write tolerances and prizes into config.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import re

import numpy as np

from poc import config as C
from poc import reference, truth
from poc.lab import PositionsOnlyExecutor
from poc.pricing import experiment_price

TARGET_COST_SHARE = 0.3
MIN_PASS_RATE = 0.9
OUT = C.ROOT / "output" / "poc_calibration.json"


def _round_up(x: float, sig: int = 2) -> float:
    if x <= 0:
        return 0.0
    step = 10 ** (math.floor(math.log10(x)) - sig + 1)
    return round(math.ceil(x / step - 1e-9) * step, 12)


def reference_cost(cfg: C.Config, hyp: C.Hypothesis) -> float:
    exps = reference.design(hyp.world)
    return sum(experiment_price(e, cfg, hyp)[0] for e in exps) + 2 * cfg.round_fee


def calibrate(cfg: C.Config, hyp: C.Hypothesis, seeds=reference.CALIBRATION_SEEDS) -> dict:
    from scienceagent.worlds import get_world

    true = truth.true_values(hyp)
    errors: dict[str, list[float]] = {q.name: [] for q in hyp.quantities}
    for seed in seeds:
        world = get_world(hyp.world, engine=C.ENGINE, noise_std=cfg.noise_std, noise_seed=seed)
        lab = PositionsOnlyExecutor(world["executor"])
        design = reference.design(hyp.world)
        est = reference.analyse(hyp.world, design, lab.run(design))
        for q in hyp.quantities:
            errors[q.name].append(est.values[q.name] - true[q.name])
    quantities = {}
    for q in hyp.quantities:
        e = np.asarray(errors[q.name])
        rms = float(np.sqrt(np.mean(e ** 2)))
        tol = _round_up(3 * rms)
        quantities[q.name] = {"rms": rms, "bias": float(e.mean()), "tolerance": tol,
                              "configured_tolerance": q.tolerance,
                              "pass_rate_at_configured": float(np.mean(np.abs(e) <= q.tolerance))}
    rule = hyp.supported_if
    margin = rule.margin(true[rule.quantity])
    hits = np.all([np.abs(np.asarray(errors[q.name])) <= quantities[q.name]["tolerance"]
                   for q in hyp.quantities], axis=0)
    cost = reference_cost(cfg, hyp)
    return {
        "hypothesis": hyp.id, "world": hyp.world, "seeds": list(seeds),
        "quantities": quantities,
        "decision_margin": margin,
        "decision_tolerance": quantities[rule.quantity]["tolerance"],
        "pass_rate": float(hits.mean()),
        "reference_cost": cost,
        "prize": float(math.ceil(cost / TARGET_COST_SHARE / 10) * 10),
        "solvable": bool(hits.mean() >= MIN_PASS_RATE
                         and margin > quantities[rule.quantity]["tolerance"]),
    }


def write_config(results: list[dict], path=C.CONFIG_PATH) -> None:
    """Rewrite the tolerance and prize values in config.yaml, keeping its comments."""
    text = path.read_text()
    for r in results:
        block_start = text.index(f"  - id: {r['hypothesis']}\n")
        nxt = text.find("\n  - id: ", block_start + 1)
        end = len(text) if nxt == -1 else nxt
        block = text[block_start:end]
        for name, q in r["quantities"].items():
            block = re.sub(rf"(\{{name: {re.escape(name)}, .*tolerance: )[0-9.eE+-]+(\}})",
                           rf"\g<1>{q['tolerance']:g}\g<2>", block)
        block = re.sub(r"(\n    prize: )[0-9.]+", rf"\g<1>{r['prize']:g}", block)
        text = text[:block_start] + block + text[end:]
    path.write_text(text)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hypotheses", nargs="+")
    ap.add_argument("--write", action="store_true", help="write tolerances and prizes to config.yaml")
    args = ap.parse_args(argv)
    cfg = C.load()
    hyps = [cfg.hypothesis(h) for h in args.hypotheses] if args.hypotheses else cfg.hypotheses
    results = []
    for h in hyps:
        r = calibrate(cfg, h)
        results.append(r)
        qs = "  ".join(f"{k}: rms {v['rms']:.3g} tol {v['tolerance']:g}"
                       for k, v in r["quantities"].items())
        print(f"{h.id:26s} pass {r['pass_rate']:.2f} margin {r['decision_margin']:.3g} "
              f"cost {r['reference_cost']:g} prize {r['prize']:g} "
              f"{'SOLVABLE' if r['solvable'] else 'DROP'}  {qs}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"written to {OUT.relative_to(C.ROOT)}")
    if args.write:
        write_config(results)
        print("tolerances and prizes written to poc/config.yaml")


if __name__ == "__main__":
    main()
