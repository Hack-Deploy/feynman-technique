"""Settle ForceBench attempts through ``dm.settle`` on the oracle's hidden cases.

Run: ``uv run python -m tests.forcebench_settle [--worlds gravity,fractional] [--solvers bayes_lite]``.
There is one preregistration per world (``test_seed=0``), like one posted prize per world.
Each attempt's law is settled, and so is a 1/r-with-fitted-k baseline that carries the same
paid training data, on the same preregistration.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import subprocess
import time
from dataclasses import replace

from dm.settle import prereg_for, settle
from dm.venues.forcebench import WORLDS, run_attempt
from dm.wallet import Wallet
from tests.forcebench_local import (
    BASELINE_INV_R_LAW,
    ROOT,
    SEEDS,
    SOLVERS,
    _top_model,
    identify_model,
)

OUTPUT_PATH = ROOT / "output" / "forcebench_settle.json"
TEST_SEED = 0


def _nmse(verdict: dict) -> float | None:
    value = verdict.get("normalised_mse")
    return value if value is not None and math.isfinite(value) else None


def _cell(cell: tuple[str, str, int]) -> dict:
    solver_name, world, seed = cell
    wallet = Wallet(owner=f"{solver_name}-{world}-{seed}", balance=100.0)
    started = time.perf_counter()
    attempt = run_attempt(solver_name, world, seed, wallet, 1.0)
    run_seconds = time.perf_counter() - started
    prereg = prereg_for("forcebench", world, TEST_SEED)
    started = time.perf_counter()
    record = settle(prereg, attempt)
    settle_seconds = time.perf_counter() - started
    baseline = settle(
        prereg,
        replace(attempt, solver="baseline_inv_r_fit", submitted_law=BASELINE_INV_R_LAW),
    ).verdict
    identified, _ = identify_model(world, _top_model(attempt))
    return {
        "solver": solver_name,
        "world": world,
        "seed": seed,
        "experiments": attempt.experiments,
        "stated_p": attempt.stated_p_success,
        "identified": identified,
        "nmse": _nmse(record.verdict),
        "passed": bool(record.verdict["passed"]),
        "reason": record.verdict.get("reason"),
        "public_tests": record.verdict.get("public_tests"),
        "prereg_commitment": record.verdict.get("prereg_commitment"),
        "attempt_id": record.attempt_id,
        "baseline_nmse": _nmse(baseline),
        "baseline_passed": bool(baseline["passed"]),
        "baseline_reason": baseline.get("reason"),
        "run_seconds": run_seconds,
        "settle_seconds": settle_seconds,
    }


def _fmt(values: list[float | None]) -> str:
    return ", ".join("fail" if v is None else f"{v:.3g}" for v in values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", default=",".join(WORLDS))
    parser.add_argument("--solvers", default=",".join(SOLVERS))
    parser.add_argument("--processes", type=int, default=6)
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()
    solvers, worlds = args.solvers.split(","), args.worlds.split(",")
    cells = [(s, w, seed) for s in solvers for w in worlds for seed in SEEDS]
    with multiprocessing.get_context("spawn").Pool(args.processes) as pool:
        results = pool.map(_cell, cells)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    out = ROOT / args.output if not args.output.startswith("/") else args.output
    with open(out, "w") as handle:
        json.dump({"head": sha, "test_seed": TEST_SEED, "results": results},
                  handle, indent=1, sort_keys=True)
    print(f"scored via dm.settle (head {sha}, test_seed={TEST_SEED})")
    print("| solver | world | pass | id. | mean exp. | nMSE per seed | 1/r+fit pass | 1/r+fit nMSE |")
    print("|---|---|---|---|---|---|---|---|")
    for s in solvers:
        for w in worlds:
            rows = [r for r in results if r["solver"] == s and r["world"] == w]
            n = len(rows)
            print(f"| {s} | {w} | {sum(r['passed'] for r in rows)}/{n} | "
                  f"{sum(r['identified'] for r in rows)}/{n} | "
                  f"{sum(r['experiments'] for r in rows) / n:.1f} | "
                  f"{_fmt([r['nmse'] for r in rows])} | "
                  f"{sum(r['baseline_passed'] for r in rows)}/{n} | "
                  f"{_fmt([r['baseline_nmse'] for r in rows])} |")
    print("reasons:", sorted({str(r["reason"]) for r in results}),
          "baseline reasons:", sorted({str(r["baseline_reason"]) for r in results}),
          "public_tests:", sorted({str(r["public_tests"]) for r in results}))
    print("run s: mean %.1f max %.1f; settle s: mean %.1f max %.1f" % (
        sum(r["run_seconds"] for r in results) / len(results),
        max(r["run_seconds"] for r in results),
        sum(r["settle_seconds"] for r in results) / len(results),
        max(r["settle_seconds"] for r in results)))


if __name__ == "__main__":
    main()
