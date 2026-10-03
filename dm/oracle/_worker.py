"""Trusted scoring worker: holds the ground truth, never runs submitted code.

Reads a JSON job on stdin, writes one JSON result on stdout. Launched by
``dm.oracle.score`` with a wall-clock timeout and a scrubbed environment.

1. Builds the world and computes the noise-free ground truth on the hidden cases.
2. Starts ``dm.oracle._sandbox`` with the cases and *zeroed* stand-in trajectories
   (same shapes and agent-visible fields, no answers). The sandbox runs the law
   through the vendor evaluator and returns its predictions.
3. Computes the squared position errors here, exactly as the vendor evaluators
   do (same particles, same averaging), so ``mean_pos_error`` is unchanged.
"""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

# Trajectory keys holding answers; zeroed before anything reaches the sandbox.
SECRET_KEYS = ("pos1", "pos2", "velocity1", "velocity2", "positions", "velocities",
               "dark_initial_positions")
# Agent-visible executor attributes the vendor evaluators read.
VISIBLE_ATTRS = ("N_BACKGROUND", "N_PROBES", "N_VISIBLE", "DEFAULT_PROBE_MASS",
                 "_bg_positions_rel", "_bg_velocities", "_bg_masses", "_visible_velocities")
# Scored particles per world (mirrors the vendor evaluators).
PROBE_SLICES = {"dark_matter": slice(20, 25), "ether": slice(21, 26), "hubble": slice(21, 26)}
TWO_PARTICLE = {"gravity", "yukawa", "coulomb_easy", "oscillator", "fractional",
                "extra_dimensions"}


def _zeroed(output: dict) -> dict:
    out = copy.deepcopy(output)
    for k in SECRET_KEYS:
        if k in out:
            out[k] = np.zeros_like(np.asarray(out[k], dtype=float)).tolist()
    if "field_snapshots" in out:  # the hidden field itself; evaluators only plot it
        out["field_snapshots"] = []
    return out


def _attrs(executor) -> dict:
    attrs = {}
    for name in VISIBLE_ATTRS:
        if hasattr(executor, name):
            v = getattr(executor, name)
            attrs[name] = np.asarray(v).tolist() if not isinstance(v, (int, float)) else v
    return attrs


def _case_errors(world: str, gt: dict, pred) -> list[float]:
    """Squared L2 errors for one case, as the vendor evaluator for ``world`` computes them."""
    inf = [float("inf")]
    if pred is None:
        return inf
    try:
        if world in TWO_PARTICLE:
            truth = np.asarray(gt["pos2"], dtype=float)                   # (T, 2)
            p = np.asarray(pred, dtype=float).reshape(truth.shape)
            errs = np.sum((p - truth) ** 2, axis=-1)                      # per time
        else:
            truth = np.asarray(gt["positions"], dtype=float)              # (T, N, 2)
            p = np.asarray(pred, dtype=float).reshape(truth.shape)
            sl = PROBE_SLICES.get(world, slice(None))
            errs = np.sum((p[:, sl] - truth[:, sl]) ** 2, axis=-1).ravel()  # per (time, particle)
    except (ValueError, TypeError):
        return inf
    errs = errs.tolist()
    return errs if all(math.isfinite(e) for e in errs) else inf


def _run_sandbox(job: dict, timeout_s: float) -> dict:
    env = {k: v for k, v in os.environ.items()
           if not any(m in k.upper() for m in ("KEY", "TOKEN", "SECRET", "PASSWORD"))}
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    with tempfile.TemporaryDirectory(prefix="dm-sandbox-") as cwd:
        try:
            proc = subprocess.run([sys.executable, "-m", "dm.oracle._sandbox"],
                                  input=json.dumps(job), capture_output=True, text=True,
                                  timeout=max(timeout_s, 1.0), cwd=cwd, env=env)
        except subprocess.TimeoutExpired:
            return {"ok": False, "preds": None, "reason": f"timeout after {timeout_s:.0f}s"}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {"ok": False, "preds": None,
                "reason": f"law process exited without a result (exit {proc.returncode}): "
                          f"{proc.stderr.strip()[-300:]}"}


def main() -> None:
    started = time.monotonic()
    job = json.loads(sys.stdin.read())
    world, cases = job["world"], job["test_cases"]
    budget = float(job.get("timeout_s", 120.0))

    from dm.oracle.cases import build_world
    executor = build_world(world)["executor"]
    with executor.noise_disabled():
        truths = executor.run(cases)
    full = executor.run_full(cases) if hasattr(executor, "run_full") else None

    sandbox_job = {
        "world": world,
        "cases": cases,
        "law_source": job["law_source"],
        "training": job.get("training") or [],
        "outputs": [_zeroed(t) for t in truths],
        "full_outputs": [_zeroed(f) for f in full] if full else None,
        "attrs": _attrs(executor),
    }
    remaining = budget - (time.monotonic() - started) - 2.0
    child = _run_sandbox(sandbox_job, remaining)

    result = {"ok": bool(child.get("ok")), "reason": child.get("reason"),
              "fit": child.get("fit"), "mean_pos_error": None, "per_case": []}
    preds = child.get("preds")
    if preds is not None:
        all_errors, per_case = [], []
        for i, gt in enumerate(truths):
            errs = _case_errors(world, gt, preds[i] if i < len(preds) else None)
            all_errors.extend(errs)
            per_case.append(float(np.mean(errs)))
        mse = float(np.mean(all_errors)) if all_errors else float("inf")
        result["per_case"] = [c if math.isfinite(c) else None for c in per_case]
        if math.isfinite(mse):
            result["mean_pos_error"] = mse
        elif result["reason"] is None:
            result["reason"] = "non-finite error (law raised or diverged on a test case)"
    sys.stdout.write(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
