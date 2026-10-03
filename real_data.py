"""Discovery Market – data for the simulation and live attempt APIs.

Two sources, both real attempts rather than coin flips:
- ARA replay: published DiscoverPhysics runs (attempts/ara.jsonl) replayed through
  the market (output/replay_ara/summary.json, from `python -m dm.replay ara`).
- ForceBench: offline solvers that pay per launch, scored by the oracle on hidden
  cases (output/forcebench_settle.json, from `python -m tests.forcebench_settle`),
  plus single attempts run on demand by `run_forcebench_attempt`.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
ARA_STORE = ROOT / "attempts" / "ara.jsonl"
ARA_SUMMARY = OUTPUT_DIR / "replay_ara" / "summary.json"
FORCEBENCH_GRID = OUTPUT_DIR / "forcebench_settle.json"
FORCEBENCH_SNAPSHOT = ROOT / "web" / "data" / "forcebench_settle.json"
SIM_SUMMARY = OUTPUT_DIR / "summary.json"
# Committed snapshots, used when the generated files above are missing (fresh clone,
# offline demo). See attempts/fixtures/demo/README.md.
SNAPSHOT_DIR = ROOT / "attempts" / "fixtures" / "demo"
SNAPSHOTS = {
    ARA_STORE: SNAPSHOT_DIR / "ara.jsonl",
    ARA_SUMMARY: SNAPSHOT_DIR / "replay_ara_summary.json",
    FORCEBENCH_GRID: SNAPSHOT_DIR / "forcebench_settle.json",
}
LIVE_WALLET = 10.0
LIVE_PRICE = 1.0
TEST_SEED = 0
# The simulated sweeps call this world "coulomb".
SIM_WORLD_ALIASES = {"coulomb": "coulomb_easy"}
SOLVERS = ("bayes_lite", "random_menu")


def _source(path: Path) -> tuple[Path, bool]:
    """The generated file if it exists, else its committed snapshot. (path, is_snapshot)"""
    if path.exists() or path not in SNAPSHOTS:
        return path, False
    snap = SNAPSHOTS[path]
    return (snap, True) if snap.exists() else (path, False)


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _finite(v) -> float | None:
    return v if isinstance(v, (int, float)) and math.isfinite(v) else None


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def ara_attempts() -> list[dict]:
    store, _ = _source(ARA_STORE)
    if not store.exists():
        return []
    from dm.store import AttemptStore

    rows = []
    for r in AttemptStore(store).load():
        extra = r.extra or {}
        rows.append({
            "model": r.solver.removeprefix("ara:"),
            "world": r.world,
            "passed": r.passed,
            "nmse": _finite(r.verdict.get("normalised_mse")),
            "explanation": _finite(r.verdict.get("explanation_score")),
            "ara_verdict": extra.get("ara_verdict"),
            "ara_passed": extra.get("ara_passed"),
            "rounds": r.rounds,
            "experiments": r.experiments,
            "usd": _finite((r.llm_usage or {}).get("usd")),
            "dataset_url": extra.get("dataset_url"),
        })
    rows.sort(key=lambda x: (x["world"], x["model"]))
    return rows


def ara_data() -> dict:
    summary_path, snapshot = _source(ARA_SUMMARY)
    summary = _read_json(summary_path)
    sim = _read_json(SIM_SUMMARY) or {}
    sim_clearing = {
        source: {SIM_WORLD_ALIASES.get(w, w): v for w, v in worlds.items()}
        for source, worlds in (sim.get("h2_clearing_prizes") or {}).items()
    }
    attempts = ara_attempts()
    if summary is None:
        return {"available": False, "attempts": attempts, "sim_clearing": sim_clearing}
    return {
        "available": True,
        "snapshot": snapshot,
        "attribution": "ARA Labs (AgentNativeResearchLab), CC BY 4.0",
        "attribution_url": "https://huggingface.co/AgentNativeResearchLab",
        "caveat": summary["caveat"],
        "config": summary["config"],
        "checks": summary["checks"],
        "verdict_rule": summary["verdict_rule"],
        "worlds": summary["worlds"],
        "solvers": summary["solvers"],
        "prizes": summary["config"]["prizes"],
        "clearing": summary["clearing_prizes"],
        "solved_seed_counts": summary["solved_seed_counts"],
        "profit_and_bids": summary["profit_and_bids"],
        "lab_revenue": summary["lab_revenue"],
        "sim_clearing": sim_clearing,
        "attempts": attempts,
    }


def forcebench_data() -> dict:
    from dm.venues.forcebench import (
        BUDGET,
        MENU,
        MEASUREMENT_TIMES,
        NOISE_STD,
        SEED_ACTION,
        WORLDS,
    )
    from dm.solvers._inference import MODEL_FAMILIES, ROLES

    grid = _read_json(FORCEBENCH_GRID)
    snapshot = False
    source = "output"
    if grid is None:
        grid = _read_json(FORCEBENCH_SNAPSHOT)
        if grid is not None:
            snapshot = True
            source = "snapshot"
        else:
            grid_path, snapshot = _source(FORCEBENCH_GRID)
            grid = _read_json(grid_path)
            source = "snapshot" if snapshot and grid is not None else None
    out = {"worlds": list(WORLDS), "solvers": list(SOLVERS), "seeds": [0, 1, 2, 3, 4],
           "wallet": LIVE_WALLET, "price": LIVE_PRICE, "available": grid is not None,
           "snapshot": bool(snapshot) if grid is not None else False,
           "source": source if grid is not None else None, "table": [], "attempts": [],
           "menu": [
               {"action": launch.action, "r0": launch.r0, "vx": launch.v[0],
                "vy": launch.v[1], "p1": launch.p1, "p2": launch.p2,
                "seed": launch.action == SEED_ACTION}
               for launch in MENU
           ],
           "venue": {"budget": BUDGET, "noise_std": NOISE_STD,
                     "measurement_times": list(MEASUREMENT_TIMES),
                     "seed_action": SEED_ACTION, "threshold": 0.1},
           "library": {"families": list(MODEL_FAMILIES), "roles": list(ROLES)}}
    if grid is None:
        return out
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    results = grid.get("results") or []
    for r in results:
        cells[(r["solver"], r["world"])].append(r)
    table = []
    for (solver, world), rows in sorted(cells.items()):
        n = len(rows)
        table.append({
            "solver": solver, "world": world, "n": n,
            "passed": sum(r["passed"] for r in rows),
            "identified": sum(r["identified"] for r in rows),
            "baseline_passed": sum(r["baseline_passed"] for r in rows),
            "mean_experiments": sum(r["experiments"] for r in rows) / n,
            "mean_stated_p": sum(r["stated_p"] or 0 for r in rows) / n,
            "nmse": [r["nmse"] for r in sorted(rows, key=lambda r: r["seed"])],
        })
    world_order = {world: i for i, world in enumerate(WORLDS)}
    attempt_fields = (
        "solver", "world", "seed", "passed", "identified", "nmse", "baseline_passed",
        "baseline_nmse", "experiments", "stated_p", "top_model", "top_params",
        "stopped_reason",
    )
    attempts = [{key: row.get(key) for key in attempt_fields} for row in results]
    attempts.sort(key=lambda row: (
        world_order.get(row["world"], len(WORLDS)),
        str(row["solver"] or ""),
        row["seed"] if isinstance(row["seed"], int) else -1,
    ))
    out.update(head=grid.get("head"), test_seed=grid.get("test_seed"), table=table)
    out["attempts"] = attempts
    return out


def build_real_data() -> dict:
    return {"ara": ara_data(), "forcebench": forcebench_data()}


def _model_label(m: dict | None) -> str | None:
    if not m:
        return None
    return f"{m.get('family')} / {m.get('role')}"


def run_forcebench_attempt(world: str, solver: str, seed: int) -> dict:
    """Post a prize, let the solver buy launches, settle its law on the hidden cases."""
    from dm.settle import prereg_for, settle
    from dm.venues.forcebench import MENU, WORLDS, run_attempt
    from dm.wallet import Wallet

    if world not in WORLDS:
        raise ValueError(f"unknown world {world!r}")
    if solver not in SOLVERS:
        raise ValueError(f"unknown solver {solver!r}")
    if seed not in range(5):
        raise ValueError("seed must be 0-4")

    prereg = prereg_for("forcebench", world, TEST_SEED)
    wallet = Wallet(solver, balance=LIVE_WALLET)
    attempt = run_attempt(solver, world, seed, wallet, LIVE_PRICE)
    conserved = math.isclose(wallet.balance + wallet.lab_revenue, LIVE_WALLET)
    record = settle(prereg, attempt)
    verdict = record.verdict

    transcript = json.loads((ROOT / attempt.transcript_path).read_text())

    def launch(choice: int | None, inp: dict, output: dict) -> dict:
        times = output.get("measurement_times") or inp.get("measurement_times") or []
        radii = [math.hypot(p[0] - q[0], p[1] - q[1])
                 for p, q in zip(output.get("pos2") or [], output.get("pos1") or [])]
        return {"action": choice, "r0": inp["pos2"][0], "velocity": inp["velocity2"],
                "p1": inp["p1"], "p2": inp["p2"], "times": times, "radii": radii}

    seed_data = transcript["seed_data"]
    rounds = []
    for r in transcript["rounds"]:
        models = sorted((r.get("solver") or {}).get("models") or [],
                        key=lambda m: -m.get("weight", 0))
        rounds.append({
            "round": r["round"],
            "launch": launch(r["choice"], r["input"], r["output"]),
            "top_models": [{"label": _model_label(m), "weight": m.get("weight")}
                           for m in models[:3]],
        })
    charges = [e for e in wallet.events if e["type"] == "experiment_charged"]
    sub = transcript["submission"]
    from tests.forcebench_local import identify_model

    identified, _ = identify_model(world, sub.get("top_model") or {})
    return {
        "ok": True,
        "world": world, "solver": solver, "seed": seed,
        "prereg": {"question_id": prereg.question_id, "commitment": prereg.commitment(),
                   "test_seed": TEST_SEED},
        "wallet": {"start": LIVE_WALLET, "end": wallet.balance,
                   "lab_revenue": wallet.lab_revenue, "conserved": conserved},
        "charges": [{"round": e["round"], "amount": e["amount"]} for e in charges],
        "insufficient": [e for e in wallet.events if e["type"] == "insufficient_credits"],
        "seed_launch": launch(seed_data.get("action"), seed_data["input"], seed_data["output"]),
        "rounds": rounds,
        "stopped_reason": transcript.get("stopped_reason"),
        "experiments": attempt.experiments,
        "stated_p": attempt.stated_p_success,
        "top_model": _model_label(sub.get("top_model")),
        "top_params": (sub.get("top_model") or {}).get("params"),
        "explanation": sub.get("explanation"),
        "law": attempt.submitted_law,
        "verdict": {
            "passed": verdict.get("passed"),
            "nmse": _finite(verdict.get("normalised_mse")),
            "threshold": 0.1,
            "public_tests": verdict.get("public_tests"),
            "reason": verdict.get("reason"),
            "commitment_match": verdict.get("prereg_commitment") == prereg.commitment(),
        },
        "identified": identified,
        "attempt_id": record.attempt_id,
        "transcript_path": _relative(ROOT / attempt.transcript_path),
        "menu_size": len(MENU),
    }
