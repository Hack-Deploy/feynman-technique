"""Import local vendor DiscoverPhysics run JSONs as ``AttemptRecord``s.

A run JSON has ``world``, ``model``, ``evaluation.mean_pos_error``, an explanation
score (``explanation.score``, or ``evaluation.explanation.score`` as the vendor's
aggregator reads it), ``law`` and ``rounds`` (a count or the list of rounds).

    uv run python -m dm.importers.vendor_runs <dir-or-json>... [--out attempts/vendor_runs.jsonl]
"""

from __future__ import annotations

import argparse
import json
import math
import uuid
from pathlib import Path
from typing import Any

from dm.importers.ara import (PASS_THRESHOLD, WORLD_VARS, count_experiments,
                               finite_or_none, json_safe, write_store)
from dm.store import ATTEMPTS_DIR
from dm.types import AttemptRecord

PROTOCOL = "discoverphysics_native"
VENUE = "discoverphysics"
STORE_PATH = ATTEMPTS_DIR / "vendor_runs.jsonl"


def attempt_id(model: str, world: str, seed: int, path: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL,
                          f"vendor:{model}:{world}:{seed}:{PROTOCOL}:{path}"))


def _explanation_score(d: dict) -> float | None:
    for e in (d.get("explanation"), (d.get("evaluation") or {}).get("explanation")):
        if isinstance(e, dict) and isinstance(e.get("score"), (int, float)):
            return finite_or_none(float(e["score"]))
    return None


def build_record(d: dict, path: str) -> AttemptRecord | None:
    """None if the JSON is not a scored vendor run."""
    world, model = d.get("world"), d.get("model")
    ev = d.get("evaluation") or {}
    if not world or not model or not isinstance(ev, dict):
        return None
    mpe = ev.get("mean_pos_error")
    var = WORLD_VARS.get(world)
    nmse = (float(mpe) / var if isinstance(mpe, (int, float)) and var
            and math.isfinite(mpe) else None)
    rounds_field = d.get("rounds")
    if isinstance(rounds_field, list):
        rounds = len(rounds_field)
        experiments = count_experiments({"rounds": rounds_field})
        exp_source = "rounds[].experiment_input"
    else:
        rounds = int(rounds_field or 0)
        experiments = 0
        exp_source = "unavailable: rounds is a count, recorded as 0"
    seed = int(d.get("seed", d.get("noise_seed", 0)) or 0)
    extra: dict[str, Any] = {
        "source_file": path,
        "mean_pos_error": mpe,
        "world_var": var,
        "noise_std": d.get("noise_std"),
        "experiments_source": exp_source,
        "numeric_rule": "norm_MSE < 0.1",
    }
    if nmse is None:
        extra["nmse_raw"] = repr(mpe)
    return AttemptRecord(
        attempt_id=attempt_id(model, world, seed, path),
        source="live", protocol=PROTOCOL, venue=VENUE, world=world,
        solver=str(model), seed=seed, stated_p_success=None,
        rounds=rounds, experiments=experiments, lab_cost=float(rounds),
        llm_usage={}, submitted_law=d.get("law") or d.get("final_law"),
        verdict={"normalised_mse": nmse,
                 "passed": bool(nmse is not None and nmse < PASS_THRESHOLD),
                 "prereg_commitment": None, "public_tests": True,
                 "explanation_score": _explanation_score(d)},
        created_at=str(d.get("timestamp", "")),
        extra=json_safe(extra),
    )


def import_paths(paths: list[Path], root: Path | None = None) -> list[AttemptRecord]:
    files: list[Path] = []
    for p in paths:
        files.extend(sorted(p.rglob("*.json")) if p.is_dir() else [p])
    out: list[AttemptRecord] = []
    for f in files:
        try:
            d = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(d, dict):
            continue
        rel = str(f.relative_to(root)) if root and f.is_relative_to(root) else f.name
        rec = build_record(d, rel)
        if rec is not None:
            out.append(rec)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=STORE_PATH)
    args = ap.parse_args(argv)
    records = import_paths(args.paths, root=Path.cwd())
    n = write_store(records, args.out)
    print(f"{len(records)} vendor run records, {n} appended to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
