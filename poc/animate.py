"""Animation data for the /experiments page: one real run, replayed from the inside.

The run shown is the hook: a real model's false claim that the naive rule paid, in a power-law
world (so its claimed force law can be drawn). For it: each launch's noisy snapshots (what the
model saw), the same launch replayed noise-free (the hidden true path), the path its claimed law
predicts, its own words and confidence each round, the expected / claimed / true force laws, and
the payout under each rule.

This is presentation code on the judge's side: it reads ``poc.truth`` after the run is over.

    uv run python -m poc.animate      # writes web/data/experiments.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from dm.store import AttemptStore
from poc import config as C
from poc import estimate, truth
from poc.baselines import SUPPORT_GUESS

OUT = C.ROOT / "web" / "data" / "experiments.json"
FRAMES = 120
POWER_LAW = ("gravity", "fractional")
LABELS = {
    "claude-opus-5-5": "Claude Opus 5.5",
    "claude-sonnet-5-5": "Claude Sonnet 5.5",
    "claude-sonnet-5": "Claude Sonnet 5",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
}


def _r(x, nd: int = 4):
    return np.round(np.asarray(x, dtype=float), nd).tolist()


def _pair(out) -> np.ndarray | None:
    if not isinstance(out, dict) or "pos1" not in out or "pos2" not in out:
        return None
    return np.stack([np.asarray(out["pos1"], float), np.asarray(out["pos2"], float)], axis=1)


def _true_path(world: str, seed: int, inp: dict, t_end: float):
    from scienceagent.worlds import get_world

    ex = get_world(world, engine=C.ENGINE, noise_std=0.0, noise_seed=seed)["executor"]
    times = np.linspace(0.0, t_end, FRAMES + 1).round(4).tolist()
    try:
        bodies = _pair(ex.run([{**inp, "measurement_times": times}])[0])
    except Exception:  # a malformed experiment: show its snapshots only
        return None
    return None if bodies is None else (times, bodies)


def _law_path(law: dict, inp: dict, times: list) -> np.ndarray | None:
    """The probe path a power law a = a3 · (3 / r)^n predicts for this launch."""
    try:
        n, a3 = float(law["n"]), float(law["a3"])
        k = a3 * math.hypot(3.0, estimate.SOFT) ** n
        role = float(inp.get("p1", 1.0)) / float(inp.get("p2", 1.0))
        accel = estimate._central(lambda r, t: k / r ** n, role)
        path = estimate._integrate(accel, np.asarray([inp["pos2"]], float),
                                   np.asarray([inp.get("velocity2", [0, 0])], float), times)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return path[:, 0, :] if np.all(np.isfinite(path)) else None


def _launch(world: str, seed: int, run: dict, claimed: dict) -> dict | None:
    inp, obs = run.get("input") or {}, _pair(run.get("output"))
    if obs is None or "pos2" not in inp:
        return None
    obs_t = run["output"].get("measurement_times") or sorted(inp.get("measurement_times") or [])
    if len(obs_t) != len(obs):
        return None
    true = _true_path(world, seed, inp, float(max(obs_t)))
    predicted = _law_path(claimed, inp, true[0]) if true else None
    return {"input": {k: v for k, v in inp.items() if k != "measurement_times"},
            "obs_t": _r(obs_t), "obs": _r(obs),
            "true_t": _r(true[0]) if true else None, "true": _r(true[1]) if true else None,
            "claimed": _r(predicted) if predicted is not None else None}


def hook_record(records: list, cfg: C.Config):
    """The real-model false claim the naive rule paid most for, in a power-law world."""
    cands = [r for r in records if not r.solver.startswith("baseline:") and r.solver != "fake"
             and r.verdict.get("outcome") == "false_claim"
             and cfg.hypothesis(r.extra["hypothesis_id"]).world in POWER_LAW
             and ((r.extra.get("settlements") or {}).get("naive") or {}).get("profit", 0) > 0]
    return max(cands, key=lambda r: r.extra["settlements"]["naive"]["profit"], default=None)


def scene(record, cfg: C.Config) -> dict:
    x = record.extra
    hyp = cfg.hypothesis(x["hypothesis_id"])
    ruling = record.verdict.get("ruling") or {}
    est = x.get("estimates") or {}
    claimed = {k: (est.get(k) or {}).get("value") for k in ("n", "a3")}
    true_values = truth.true_values(hyp)
    st = x.get("settlements") or {}
    return {
        "id": record.attempt_id,
        "agent": LABELS.get(record.solver, record.solver),
        "hypothesis_id": hyp.id, "hypothesis": hyp.hypothesis, "seed": record.seed,
        "answer": hyp.answer, "verdict": x.get("agent_verdict"), "prize": hyp.prize,
        "outcome": record.verdict.get("outcome"),
        "reasons": ruling.get("reasons") or [],
        "flags": ruling.get("flags") or [],
        "laws": {
            "expected": {k: SUPPORT_GUESS[hyp.id][k] for k in ("n", "a3")},
            "claimed": claimed,
            "true": {k: round(true_values[k], 5) for k in ("n", "a3")},
        },
        "tolerance": {q.name: q.tolerance for q in hyp.quantities},
        "rounds": [{"round": e.get("round"), "action": e.get("action"),
                    "assessment": e.get("assessment"), "p_success": e.get("p_success"),
                    "experiments": e.get("experiments")} for e in x.get("round_log") or []],
        "evidence": x.get("evidence"),
        "spent": record.lab_cost,
        "naive": (st.get("naive") or {}).get("profit"),
        "market": (st.get("market") or {}).get("profit"),
        "launches": [l for l in (_launch(hyp.world, record.seed, run, claimed)
                                 for run in x.get("runs") or []) if l],
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=str(C.ATTEMPTS_PATH))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    cfg = C.load()
    record = hook_record(AttemptStore(args.store).load(lambda r: r.protocol == C.PROTOCOL), cfg)
    if record is None:
        raise SystemExit("no real-model false claim in a power-law world yet")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scene(record, cfg), separators=(",", ":"), allow_nan=False))
    print(f"{record.solver} on {record.extra['hypothesis_id']}, "
          f"{out.stat().st_size / 1e3:.0f} kB -> {out}")


if __name__ == "__main__":
    main()
