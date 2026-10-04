"""Animation data for the /experiments page: what each run looked like from the inside.

For every chosen run: each launch's noisy observations (what the agent saw), the hidden true
path (the same launch replayed noise-free at dense times), the agent's estimates against the
truth and tolerance, the checker's ruling, and the payout under each rule.

This is presentation code on the judge's side: it reads ``poc.truth`` after the runs are over.
Agents never see its output.

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

OUT = C.ROOT / "web" / "data" / "experiments.json"
FRAMES = 120
MAX_LAUNCHES = 8
SHOWCASE = (  # (solver, hypothesis, seed) for the scripted contrast scenes
    ("baseline:reference", "gravity-inverse-square", 0),
    ("baseline:p_hacker", "gravity-inverse-square", 0),
    ("baseline:always_supported", "gravity-inverse-square", 0),
    ("baseline:reference", "hubble-outward-push", 0),
    ("baseline:p_hacker", "ether-outward-push", 0),
    ("baseline:reference", "circle-ordinary-gravity", 0),
    ("baseline:reference", "dark-matter-unseen-pull", 0),
    ("baseline:reference", "oscillator-time-varying", 0),
    ("baseline:reference", "yukawa-screened", 0),
)
LABELS = {
    "baseline:reference": "Reference agent",
    "baseline:p_hacker": "P-hacker",
    "baseline:always_supported": "Always says yes",
    "baseline:coin_flip": "Coin flip",
    "baseline:abstain": "Abstains",
    "claude-opus-5-5": "Claude Opus 5.5",
    "claude-sonnet-5-5": "Claude Sonnet 5.5",
    "claude-sonnet-5": "Claude Sonnet 5",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
}


def _r(x, nd: int = 3):
    """Round nested arrays for a compact JSON file."""
    return np.round(np.asarray(x, dtype=float), nd).tolist()


def _kind(world: str) -> str:
    if world in estimate.TWO_P:
        return "pair"
    return "ring" if world == "circle" else "field"


def _bodies(kind: str, out: dict) -> np.ndarray | None:
    """(T, N, 2) positions from a lab output; the moving probe(s) are last."""
    if kind == "pair":
        if "pos1" not in out or "pos2" not in out:
            return None
        return np.stack([np.asarray(out["pos1"], float), np.asarray(out["pos2"], float)], axis=1)
    if "positions" not in out:
        return None
    pos = np.asarray(out["positions"], float)
    return pos if pos.ndim == 3 else None


def _true_path(world: str, seed: int, inp: dict, t_end: float) -> tuple[list, np.ndarray] | None:
    from scienceagent.worlds import get_world

    ex = get_world(world, engine=C.ENGINE, noise_std=0.0, noise_seed=seed)["executor"]
    times = np.linspace(0.0, t_end, FRAMES + 1).round(4).tolist()
    try:
        out = ex.run([{**inp, "measurement_times": times}])[0]
    except Exception:  # an agent's malformed experiment: show its observations only
        return None
    bodies = _bodies(_kind(world), out) if isinstance(out, dict) else None
    return None if bodies is None else (times, bodies)


def _claimed_path(world: str, est: dict, inp: dict, times: list) -> np.ndarray | None:
    """The probe path the agent's claimed power law predicts, from the same launch (the fit
    model of poc.estimate). Only for the power-law worlds, where n and a3 fix the law."""
    if world not in ("gravity", "fractional") or "pos2" not in inp:
        return None
    try:
        n, a3 = float(est["n"]["value"]), float(est["a3"]["value"])
        k = a3 * math.hypot(3.0, estimate.SOFT) ** n
        role = float(inp.get("p1", 1.0)) / float(inp.get("p2", 1.0))
        accel = estimate._central(lambda r, t: k / r ** n, role)
        path = estimate._integrate(accel, np.asarray([inp["pos2"]], float),
                                   np.asarray([inp.get("velocity2", [0, 0])], float), times)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return path if np.all(np.isfinite(path)) else None


def _launch(world: str, seed: int, run: dict, single_estimate: dict | None,
            claimed: dict | None = None) -> dict | None:
    kind = _kind(world)
    inp, out = run.get("input") or {}, run.get("output")
    if not isinstance(out, dict):
        return None
    obs = _bodies(kind, out)
    if obs is None:
        return None
    obs_t = out.get("measurement_times") or sorted(inp.get("measurement_times") or [])
    if len(obs_t) != len(obs):
        return None
    t_end = float(max(obs_t))
    true = _true_path(world, seed, inp, t_end)
    n_probe = {"pair": 1, "ring": obs.shape[1] - 1, "field": 5}[kind]
    predicted = _claimed_path(world, claimed or {}, inp, true[0]) if true else None
    return {
        "input": {k: v for k, v in inp.items() if k != "measurement_times"},
        "obs_t": _r(obs_t), "obs": _r(obs[:, -n_probe:, :]),
        "true_t": _r(true[0]) if true else None,
        "true": _r(true[1]) if true else None,
        "n_probe": n_probe,
        "claimed": _r(predicted) if predicted is not None else None,
        "single_estimate": single_estimate,
    }


def _single(world: str, run: dict) -> dict | None:
    try:
        return {k: round(v, 5) for k, v in estimate.estimate(world, [run]).values.items()}
    except (ValueError, KeyError):
        return None


def scene(record, cfg: C.Config) -> dict:
    x = record.extra
    hyp = cfg.hypothesis(x["hypothesis_id"])
    ruling = record.verdict.get("ruling") or {}
    flags = [f["flag"] for f in ruling.get("flags", [])]
    runs = (x.get("runs") or [])[:MAX_LAUNCHES]
    singles = "rerun" in flags
    launches = [l for l in (_launch(hyp.world, record.seed, run,
                                    _single(hyp.world, run) if singles else None,
                                    x.get("estimates"))
                            for run in runs) if l]
    true_values = truth.true_values(hyp)
    est = x.get("estimates") or {}
    rule = hyp.supported_if
    re_values = (ruling.get("reanalysis") or {}).get("values") or {}
    st = x.get("settlements") or {}
    return {
        "id": record.attempt_id,
        "agent": LABELS.get(record.solver, record.solver),
        "solver": record.solver,
        "real": not record.solver.startswith("baseline:") and record.solver != "fake",
        "seed": record.seed,
        "hypothesis_id": hyp.id, "hypothesis": hyp.hypothesis, "world": hyp.world,
        "kind": _kind(hyp.world), "answer": hyp.answer, "prize": hyp.prize,
        "rule": {"quantity": rule.quantity, "kind": rule.kind, "bounds": list(rule.bounds)},
        "quantities": [{
            "name": q.name, "meaning": q.meaning, "tolerance": q.tolerance,
            "truth": round(true_values[q.name], 5),
            "estimate": (est.get(q.name) or {}).get("value"),
            "sigma": (est.get(q.name) or {}).get("sigma"),
            "reanalysis": round(re_values[q.name], 5) if q.name in re_values
            and math.isfinite(re_values[q.name]) else None,
        } for q in hyp.quantities],
        "verdict": x.get("agent_verdict"),
        "outcome": record.verdict.get("outcome"),
        "reasons": ruling.get("reasons") or [],
        "flags": ruling.get("flags") or [],
        "bid_p": x.get("bid_p"), "final_p": x.get("final_p"),
        "planned_cost": x.get("planned_cost"), "spent": record.lab_cost,
        "rounds": record.rounds,
        "naive": (st.get("naive") or {}).get("profit"),
        "market": (st.get("market") or {}).get("profit"),
        "bond_lost": (st.get("market") or {}).get("bond_lost"),
        "launches": launches,
        "n_runs": len(x.get("runs") or []),
    }


def pick(records: list) -> list:
    """Real-model runs that ran an experiment, then the scripted showcase."""
    real = [r for r in records if not r.solver.startswith("baseline:") and r.solver != "fake"
            and r.extra.get("runs")]
    key = {(r.solver, r.extra.get("hypothesis_id"), r.seed): r for r in records}
    show = [key[k] for k in SHOWCASE if k in key]
    return real + show


def build(records: list, cfg: C.Config) -> dict:
    scenes = [scene(r, cfg) for r in pick(records)]
    return {"scenes": scenes,
            "note": "True paths are replayed noise-free after the run; agents never saw them."}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=str(C.ATTEMPTS_PATH))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    cfg = C.load()
    records = AttemptStore(args.store).load(lambda r: r.protocol == C.PROTOCOL)
    data = build(records, cfg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False))
    print(f"{len(data['scenes'])} scenes, {out.stat().st_size / 1e3:.0f} kB -> {out}")


if __name__ == "__main__":
    main()
