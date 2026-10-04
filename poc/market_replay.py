"""Judge-side presentation of recorded experiments for every market claim.

Only called after resolution. Motion plots use recorded positions, not invented
force estimates. Noise-free reference paths rerun the exact purchased experiment.
"""
from __future__ import annotations

import numpy as np

from poc import config as C, truth


def displacement(inp: dict, out: dict) -> list[float] | None:
    """Mean inward radial displacement of probes (or the ring), in simulator units."""
    try:
        if "pos2" in out and "pos2" in inp:
            origin = np.asarray(inp.get("pos1", [0, 0]), float)
            start = np.asarray([inp["pos2"]], float) - origin
            positions = np.asarray(out["pos2"], float)[:, None, :] - origin
        elif "probe_positions" in inp:
            start = np.asarray(inp["probe_positions"], float)
            positions = np.asarray(out["positions"], float)[:, -len(start):, :]
        elif "ring_radius" in inp:
            positions = np.asarray(out["positions"], float)[:, 1:, :]
            radii = np.linalg.norm(positions, axis=2)
            values = float(inp["ring_radius"]) - radii.mean(axis=1)
            return values.tolist() if np.all(np.isfinite(values)) else None
        else:
            return None
        values = (np.linalg.norm(start, axis=1) - np.linalg.norm(positions, axis=2)).mean(axis=1)
        return values.tolist() if np.all(np.isfinite(values)) else None
    except (KeyError, ValueError, TypeError, IndexError):
        return None


def build(record, hyp, cfg) -> dict:
    from scienceagent.worlds import get_world

    x = record.extra
    executor = get_world(hyp.world, engine=C.ENGINE, noise_std=0.0,
                         noise_seed=record.seed)["executor"]
    traces = []
    for run in x.get("runs") or []:
        inp, out = run.get("input") or {}, run.get("output") or {}
        times = out.get("measurement_times") or inp.get("measurement_times") or []
        observed = displacement(inp, out)
        if observed is None or len(times) != len(observed) or not times:
            continue
        reference = None
        reference_times = np.linspace(0, max(times), 41).tolist()
        try:
            reference = displacement(inp, executor.run([
                {**inp, "measurement_times": reference_times}
            ])[0])
        except Exception:
            # A failed reference rendering never changes a settled run's outcome.
            pass
        traces.append({"round": run["round"], "times": times, "observed": observed,
                       "reference_times": reference_times if reference is not None else [],
                       "reference": reference or []})
    paid = x.get("prize_paid", 0.0)
    return {"kind": "quantity", "id": record.attempt_id, "world": hyp.world,
            "agent": record.solver, "hypothesis_id": hyp.id, "hypothesis": hyp.hypothesis,
            "resolution_criteria": hyp.resolution_criteria, "seed": record.seed,
            "answer": hyp.answer, "verdict": x.get("agent_verdict"),
            "outcome": record.verdict.get("outcome"), "prize": hyp.prize,
            "quantities": [{"name": q.name, "meaning": q.meaning, "tolerance": q.tolerance}
                           for q in hyp.quantities],
            "primary": hyp.supported_if.quantity, "true": truth.true_values(hyp),
            "rule": {"kind": hyp.supported_if.kind, "bounds": list(hyp.supported_if.bounds)},
            "rounds": [{**rd, "estimates": rd.get("estimates") or {}}
                       for rd in x.get("round_log") or []],
            "traces": traces, "max_rounds": cfg.max_rounds, "spent": record.lab_cost,
            "prize_paid": paid, "profit": paid - record.lab_cost,
            "usd": (record.llm_usage or {}).get("usd", 0.0)}
