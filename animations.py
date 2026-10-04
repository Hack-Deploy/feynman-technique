"""Discovery Market – experiment replays for the /experiments page.

Each recorded live run (attempts/transcripts/poc_dp/<claim>/<model>_seed<n>.json)
stores the experiments the AI scientist bought: their initial conditions and a
handful of noisy measurements. To animate them, every experiment is re-run
through the same world simulator with no noise and dense time steps. The page
draws that smooth path and overlays the noisy points the AI actually paid for.

The re-simulation reuses the transcript's own inputs, so the motion shown is
the world's true physics for exactly the experiment the AI chose. Replays are
cached in output/animations/<claim>.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from poc import config as C

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "output" / "animations"
FRAMES = 72
CACHE_VERSION = 1


def _round(x, nd: int = 3):
    if isinstance(x, list):
        return [_round(v, nd) for v in x]
    if isinstance(x, float):
        return round(x, nd)
    return x


def _particles(out: dict) -> list:
    """Normalise both executor output shapes to frames × particles × [x, y]."""
    if "positions" in out:
        return out["positions"]
    return [[p1, p2] for p1, p2 in zip(out["pos1"], out["pos2"])]


def _probe_count(inp: dict, n_particles: int) -> int:
    """How many particles are the AI's own probes (always the last ones)."""
    if "probe_positions" in inp:
        return len(inp["probe_positions"])
    if "pos2" in inp:
        return 1  # particle 2 is the probe, particle 1 the source
    return n_particles  # circle: every ring particle is placed by the AI


def _replay(executor, inp: dict, out: dict | None) -> dict | None:
    times = [float(t) for t in inp.get("measurement_times") or []]
    if not times:
        return None
    t_end = max(times + [float(inp.get("duration") or 0)])
    dense = dict(inp, measurement_times=[t_end * i / FRAMES for i in range(1, FRAMES + 1)])
    try:
        sim = executor.run([dense])[0]
    except Exception as exc:  # a malformed experiment should not break the page
        return {"error": str(exc)[:200]}
    frames = _particles(sim)
    n = len(frames[0]) if frames else 0
    measured = []
    if out and not out.get("error"):
        try:
            measured = [{"t": t, "points": pts}
                        for t, pts in zip(out["measurement_times"], _particles(out))]
        except (KeyError, TypeError):
            measured = []
    return {
        "times": _round(dense["measurement_times"]),
        "frames": _round(frames, 3),
        "probes": _probe_count(inp, n),
        "measured": _round(measured, 3),
        "input": _round({k: v for k, v in inp.items() if k != "measurement_times"}, 3),
        "measurement_times": _round(times),
    }


def claims() -> list[dict]:
    cfg = C.load()
    out = []
    for h in cfg.hypotheses:
        d = C.TRANSCRIPTS_DIR / h.id
        models = sorted(p.stem for p in d.glob("*.json")) if d.exists() else []
        out.append({"id": h.id, "world": h.world, "hypothesis": h.hypothesis,
                    "answer": h.answer, "runs": models})
    return out


def build(claim_id: str, use_cache: bool = True) -> dict:
    cfg = C.load()
    hyp = cfg.hypothesis(claim_id)
    cache = CACHE / f"{claim_id}.json"
    sources = sorted((C.TRANSCRIPTS_DIR / claim_id).glob("*.json"))
    stamp = [CACHE_VERSION] + [[p.name, p.stat().st_mtime_ns] for p in sources]
    if use_cache and cache.exists():
        cached = json.loads(cache.read_text())
        if cached.get("stamp") == stamp:
            return cached

    from scienceagent.worlds import get_world

    executor = get_world(hyp.world, engine=C.ENGINE, noise_std=0.0, noise_seed=0)["executor"]
    runs = {}
    for path in sources:
        t = json.loads(path.read_text())
        rounds = []
        for r in t.get("rounds", []):
            exps = []
            outs = r.get("experiment_output") or []
            for i, inp in enumerate(r.get("experiment_input") or []):
                rep = _replay(executor, inp, outs[i] if i < len(outs) else None)
                if rep:
                    exps.append(rep)
            rounds.append({
                "round": r.get("round"), "action": r.get("action"),
                "p_success": r.get("p_success"), "assessment": r.get("assessment"),
                "verdict": r.get("verdict"), "evidence": r.get("evidence"),
                "cost": r.get("experiments_cost"), "experiments": exps,
            })
        verdict = t.get("verdict")
        runs[path.stem] = {
            "model": t.get("model"), "outcome": t.get("outcome"), "verdict": verdict,
            "correct": verdict == hyp.answer if verdict in C.ANSWERS else None,
            "rounds": rounds,
        }
    data = {
        "stamp": stamp, "id": hyp.id, "world": hyp.world, "hypothesis": hyp.hypothesis,
        "criteria": hyp.resolution_criteria, "answer": hyp.answer, "prize": hyp.prize,
        "runs": runs,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data, separators=(",", ":")))
    return data


if __name__ == "__main__":
    for c in claims():
        d = build(c["id"], use_cache=False)
        n = sum(len(r["experiments"]) for run in d["runs"].values() for r in run["rounds"])
        size = (CACHE / f"{c['id']}.json").stat().st_size // 1024
        print(f"{c['id']:32s} {len(d['runs'])} runs, {n:3d} experiments, {size} KB")
