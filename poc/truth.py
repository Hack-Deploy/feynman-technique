"""True values of every posted quantity, measured on the vendor executors with noise off.

Hidden: only the checker and the calibration script may import this module (a test enforces
it). Each value is the quantity exactly as the posting defines it, read from the noise-free
simulator: a probe released at rest, its velocity after DT divided by DT. Nothing here is
written to the config or shown in a prompt.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from poc import config as C

DT = 0.01  # release time for an acceleration reading; executor steps are 0.005


@lru_cache(maxsize=None)
def _executor(world: str):
    from scienceagent.worlds import get_world
    return get_world(world, engine=C.ENGINE, noise_std=0.0, noise_seed=0)["executor"]


def _inward_2p(world: str, r: float, start_time: float = 0.0) -> float:
    """Inward acceleration of particle 2 released at rest at (r, 0), p1 = p2 = 1."""
    exp = {"p1": 1.0, "p2": 1.0, "pos2": [r, 0.0], "velocity2": [0.0, 0.0],
           "measurement_times": [DT]}
    if start_time:
        exp["start_time"] = start_time
    out = _executor(world).run([exp])[0]
    return -out["velocity2"][-1][0] / DT


def _probe_accels(world: str, points: list[list[float]]) -> np.ndarray:
    """Accelerations (N, 2) of neutral probes released at rest at ``points`` (N <= 5)."""
    pad = points + [[30.0 + 3 * i, 30.0] for i in range(5 - len(points))]
    out = _executor(world).run([{"probe_positions": pad, "probe_velocities": [[0.0, 0.0]] * 5,
                                 "measurement_times": [DT]}])[0]
    v = np.asarray(out["velocities"])[-1][-5:]
    return v[: len(points)] / DT


def _ring_inward(r: float) -> float:
    """Mean inward acceleration of the circle world's ring particles at radius r."""
    out = _executor("circle").run([{"ring_radius": r, "initial_tangential_velocity": 0.0,
                                    "measurement_times": [DT]}])[0]
    pos = np.asarray(out["positions"])[0]  # (11, 2): centre then ring
    vel = np.asarray(out["velocities"])[-1]
    ring = slice(1, None)
    rhat = pos[ring] / np.linalg.norm(pos[ring], axis=1, keepdims=True)
    return float(np.mean(-np.sum(vel[ring] * rhat, axis=1)) / DT)


def _power_and_a3(world: str) -> dict[str, float]:
    a2, a3, a6 = (_inward_2p(world, r) for r in (2.0, 3.0, 6.0))
    return {"n": -math.log(a6 / a2) / math.log(3.0), "a3": a3}


def _hubble_ether(world: str) -> dict[str, float]:
    radial = {}
    drift = np.zeros(2)
    for r in (5.0, 10.0):
        pts = [[r, 0.0], [0.0, r], [-r, 0.0], [0.0, -r]]
        acc = _probe_accels(world, pts)
        rhat = np.asarray(pts) / r
        radial[r] = float(np.mean(np.sum(acc * rhat, axis=1)))  # outward
        drift += acc.mean(axis=0) / 2
    # outward a_r(r) = -Q / r + H r  →  two radii, two unknowns
    r1, r2 = 5.0, 10.0
    H = (radial[r2] * r2 - radial[r1] * r1) / (r2 ** 2 - r1 ** 2)
    return {"H": H, "drift": float(np.linalg.norm(drift))}


def _dark_matter() -> dict[str, float]:
    pts = [[4.0, 0.0], [0.0, 4.0], [-4.0, 0.0], [0.0, -4.0]]
    acc = _probe_accels("dark_matter", pts)
    rhat = np.asarray(pts) / 4.0
    return {"g4": float(np.mean(-np.sum(acc * rhat, axis=1)) * 4.0)}


def _oscillator() -> dict[str, float]:
    from scienceagent.executor import NBodyOscillatorExecutor as Osc
    return {"ratio": _inward_2p("oscillator", 3.0, 2.0) / _inward_2p("oscillator", 3.0, 0.0),
            "period": 2 * math.pi / Osc.OMEGA}


def _measure(hyp: C.Hypothesis) -> dict[str, float]:
    w = hyp.world
    if w in ("gravity", "fractional"):
        return _power_and_a3(w)
    if w == "yukawa":
        return {"drop": _inward_2p(w, 1.0) * 1.0 / (_inward_2p(w, 6.0) * 6.0)}
    if w == "oscillator":
        return _oscillator()
    if w in ("hubble", "ether"):
        return _hubble_ether(w)
    if w == "dark_matter":
        return _dark_matter()
    if w == "circle":
        return {"n": -math.log(_ring_inward(4.0) / _ring_inward(2.0)) / math.log(2.0)}
    raise KeyError(f"no true value defined for world {w!r}")


@lru_cache(maxsize=None)
def _cached(hid: str, world: str) -> tuple[tuple[str, float], ...]:
    hyp = next(h for h in C.load().hypotheses if h.id == hid)
    return tuple(sorted(_measure(hyp).items()))


def true_values(hyp: C.Hypothesis) -> dict[str, float]:
    values = dict(_cached(hyp.id, hyp.world))
    missing = [q.name for q in hyp.quantities if q.name not in values]
    if missing:
        raise KeyError(f"{hyp.id}: no true value for {missing}")
    return {q.name: values[q.name] for q in hyp.quantities}
