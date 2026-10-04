"""Estimate each posted quantity from noisy positions: least-squares fits of simple force models.

Pure and truth-free. Used by the scripted reference agent, the scripted p-hacker, and the
checker's re-analysis of the data an agent paid for. ``runs`` is the list the attempt record
keeps: ``[{"input": experiment, "output": lab result}, ...]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import least_squares
from scipy.special import k1

SOFT = 0.05   # vendor softening: force magnitude uses sqrt(r² + SOFT²)
DT = 0.01
TWO_P = ("gravity", "fractional", "yukawa", "oscillator")
FIELD = ("hubble", "ether", "dark_matter")


@dataclass(frozen=True)
class Estimate:
    values: dict[str, float]
    sigmas: dict[str, float | None]  # None when the fit's covariance is unusable
    n_points: int


# ------------------------------------------------------------------ integrators

def _steps(times) -> list[int]:
    return [max(int(round(float(t) / DT)), 0) for t in times]


def _integrate(accel: Callable, x0: np.ndarray, v0: np.ndarray, times, t0: float = 0.0):
    """KDK leapfrog for independent test particles. x0, v0: (N, 2). Returns (T, N, 2)."""
    idx = _steps(times)
    x, v = x0.astype(float).copy(), v0.astype(float).copy()
    out = np.empty((len(idx), *x.shape))
    want: dict[int, list[int]] = {}
    for slot, s in enumerate(idx):
        want.setdefault(s, []).append(slot)
    for slot in want.get(0, ()):
        out[slot] = x
    with np.errstate(all="ignore"):
        a = accel(x, t0)
        for step in range(max(idx, default=0)):
            v += 0.5 * DT * a
            x += DT * v
            a = accel(x, t0 + (step + 1) * DT)
            v += 0.5 * DT * a
            for slot in want.get(step + 1, ()):
                out[slot] = x
    return np.nan_to_num(out, nan=1e3, posinf=1e3, neginf=-1e3)


def _central(force: Callable, scale: float):
    """Acceleration toward the origin with magnitude scale * force(r_eff, t)."""
    def accel(x, t):
        r = np.linalg.norm(x, axis=-1, keepdims=True)
        f = force(np.sqrt(r ** 2 + SOFT ** 2), t)
        return -scale * f * x / np.maximum(r, 1e-12)
    return accel


# ------------------------------------------------------------------ fitting

def _fit(residuals: Callable, starts: list[np.ndarray], derived: Callable,
         n_points: int) -> Estimate:
    best = None
    for p0 in starts:
        try:
            res = least_squares(residuals, p0, method="lm", max_nfev=400 * (len(p0) + 1))
        except Exception:
            continue
        if np.all(np.isfinite(res.fun)) and (best is None or res.cost < best.cost):
            best = res
    if best is None:
        raise ValueError("no fit converged")
    p = best.x
    try:
        values = derived(p)
    except (ArithmeticError, ValueError) as exc:
        raise ValueError(f"fit gave no usable estimate: {exc}") from exc
    if not all(np.isfinite(v) for v in values.values()):
        raise ValueError("fit gave a non-finite estimate")
    sigmas: dict[str, float | None] = {k: None for k in values}
    dof = max(len(best.fun) - len(p), 1)
    try:
        cov = np.linalg.inv(best.jac.T @ best.jac) * (2 * best.cost / dof)
        for k in values:
            grad = np.zeros_like(p)
            for i in range(len(p)):
                h = 1e-6 * max(abs(p[i]), 1.0)
                q = p.copy()
                q[i] += h
                try:
                    grad[i] = (derived(q)[k] - values[k]) / h
                except (ArithmeticError, ValueError):
                    grad[i] = np.nan
            var = float(grad @ cov @ grad)
            sigmas[k] = math.sqrt(max(var, 0.0)) if math.isfinite(var) else None
    except np.linalg.LinAlgError:
        pass
    return Estimate({k: float(v) for k, v in values.items()}, sigmas, n_points)


def _two_particle_data(runs: list[dict]):
    """Per launch: (x0, v0, times, observed (T, 2), p1/p2, start_time)."""
    data = []
    for run in runs:
        inp, out = run["input"], run["output"]
        if not isinstance(out, dict) or "pos2" not in out:
            continue
        obs = np.asarray(out["pos2"], dtype=float)
        times = out.get("measurement_times") or sorted(inp["measurement_times"])
        role = float(inp.get("p1", 1.0)) / float(inp.get("p2", 1.0))
        data.append((np.asarray([inp["pos2"]], float), np.asarray([inp.get("velocity2", [0, 0])], float),
                     times, obs, role, float(inp.get("start_time", 0.0) or 0.0)))
    if not data:
        raise ValueError("no two-particle data")
    return data


def _fit_two_particle(runs, force: Callable, starts, derived) -> Estimate:
    data = _two_particle_data(runs)

    def residuals(p):
        parts = []
        for x0, v0, times, obs, role, t0 in data:
            pred = _integrate(_central(lambda r, t: force(p, r, t), role), x0, v0, times, t0)[:, 0]
            parts.append((pred - obs).ravel())
        return np.concatenate(parts)

    n = sum(len(d[3]) for d in data)
    return _fit(residuals, starts, derived, n)


def _power_law(runs) -> Estimate:
    def force(p, r, t):
        return math.exp(p[0]) / r ** p[1]
    r3 = math.hypot(3.0, SOFT)
    starts = [np.array([math.log(k), n]) for k in (0.02, 0.2) for n in (1.0, 2.0)]
    return _fit_two_particle(runs, force, starts,
                             lambda p: {"n": p[1], "a3": math.exp(p[0]) / r3 ** p[1]})


def _yukawa_drop(runs) -> Estimate:
    # Screened 2D force: a = k K1(r/λ) / λ (the 2D Green's function of a screened field).
    def force(p, r, t):
        lam = math.exp(p[1])
        return math.exp(p[0]) * k1(r / lam) / lam

    def derived(p):
        lam = math.exp(p[1])

        def a(r):
            return math.exp(p[0]) * float(k1(math.hypot(r, SOFT) / lam)) / lam
        return {"drop": a(1.0) * 1.0 / (a(6.0) * 6.0)}

    starts = [np.array([math.log(k), math.log(lam)]) for k in (0.1, 1.0) for lam in (1.0, 5.0)]
    return _fit_two_particle(runs, force, starts, derived)


def _oscillator(runs) -> Estimate:
    def force(p, r, t):
        return p[0] * np.cos(p[1] * t + p[2]) / r

    def derived(p):
        w = abs(p[1])
        return {"ratio": math.cos(2 * p[1] + p[2]) / math.cos(p[2]) if abs(math.cos(p[2])) > 1e-9
                else float("nan"),
                "period": 2 * math.pi / w if w > 1e-9 else float("inf")}

    starts = [np.array([k, w, ph]) for k in (0.5, 1.0) for w in (0.3, 0.8, 1.5, 2.5)
              for ph in (0.0, 1.5)]
    return _fit_two_particle(runs, force, starts, derived)


# ------------------------------------------------------------------ many-body worlds

def _probe_data(runs, n_total: int):
    data = []
    for run in runs:
        inp, out = run["input"], run["output"]
        if not isinstance(out, dict) or "positions" not in out or "probe_positions" not in inp:
            continue
        pos = np.asarray(out["positions"], dtype=float)
        if pos.ndim != 3 or pos.shape[1] != n_total:
            continue
        times = out.get("measurement_times") or sorted(inp["measurement_times"])
        data.append((np.asarray(inp["probe_positions"], float),
                     np.asarray(inp.get("probe_velocities") or [[0, 0]] * 5, float),
                     times, pos[:, -5:, :]))
    if not data:
        raise ValueError("no probe data")
    return data


def _fit_field(runs, n_total: int, accel_of: Callable, starts, derived) -> Estimate:
    groups: dict[tuple, list] = {}
    for x0, v0, times, obs in _probe_data(runs, n_total):
        groups.setdefault(tuple(times), []).append((x0, v0, obs))
    batches = [(np.concatenate([g[0] for g in grp]), np.concatenate([g[1] for g in grp]),
                list(times), np.concatenate([g[2] for g in grp], axis=1))
               for times, grp in groups.items()]

    def residuals(p):
        accel = accel_of(p)
        return np.concatenate([(_integrate(accel, x0, v0, times) - obs).ravel()
                               for x0, v0, times, obs in batches])

    n = sum(b[3].shape[0] * b[3].shape[1] for b in batches)
    return _fit(residuals, starts, derived, n)


def _hubble_ether(runs) -> Estimate:
    # a = -q x / r_eff² + H x + d
    def accel_of(p):
        q, H, dx, dy = p

        def accel(x, t):
            r2 = np.sum(x * x, axis=-1, keepdims=True) + SOFT ** 2
            return -q * x / r2 + H * x + np.array([dx, dy])
        return accel

    starts = [np.array([q, 0.0, 0.0, 0.0]) for q in (3.0, 15.0)]
    return _fit_field(runs, 26, accel_of, starts,
                      lambda p: {"H": p[1], "drift": math.hypot(p[2], p[3])})


def _dark_matter(runs) -> Estimate:
    # a = -g x / r_eff² + d: one central source plus a uniform background pull
    def accel_of(p):
        g, dx, dy = p

        def accel(x, t):
            r2 = np.sum(x * x, axis=-1, keepdims=True) + SOFT ** 2
            return -g * x / r2 + np.array([dx, dy])
        return accel

    starts = [np.array([g, 0.0, 0.0]) for g in (1.0, 5.0, 15.0)]
    return _fit_field(runs, 25, accel_of, starts, lambda p: {"g4": p[0]})


def _circle(runs) -> Estimate:
    """Mean ring radius R(t); R'' = -c R^-n + L²/R³ with L from the launch speed."""
    data = []
    for run in runs:
        inp, out = run["input"], run["output"]
        if not isinstance(out, dict) or "positions" not in out:
            continue
        pos = np.asarray(out["positions"], dtype=float)
        if pos.ndim != 3 or pos.shape[1] != 11:
            continue
        R = np.linalg.norm(pos[:, 1:, :] - pos[:, :1, :], axis=2).mean(axis=1)
        R0 = float(inp.get("ring_radius", 5.0))
        L = R0 * float(inp.get("initial_tangential_velocity", 0.0))
        times = out.get("measurement_times") or sorted(inp["measurement_times"])
        data.append((R0, L, times, R))
    if not data:
        raise ValueError("no ring data")

    def residuals(p):
        c, n = math.exp(p[0]), p[1]
        parts = []
        for R0, L, times, R in data:
            def accel(x, t):
                r = np.maximum(x[..., :1], 1e-6)
                return np.concatenate([-c * r ** -n + L * L / r ** 3, 0 * r], axis=-1)
            pred = _integrate(accel, np.array([[R0, 0.0]]), np.zeros((1, 2)), times)[:, 0, 0]
            parts.append(pred - R)
        return np.concatenate(parts)

    starts = [np.array([math.log(c), n]) for c in (0.1, 1.0) for n in (1.0, 2.0)]
    return _fit(residuals, starts, lambda p: {"n": p[1]}, sum(len(d[3]) for d in data))


ESTIMATORS: dict[str, Callable[[list[dict]], Estimate]] = {
    "gravity": _power_law, "fractional": _power_law, "yukawa": _yukawa_drop,
    "oscillator": _oscillator, "hubble": _hubble_ether, "ether": _hubble_ether,
    "dark_matter": _dark_matter, "circle": _circle,
}


def estimate(world: str, runs: list[dict]) -> Estimate:
    """Fit this world's quantities to every run with usable data. Raises ValueError if the
    data do not support a fit."""
    if world not in ESTIMATORS:
        raise KeyError(f"no estimator for world {world!r}")
    try:
        return ESTIMATORS[world](runs)
    except (ArithmeticError, np.linalg.LinAlgError, KeyError, TypeError, IndexError) as exc:
        raise ValueError(f"no fit: {type(exc).__name__}: {exc}") from exc
