"""Hidden test cases and normalising variance for each world.

Two-particle worlds get ForceBench-style interventional cases (MDA arXiv
2608.09696v3, Appendix C.2): one long-horizon probe plus single-knob
interventions with p1 ∈ {3, 4, 5} (p2 = 1) and p2 ∈ {3, 5} (p1 = 1), all from one
base launch drawn from the test seed. Multi-particle worlds get the same keys as
their default (public) cases with initial conditions jittered from the test seed.

``norm_variance`` follows the vendor's ``_WORLD_VARS`` convention exactly
(verified: it reproduces all 11 published values on the default cases):
``np.var`` over the flattened noise-free agent-facing positions, i.e. ``pos2`` for
two-particle worlds and every particle in ``positions`` otherwise (even where the
evaluator scores only the probes).
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

TWO_PARTICLE_WORLDS = ("gravity", "yukawa", "coulomb_easy", "oscillator",
                       "fractional", "extra_dimensions")
MULTI_PARTICLE_WORLDS = ("circle", "three_species", "dark_matter", "ether", "hubble")
PUBLIC_WORLDS = TWO_PARTICLE_WORLDS + MULTI_PARTICLE_WORLDS

LONG_TIMES = [float(t) for t in range(1, 11)]          # 1..10
SHORT_TIMES = [float(t) for t in range(1, 11)]          # 1..10 (design V1)
R0_RANGE = (3.0, 6.0)    # base launch radius (design V1: 1/r must fail yukawa)
SPEED_RANGE = (0.2, 0.5)  # tangential launch speed
MIN_RADIUS = 0.5         # reject two-particle cases that pass this close to the source
MAX_REDRAWS = 200


def build_world(world: str, noise_std: float = 0.0, noise_seed: int = 0) -> dict:
    from scienceagent.worlds import get_world
    return get_world(world, engine="nbody", noise_std=noise_std, noise_seed=noise_seed)


def default_cases(world: str) -> list[dict]:
    import scienceagent.evaluator as E
    table = {
        "circle": E._CIRCLE_TEST_CASES,
        "three_species": E._THREE_SPECIES_TEST_CASES,
        "dark_matter": E._DARK_MATTER_TEST_CASES,
        "ether": E._ETHER_TEST_CASES,
        "hubble": E._HUBBLE_TEST_CASES,
    }
    return copy.deepcopy(table.get(world, E._DEFAULT_TEST_CASES))


def scored_positions(world: str, ground_truths: list[dict]) -> np.ndarray:
    """Positions the normalising variance is taken over (vendor convention)."""
    key = "pos2" if world in TWO_PARTICLE_WORLDS else "positions"
    return np.concatenate([np.asarray(g[key], dtype=float).reshape(-1, 2)
                           for g in ground_truths])


def norm_variance(world: str, cases: list[dict], executor=None) -> float:
    executor = executor or build_world(world)["executor"]
    with executor.noise_disabled():
        gts = executor.run(cases)
    return float(np.var(scored_positions(world, gts)))


# ------------------------------------------------------------ two-particle

def _min_radius(executor, case: dict) -> float:
    """Closest approach to the source, sampled densely (noise-free)."""
    horizon = max(case["measurement_times"])
    dense = dict(case, measurement_times=[round(0.05 * k, 2)
                                          for k in range(1, int(horizon / 0.05) + 1)])
    with executor.noise_disabled():
        out = executor.run([dense])[0]
    rel = np.asarray(out["pos2"]) - np.asarray(out["pos1"])
    return float(np.min(np.linalg.norm(rel, axis=1)))


def _entropy(test_seed: int, domain: int, salt: str) -> list[int]:
    """RNG seed material. With a salt (secret-derived hex), the cases cannot be
    regenerated from the public code and a guessed test seed."""
    words = [int(salt[i:i + 8], 16) for i in range(0, len(salt), 8)] if salt else []
    return [test_seed, domain, *words]


def two_particle_cases(world: str, test_seed: int, executor=None, salt: str = "") -> list[dict]:
    executor = executor or build_world(world)["executor"]
    rng = np.random.default_rng(_entropy(test_seed, 7919, salt))  # 7919: domain separation
    for _ in range(MAX_REDRAWS):
        r0 = float(rng.uniform(*R0_RANGE))
        theta = float(rng.uniform(0.0, 2 * np.pi))
        speed = float(rng.uniform(*SPEED_RANGE))
        pos2 = [round(r0 * np.cos(theta), 4), round(r0 * np.sin(theta), 4)]
        # tangential launch (counter-clockwise), so no case is a head-on fall
        vel2 = [round(-speed * np.sin(theta), 4), round(speed * np.cos(theta), 4)]
        base = {"pos2": pos2, "velocity2": vel2}
        cases = [{"p1": 1.0, "p2": 1.0, **base, "measurement_times": LONG_TIMES}]
        cases += [{"p1": p1, "p2": 1.0, **base, "measurement_times": SHORT_TIMES}
                  for p1 in (3.0, 4.0, 5.0)]
        cases += [{"p1": 1.0, "p2": p2, **base, "measurement_times": SHORT_TIMES}
                  for p2 in (3.0, 5.0)]
        if all(_min_radius(executor, c) >= MIN_RADIUS for c in cases):
            return cases
    raise RuntimeError(f"no valid hidden cases for {world} after {MAX_REDRAWS} draws")


# ---------------------------------------------------------- multi-particle

def _jitter_xy(rng, xy, sd):
    return [[round(float(x + rng.normal(0, sd)), 4), round(float(y + rng.normal(0, sd)), 4)]
            for x, y in xy]


def multi_particle_cases(world: str, test_seed: int, salt: str = "") -> list[dict]:
    rng = np.random.default_rng(_entropy(test_seed, 104729, salt))
    cases = default_cases(world)
    for c in cases:
        if world == "circle":
            c["ring_radius"] = round(float(c["ring_radius"] * rng.uniform(0.85, 1.15)), 4)
            c["initial_tangential_velocity"] = round(
                float(c["initial_tangential_velocity"] * rng.uniform(0.8, 1.2)), 4)
            continue
        c["probe_positions"] = _jitter_xy(rng, c["probe_positions"], 0.75)
        c["probe_velocities"] = _jitter_xy(rng, c["probe_velocities"], 0.1)
        if "probe_masses" in c:
            c["probe_masses"] = [float(m) for m in rng.permutation(c["probe_masses"])]
    return cases


def _valid(world: str, cases: list[dict], executor) -> bool:
    try:
        with executor.noise_disabled():
            gts = executor.run(cases)
        pos = scored_positions(world, gts)
        return bool(np.all(np.isfinite(pos)) and np.max(np.abs(pos)) < 1e3)
    except Exception:
        return False


def hidden_cases(world: str, test_seed: int, salt: str = "") -> tuple[list[dict], float, bool]:
    """(test_cases, norm_variance, public_tests) for one world and test seed."""
    if world not in PUBLIC_WORLDS:
        raise ValueError(f"unknown or non-public world {world!r}")
    executor = build_world(world)["executor"]
    if world in TWO_PARTICLE_WORLDS:
        cases, public = two_particle_cases(world, test_seed, executor, salt), False
    else:
        cases, public = multi_particle_cases(world, test_seed, salt), False
        if not _valid(world, cases, executor):
            cases, public = default_cases(world), True
    return cases, norm_variance(world, cases, executor), public


def jsonable(obj: Any) -> Any:
    """Plain floats/lists so the preregistration hashes the same everywhere."""
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj
