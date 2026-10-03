"""Numerical fitting, generated-law, and end-to-end ForceBench solver tests."""

from __future__ import annotations

import json
import math
import time

import numpy as np
import pytest

from dm.solvers._inference import (
    ModelInference,
    _fit_model,
    generate_law,
    simulate,
)
from dm.solvers import Observation
from dm.venues.forcebench import MENU, WORLDS, SubmittedAttempt
from tests.forcebench_local import run_grid


def _launches() -> list[dict]:
    return [
        MENU[index].experiment()
        for index in (1, 3, 8, 10, 11)
    ]


def test_all_model_role_laws_match_integrator_and_run_quickly() -> None:
    launches = _launches()
    times = launches[0]["measurement_times"]
    model_params = {
        "power": (-0.7, 1.4),
        "log": (0.16,),
        "yukawa": (0.08, 2.0),
        "timemod": (1.2, 1.3, 0.7),
        "crossover": (0.2, -0.1),
    }
    roles = ("ratio", "p1", "product")
    for family, params in model_params.items():
        for role in roles:
            expected = simulate(family, role, params, launches)
            law_source = generate_law(family, role, params)
            namespace: dict = {}
            exec(law_source, namespace)
            law = namespace["discovered_law"]
            for launch_index, launch in enumerate(launches):
                for time_index, duration in enumerate(times):
                    started = time.perf_counter()
                    position, _ = law(
                        pos1=[0.0, 0.0],
                        pos2=launch["pos2"],
                        p1=launch["p1"],
                        p2=launch["p2"],
                        velocity2=launch["velocity2"],
                        duration=duration,
                    )
                    elapsed = time.perf_counter() - started
                    assert elapsed < 0.5
                    assert np.allclose(
                        position,
                        expected[0, launch_index, time_index],
                        rtol=0,
                        atol=1e-9,
                    )
            # Longer trajectories are part of the external evaluator's default cases.
            started = time.perf_counter()
            law(
                pos1=[0.0, 0.0],
                pos2=[3.0, 0.0],
                p1=1.0,
                p2=1.0,
                velocity2=[0.0, 0.5],
                duration=10.0,
            )
            assert time.perf_counter() - started < 0.5


def test_log_ratio_recovered_from_noisy_synthetic_data() -> None:
    launches = [MENU[index] for index in (0, 2, 4, 8)]
    rng = np.random.default_rng(19)
    predictions = simulate(
        "log",
        "ratio",
        [0.16],
        [launch.experiment() for launch in launches],
    )[0]
    observations = []
    for index, (launch, output) in enumerate(zip(launches, predictions)):
        noisy = output + rng.normal(0.0, 0.03, size=output.shape)
        observations.append(
            Observation(
                launch.action,
                launch.experiment(),
                {"measurement_times": list(launch.experiment()["measurement_times"]), "pos2": noisy.tolist()},
            )
        )
    inference = ModelInference(MENU, 0.03, n_starts=1)
    for observation in observations:
        inference.add_observation(observation)
    assert inference.top.family in {"log", "power"}
    if inference.top.family == "log":
        assert inference.top.params[0] == pytest.approx(0.16, rel=0.1)
    else:
        assert inference.top.params[1] == pytest.approx(1.0, abs=0.1)
        assert inference.top.params[0] == pytest.approx(0.16, rel=0.1)


def test_repulsive_power_sign_and_charge_role_recovered() -> None:
    selected = [MENU[index] for index in (0, 3, 10, 11, 12)]
    launches = [launch.experiment() for launch in selected]
    positions = simulate("power", "p1", [-1.0, 2.0], launches)[0]
    rng = np.random.default_rng(7)
    observations = positions + rng.normal(0.0, 0.03, size=positions.shape)
    params, rss = _fit_model(
        "power",
        "p1",
        launches,
        observations,
        None,
        n_starts=1,
    )
    assert math.isfinite(rss)
    assert params[0] < 0
    assert params[0] == pytest.approx(-1.0, rel=0.1)
    assert params[1] == pytest.approx(2.0, abs=0.1)


@pytest.mark.slow
def test_full_forcebench_grid_conserves_credits_and_recovers_two_worlds(tmp_path) -> None:
    results = run_grid(output_path=tmp_path / "forcebench_grid.json", processes=6)
    assert len(results) == len(WORLDS) * 2 * 5
    assert all(isinstance(item["attempt"], SubmittedAttempt) for item in results)
    assert all(item["experiments"] <= 8 for item in results)
    assert all(item["rounds"] == item["experiments"] for item in results)
    assert all(item["wallet_conserved"] for item in results)
    payload = json.loads((tmp_path / "forcebench_grid.json").read_text())
    assert len(payload["results"]) == len(results)
    assert len(payload["identification_summary"]) == len(WORLDS) * 2
    assert all(
        "baseline" in item and "identified" in item
        for item in payload["results"]
    )
    bayes_gravity = [
        item for item in results
        if item["solver"] == "bayes_lite" and item["world"] == "gravity"
    ]
    bayes_fractional = [
        item for item in results
        if item["solver"] == "bayes_lite" and item["world"] == "fractional"
    ]
    assert sum(item["passed"] for item in bayes_gravity) >= 3
    assert sum(item["passed"] for item in bayes_fractional) >= 3
    assert (tmp_path / "forcebench_grid.json").exists()
