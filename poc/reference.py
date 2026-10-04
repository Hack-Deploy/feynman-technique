"""The reference experimenter: a fixed, cheap design per world plus a least-squares fit.

It never sees the truth. Calibration runs it on seeds the benchmark never uses
(``CALIBRATION_SEEDS``) to set each quantity's tolerance (3 × its RMS error) and each prize
(design cost ≈ 30% of the prize); ``poc.calibrate`` does that and checks the result.
"""

from __future__ import annotations

from poc import estimate

CALIBRATION_SEEDS = tuple(range(100, 120))


def _launch(r: float, p1: float, times: list[float], p2: float = 1.0, **extra) -> dict:
    return {"p1": p1, "p2": p2, "pos2": [r, 0.0], "velocity2": [0.0, 0.0],
            "measurement_times": times, **extra}


def _steps(stop: float, step: float) -> list[float]:
    return [round(step * i, 2) for i in range(1, int(round(stop / step)) + 1)]


def _probes(points: list[list[float]], times: list[float]) -> dict:
    return {"probe_positions": points, "probe_velocities": [[0.0, 0.0]] * 5,
            "measurement_times": times}


# p1, p2 stay inside the vendor's documented range [0.1, 10]; launches end before the probe
# falls near the source, where softening and integrator details dominate.
DESIGNS: dict[str, list[dict]] = {
    "gravity": [_launch(r, 4.0, _steps(3.0, 0.5)) for r in (2.0, 4.0, 6.0)],
    "fractional": [_launch(r, p1, _steps(3.0, 0.5))
                   for r, p1 in ((1.5, 1.0), (3.0, 5.0), (6.0, 10.0))],
    "yukawa": [_launch(1.0, 1.0, _steps(1.5, 0.1), p2=0.3), _launch(2.0, 3.0, _steps(2.5, 0.5)),
               _launch(4.0, 10.0, _steps(3.0, 0.5), p2=0.5),
               _launch(6.0, 10.0, _steps(3.0, 0.25), p2=0.2)],
    "oscillator": [_launch(3.0, 1.0, _steps(6.0, 0.5))],
    "hubble": [_probes([[12, 0], [0, 12], [-12, 0], [0, -12], [18, 0]], _steps(3.0, 0.5)),
               _probes([[-18, 0], [0, 18], [0, -18], [12, 12], [-12, -12]], _steps(3.0, 0.5))],
    "dark_matter": [_probes([[4, 0], [0, 4], [-4, 0], [0, -4], [3, 3]], _steps(1.0, 0.25))],
    "circle": [{"ring_radius": r, "initial_tangential_velocity": 0.0,
                "measurement_times": _steps(2.0, 0.25)} for r in (1.5, 3.0, 6.0)],
}
DESIGNS["ether"] = DESIGNS["hubble"]


def design(world: str) -> list[dict]:
    return [dict(e) for e in DESIGNS[world]]


def analyse(world: str, inputs: list[dict], outputs: list[dict]) -> estimate.Estimate:
    return estimate.estimate(world, [{"input": i, "output": o} for i, o in zip(inputs, outputs)])
