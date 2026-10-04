"""The lab the agent buys experiments from: a vendor executor that reports positions only.

The vendor executors add noise to positions but return exact velocities, so one cheap
experiment would reveal every force law and make noise, precision and price irrelevant.
This wrapper drops velocity fields from every result; the agent is told it observes noisy
positions only. The vendor code is not edited.
"""

from __future__ import annotations

VELOCITY_KEYS = ("velocity1", "velocity2", "velocities", "background_initial_velocities")

LAB_NOTE = (
    "## WHAT THE LAB RETURNS (this overrides the output format described above)\n"
    "The lab reports measured positions only, each with Gaussian noise (σ = {noise:g} per "
    "coordinate). Velocity fields are not returned. Infer velocities and accelerations from "
    "the positions.")


def strip_velocities(result: dict) -> dict:
    return {k: v for k, v in result.items() if k not in VELOCITY_KEYS}


class PositionsOnlyExecutor:
    """Wraps a vendor executor; ``run`` returns its results without velocities."""

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    def run(self, experiments: list[dict]) -> list[dict]:
        return [strip_velocities(r) for r in self.inner.run(experiments)]
