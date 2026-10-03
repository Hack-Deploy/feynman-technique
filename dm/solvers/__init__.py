"""World-blind offline solver interfaces and registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from dm.venues.forcebench import Launch


@dataclass(frozen=True)
class Observation:
    action: int
    input: dict
    output: dict


@dataclass(frozen=True)
class Submission:
    law_source: str
    stated_p: float
    explanation: str
    summary: dict


class MenuSolver(Protocol):
    name: str

    def start(
        self,
        *,
        seed: int,
        menu: tuple[Launch, ...],
        noise_std: float,
        budget: int,
        seed_obs: Observation,
    ) -> None: ...

    def next_action(self, *, unused: tuple[int, ...], price: float) -> int | None: ...

    def observe(self, obs: Observation) -> None: ...

    def submit(self) -> Submission: ...

    def round_log(self) -> dict: ...


def get_solver(name: str) -> MenuSolver:
    """Instantiate one of the registered deterministic menu solvers."""
    if name == "bayes_lite":
        from dm.solvers.bayes_lite import BayesLite

        return BayesLite()
    if name == "random_menu":
        from dm.solvers.random_menu import RandomMenu

        return RandomMenu()
    raise ValueError(f"Unknown ForceBench solver: {name!r}")
