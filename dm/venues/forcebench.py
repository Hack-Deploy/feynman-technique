"""Fixed-menu ForceBench venue backed by DiscoverPhysics' N-body worlds."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from dm.store import ATTEMPTS_DIR
from dm.types import InsufficientCredits, SubmittedAttempt
from dm.wallet import Wallet

if TYPE_CHECKING:
    from dm.solvers import MenuSolver, Observation


WORLDS = (
    "gravity",
    "yukawa",
    "coulomb_easy",
    "oscillator",
    "fractional",
    "extra_dimensions",
)
NOISE_STD = 0.03
BUDGET = 8
MEASUREMENT_TIMES = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
SEED_ACTION = 3


@dataclass(frozen=True)
class Launch:
    action: int
    r0: float
    v: tuple[float, float]
    p1: float
    p2: float

    def experiment(self) -> dict:
        return {
            "p1": float(self.p1),
            "p2": float(self.p2),
            "pos2": [float(self.r0), 0.0],
            "velocity2": [float(self.v[0]), float(self.v[1])],
            "measurement_times": [float(t) for t in MEASUREMENT_TIMES],
        }


MENU: tuple[Launch, ...] = tuple(
    [Launch(i + 1, r0, (0.0, 0.0), 1.0, 1.0) for i, r0 in enumerate((1.5, 2, 3, 4, 5, 6, 8, 10))]
    + [
        Launch(9, 2, (0.0, 0.4), 1.0, 1.0),
        Launch(10, 4, (0.0, 0.4), 1.0, 1.0),
        Launch(11, 4, (0.0, 0.0), 2.0, 1.0),
        Launch(12, 3, (0.0, 0.0), 1.0, 2.0),
        Launch(13, 4, (0.0, 0.0), 2.0, 2.0),
    ]
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _relative_or_absolute(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(_repo_root()))
    except ValueError:
        return str(resolved)


def run_attempt(
    solver: MenuSolver | str,
    world: str,
    seed: int,
    wallet: Wallet,
    price: float,
    *,
    transcript_dir: Path | str | None = None,
) -> SubmittedAttempt:
    """Run a solver against one noisy world, charging before each paid launch."""
    if world not in WORLDS:
        raise ValueError(f"Unknown ForceBench world: {world!r}")
    if isinstance(solver, str):
        from dm.solvers import get_solver

        solver = get_solver(solver)

    from scienceagent.worlds import get_world
    from dm.solvers import Observation

    executor = get_world(
        world, engine="nbody", noise_std=NOISE_STD, noise_seed=seed
    )["executor"]
    initial_credits = wallet.balance + wallet.lab_revenue
    revenue_before = wallet.lab_revenue
    events_before = len(wallet.events)
    seed_launch = next(launch for launch in MENU if launch.action == SEED_ACTION)
    seed_input = seed_launch.experiment()
    seed_output = executor.run([seed_input])[0]
    seed_obs = Observation(SEED_ACTION, seed_input, seed_output)
    solver.start(
        seed=seed,
        menu=MENU,
        noise_std=NOISE_STD,
        budget=BUDGET,
        seed_obs=seed_obs,
    )

    unused = {launch.action for launch in MENU if launch.action != SEED_ACTION}
    paid_training: list[dict] = []
    rounds_log: list[dict] = []
    stopped_reason = "budget"
    for round_num in range(1, BUDGET + 1):
        action = solver.next_action(unused=tuple(sorted(unused)), price=price)
        if action is None:
            stopped_reason = "solver_stop"
            break
        if action not in unused:
            raise ValueError(f"Solver selected unavailable menu action {action}")
        try:
            wallet.charge(1, price, world=world, round_num=round_num)
        except InsufficientCredits:
            stopped_reason = "insufficient_credits"
            break
        launch = next(launch for launch in MENU if launch.action == action)
        experiment = launch.experiment()
        output = executor.run([experiment])[0]
        observation = Observation(action, experiment, output)
        solver.observe(observation)
        paid_training.append({"input": experiment, "output": output})
        rounds_log.append(
            {
                "round": round_num,
                "choice": action,
                "input": experiment,
                "output": output,
                "solver": solver.round_log(),
            }
        )
        unused.remove(action)
    else:
        stopped_reason = "budget"

    submission = solver.submit()
    experiments = len(paid_training)
    attempt_charges = [
        event
        for event in wallet.events[events_before:]
        if event["type"] == "experiment_charged"
    ]
    lab_cost = wallet.lab_revenue - revenue_before
    if abs((wallet.balance + wallet.lab_revenue) - initial_credits) > 1e-12:
        raise AssertionError("Wallet credit conservation violated during ForceBench attempt")
    if len(attempt_charges) != experiments or abs(
        lab_cost - sum(event["amount"] for event in attempt_charges)
    ) > 1e-12:
        raise AssertionError("Attempt charges do not match its paid launches")

    attempt = SubmittedAttempt(
        source="live",
        protocol="forcebench_menu",
        venue="forcebench",
        world=world,
        solver=solver.name,
        seed=seed,
        stated_p_success=submission.stated_p,
        rounds=experiments,
        experiments=experiments,
        lab_cost=lab_cost,
        submitted_law=submission.law_source,
        explanation=submission.explanation,
        training=paid_training,
        transcript_path=None,
        created_at="",
        extra={
            "noise_std": NOISE_STD,
            "menu_source": "MDA arXiv 2608.09696 v3 App. C Table 5",
            "seed_action": SEED_ACTION,
            "seed_data": {"input": seed_input, "output": seed_output},
            "actions": [row["choice"] for row in rounds_log],
            "stopped_reason": stopped_reason,
            "price": price,
        },
    )
    target_dir = (
        Path(transcript_dir)
        if transcript_dir is not None
        else ATTEMPTS_DIR / "transcripts" / "forcebench"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = target_dir / f"{solver.name}_{world}_s{seed}.json"
    transcript = {
        "solver": solver.name,
        "world": world,
        "seed": seed,
        "price": price,
        "stopped_reason": stopped_reason,
        "menu": [launch.experiment() | {"action": launch.action} for launch in MENU],
        "seed_data": {"action": SEED_ACTION, "input": seed_input, "output": seed_output},
        "rounds": rounds_log,
        "submission": {
            "law": submission.law_source,
            "stated_p": submission.stated_p,
            "explanation": submission.explanation,
            "summary": submission.summary,
            "top_model": submission.summary.get("top_model"),
        },
    }
    transcript_path.write_text(json.dumps(transcript, sort_keys=True, indent=1) + "\n")
    attempt = replace(attempt, transcript_path=_relative_or_absolute(transcript_path))
    return attempt
