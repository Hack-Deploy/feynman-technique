"""ForceBench menu, accounting, transcript, and local scoring coverage."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from dm.solvers import Submission
from dm.solvers.bayes_lite import BayesLite
from dm.types import InsufficientCredits
from dm.venues.forcebench import (
    BUDGET,
    MENU,
    MEASUREMENT_TIMES,
    SEED_ACTION,
    SubmittedAttempt,
    run_attempt,
)
from dm.wallet import Wallet
from tests.forcebench_local import identify_model, score_baseline, score_local


def test_menu_matches_forcebench_table_and_vendor_shape() -> None:
    expected = [
        (1, 1.5, (0, 0), 1, 1),
        (2, 2, (0, 0), 1, 1),
        (3, 3, (0, 0), 1, 1),
        (4, 4, (0, 0), 1, 1),
        (5, 5, (0, 0), 1, 1),
        (6, 6, (0, 0), 1, 1),
        (7, 8, (0, 0), 1, 1),
        (8, 10, (0, 0), 1, 1),
        (9, 2, (0, 0.4), 1, 1),
        (10, 4, (0, 0.4), 1, 1),
        (11, 4, (0, 0), 2, 1),
        (12, 3, (0, 0), 1, 2),
        (13, 4, (0, 0), 2, 2),
    ]
    assert len(MENU) == 13
    assert [
        (item.action, item.r0, item.v, item.p1, item.p2) for item in MENU
    ] == expected
    assert SEED_ACTION == 3
    for launch in MENU:
        experiment = launch.experiment()
        assert experiment == {
            "p1": float(launch.p1),
            "p2": float(launch.p2),
            "pos2": [float(launch.r0), 0.0],
            "velocity2": [float(launch.v[0]), float(launch.v[1])],
            "measurement_times": list(MEASUREMENT_TIMES),
        }


def test_wallet_exact_fractional_charge() -> None:
    wallet = Wallet("fractional", 0.3)
    wallet.charge(3, 0.1, world="gravity", round_num=1)
    assert wallet.balance == 0.0
    assert wallet.lab_revenue == 0.3
    assert wallet.events == [
        {
            "type": "experiment_charged",
            "from": "agent:fractional",
            "to": "lab",
            "amount": 0.3,
            "count": 3,
            "world": "gravity",
            "round": 1,
        }
    ]


def test_wallet_thousand_fractional_charges_conserve_exactly() -> None:
    wallet = Wallet("many", 100.0)
    for round_num in range(1000):
        wallet.charge(1, 0.1, world="yukawa", round_num=round_num)
        assert wallet.balance + wallet.lab_revenue == 100.0
    assert wallet.balance == 0.0
    assert wallet.lab_revenue == 100.0


def test_wallet_insufficient_charge_preserves_balances_and_prior_events() -> None:
    wallet = Wallet("short", 1.0)
    wallet.charge(1, 0.5, world="gravity", round_num=1)
    balance_before = wallet.balance
    revenue_before = wallet.lab_revenue
    earlier_events = list(wallet.events)
    with pytest.raises(InsufficientCredits):
        wallet.charge(2, 0.3, world="gravity", round_num=2)
    assert wallet.balance == balance_before
    assert wallet.lab_revenue == revenue_before
    assert wallet.events[:-1] == earlier_events
    assert wallet.events[-1] == {
        "type": "insufficient_credits",
        "from": "agent:short",
        "to": "lab",
        "amount": 0.6,
        "count": 2,
        "balance": 0.5,
        "world": "gravity",
        "round": 2,
    }


def test_wallet_rejects_non_milli_credit_values() -> None:
    with pytest.raises(ValueError):
        Wallet("bad", 0.0001)
    wallet = Wallet("bad", 1.0)
    with pytest.raises(ValueError):
        wallet.charge(1, 0.0001, world="gravity", round_num=1)


@pytest.fixture
def bayes_gravity_attempt(tmp_path) -> tuple[SubmittedAttempt, Wallet]:
    wallet = Wallet("bayes-gravity", 100.0)
    attempt = run_attempt(
        BayesLite(n_starts=1),
        "gravity",
        0,
        wallet,
        1.0,
        transcript_dir=tmp_path / "transcripts",
    )
    return attempt, wallet


def test_run_attempt_transcript_conservation_baseline_and_determinism(
    bayes_gravity_attempt: tuple[SubmittedAttempt, Wallet],
    tmp_path,
) -> None:
    attempt, wallet = bayes_gravity_attempt
    assert attempt.experiments <= BUDGET
    assert attempt.rounds == attempt.experiments
    assert attempt.lab_cost == attempt.experiments
    assert len(attempt.training) == attempt.experiments
    assert wallet.balance + wallet.lab_revenue == 100.0
    charges = [
        event for event in wallet.events if event["type"] == "experiment_charged"
    ]
    assert len(charges) == attempt.experiments
    assert all(
        row["input"]["pos2"] and len(row["output"]["pos2"]) == 6
        for row in attempt.training
    )
    assert attempt.submitted_law and "def discovered_law" in attempt.submitted_law
    assert attempt.transcript_path
    transcript = Path(attempt.transcript_path)
    if not transcript.is_absolute():
        transcript = Path(__file__).resolve().parents[1] / transcript
    assert transcript.exists()
    prior_events = list(wallet.events)
    prior_balance = wallet.balance
    prior_revenue = wallet.lab_revenue
    result = score_local(attempt, "gravity")
    assert result["passed"], result
    baseline = score_baseline(attempt, "gravity")
    assert baseline["passed"], baseline
    assert baseline["fitted_k"] == pytest.approx(1 / (2 * 3.141592653589793), rel=0.2)
    assert wallet.events == prior_events
    assert wallet.balance == prior_balance
    assert wallet.lab_revenue == prior_revenue
    repeated = run_attempt(
        BayesLite(n_starts=1),
        "gravity",
        0,
        Wallet("repeated-bayes-gravity", 100.0),
        1.0,
        transcript_dir=tmp_path / "repeated",
    )
    repeated_path = Path(repeated.transcript_path)
    assert transcript.read_bytes() == repeated_path.read_bytes()
    assert attempt.submitted_law == repeated.submitted_law


class _AlwaysChooseSolver:
    name = "test_always_choose"

    def start(self, *, seed, menu, noise_std, budget, seed_obs) -> None:
        self.menu = menu
        self.used = {seed_obs.action}

    def next_action(self, *, unused, price) -> int | None:
        return min(unused)

    def observe(self, obs) -> None:
        self.used.add(obs.action)

    def submit(self) -> Submission:
        return Submission(
            "def discovered_law(pos1, pos2, p1, p2, velocity2, duration):\n"
            "    return list(pos2), list(velocity2)\n",
            0.1,
            "A deterministic test law.",
            {"top_model": {"family": "test", "role": "ratio", "params": []}},
        )

    def round_log(self) -> dict:
        return {"models": [], "voi": {}}


def test_too_small_wallet_stops_before_simulator_and_submits(tmp_path) -> None:
    wallet = Wallet("short-run", 2.0)
    attempt = run_attempt(
        _AlwaysChooseSolver(),
        "gravity",
        2,
        wallet,
        1.0,
        transcript_dir=tmp_path,
    )
    assert attempt.experiments == 2
    assert attempt.rounds == 2
    assert any(event["type"] == "insufficient_credits" for event in wallet.events)
    assert attempt.extra["stopped_reason"] == "insufficient_credits"
    assert attempt.submitted_law
    assert wallet.balance == 0.0
    assert wallet.lab_revenue == 2.0


@pytest.mark.parametrize(
    ("world", "model", "expected", "label"),
    [
        (
            "gravity",
            {"family": "log", "role": "ratio", "params": [0.16]},
            True,
            "exact",
        ),
        (
            "gravity",
            {"family": "power", "role": "ratio", "params": [0.16, 1.05]},
            True,
            "exact",
        ),
        (
            "gravity",
            {"family": "power", "role": "ratio", "params": [0.16, 1.1]},
            False,
            None,
        ),
        (
            "fractional",
            {"family": "power", "role": "ratio", "params": [0.16, 2.0]},
            True,
            "exact",
        ),
        (
            "yukawa",
            {"family": "yukawa", "role": "ratio", "params": [0.16, 2.0]},
            True,
            "exact",
        ),
        (
            "oscillator",
            {"family": "timemod", "role": "ratio", "params": [0.8, 1.5, 0]},
            True,
            "exact",
        ),
        (
            "coulomb_easy",
            {"family": "power", "role": "p1", "params": [-1.0, 2.0]},
            True,
            "exact",
        ),
        (
            "extra_dimensions",
            {"family": "crossover", "role": "ratio", "params": [0.1, 0.2]},
            False,
            "closest: crossover/ratio",
        ),
    ],
)
def test_identify_model_uses_world_truth_mapping(
    world: str, model: dict, expected: bool, label: str | None
) -> None:
    assert identify_model(world, model) == (expected, label)


def test_venue_and_solver_sources_do_not_import_or_dynamically_load_oracle() -> None:
    root = Path(__file__).resolve().parents[1]
    for folder in (root / "dm" / "venues", root / "dm" / "solvers"):
        for path in folder.glob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    assert all(not alias.name.startswith("dm.oracle") for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    assert not (node.module or "").startswith("dm.oracle")
                    if node.module == "dm":
                        assert all(alias.name != "oracle" for alias in node.names)
                elif isinstance(node, ast.Call):
                    func = node.func
                    is_import_call = (
                        isinstance(func, ast.Attribute)
                        and isinstance(func.value, ast.Name)
                        and func.value.id == "importlib"
                        and func.attr in {"import_module", "__import__"}
                    ) or (isinstance(func, ast.Name) and func.id == "__import__")
                    if is_import_call and node.args and isinstance(node.args[0], ast.Constant):
                        assert not str(node.args[0].value).startswith("dm.oracle")
