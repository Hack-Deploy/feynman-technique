"""Adversarial readiness checks for a deterministic ForceBench replay market."""

import ast
import hashlib
import json
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import pytest

from analysis import compute_final_balances
from dm.outcomes import ReplayPool, experiments_cost, recorded_cost
from dm.store import AttemptStore
from dm.types import AttemptRecord
from market import MarketRun, run_market


WORLDS = [
    "gravity",
    "yukawa",
    "coulomb_easy",
    "oscillator",
    "fractional",
    "extra_dimensions",
]
AGENTS = ["bayes_lite", "random_menu", "llm_menu"]
PASS_SEEDS = {
    "gravity": set(range(5)),
    "fractional": set(range(5)),
    "yukawa": {0, 1, 2},
    "oscillator": {1, 2, 3},
    "coulomb_easy": {0},
    "extra_dimensions": {3, 4},
}
REPO_ROOT = Path(__file__).resolve().parents[2]


def forcebench_pool(price: float) -> list[AttemptRecord]:
    records = []
    for world in WORLDS:
        for seed in range(5):
            passed = seed in PASS_SEEDS[world]
            for solver, experiments, stated_p in (
                ("bayes_lite", [3, 4, 5, 3, 4][seed], 0.7),
                ("random_menu", 8, 0.5),
            ):
                records.append(AttemptRecord(
                    attempt_id=f"fb-{solver}-{world}-{seed}",
                    source="sim",
                    protocol="forcebench_menu",
                    venue="forcebench",
                    world=world,
                    solver=solver,
                    seed=seed,
                    stated_p_success=stated_p,
                    rounds=experiments,
                    experiments=experiments,
                    lab_cost=experiments * price,
                    verdict={
                        "normalised_mse": 0.01 if passed else 0.5,
                        "passed": passed,
                        "prereg_commitment": f"c-{world}-{seed}",
                        "explanation_score": None,
                    },
                    extra={"price": price, "seed_launch_free": True},
                ))
            if world in ("gravity", "yukawa"):
                passed = seed == 0
                records.append(AttemptRecord(
                    attempt_id=f"fb-llm_menu-{world}-{seed}",
                    source="sim",
                    protocol="forcebench_menu",
                    venue="forcebench",
                    world=world,
                    solver="llm_menu",
                    seed=seed,
                    stated_p_success=0.9,
                    rounds=8,
                    experiments=8,
                    lab_cost=8 * price,
                    verdict={
                        "normalised_mse": 0.01 if passed else 0.5,
                        "passed": passed,
                        "prereg_commitment": f"c-{world}-{seed}",
                        "explanation_score": None,
                    },
                    extra={"price": price, "seed_launch_free": True},
                ))
    return records


def fb_run(
    seed: int,
    prize: float,
    price: float,
    ticks: int = 60,
    records: list[AttemptRecord] | None = None,
    starting: float = 100,
    pool: ReplayPool | None = None,
) -> list[dict]:
    if records is None:
        records = forcebench_pool(price)
    if pool is None:
        pool = ReplayPool(records, experiments_cost(price))
    cfg = MarketRun(
        run_id=f"forcebench_fixture_seed{seed}_prize{prize:g}_price{price:g}",
        seed=seed,
        ticks=ticks,
        worlds=WORLDS,
        agents=AGENTS,
        prizes={world: prize for world in WORLDS},
        starting_credits=starting,
        cost_model=pool,
        true_probs=None,
        initial_beliefs={
            agent: {world: 0.5 for world in WORLDS} for agent in AGENTS
        },
        belief_weight=2,
        track="forcebench_fixture",
        probability_source="replay:forcebench_fixture",
        outcome_source=pool,
        state_confidence=True,
    )
    events, _ = run_market(cfg)
    return [event.to_dict() for event in events]


@pytest.mark.parametrize(
    ("prize", "price", "seed"),
    [
        (prize, price, seed)
        for prize in [5, 10, 20, 40, 80]
        for price in [0.25, 0.5, 1, 2]
        for seed in range(5)
    ],
)
def test_conservation_grid(prize, price, seed):
    events = fb_run(seed, prize=prize, price=price)
    balances = compute_final_balances(events)

    assert abs(sum(balances.values())) < 1e-9
    assert abs(balances.get("escrow", 0.0)) < 1e-9

    funded = sum(
        event["amount"] for event in events if event["type"] == "account_funded"
    )
    participant_accounts = ["researcher", "lab", *(f"agent:{a}" for a in AGENTS)]
    assert sum(balances.get(account, 0.0) for account in participant_accounts) == pytest.approx(
        funded, abs=1e-9
    )

    charges = [event for event in events if event["type"] == "experiment_charged"]
    assert balances.get("lab", 0.0) == pytest.approx(
        sum(event["amount"] for event in charges), abs=1e-9
    )
    assert balances.get("lab", 0.0) == pytest.approx(
        price * sum(event["count"] for event in charges), abs=1e-9
    )


def test_replay_is_deterministic_for_repeated_and_shuffled_inputs():
    records = forcebench_pool(0.5)
    expected = fb_run(3, prize=20, price=0.5, records=records)
    assert fb_run(3, prize=20, price=0.5, records=records) == expected

    shuffled = list(records)
    random.Random(17).shuffle(shuffled)
    assert fb_run(3, prize=20, price=0.5, records=shuffled) == expected


def test_shared_pool_sweep_is_order_independent():
    shared_pool = ReplayPool(forcebench_pool(0.5), experiments_cost(0.5))
    configs = [(prize, seed) for prize in [5, 20, 80] for seed in range(5)]

    forward = {
        config: fb_run(config[1], prize=config[0], price=0.5, pool=shared_pool)
        for config in configs
    }
    reverse = {
        config: fb_run(config[1], prize=config[0], price=0.5, pool=shared_pool)
        for config in reversed(configs)
    }

    assert forward == reverse


def test_replay_hash_is_independent_of_pythonhashseed():
    module_path = str(Path(__file__).resolve())
    code = (
        "import hashlib, importlib.util, json, sys; "
        f"sys.path.insert(0, {str(REPO_ROOT)!r}); "
        "spec = importlib.util.spec_from_file_location("
        f"'forcebench_readiness_child', {module_path!r}); "
        "module = importlib.util.module_from_spec(spec); "
        "sys.modules[spec.name] = module; "
        "spec.loader.exec_module(module); "
        "events = module.fb_run(4, prize=20, price=0.5); "
        "print(hashlib.sha256(json.dumps(events, sort_keys=True).encode()).hexdigest())"
    )
    hashes = []
    for hash_seed in ("0", "123"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        hashes.append(result.stdout.strip())
    assert hashes[0] == hashes[1]


@pytest.mark.parametrize("seed", range(5))
def test_attempt_accounting_and_lifecycle(seed):
    records = forcebench_pool(0.5)
    by_id = {record.attempt_id: record for record in records}
    events = fb_run(seed, prize=500, price=0.5, ticks=200, records=records)

    started = [event for event in events if event["type"] == "attempt_started"]
    charged = [event for event in events if event["type"] == "experiment_charged"]
    verdicts = [event for event in events if event["type"] == "verdict_issued"]
    charges_by_id = Counter(event["attempt_id"] for event in charged)
    verdicts_by_id = Counter(event["attempt_id"] for event in verdicts)

    assert started
    assert len({event["attempt_id"] for event in started}) == len(started)
    assert set(charges_by_id) == {event["attempt_id"] for event in started}
    assert set(verdicts_by_id) == {event["attempt_id"] for event in started}
    assert all(count == 1 for count in charges_by_id.values())
    assert all(count == 1 for count in verdicts_by_id.values())
    for event in charged:
        record = by_id[event["attempt_id"]]
        assert event["count"] == record.experiments
        assert event["amount"] == pytest.approx(event["count"] * 0.5)
    for event in verdicts:
        record = by_id[event["attempt_id"]]
        assert event["detail"]["passed"] == record.passed

    passed_worlds = {
        event["world"] for event in events if event["type"] == "attempt_passed"
    }
    paid_worlds = [
        event["world"] for event in events if event["type"] == "prize_paid"
    ]
    assert len(paid_worlds) == len(passed_worlds)
    assert len(paid_worlds) == len(set(paid_worlds))
    assert set(paid_worlds) == passed_worlds
    assert all(
        event.get("world") in {"gravity", "yukawa"}
        for event in events
        if event.get("agent") == "llm_menu"
    )


def h3_table(events: list[dict], starting: float = 100) -> dict[str, dict]:
    """Calculate H3 outcomes using only serialized market events."""
    bidders = {
        event["agent"] for event in events if event["type"] == "bid_placed"
    }
    attempts = Counter(
        event["agent"]
        for event in events
        if event["type"] == "attempt_started"
    )
    passes = Counter(
        event["agent"]
        for event in events
        if event["type"] == "verdict_issued" and event["detail"]["passed"]
    )
    experiment_events = [
        event for event in events if event["type"] == "experiment_charged"
    ]
    experiments = Counter()
    spend = Counter()
    for event in experiment_events:
        experiments[event["agent"]] += event["count"]
        spend[event["agent"]] += event["amount"]
    prize_income = Counter(
        {agent: 0.0 for agent in bidders}
    )
    for event in events:
        if event["type"] == "prize_paid":
            prize_income[event["agent"]] += event["amount"]

    balances = compute_final_balances(events)
    return {
        agent: {
            "attempts": attempts[agent],
            "passes": passes[agent],
            "experiments": experiments[agent],
            "mean_experiments_per_attempt": (
                experiments[agent] / attempts[agent] if attempts[agent] else None
            ),
            "spend": spend[agent],
            "prize_income": prize_income[agent],
            "profit": balances.get(f"agent:{agent}", 0.0) - starting,
        }
        for agent in sorted(bidders)
    }


def test_h3_is_measurable_and_bayes_lite_outperforms_random_menu():
    pool = ReplayPool(forcebench_pool(1.0), experiments_cost(1.0))
    totals = defaultdict(
        lambda: {
            "attempts": 0,
            "passes": 0,
            "experiments": 0,
            "spend": 0.0,
            "prize_income": 0.0,
            "profit": 0.0,
        }
    )
    for seed in range(10):
        table = h3_table(fb_run(seed, prize=20, price=1.0, pool=pool))
        assert table
        for agent, stats in table.items():
            assert all(value is not None for value in stats.values())
            for field, value in stats.items():
                if field != "mean_experiments_per_attempt":
                    totals[agent][field] += value

    aggregated = {}
    for agent, stats in sorted(totals.items()):
        aggregated[agent] = {
            **stats,
            "mean_experiments_per_attempt": (
                stats["experiments"] / stats["attempts"] if stats["attempts"] else None
            ),
        }
    print("Aggregated H3 table (seeds 0–9, prize 20, price 1):")
    print(json.dumps(aggregated, sort_keys=True, indent=2))
    assert aggregated["bayes_lite"]["mean_experiments_per_attempt"] < aggregated[
        "random_menu"
    ]["mean_experiments_per_attempt"]
    assert aggregated["bayes_lite"]["profit"] > aggregated["random_menu"]["profit"]


def _single_record(
    *,
    attempt_id: str = "fb-single-gravity-0",
    solver: str = "bayes_lite",
    world: str = "gravity",
    seed: int = 0,
    experiments: int = 1,
    price: float = 1.0,
    passed: bool = False,
    normalised_mse: float = 0.5,
    stated_p: float = 0.7,
) -> AttemptRecord:
    return AttemptRecord(
        attempt_id=attempt_id,
        source="sim",
        protocol="forcebench_menu",
        venue="forcebench",
        world=world,
        solver=solver,
        seed=seed,
        stated_p_success=stated_p,
        rounds=experiments,
        experiments=experiments,
        lab_cost=experiments * price,
        verdict={
            "normalised_mse": normalised_mse,
            "passed": passed,
            "prereg_commitment": f"c-{world}-{seed}",
            "explanation_score": None,
        },
        extra={"price": price, "seed_launch_free": True},
    )


def test_zero_experiment_attempt_is_charged_and_conserved():
    events = fb_run(
        0, prize=20, price=1.0, records=[_single_record(experiments=0)]
    )
    charge = next(
        event for event in events if event["type"] == "experiment_charged"
    )
    assert charge["amount"] == 0.0
    assert charge["count"] == 0
    assert abs(sum(compute_final_balances(events).values())) < 1e-9


@pytest.mark.xfail(
    strict=True,
    reason="BUG: inf normalised_mse leaks into event log as non-standard JSON Infinity",
)
def test_infinite_score_events_are_strict_json():
    record = _single_record(normalised_mse=float("inf"))
    events = fb_run(0, prize=20, price=1.0, records=[record])
    assert any(event["type"] == "attempt_started" for event in events)
    json.dumps(events, allow_nan=False)


def _reject_json_constant(value: str):
    raise ValueError(f"invalid JSON constant: {value}")


@pytest.mark.xfail(
    strict=True,
    reason="BUG: inf normalised_mse leaks into event log as non-standard JSON Infinity",
)
def test_attempt_store_lines_are_strict_json(tmp_path):
    record = _single_record(normalised_mse=float("inf"))
    store = AttemptStore(tmp_path / "fb.jsonl")
    store.append([record])
    line = store.path.read_text()
    json.loads(line, parse_constant=_reject_json_constant)


@pytest.mark.xfail(
    strict=True, reason="BUG: ReplayPool accepts duplicate attempt_ids"
)
def test_replay_pool_rejects_duplicate_attempt_ids():
    first = _single_record()
    second = replace(first, solver="random_menu")
    with pytest.raises(ValueError):
        ReplayPool([first, second], recorded_cost())


@pytest.mark.xfail(
    strict=True,
    reason="BUG: ReplayPool keys by (solver, world) and mixes venues",
)
def test_replay_pool_does_not_mix_venues():
    forcebench = _single_record(attempt_id="forcebench-0")
    discoverphysics = replace(
        forcebench,
        attempt_id="discoverphysics-1",
        venue="discoverphysics",
        protocol="discoverphysics_native",
    )
    with pytest.raises(ValueError):
        ReplayPool([forcebench, discoverphysics], experiments_cost(1.0))


@pytest.mark.parametrize("stated_p", [1.5, -0.1])
@pytest.mark.xfail(strict=True, reason="BUG: AttemptRecord accepts invalid stated_p_success")
def test_attempt_record_rejects_out_of_range_stated_probability(stated_p):
    with pytest.raises(ValueError):
        _single_record(stated_p=stated_p)


@pytest.mark.xfail(
    strict=True, reason="BUG: ReplayPool ignores recorded price mismatch"
)
def test_replay_pool_rejects_experiment_price_mismatch():
    records = forcebench_pool(1.0)
    with pytest.raises(ValueError):
        ReplayPool(records, experiments_cost(0.25))


def test_recorded_cost_charges_each_drawn_records_lab_cost():
    records = forcebench_pool(0.5)
    by_id = {record.attempt_id: record for record in records}
    pool = ReplayPool(records, recorded_cost())
    events = fb_run(2, prize=500, price=0.5, ticks=200, records=records, pool=pool)

    charges = [event for event in events if event["type"] == "experiment_charged"]
    assert charges
    for event in charges:
        assert event["amount"] == by_id[event["attempt_id"]].lab_cost


def test_llm_pool_exhaustion_never_draws_more_than_five_gravity_records():
    events = fb_run(0, prize=500, price=0.5, ticks=200)
    gravity_attempts = [
        event
        for event in events
        if event["type"] == "attempt_started"
        and event["agent"] == "llm_menu"
        and event["world"] == "gravity"
    ]
    assert len(gravity_attempts) <= 5
    assert len({event["attempt_id"] for event in gravity_attempts}) == len(
        gravity_attempts
    )


def test_insufficient_credits_is_reported_once_and_never_charged():
    events = fb_run(0, prize=500, price=1.0, starting=3)
    notices = [
        event
        for event in events
        if event["type"] == "insufficient_credits"
        and event["agent"] == "random_menu"
    ]
    counts = Counter((event["agent"], event["world"]) for event in notices)
    assert all(count <= 1 for count in counts.values())
    assert not [
        event
        for event in events
        if event["type"] == "experiment_charged"
        and event["agent"] == "random_menu"
    ]


def test_fractional_price_conserves_credits():
    events = fb_run(0, prize=500, price=0.1, ticks=200)
    assert abs(sum(compute_final_balances(events).values())) < 1e-9


def test_attempt_store_corrections_supersede_without_changing_count(tmp_path):
    path = tmp_path / "fb.jsonl"
    store = AttemptStore(path)
    records = forcebench_pool(0.5)
    assert store.append(records) == len(records)
    original_count = len(store.load())

    original = next(
        record
        for record in records
        if record.venue == "forcebench"
        and record.world == "gravity"
        and record.solver == "bayes_lite"
        and record.seed == 0
    )
    corrected = replace(
        original,
        verdict={**original.verdict, "passed": not original.passed},
    )
    assert store.append([corrected]) == 1
    loaded = store.load()
    assert len(loaded) == original_count
    assert next(record for record in loaded if record.attempt_id == original.attempt_id) == corrected
    assert store.has("forcebench", "gravity", "bayes_lite", 0)
    assert not store.has("forcebench", "gravity", "bayes_lite", 99)


def oracle_imports(path: Path) -> list[str]:
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    findings = []

    def record(node: ast.AST):
        findings.append(
            f"{path}:{getattr(node, 'lineno', 1)}: "
            f"{ast.get_source_segment(source, node) or ast.dump(node)}"
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name == "dm.oracle" or alias.name.startswith("dm.oracle.")
                for alias in node.names
            ):
                record(node)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "dm.oracle" or (
                node.module and node.module.startswith("dm.oracle.")
            ):
                record(node)
            elif node.module == "dm" and any(
                alias.name == "oracle" for alias in node.names
            ):
                record(node)
        elif isinstance(node, ast.Call):
            func = node.func
            dynamic_import = (
                isinstance(func, ast.Name) and func.id == "__import__"
            ) or (
                isinstance(func, ast.Attribute)
                and func.attr == "import_module"
                and isinstance(func.value, ast.Name)
                and func.value.id == "importlib"
            )
            if dynamic_import and any(
                isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
                and "dm.oracle" in argument.value
                for argument in node.args
            ):
                record(node)
    return findings


def test_oracle_import_scanner_detects_import_forms(tmp_path):
    forms = [
        "import dm.oracle",
        "import dm.oracle.cases as cases",
        "from dm.oracle import evaluate",
        "from dm.oracle.cases import CASES",
        "from dm import oracle",
        'import importlib; importlib.import_module("dm.oracle.worker")',
        '__import__("prefix dm.oracle suffix")',
    ]
    for index, source in enumerate(forms):
        path = tmp_path / f"oracle_import_{index}.py"
        path.write_text(source)
        assert oracle_imports(path), source

    clean = tmp_path / "clean.py"
    clean.write_text("import dm.types\nfrom dm import outcomes\n")
    assert oracle_imports(clean) == []


def test_solvers_and_venues_do_not_import_oracle():
    directories = [
        directory
        for directory in (REPO_ROOT / "dm/solvers", REPO_ROOT / "dm/venues")
        if directory.exists()
    ]
    if not directories:
        pytest.skip("dm/solvers and dm/venues do not exist yet")
    violations = [
        finding
        for directory in directories
        for path in sorted(directory.rglob("*.py"))
        for finding in oracle_imports(path)
    ]
    assert not violations, "\n".join(violations)
