"""Adversarial regression coverage for the market engine and attempt store."""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import pytest

import market
from analysis import compute_final_balances
from dm.outcomes import (
    ReplayPool,
    experiments_cost,
    recorded_cost,
    rounds_cost,
)
from dm.store import FIXTURES_DIR, AttemptStore
from dm.types import AttemptRecord, Preregistration
from market import (
    BernoulliTable,
    MarketRun,
    TrackACostModel,
    TrackCCostModel,
    run_market,
)
from runner import run_track_a


def rec(i, solver="s", world="w", exps=4, rounds=2, passed=True, **kw):
    """Build a small replay record."""
    verdict = kw.pop(
        "verdict",
        {
            "passed": passed,
            "normalised_mse": 0.01 if passed else 1.0,
            "prereg_commitment": None,
        },
    )
    return AttemptRecord(
        attempt_id=f"r{i}",
        source="sim",
        protocol="fixture",
        venue=kw.pop("venue", "fixture"),
        world=world,
        solver=solver,
        seed=i,
        stated_p_success=kw.pop("p", 0.5),
        rounds=rounds,
        experiments=exps,
        lab_cost=kw.pop("lab_cost", float(exps)),
        verdict=verdict,
        **kw,
    )


def cfg(
    pool,
    worlds=("w",),
    agents=("s",),
    prize=50.0,
    credits=100.0,
    ticks=20,
    seed=0,
    cost_model=None,
    **kw,
):
    """Build a replay run in the style used by the phase-one tests."""
    worlds = list(worlds)
    agents = list(agents)
    return MarketRun(
        run_id="x",
        seed=seed,
        ticks=ticks,
        worlds=worlds,
        agents=agents,
        prizes={w: prize for w in worlds},
        starting_credits=credits,
        cost_model=cost_model or pool,
        true_probs=None,
        initial_beliefs={a: {w: 0.5 for w in worlds} for a in agents},
        belief_weight=2,
        track="t",
        probability_source="replay",
        outcome_source=pool,
        state_confidence=True,
        **kw,
    )


def _fixture_replay_run(seed, prize=60.0, cost="experiments", ticks=50):
    records = AttemptStore(FIXTURES_DIR / "replay_pool.jsonl").load()
    if cost == "rounds":
        pool = ReplayPool(records, rounds_cost(1.0), charge_event="round_charged")
    else:
        pool = ReplayPool(records, experiments_cost(0.5))
    worlds = ["gravity", "yukawa", "coulomb_easy"]
    agents = ["fast", "slow"]
    run_cfg = cfg(
        pool,
        worlds=worlds,
        agents=agents,
        prize=prize,
        credits=100,
        ticks=ticks,
        seed=seed,
    )
    events, ledger = run_market(run_cfg)
    return records, [event.to_dict() for event in events], [
        row.to_dict() for row in ledger
    ]


def _min_agent_balance(events):
    """Return the minimum agent balance observed while replaying event transfers."""
    balances = defaultdict(float)
    minimum = 0.0
    for event in events:
        if isinstance(event, dict):
            get = event.get
        else:
            get = lambda key, default=None: getattr(event, key, default)
        amount = get("amount")
        if amount is None:
            continue
        source = get("from_account", get("from"))
        target = get("to_account", get("to"))
        if source is not None:
            balances[source] -= amount
        if target is not None:
            balances[target] += amount
        for account, balance in balances.items():
            if account.startswith("agent:"):
                minimum = min(minimum, balance)
    return minimum


def _assert_tick_conservation(events, expected_total):
    """Recompute all account balances and check the total at every tick."""
    balances = defaultdict(float)
    by_tick = defaultdict(list)
    for event in events:
        by_tick[event["tick"]].append(event)
    for tick in sorted(by_tick):
        for event in by_tick[tick]:
            amount = event.get("amount")
            if amount is None:
                continue
            source = event.get("from", event.get("from_account"))
            target = event.get("to", event.get("to_account"))
            if source is not None and source != "external":
                balances[source] -= amount
            if target is not None and target != "external":
                balances[target] += amount
        assert sum(balances.values()) == pytest.approx(expected_total, abs=1e-6)


def _parallel_store_writer(path, tag, record_count, law_size):
    """Append records one at a time like independent store clients."""
    store = AttemptStore(Path(path))
    law = tag * law_size
    for index in range(record_count):
        store.append(
            [
                AttemptRecord(
                    attempt_id=f"{tag}-{index}",
                    source="sim",
                    protocol="fixture",
                    venue="fixture",
                    world="w",
                    solver=tag,
                    seed=index,
                    stated_p_success=None,
                    rounds=1,
                    experiments=1,
                    lab_cost=1.0,
                    submitted_law=law,
                )
            ]
        )


def _raise_json_constant(value):
    raise ValueError(f"invalid JSON constant: {value}")


class TestStrictBugRegressions:
    """Tests that describe correct behavior for known engine bugs."""

    @pytest.mark.xfail(strict=True, reason="BUG-E1: string verdicts are truthy")
    def test_string_false_verdict_cannot_win(self):
        """Reject a string-valued false verdict or refuse its prize."""
        try:
            pool = ReplayPool(
                [rec(0, verdict={"passed": "False", "normalised_mse": 5.0})],
                experiments_cost(1),
            )
        except (ValueError, TypeError):
            return
        events, _ = run_market(cfg(pool))
        assert not any(event.type == "prize_paid" for event in events)

    @pytest.mark.xfail(strict=True, reason="BUG-E2: missing passed defaults to false")
    def test_missing_passed_verdict_is_rejected(self):
        """Require every replay verdict to include a passed key."""
        with pytest.raises((ValueError, KeyError)):
            ReplayPool(
                [rec(0, verdict={"normalised_mse": 0.01})],
                experiments_cost(1),
            )

    @pytest.mark.xfail(strict=True, reason="BUG-E3: replay commitment is unchecked")
    def test_mismatched_prereg_commitment_cannot_win(self):
        """Refuse to pay a replay verdict scored against another preregistration."""
        prereg = Preregistration(
            question_id="fixture/w",
            venue="fixture",
            world="w",
            test_seed=1,
            test_cases=[{"a": 1}],
            norm_variance=1.0,
            oracle_version="v",
        )
        mismatched = replace(prereg, test_seed=2)
        pool = ReplayPool(
            [
                rec(
                    0,
                    verdict={
                        "passed": True,
                        "normalised_mse": 0.01,
                        "prereg_commitment": mismatched.commitment(),
                    },
                )
            ],
            experiments_cost(1),
        )
        try:
            events, _ = run_market(cfg(pool, preregs={"w": prereg}))
        except ValueError:
            return
        assert not any(event.type == "prize_paid" for event in events)

    @pytest.mark.xfail(strict=True, reason="BUG-E4: replay pool key drops venue")
    def test_replay_pool_separates_venues(self):
        """Keep records from different venues in separate pool buckets."""
        try:
            pool = ReplayPool(
                [
                    rec(0, venue="forcebench"),
                    rec(1, venue="discoverphysics"),
                ],
                experiments_cost(1),
            )
        except ValueError:
            return
        assert len(pool.pool) == 2

    @pytest.mark.xfail(strict=True, reason="BUG-E5: replay cost can overdraw agents")
    def test_replay_charge_never_makes_agent_balance_negative(self):
        """Prevent replay charges from exceeding the agent's available credits."""
        pool = ReplayPool(
            [rec(i, exps=90, passed=False) for i in range(3)],
            experiments_cost(1),
        )
        try:
            events, _ = run_market(
                cfg(
                    pool,
                    credits=100,
                    prize=500,
                    cost_model=TrackCCostModel({"s": 1}, 1.0),
                )
            )
        except ValueError:
            return
        assert _min_agent_balance(events) >= 0

    @pytest.mark.xfail(strict=True, reason="BUG-E6: rounds charges omit count")
    def test_replay_charge_event_always_has_integer_count(self):
        """Emit the experiment count even with the default charge event."""
        record = rec(0)
        pool = ReplayPool([record], rounds_cost(1))
        events, _ = run_market(cfg(pool))
        charged = [
            event
            for event in events
            if event.type in {"experiment_charged", "round_charged"}
        ]
        assert charged
        assert all(type(event.count) is int for event in charged)
        for event in charged:
            expected_count = (
                record.rounds
                if event.type == "round_charged"
                else record.experiments
            )
            assert event.count == expected_count

    @pytest.mark.parametrize(
        ("kwargs", "cost_fn"),
        [
            ({"exps": -10}, experiments_cost(1)),
            ({"rounds": -10}, rounds_cost(1)),
            ({"lab_cost": -10.0}, recorded_cost()),
        ],
    )
    @pytest.mark.xfail(strict=True, reason="BUG-E7: replay accepts negative record costs")
    def test_replay_rejects_negative_record_counts_and_costs(self, kwargs, cost_fn):
        """Reject negative experiments, rounds, or lab cost at pool creation."""
        with pytest.raises(ValueError):
            ReplayPool([rec(0, **kwargs)], cost_fn)

    @pytest.mark.xfail(strict=True, reason="BUG-E8: floating-point conservation false alarm")
    def test_large_fractional_run_completes_when_credits_are_conserved(self):
        """Allow a conserved large-credit simulation to finish despite float drift."""
        pool = ReplayPool(
            [rec(i, exps=3, passed=False) for i in range(50)],
            experiments_cost(0.1),
        )
        run_market(cfg(pool, prize=1e7, credits=1e9, ticks=60))

    @pytest.mark.xfail(strict=True, reason="BUG-E9: preregistration cases are mutable")
    def test_prereg_commitment_cannot_change_through_test_cases(self):
        """Keep a frozen preregistration commitment stable after construction."""
        prereg = Preregistration(
            question_id="fixture/w",
            venue="fixture",
            world="w",
            test_seed=1,
            test_cases=[{"a": 1}],
            norm_variance=1.0,
            oracle_version="v",
        )
        before = prereg.commitment()
        try:
            prereg.test_cases.append({"b": 2})
        except (AttributeError, TypeError):
            return
        assert prereg.commitment() == before

    @pytest.mark.xfail(strict=True, reason="BUG-E10: dropped mid-tick bid is silent")
    def test_mid_tick_dropped_bid_emits_an_affordability_event(self):
        """Report when an earlier bid makes another world unaffordable."""
        pool = ReplayPool(
            [
                rec(0, world="a", exps=60, passed=False),
                rec(1, world="b", exps=60, passed=False),
            ],
            experiments_cost(1),
        )
        events, _ = run_market(
            cfg(pool, worlds=("a", "b"), prize=500, credits=100, ticks=1)
        )
        assert any(
            event.type in {"bid_cancelled", "insufficient_credits"}
            and event.agent == "s"
            and event.world == "b"
            for event in events
        )

    @pytest.mark.parametrize("duplicate", ["agents", "worlds"])
    @pytest.mark.xfail(strict=True, reason="BUG-E11: duplicate market entities are accepted")
    def test_duplicate_agents_or_worlds_are_rejected(self, duplicate):
        """Reject repeated agent or world identifiers before market execution."""
        pool = ReplayPool([rec(0, passed=False)], experiments_cost(1))
        run_cfg = (
            cfg(pool, agents=("s", "s"))
            if duplicate == "agents"
            else cfg(pool, worlds=("w", "w"))
        )
        with pytest.raises(ValueError):
            run_market(run_cfg)

    @pytest.mark.xfail(strict=True, reason="BUG-E12: truncated final JSONL line aborts load")
    def test_store_ignores_a_truncated_final_line(self, tmp_path):
        """Load valid records even if the final JSONL append was truncated."""
        path = tmp_path / "attempts.jsonl"
        store = AttemptStore(path)
        store.append([rec(0)])
        with path.open("ab") as output:
            output.write(b'{"attempt_id": "t-1", "sour')
        assert len(store.load()) == 1

    @pytest.mark.xfail(strict=True, reason="BUG-E13: unknown record fields are dropped")
    def test_store_preserves_unknown_fields_on_round_trip(self, tmp_path):
        """Preserve unrecognized record fields when a loaded record is appended."""
        path = tmp_path / "attempts.jsonl"
        record = rec(0).to_dict()
        record["question_id"] = "discoverphysics/gravity"
        path.write_text(json.dumps(record) + "\n")
        store = AttemptStore(path)
        store.append(store.load())
        data = json.loads(path.read_text().splitlines()[-1])
        assert (
            data.get("question_id") == "discoverphysics/gravity"
            or data.get("extra", {}).get("_unknown", {}).get("question_id")
            == "discoverphysics/gravity"
        )

    @pytest.mark.xfail(strict=True, reason="BUG-E14: store writes NaN as invalid JSON")
    def test_store_rejects_nan_or_writes_strict_json(self, tmp_path):
        """Reject NaN record values or guarantee strict JSON output."""
        path = tmp_path / "attempts.jsonl"
        try:
            record = rec(0, lab_cost=math.nan)
            AttemptStore(path).append([record])
        except ValueError:
            return
        line = path.read_text().splitlines()[-1]
        json.loads(line, parse_constant=_raise_json_constant)

    @pytest.mark.xfail(strict=True, reason="BUG-E15: record parser accepts string counts")
    def test_record_parser_rejects_string_experiments(self):
        """Require experiments to remain an integer when parsing a record."""
        data = rec(0).to_dict()
        data["experiments"] = "4"
        try:
            parsed = AttemptRecord.from_dict(data)
        except (TypeError, ValueError):
            return
        assert type(parsed.experiments) is int


class TestPassingRegressions:
    """Tests for engine behavior that should pass already."""

    def test_empty_pool_has_no_pair_events_and_refunds_unused_funds(self):
        """Do not emit pair activity for empty pools and preserve agent funding."""
        pool = ReplayPool([], experiments_cost(1))
        run_cfg = cfg(pool, worlds=("w", "empty"), prize=20, credits=100)
        events, _ = run_market(run_cfg)
        assert not [
            event
            for event in events
            if event.agent == "s" and event.world in {"w", "empty"}
        ]
        balances = compute_final_balances([event.to_dict() for event in events])
        assert balances["agent:s"] == 100

    def test_one_failing_record_is_attempted_once(self):
        """Stop bidding after a single-record pool has been exhausted."""
        pool = ReplayPool([rec(0, passed=False)], experiments_cost(0.5))
        events, _ = run_market(cfg(pool, prize=500, ticks=200))
        pair_events = [
            event
            for event in events
            if event.agent == "s" and event.world == "w"
        ]
        assert sum(event.type == "attempt_started" for event in pair_events) == 1
        started = next(
            index for index, event in enumerate(pair_events)
            if event.type == "attempt_started"
        )
        assert not any(
            event.type == "insufficient_credits"
            for event in pair_events[started + 1 :]
        )

    def test_exhausted_pool_has_no_pair_activity_after_last_attempt(self):
        """Do not emit further solver-world events after the last replay."""
        pool = ReplayPool([rec(i, passed=False) for i in range(3)], experiments_cost(1))
        events, _ = run_market(cfg(pool, prize=500, ticks=30))
        assert sum(
            event.type == "attempt_started"
            and event.agent == "s"
            and event.world == "w"
            for event in events
        ) == 3
        last_completion = max(
            index
            for index, event in enumerate(events)
            if event.type == "estimate_updated"
            and event.agent == "s"
            and event.world == "w"
        )
        assert not any(
            event.agent == "s" and event.world == "w"
            for event in events[last_completion + 1 :]
        )

    def test_solver_with_records_on_one_world_never_bids_on_another(self):
        """Keep a solver off worlds absent from its replay pool."""
        pool = ReplayPool([rec(0, world="gravity")], experiments_cost(1))
        events, _ = run_market(
            cfg(pool, worlds=("gravity", "yukawa"), prize=500)
        )
        assert not any(
            event.agent == "s" and event.world == "yukawa" for event in events
        )

    @pytest.mark.parametrize("seed", [0, 2, 4])
    def test_each_replay_bid_has_a_complete_real_attempt_lifecycle(self, seed):
        """Check event order and record identity for every fixture replay bid."""
        records, events, _ = _fixture_replay_run(seed, prize=500, ticks=200)
        by_id = {record.attempt_id: record for record in records}
        expected = [
            "attempt_started",
            "confidence_stated",
            "experiment_charged",
            "attempt_submitted",
            "verdict_issued",
        ]
        for index, event in enumerate(events):
            if event["type"] != "bid_placed":
                continue
            lifecycle = events[index + 1 : index + 7]
            assert [item["type"] for item in lifecycle[:5]] == expected
            assert lifecycle[5]["type"] in {"attempt_passed", "attempt_failed"}
            attempt_id = lifecycle[0]["attempt_id"]
            assert attempt_id in by_id
            assert all(
                item["attempt_id"] == attempt_id for item in lifecycle[:5]
            )
            assert "attempt_id" not in lifecycle[5]
            record = by_id[attempt_id]
            charge = lifecycle[2]
            assert charge["count"] == record.experiments
            assert charge["amount"] == pytest.approx(record.experiments * 0.5)

    def test_bernoulli_real_mode_states_initial_belief_and_charges_rounds(self):
        """State initial Bernoulli confidence and preserve round-charge semantics."""
        cost_model = TrackACostModel(
            cost_per_round=1, rounds_min=4, rounds_max=4
        )
        run_cfg = MarketRun(
            run_id="bernoulli-confidence",
            seed=3,
            ticks=4,
            worlds=["w"],
            agents=["s"],
            prizes={"w": 500},
            starting_credits=1000,
            cost_model=cost_model,
            true_probs={"s": {"w": 0.2}},
            initial_beliefs={"s": {"w": 0.7}},
            belief_weight=2,
            track="A",
            probability_source="raw",
            state_confidence=True,
        )
        events, _ = run_market(run_cfg)
        stated = [
            event for event in events if event.type == "confidence_stated"
        ]
        assert stated
        assert stated[0].p == pytest.approx(0.7)
        assert all(event.attempt_id is None for event in stated)
        assert any(event.type == "round_charged" for event in events)
        assert not any(event.type == "experiment_charged" for event in events)

    @pytest.mark.parametrize(
        ("credits", "prize", "expected_insufficient"),
        [(10, 500, 1), (10, 1, 0)],
    )
    def test_insufficient_credits_is_reported_once_only_when_bidding(
        self, credits, prize, expected_insufficient
    ):
        """Emit one affordability warning only when the solver wants to bid."""
        pool = ReplayPool([rec(0, exps=50, passed=False)], experiments_cost(1))
        events, _ = run_market(
            cfg(pool, credits=credits, prize=prize, ticks=50)
        )
        assert sum(
            event.type == "insufficient_credits"
            and event.agent == "s"
            and event.world == "w"
            for event in events
        ) == expected_insufficient
        assert not any(event.type == "bid_placed" for event in events)

    def test_missing_optional_verdict_fields_remain_supported(self):
        """Run replay records without optional metric and commitment fields."""
        pool = ReplayPool(
            [rec(0, verdict={"passed": True})],
            experiments_cost(1),
        )
        events, ledger = run_market(cfg(pool))
        verdict = next(event for event in events if event.type == "verdict_issued")
        assert verdict.detail["normalised_mse"] is None
        assert ledger[0].outcome["metric"] is None

    def test_infinite_cost_is_not_bid_or_charged(self):
        """Treat infinite expected costs as unaffordable without raising."""
        record = rec(0)
        cost_fn = lambda _: (math.inf, {"experiments": 4, "count": 4})
        pool = ReplayPool([record], cost_fn)
        events, _ = run_market(cfg(pool, prize=500))
        assert not any(event.type == "bid_placed" for event in events)
        assert not any("charged" in event.type for event in events)


class TestDeterminism:
    """Tests for stable replay outputs and seed use."""

    def test_shuffled_replay_records_have_identical_outputs(self):
        """Make replay results independent of source record order."""
        records = AttemptStore(FIXTURES_DIR / "replay_pool.jsonl").load()
        first = ReplayPool(records, experiments_cost(0.5))
        second = ReplayPool(list(reversed(records)), experiments_cost(0.5))
        first_events, first_ledger = run_market(
            cfg(first, worlds=("gravity", "yukawa", "coulomb_easy"),
                agents=("fast", "slow"), prize=500, ticks=200, seed=2)
        )
        second_events, second_ledger = run_market(
            cfg(second, worlds=("gravity", "yukawa", "coulomb_easy"),
                agents=("fast", "slow"), prize=500, ticks=200, seed=2)
        )
        assert [event.to_dict() for event in first_events] == [
            event.to_dict() for event in second_events
        ]
        assert [row.to_dict() for row in first_ledger] == [
            row.to_dict() for row in second_ledger
        ]

    def test_fixture_sweep_is_independent_of_python_hash_seed(self):
        """Match event-and-ledger digests across fresh seeded subprocesses."""
        root = Path(__file__).resolve().parents[2]
        code = """
import hashlib, json
from dm.outcomes import ReplayPool, experiments_cost
from dm.store import AttemptStore, FIXTURES_DIR
from market import MarketRun, run_market
digests = []
records = AttemptStore(FIXTURES_DIR / "replay_pool.jsonl").load()
for seed in range(5):
    pool = ReplayPool(records, experiments_cost(0.5))
    worlds = ["gravity", "yukawa", "coulomb_easy"]
    agents = ["fast", "slow"]
    cfg = MarketRun(
        run_id=f"replay_fixture_seed{seed}", seed=seed, ticks=200,
        worlds=worlds, agents=agents, prizes={w: 500 for w in worlds},
        starting_credits=100, cost_model=pool, true_probs=None,
        initial_beliefs={a: {w: 0.5 for w in worlds} for a in agents},
        belief_weight=2, track="replay_fixture", probability_source="replay",
        outcome_source=pool, state_confidence=True,
    )
    events, ledger = run_market(cfg)
    payload = json.dumps(
        [[e.to_dict() for e in events], [r.to_dict() for r in ledger]],
        sort_keys=True,
    )
    digests.append(hashlib.sha256(payload.encode()).hexdigest())
print(json.dumps(digests))
"""
        digests = []
        for hash_seed in ("0", "1", "2"):
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONHASHSEED": hash_seed},
            )
            digests.append(json.loads(result.stdout))
        assert digests[0] == digests[1] == digests[2]

    def test_attempt_order_changes_with_seed(self):
        """Exercise the seeded permutation by comparing ten replay orderings."""
        orders = {
            tuple(
                event["attempt_id"]
                for event in _fixture_replay_run(seed, prize=500, ticks=200)[1]
                if event["type"] == "attempt_started"
            )
            for seed in range(10)
        }
        assert len(orders) > 1

    def test_bernoulli_rng_draw_order_affects_market_events(self, monkeypatch):
        """Prove that swapping cost and outcome RNG draws changes event output."""
        baseline, _ = run_track_a(
            prize=50, probability_source="raw", seed=0
        )

        def draw_pass_before_cost(self, agent, world, rng):
            passed = bool(rng.random() < self.true_probs[agent].get(world, 0.0))
            cost, detail = self.cost_model.draw_cost(agent, world, rng)
            return passed, {**detail, "credits": cost}, None

        monkeypatch.setattr(market.BernoulliTable, "draw", draw_pass_before_cost)
        reordered, _ = run_track_a(
            prize=50, probability_source="raw", seed=0
        )
        assert [event.to_dict() for event in baseline] != [
            event.to_dict() for event in reordered
        ]


class TestAttemptStore:
    """Tests for append-only store behavior."""

    def test_superseding_record_keeps_its_first_seen_position(self, tmp_path):
        """Load the latest same-id version in the original record position."""
        store = AttemptStore(tmp_path / "attempts.jsonl")
        store.append(
            [
                rec(0, verdict={"passed": False, "version": 1}),
                rec(1, solver="other"),
                rec(0, verdict={"passed": True, "version": 2}),
            ]
        )
        loaded = store.load()
        assert [record.attempt_id for record in loaded] == ["r0", "r1"]
        assert loaded[0].verdict["version"] == 2

    def test_append_does_not_rewrite_existing_bytes(self, tmp_path):
        """Preserve the exact existing JSONL byte prefix on append."""
        path = tmp_path / "attempts.jsonl"
        store = AttemptStore(path)
        store.append([rec(0)])
        prefix = path.read_bytes()
        store.append([rec(1)])
        assert path.read_bytes().startswith(prefix)

    def test_concurrent_large_appends_produce_parseable_records(self, tmp_path):
        """Keep 400 concurrent 20 KB appends as intact JSONL records."""
        path = tmp_path / "concurrent.jsonl"
        context = multiprocessing.get_context("fork")
        processes = [
            context.Process(
                target=_parallel_store_writer,
                args=(str(path), tag, 200, 20_000),
            )
            for tag in ("a", "b")
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
        assert [process.exitcode for process in processes] == [0, 0]
        lines = path.read_text().splitlines()
        assert len(lines) == 400
        parsed = [json.loads(line) for line in lines]
        assert all(len(record["submitted_law"]) == 20_000 for record in parsed)
        assert len(AttemptStore(path).load()) == 400


class TestHypothesisProperties:
    """Bounded generative properties for replay and Bernoulli market runs."""

    def test_random_replay_pool_properties(self):
        """Check conservation, affordability, replay identity, and determinism."""
        hypothesis = pytest.importorskip("hypothesis")
        from hypothesis import given, settings, strategies as st

        record_spec = st.tuples(
            st.integers(min_value=0, max_value=30),
            st.integers(min_value=0, max_value=16),
            st.booleans(),
            st.one_of(st.none(), st.floats(min_value=0, max_value=1)),
        )
        grid_strategy = st.lists(
            st.lists(
                st.lists(record_spec, min_size=0, max_size=4),
                min_size=1,
                max_size=3,
            ),
            min_size=1,
            max_size=3,
        )

        @settings(max_examples=150, deadline=None)
        @given(
            grid=grid_strategy,
            prize=st.integers(min_value=0, max_value=300),
            price=st.floats(min_value=0.05, max_value=3, allow_nan=False),
            credits=st.integers(min_value=0, max_value=200),
            ticks=st.integers(min_value=1, max_value=40),
            seed=st.integers(min_value=0, max_value=2**16),
            cost_kind=st.sampled_from(("experiments", "rounds")),
        )
        def check(
            grid, prize, price, credits, ticks, seed, cost_kind
        ):
            """Validate one generated replay market and its event log."""
            agents = [f"s{i}" for i in range(len(grid))]
            worlds = [f"w{i}" for i in range(max(map(len, grid)))]
            records = []
            counter = 0
            for solver_index, solver_worlds in enumerate(grid):
                for world_index, specs in enumerate(solver_worlds):
                    for exps, rounds, passed, stated_p in specs:
                        records.append(
                            rec(
                                counter,
                                solver=agents[solver_index],
                                world=worlds[world_index],
                                exps=exps,
                                rounds=rounds,
                                passed=passed,
                                p=stated_p,
                            )
                        )
                        counter += 1
            if cost_kind == "experiments":
                cost_fn = experiments_cost(price)
                charge_event = "experiment_charged"
            else:
                cost_fn = rounds_cost(price)
                charge_event = "round_charged"
            pool = ReplayPool(records, cost_fn, charge_event=charge_event)
            run_cfg = cfg(
                pool,
                worlds=worlds,
                agents=agents,
                prize=prize,
                credits=credits,
                ticks=ticks,
                seed=seed,
            )
            events, ledger = run_market(run_cfg)
            repeated_events, _ = run_market(run_cfg)
            event_dicts = [event.to_dict() for event in events]
            assert event_dicts == [
                event.to_dict() for event in repeated_events
            ]
            assert _min_agent_balance(event_dicts) >= -1e-9
            _assert_tick_conservation(
                event_dicts, len(agents) * credits + len(worlds) * prize
            )

            by_id = {record.attempt_id: record for record in records}
            starts = [
                event for event in event_dicts
                if event["type"] == "attempt_started"
            ]
            started_ids = [event["attempt_id"] for event in starts]
            assert len(started_ids) == len(set(started_ids))
            charges = [
                event for event in event_dicts
                if event["type"] == charge_event
            ]
            assert all(
                event["amount"] == pytest.approx(
                    cost_fn(by_id[event["attempt_id"]])[0]
                )
                for event in charges
            )
            prizes_paid = [
                event for event in event_dicts
                if event["type"] == "prize_paid"
            ]
            assert len({event["world"] for event in prizes_paid}) == len(
                prizes_paid
            )
            for event in event_dicts:
                if event["type"] == "verdict_issued":
                    assert event["detail"]["passed"] == by_id[
                        event["attempt_id"]
                    ].passed
            insufficient = [
                (event["agent"], event["world"])
                for event in event_dicts
                if event["type"] == "insufficient_credits"
            ]
            assert len(insufficient) == len(set(insufficient))

        check()

    def test_random_bernoulli_market_properties(self):
        """Check bounded Track A/C conservation, affordability, and determinism."""
        hypothesis = pytest.importorskip("hypothesis")
        from hypothesis import given, settings, strategies as st

        probability_grid = st.lists(
            st.lists(
                st.floats(min_value=0, max_value=1, allow_nan=False),
                min_size=1,
                max_size=3,
            ),
            min_size=1,
            max_size=3,
        )
        numeric = st.floats(min_value=0.05, max_value=3, allow_nan=False)

        @settings(max_examples=150, deadline=None)
        @given(
            beliefs=probability_grid,
            true_probs=probability_grid,
            prize=st.integers(min_value=0, max_value=300),
            credits=st.integers(min_value=0, max_value=200),
            ticks=st.integers(min_value=1, max_value=40),
            seed=st.integers(min_value=0, max_value=2**16),
            price=numeric,
            is_track_c=st.booleans(),
            state_confidence=st.booleans(),
        )
        def check(
            beliefs,
            true_probs,
            prize,
            credits,
            ticks,
            seed,
            price,
            is_track_c,
            state_confidence,
        ):
            """Validate one generated Bernoulli market and its event log."""
            agents = [f"s{i}" for i in range(len(beliefs))]
            world_count = max(
                max(map(len, beliefs)), max(map(len, true_probs))
            )
            worlds = [f"w{i}" for i in range(world_count)]
            belief_map = {
                agent: {
                    world: beliefs[agent_index][world_index]
                    if world_index < len(beliefs[agent_index])
                    else 0.5
                    for world_index, world in enumerate(worlds)
                }
                for agent_index, agent in enumerate(agents)
            }
            true_map = {
                agent: {
                    world: true_probs[agent_index][world_index]
                    if agent_index < len(true_probs)
                    and world_index < len(true_probs[agent_index])
                    else 0.0
                    for world_index, world in enumerate(worlds)
                }
                for agent_index, agent in enumerate(agents)
            }
            if is_track_c:
                cost_model = TrackCCostModel(
                    {
                        agent: (seed + index * 7) % 50 + 1
                        for index, agent in enumerate(agents)
                    },
                    price,
                )
                track = "C"
            else:
                cost_model = TrackACostModel(
                    cost_per_round=price,
                    rounds_min=4,
                    rounds_max=16,
                )
                track = "A"
            run_cfg = MarketRun(
                run_id="hypothesis-bernoulli",
                seed=seed,
                ticks=ticks,
                worlds=worlds,
                agents=agents,
                prizes={world: prize for world in worlds},
                starting_credits=credits,
                cost_model=cost_model,
                true_probs=true_map,
                initial_beliefs=belief_map,
                belief_weight=2,
                track=track,
                probability_source="property",
                state_confidence=state_confidence,
            )
            events, _ = run_market(run_cfg)
            repeated_events, _ = run_market(run_cfg)
            event_dicts = [event.to_dict() for event in events]
            assert event_dicts == [
                event.to_dict() for event in repeated_events
            ]
            assert _min_agent_balance(event_dicts) >= -1e-9
            _assert_tick_conservation(
                event_dicts, len(agents) * credits + len(worlds) * prize
            )

        check()
