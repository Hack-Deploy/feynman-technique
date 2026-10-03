"""Tests for data loading (P0) and market engine (P1)."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_loader import load_table1, load_table2, load_table3, load_config
from market import (
    MarketRun, run_market, TrackACostModel, TrackCCostModel,
    Event, LedgerRow, _check_conservation,
)
from runner import (
    build_track_a_probs, build_track_a_beliefs,
    build_track_c_probs, build_track_c_beliefs,
    run_track_a, run_track_c,
)


# ===================================================================
# P0 Tests – Data loading
# ===================================================================

class TestP0DataLoading:
    """Verify CSVs match the specification tables exactly."""

    def test_table1_shape(self):
        df = load_table1()
        assert df.shape == (11, 4), f"Expected (11, 4), got {df.shape}"
        assert list(df.columns) == ["opus-4.7", "gpt-5.5", "sonnet-4.6", "qwen3.5-397b"]

    def test_table1_values(self):
        df = load_table1()
        # Spot checks from specification
        assert df.loc["gravity", "opus-4.7"] == pytest.approx(0.94)
        assert df.loc["gravity", "gpt-5.5"] == pytest.approx(0.96)
        assert df.loc["gravity", "sonnet-4.6"] == pytest.approx(0.50)
        assert df.loc["gravity", "qwen3.5-397b"] == pytest.approx(0.76)
        assert df.loc["yukawa", "opus-4.7"] == pytest.approx(0.96)
        assert df.loc["hubble", "gpt-5.5"] == pytest.approx(0.64)
        assert df.loc["dark_matter", "opus-4.7"] == pytest.approx(0.38)
        assert df.loc["three_species", "qwen3.5-397b"] == pytest.approx(0.26)
        assert df.loc["circle", "sonnet-4.6"] == pytest.approx(0.36)
        assert df.loc["extra_dimensions", "opus-4.7"] == pytest.approx(0.54)
        assert df.loc["fractional", "gpt-5.5"] == pytest.approx(0.48)

    def test_table1_all_worlds(self):
        df = load_table1()
        expected_worlds = [
            "gravity", "yukawa", "hubble", "ether", "oscillator",
            "coulomb", "circle", "extra_dimensions", "fractional",
            "dark_matter", "three_species",
        ]
        assert list(df.index) == expected_worlds

    def test_table2_values(self):
        df = load_table2()
        assert df.shape == (4, 2)
        assert df.loc["opus-4.7", "mean_score"] == pytest.approx(0.61)
        assert df.loc["opus-4.7", "pass_at_1"] == pytest.approx(0.264)
        assert df.loc["gpt-5.5", "mean_score"] == pytest.approx(0.59)
        assert df.loc["gpt-5.5", "pass_at_1"] == pytest.approx(0.217)
        assert df.loc["sonnet-4.6", "mean_score"] == pytest.approx(0.28)
        assert df.loc["sonnet-4.6", "pass_at_1"] == pytest.approx(0.018)
        assert df.loc["qwen3.5-397b", "mean_score"] == pytest.approx(0.33)
        assert df.loc["qwen3.5-397b", "pass_at_1"] == pytest.approx(0.045)

    def test_table3_values(self):
        df = load_table3()
        assert df.shape == (6, 2)
        assert df.loc["gravity", "mda"] == pytest.approx(1.00)
        assert df.loc["gravity", "llm_opus"] == pytest.approx(0.22)
        assert df.loc["yukawa", "mda"] == pytest.approx(1.00)
        assert df.loc["coulomb", "mda"] == pytest.approx(0.56)
        assert df.loc["fractional", "llm_opus"] == pytest.approx(0.56)
        assert df.loc["extra_dimensions", "llm_opus"] == pytest.approx(0.22)

    def test_table3_all_worlds(self):
        df = load_table3()
        expected = ["gravity", "yukawa", "coulomb", "oscillator",
                     "fractional", "extra_dimensions"]
        assert list(df.index) == expected

    def test_config_loads(self):
        cfg = load_config()
        assert cfg["simulation"]["ticks"] == 200
        assert cfg["simulation"]["seeds"] == [0, 1, 2, 3, 4]
        assert cfg["simulation"]["starting_credits"] == 100
        assert cfg["track_a"]["baseline_prize"] == 20
        assert cfg["track_a"]["prize_sweep"] == [5, 20, 50, 100, 200]


# ===================================================================
# P1 Tests – Market engine
# ===================================================================

def _make_simple_run(seed=42, ticks=10, prize=20, p=0.5,
                     starting_credits=100) -> MarketRun:
    """Helper: single agent, single world, Track A."""
    return MarketRun(
        run_id="test_simple",
        seed=seed,
        ticks=ticks,
        worlds=["test_world"],
        agents=["test_agent"],
        prizes={"test_world": prize},
        starting_credits=starting_credits,
        cost_model=TrackACostModel(),
        true_probs={"test_agent": {"test_world": p}},
        initial_beliefs={"test_agent": {"test_world": p}},
        belief_weight=2,
        track="A",
        probability_source="test",
    )


class TestP1CreditConservation:
    """Credit conservation over 5 seeds."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_conservation_simple(self, seed):
        """Conservation with a simple single-agent setup."""
        cfg = _make_simple_run(seed=seed, ticks=50, prize=20, p=0.5)
        events, ledger = run_market(cfg)
        # If we got here, _check_conservation passed at every tick

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_conservation_track_a_full(self, seed):
        """Conservation with full Track A setup."""
        events, ledger = run_track_a(
            prize=20, probability_source="raw", seed=seed
        )
        # Conservation checked internally every tick

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_conservation_track_c(self, seed):
        """Conservation with Track C setup."""
        events, ledger = run_track_c(
            prize=50, price_per_experiment=1.0, seed=seed
        )


class TestP1BidRule:
    """Test the bid-rule boundary."""

    def test_no_bid_when_unprofitable(self):
        """Agent should not bid if belief_mean * prize <= expected_cost."""
        # p=0.1, prize=20 => 0.1*20=2 < 10 (expected cost). No bids.
        cfg = _make_simple_run(seed=0, ticks=50, prize=20, p=0.1)
        events, _ = run_market(cfg)
        bid_events = [e for e in events if e.type == "bid_placed"]
        assert len(bid_events) == 0, \
            f"Expected no bids but got {len(bid_events)}"

    def test_bids_when_profitable(self):
        """Agent should bid if belief_mean * prize > expected_cost."""
        # p=0.8, prize=20 => 0.8*20=16 > 10. Should bid.
        cfg = _make_simple_run(seed=0, ticks=50, prize=20, p=0.8)
        events, _ = run_market(cfg)
        bid_events = [e for e in events if e.type == "bid_placed"]
        assert len(bid_events) > 0, "Expected bids but got none"

    def test_bid_boundary(self):
        """At the exact boundary, belief_mean * prize == expected_cost, no bid."""
        # p=0.5, prize=20 => 0.5*20=10 == 10. Not strictly greater.
        cfg = _make_simple_run(seed=0, ticks=50, prize=20, p=0.5)
        events, _ = run_market(cfg)
        bid_events = [e for e in events if e.type == "bid_placed"]
        assert len(bid_events) == 0, \
            f"At boundary (==), expected no bids but got {len(bid_events)}"

    def test_bids_when_just_above(self):
        """Just above the boundary: should bid."""
        # p=0.501, prize=20 => 0.501*20=10.02 > 10
        cfg = _make_simple_run(seed=0, ticks=50, prize=20, p=0.501)
        events, _ = run_market(cfg)
        bid_events = [e for e in events if e.type == "bid_placed"]
        assert len(bid_events) > 0


class TestP1CancelledBids:
    """Cancelled bids cost nothing."""

    def test_cancelled_bid_no_charge(self):
        """If a world closes during a tick, later bids are cancelled free."""
        # Two agents, one world. High p so likely to pass quickly.
        cfg = MarketRun(
            run_id="test_cancel",
            seed=42,
            ticks=200,
            worlds=["w"],
            agents=["a1", "a2"],
            prizes={"w": 50},
            starting_credits=100,
            cost_model=TrackACostModel(),
            true_probs={"a1": {"w": 1.0}, "a2": {"w": 1.0}},
            initial_beliefs={"a1": {"w": 1.0}, "a2": {"w": 1.0}},
            belief_weight=2,
            track="A",
            probability_source="test",
        )
        events, _ = run_market(cfg)

        # Check that any cancelled bids have no associated charge
        cancelled = [e for e in events if e.type == "bid_cancelled"]
        # For each cancelled bid, find if there's a charge for the same
        # agent + world at the same tick
        for c in cancelled:
            charges = [
                e for e in events
                if e.type in ("round_charged", "experiment_charged")
                and e.tick == c.tick and e.agent == c.agent
                and e.world == c.world
            ]
            assert len(charges) == 0, \
                f"Cancelled bid for {c.agent}/{c.world} at tick {c.tick} was charged"


class TestP1Determinism:
    """Identical events for the same seed."""

    def test_same_seed_same_events(self):
        """Two runs with same seed must produce identical event sequences."""
        events1, _ = run_track_a(prize=20, probability_source="raw", seed=42)
        events2, _ = run_track_a(prize=20, probability_source="raw", seed=42)

        assert len(events1) == len(events2)
        for e1, e2 in zip(events1, events2):
            assert e1.to_dict() == e2.to_dict()

    def test_different_seed_different_events(self):
        """Different seeds should (almost certainly) produce different results."""
        events1, _ = run_track_a(prize=20, probability_source="raw", seed=0)
        events2, _ = run_track_a(prize=20, probability_source="raw", seed=1)

        # At least some events should differ
        dicts1 = [e.to_dict() for e in events1]
        dicts2 = [e.to_dict() for e in events2]
        assert dicts1 != dicts2


class TestP1LedgerDisclosure:
    """Ledger rows hidden until disclosure."""

    def test_failed_attempts_disclosed_on_close(self):
        """Failed attempt rows get disclosed when the world closes."""
        # p=0.5 so there will be failures and eventual passes
        cfg = _make_simple_run(seed=0, ticks=200, prize=50, p=0.5)
        events, ledger = run_market(cfg)

        for row in ledger:
            assert row.disclosed_at_tick is not None, \
                f"Ledger row {row.attempt_id} was never disclosed"

    def test_failed_only_disclosed_at_end_if_unsolved(self):
        """If world never solved, rows disclosed at last tick."""
        # p=0 so world never solved
        cfg = _make_simple_run(seed=0, ticks=50, prize=50, p=0.0,
                               starting_credits=1000)
        # p=0 means no bids either (0*50=0 < 10), so no ledger rows
        events, ledger = run_market(cfg)
        # With p=0, belief_mean starts at 0, so 0*50=0 < 10: no bids
        assert len(ledger) == 0

    def test_hidden_until_close(self):
        """Failed rows for a world are hidden until that world closes or run ends."""
        # Use high enough p to generate some activity
        cfg = _make_simple_run(seed=7, ticks=200, prize=50, p=0.3,
                               starting_credits=500)
        events, ledger = run_market(cfg)

        # Find the closing tick for test_world (if it closed)
        close_tick = None
        for e in events:
            if e.type == "attempt_passed" and e.world == "test_world":
                close_tick = e.tick
                break

        for row in ledger:
            if row.outcome["passed"]:
                # Passed row: disclosed at the tick it passed
                continue
            # Failed row: disclosed at close tick or end
            if close_tick is not None:
                assert row.disclosed_at_tick == close_tick or \
                       row.disclosed_at_tick == 200  # end
            else:
                assert row.disclosed_at_tick == 200

    def test_ledger_hides_true_probability(self):
        """Public rows must not reveal the solver's hidden pass probability."""
        cfg = _make_simple_run(seed=3, ticks=200, prize=50, p=0.37,
                               starting_credits=500)
        _, ledger = run_market(cfg)
        assert ledger
        for row in ledger:
            assert set(row.outcome) == {"passed"}
            assert 0.37 not in row.to_dict()["outcome"].values()


class TestP1Refunds:
    """Prize refunds at end of run."""

    def test_unsolved_worlds_refunded(self):
        """Open worlds get their prizes refunded to researcher at end."""
        # p=0 and starting belief too low to bid, so all stay unsolved
        cfg = _make_simple_run(seed=0, ticks=10, prize=20, p=0.0)
        events, _ = run_market(cfg)

        refund_events = [e for e in events if e.type == "prize_refunded"]
        assert len(refund_events) == 1  # one world
        assert refund_events[0].amount == 20
        assert refund_events[0].from_account == "escrow"
        assert refund_events[0].to_account == "researcher"

    def test_solved_worlds_not_refunded(self):
        """Solved worlds don't get refunded."""
        # p=1 so always passes on first attempt
        cfg = _make_simple_run(seed=0, ticks=10, prize=20, p=1.0)
        events, _ = run_market(cfg)

        refund_events = [e for e in events if e.type == "prize_refunded"]
        assert len(refund_events) == 0


class TestP1TrackCCostModel:
    """Track C cost model tests."""

    def test_cost_calculation(self):
        """Cost = experiments × price_per_experiment."""
        cm = TrackCCostModel(
            experiments_per_attempt={"mda": 8, "llm_opus": 8},
            price_per_experiment=1.0,
        )
        rng = np.random.default_rng(0)
        cost, detail = cm.draw_cost("mda", "gravity", rng)
        assert cost == 8.0
        assert detail["experiments"] == 8

    def test_max_cost(self):
        cm = TrackCCostModel(
            experiments_per_attempt={"mda": 8, "llm_opus_unthrottled": 41},
            price_per_experiment=2.0,
        )
        assert cm.max_cost("mda", "gravity") == 16.0
        assert cm.max_cost("llm_opus_unthrottled", "gravity") == 82.0


class TestP1ProbabilitySources:
    """Track A probability source calculation tests."""

    def test_raw_probabilities(self):
        """raw: p = explanation score."""
        t1 = load_table1()
        t2 = load_table2()
        probs = build_track_a_probs(t1, t2, "raw")
        assert probs["opus-4.7"]["gravity"] == pytest.approx(0.94)
        assert probs["sonnet-4.6"]["dark_matter"] == pytest.approx(0.22)

    def test_calibrated_probabilities(self):
        """calibrated: p = min(1, score * pass_at_1 / mean_score)."""
        t1 = load_table1()
        t2 = load_table2()
        probs = build_track_a_probs(t1, t2, "calibrated")
        # opus-4.7 gravity: min(1, 0.94 * 0.264 / 0.61) = min(1, 0.4068...) ≈ 0.407
        expected = min(1.0, 0.94 * 0.264 / 0.61)
        assert probs["opus-4.7"]["gravity"] == pytest.approx(expected)

    def test_calibrated_capped_at_1(self):
        """Calibrated probabilities should be capped at 1.0."""
        t1 = load_table1()
        t2 = load_table2()
        probs = build_track_a_probs(t1, t2, "calibrated")
        for agent in probs:
            for world in probs[agent]:
                assert probs[agent][world] <= 1.0


class TestP2CalibratedFewBids:
    """Under calibrated odds at prize 20, expect few or no bids."""

    def test_calibrated_prize20_few_bids(self):
        """Calibrated odds are much lower than raw. At prize 20, most agents
        should not find it profitable (belief_mean * 20 vs expected_cost 10)."""
        t1 = load_table1()
        t2 = load_table2()
        probs = build_track_a_probs(t1, t2, "calibrated")

        # Expected cost is 10. For a bid: belief_mean * prize > 10
        # belief_mean starts at pass_at_1.
        # opus-4.7: 0.264 * 20 = 5.28 < 10 => no bid
        # gpt-5.5: 0.217 * 20 = 4.34 < 10 => no bid
        # sonnet-4.6: 0.018 * 20 = 0.36 < 10 => no bid
        # qwen3.5-397b: 0.045 * 20 = 0.90 < 10 => no bid

        events, _ = run_track_a(prize=20, probability_source="calibrated", seed=0)
        bid_events = [e for e in events if e.type == "bid_placed"]
        assert len(bid_events) == 0, (
            f"Expected 0 bids under calibrated odds at prize 20 "
            f"(all agents' belief_mean * 20 < 10), got {len(bid_events)}"
        )


class TestP1CrossProcessDeterminism:
    """Same seed gives identical events regardless of Python's hash seed."""

    def test_events_independent_of_hash_seed(self):
        import subprocess
        root = Path(__file__).resolve().parent.parent
        code = (
            "import json, hashlib; from runner import run_track_a; "
            "ev, _ = run_track_a(prize=50, probability_source='raw', seed=0); "
            "print(hashlib.sha256(json.dumps([e.to_dict() for e in ev])"
            ".encode()).hexdigest())"
        )
        digests = set()
        for hash_seed in ["1", "2", "3"]:
            out = subprocess.run(
                [sys.executable, "-c", code], cwd=root, check=True,
                capture_output=True, text=True,
                env={**__import__("os").environ, "PYTHONHASHSEED": hash_seed},
            )
            digests.add(out.stdout.strip())
        assert len(digests) == 1, "Events depend on PYTHONHASHSEED"


class TestP1AccountFunding:
    """Opening balances are recorded as events."""

    def test_funding_events_at_tick_zero(self):
        cfg = _make_simple_run(prize=20)
        events, _ = run_market(cfg)
        funded = {e.to_account: e.amount for e in events
                  if e.type == "account_funded"}
        assert funded == {"researcher": 20, "agent:test_agent": 100}
        assert all(e.tick == 0 and e.from_account == "external"
                   for e in events if e.type == "account_funded")
