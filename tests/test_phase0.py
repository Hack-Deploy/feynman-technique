"""Phase 0: leak fix, CostModel.expected_cost, vendor smoke test."""

import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import market
from market import TrackACostModel, TrackCCostModel
from runner import run_track_a, run_track_c


class TestLedgerLeak:
    """Ledger rows must not carry the hidden true probability."""

    @pytest.mark.parametrize("run", [
        lambda: run_track_a(prize=100, probability_source="raw", seed=0),
        lambda: run_track_c(prize=100, price_per_experiment=0.5, seed=0),
    ])
    def test_outcome_metric_is_null(self, run):
        _, ledger = run()
        assert ledger, "expected at least one attempt"
        assert all(r.outcome["metric"] is None for r in ledger)


class TestExpectedCost:

    def test_track_a(self):
        assert TrackACostModel().expected_cost("a", "w") == 10

    def test_track_c(self):
        cm = TrackCCostModel({"mda": 8, "x": 41}, price_per_experiment=0.5)
        assert cm.expected_cost("mda", "w") == 4
        assert cm.expected_cost("x", "w") == 20.5

    def test_engine_has_no_isinstance_dispatch(self):
        assert "isinstance" not in inspect.getsource(market.run_market)
        assert not hasattr(market, "_expected_cost")


class TestVendorSmoke:
    """DiscoverPhysics installs via uv and runs one experiment."""

    def test_gravity_one_experiment(self):
        from scienceagent.worlds import get_world

        world = get_world("gravity", engine="nbody", noise_std=0.075, noise_seed=0)
        assert {"executor", "mission", "true_law", "law_stub"} <= world.keys()
        out = world["executor"].run([{
            "p1": 1.0, "p2": 1.0, "pos2": [3.0, 0.0], "velocity2": [0.0, 0.0],
            "measurement_times": [0.5, 1.0],
        }])
        assert len(out) == 1
        assert len(out[0]["pos2"]) == 2
        # Attractive: the probe accelerates towards the source. Velocities are
        # never noised (positions are, with sigma larger than the 1 s displacement).
        assert out[0]["velocity2"][-1][0] < 0
