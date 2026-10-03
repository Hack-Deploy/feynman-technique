"""Tests for analysis module (P4) – every number is recomputed from the event log."""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import (
    compute_final_balances, compute_balances_over_time,
    find_worlds_solved, compute_run_summary, group_events_by_run,
    group_ledger_by_run, compute_clearing_prizes, compute_agent_profits,
    build_full_summary,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def _ensure_outputs():
    """output/ is gitignored; regenerate it on a fresh clone."""
    if not (OUTPUT_DIR / "summary.json").exists():
        import run
        run.main()


def _load_events():
    with open(OUTPUT_DIR / "events.json") as f:
        return json.load(f)


def _load_ledger():
    with open(OUTPUT_DIR / "ledger.json") as f:
        return json.load(f)


def _load_summary():
    with open(OUTPUT_DIR / "summary.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def all_events():
    _ensure_outputs()
    return _load_events()


@pytest.fixture(scope="module")
def all_ledger():
    _ensure_outputs()
    return _load_ledger()


@pytest.fixture(scope="module")
def summary():
    _ensure_outputs()
    return _load_summary()


class TestP4BalancesFromEvents:
    """Verify final balances can be recomputed from events alone."""

    def test_recompute_all_runs(self, all_events, summary):
        """Every run's final balances match the summary."""
        by_run = group_events_by_run(all_events)
        for run_id, events in by_run.items():
            recomputed = compute_final_balances(events)
            stored = summary["run_summaries"].get(run_id, {}).get("final_balances", {})
            for acct in set(list(recomputed.keys()) + list(stored.keys())):
                assert recomputed.get(acct, 0) == pytest.approx(stored.get(acct, 0), abs=1e-9), \
                    f"Balance mismatch for {acct} in {run_id}"

    def test_credit_conservation_all_runs(self, all_events):
        """Total credits = sum of initial credits in every run.

        Since events only record movements, the sum of all movements must
        net to zero for each account type (credits are conserved).
        """
        by_run = group_events_by_run(all_events)
        for run_id, events in by_run.items():
            balances = compute_final_balances(events)
            total = sum(balances.values())
            assert abs(total) < 1e-9, \
                f"Conservation violated in {run_id}: total movements={total}"


class TestP4WorldsSolved:
    """Verify world solution data matches events."""

    def test_solved_worlds_match_summary(self, all_events, summary):
        """Solved worlds recomputed from events match summary."""
        by_run = group_events_by_run(all_events)
        for run_id, events in by_run.items():
            recomputed = find_worlds_solved(events)
            stored = summary["run_summaries"].get(run_id, {}).get("worlds_solved", {})
            assert set(recomputed.keys()) == set(stored.keys()), \
                f"Solved worlds mismatch in {run_id}"


class TestP4ClearingPrizes:
    """Verify clearing prizes are recomputed from events."""

    def test_h2_raw_clearing(self, all_events, summary):
        """H2 raw clearing prizes match recomputation."""
        worlds = [
            "gravity", "yukawa", "hubble", "ether", "oscillator",
            "coulomb", "circle", "extra_dimensions", "fractional",
            "dark_matter", "three_species",
        ]
        recomputed = compute_clearing_prizes(
            all_events, worlds, [5, 20, 50, 100, 200],
            [0, 1, 2, 3, 4], track="A", probability_source="raw",
        )
        stored = summary["h2_clearing_prizes"]["raw"]
        for world in worlds:
            rc = recomputed[world]
            sc = stored[world]
            if sc == "never":
                assert rc is None, f"{world}: expected never, got {rc}"
            else:
                assert rc == sc, f"{world}: recomputed={rc}, stored={sc}"

    def test_h2_calibrated_clearing(self, all_events, summary):
        """H2 calibrated clearing prizes match recomputation."""
        worlds = [
            "gravity", "yukawa", "hubble", "ether", "oscillator",
            "coulomb", "circle", "extra_dimensions", "fractional",
            "dark_matter", "three_species",
        ]
        recomputed = compute_clearing_prizes(
            all_events, worlds, [5, 20, 50, 100, 200],
            [0, 1, 2, 3, 4], track="A", probability_source="calibrated",
        )
        stored = summary["h2_clearing_prizes"]["calibrated"]
        for world in worlds:
            rc = recomputed[world]
            sc = stored[world]
            if sc == "never":
                assert rc is None
            else:
                assert rc == sc, f"{world}: recomputed={rc}, stored={sc}"


class TestP4AgentProfits:
    """Verify agent profit computations match the summary."""

    def test_h1_raw_profits_recompute(self, all_events, summary):
        """H1 raw profits are recomputable from events."""
        for prize_str, data in summary["h1_analysis"].get("raw", {}).items():
            template = f"track_a_raw_prize{prize_str}_seed{{seed}}"
            recomputed = compute_agent_profits(
                all_events, [0, 1, 2, 3, 4], template
            )
            stored = data.get("profits", {})
            for agent in set(list(recomputed.keys()) + list(stored.keys())):
                if agent in recomputed and agent in stored:
                    assert recomputed[agent]["mean_profit"] == pytest.approx(
                        stored[agent]["mean_profit"], abs=0.1
                    ), f"Profit mismatch for {agent} at prize={prize_str}"


class TestP4LedgerSize:
    """Verify ledger sizes match."""

    def test_ledger_sizes(self, all_events, all_ledger, summary):
        """Ledger size per run matches summary."""
        by_run_ledger = group_ledger_by_run(all_ledger)
        for run_id, stored_summary in summary["run_summaries"].items():
            recomputed = len(by_run_ledger.get(run_id, []))
            stored = stored_summary.get("ledger_size", 0)
            assert recomputed == stored, \
                f"Ledger size mismatch in {run_id}: {recomputed} vs {stored}"


class TestP4LabRevenue:
    """Verify lab revenue is recomputable."""

    def test_lab_revenue(self, all_events, summary):
        """Lab revenue matches recomputation from charge events."""
        by_run = group_events_by_run(all_events)
        for run_id, events in by_run.items():
            recomputed = sum(
                e["amount"] for e in events
                if e["type"] in ("round_charged", "experiment_charged")
            )
            stored = summary["run_summaries"][run_id]["lab_revenue"]
            assert recomputed == pytest.approx(stored, abs=1e-9), \
                f"Lab revenue mismatch in {run_id}"


# ===================================================================
# Profit and H1 verdict – checked against hand-computed values
# ===================================================================

from market import MarketRun, run_market, TrackCCostModel
from analysis import _h1_verdict


def _sure_win_events(seed: int) -> list[dict]:
    """One world, prize 30. 'winner' always passes at cost 8 × 1 = 8;
    'idle' believes p=0 and never bids."""
    cfg = MarketRun(
        run_id=f"hand_seed{seed}", seed=seed, ticks=5,
        worlds=["w"], agents=["winner", "idle"], prizes={"w": 30},
        starting_credits=100,
        cost_model=TrackCCostModel({"winner": 8, "idle": 8}, 1.0),
        true_probs={"winner": {"w": 1.0}, "idle": {"w": 0.0}},
        initial_beliefs={"winner": {"w": 1.0}, "idle": {"w": 0.0}},
        belief_weight=2, track="C", probability_source="test",
    )
    events, _ = run_market(cfg)
    return [e.to_dict() for e in events]


class TestProfitsHandComputed:

    def test_profit_equals_prize_minus_cost(self):
        events = _sure_win_events(0) + _sure_win_events(1)
        profits = compute_agent_profits(events, [0, 1], "hand_seed{seed}")
        assert profits["winner"]["mean_profit"] == pytest.approx(30 - 8)
        assert profits["idle"]["mean_profit"] == pytest.approx(0)

    def test_final_balances_are_absolute(self):
        balances = compute_final_balances(_sure_win_events(0))
        assert balances["agent:winner"] == pytest.approx(100 + 30 - 8)
        assert balances["agent:idle"] == pytest.approx(100)

    def test_non_bidders_counted_with_zero_bids(self):
        from analysis import compute_agent_bid_counts
        events = _sure_win_events(0)
        bids = compute_agent_bid_counts(events, [0], "hand_seed{seed}")
        assert bids == {"winner": 1.0, "idle": 0.0}


def _h1_data(**prize_rows):
    """prize -> {agent: (mean_profit, mean_bids)} into h1_results shape."""
    return {
        prize: {
            "profits": {a: {"mean_profit": p, "std_profit": 0.0}
                        for a, (p, _) in row.items()},
            "mean_bids": {a: b for a, (_, b) in row.items()},
        }
        for prize, row in prize_rows.items()
    }


ORDER = ["s1", "s2", "w1", "w2"]


class TestH1Verdict:

    def test_supported_when_strong_profit_and_weak_abstain(self):
        data = _h1_data(**{"50": {"s1": (10, 3), "s2": (5, 2),
                                  "w1": (0, 0), "w2": (0, 0)}})
        v = _h1_verdict("raw", data, ORDER)
        assert "→ SUPPORTED at prize=50" in v
        assert "never placed a bid" in v

    def test_not_supported_when_a_strong_agent_loses(self):
        # Old verdict called this SUPPORTED (top agent > 0 was enough)
        data = _h1_data(**{"50": {"s1": (10, 3), "s2": (-5, 2),
                                  "w1": (0, 0), "w2": (0, 0)}})
        v = _h1_verdict("raw", data, ORDER)
        assert "NOT SUPPORTED at any prize" in v

    def test_partial_when_weak_bid_and_lose(self):
        data = _h1_data(**{"50": {"s1": (10, 3), "s2": (5, 2),
                                  "w1": (-8, 1), "w2": (0, 0)}})
        v = _h1_verdict("raw", data, ORDER)
        assert "→ PARTIAL at prize=50" in v
        assert "never placed a bid" not in v

    def test_no_market_when_nobody_bids(self):
        data = _h1_data(**{"5": {a: (0, 0) for a in ORDER}})
        v = _h1_verdict("raw", data, ORDER)
        assert "no market" in v
