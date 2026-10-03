"""Phase 1: OutcomeSource generalisation.

BernoulliTable must reproduce the Phase 0 outputs byte for byte; ReplayPool must
conserve credits, be deterministic, draw without replacement within a run, and
keep solvers off worlds where their pool is empty.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from analysis import build_full_summary, compute_final_balances, save_summary
from dm.outcomes import ReplayPool, experiments_cost, rounds_cost
from dm.store import FIXTURES_DIR, AttemptStore
from market import MarketRun, TrackACostModel, BernoulliTable, run_market
from runner import run_all_track_a, run_all_track_c, save_results

BASELINE = json.loads(
    (Path(__file__).parent / "fixtures" / "baseline_sha256.json").read_text())


class TestBernoulliByteIdentical:

    def test_run_outputs_match_phase0_baseline(self, tmp_path):
        results = {**run_all_track_a(), **run_all_track_c()}
        save_results(results, tmp_path)
        events = json.loads((tmp_path / "events.json").read_text())
        ledger = json.loads((tmp_path / "ledger.json").read_text())
        save_summary(build_full_summary(events, ledger), tmp_path)
        for name in ["events.json", "ledger.json", "summary.json"]:
            digest = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
            assert digest == BASELINE[name], f"{name} changed"

    def test_explicit_bernoulli_source_equals_default(self):
        kw = dict(run_id="t", seed=3, ticks=30, worlds=["w"], agents=["a"],
                  prizes={"w": 40}, starting_credits=100,
                  cost_model=TrackACostModel(),
                  true_probs={"a": {"w": 0.4}}, initial_beliefs={"a": {"w": 0.6}},
                  belief_weight=2, track="A", probability_source="t")
        default = run_market(MarketRun(**kw))
        explicit = run_market(MarketRun(
            **kw, outcome_source=BernoulliTable(kw["cost_model"], kw["true_probs"])))
        assert [e.to_dict() for e in default[0]] == [e.to_dict() for e in explicit[0]]


# ---------------------------------------------------------------- ReplayPool

def _pool(cost="experiments"):
    recs = AttemptStore(FIXTURES_DIR / "replay_pool.jsonl").load()
    if cost == "rounds":
        return recs, ReplayPool(recs, rounds_cost(1.0), charge_event="round_charged")
    return recs, ReplayPool(recs, experiments_cost(0.5))


def _replay_run(seed, prize=60.0, cost="experiments", ticks=50):
    recs, pool = _pool(cost)
    worlds = ["gravity", "yukawa", "coulomb_easy"]
    agents = ["fast", "slow"]
    cfg = MarketRun(
        run_id=f"replay_fixture_seed{seed}", seed=seed, ticks=ticks,
        worlds=worlds, agents=agents, prizes={w: prize for w in worlds},
        starting_credits=100, cost_model=pool, true_probs=None,
        initial_beliefs={a: {w: 0.5 for w in worlds} for a in agents},
        belief_weight=2, track="replay_fixture", probability_source="replay:fixture",
        outcome_source=pool, state_confidence=True,
    )
    events, ledger = run_market(cfg)
    return recs, [e.to_dict() for e in events], [r.to_dict() for r in ledger]


class TestReplayPool:

    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("cost", ["experiments", "rounds"])
    def test_conservation(self, seed, cost):
        # run_market asserts conservation every tick; also check from the log.
        _, events, _ = _replay_run(seed, cost=cost)
        assert abs(sum(compute_final_balances(events).values())) < 1e-9

    @pytest.mark.parametrize("seed", range(3))
    def test_deterministic(self, seed):
        assert _replay_run(seed)[1] == _replay_run(seed)[1]

    @pytest.mark.parametrize("seed", range(5))
    def test_without_replacement_within_run(self, seed):
        _, events, _ = _replay_run(seed, prize=500.0, ticks=200)
        drawn = Counter(e["attempt_id"] for e in events if e["type"] == "attempt_started")
        assert drawn and max(drawn.values()) == 1

    def test_with_replacement_across_seeds(self):
        used = [{e["attempt_id"] for e in _replay_run(s, prize=500.0)[1]
                 if e["type"] == "attempt_started"} for s in range(3)]
        assert used[0] & used[1] & used[2], "same records should be reusable across runs"

    @pytest.mark.parametrize("seed", range(5))
    def test_empty_pool_cannot_bid(self, seed):
        _, events, _ = _replay_run(seed, prize=500.0)
        assert not [e for e in events if e.get("agent") == "fast"
                    and e.get("world") == "coulomb_easy"]

    def test_charges_match_records(self):
        recs, events, _ = _replay_run(0, prize=500.0)
        by_id = {r.attempt_id: r for r in recs}
        charges = [e for e in events if e["type"] == "experiment_charged"]
        assert charges
        for e in charges:
            r = by_id[e["attempt_id"]]
            assert e["count"] == r.experiments
            assert e["amount"] == pytest.approx(r.experiments * 0.5)

    def test_outcomes_match_records(self):
        recs, events, _ = _replay_run(1, prize=500.0)
        by_id = {r.attempt_id: r for r in recs}
        for e in events:
            if e["type"] == "verdict_issued":
                assert e["detail"]["passed"] == by_id[e["attempt_id"]].passed

    def test_confidence_stated_carries_record_p(self):
        recs, events, _ = _replay_run(2, prize=500.0)
        by_id = {r.attempt_id: r for r in recs}
        stated = [e for e in events if e["type"] == "confidence_stated"]
        assert stated
        for e in stated:
            assert e["p"] == by_id[e["attempt_id"]].stated_p_success

    def test_lifecycle_order(self):
        _, events, _ = _replay_run(0, prize=500.0)
        order = ["bid_placed", "attempt_started", "confidence_stated",
                 "experiment_charged", "attempt_submitted", "verdict_issued"]
        types = [e["type"] for e in events]
        i = types.index("bid_placed")
        assert types[i:i + len(order)] == order

    def test_ledger_metric_is_measured_score_not_truth(self):
        recs, _, ledger = _replay_run(0, prize=500.0)
        by_id = {r.attempt_id: r for r in recs}
        for row in ledger:
            rec = by_id[row["context"]["replayed_attempt_id"]]
            assert row["outcome"]["metric"] == rec.verdict["normalised_mse"]
