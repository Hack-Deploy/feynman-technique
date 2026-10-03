import pytest

from dm.hypotheses import evaluate_hypotheses, strength_split
from dm.replay import run_sweep
from dm.types import AttemptRecord


def _record(solver, world, passed, *, experiments=1, stated_p=None, seed=0):
    return AttemptRecord(
        attempt_id=f"{solver}:{world}:{seed}",
        source="live",
        protocol="forcebench_menu",
        venue="forcebench",
        world=world,
        solver=solver,
        seed=seed,
        stated_p_success=stated_p,
        rounds=experiments,
        experiments=experiments,
        lab_cost=float(experiments),
        verdict={"passed": passed},
    )


def test_strength_split_uses_largest_gap_and_first_tie():
    records = [
        _record("a", "w", True),
        _record("b", "w", True, seed=1),
        _record("b", "w", False, seed=2),
        _record("c", "w", False),
        _record("c", "w", False, seed=1),
    ]
    # Sorted pass rates are 1.0, 0.5, 0.0; both gaps are 0.5.
    split = strength_split(records)
    assert split["ordered"] == ["a", "b", "c"]
    assert split["gap"] == pytest.approx(0.5)
    assert split["strong"] == ["a"]
    assert split["weak"] == ["b", "c"]


def test_equal_strength_forcebench_agents_make_h1_unavailable_and_h3_supported():
    records = [
        _record("cheap", "world", True, experiments=1),
        _record("dear", "world", True, experiments=5),
    ]
    events, _ = run_sweep(records, "h3cheap", prizes=[20], seeds=list(range(1000)),
                          pool="forcebench")
    hypotheses = evaluate_hypotheses(
        events, records, "h3cheap", [20], list(range(1000)), "experiment_charged")
    assert hypotheses["H1"]["verdict"] == "UNAVAILABLE"
    assert "about equally strong" in hypotheses["H1"]["reason"]
    assert hypotheses["H3"]["verdict"] == "SUPPORTED"
    cheap = hypotheses["H3"]["evidence"]["per_agent"]["cheap"]
    assert cheap == {
        "pass_rate": 1.0,
        "experiments_per_attempt": 1.0,
        "attempts": 1,
        "attempt_started_events": sum(
            event.get("agent") == "cheap" and event["type"] == "attempt_started"
            for event in events
        ),
    }


def test_h1_strong_and_weak_verdict_follows_profit_and_bid_rule():
    records = [
        _record("strong", "world-a", True),
        _record("strong", "world-b", True, seed=1),
        _record("weak", "world-a", False),
        _record("weak", "world-b", False, seed=1),
    ]
    events, _ = run_sweep(records, "h1split", prizes=[20], seeds=[0, 1],
                          pool="forcebench")
    result = evaluate_hypotheses(
        events, records, "h1split", [20], [0, 1], "experiment_charged")["H1"]
    assert result["verdict"] == "SUPPORTED"
    assert result["evidence"]["strong"] == ["strong"]
    assert result["evidence"]["weak"] == ["weak"]
    assert result["evidence"]["per_prize"][20]["status"] == "SUPPORTED"


def test_h4_uses_stated_probability_joined_to_outcome():
    records = [_record("solver", "world", True, stated_p=0.2)]
    events, _ = run_sweep(records, "h4cal", prizes=[20], seeds=[0],
                          pool="forcebench")
    result = evaluate_hypotheses(
        events, records, "h4cal", [20], [0], "experiment_charged")["H4"]
    assert result["verdict"] == "NOT_SUPPORTED"
    assert result["evidence"]["pooled"]["n"] == 1
    assert result["evidence"]["pooled"]["brier"] == pytest.approx(0.64)
    assert result["evidence"]["pooled"]["calibration_in_the_large"] == pytest.approx(-0.8)
    assert result["evidence"]["pooled"]["mean_stated_p"] == pytest.approx(0.2)
    assert result["evidence"]["pooled"]["pass_rate"] == 1.0


def test_ara_rounds_pool_h3_and_h4_are_unavailable_with_reasons():
    record = AttemptRecord(
        attempt_id="ara",
        source="published_replay",
        protocol="ara_harness",
        venue="discoverphysics",
        world="world",
        solver="solver",
        seed=0,
        stated_p_success=None,
        rounds=2,
        experiments=4,
        lab_cost=2.0,
        verdict={"passed": True},
    )
    events, _ = run_sweep([record], "arahyp", prizes=[20], seeds=[0])
    hypotheses = evaluate_hypotheses(
        events, [record], "arahyp", [20], [0], "round_charged")
    assert hypotheses["H3"]["verdict"] == "UNAVAILABLE"
    assert hypotheses["H3"]["reason"] == (
        "pool charges rounds, not experiments; ARA experiment counts are not exact (PLAN C7)")
    assert hypotheses["H4"]["verdict"] == "UNAVAILABLE"
    assert hypotheses["H4"]["reason"] == (
        "no stated probabilities in this pool (ARA records carry none); "
        "models state their chances on /live")
