"""Phase 4: replay-market sweep on imported ARA records (offline sample)."""

import json
from collections import Counter

import pytest

from analysis import compute_balances_over_time, compute_final_balances
from dm import replay
from dm.importers import ara
from dm.store import FIXTURES_DIR, AttemptStore

SAMPLE = FIXTURES_DIR / "ara_sample"
LABEL = "sample"


@pytest.fixture(scope="module")
def records():
    recs, _ = ara.import_cache(SAMPLE, ["fable", "gpt5.5", "opus4.8-max"])
    return recs


@pytest.fixture(scope="module")
def sweep(records):
    return replay.run_sweep(records, LABEL)


def _runs(events):
    out = {}
    for e in events:
        out.setdefault(e["run_id"], []).append(e)
    return out


def test_sweep_shape(sweep):
    runs = _runs(sweep[0])
    assert len(runs) == len(replay.PRIZES) * len(replay.SEEDS)
    assert replay.run_id(LABEL, 50, 3) in runs


def test_conserves_credits(sweep):
    for ev in _runs(sweep[0]).values():
        assert abs(sum(compute_final_balances(ev).values())) < 1e-9


def test_no_agent_overdrawn(sweep):
    for ev in _runs(sweep[0]).values():
        for acct, hist in compute_balances_over_time(ev).items():
            if acct.startswith("agent:"):
                assert min(b for _, b in hist) >= -1e-9


def test_deterministic(records, sweep):
    assert replay.run_sweep(records, LABEL)[0] == sweep[0]


def test_no_record_drawn_twice_within_run(sweep):
    assert replay.max_draws_per_record(sweep[0]) == 1
    assert replay.max_attempts_per_solver_world(sweep[0]) == 1


def test_charges_are_rounds(records, sweep):
    by_id = {r.attempt_id: r for r in records}
    charges = [e for e in sweep[0] if e["type"] == "round_charged"]
    assert charges
    for e in charges:
        assert e["amount"] == by_id[e["attempt_id"]].rounds


def test_verdicts_match_records(records, sweep):
    by_id = {r.attempt_id: r for r in records}
    for e in sweep[0]:
        if e["type"] == "verdict_issued":
            assert e["detail"]["passed"] == by_id[e["attempt_id"]].passed


def test_leave_one_out_beliefs(records):
    pool = replay.make_pool(records)
    b = replay.loo_beliefs(pool)
    for s in pool.solvers():
        for w in pool.worlds():
            others = [r for r in records if r.solver == s and r.world != w]
            assert b[s][w] == pytest.approx(sum(r.passed for r in others) / len(others))


def test_loo_fallback_when_no_other_worlds(records):
    one = [r for r in records if r.world == "gravity"]
    b = replay.loo_beliefs(replay.make_pool(one))
    assert all(v == {"gravity": 0.5} for v in b.values())


def test_clearing_prize_rule():
    def run(prize, seed, solved):
        ev = [{"run_id": replay.run_id("t", prize, seed), "type": "prize_posted",
               "world": w, "tick": 0} for w in ("a", "b", "c")]
        ev += [{"run_id": replay.run_id("t", prize, seed), "type": "attempt_passed",
                "world": w, "agent": "x", "tick": 1} for w in solved]
        return ev
    events = []
    for seed in range(5):
        events += run(5, seed, ["a"] if seed < 2 else [])        # a: 2/5 at 5
        events += run(20, seed, ["a"] if seed < 3 else [])       # a: 3/5 at 20
        events += run(50, seed, ["a", "b"])                      # b: 5/5 at 50
    got = replay.clearing_prizes(events, ["a", "b", "c"], "t", prizes=[5, 20, 50])
    assert got == {"a": 20, "b": 50, "c": "never"}


def test_summary_from_event_log(records, sweep):
    s = replay.summarise(*sweep, records, LABEL)
    assert s["checks"]["credit_sum_per_run_max_abs"] < 1e-9
    assert s["checks"]["max_draws_per_record_per_run"] == 1
    assert set(s["clearing_prizes"]) == set(ara.WORLD_VARS)
    assert "close to deterministic" in s["caveat"]
    # Profit = final - opening balance, so the agents' total profit plus the lab's
    # revenue equals the prizes paid out.
    runs = _runs(sweep[0])
    for p in replay.PRIZES:
        paid = sum(e["amount"] for s_ in replay.SEEDS
                   for e in runs[replay.run_id(LABEL, p, s_)] if e["type"] == "prize_paid")
        prof = sum(v["profit"]["mean"] for v in s["profit_and_bids"][p].values()) * 5
        assert prof + s["lab_revenue"][p]["mean"] * 5 == pytest.approx(paid)


def test_outcomes_identical_across_seeds(records, sweep):
    # The caveat: one record per (solver, world) → same outcome in every seed.
    outcomes: dict = {}
    for e in sweep[0]:
        if e["type"] == "verdict_issued":
            outcomes.setdefault((e["agent"], e["world"]), set()).add(e["detail"]["passed"])
    assert outcomes and all(len(v) == 1 for v in outcomes.values())


def test_ara_verdict_sensitivity(records):
    strict = replay.with_verdict(records, "ara")
    assert [r.passed for r in strict] == [r.extra["ara_passed"] for r in strict]
    assert sum(r.passed for r in strict) <= sum(r.passed for r in records)
    with pytest.raises(ValueError):
        replay.with_verdict(records, "bogus")


def test_run_and_save(tmp_path, records):
    store = tmp_path / "ara.jsonl"
    AttemptStore(store).append(records)
    s = replay.run_and_save("ara", out_dir=tmp_path / "out", store_path=store)
    for name in ["events.json", "ledger.json", "summary.json"]:
        assert (tmp_path / "out" / name).exists()
    events = json.loads((tmp_path / "out" / "events.json").read_text())
    assert Counter(e["type"] for e in events)["account_funded"] == 25 * (1 + 3)
    assert s["verdict_rule"] == "numeric"
