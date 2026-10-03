"""Adversarial readiness tests for the replay market, analysis and calibration
(area: live-market). Everything is recomputed from the event log.

Tests marked ``xfail(strict=True)`` document a bug or gap found on ``real-attempts``;
they start failing (XPASS) once it is fixed, so the marker must then be removed.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import threading
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from analysis import (build_full_summary, compute_agent_profits,
                      compute_clearing_prizes, compute_final_balances,
                      compute_run_summary)
from dm.outcomes import ReplayPool, experiments_cost, recorded_cost, rounds_cost
from dm.store import FIXTURES_DIR, AttemptStore
from dm.types import AttemptRecord, Preregistration, canonical_json
from market import MarketRun, TrackACostModel, run_market

WORLDS = ["gravity", "yukawa", "coulomb_easy"]
AGENTS = ["fast", "slow"]
PRIZES = [5, 20, 60, 100, 500]
PRICES = [0.25, 0.5, 1.0, 2.0]
SEEDS = range(5)
START = 100.0


def _records() -> list[AttemptRecord]:
    return AttemptStore(FIXTURES_DIR / "replay_pool.jsonl").load()


def _cfg(pool, *, seed=0, prize=60.0, price=None, credits=START, ticks=50,
         worlds=WORLDS, agents=AGENTS, cost_model=None, preregs=None):
    tag = f"price{price}_" if price is not None else ""
    return MarketRun(
        run_id=f"replay_fixture_{tag}prize{int(prize)}_seed{seed}", seed=seed, ticks=ticks,
        worlds=list(worlds), agents=list(agents), prizes={w: float(prize) for w in worlds},
        starting_credits=credits, cost_model=cost_model or pool, true_probs=None,
        initial_beliefs={a: {w: 0.5 for w in worlds} for a in agents},
        belief_weight=2, track="replay_fixture", probability_source="replay:fixture",
        outcome_source=pool, state_confidence=True, preregs=preregs)


def _run(pool, **kw):
    events, ledger = run_market(_cfg(pool, **kw))
    return [e.to_dict() for e in events], [r.to_dict() for r in ledger]


def _sweep():
    recs = _records()
    for price in PRICES:
        pool = ReplayPool(recs, experiments_cost(price))
        for prize in PRIZES:
            for seed in SEEDS:
                ev, led = _run(pool, seed=seed, prize=prize, price=price)
                yield price, prize, seed, ev, led


_SWEEP = None


def sweep():
    global _SWEEP
    if _SWEEP is None:
        _SWEEP = list(_sweep())
    return _SWEEP


# ------------------------------------------------------------ recomputation

def is_charge(e: dict) -> bool:
    return e["type"].endswith("_charged")


def running_balances(events: list[dict]):
    """Yield (event, balances) after applying each transfer."""
    bal: dict[str, float] = defaultdict(float)
    for e in sorted(events, key=lambda e: e["seq"]):
        if "from" in e or "to" in e or "amount" in e:
            if "from" in e and "to" in e and "amount" in e:
                bal[e["from"]] -= e["amount"]
                bal[e["to"]] += e["amount"]
        yield e, dict(bal)


def clearing_from_log(events: list[dict], world: str, prizes, seeds,
                      price, min_solved=3) -> float | None:
    """Lowest prize solved in >= 3 of the seeds, read from prize_posted/attempt_passed."""
    solved = defaultdict(set)  # prize -> seeds where world was solved
    posted = {}
    for e in events:
        if e["type"] == "prize_posted" and e["world"] == world:
            posted[e["run_id"]] = e["amount"]
    for e in events:
        if e["type"] == "attempt_passed" and e["world"] == world:
            solved[posted[e["run_id"]]].add(e["seed"])
    for p in sorted(prizes):
        if len(solved[float(p)] & set(seeds)) >= min_solved:
            return p
    return None


def join_confidence_verdict(events: list[dict]) -> list[tuple[str, float, bool]]:
    """(solver, stated p, passed) per attempt, joined on (run_id, attempt_id)."""
    conf, verd = {}, {}
    for e in events:
        key = (e["run_id"], e.get("attempt_id"))
        if e["type"] == "confidence_stated":
            assert key not in conf, f"duplicate confidence key {key}"
            conf[key] = (e["agent"], e["p"])
        elif e["type"] == "verdict_issued":
            assert key not in verd, f"duplicate verdict key {key}"
            verd[key] = e["detail"]["passed"]
    assert conf.keys() == verd.keys()
    return [(conf[k][0], conf[k][1], verd[k]) for k in sorted(conf)]


def brier(rows) -> dict[str, float]:
    by = defaultdict(list)
    for solver, p, passed in rows:
        by[solver].append((p - float(passed)) ** 2)
    return {s: sum(v) / len(v) for s, v in by.items()}


def reliability(rows, edges=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001)):
    """solver -> list of (bin_lo, bin_hi, n, mean_p, pass_rate)."""
    out = defaultdict(list)
    for s in sorted({r[0] for r in rows}):
        for lo, hi in zip(edges, edges[1:]):
            b = [(p, ok) for sv, p, ok in rows if sv == s and lo <= p < hi]
            if b:
                out[s].append((lo, min(hi, 1.0), len(b), sum(p for p, _ in b) / len(b),
                               sum(ok for _, ok in b) / len(b)))
    return dict(out)


# ------------------------------------------------------------ sweep checks

class TestReplaySweep:
    """prize × price × seed over the fixture pool (100 runs)."""

    def test_conservation_and_escrow_from_log(self):
        for price, prize, seed, ev, _ in sweep():
            funded = sum(e["amount"] for e in ev if e["type"] == "account_funded")
            bal = compute_final_balances(ev)
            internal = {k: v for k, v in bal.items() if k != "external"}
            assert sum(internal.values()) == pytest.approx(funded, abs=1e-9)
            assert bal["escrow"] == pytest.approx(0.0, abs=1e-9)

    def test_balances_never_negative_during_run(self):
        for price, prize, seed, ev, _ in sweep():
            for e, bal in running_balances(ev):
                bad = {k: v for k, v in bal.items()
                       if k != "external" and v < -1e-9}
                assert not bad, (price, prize, seed, e, bad)

    def test_balances_match_ledger(self):
        """Agent final balance = start − Σ charges + Σ prizes, from the ledger alone."""
        for price, prize, seed, ev, led in sweep():
            bal = compute_final_balances(ev)
            for a in AGENTS:
                rows = [r for r in led if r["solver"] == a]
                expect = (START - sum(r["effort"]["credits"] for r in rows)
                          + prize * sum(r["outcome"]["passed"] for r in rows))
                assert bal[f"agent:{a}"] == pytest.approx(expect)
            lab = sum(e["amount"] for e in ev if is_charge(e))
            assert bal.get("lab", 0.0) == pytest.approx(lab)
            assert compute_run_summary(ev, led)["lab_revenue"] == pytest.approx(lab)
            assert lab == pytest.approx(sum(r["effort"]["credits"] for r in led))

    def test_charge_equals_count_times_price(self):
        by_id = {r.attempt_id: r for r in _records()}
        for price, prize, seed, ev, _ in sweep():
            for e in ev:
                if is_charge(e):
                    assert e["type"] == "experiment_charged"
                    assert e["count"] == by_id[e["attempt_id"]].experiments
                    assert e["amount"] == pytest.approx(e["count"] * price)

    def test_lifecycle_per_attempt(self):
        for price, prize, seed, ev, _ in sweep():
            ev = sorted(ev, key=lambda e: e["seq"])
            assert [e["seq"] for e in ev] == list(range(len(ev)))
            for i, e in enumerate(ev):
                if e["type"] != "bid_placed":
                    continue
                key = (e["tick"], e["agent"], e["world"])
                head = [x["type"] for x in ev[i:i + 6]]
                assert head == ["bid_placed", "attempt_started", "confidence_stated",
                                "experiment_charged", "attempt_submitted",
                                "verdict_issued"], (price, prize, seed, key, head)
                same = all((x["tick"], x["agent"], x["world"]) == key for x in ev[i:i + 6])
                assert same
                ids = {x.get("attempt_id") for x in ev[i + 1:i + 6]}
                assert len(ids) == 1 and None not in ids
                passed = ev[i + 5]["detail"]["passed"]
                tail = [x["type"] for x in ev[i + 6:i + 9]]
                if passed:
                    assert tail == ["attempt_passed", "prize_paid", "estimate_updated"]
                else:
                    assert tail[:2] == ["attempt_failed", "estimate_updated"]

    def test_each_prize_paid_or_refunded_exactly_once(self):
        for price, prize, seed, ev, _ in sweep():
            last_tick = max(e["tick"] for e in ev)
            for w in WORLDS:
                paid = [e for e in ev if e["type"] == "prize_paid" and e["world"] == w]
                ref = [e for e in ev if e["type"] == "prize_refunded" and e["world"] == w]
                assert len(paid) + len(ref) == 1
                for e in paid + ref:
                    assert e["amount"] == prize
                for e in ref:
                    assert e["tick"] == last_tick and e["to"] == "researcher"
                if paid:  # nothing happens on a world after it is paid
                    after = [e for e in ev if e.get("world") == w
                             and e["seq"] > paid[0]["seq"]
                             and e["type"] in ("bid_placed", "attempt_started")]
                    assert not after

    def test_insufficient_credits_is_not_a_transfer_and_once_per_pair(self):
        seen_any = False
        for price, prize, seed, ev, _ in sweep():
            ins = [e for e in ev if e["type"] == "insufficient_credits"]
            seen_any |= bool(ins)
            for e in ins:
                assert "from" not in e and "to" not in e
            pairs = Counter((e["agent"], e["world"]) for e in ins)
            assert not pairs or max(pairs.values()) == 1
        assert seen_any, "sweep should exercise insufficient_credits"

    def test_without_replacement_and_empty_pool(self):
        for price, prize, seed, ev, _ in sweep():
            drawn = Counter(e["attempt_id"] for e in ev if e["type"] == "attempt_started")
            assert not drawn or max(drawn.values()) == 1
            assert not [e for e in ev if e.get("agent") == "fast"
                        and e.get("world") == "coulomb_easy"]


# ------------------------------------------------------------ determinism

class TestDeterminism:

    def test_pool_reuse_order_does_not_matter(self):
        recs = _records()
        pool = ReplayPool(recs, experiments_cost(0.5))
        first = _run(pool, seed=0, prize=500)
        _run(pool, seed=1, prize=500)
        _run(pool, seed=3, prize=20)
        assert _run(pool, seed=0, prize=500) == first

    def test_record_file_order_does_not_matter(self):
        recs = _records()
        shuffled = list(recs)
        random.Random(7).shuffle(shuffled)
        a = _run(ReplayPool(recs, experiments_cost(0.5)), seed=2, prize=500)
        b = _run(ReplayPool(shuffled, experiments_cost(0.5)), seed=2, prize=500)
        assert a == b

    def test_created_at_does_not_affect_outcomes(self):
        recs = _records()
        moved = [AttemptRecord(**{**r.to_dict(), "created_at": "1999-01-01T00:00:00"})
                 for r in recs]
        assert (_run(ReplayPool(recs, experiments_cost(1)), seed=4, prize=100)
                == _run(ReplayPool(moved, experiments_cost(1)), seed=4, prize=100))


# ------------------------------------------------------------ analysis.py

class TestAnalysisOnReplayRuns:

    def test_profits_recomputed_from_log(self):
        events = [e for p, z, s, ev, _ in sweep() if p == 1.0 and z == 60 for e in ev]
        got = compute_agent_profits(events, list(SEEDS),
                                    "replay_fixture_price1.0_prize60_seed{seed}")
        for a in AGENTS:
            mine = []
            for s in SEEDS:
                run = [e for e in events if e["seed"] == s]
                mine.append(compute_final_balances(run)[f"agent:{a}"] - START)
            assert got[a]["mean_profit"] == pytest.approx(sum(mine) / len(mine))
            assert got[a]["num_seeds"] == 5

    def test_clearing_prizes_recomputed_from_log(self):
        """Hand recompute works; this is the reference the analysis must match."""
        events = [e for p, z, s, ev, _ in sweep() if p == 1.0 for e in ev]
        clearing = {w: clearing_from_log(events, w, PRIZES, SEEDS, 1.0) for w in WORLDS}
        assert clearing["coulomb_easy"] is None  # slow's only attempt fails
        assert clearing["gravity"] is not None

    def test_analysis_clearing_prizes_match_log_for_replay(self):
        events = [e for p, z, s, ev, _ in sweep() if p == 1.0 for e in ev]
        mine = {w: clearing_from_log(events, w, PRIZES, SEEDS, 1.0) for w in WORLDS}
        got = compute_clearing_prizes(
            events, WORLDS, PRIZES, list(SEEDS), track="replay", price=1.0,
            run_id_template="replay_fixture_price{price}_prize{prize}_seed{seed}")
        assert got == mine

    def test_summary_does_not_invent_verdicts_for_missing_tracks(self):
        ev, led = _run(ReplayPool(_records(), experiments_cost(0.5)), prize=60)
        s = build_full_summary(ev, led)
        assert "NOT SUPPORTED" not in s["verdicts"]["h3"]
        assert s["verdicts"]["h3"].startswith("UNAVAILABLE")

    def test_summary_counts_both_charge_types_as_lab_revenue(self):
        recs = _records()
        ev_r, led_r = _run(ReplayPool(recs, rounds_cost(1.0), charge_event="round_charged"),
                           prize=500)
        ev_e, led_e = _run(ReplayPool(recs, experiments_cost(1.0)), prize=500)
        for ev, led in [(ev_r, led_r), (ev_e, led_e)]:
            lab = sum(e["amount"] for e in ev if is_charge(e))
            assert compute_run_summary(ev, led)["lab_revenue"] == pytest.approx(lab)


# ------------------------------------------------------------ calibration (H4)

class TestCalibrationFromEvents:

    def test_brier_from_events_matches_records(self):
        by_id = {r.attempt_id: r for r in _records()}
        events = [e for _, _, _, ev, _ in sweep() for e in ev]
        rows = join_confidence_verdict(events)
        assert rows
        got = brier(rows)
        ref = defaultdict(list)
        for e in events:
            if e["type"] == "attempt_started":
                r = by_id[e["attempt_id"]]
                ref[r.solver].append((r.stated_p_success - float(r.passed)) ** 2)
        for s, v in ref.items():
            assert got[s] == pytest.approx(sum(v) / len(v))

    def test_hand_computed_brier_and_reliability_single_run(self):
        ev, _ = _run(ReplayPool(_records(), experiments_cost(0.25)), seed=0,
                     prize=500, ticks=200)
        rows = join_confidence_verdict(ev)
        # prize 500 / cheap: every record of both solvers is drawn exactly once
        # until its world closes. Hand table for whatever was drawn:
        drawn = sorted((s, p, ok) for s, p, ok in rows)
        by_hand = defaultdict(lambda: [0.0, 0])
        for s, p, ok in drawn:
            by_hand[s][0] += (p - ok) ** 2
            by_hand[s][1] += 1
        assert brier(rows) == {s: t / n for s, (t, n) in by_hand.items()}
        rel = reliability(rows)
        for s, bins in rel.items():
            assert sum(b[2] for b in bins) == by_hand[s][1]
            for lo, hi, n, mp, pr in bins:
                assert lo <= mp <= hi and 0.0 <= pr <= 1.0

    def test_join_key_unique_without_attempt_ids(self):
        """BernoulliTable + state_confidence: attempt_id is None, so the only join key
        is (run_id, tick, agent, world); it must be unique."""
        cfg = MarketRun(run_id="b", seed=1, ticks=40, worlds=["w1", "w2"], agents=["a", "b"],
                        prizes={"w1": 40, "w2": 40}, starting_credits=100,
                        cost_model=TrackACostModel(), true_probs={"a": {"w1": .3, "w2": .5},
                                                                  "b": {"w1": .6, "w2": .2}},
                        initial_beliefs={"a": {"w1": .6, "w2": .6}, "b": {"w1": .6, "w2": .6}},
                        belief_weight=2, track="A", probability_source="t",
                        state_confidence=True)
        ev = [e.to_dict() for e in run_market(cfg)[0]]
        keys = Counter((e["run_id"], e["tick"], e["agent"], e["world"]) for e in ev
                       if e["type"] == "confidence_stated")
        assert keys and max(keys.values()) == 1
        assert all(e.get("attempt_id") is None for e in ev if e["type"] == "verdict_issued")

    def test_confidence_event_marks_fallback(self):
        recs = [AttemptRecord(**{**r.to_dict(), "stated_p_success": None})
                for r in _records()]
        ev, _ = _run(ReplayPool(recs, experiments_cost(0.5)), prize=500)
        cs = [e for e in ev if e["type"] == "confidence_stated"]
        assert cs and all((e.get("detail") or {}).get("p_source") == "belief" for e in cs)

    def test_events_link_to_ledger_rows(self):
        ev, led = _run(ReplayPool(_records(), experiments_cost(0.5)), prize=500)
        ids = {r["attempt_id"] for r in led}
        v = [e for e in ev if e["type"] == "verdict_issued"]
        assert all((e.get("detail") or {}).get("market_attempt_id") in ids for e in v)


# ------------------------------------------------------------ preregistration

def _preregs():
    return {w: Preregistration(question_id=f"fixture/{w}", venue="fixture", world=w,
                               test_seed=11, test_cases=[{"SECRET_CASE": w, "p1": 3.0}],
                               norm_variance=4.2, oracle_version="test")
            for w in WORLDS}


class TestPreregLifecycle:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_commit_reveal_and_secrecy(self, seed):
        pr = _preregs()
        # Since review/oracle-fixes, run_market refuses replayed verdicts that were not
        # scored against the posted preregistration, so bind each record to its world's.
        bound = [AttemptRecord(**{**r.to_dict(), "verdict": {
                     **r.verdict, "prereg_commitment": pr[r.world].commitment()}})
                 for r in _records() if r.world in pr]
        ev, _ = _run(ReplayPool(bound, experiments_cost(1)), seed=seed, prize=60,
                     preregs=pr)
        ev = sorted(ev, key=lambda e: e["seq"])
        for w in WORLDS:
            c = pr[w].commitment()
            posted = [e for e in ev if e["type"] == "prize_posted" and e["world"] == w]
            committed = [e for e in ev if e["type"] == "prereg_committed" and e["world"] == w]
            revealed = [e for e in ev if e["type"] == "prereg_revealed" and e["world"] == w]
            assert [e["commitment"] for e in posted + committed + revealed] == [c, c, c]
            d = revealed[0]["detail"]
            assert hashlib.sha256(canonical_json(d).encode()).hexdigest() == c
            close = [e for e in ev if e.get("world") == w
                     and e["type"] in ("prize_paid", "prize_refunded")][0]
            assert revealed[0]["tick"] == close["tick"]
            assert revealed[0]["seq"] > close["seq"]
            before = json.dumps([e for e in ev if e["seq"] < revealed[0]["seq"]])
            assert f'"SECRET_CASE": "{w}"' not in before

    def test_verdict_commitment_matches_market_prereg(self):
        pr = _preregs()
        records = [
            AttemptRecord(**{
                **record.to_dict(),
                "verdict": {
                    **record.verdict,
                    "prereg_commitment": pr[record.world].commitment(),
                },
            })
            for record in _records()
        ]
        ev, _ = _run(ReplayPool(records, experiments_cost(1)), prize=500, preregs=pr)
        for e in ev:
            if e["type"] == "verdict_issued":
                assert e["detail"]["prereg_match"] is True

        with pytest.raises(ValueError, match="not scored against"):
            _run(ReplayPool(_records(), experiments_cost(1)), prize=500, preregs=pr)


# ------------------------------------------------------------ engine guards

class TestEngineGuards:

    def test_mismatched_cost_model_cannot_overdraw(self):
        pool = ReplayPool(_records(), experiments_cost(5.0))
        with pytest.raises(ValueError, match="prices its own attempts"):
            _run(pool, prize=500, cost_model=TrackACostModel())

    def test_negative_cost_rejected(self):
        with pytest.raises(ValueError, match="lab_cost"):
            AttemptRecord(**{**_records()[0].to_dict(), "lab_cost": -10.0})

    def test_every_collected_bid_leaves_an_event(self):
        # One solver, two worlds, enough for one 41-credit attempt but not two.
        dropped = []
        for seed in range(10):
            ev, _ = _run(ReplayPool(_records(), experiments_cost(1.0)), seed=seed,
                         prize=500, credits=75, worlds=["gravity", "yukawa"],
                         agents=["slow"])
            t1 = {e["world"] for e in ev if e["tick"] == 1 and e.get("agent") == "slow"}
            if t1 != {"gravity", "yukawa"}:
                dropped.append(seed)
        assert not dropped, f"bids silently dropped in seeds {dropped}"

    def test_non_finite_metric_is_json_safe(self):
        recs = _records()
        next(r for r in recs if not r.passed).verdict["normalised_mse"] = math.inf
        ev, led = _run(ReplayPool(recs, experiments_cost(0.5)), prize=500)
        json.dumps(ev, allow_nan=False)
        json.dumps(led, allow_nan=False)


# ------------------------------------------------------------ store

class TestStore:

    def _rec(self, i, **kw):
        base = _records()[0].to_dict()
        return AttemptRecord(**{**base, "attempt_id": f"r{i}", **kw})

    def test_supersede_keeps_first_seen_order(self, tmp_path):
        s = AttemptStore(tmp_path / "a.jsonl")
        s.append([self._rec(1), self._rec(2)])
        s.append([self._rec(1, rounds=99)])
        got = s.load()
        assert [r.attempt_id for r in got] == ["r1", "r2"] and got[0].rounds == 99

    def test_concurrent_appends_do_not_interleave(self, tmp_path):
        s = AttemptStore(tmp_path / "c.jsonl")
        big = "x" * 20000  # > io.DEFAULT_BUFFER_SIZE, forces multi-write lines

        def worker(k):
            for j in range(25):
                s.append([self._rec(f"{k}-{j}", submitted_law=big)])
        ts = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        lines = (tmp_path / "c.jsonl").read_text().splitlines()
        assert len(lines) == 200
        for ln in lines:
            json.loads(ln)

    def test_partial_trailing_line_tolerated(self, tmp_path):
        p = tmp_path / "t.jsonl"
        AttemptStore(p).append([self._rec(1)])
        with p.open("a") as f:
            f.write('{"attempt_id": "r2", "sour')
        with pytest.warns(UserWarning, match="truncated final line"):
            assert [r.attempt_id for r in AttemptStore(p).load()] == ["r1"]

    def test_store_writes_strict_json(self, tmp_path):
        p = tmp_path / "i.jsonl"
        record = self._rec(1)
        record.verdict["normalised_mse"] = math.inf
        with pytest.raises(ValueError):
            AttemptStore(p).append([record])

    def test_unknown_fields_round_trip(self):
        d = {**self._rec(1).to_dict(), "schema_version": 2}
        with pytest.raises(ValueError, match="schema_version"):
            AttemptRecord.from_dict(d)


# ------------------------------------------------------------ ARA-shaped pools (C12)

class TestAraShapedPool:
    """One attempt per (solver, world), own verdict kept in extra, numeric verdict settles."""

    def _ara(self):
        rows = [("m1", "gravity", 7, 20, 9e-5, True, True),
                ("m1", "yukawa", 8, 24, 2e-9, True, True),
                ("m1", "fractional", 8, 22, 2.8e-7, True, False),   # numeric PASS, own FAIL
                ("m2", "gravity", 7, 32, 1.2e-3, True, True),
                ("m2", "coulomb_easy", 8, 16, 2.45, False, False),
                ("m2", "extra_dimensions", 14, 49, 0.0985, True, False)]
        return [AttemptRecord(attempt_id=f"ara:{m}:{w}", source="published_replay",
                              protocol="ara_harness", venue="discoverphysics", world=w,
                              solver=m, seed=0, stated_p_success=None, rounds=r,
                              experiments=x, lab_cost=float(r),
                              verdict={"normalised_mse": n, "passed": num,
                                       "prereg_commitment": None, "explanation_score": None},
                              extra={"own_passed": own})
                for m, w, r, x, n, num, own in rows]

    @pytest.mark.parametrize("seed", SEEDS)
    def test_each_solver_attempts_each_world_at_most_once(self, seed):
        recs = self._ara()
        worlds = sorted({r.world for r in recs})
        pool = ReplayPool(recs, rounds_cost(1.0), charge_event="round_charged")
        ev, led = _run(pool, seed=seed, prize=100, worlds=worlds, agents=["m1", "m2"],
                       ticks=200)
        starts = Counter((e["agent"], e["world"]) for e in ev if e["type"] == "attempt_started")
        assert max(starts.values()) == 1
        for e in ev:
            if e["type"] == "round_charged":
                rec = next(r for r in recs if r.attempt_id == e["attempt_id"])
                assert e["amount"] == rec.rounds and "count" not in e
            if e["type"] == "verdict_issued":
                rec = next(r for r in recs if r.attempt_id == e["attempt_id"])
                assert e["detail"]["passed"] == rec.verdict["passed"]
