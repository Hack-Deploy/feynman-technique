"""Readiness tests for the Phase 3 DiscoverPhysics venue (area: dp-venue).

Phase 3 (``dm/venues/discoverphysics.py``, ``dm/wallet.py``, ``dm/testing/fake_llm.py``)
is not built yet. These tests pin down the vendor behaviour the venue will rely on,
using throwaway spike helpers defined here (``_SpikeWallet``, ``_SpikeMeter``,
``_ScriptedLLM``). They never edit vendor files and never call a real LLM.

Tests marked ``xfail(strict=True)`` encode the behaviour Phase 3 needs but the
current scaffolding does not provide; they flip to XPASS (and fail) once fixed.
"""

from __future__ import annotations

import contextlib
import json
import math
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from dm.store import AttemptStore
from dm.types import AttemptRecord, InsufficientCredits, canonical_json

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor" / "discovery-agents"

EXP = {"p1": 1.0, "p2": 1.0, "pos2": [3.0, 0.0], "velocity2": [0.0, 0.0],
       "measurement_times": [0.5, 1.0, 2.0]}

# 1/r law with k = 1/(2π): a = -k·p1/p2·r̂/r  (the fixture from STATUS.md Phase 0).
LAW_1_OVER_R = '''
def discovered_law(pos1, pos2, p1, p2, velocity2, duration):
    import numpy as np
    from scipy.integrate import solve_ivp
    k = 1.0 / (2.0 * np.pi)
    p1x = np.asarray(pos1, dtype=float)
    def rhs(t, y):
        d = y[:2] - p1x
        r = np.hypot(d[0], d[1])
        a = -k * p1 / p2 * d / r**2
        return [y[2], y[3], a[0], a[1]]
    y0 = [pos2[0], pos2[1], velocity2[0], velocity2[1]]
    sol = solve_ivp(rhs, (0.0, duration), y0, rtol=1e-9, atol=1e-11)
    y = sol.y[:, -1]
    return np.array(y[:2]), np.array(y[2:])
'''

EXPLANATION = ("A static 2D Poisson field sourced by p1 accelerates particle 2 by -grad(phi)/p2; "
               "the force falls off as 1/r.")


def _world(seed=0, name="gravity"):
    from scienceagent.worlds import get_world
    return get_world(name, engine="nbody", noise_std=0.075, noise_seed=seed)


# --------------------------------------------------------------------- spike helpers

class _SpikeWallet:
    """Naive float wallet: the hazard demonstration, not the proposed design."""

    def __init__(self, balance: float):
        self.balance = balance
        self.lab = 0.0
        self.charges: list[tuple[int, float]] = []

    def charge(self, count: int, price: float) -> None:
        amount = count * price
        if amount > self.balance:
            raise InsufficientCredits(
                f"need {amount} credits for {count} experiment(s) at {price}, "
                f"balance {self.balance}")
        self.balance -= amount
        self.lab += amount
        self.charges.append((count, amount))


class _SpikeMeter:
    """Minimal metering wrapper: charge len(list) × price *before* delegating."""

    def __init__(self, inner, wallet: _SpikeWallet, price: float, validate: bool = True):
        self._inner = inner
        self._wallet = wallet
        self._price = price
        self._validate = validate
        self.rounds = 0
        self.delegated: list[int] = []

    def run(self, experiments):
        self.rounds += 1
        if self._validate:
            if not isinstance(experiments, list) or not all(
                    isinstance(e, dict) for e in experiments):
                raise ValueError("experiments must be a JSON list of objects")
        self._wallet.charge(len(experiments), self._price)
        self.delegated.append(len(experiments))
        return self._inner.run(experiments)

    def __getattr__(self, name):  # noise_disabled, noise_std, private executor state...
        return getattr(self._inner, name)


class _ScriptedLLM:
    """Scripted stand-in for ``scienceagent.llm_client.complete``."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def __call__(self, model, messages, system=None, max_tokens=4096):
        self.calls.append({"model": model, "messages": [dict(m) for m in messages],
                           "system": system, "max_tokens": max_tokens})
        i = len(self.calls) - 1
        return self.replies[min(i, len(self.replies) - 1)]

    @contextlib.contextmanager
    def installed(self):
        from scienceagent import llm_client
        saved = llm_client.complete
        llm_client.complete = self
        try:
            yield self
        finally:
            llm_client.complete = saved


def _one_experiment_then_law(exps=None, explanation=True):
    exps = exps if exps is not None else [EXP]
    law = f"<final_law>{LAW_1_OVER_R}</final_law>"
    if explanation:
        law += f"\n<explanation>{EXPLANATION}</explanation>"
    return [f"<run_experiment>{json.dumps(exps)}</run_experiment>", law]


def _agent(executor, mission="find the law", max_rounds=16, **kw):
    from scienceagent.agent import DiscoveryAgent
    w = _world()
    return DiscoveryAgent(model="fake-model", executor=executor, mission=mission,
                          verbose=False, system_prompt_path=w["system_prompt"],
                          instructions_path=w["instructions"], law_stub=w["law_stub"],
                          experiment_format=w["experiment_format"],
                          max_rounds=max_rounds, **kw)


# --------------------------------------------------------------------- AttemptRecord

def _full_record(**over) -> AttemptRecord:
    d = dict(attempt_id="dp-gravity-fake-0", source="live",
             protocol="discoverphysics_native", venue="discoverphysics", world="gravity",
             solver="fake-1overr", seed=0, stated_p_success=0.6, rounds=2, experiments=1,
             lab_cost=0.5,
             llm_usage={"calls": 3, "input_tokens": 1234, "output_tokens": 210,
                        "usd": 0.0, "estimated": True},
             submitted_law=LAW_1_OVER_R,
             verdict={"normalised_mse": 0.0, "passed": True, "prereg_commitment": "ab" * 32,
                      "explanation_score": None},
             transcript_path="attempts/transcripts/dp-gravity-fake-0.json",
             created_at="2026-10-03T00:00:00")
    d.update(over)
    return AttemptRecord(**d)


class TestAttemptRecordHoldsRunAttemptOutput:

    def test_all_fields_present(self):
        names = set(AttemptRecord.__dataclass_fields__)
        needed = {"submitted_law", "rounds", "experiments", "lab_cost", "llm_usage",
                  "transcript_path", "stated_p_success", "verdict", "seed", "world"}
        assert needed <= names

    def test_canonical_json_round_trip(self, tmp_path):
        rec = _full_record()
        s = canonical_json(rec.to_dict())
        assert AttemptRecord.from_dict(json.loads(s)) == rec
        store = AttemptStore(tmp_path / "a.jsonl")
        store.append([rec])
        assert store.load() == [rec]
        assert store.load()[0].llm_usage["estimated"] is True

    def test_law_with_quotes_and_newlines_survives(self, tmp_path):
        rec = _full_record(submitted_law='def f():\n    return "a\\tb" \'c\'\n')
        AttemptStore(tmp_path / "a.jsonl").append([rec])
        assert AttemptStore(tmp_path / "a.jsonl").load()[0].submitted_law == rec.submitted_law

    def test_frozen_top_level(self):
        rec = _full_record()
        with pytest.raises(FrozenInstanceError):
            rec.rounds = 3  # type: ignore[misc]

    @pytest.mark.xfail(strict=True, reason="BUG: frozen record holds mutable dicts; "
                       "verdict can be edited in place after settlement")
    def test_verdict_is_immutable(self):
        rec = _full_record(verdict={"passed": False, "normalised_mse": 3.0})
        with contextlib.suppress(TypeError):
            rec.verdict["passed"] = True
        assert rec.passed is False

    @pytest.mark.xfail(strict=True, reason="BUG: numpy scalars (np.bool_ from `nmse < 0.1` on an "
                       "np.var-normalised MSE) are accepted but not JSON-serialisable")
    def test_numpy_verdict_serialises(self, tmp_path):
        rec = _full_record(verdict={"normalised_mse": np.float64(0.01),
                                    "passed": np.bool_(True)})
        AttemptStore(tmp_path / "a.jsonl").append([rec])  # json.dumps raises on np.bool_
        canonical_json(rec.to_dict())

    @pytest.mark.xfail(strict=True, reason="BUG: a diverging law gives mean_pos_error=NaN (verified); "
                       "store writes non-standard 'Infinity' while canonical_json raises")
    def test_nonfinite_mse_consistent(self, tmp_path):
        rec = _full_record(verdict={"normalised_mse": math.inf, "passed": False})
        path = tmp_path / "a.jsonl"
        AttemptStore(path).append([rec])
        json.loads(path.read_text(), parse_constant=lambda c: (_ for _ in ()).throw(
            ValueError(c)))  # strict JSON readers (JS dashboard) reject Infinity
        canonical_json(rec.to_dict())

    @pytest.mark.xfail(strict=True, reason="GAP: from_dict silently drops unknown keys, "
                       "so a newer record read by older code loses data without warning")
    def test_unknown_keys_not_silently_dropped(self):
        d = _full_record().to_dict() | {"training_ref": "x.json"}
        with pytest.raises((TypeError, ValueError, KeyError)):
            AttemptRecord.from_dict(d)

    @pytest.mark.xfail(strict=True, reason="GAP: no validation; stated_p_success=1.7 and "
                       "negative lab_cost are accepted")
    def test_rejects_out_of_range_values(self):
        with pytest.raises((TypeError, ValueError)):
            _full_record(stated_p_success=1.7, lab_cost=-1.0)


# --------------------------------------------------------------------- metering

class TestMeteringSpike:

    def test_charges_before_delegating(self):
        wallet = _SpikeWallet(10.0)
        seen = []

        class Spy:
            def run(self, exps):
                seen.append(wallet.balance)
                return [{} for _ in exps]

        _SpikeMeter(Spy(), wallet, 0.5).run([EXP, EXP])
        assert seen == [9.0]  # already debited when the simulator ran

    def test_empty_list_charges_nothing_and_still_counts_a_round(self):
        wallet = _SpikeWallet(1.0)
        m = _SpikeMeter(_world()["executor"], wallet, 0.5)
        assert m.run([]) == []
        assert wallet.balance == 1.0 and wallet.charges == [(0, 0.0)] and m.rounds == 1

    def test_long_list_charged_in_full(self):
        wallet = _SpikeWallet(100.0)
        m = _SpikeMeter(_world()["executor"], wallet, 0.25)
        out = m.run([EXP] * 40)
        assert len(out) == 40 and wallet.balance == 90.0 and wallet.lab == 10.0

    def test_over_balance_charges_nothing_and_never_runs(self):
        wallet = _SpikeWallet(1.0)
        m = _SpikeMeter(_world()["executor"], wallet, 0.5)
        with pytest.raises(InsufficientCredits):
            m.run([EXP] * 3)
        assert wallet.balance == 1.0 and wallet.lab == 0.0 and m.delegated == []

    def test_exact_balance_integer_price(self):
        wallet = _SpikeWallet(1.0)
        _SpikeMeter(_world()["executor"], wallet, 0.5).run([EXP, EXP])
        assert wallet.balance == 0.0

    @pytest.mark.xfail(strict=True, reason="HAZARD: float credits; 3 × 0.1 = "
                       "0.30000000000000004 > 0.3, so an exactly affordable batch is refused")
    def test_fractional_price_exact_balance(self):
        wallet = _SpikeWallet(0.3)
        _SpikeMeter(_world()["executor"], wallet, 0.1).run([EXP] * 3)

    def test_fractional_price_conservation_drift(self):
        """1000 charges of 0.1: lab ends at 99.9999999999986, not 100. The sum is still
        within the engine's 1e-9 tolerance, but individual balances drift."""
        wallet = _SpikeWallet(100.0)
        for _ in range(1000):
            wallet.charge(1, 0.1)
        assert abs(wallet.balance + wallet.lab - 100.0) < 1e-9
        assert wallet.lab != 100.0 and wallet.balance != 0.0

    def test_non_list_input_overcharges_without_validation(self):
        """`json.loads` of the agent's block may be a dict; len(dict) = number of keys."""
        wallet = _SpikeWallet(10.0)
        m = _SpikeMeter(_world()["executor"], wallet, 1.0, validate=False)
        with pytest.raises(TypeError):
            m.run(dict(EXP))  # vendor iterates keys → "string indices must be integers"
        assert wallet.lab == 5.0  # charged 5 credits for zero experiments

    def test_malformed_item_is_charged_then_fails(self):
        """Missing keys raise inside the vendor *after* the charge (KeyError 'p2')."""
        wallet = _SpikeWallet(10.0)
        m = _SpikeMeter(_world()["executor"], wallet, 1.0)
        with pytest.raises(KeyError):
            m.run([EXP, {"p1": 1.0}])
        assert wallet.lab == 2.0  # the valid one's data is lost too

    def test_wrapper_forwards_noise_controls(self):
        m = _SpikeMeter(_world()["executor"], _SpikeWallet(10.0), 1.0)
        assert m.noise_std == 0.075
        with m.noise_disabled():
            assert m.noise_std == 0.0
        assert m.noise_std == 0.075

    def test_evaluator_on_metered_executor_charges_solver(self):
        """HAZARD: reusing the venue executor for scoring charges the test cases to the
        solver and routes hidden cases through the solver-side object. The oracle must
        build its own executor."""
        from scienceagent.evaluator import Evaluator
        wallet = _SpikeWallet(100.0)
        m = _SpikeMeter(_world()["executor"], wallet, 1.0)
        Evaluator(m).evaluate(LAW_1_OVER_R, verbose=False)
        assert wallet.lab == 3.0  # gravity's 3 default (public) test cases


# --------------------------------------------------------------------- agent loop offline

class TestAgentLoopOffline:

    def test_patching_module_attribute_intercepts_agent(self):
        llm = _ScriptedLLM(_one_experiment_then_law())
        wallet = _SpikeWallet(10.0)
        with llm.installed():
            law = _agent(_SpikeMeter(_world()["executor"], wallet, 0.5)).run()
        assert law is not None and "discovered_law" in law
        assert len(llm.calls) == 2
        from scienceagent import llm_client
        assert llm_client.complete is not llm  # restored

    def test_full_fake_attempt_passes_and_charges_exactly(self, tmp_path):
        from scienceagent.evaluator import Evaluator, _extract_training_trajectories
        llm = _ScriptedLLM(_one_experiment_then_law(exps=[EXP, dict(EXP, p1=2.0)]))
        wallet = _SpikeWallet(10.0)
        meter = _SpikeMeter(_world(seed=3)["executor"], wallet, 0.5)
        with llm.installed():
            agent = _agent(meter)
            law = agent.run()
        n_exp = sum(len(e["experiment_input"]) for e in agent.conversation_log
                    if e["action"] == "experiment" and e["experiment_output"] is not None)
        assert n_exp == 2 and wallet.lab == 2 * 0.5 and wallet.balance + wallet.lab == 10.0
        assert len(agent.conversation_log) == 2 and meter.rounds == 1
        assert agent.discovered_explanation == EXPLANATION
        training = _extract_training_trajectories(agent.conversation_log)
        assert len(training) == 2
        fresh = _world(seed=3)["executor"]  # oracle-side, never the metered one
        res = Evaluator(fresh).evaluate(law, verbose=False, training_trajectories=training)
        assert res["mean_pos_error"] < 1e-6
        # transcript: our own file at a path we choose
        path = tmp_path / "t" / "attempt.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"mission": agent.mission, "system": agent._system,
                                    "rounds": agent.conversation_log, "final_law": law,
                                    "explanation": agent.discovered_explanation}))
        assert json.loads(path.read_text())["rounds"][0]["action"] == "experiment"

    def test_insufficient_credits_shown_to_agent_and_agent_submits(self):
        llm = _ScriptedLLM(_one_experiment_then_law(exps=[EXP] * 3))
        wallet = _SpikeWallet(1.0)
        with llm.installed():
            agent = _agent(_SpikeMeter(_world()["executor"], wallet, 0.5))
            law = agent.run()
        assert law is not None  # attempt still ends with a law to settle
        assert wallet.lab == 0.0 and wallet.balance == 1.0
        r1 = agent.conversation_log[0]
        assert r1["action"] == "experiment" and r1["experiment_output"] is None
        assert "need 1.5 credits" in r1["experiment_error"]
        shown = llm.calls[1]["messages"][-1]["content"]
        assert shown.startswith("<experiment_output>\nError running experiment: need 1.5")
        from scienceagent.evaluator import _extract_training_trajectories
        assert _extract_training_trajectories(agent.conversation_log) == []

    def test_insufficient_credits_then_smaller_batch(self):
        replies = [f"<run_experiment>{json.dumps([EXP] * 3)}</run_experiment>",
                   f"<run_experiment>{json.dumps([EXP] * 2)}</run_experiment>",
                   _one_experiment_then_law()[1]]
        llm = _ScriptedLLM(replies)
        wallet = _SpikeWallet(1.0)
        with llm.installed():
            agent = _agent(_SpikeMeter(_world()["executor"], wallet, 0.5))
            assert agent.run() is not None
        assert wallet.balance == 0.0 and wallet.charges == [(2, 1.0)]

    def test_min_rounds_rejects_round_one_submission(self):
        llm = _ScriptedLLM([_one_experiment_then_law()[1]])
        with llm.installed():
            agent = _agent(_world()["executor"])
            assert agent.run() is not None
        assert agent.conversation_log[0]["action"] == "warning"
        assert len(llm.calls) == 2  # round 1 refused, round 2 accepted

    def test_missing_explanation_costs_an_extra_llm_call(self):
        llm = _ScriptedLLM(_one_experiment_then_law(explanation=False)
                           + [f"<explanation>{EXPLANATION}</explanation>"])
        with llm.installed():
            agent = _agent(_world()["executor"])
            agent.run()
        assert len(llm.calls) == 3  # usage accounting must include the follow-up
        assert len(agent.conversation_log) == 2  # ... but rounds stay 2

    def test_max_rounds_16_no_submission_returns_none(self):
        llm = _ScriptedLLM([f"<run_experiment>{json.dumps([EXP])}</run_experiment>"])
        wallet = _SpikeWallet(100.0)
        with llm.installed():
            agent = _agent(_SpikeMeter(_world()["executor"], wallet, 0.5), max_rounds=16)
            assert agent.run() is None  # venue must still produce a failing verdict
        assert len(llm.calls) == 16 and len(agent.conversation_log) == 16
        assert wallet.lab == 16 * 0.5  # round 16 still runs the experiment it was told not to
        msgs = llm.calls[14]["messages"]
        assert any("round 15 of 16" in m["content"] for m in msgs)
        assert "EXACTLY 16 round(s)" in llm.calls[0]["system"]

    def test_usage_not_exposed_by_vendor_client(self):
        import inspect
        from scienceagent import llm_client
        assert inspect.signature(llm_client.complete).return_annotation is str
        src = inspect.getsource(llm_client)
        assert "usage" not in src  # every provider path returns text only

    def test_critic_and_judge_route_through_same_function(self):
        import inspect
        from scienceagent import critic, evaluator
        assert "llm_client.complete(" in inspect.getsource(critic)
        assert "llm_client.complete(" in inspect.getsource(evaluator.ExplanationJudge)

    def test_agent_call_carries_no_attempt_context(self):
        """The patched function only sees (model, messages, system, max_tokens): to
        attribute usage per attempt the adapter must be installed per attempt."""
        llm = _ScriptedLLM(_one_experiment_then_law())
        with llm.installed():
            _agent(_world()["executor"]).run()
        assert set(llm.calls[0]) == {"model", "messages", "system", "max_tokens"}


# --------------------------------------------------------------------- mission / market_aware

class TestMarketAwareMission:

    def test_mission_is_first_user_message_and_note_appends(self):
        mission = _world()["mission"]
        note = ("\n\nMARKET TERMS: each experiment costs 0.5 credits, charged before it runs. "
                "Your balance is 10 credits; the prize is 60 credits. If you cannot pay, the "
                "experiment returns an error and you must submit your <final_law>.")
        calls = {}
        for aware in (False, True):
            llm = _ScriptedLLM(_one_experiment_then_law())
            with llm.installed():
                _agent(_world()["executor"], mission=mission + (note if aware else "")).run()
            calls[aware] = llm.calls[0]
        assert calls[False]["system"] == calls[True]["system"]  # protocol-identical system
        assert calls[False]["messages"][0]["content"] == mission
        assert calls[True]["messages"][0]["content"] == mission + note

    def test_mission_is_identical_across_seeds(self):
        assert _world(0)["mission"] == _world(7)["mission"]


# --------------------------------------------------------------------- determinism

def _traj(seed, batches):
    ex = _world(seed)["executor"]
    out = []
    for b in batches:
        out += ex.run(b)
    return out


class TestDeterminism:

    def test_same_seed_same_noise(self):
        assert _traj(5, [[EXP]]) == _traj(5, [[EXP]])

    def test_different_seed_different_noise(self):
        assert _traj(5, [[EXP]])[0]["pos2"] != _traj(6, [[EXP]])[0]["pos2"]

    def test_noise_stream_is_sequential_not_per_experiment(self):
        """Batching does not matter, but order does: noise depends on call history."""
        a, b = EXP, dict(EXP, p1=2.0)
        assert _traj(1, [[a, b]]) == _traj(1, [[a], [b]])
        assert _traj(1, [[a, b]])[1] != _traj(1, [[b]])[0]

    def test_noise_seed_none_is_not_reproducible(self):
        from scienceagent.worlds import get_world
        r = [get_world("gravity", engine="nbody", noise_std=0.075,
                       noise_seed=None)["executor"].run([EXP])[0]["pos2"] for _ in range(2)]
        assert r[0] != r[1]  # run_attempt must always pass an explicit seed

    def test_velocities_unnoised(self):
        assert _traj(1, [[EXP]])[0]["velocity2"] == _traj(2, [[EXP]])[0]["velocity2"]

    def test_bit_identical_across_processes(self):
        code = ("import json;from scienceagent.worlds import get_world;"
                "e=get_world('gravity',engine='nbody',noise_std=0.075,noise_seed=11)['executor'];"
                f"print(json.dumps(e.run([{EXP!r}])))")
        outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                               check=True).stdout for _ in range(2)}
        assert len(outs) == 1 and json.loads(outs.pop()) == _traj(11, [[EXP]])


# --------------------------------------------------------------------- transcript / logger

class TestTranscriptAndTrajectoryLog:

    def test_trajectory_logger_path_is_ours_and_enables_mse_fit(self, tmp_path):
        from scienceagent.trajectory_logger import TrajectoryLogger
        ex = _SpikeMeter(_world()["executor"], _SpikeWallet(10.0), 0.5)
        logger = TrajectoryLogger(world="gravity", executor=ex,
                                  csv_path=tmp_path / "traj.csv", run_id="attempt-0")
        llm = _ScriptedLLM(_one_experiment_then_law())
        with llm.installed():
            agent = _agent(ex, trajectory_logger=logger)
            agent.run()
        assert "<run_mse_fit>" in agent._system  # tool advertised only with a logger
        rows = (tmp_path / "traj.csv").read_text().splitlines()
        assert rows[0].startswith("run_id,") and len(rows) == 1 + 2 + 2 * 3
        assert not (VENDOR / "results").exists()

    def test_no_logger_means_mse_tool_absent(self):
        assert "<run_mse_fit>" not in _agent(_world()["executor"])._system

    def test_conversation_log_is_strict_json(self):
        llm = _ScriptedLLM(_one_experiment_then_law())
        with llm.installed():
            agent = _agent(_world()["executor"])
            agent.run()
        json.dumps(agent.conversation_log, allow_nan=False)

    def test_vendor_tree_clean(self):
        out = subprocess.run(["git", "-C", str(VENDOR), "status", "--porcelain"],
                             capture_output=True, text=True, check=True).stdout
        assert out.strip() == ""
