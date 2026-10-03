"""Phase 2: the oracle settles on hidden cases and cannot be read or gamed by the law.

Checks from the task: the 1/r fixture passes gravity and fails yukawa; an infinite
loop times out and fails; the same seed gives the same commitment. Plus: the
normalising variance matches the vendor, two-process scoring equals the vendor's
in-process scoring, and laws that try to read the answers or run the simulator fail.
"""

import pytest

import scienceagent.evaluator as E
from dm import oracle
from dm.oracle.cases import (MULTI_PARTICLE_WORLDS, PUBLIC_WORLDS, TWO_PARTICLE_WORLDS,
                             build_world, default_cases, norm_variance)
from dm.settle import prereg_for, settle
from dm.types import Preregistration, SubmittedAttempt

# k = 1/(2π), a ∝ -k·p1/p2·r̂/r : exact on gravity, wrong on yukawa.
ONE_OVER_R = '''
import numpy as np
def discovered_law(pos1, pos2, p1, p2, velocity2, duration):
    k, dt = 1.0 / (2.0 * np.pi), 0.005
    x = np.array(pos2, float) - np.array(pos1, float)
    v = np.array(velocity2, float)
    acc = lambda x: -k * p1 / p2 * x / (x @ x)
    a = acc(x)
    for _ in range(int(round(duration / dt))):
        v = v + 0.5 * dt * a
        x = x + dt * v
        a = acc(x)
        v = v + 0.5 * dt * a
    return (x + np.array(pos1, float)).tolist(), v.tolist()
'''

STAND_STILL = '''
def discovered_law(pos1, pos2, p1, p2, velocity2, duration):
    return list(pos2), list(velocity2)
'''

VENDOR_WORLD_VARS = {  # vendor scripts/run_benchmark.py::_WORLD_VARS
    "circle": 6.596, "coulomb_easy": 11.465, "dark_matter": 63.303, "ether": 21.259,
    "extra_dimensions": 4.248, "fractional": 5.721, "gravity": 4.283, "hubble": 41.189,
    "oscillator": 6.332, "three_species": 28.717, "yukawa": 5.677,
}


@pytest.fixture(autouse=True)
def _no_secret(monkeypatch):
    monkeypatch.delenv("DM_ORACLE_SECRET", raising=False)


def _submit(world, law, venue="discoverphysics"):
    return SubmittedAttempt(source="sim", protocol="test", venue=venue, world=world,
                            solver="fixture", seed=0, stated_p_success=None, rounds=0,
                            experiments=0, lab_cost=0.0, submitted_law=law)


# ------------------------------------------------------------------ settlement

class TestFixtureLaws:

    def test_one_over_r_passes_gravity(self):
        rec = settle(prereg_for("discoverphysics", "gravity", 0), _submit("gravity", ONE_OVER_R))
        assert rec.verdict["passed"] and rec.verdict["normalised_mse"] < 1e-4

    def test_one_over_r_fails_yukawa(self):
        rec = settle(prereg_for("discoverphysics", "yukawa", 0), _submit("yukawa", ONE_OVER_R))
        assert not rec.verdict["passed"] and rec.verdict["normalised_mse"] > 0.1

    def test_verdict_bound_to_prereg(self):
        p = prereg_for("discoverphysics", "gravity", 0)
        rec = settle(p, _submit("gravity", ONE_OVER_R))
        assert rec.verdict["prereg_commitment"] == p.commitment()

    def test_no_law_is_a_fail(self):
        rec = settle(prereg_for("discoverphysics", "gravity", 0), _submit("gravity", None))
        assert not rec.passed and rec.verdict["reason"] == "no law submitted"


class TestHostileLaws:
    """Every one of these must fail, with no answer leaking into the score."""

    def _score(self, law, world="gravity", timeout_s=120.0):
        return oracle.score(prereg_for("discoverphysics", world, 0), law, timeout_s=timeout_s)

    def test_infinite_loop_times_out(self):
        v = self._score("def discovered_law(*a, **k):\n    while True:\n        pass\n",
                        timeout_s=15)
        assert not v["passed"] and "timeout" in (v["reason"] or "")

    def test_reading_ground_truth_from_the_stack(self):
        law = '''
import sys
def discovered_law(pos1, pos2, p1, p2, velocity2, duration):
    f = sys._getframe(1)
    while f is not None and "gt" not in f.f_locals:
        f = f.f_back
    gt, case = f.f_locals["gt"], f.f_locals["case"]
    j = case["measurement_times"].index(duration)
    return gt["pos2"][j], gt["velocity2"][j]
'''
        v = self._score(law)
        assert not v["passed"] and v["normalised_mse"] > 0.1

    def test_running_the_true_simulator(self):
        law = '''
def discovered_law(pos1, pos2, p1, p2, velocity2, duration):
    from scienceagent.worlds import get_world
    ex = get_world("gravity", engine="nbody")["executor"]
    out = ex.run([{"p1": p1, "p2": p2, "pos2": list(pos2), "velocity2": list(velocity2),
                   "measurement_times": [duration]}])[0]
    return out["pos2"][-1], out["velocity2"][-1]
'''
        v = self._score(law)
        assert not v["passed"] and "not allowed" in (v["reason"] or "")

    def test_reading_vendor_source(self):
        law = '''
import scipy, os
SRC = open(os.path.join(os.path.dirname(scipy.__file__), "..", "..", "..", "..", "..",
                        "vendor", "discovery-agents", "ScienceAgent", "scienceagent",
                        "worlds.py")).read()
def discovered_law(pos1, pos2, p1, p2, velocity2, duration):
    return list(pos2), list(velocity2)
'''
        v = self._score(law)
        assert not v["passed"]

    def test_spawning_a_process(self):
        law = '''
import subprocess
def discovered_law(pos1, pos2, p1, p2, velocity2, duration):
    subprocess.run(["python", "-c", "print(1)"])
    return list(pos2), list(velocity2)
'''
        v = self._score(law)
        assert not v["passed"] and "not allowed" in (v["reason"] or "")

    def test_network(self):
        law = '''
import socket
def discovered_law(pos1, pos2, p1, p2, velocity2, duration):
    socket.create_connection(("example.com", 80), timeout=2)
    return list(pos2), list(velocity2)
'''
        v = self._score(law)
        assert not v["passed"] and "not allowed" in (v["reason"] or "")

    def test_sys_exit(self):
        law = "import sys\ndef discovered_law(*a, **k):\n    sys.exit(0)\n"
        v = self._score(law)
        assert not v["passed"]

    def test_nan(self):
        law = "def discovered_law(pos1, pos2, *a, **k):\n    return [float('nan')] * 2, [0, 0]\n"
        v = self._score(law)
        assert not v["passed"] and v["normalised_mse"] is None

    def test_no_secrets_in_the_law_process(self, monkeypatch):
        monkeypatch.setenv("DM_ORACLE_SECRET", "s3cret")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        law = '''
import os
LEAK = [k for k in os.environ if "SECRET" in k or "API_KEY" in k]
def discovered_law(pos1, pos2, p1, p2, velocity2, duration):
    if LEAK:
        raise RuntimeError("leaked " + ",".join(LEAK))
    return list(pos2), list(velocity2)
'''
        v = self._score(law)
        assert "leaked" not in (v["reason"] or "")


# ------------------------------------------------------------- equivalence

@pytest.mark.parametrize("world", ["gravity", "circle", "ether"])
def test_sandbox_scoring_equals_vendor_in_process(world):
    p = prereg_for("discoverphysics", world, 0)
    law = STAND_STILL if world in TWO_PARTICLE_WORLDS else (
        "def discovered_law(positions, velocities, duration, masses=None):\n"
        "    return positions\n")
    cls = {"circle": E.CircleEvaluator, "ether": E.EtherEvaluator}.get(world, E.Evaluator)
    vendor = cls(build_world(world)["executor"], test_cases=p.test_cases).evaluate(
        law, verbose=False)["mean_pos_error"]
    assert oracle.score(p, law)["mean_pos_error"] == pytest.approx(vendor, rel=1e-12)


# --------------------------------------------------------- preregistration

class TestPrereg:

    def test_same_seed_same_commitment(self):
        a = oracle._make_prereg("discoverphysics", "gravity", 0, "")
        oracle._make_prereg.cache_clear()
        b = oracle.make_prereg("discoverphysics", "gravity", 0)
        assert a.commitment() == b.commitment()

    def test_different_seed_different_cases(self):
        a = prereg_for("forcebench", "gravity", 0)
        b = prereg_for("forcebench", "gravity", 1)
        assert a.test_cases != b.test_cases and a.commitment() != b.commitment()

    def test_salt_from_secret(self, monkeypatch):
        monkeypatch.setenv("DM_ORACLE_SECRET", "one")
        a1, a2 = (oracle.make_prereg("forcebench", "gravity", 0) for _ in range(2))
        monkeypatch.setenv("DM_ORACLE_SECRET", "two")
        b = oracle.make_prereg("forcebench", "gravity", 0)
        assert a1.commitment() == a2.commitment()
        assert a1.salt and a1.salt != b.salt and a1.test_cases != b.test_cases

    def test_unsalted_warns(self):
        with pytest.warns(UserWarning, match="DM_ORACLE_SECRET"):
            oracle.make_prereg("forcebench", "yukawa", 3)

    def test_public_view_hides_cases_seed_and_salt(self, monkeypatch):
        monkeypatch.setenv("DM_ORACLE_SECRET", "s")
        pub = oracle.make_prereg("forcebench", "gravity", 0).public()
        assert not {"test_cases", "test_seed", "salt"} & pub.keys()

    def test_reveal_verifies(self):
        p = prereg_for("discoverphysics", "gravity", 0)
        assert Preregistration.verify(p.to_dict(), p.commitment()) == p

    @pytest.mark.parametrize("world", MULTI_PARTICLE_WORLDS)
    def test_multi_particle_cases_are_hidden_not_public(self, world):
        p = prereg_for("discoverphysics", world, 0)
        assert p.public_tests is False and p.test_cases != default_cases(world)

    def test_forcebench_rejects_multi_particle_worlds(self):
        with pytest.raises(ValueError):
            prereg_for("forcebench", "circle", 0)


@pytest.mark.parametrize("world", PUBLIC_WORLDS)
def test_norm_variance_reproduces_vendor_world_vars(world):
    assert norm_variance(world, default_cases(world)) == pytest.approx(
        VENDOR_WORLD_VARS[world], abs=5e-4)
