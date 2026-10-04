"""Pins for the oracle sandbox's current guard rails (see docs/oracle-sandbox-review.md).

These pass today; attacks that currently succeed are documented in the review, not tested here.
"""

import pytest

from dm import oracle
from dm.settle import prereg_for


@pytest.fixture(autouse=True)
def _no_secret(monkeypatch):
    monkeypatch.delenv("DM_ORACLE_SECRET", raising=False)


def _score(law, timeout_s=60.0):
    return oracle.score(prereg_for("discoverphysics", "gravity", 0), law,
                        timeout_s=timeout_s)


def test_reading_proc_is_blocked():
    law = '''
def discovered_law(pos1, pos2, p1, p2, velocity2, duration):
    open("/proc/self/environ", "rb").read()
    return list(pos2), list(velocity2)
'''
    v = _score(law)
    assert not v["passed"] and "not allowed" in (v["reason"] or "")


def test_fork_is_blocked():
    law = '''
import os
def discovered_law(pos1, pos2, p1, p2, velocity2, duration):
    os.fork()
    return list(pos2), list(velocity2)
'''
    v = _score(law)
    assert not v["passed"] and "not allowed" in (v["reason"] or "")


def test_ctypes_dlopen_is_blocked():
    law = '''
import ctypes
def discovered_law(pos1, pos2, p1, p2, velocity2, duration):
    ctypes.CDLL(None)
    return list(pos2), list(velocity2)
'''
    v = _score(law)
    assert not v["passed"] and "not allowed" in (v["reason"] or "")


def test_forged_result_line_cannot_pass():
    law = '''
import json, os
def discovered_law(pos1, pos2, p1, p2, velocity2, duration):
    os.write(1, (json.dumps({"ok": True, "preds": [[0.0, 0.0]], "fit": None,
                            "reason": None}) + "\\n").encode())
    os._exit(0)
'''
    v = _score(law)
    assert not v["passed"] and v["normalised_mse"] is None


def test_patching_numpy_dot_cannot_lower_the_score():
    patched = '''
import numpy
def discovered_law(pos1, pos2, p1, p2, velocity2, duration):
    numpy.dot = lambda a, b: 0.0
    return list(pos2), list(velocity2)
'''
    stand_still = '''
def discovered_law(pos1, pos2, p1, p2, velocity2, duration):
    return list(pos2), list(velocity2)
'''
    v_patched = _score(patched)
    v_plain = _score(stand_still)
    assert not v_patched["passed"]
    assert v_patched["mean_pos_error"] == pytest.approx(
        v_plain["mean_pos_error"], rel=1e-12)
    assert v_patched["mean_pos_error"] > 0


def test_ground_truth_on_the_stack_is_zeroed():
    law = '''
import sys
import numpy as np
def discovered_law(pos1, pos2, p1, p2, velocity2, duration):
    f = sys._getframe()
    while f is not None and "gt_pos2" not in f.f_locals:
        f = f.f_back
    gt_pos2 = f.f_locals["gt_pos2"]
    raise RuntimeError("maxabs=%r" % float(np.max(np.abs(gt_pos2))))
'''
    v = _score(law)
    assert "maxabs=0.0" in (v["reason"] or "")


def test_disabling_the_vendor_alarm_still_times_out():
    law = '''
import signal
def discovered_law(pos1, pos2, p1, p2, velocity2, duration):
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, signal.SIG_IGN)
    while True:
        pass
'''
    v = _score(law, timeout_s=8)
    assert not v["passed"] and "timeout" in (v["reason"] or "")
