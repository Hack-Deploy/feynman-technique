"""Experiment replays: both executor output shapes normalise to frames x particles x [x, y]."""

import animations


def test_two_body_output_becomes_two_particles():
    out = {"pos1": [[0, 0], [0, 0]], "pos2": [[2, 0], [1.9, 0]]}
    assert animations._particles(out) == [[[0, 0], [2, 0]], [[0, 0], [1.9, 0]]]


def test_multi_particle_output_passes_through():
    out = {"positions": [[[0, 0], [1, 1], [2, 2]]]}
    assert animations._particles(out) == out["positions"]


def test_probe_counts():
    assert animations._probe_count({"probe_positions": [[1, 0], [2, 0]]}, 32) == 2
    assert animations._probe_count({"pos2": [2, 0]}, 2) == 1
    assert animations._probe_count({"ring_radius": 3}, 10) == 10


def test_replay_skips_experiments_without_times():
    assert animations._replay(executor=None, inp={"measurement_times": []}, out=None) is None
