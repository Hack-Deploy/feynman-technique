from __future__ import annotations

import json
import pickle
from dataclasses import asdict

import pytest

from dm.store import AttemptStore, FIXTURES_DIR
from dm.types import (
    AttemptRecord,
    InsufficientCredits,
    SubmittedAttempt,
    canonical_json,
)


def _submitted_attempt(**overrides) -> SubmittedAttempt:
    values = {
        "venue": "discoverphysics",
        "world": "gravity",
        "solver": "solver-a",
        "seed": 4,
        "protocol": "discoverphysics_native",
        "source": "live",
        "stated_p_success": 0.6,
        "rounds": 2,
        "experiments": 3,
        "lab_cost": 1.5,
        "llm_usage": {"calls": 2},
        "submitted_law": "def discovered_law(): pass",
        "explanation": "A law.",
    }
    values.update(overrides)
    return SubmittedAttempt(**values)


def _record() -> AttemptRecord:
    return AttemptRecord(
        attempt_id="strict-json",
        source="sim",
        protocol="test",
        venue="fixture",
        world="gravity",
        solver="solver-a",
        seed=0,
        stated_p_success=0.5,
        rounds=1,
        experiments=1,
        lab_cost=0.25,
    )


def test_submitted_attempt_to_dict_json_round_trip():
    attempt = _submitted_attempt(
        training=[{"experiment": {"p1": 1.0}}],
        extra={"terms": {"price": 0.5, "market_aware": True}},
    )
    d = json.loads(canonical_json(asdict(attempt)))
    assert SubmittedAttempt(**d) == attempt


def test_insufficient_credits_attributes_message_and_pickle():
    error = InsufficientCredits(needed=1.5, balance=1.0, count=3, price=0.5)
    expected = (
        "insufficient credits: 3 experiment(s) x 0.5 = 1.5 credits needed, balance 1.0. "
        "Submit your <final_law> or run fewer experiments."
    )
    assert (error.needed, error.balance, error.count, error.price) == (1.5, 1.0, 3, 0.5)
    assert str(error) == expected

    restored = pickle.loads(pickle.dumps(error))
    assert isinstance(restored, InsufficientCredits)
    assert (restored.needed, restored.balance, restored.count, restored.price) == (
        1.5, 1.0, 3, 0.5
    )
    assert str(restored) == expected


def test_store_writes_strict_json_lines(tmp_path):
    path = tmp_path / "attempts.jsonl"
    AttemptStore(path).append([_record()])

    def reject_constant(value):
        raise ValueError(f"invalid JSON constant: {value}")

    lines = path.read_text().splitlines()
    assert len(lines) == 1
    assert isinstance(json.loads(lines[0], parse_constant=reject_constant), dict)


def test_replay_pool_fixture_loads_eight_records():
    records = AttemptStore(FIXTURES_DIR / "replay_pool.jsonl").load()
    assert len(records) == 8
