import json
from pathlib import Path

import pytest

from dm import replay
from dm.importers.forcebench import SETTLE_PATHS, default_settle_path, records_from_settle
from dm.types import AttemptRecord

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "attempts" / "fixtures" / "demo" / "forcebench_settle.json"


def _ara_record():
    return AttemptRecord(
        attempt_id="ara-record",
        source="published_replay",
        protocol="ara_harness",
        venue="discoverphysics",
        world="gravity",
        solver="ara-solver",
        seed=0,
        stated_p_success=None,
        rounds=2,
        experiments=3,
        lab_cost=2.0,
        verdict={"passed": True},
    )


def test_settle_snapshot_loads_forcebench_records():
    records = records_from_settle(SNAPSHOT)
    assert len(records) == 60
    assert {record.venue for record in records} == {"forcebench"}
    assert all(record.lab_cost == record.experiments for record in records)
    assert all(record.rounds == record.experiments for record in records)
    assert all(record.extra["settle_file"] == "attempts/fixtures/demo/forcebench_settle.json"
               for record in records)


def test_generated_settle_output_precedes_snapshot_and_snapshot_is_fallback(monkeypatch, tmp_path):
    generated, snapshot = tmp_path / "generated.json", tmp_path / "snapshot.json"
    generated.touch()
    monkeypatch.setattr("dm.importers.forcebench.SETTLE_PATHS", (generated, snapshot))
    assert default_settle_path() == generated
    generated.unlink()
    assert default_settle_path() == snapshot
    assert SETTLE_PATHS[0].name == "forcebench_settle.json"


def test_forcebench_sweep_is_deterministic_conserving_and_draws_once():
    records = records_from_settle(SNAPSHOT)
    first = replay.run_sweep(records, "fbtest", prizes=[5, 20], seeds=[0, 1],
                             pool="forcebench")
    second = replay.run_sweep(records, "fbtest", prizes=[5, 20], seeds=[0, 1],
                              pool="forcebench")
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert replay.max_draws_per_record(first[0]) == 1
    by_id = {record.attempt_id: record for record in records}
    charges = [event for event in first[0] if event["type"] == "experiment_charged"]
    assert charges
    for event in charges:
        assert event["count"] == by_id[event["attempt_id"]].experiments
    from analysis import compute_final_balances
    for run_events in replay._by_run(first[0]).values():
        assert abs(sum(compute_final_balances(run_events).values())) < 1e-9


def test_venues_cannot_mix_or_use_wrong_pool():
    forcebench = records_from_settle(SNAPSHOT)
    with pytest.raises(ValueError, match="cannot mix venues"):
        replay.run_sweep([_ara_record(), forcebench[0]], "mixed", pool="forcebench")
    with pytest.raises(ValueError, match="requires venue"):
        replay.run_sweep(forcebench, "wrong", pool="ara")


def test_run_and_save_forcebench_summary_and_ara_hypothesis_unavailability(tmp_path):
    summary = replay.run_and_save(
        "forcebench", out_dir=tmp_path / "forcebench", store_path=SNAPSHOT)
    written = json.loads((tmp_path / "forcebench" / "summary.json").read_text())
    assert set(written["hypotheses"]) == {"H1", "H2", "H3", "H4"}
    assert "calibration" in written
    assert summary["checks"]["credit_sum_per_run_max_abs"] == 0

    record = _ara_record()
    events, ledger = replay.run_sweep([record], "aramin", prizes=[20], seeds=[0])
    ara_summary = replay.summarise(events, ledger, [record], "aramin",
                                   prizes=[20], seeds=[0])
    assert ara_summary["hypotheses"]["H3"]["verdict"] == "UNAVAILABLE"
    assert ara_summary["hypotheses"]["H4"]["verdict"] == "UNAVAILABLE"


def test_default_settle_path_prefers_existing_generated_output():
    assert default_settle_path() in SETTLE_PATHS
