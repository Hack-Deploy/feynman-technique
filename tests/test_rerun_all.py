from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from dm.store import AttemptStore
from poc import archive, bench, config as C, live_cache, rerun_all, spend
from poc.spend import SpendLedger


@pytest.fixture(autouse=True)
def isolated_reruns(monkeypatch, tmp_path):
    for name in ("ANTHROPIC_API_KEY", "ENABLE_LIVE", "DM_MAX_USD", "HF_TOKEN", "DM_HF_REPO"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(bench, "load_env", lambda: None)
    monkeypatch.setattr(C, "ROOT", tmp_path)
    monkeypatch.setattr(C, "ATTEMPTS_PATH", tmp_path / "attempts.jsonl")
    monkeypatch.setattr(C, "TRANSCRIPTS_DIR", tmp_path / "transcripts")
    monkeypatch.setattr(C, "TRAJECTORIES_DIR", tmp_path / "trajectories")
    monkeypatch.setattr(live_cache, "RUNS_PATH", tmp_path / "runs.jsonl")
    monkeypatch.setattr(live_cache, "SCRIPTED_PATH", tmp_path / "scripted_demo.jsonl")
    monkeypatch.setattr(spend, "LEDGER_PATH", tmp_path / "live_spend.jsonl")


def _configuration():
    cfg = C.load()
    settings = replace(
        spend.load_settings(),
        hypotheses=tuple(hyp.id for hyp in cfg.hypotheses),
        seeds=(0,),
    )
    return cfg, settings


def _fake_configuration(monkeypatch):
    cfg = C.load()
    cfg = replace(cfg, hypotheses=cfg.hypotheses[:2])
    settings = replace(
        spend.load_settings(),
        models=spend.load_settings().models[:2],
        hypotheses=tuple(hyp.id for hyp in cfg.hypotheses),
        seeds=(0,),
    )
    monkeypatch.setattr(C, "load", lambda: cfg)
    monkeypatch.setattr(spend, "load_settings", lambda: settings)
    return cfg, settings


def test_order_is_deterministic_and_each_claim_gets_every_model():
    cfg, settings = _configuration()
    first = rerun_all._order(settings, cfg, 17)
    second = rerun_all._order(settings, cfg, 17)

    assert first == second
    _, claim_order, _ = first
    model_ids = {model.id for model in settings.models}
    assert all(set(models) == model_ids and len(models) == len(model_ids)
               for models in claim_order.values())


def test_every_model_is_first_for_some_claim_across_seed_range():
    cfg, settings = _configuration()
    first_models = set()

    for seed in range(50):
        _, claim_order, _ = rerun_all._order(settings, cfg, seed)
        first_models.update(models[0] for models in claim_order.values())

    assert first_models == {model.id for model in settings.models}


def test_order_runs_position_major():
    cfg, settings = _configuration()
    order, claim_order, _ = rerun_all._order(settings, cfg, 0)
    claims = [hyp.id for hyp in cfg.hypotheses]
    model_count = len(settings.models)
    expected = [
        (claim, claim_order[claim][position], 0)
        for position in range(model_count)
        for claim in claims
    ]

    assert order == expected


def test_cached_order_seed_resolution_rejects_mismatch_and_conflicts():
    entries = [{"order_seed": 3}, {"order_seed": 3}]
    assert rerun_all._resolve_order_seed(entries, 3) == 3
    assert rerun_all._resolve_order_seed(entries, 4, reuse_cached=False) == 4

    with pytest.raises(ValueError, match="cache was run with order seed 3"):
        rerun_all._resolve_order_seed(entries, 4)
    with pytest.raises(ValueError, match="conflicting order seeds"):
        rerun_all._resolve_order_seed([{"order_seed": 3}, {"order_seed": 4}], 3)


def _make_live_files(monkeypatch, root):
    runs = root / "attempts" / "fixtures" / "live" / "runs.jsonl"
    store = root / "attempts" / "poc_dp_bench.jsonl"
    transcripts = root / "attempts" / "transcripts" / "poc_dp"
    ledger = root / "attempts" / "live_spend.jsonl"
    scripted = root / "attempts" / "fixtures" / "live" / "scripted_demo.jsonl"
    ara_cache = root / "attempts" / "cache" / "ara.jsonl"
    old_archive = root / "attempts" / "archive" / "old" / "keep.jsonl"
    runs.parent.mkdir(parents=True)
    runs.write_text(json.dumps({
        "record": {"attempt_id": "existing"},
        "key": {"model": "model", "hypothesis_id": "claim", "seed": 0},
        "order_seed": 3,
        "order_position": 0,
    }) + "\n")
    live_cache.summary_path(runs).write_text("summary")
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text("store")
    transcripts.mkdir(parents=True)
    (transcripts / "transcript.json").write_text("transcript")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text('{"type":"reserve","id":"protected","upper_usd":0.25}\n')
    (root / "attempts" / "live_spend.jsonl.lock").write_text("lock")
    scripted.write_text("scripted")
    ara_cache.parent.mkdir(parents=True)
    ara_cache.write_text("ara")
    old_archive.parent.mkdir(parents=True)
    old_archive.write_text("archive")
    monkeypatch.setattr(live_cache, "RUNS_PATH", runs)
    monkeypatch.setattr(live_cache, "SCRIPTED_PATH", scripted)
    monkeypatch.setattr(C, "ATTEMPTS_PATH", store)
    monkeypatch.setattr(C, "TRANSCRIPTS_DIR", transcripts)
    monkeypatch.setattr(spend, "LEDGER_PATH", ledger)
    return runs, store, transcripts, ledger, scripted, ara_cache, old_archive


def test_purge_archives_live_files_and_leaves_protected_data(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "ROOT", tmp_path)
    runs, store, transcripts, ledger, scripted, ara_cache, old_archive = _make_live_files(
        monkeypatch, tmp_path
    )

    archived = rerun_all._archive(runs, store, transcripts, out=lambda _: None)

    assert len(archived) == 4
    archive_root = tmp_path / "attempts" / "archive"
    archived_root = next(path for path in archive_root.iterdir() if path.name != "old")
    for relative in (
        "attempts/fixtures/live/runs.jsonl",
        "attempts/fixtures/live/runs.summary.json",
        "attempts/poc_dp_bench.jsonl",
        "attempts/transcripts/poc_dp/transcript.json",
    ):
        assert (archived_root / relative).exists()
    assert not runs.exists()
    assert not live_cache.summary_path(runs).exists()
    assert not store.exists()
    assert not transcripts.exists()
    assert ledger.read_text() == '{"type":"reserve","id":"protected","upper_usd":0.25}\n'
    assert (tmp_path / "attempts" / "live_spend.jsonl.lock").read_text() == "lock"
    assert scripted.read_text() == "scripted"
    assert ara_cache.read_text() == "ara"
    assert old_archive.read_text() == "archive"


def test_archive_paths_retries_same_timestamp_with_suffix(monkeypatch, tmp_path):
    fixed = datetime(2025, 1, 2, 3, 4, 5, 6789, tzinfo=timezone.utc)

    class FrozenClock:
        @classmethod
        def now(cls, _timezone):
            return fixed

    monkeypatch.setattr(archive, "datetime", FrozenClock)
    first = tmp_path / "attempts" / "poc_dp_bench.jsonl"
    second = tmp_path / "attempts" / "transcripts" / "poc_dp" / "one.json"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("first")
    second.write_text("second")

    first_archived = archive.archive_paths([first], tmp_path, out=lambda _line: None)
    second_archived = archive.archive_paths([second], tmp_path, out=lambda _line: None)

    assert first_archived[0][1].read_text() == "first"
    assert second_archived[0][1].read_text() == "second"
    assert first_archived[0][1].parents[1].name == "20250102T030405006789Z"
    assert second_archived[0][1].parents[3].name == "20250102T030405006789Z-1"


def test_declined_purge_leaves_files_in_place(monkeypatch, tmp_path, capsys):
    runs, store, transcripts, ledger, scripted, ara_cache, old_archive = _make_live_files(
        monkeypatch, tmp_path
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    rerun_all.main([
        "--fake",
        "--purge",
        "--order-seed", "3",
        "--cache", str(runs),
        "--store", str(store),
        "--transcripts", str(transcripts),
    ])

    assert json.loads(runs.read_text())["order_seed"] == 3
    assert store.read_text() == "store"
    assert (transcripts / "transcript.json").read_text() == "transcript"
    archive_root = tmp_path / "attempts" / "archive"
    assert list(archive_root.iterdir()) == [old_archive.parent]
    assert ledger.read_text() == '{"type":"reserve","id":"protected","upper_usd":0.25}\n'
    assert scripted.read_text() == "scripted"
    assert ara_cache.read_text() == "ara"
    assert old_archive.read_text() == "archive"
    assert "Declined" in capsys.readouterr().out


def test_live_gate_refusal_does_not_purge(monkeypatch, tmp_path):
    runs, store, transcripts, *rest = _make_live_files(monkeypatch, tmp_path)
    old_archive = rest[-1]

    with pytest.raises(SystemExit, match="ENABLE_LIVE"):
        rerun_all.main([
            "--purge",
            "--order-seed", "3",
            "--cache", str(runs),
            "--store", str(store),
            "--transcripts", str(transcripts),
        ])

    assert json.loads(runs.read_text())["order_seed"] == 3
    assert store.read_text() == "store"
    assert (transcripts / "transcript.json").read_text() == "transcript"
    archive_root = tmp_path / "attempts" / "archive"
    assert list(archive_root.iterdir()) == [old_archive.parent]


def test_order_seed_mismatch_refuses_before_archiving_or_running(
    monkeypatch, tmp_path, capsys
):
    runs, store, transcripts, ledger, scripted, ara_cache, old_archive = _make_live_files(
        monkeypatch, tmp_path
    )

    with pytest.raises(SystemExit) as exc_info:
        rerun_all.main([
            "--fake",
            "--order-seed", "4",
            "--cache", str(runs),
            "--store", str(store),
            "--transcripts", str(transcripts),
    ])

    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert (
        "cache was run with order seed 3; use --purge to start a new order or omit "
        "--order-seed"
    ) in error
    assert json.loads(runs.read_text())["order_seed"] == 3
    assert store.read_text() == "store"
    assert (transcripts / "transcript.json").read_text() == "transcript"
    assert list((tmp_path / "attempts" / "archive").iterdir()) == [old_archive.parent]
    assert ledger.exists() and scripted.exists() and ara_cache.exists()


def test_purge_allows_a_new_order_seed(monkeypatch, tmp_path):
    runs, store, transcripts, _ledger, _scripted, _ara_cache, old_archive = _make_live_files(
        monkeypatch, tmp_path
    )
    cfg, settings = _fake_configuration(monkeypatch)

    rerun_all.main([
        "--fake",
        "--purge",
        "--yes",
        "--order-seed", "4",
        "--cache", str(runs),
        "--store", str(store),
        "--transcripts", str(transcripts),
    ])

    entries = live_cache.load(runs)
    assert len(entries) == len(cfg.hypotheses) * len(settings.models)
    assert {entry["order_seed"] for entry in entries} == {4}
    assert len(AttemptStore(store).load()) == len(entries)
    archives = list((tmp_path / "attempts" / "archive").iterdir())
    archives.remove(old_archive.parent)
    assert len(archives) == 1
    assert (archives[0] / "attempts" / "fixtures" / "live" / "runs.jsonl").exists()


def test_fake_rerun_writes_order_metadata_and_conserves_credits(
    monkeypatch, tmp_path, capsys
):
    cfg, settings = _fake_configuration(monkeypatch)
    cache = tmp_path / "runs.jsonl"
    store = tmp_path / "attempts.jsonl"
    transcripts = tmp_path / "transcripts"
    args = [
        "--fake",
        "--order-seed", "23",
        "--cache", str(cache),
        "--store", str(store),
        "--transcripts", str(transcripts),
    ]

    rerun_all.main(args)

    entries = live_cache.load(cache)
    assert len(entries) == len(cfg.hypotheses) * len(settings.models)
    assert {entry["order_seed"] for entry in entries} == {23}
    assert {entry["order_position"] for entry in entries} == set(range(len(settings.models)))
    assert all(
        round_entry["cut_off"] is False
        for entry in entries
        for round_entry in entry["rounds"]
    )
    for entry in entries:
        record = entry["record"]
        charges = sum(
            event["amount"]
            for event in record["extra"]["account_events"]
            if event["type"].endswith("_charged")
        )
        assert record["lab_cost"] == pytest.approx(charges)
    assert len(AttemptStore(store).load()) == len(entries)

    summary = json.loads(live_cache.summary_path(cache).read_text())
    assert summary["order_seed"] == 23
    assert set(summary["claim_order"]) == {hyp.id for hyp in cfg.hypotheses}

    monkeypatch.setattr(
        rerun_all.secrets,
        "randbelow",
        lambda _limit: pytest.fail("cached order seed should be reused"),
    )
    rerun_all.main([
        "--fake",
        "--cache", str(cache),
        "--store", str(store),
        "--transcripts", str(transcripts),
    ])

    assert len(live_cache.load(cache)) == len(entries)
    assert len(AttemptStore(store).load()) == len(entries)
    assert json.loads(live_cache.summary_path(cache).read_text())["order_seed"] == 23
    assert "No runs to do" in capsys.readouterr().out
