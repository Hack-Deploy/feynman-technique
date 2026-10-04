"""Offline coverage for multi-model live pricing, caching, and recorded-run APIs."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path

import pytest
import scienceagent.llm_client
import yaml

import app
import live_market
from dm.types import AttemptRecord, SubmittedAttempt
from poc import bench, config as C, demo_grid, fake_llm, live_cache, report, spend
from poc.config import ROOT as REPO_ROOT
from poc.llm import MeteredLLM, redact
from poc.spend import CapReached, ModelPrice, SpendLedger

_LOAD_ENV = bench.load_env


@pytest.fixture(autouse=True)
def isolated_live_files(monkeypatch, tmp_path):
    for name in ("ANTHROPIC_API_KEY", "ENABLE_LIVE", "DM_MAX_USD", "DM_LIVE_MODELS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(bench, "load_env", lambda: None)
    monkeypatch.setattr(
        scienceagent.llm_client,
        "complete",
        lambda **kwargs: pytest.fail("tests must never call a paid API"),
    )
    monkeypatch.setattr(
        "poc.llm.anthropic_transport",
        lambda *args, **kwargs: pytest.fail("tests must not call Anthropic"),
    )
    monkeypatch.setattr(C, "ROOT", tmp_path)
    monkeypatch.setattr(C, "ATTEMPTS_PATH", tmp_path / "attempts.jsonl")
    monkeypatch.setattr(C, "TRANSCRIPTS_DIR", tmp_path / "transcripts")
    monkeypatch.setattr(C, "TRAJECTORIES_DIR", tmp_path / "trajectories")
    monkeypatch.setattr(live_market, "DEMO_PATH", tmp_path / "demo.jsonl")
    monkeypatch.setattr(live_market, "_JOBS", {})
    monkeypatch.setattr(live_market, "_ACTIVE_JOB", None)
    monkeypatch.setattr(live_market, "_PROMPT_CHARS_CACHE", {})
    monkeypatch.setattr(spend, "LEDGER_PATH", tmp_path / "live_spend.jsonl")
    monkeypatch.setattr(live_cache, "RUNS_PATH", tmp_path / "runs.jsonl")
    monkeypatch.setattr(live_cache, "SCRIPTED_PATH", tmp_path / "scripted.jsonl")


@pytest.fixture
def app_server():
    server = app.ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _request(url):
    try:
        with urllib.request.urlopen(url) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def _resolved_entry(model: str, source: str, hyp_id: str, seed: int = 0) -> dict:
    cfg = C.load()
    hyp = cfg.hypothesis(hyp_id)
    record = AttemptRecord(
        attempt_id=f"{source}-{model}-{hyp_id}-{seed}",
        source=source,
        protocol=C.PROTOCOL,
        venue=C.VENUE,
        world=hyp.world,
        solver=f"scripted:{model}" if source == "scripted" else model,
        seed=seed,
        stated_p_success=0.7,
        rounds=1,
        experiments=0,
        lab_cost=10,
        llm_usage={"calls": 1, "usd": 0.001},
        verdict={"passed": True, "agent_verdict": hyp.answer, "answer": hyp.answer},
        extra={
            "hypothesis_id": hyp_id,
            "outcome": "verdict",
            "agent_verdict": hyp.answer,
            "final_p": 0.7,
            "prize_paid": hyp.prize,
        },
    )
    return {
        "schema": 1,
        "source": source,
        "key": {"model": model, "hypothesis_id": hyp_id, "seed": seed},
        "model_label": f"Label {model}",
        "settings": {"max_rounds": 3, "max_tokens": 3072, "prices": {"input": 2, "output": 10}},
        "record": record.to_dict(),
        "rounds": [{"round": 1, "usd_so_far": 0.001}],
        "usd": 0.001,
    }


def _verdict_transport():
    reply = (
        "<assessment>Evidence is sufficient.</assessment><p_success>0.7</p_success>"
        "<verdict>supported</verdict><evidence>Offline scripted transport.</evidence>"
    )
    return fake_llm.scripted_transport(
        [reply],
        usage_fn=lambda system, messages, text: {
            "input_tokens": 25,
            "output_tokens": 8,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    )


def _run_grid(settings, cache, ledger, factory, confirm=True, out=lambda *_: None):
    return demo_grid.run_grid(
        settings,
        cache,
        ledger,
        transport_factory=factory,
        confirm=confirm,
        out=out,
    )


def test_projection_and_usage_pricing_are_exact():
    sonnet = ModelPrice("Sonnet", "sonnet", 2, 10)
    projection = spend.project_run_usd(sonnet, 100, 1, 10, 2, 20)
    assert projection == {
        "calls": 3,
        "input_tokens": 210,
        "output_tokens": 30,
        "usd": 0.00072,
    }
    assert spend.usd_for_usage(sonnet, {
        "input_tokens": 100,
        "output_tokens": 20,
        "cache_creation_input_tokens": 40,
        "cache_read_input_tokens": 10,
    }) == 0.00052

    settings = spend.load_settings()
    by_id = {model.id: model for model in settings.models}
    usage = {"input_tokens": 1000, "output_tokens": 200}
    costs = {model_id: spend.usd_for_usage(model, usage) for model_id, model in by_id.items()}
    assert costs["claude-opus-5-5"] > costs["claude-sonnet-5-5"]
    assert costs["claude-sonnet-5-5"] == costs["claude-sonnet-5"]
    assert costs["claude-sonnet-5"] > costs["claude-haiku-4-5-20251001"]
    base = spend.project_run_usd(sonnet, 100, 1, 10, 2, 20)["usd"]
    assert spend.project_run_usd(sonnet, 100, 2, 10, 2, 20)["usd"] > base
    assert spend.project_run_usd(sonnet, 100, 1, 20, 2, 20)["usd"] > base


def test_live_models_table_and_validation(tmp_path):
    settings = spend.load_settings()
    assert [model.id for model in settings.models] == [
        "claude-sonnet-5-5",
        "claude-sonnet-5",
        "claude-opus-5-5",
        "claude-haiku-4-5-20251001",
    ]
    assert settings.source["models"].startswith("https://")
    assert settings.source["pricing"].startswith("https://")
    assert settings.source["checked"] == "2026-10-03"

    original = yaml.safe_load(spend.LIVE_MODELS_PATH.read_text())
    duplicate = json.loads(json.dumps(original))
    duplicate["models"][1]["id"] = duplicate["models"][0]["id"]
    path = tmp_path / "duplicate.yaml"
    path.write_text(yaml.safe_dump(duplicate))
    with pytest.raises(ValueError, match="unique"):
        spend.load_settings(path)

    negative = json.loads(json.dumps(original))
    negative["models"][0]["input"] = -1
    path.write_text(yaml.safe_dump(negative))
    with pytest.raises(ValueError, match="positive finite"):
        spend.load_settings(path)


def test_spend_ledger_totals_restart_caps_void_truncation_and_threads(tmp_path):
    path = tmp_path / "spend.jsonl"
    ledger = SpendLedger(path, cap=0.5)
    reservation = ledger.reserve("run-1", "model", 0.3)
    assert ledger.totals() == {
        "actual_usd": 0.0,
        "open_usd": 0.3,
        "committed_usd": 0.3,
        "calls": 0,
    }
    ledger.settle(reservation, "run-1", "model", {
        "input_tokens": 10,
        "output_tokens": 3,
    }, 0.2)
    restarted = SpendLedger(path, cap=0.5)
    assert restarted.totals() == {
        "actual_usd": 0.2,
        "open_usd": 0.0,
        "committed_usd": 0.2,
        "calls": 1,
    }
    voided = restarted.reserve("run-2", "model", 0.1)
    restarted.void(voided, "api_error")
    assert restarted.totals()["committed_usd"] == 0.2
    with pytest.raises(CapReached):
        restarted.reserve("run-3", "model", 0.31)
    with path.open("a") as stream:
        stream.write('{"type":"reserve","id":"partial"')
    with pytest.warns(RuntimeWarning, match="truncated"):
        assert restarted.totals()["committed_usd"] == 0.2
    corrupt = tmp_path / "corrupt.jsonl"
    corrupt.write_text('{"complete": invalid}\n')
    with pytest.raises(json.JSONDecodeError):
        SpendLedger(corrupt).totals()

    threaded = SpendLedger(tmp_path / "threaded.jsonl", cap=0.05)
    outcomes = []

    def reserve():
        try:
            threaded.reserve("thread", "model", 0.04)
        except CapReached:
            outcomes.append(False)
        else:
            outcomes.append(True)

    workers = [threading.Thread(target=reserve) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=5)
    assert sorted(outcomes) == [False, True]
    assert threaded.totals()["committed_usd"] <= 0.05


def test_metered_llm_bounds_calls_and_never_sends_over_cap(tmp_path):
    price = ModelPrice("Sonnet", "sonnet", 2, 10)
    ledger = SpendLedger(tmp_path / "metered.jsonl", cap=1)
    requests = []

    def transport(model, system, messages, max_tokens):
        requests.append((model, system, messages, max_tokens))
        return "ok", {"input_tokens": 5, "output_tokens": 2}

    metered = MeteredLLM("sonnet", price, ledger, "metered-run", transport=transport)
    messages = [{"role": "user", "content": "é"}]
    assert metered("sonnet", messages, system="s", max_tokens=10) == "ok"
    assert metered("sonnet", messages + [{"role": "user", "content": "abc"}],
                   system="s", max_tokens=10) == "ok"
    entries = [json.loads(line) for line in (tmp_path / "metered.jsonl").read_text().splitlines()]
    reservations = [entry for entry in entries if entry["type"] == "reserve"]
    assert reservations[0]["upper_usd"] == pytest.approx(0.000122)
    assert reservations[1]["upper_usd"] == pytest.approx(0.000136)
    assert metered.usage == {
        "calls": 2,
        "input_tokens": 10,
        "output_tokens": 4,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "usd": 0.00006,
    }
    assert ledger.totals()["actual_usd"] == 0.00006

    calls = []
    tiny = MeteredLLM(
        "sonnet",
        price,
        SpendLedger(tmp_path / "tiny.jsonl", cap=0.0001),
        "tiny-run",
        transport=lambda *args: calls.append(args) or ("no", {}),
    )
    with pytest.raises(CapReached):
        tiny("sonnet", messages, system="s", max_tokens=10)
    assert calls == []


def test_metered_llm_records_stop_reasons_and_defaults_to_16000_tokens(tmp_path):
    price = ModelPrice("Sonnet", "sonnet", 2, 10)
    requests = []
    stop_reasons = iter(("max_tokens", "end_turn"))

    def transport(model, system, messages, max_tokens):
        requests.append(max_tokens)
        return "ok", {
            "input_tokens": 5,
            "output_tokens": 2,
            "stop_reason": next(stop_reasons),
        }

    metered = MeteredLLM(
        "sonnet",
        price,
        SpendLedger(tmp_path / "metered-stop-reasons.jsonl", cap=1),
        "metered-stop-reasons",
        transport=transport,
    )

    metered("sonnet", [{"role": "user", "content": "first"}])
    metered("sonnet", [{"role": "user", "content": "second"}], max_tokens=10)

    assert requests == [16000, 10]
    assert metered.stop_reasons == ["max_tokens", "end_turn"]
    assert metered.take_cut_off() is True
    assert metered.take_cut_off() is False


def test_demo_grid_marks_only_the_round_that_hit_the_token_limit(tmp_path):
    settings = spend.load_settings()
    hyp = C.load().hypotheses[0]
    settings = replace(
        settings,
        models=settings.models[:1],
        hypotheses=(hyp.id,),
        seeds=(0,),
        max_rounds=3,
        max_tokens=32,
    )
    replies = iter((
        ("", "max_tokens"),
        (
            "<assessment>No response content yet.</assessment><p_success>0.4</p_success>",
            "end_turn",
        ),
        (
            "<assessment>Evidence is sufficient.</assessment><p_success>0.7</p_success>"
            "<verdict>supported</verdict><evidence>Offline response.</evidence>",
            "end_turn",
        ),
    ))

    def transport(_model, _system, _messages, _max_tokens):
        text, stop_reason = next(replies)
        return text, {
            "input_tokens": 25,
            "output_tokens": 0 if not text else 8,
            "stop_reason": stop_reason,
        }

    cache = tmp_path / "cutoff-grid.jsonl"
    messages = []
    result = _run_grid(
        settings,
        cache,
        SpendLedger(tmp_path / "cutoff-grid-spend.jsonl", cap=1),
        lambda _model: transport,
        out=lambda *parts: messages.append(" ".join(map(str, parts))),
    )

    assert result["done"] == 1, (result, messages)
    rounds = live_cache.load(cache)[0]["rounds"]
    assert [round_entry["cut_off"] for round_entry in rounds] == [True, False]


def test_run_grid_repairs_cache_store_split_after_store_append_failure(
    monkeypatch, tmp_path
):
    settings = spend.load_settings()
    hyp = C.load().hypotheses[0]
    settings = replace(
        settings,
        models=settings.models[:1],
        hypotheses=(hyp.id,),
        seeds=(0,),
        max_rounds=3,
        max_tokens=32,
    )
    cache = tmp_path / "runs.jsonl"
    store = tmp_path / "attempts.jsonl"
    transport_calls = []
    original_append = demo_grid.AttemptStore.append
    append_failed = False

    def fail_once(self, records):
        nonlocal append_failed
        if not append_failed:
            append_failed = True
            raise OSError("simulated store failure")
        return original_append(self, records)

    monkeypatch.setattr(demo_grid.AttemptStore, "append", fail_once)

    def transport_factory(model_id):
        transport = _verdict_transport()

        def call(*args):
            transport_calls.append(model_id)
            return transport(*args)

        return call

    run_options = {
        "settings": settings,
        "cache_path": cache,
        "ledger": SpendLedger(tmp_path / "spend.jsonl", cap=1),
        "transport_factory": transport_factory,
        "confirm": True,
        "out": lambda *_: None,
        "store_path": store,
        "show_preflight": False,
    }
    first = demo_grid.run_grid(**run_options)

    assert first["stopped_reason"] == "error"
    cached_entries = live_cache.load(cache)
    assert len(cached_entries) == 1
    assert demo_grid.AttemptStore(store).load() == []

    resumed = demo_grid.run_grid(**run_options)

    stored_records = demo_grid.AttemptStore(store).load()
    assert resumed["done"] == 0
    assert resumed["skipped"] == 1
    assert [record.attempt_id for record in stored_records] == [
        cached_entries[0]["record"]["attempt_id"]
    ]
    assert transport_calls == [settings.models[0].id]


def test_metered_llm_voids_provider_status_errors_and_keeps_unknown_reservations(
    monkeypatch, tmp_path
):
    import anthropic

    class StatusError(Exception):
        pass

    monkeypatch.setattr(anthropic, "APIStatusError", StatusError)
    price = ModelPrice("Sonnet", "sonnet", 2, 10)
    status_ledger = SpendLedger(tmp_path / "status.jsonl", cap=1)
    status_meter = MeteredLLM(
        "sonnet",
        price,
        status_ledger,
        "status-run",
        transport=lambda *args: (_ for _ in ()).throw(StatusError("provider refused")),
    )
    with pytest.raises(StatusError, match="provider refused"):
        status_meter("sonnet", [{"role": "user", "content": "test"}], max_tokens=10)
    assert status_ledger.totals()["committed_usd"] == 0

    unknown_ledger = SpendLedger(tmp_path / "unknown.jsonl", cap=1)
    unknown_meter = MeteredLLM(
        "sonnet",
        price,
        unknown_ledger,
        "unknown-run",
        transport=lambda *args: (_ for _ in ()).throw(RuntimeError("unknown outcome")),
    )
    with pytest.raises(RuntimeError, match="unknown outcome"):
        unknown_meter("sonnet", [{"role": "user", "content": "test"}], max_tokens=10)
    assert unknown_ledger.totals()["open_usd"] > 0


def test_live_market_reads_cumulative_spend_and_enforces_hard_cap(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ENABLE_LIVE", "1")
    monkeypatch.setenv("DM_MAX_USD", "50")
    ledger = SpendLedger(spend.LEDGER_PATH, cap=50)
    reservation = ledger.reserve("old-run", "claude-sonnet-5-5", 49.99)
    ledger.settle(reservation, "old-run", "claude-sonnet-5-5", {}, 49.99)

    info = live_market.info()
    assert info["live"]["hard_cap_usd"] == 50
    assert info["live"]["max_usd"] == 50
    assert info["live"]["spent_usd"] == 49.99
    assert info["live"]["actual_usd"] == 49.99
    assert info["live"]["remaining_usd"] == 0.01
    with pytest.raises(PermissionError, match="Projected run spend exceeds.*DM_MAX_USD"):
        live_market.start("gravity-inverse-square", "claude-sonnet-5-5", scripted=False)


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"ENABLE_LIVE": "1", "ANTHROPIC_API_KEY": "test-key"},
        {"ENABLE_LIVE": "1", "DM_MAX_USD": "5"},
    ],
)
def test_demo_grid_refuses_live_mode_without_each_guard(monkeypatch, tmp_path, env):
    for name in ("ANTHROPIC_API_KEY", "ENABLE_LIVE", "DM_MAX_USD"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    cache = tmp_path / "runs.jsonl"
    ledger = tmp_path / "spend.jsonl"
    with pytest.raises(SystemExit, match="refusing paid calls|ANTHROPIC_API_KEY"):
        demo_grid.main(["--cache", str(cache), "--ledger", str(ledger)])
    assert not ledger.exists()
    assert not cache.exists()


@pytest.mark.allow_live_env
def test_demo_grid_preflight_loads_cap_from_poc_env(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("DM_MAX_USD", raising=False)
    env_path = tmp_path / "poc.env"
    env_path.write_text("DM_MAX_USD=50\n")
    monkeypatch.setattr(bench, "load_env", lambda: _LOAD_ENV(env_path))
    demo_grid.main([
        "--preflight",
        "--cache", str(tmp_path / "runs.jsonl"),
        "--ledger", str(tmp_path / "spend.jsonl"),
    ])
    assert spend.effective_cap(spend.load_settings()) == 50
    output = capsys.readouterr().out
    assert "effective cap: $50.000000" in output

    monkeypatch.setenv("DM_MAX_USD", "100")
    bench.load_env()
    assert spend.effective_cap(spend.load_settings()) == 50
    demo_grid.main([
        "--preflight",
        "--cache", str(tmp_path / "runs-uncapped.jsonl"),
        "--ledger", str(tmp_path / "spend-uncapped.jsonl"),
    ])
    assert "the hard cap is applied" in capsys.readouterr().out
    monkeypatch.setenv("DM_MAX_USD", "40")
    bench.load_env()
    assert spend.effective_cap(spend.load_settings()) == 40


def test_bench_live_uses_metered_ledger_and_stops_when_cap_is_reached(
    monkeypatch, tmp_path, capsys
):
    settings = spend.load_settings()
    model = settings.models[0].id
    monkeypatch.setenv("ENABLE_LIVE", "1")
    monkeypatch.setenv("DM_MAX_USD", "5")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "offline-test-key")
    monkeypatch.setattr(bench, "load_env", lambda: None)

    calls = []
    scripted = _verdict_transport()

    def transport(model_id, system, messages, max_tokens):
        calls.append(model_id)
        return scripted(model_id, system, messages, max_tokens)

    monkeypatch.setattr("poc.llm.anthropic_transport", transport)
    store_path = tmp_path / "bench.jsonl"
    args = [
        "--models", model,
        "--seeds", "0",
        "--store", str(store_path),
        "--usd-per-call", "0.01",
    ]
    bench.main([*args, "--hypotheses", "gravity-inverse-square"])

    assert calls == [model]
    saved = json.loads(store_path.read_text().splitlines()[0])
    assert saved["llm_usage"]["calls"] == 1
    ledger = SpendLedger(spend.LEDGER_PATH, cap=5)
    assert ledger.totals()["actual_usd"] > 0
    assert "--usd-per-call is accepted but ignored" in capsys.readouterr().out

    reservation = ledger.reserve("test-over-cap", model, 4.99)
    ledger.settle(reservation, "test-over-cap", model, {"input_tokens": 1}, 4.99)
    bench.main([*args, "--hypotheses", "coulomb-source-strength"])

    output = capsys.readouterr().out
    assert "spend cap reached; stopping live grid" in output
    assert calls == [model]
    assert len(store_path.read_text().splitlines()) == 1


def test_bench_live_refuses_models_without_configured_prices(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("ENABLE_LIVE", "1")
    monkeypatch.setenv("DM_MAX_USD", "5")
    monkeypatch.setattr(bench, "load_env", lambda: None)
    store_path = tmp_path / "bench.jsonl"
    with pytest.raises(SystemExit, match="not in poc/live_models.yaml"):
        bench.main([
            "--models", "unpriced-model",
            "--hypotheses", "gravity-inverse-square",
            "--store", str(store_path),
        ])
    assert not spend.LEDGER_PATH.exists()
    assert not store_path.exists()


def test_demo_grid_resume_confirmation_usage_interruption_and_cap(tmp_path):
    settings = spend.load_settings()
    cache = tmp_path / "grid.jsonl"
    ledger = SpendLedger(tmp_path / "grid-spend.jsonl", cap=5)
    output = []
    factory_calls = []
    result = _run_grid(
        settings,
        cache,
        ledger,
        lambda model: factory_calls.append(model) or _verdict_transport(),
        confirm=False,
        out=output.append,
    )
    assert result["done"] == 0
    assert result["stopped_reason"] == "declined"
    assert len(factory_calls) == 0
    assert ledger.totals()["committed_usd"] == 0
    assert not cache.exists()
    assert all(model.label in "\n".join(output) for model in settings.models)

    result = _run_grid(
        settings,
        cache,
        ledger,
        lambda _model: _verdict_transport(),
        confirm=True,
        out=lambda *_: None,
    )
    assert result["done"] == 8
    entries = live_cache.load(cache)
    assert len(entries) == 8
    assert all(entry["record"]["llm_usage"]["input_tokens"] > 0 for entry in entries)
    assert live_cache.comparison(entries, settings.models)[0]["usd_spent"] > 0

    calls = []
    resumed = _run_grid(
        settings,
        cache,
        ledger,
        lambda model: calls.append(model) or _verdict_transport(),
        confirm=True,
        out=lambda *_: None,
    )
    assert resumed["done"] == 0
    assert resumed["skipped"] == 8
    assert calls == []

    interrupted_cache = tmp_path / "interrupted.jsonl"
    interrupted_ledger = SpendLedger(tmp_path / "interrupted-spend.jsonl", cap=5)
    calls_seen = 0

    def interrupt_on_fourth_call(_model):
        def transport(model, system, messages, max_tokens):
            nonlocal calls_seen
            calls_seen += 1
            if calls_seen == 4:
                raise KeyboardInterrupt
            return _verdict_transport()(model, system, messages, max_tokens)
        return transport

    interrupted = _run_grid(
        settings,
        interrupted_cache,
        interrupted_ledger,
        interrupt_on_fourth_call,
        confirm=True,
        out=lambda *_: None,
    )
    assert interrupted["done"] == 3
    assert interrupted["stopped_reason"] == "interrupted"
    assert len(live_cache.load(interrupted_cache)) == 3
    resumed = _run_grid(
        settings,
        interrupted_cache,
        interrupted_ledger,
        lambda _model: _verdict_transport(),
        confirm=True,
        out=lambda *_: None,
    )
    assert resumed["done"] == 5
    assert len(live_cache.load(interrupted_cache)) == 8

    capped_ledger = SpendLedger(tmp_path / "capped-spend.jsonl", cap=0.00001)
    capped = _run_grid(
        settings,
        tmp_path / "capped.jsonl",
        capped_ledger,
        lambda _model: pytest.fail("a projected-over-cap run must not call transport"),
        confirm=True,
        out=output.append,
    )
    assert capped["stopped_reason"] == "cap"
    assert "spend cap reached" in "\n".join(output)
    assert capped_ledger.totals()["committed_usd"] <= 0.00001


def test_live_cache_round_trip_redacts_secrets(tmp_path, monkeypatch):
    model = spend.load_settings().models[0]
    hyp = C.load().hypothesis("gravity-inverse-square")
    record = AttemptRecord(
        attempt_id="round-trip",
        source="live",
        protocol=C.PROTOCOL,
        venue=C.VENUE,
        world=hyp.world,
        solver=model.id,
        seed=0,
        stated_p_success=0.5,
        rounds=1,
        experiments=0,
        lab_cost=0,
        llm_usage={"calls": 1, "input_tokens": 5, "usd": 0.00001},
        verdict={"passed": False},
        extra={"hypothesis_id": hyp.id, "outcome": "verdict", "final_p": 0.5},
    )
    secret = "fake-secret-value"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    assert redact(f"request failed with {secret}") == "request failed with [redacted]"
    cache_path = tmp_path / "cache.jsonl"
    ledger_path = tmp_path / "ledger.jsonl"
    entry = {
        "schema": 1,
        "source": "real",
        "key": {"model": model.id, "hypothesis_id": hyp.id, "seed": 0},
        "model_label": model.label,
        "settings": {"max_rounds": 3, "max_tokens": 3072, "prices": {"input": 2, "output": 10}},
        "record": record.to_dict(),
        "rounds": [{"round": 1, "usd_so_far": 0.00001}],
        "usd": 0.00001,
    }
    live_cache.append(cache_path, entry)
    ledger = SpendLedger(ledger_path, cap=1)
    rid = ledger.reserve("round-trip", model.id, 0.01)
    ledger.settle(rid, "round-trip", model.id, {"input_tokens": 1}, 0.000002)
    loaded = live_cache.load(cache_path)
    assert AttemptRecord.from_dict(loaded[0]["record"]) == record
    assert loaded[0]["rounds"] == entry["rounds"]
    with cache_path.open("a") as stream:
        stream.write('{"partial":')
    with pytest.warns(RuntimeWarning, match="truncated"):
        assert len(live_cache.load(cache_path)) == 1
    files = cache_path.read_text() + ledger_path.read_text()
    assert secret not in files
    assert "x-api-key" not in files.lower()
    assert "authorization" not in files.lower()


def test_recorded_api_selects_real_runs_and_returns_details(app_server):
    settings = spend.load_settings()
    scripted_entries = [
        _resolved_entry(model.id, "scripted", "gravity-inverse-square")
        for model in settings.models
    ]
    for entry in scripted_entries:
        live_cache.append(live_cache.SCRIPTED_PATH, entry)
    status, body = _request(f"{app_server}/api/live/recorded")
    assert status == 200
    scripted = json.loads(body)
    assert scripted["source"] == "scripted"
    assert len(scripted["comparison"]) == 4
    assert len(scripted["runs"]) == 4
    for row, entry in zip(scripted["comparison"], scripted_entries):
        record = AttemptRecord.from_dict(entry["record"])
        expected = report.summarise([record])["models"][record.solver]["brier_final_p"]
        assert row["brier"] == expected

    real_entry = _resolved_entry(
        settings.models[0].id,
        "real",
        "gravity-inverse-square",
        seed=1,
    )
    real_entry["rounds"][0]["cut_off"] = True
    live_cache.append(live_cache.RUNS_PATH, real_entry)
    status, body = _request(f"{app_server}/api/live/recorded")
    assert status == 200
    real = json.loads(body)
    assert real["source"] == "real"
    assert real["real_count"] == 1
    assert real["scripted_count"] == 4
    assert len(real["runs"]) == 1

    attempt_id = real_entry["record"]["attempt_id"]
    status, body = _request(
        f"{app_server}/api/live/recorded/run?id={attempt_id}"
    )
    assert status == 200
    detail = json.loads(body)
    assert detail["source"] == "real"
    assert detail["rounds"] == real_entry["rounds"]
    assert detail["rounds"][0]["cut_off"] is True
    status, body = _request(f"{app_server}/api/live/recorded/run?id=unknown")
    assert status == 404
    assert json.loads(body) == {"error": "not found"}


def test_committed_scripted_grid_fixture_has_truthy_and_false_claims():
    entries = live_cache.load(
        Path(REPO_ROOT / "attempts/fixtures/live/scripted_demo.jsonl")
    )
    settings = spend.load_settings()
    assert len(entries) == 8
    assert all(entry["source"] == "scripted" for entry in entries)
    assert {entry["key"]["model"] for entry in entries} == {
        model.id for model in settings.models
    }
    cfg = C.load()
    for model in settings.models:
        model_records = [
            entry for entry in entries if entry["key"]["model"] == model.id
        ]
        assert len(model_records) == 2
        assert {
            cfg.hypothesis(entry["key"]["hypothesis_id"]).answer == "supported"
            for entry in model_records
        } == {True, False}
