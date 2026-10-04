"""HTTP and scripted-run coverage for the local live-market APIs."""

import json
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import anthropic
import pytest
import scienceagent.llm_client

import app
import live_market
from dm.store import AttemptStore
from dm.types import AttemptRecord, SubmittedAttempt
from poc import bench, config as C, live_cache, spend
from poc.llm import MeteredLLM, anthropic_transport


@pytest.fixture(autouse=True)
def isolated_live_market(monkeypatch, tmp_path):
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
    monkeypatch.setattr(C, "ATTEMPTS_PATH", tmp_path / "live.jsonl")
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


def _request(url, method="GET", body=None, headers=None):
    request = urllib.request.Request(url, data=body, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers, error.read()


def _post_start(base_url, hypothesis_id="gravity-inverse-square", scripted=True, model=""):
    return _request(
        f"{base_url}/api/live/start",
        method="POST",
        headers={"Content-Type": "application/json"},
        body=json.dumps({
            "hypothesis_id": hypothesis_id,
            "model": model,
            "scripted": scripted,
        }).encode(),
    )


def _wait_for_job(base_url, job_id, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status, _, body = _request(f"{base_url}/api/live/job?id={job_id}")
        assert status == 200
        result = json.loads(body)
        if result["state"] != "running":
            return result
        time.sleep(0.1)
    pytest.fail(f"scripted job {job_id} did not finish within {timeout} seconds")


def test_anthropic_transport_ignores_thinking_blocks(monkeypatch):
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="reasoning"),
            SimpleNamespace(type="text", text="ANSWER"),
        ],
        usage=SimpleNamespace(
            input_tokens=12,
            output_tokens=3,
            cache_creation_input_tokens=2,
            cache_read_input_tokens=1,
        ),
        stop_reason="max_tokens",
    )
    client_kwargs = {}

    class FakeMessages:
        def create(self, **kwargs):
            return response

    class FakeAnthropic:
        def __init__(self, **kwargs):
            client_kwargs.update(kwargs)
            self.messages = FakeMessages()

    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)

    text, usage = anthropic_transport("model", None, [], 64)

    assert text == "ANSWER"
    assert usage == {
        "input_tokens": 12,
        "output_tokens": 3,
        "cache_creation_input_tokens": 2,
        "cache_read_input_tokens": 1,
        "stop_reason": "max_tokens",
    }
    assert client_kwargs["timeout"] == 600


def test_real_live_job_passes_metered_llm_to_run_attempt(monkeypatch, tmp_path):
    cfg = C.load()
    hyp = cfg.hypothesis("gravity-inverse-square")
    settings = spend.load_settings()
    model = "claude-sonnet-5-5"
    model_price = spend.price(settings, model)
    spend_ledger = spend.SpendLedger(tmp_path / "live_spend.jsonl", cap=settings.max_usd)
    store_path = tmp_path / "live.jsonl"
    job_id = "real-metering-test"
    live_market._JOBS[job_id] = {
        "state": "running",
        "hypothesis_id": hyp.id,
        "model": model,
        "seed": 0,
        "scripted": False,
        "rounds": [],
        "run": None,
        "error": None,
    }
    live_market._ACTIVE_JOB = job_id
    captured = []

    def fake_run_attempt(*args, complete=None, **kwargs):
        captured.append(complete)
        raise RuntimeError("captured complete")

    monkeypatch.setattr(live_market, "run_attempt", fake_run_attempt)

    live_market._run_job(
        job_id,
        hyp,
        model,
        0,
        False,
        cfg,
        [],
        store_path,
        spend_ledger,
        model_price,
        settings,
    )

    assert len(captured) == 1
    assert isinstance(captured[0], MeteredLLM)
    assert captured[0] is not scienceagent.llm_client.complete


def test_info_hides_answers_and_redacts_api_key(app_server, monkeypatch):
    disabled = live_market.info()
    assert disabled["live"]["enabled"] is False
    assert len(disabled["live"]["reasons"]) == 3
    assert all("answer" not in hyp for hyp in disabled["hypotheses"])
    assert '"answer":' not in json.dumps(disabled)
    assert disabled["ledger_read_fee"] == C.load().ledger_read_fee
    assert all(hyp["solved_by"] is None for hyp in disabled["hypotheses"])
    assert disabled["noise_std"] == C.load().noise_std
    assert disabled["velocity_noise_std"] == C.load().velocity_noise_std

    secret = "test-secret-never-return"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    monkeypatch.setenv("ENABLE_LIVE", "1")
    monkeypatch.setenv("DM_MAX_USD", "100")
    monkeypatch.setenv("DM_LIVE_MODELS", "claude-sonnet-5-5, claude-opus-5-5, custom-model")
    enabled = live_market.info()
    serialized = json.dumps(enabled)
    assert enabled["live"]["enabled"] is True
    assert enabled["models"] == [
        "claude-sonnet-5-5",
        "claude-opus-5-5",
        "custom-model",
    ]
    assert secret not in serialized
    assert '"answer":' not in serialized
    assert enabled["live"]["hard_cap_usd"] == 50
    assert enabled["live"]["max_usd"] == 50
    assert "the hard cap is applied" in enabled["live"]["cap_note"]
    assert enabled["live"]["projected_usd_per_run"]["claude-sonnet-5-5"] > 0
    assert len(enabled["model_table"]) == 4
    status, headers, body = _request(f"{app_server}/api/live/info")
    assert status == 200
    assert "json" in headers.get("Content-Type", "").lower()
    api_info = json.loads(body)
    assert api_info["live"]["enabled"] is True
    assert api_info["noise_std"] == C.load().noise_std
    assert api_info["velocity_noise_std"] == C.load().velocity_noise_std


def _solved_record(hypothesis_id, solver="prior-solver"):
    hyp = C.load().hypothesis(hypothesis_id)
    return AttemptRecord(
        attempt_id=f"solved-{hypothesis_id}",
        source="live",
        protocol=C.PROTOCOL,
        venue=C.VENUE,
        world=hyp.world,
        solver=solver,
        seed=0,
        stated_p_success=0.9,
        rounds=1,
        experiments=0,
        lab_cost=10,
        verdict={"passed": True, "agent_verdict": hyp.answer, "answer": hyp.answer},
        extra={"hypothesis_id": hypothesis_id},
    )


def test_info_reports_solved_by_and_record_fee():
    hyp_id = "gravity-inverse-square"
    AttemptStore(C.ATTEMPTS_PATH).append([_solved_record(hyp_id)])

    data = live_market.info()

    assert data["ledger_read_fee"] == C.load().ledger_read_fee
    claim = next(hyp for hyp in data["hypotheses"] if hyp["id"] == hyp_id)
    assert claim["solved_by"] == "prior-solver"


def test_real_live_start_refuses_solved_claim_without_reserving_spend(
    app_server,
):
    hyp_id = "gravity-inverse-square"
    AttemptStore(C.ATTEMPTS_PATH).append([_solved_record(hyp_id)])

    status, _, body = _post_start(
        app_server, hypothesis_id=hyp_id, scripted=False, model="claude-sonnet-5-5"
    )

    assert status == 400
    assert json.loads(body)["error"] == (
        f"{hyp_id} was solved by prior-solver; it is off the market."
    )
    assert spend.SpendLedger(spend.LEDGER_PATH).totals()["committed_usd"] == 0


def test_scripted_live_start_remains_available_for_solved_claim(monkeypatch):
    hyp_id = "gravity-inverse-square"
    AttemptStore(C.ATTEMPTS_PATH).append([_solved_record(hyp_id)])

    def fake_run(model, hypothesis_id, seed, ledger, cfg=None, **_kwargs):
        hyp = cfg.hypothesis(hypothesis_id)
        return SubmittedAttempt(
            source="live",
            protocol=C.PROTOCOL,
            venue=C.VENUE,
            world=hyp.world,
            solver=model,
            seed=seed,
            stated_p_success=0.8,
            rounds=1,
            experiments=0,
            lab_cost=0,
            extra={
                "hypothesis_id": hyp.id,
                "outcome": "verdict",
                "agent_verdict": hyp.answer,
                "prize": hyp.prize,
                "first_p": 0.8,
                "final_p": 0.8,
            },
        )

    monkeypatch.setattr(live_market, "run_attempt", fake_run)
    started = live_market.start(hyp_id, "scripted-demo", scripted=True)

    deadline = time.monotonic() + 5
    current = live_market.job(started["job_id"])
    while current["state"] == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
        current = live_market.job(started["job_id"])
    assert current["state"] == "done"
    assert current["run"]["hypothesis_id"] == hyp_id


def test_live_start_refuses_disabled_and_over_budget_runs(app_server, monkeypatch):
    status, _, body = _post_start(app_server, scripted=False, model="claude-sonnet-5-5")
    assert status == 403
    assert "ENABLE_LIVE" in json.loads(body)["error"]

    status, _, body = _post_start(app_server, scripted=False, model="not-allowed")
    assert status == 400
    assert "allowlist" in json.loads(body)["error"]

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ENABLE_LIVE", "1")
    monkeypatch.setenv("DM_MAX_USD", "0.1")
    status, _, body = _post_start(app_server, scripted=False, model="claude-sonnet-5-5")
    assert status == 403
    assert "Projected run spend exceeds the spend cap" in json.loads(body)["error"]
    assert live_market.info()["live"]["spent_usd"] == 0


def test_live_start_tracks_projection_without_calling_provider(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ENABLE_LIVE", "1")
    monkeypatch.setenv("DM_MAX_USD", "5")
    entered, release = threading.Event(), threading.Event()

    def fake_run(model, hypothesis_id, seed, ledger, cfg=None, **kwargs):
        entered.set()
        release.wait(timeout=5)
        hyp = cfg.hypothesis(hypothesis_id)
        return SubmittedAttempt(
            source="live",
            protocol=C.PROTOCOL,
            venue=C.VENUE,
            world=hyp.world,
            solver=model,
            seed=seed,
            stated_p_success=0.5,
            rounds=1,
            experiments=0,
            lab_cost=0,
            extra={
                "hypothesis_id": hyp.id,
                "outcome": "verdict",
                "agent_verdict": hyp.answer,
                "prize": hyp.prize,
                "first_p": 0.5,
                "final_p": 0.5,
            },
        )

    monkeypatch.setattr(live_market, "run_attempt", fake_run)
    result = live_market.start(
        "gravity-inverse-square", "claude-sonnet-5-5", scripted=False
    )
    assert entered.wait(timeout=5)
    projection = live_market.info()["live"]["projected_usd_per_run"]["claude-sonnet-5-5"]
    assert 0 < result["projected_usd"] <= projection
    assert live_market.info()["live"]["spent_usd"] == 0
    current = live_market.job(result["job_id"])
    assert current["state"] == "running"
    assert current["run"] is None
    assert "answer" not in json.dumps(current["rounds"])
    release.set()

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        current = live_market.job(result["job_id"])
        if current["state"] != "running":
            break
        time.sleep(0.01)
    assert current["state"] == "done"
    assert current["run"]["answer"] in ("supported", "refuted")
    recorded = live_cache.load(live_cache.RUNS_PATH)
    assert len(recorded) == 1
    assert recorded[0]["source"] == "real"
    assert recorded[0]["key"]["model"] == "claude-sonnet-5-5"


def test_scripted_http_run_and_seed_increment(app_server, monkeypatch):
    status, _, body = _post_start(app_server)
    assert status == 200
    first = json.loads(body)
    assert first["seed"] == 0
    assert first["scripted"] is True
    assert first["projected_usd"] == 0

    result = _wait_for_job(app_server, first["job_id"])
    assert result["state"] == "done"
    assert result["hypothesis_id"] == "gravity-inverse-square"
    assert len(result["rounds"]) >= 2
    assert result["rounds"][0]["n_experiments"] == 1
    assert result["rounds"][0]["experiment_input"][0]["p1"] == 1
    assert "answer" not in json.dumps(result["rounds"])
    assert result["run"]["model"] == "scripted-demo"
    assert result["run"]["rounds"] >= 2
    assert len(AttemptStore(live_market.DEMO_PATH).load()) == 1

    scripted_record = AttemptStore(live_market.DEMO_PATH).load()[0]
    scripted_entry = {
        "record": scripted_record.to_dict(),
        "rounds": [dict(round_entry) for round_entry in result["rounds"]],
        "settings": {},
        "source": "scripted",
    }
    scripted_entry["rounds"][0].pop("cut_off", None)
    with monkeypatch.context() as patch:
        patch.setattr(
            live_cache,
            "load",
            lambda path: [scripted_entry] if path == live_cache.SCRIPTED_PATH else [],
        )
        replay = live_market.recorded_run(scripted_record.attempt_id)
    assert all(round_entry["cut_off"] is False for round_entry in replay["rounds"])

    status, _, body = _post_start(app_server)
    assert status == 200
    second = json.loads(body)
    assert second["seed"] == 1
    assert _wait_for_job(app_server, second["job_id"])["state"] == "done"

    status, _, body = _request(f"{app_server}/api/live/runs")
    assert status == 200
    records = json.loads(body)
    assert len(records["demo"]) == 2
    assert records["summary"] == live_market.runs()["summary"]


def test_busy_start_unknown_hypothesis_and_unknown_job(app_server, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def blocked_run(*args, **kwargs):
        entered.set()
        release.wait(timeout=5)
        raise RuntimeError("scripted test run stopped")

    monkeypatch.setattr(live_market, "run_attempt", blocked_run)
    status, _, body = _post_start(app_server)
    assert status == 200
    job_id = json.loads(body)["job_id"]
    assert entered.wait(timeout=5)
    try:
        status, _, body = _post_start(app_server)
        assert status == 409
        assert "already in progress" in json.loads(body)["error"]
    finally:
        release.set()
    assert _wait_for_job(app_server, job_id)["state"] == "error"

    status, _, body = _post_start(app_server, hypothesis_id="not-a-hypothesis")
    assert status == 400
    assert "unknown hypothesis" in json.loads(body)["error"]

    status, _, _ = _request(f"{app_server}/api/live/job?id=missing")
    assert status == 404


def test_scripted_experiment_is_valid_for_each_hypothesis():
    from dataclasses import replace

    cfg = replace(C.load(), max_rounds=2)
    for hyp in cfg.hypotheses:
        submitted = live_market.run_attempt(
            "scripted-demo",
            hyp.id,
            0,
            [],
            cfg=cfg,
            complete=live_market.scripted_llm(hyp),
        )
        assert submitted.experiments == 1, hyp.id
        assert submitted.rounds == 2, hyp.id


def test_job_round_payload_caps_large_fields():
    compact = live_market._compact_round({
        "round": 1,
        "action": "experiment",
        "experiment_input": [{"payload": "x" * 700}],
        "llm_reply": "y" * 5000,
        "mse_fit_output": "z" * 900,
        "record_bought": True,
    })

    assert len(compact["experiment_input"][0]["truncated"]) == 600
    assert len(compact["reply"]) == 4000
    assert len(compact["mse_fit"]) == 800
    assert compact["cut_off"] is False
    assert compact["record_bought"] is True


def test_job_round_callback_marks_only_new_max_token_cutoffs():
    job_id = "cutoff"
    live_market._JOBS[job_id] = {"rounds": []}
    cutoffs = iter((True, False))
    metered = SimpleNamespace(usd=0.0, take_cut_off=lambda: next(cutoffs))
    callback = live_market._job_round_callback(job_id, metered)

    callback({"round": 1})
    callback({"round": 2})

    assert [row["cut_off"] for row in live_market._JOBS[job_id]["rounds"]] == [
        True,
        False,
    ]

    scripted_id = "scripted"
    live_market._JOBS[scripted_id] = {"rounds": []}
    live_market._job_round_callback(scripted_id)({"round": 1})
    assert live_market._JOBS[scripted_id]["rounds"][0]["cut_off"] is False
