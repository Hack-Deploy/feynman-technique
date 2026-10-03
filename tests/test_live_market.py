"""HTTP and scripted-run coverage for the local live-market APIs."""

import json
import threading
import time
import urllib.error
import urllib.request

import pytest
import scienceagent.llm_client

import app
import live_market
from dm.store import AttemptStore
from dm.types import SubmittedAttempt
from poc import bench, config as C


@pytest.fixture(autouse=True)
def isolated_live_market(monkeypatch, tmp_path):
    for name in ("ANTHROPIC_API_KEY", "ENABLE_LIVE", "DM_MAX_USD", "DM_USD_PER_CALL",
                 "DM_LIVE_MODELS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(bench, "load_env", lambda: None)
    monkeypatch.setattr(
        scienceagent.llm_client,
        "complete",
        lambda **kwargs: pytest.fail("tests must never call a paid API"),
    )
    monkeypatch.setattr(C, "ATTEMPTS_PATH", tmp_path / "live.jsonl")
    monkeypatch.setattr(live_market, "DEMO_PATH", tmp_path / "demo.jsonl")
    monkeypatch.setattr(live_market, "_JOBS", {})
    monkeypatch.setattr(live_market, "_ACTIVE_JOB", None)
    monkeypatch.setattr(live_market, "_PROJECTED_USD_SPENT", 0.0)


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


def test_info_hides_answers_and_redacts_api_key(app_server, monkeypatch):
    disabled = live_market.info()
    assert disabled["live"]["enabled"] is False
    assert len(disabled["live"]["reasons"]) == 3
    assert all("answer" not in hyp for hyp in disabled["hypotheses"])
    assert '"answer":' not in json.dumps(disabled)

    secret = "test-secret-never-return"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    monkeypatch.setenv("ENABLE_LIVE", "1")
    monkeypatch.setenv("DM_MAX_USD", "5")
    monkeypatch.setenv("DM_LIVE_MODELS", "claude-sonnet-4-6, claude-opus-4-1")
    enabled = live_market.info()
    serialized = json.dumps(enabled)
    assert enabled["live"]["enabled"] is True
    assert enabled["models"] == ["claude-sonnet-4-6", "claude-opus-4-1"]
    assert secret not in serialized
    assert '"answer":' not in serialized
    assert enabled["live"]["projected_usd_per_run"] == pytest.approx(
        (2 * C.load().max_rounds + 1) * enabled["live"]["usd_per_call"]
    )
    status, headers, body = _request(f"{app_server}/api/live/info")
    assert status == 200
    assert "json" in headers.get("Content-Type", "").lower()
    assert json.loads(body)["live"]["enabled"] is True


def test_live_start_refuses_disabled_and_over_budget_runs(app_server, monkeypatch):
    status, _, body = _post_start(app_server, scripted=False, model="claude-sonnet-4-6")
    assert status == 403
    assert "ENABLE_LIVE" in json.loads(body)["error"]

    status, _, body = _post_start(app_server, scripted=False, model="not-allowed")
    assert status == 400
    assert "allowlist" in json.loads(body)["error"]

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ENABLE_LIVE", "1")
    monkeypatch.setenv("DM_MAX_USD", "0.1")
    status, _, body = _post_start(app_server, scripted=False, model="claude-sonnet-4-6")
    assert status == 403
    assert "Projected run spend exceeds DM_MAX_USD" in json.loads(body)["error"]
    assert live_market.info()["live"]["projected_usd_spent"] == 0


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
        "gravity-inverse-square", "claude-sonnet-4-6", scripted=False
    )
    assert entered.wait(timeout=5)
    projection = live_market.info()["live"]["projected_usd_per_run"]
    assert result["projected_usd"] == projection
    assert live_market.info()["live"]["projected_usd_spent"] == projection
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


def test_scripted_http_run_and_seed_increment(app_server):
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
    })

    assert len(compact["experiment_input"][0]["truncated"]) == 600
    assert len(compact["reply"]) == 4000
    assert len(compact["mse_fit"]) == 800
