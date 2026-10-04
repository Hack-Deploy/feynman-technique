"""Tests for real-attempt data and routes in the local results app."""

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

import app
import real_data
from dm.store import AttemptStore
from dm.types import AttemptRecord
from dm.venues import forcebench

VENDOR_SCRIPTS = (
    Path(__file__).resolve().parents[1] / "vendor" / "discovery-agents" / "scripts"
)


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
    request = urllib.request.Request(
        url, data=body, headers=headers or {}, method=method
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers, error.read()


def _record(world, model, seed, nmse, passed):
    return AttemptRecord(
        attempt_id=f"ara:{model}:{world}",
        source="published_replay",
        protocol="ara_harness",
        venue="discoverphysics",
        world=world,
        solver=f"ara:{model}",
        seed=seed,
        stated_p_success=None,
        rounds=2,
        experiments=4,
        lab_cost=2.0,
        verdict={"normalised_mse": nmse, "passed": passed},
    )


def test_build_real_data_when_files_are_missing(monkeypatch, tmp_path):
    for name in (
        "ARA_STORE", "ARA_SUMMARY", "FORCEBENCH_GRID", "SIM_SUMMARY"
    ):
        monkeypatch.setattr(real_data, name, tmp_path / f"missing-{name}.json")

    data = real_data.build_real_data()

    assert data["ara"]["available"] is False
    assert data["ara"]["attempts"] == []
    assert data["forcebench"]["available"] is False
    assert data["forcebench"]["worlds"] == list(forcebench.WORLDS)
    assert data["forcebench"]["solvers"] == ["bayes_lite", "random_menu"]
    assert data["forcebench"]["source"] is None


def test_forcebench_grid_aggregation(monkeypatch, tmp_path):
    rows = [
        {
            "solver": "bayes_lite",
            "world": "gravity",
            "seed": 1,
            "passed": False,
            "identified": True,
            "baseline_passed": False,
            "experiments": 5,
            "stated_p": 0.8,
            "nmse": 0.2,
        },
        {
            "solver": "bayes_lite",
            "world": "gravity",
            "seed": 0,
            "passed": True,
            "identified": False,
            "baseline_passed": True,
            "experiments": 3,
            "stated_p": 0.4,
            "nmse": 0.01,
        },
        {
            "solver": "random_menu",
            "world": "yukawa",
            "seed": 2,
            "passed": True,
            "identified": True,
            "baseline_passed": False,
            "experiments": 2,
            "stated_p": 0.5,
            "nmse": 0.03,
        },
    ]
    grid_path = tmp_path / "forcebench_settle.json"
    grid_path.write_text(json.dumps({"results": rows}))
    monkeypatch.setattr(real_data, "FORCEBENCH_GRID", grid_path)

    data = real_data.forcebench_data()
    cells = {(row["solver"], row["world"]): row for row in data["table"]}

    gravity = cells[("bayes_lite", "gravity")]
    assert gravity["n"] == 2
    assert gravity["passed"] == 1
    assert gravity["identified"] == 1
    assert gravity["baseline_passed"] == 1
    assert gravity["mean_experiments"] == 4
    assert gravity["nmse"] == [0.01, 0.2]

    yukawa = cells[("random_menu", "yukawa")]
    assert yukawa["n"] == 1
    assert yukawa["passed"] == 1
    assert yukawa["identified"] == 1
    assert yukawa["baseline_passed"] == 0
    assert yukawa["mean_experiments"] == 2
    assert yukawa["nmse"] == [0.03]
    assert data["source"] == "output"
    assert [(row["world"], row["solver"], row["seed"]) for row in data["attempts"]] == [
        ("gravity", "bayes_lite", 0),
        ("gravity", "bayes_lite", 1),
        ("yukawa", "random_menu", 2),
    ]
    assert data["attempts"][0]["top_model"] is None
    assert data["attempts"][0]["top_params"] is None
    assert data["attempts"][0]["stopped_reason"] is None
    assert data["attempts"][0]["baseline_nmse"] is None


def test_forcebench_snapshot_fallback_and_menu(monkeypatch, tmp_path):
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps({
        "head": "abc123",
        "test_seed": 0,
        "results": [{
            "solver": "bayes_lite", "world": "gravity", "seed": 0,
            "passed": True, "identified": True, "nmse": 0.01,
            "baseline_passed": False, "experiments": 2, "stated_p": 0.6,
        }],
    }))
    grid_path = tmp_path / "missing-output.json"
    monkeypatch.setattr(real_data, "FORCEBENCH_GRID", grid_path)
    monkeypatch.setattr(real_data, "SNAPSHOTS", {grid_path: snapshot_path})

    data = real_data.forcebench_data()

    assert data["source"] == "snapshot"
    assert data["available"] is True
    assert data["head"] == "abc123"
    assert len(data["menu"]) == 13
    assert sum(item["seed"] for item in data["menu"]) == 1
    assert data["venue"] == {
        "budget": forcebench.BUDGET,
        "noise_std": forcebench.NOISE_STD,
        "measurement_times": list(forcebench.MEASUREMENT_TIMES),
        "seed_action": forcebench.SEED_ACTION,
        "threshold": 0.1,
    }


def test_ara_attempts_summary_and_sim_clearing(monkeypatch, tmp_path):
    store_path = tmp_path / "ara.jsonl"
    summary_path = tmp_path / "ara-summary.json"
    sim_summary_path = tmp_path / "sim-summary.json"
    monkeypatch.setattr(real_data, "ARA_STORE", store_path)
    monkeypatch.setattr(real_data, "ARA_SUMMARY", summary_path)
    monkeypatch.setattr(real_data, "SIM_SUMMARY", sim_summary_path)
    AttemptStore(store_path).append(
        [
            _record("yukawa", "opus", 0, None, False),
            _record("gravity", "gpt5.5", 0, 0.02, True),
        ]
    )
    summary_path.write_text(
        json.dumps(
            {
                "caveat": "Published attempts are replayed through the market.",
                "config": {"prizes": {"gravity": 100}},
                "checks": {"verified": True},
                "verdict_rule": "independent oracle",
                "worlds": ["gravity", "yukawa"],
                "solvers": ["gpt5.5", "opus"],
                "clearing_prizes": {"gravity": 100},
                "solved_seed_counts": {"gravity": 1},
                "profit_and_bids": {"gpt5.5": {}},
                "lab_revenue": 4.0,
            }
        )
    )
    sim_summary_path.write_text(
        json.dumps({"h2_clearing_prizes": {"raw": {"coulomb": 25.0}}})
    )

    data = real_data.ara_data()

    assert data["available"] is True
    assert "ARA Labs (AgentNativeResearchLab)" in data["attribution"]
    assert data["attribution_url"] == "https://huggingface.co/AgentNativeResearchLab"
    assert [(row["world"], row["model"]) for row in data["attempts"]] == [
        ("gravity", "gpt5.5"),
        ("yukawa", "opus"),
    ]
    assert data["attempts"][0]["model"] == "gpt5.5"
    assert data["attempts"][1]["nmse"] is None
    assert data["sim_clearing"]["raw"]["coulomb_easy"] == 25.0


@pytest.mark.parametrize(
    ("world", "solver", "seed"),
    [
        ("unknown", "bayes_lite", 0),
        ("gravity", "unknown", 0),
        ("gravity", "bayes_lite", 7),
    ],
)
def test_forcebench_attempt_rejects_invalid_inputs_before_simulation(
    monkeypatch, world, solver, seed
):
    def unexpected_run(*args, **kwargs):
        pytest.fail("invalid input reached the ForceBench simulator")

    monkeypatch.setattr(forcebench, "run_attempt", unexpected_run)

    with pytest.raises(ValueError):
        real_data.run_forcebench_attempt(world, solver, seed)


def test_three_pages_removed_routes_static_and_api_data(app_server, monkeypatch):
    monkeypatch.setattr(
        real_data,
        "build_real_data",
        lambda: {"ara": {"available": False}, "forcebench": {"available": False}},
    )

    for path in ("/", "/simulation", "/live"):
        status, headers, _ = _request(f"{app_server}{path}")
        assert status == 200
        assert "text/html" in headers.get("Content-Type", "").lower()

    for path in (
        "/report", "/real", "/index.html", "/experiments", "/api/data", "/api/bounty",
        "/api/bounty/info", "/api/run", "/api/test",
    ):
        assert _request(f"{app_server}{path}")[0] == 404
    for path in ("/api/bounty", "/api/run", "/api/test"):
        assert _request(f"{app_server}{path}", method="POST", body=b"{}")[0] == 404

    status, headers, _ = _request(f"{app_server}/web/style.css")
    assert status == 200
    assert "text/css" in headers.get("Content-Type", "").lower()
    assert _request(f"{app_server}/web/%2e%2e/app.py")[0] == 404
    assert _request(f"{app_server}/web/data/%2e%2e/style.css")[0] == 404
    assert _request(f"{app_server}/web/data/missing.json")[0] == 404
    status, _, body = _request(f"{app_server}/web/data/experiments.json")
    assert status == 200 and json.loads(body)["launches"]

    status, headers, body = _request(f"{app_server}/api/real")
    assert status == 200
    assert "json" in headers.get("Content-Type", "").lower()
    assert set(json.loads(body)) == {"ara", "forcebench"}


def test_real_attempt_api_validation_busy_and_success(app_server, monkeypatch):
    headers = {"Content-Type": "application/json"}
    status, _, body = _request(
        f"{app_server}/api/real/attempt",
        method="POST",
        body=b"not-json",
        headers=headers,
    )
    assert status == 400
    assert json.loads(body)["ok"] is False

    status, _, body = _request(
        f"{app_server}/api/real/attempt",
        method="POST",
        body=json.dumps(
            {"world": "nope", "solver": "bayes_lite", "seed": 0}
        ).encode(),
        headers=headers,
    )
    assert status == 400
    assert json.loads(body)["ok"] is False

    result = {"ok": True, "x": 1}
    monkeypatch.setattr(
        real_data,
        "run_forcebench_attempt",
        lambda world, solver, seed: result,
    )
    status, _, body = _request(
        f"{app_server}/api/real/attempt",
        method="POST",
        body=json.dumps(
            {"world": "gravity", "solver": "bayes_lite", "seed": 0}
        ).encode(),
        headers=headers,
    )
    assert status == 200
    assert json.loads(body) == result

    assert app._BUSY.acquire(blocking=False)
    try:
        status, _, body = _request(
            f"{app_server}/api/real/attempt",
            method="POST",
            body=json.dumps(
                {"world": "gravity", "solver": "bayes_lite", "seed": 0}
            ).encode(),
            headers=headers,
        )
    finally:
        app._BUSY.release()
    assert status == 409
    assert json.loads(body)["ok"] is False

    status, _, _ = _request(
        f"{app_server}/api/real/attempt",
        method="POST",
        body=json.dumps(
            {"world": "gravity", "solver": "bayes_lite", "seed": 0}
        ).encode(),
        headers={**headers, "Origin": "http://evil.example"},
    )
    assert status == 403


def test_real_replay_and_grid_dispatch(app_server, monkeypatch):
    chain_calls = []
    command_calls = []

    def fake_chain(commands):
        chain_calls.append(commands)
        return {"ok": True}

    def fake_command(command, timeout=600):
        command_calls.append((command, timeout))
        return {"ok": True}

    monkeypatch.setattr(app, "run_chain", fake_chain)
    monkeypatch.setattr(app, "run_command", fake_command)

    status, _, body = _request(
        f"{app_server}/api/real/replay", method="POST", body=b""
    )
    assert status == 200
    assert json.loads(body)["ok"] is True
    assert chain_calls == [app.CHAINS["/api/real/replay"]]

    status, _, body = _request(
        f"{app_server}/api/real/grid", method="POST", body=b""
    )
    assert status == 200
    assert json.loads(body)["ok"] is True
    assert command_calls == [
        (app.COMMANDS["/api/real/grid"], 1800),
    ]


@pytest.mark.slow
@pytest.mark.skipif(
    not VENDOR_SCRIPTS.is_dir(),
    reason="DiscoverPhysics vendor scripts are not present",
)
def test_forcebench_gravity_attempt_end_to_end():
    result = real_data.run_forcebench_attempt("gravity", "bayes_lite", 0)

    assert result["ok"] is True
    assert result["verdict"]["passed"] is True
    assert result["verdict"]["commitment_match"] is True
    assert result["wallet"]["conserved"] is True
    assert result["experiments"] == len(result["charges"])
    assert result["identified"] is True
