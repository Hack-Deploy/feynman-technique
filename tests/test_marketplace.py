"""Marketplace: submission checks, scripted agent runs judged and settled, boards, HTTP routes."""

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

import app
import marketplace
from poc import config as C


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "ROOT", tmp_path)
    monkeypatch.setattr(C, "ATTEMPTS_PATH", tmp_path / "bench.jsonl")
    monkeypatch.setattr(C, "TRANSCRIPTS_DIR", tmp_path / "transcripts")
    monkeypatch.setattr(C, "TRAJECTORIES_DIR", tmp_path / "trajectories")
    monkeypatch.setattr(marketplace, "SUBMISSIONS_PATH", tmp_path / "submissions.jsonl")
    monkeypatch.setattr(marketplace, "_JOBS", {})
    monkeypatch.setattr(marketplace, "_ACTIVE_JOB", None)


@pytest.fixture
def server():
    srv = app.ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{srv.server_port}"
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)


def _request(url, method="GET", body=None):
    req = urllib.request.Request(url, data=body, method=method,
                                 headers={"Content-Type": "application/json"} if body else {})
    try:
        with urllib.request.urlopen(req, timeout=30) as res:
            return res.status, res.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


@pytest.mark.parametrize("args, message", [
    (("", "careful", ["gravity-inverse-square"], 1), "name"),
    (("x" * 41, "careful", ["gravity-inverse-square"], 1), "name"),
    (("---", "careful", ["gravity-inverse-square"], 1), "letter"),
    (("A", "oracle", ["gravity-inverse-square"], 1), "strategy"),
    (("A", "careful", ["no-such-bounty"], 1), "bounty"),
    (("A", "careful", [], 1), "bounty"),
    (("A", "careful", ["gravity-inverse-square"], 0), "seeds"),
    (("A", "careful", ["gravity-inverse-square"], 4), "seeds"),
    (("A", "careful", ["gravity-inverse-square"], True), "seeds"),
])
def test_plan_rejects_bad_submissions(args, message):
    with pytest.raises(ValueError, match=message):
        marketplace.plan(*args)


def test_plan_orders_runs_and_takes_next_free_seed():
    p = marketplace.plan("  Shortcut   Labs ", "guesser",
                         ["hubble-outward-push", "gravity-inverse-square"], 2)
    assert p["name"] == "Shortcut Labs" and p["solver"] == "agent:shortcut-labs"
    assert p["runs"] == [{"bounty": "gravity-inverse-square", "seed": 0},
                         {"bounty": "gravity-inverse-square", "seed": 1},
                         {"bounty": "hubble-outward-push", "seed": 0},
                         {"bounty": "hubble-outward-push", "seed": 1}]
    marketplace.run_one(p, "gravity-inverse-square", 0)
    again = marketplace.plan("Shortcut Labs", "guesser", ["gravity-inverse-square"], 1)
    assert again["runs"] == [{"bounty": "gravity-inverse-square", "seed": 1}]
    with pytest.raises(ValueError, match="another strategy"):
        marketplace.plan("shortcut labs", "careful", ["gravity-inverse-square"], 1)


def test_guesser_tops_naive_board_and_falls_on_market_board():
    good = marketplace.plan("Careful Lab", "careful", ["gravity-inverse-square"], 1)
    bad = marketplace.plan("Shortcut Labs", "guesser", ["gravity-inverse-square"], 1)
    g = marketplace.run_one(good, "gravity-inverse-square", 0)
    b = marketplace.run_one(bad, "gravity-inverse-square", 0)
    assert g.passed and g.extra["agent_name"] == "Careful Lab" and g.extra["strategy"] == "careful"
    assert b.verdict["outcome"] == "false_claim"
    assert b.extra["settlements"]["naive"]["paid"] and not b.extra["settlements"]["market"]["paid"]
    assert b.extra["settlements"]["market"]["profit"] < 0

    view = marketplace.venue("discoverphysics", "gravity-inverse-square")
    rows = {r["name"]: r for r in view["board"]}
    assert rows["Careful Lab"]["market_rank"] == 1 and rows["Shortcut Labs"]["market_rank"] == 2
    assert rows["Shortcut Labs"]["naive_rank"] == 1  # a guess costs nothing; naive pays it most
    assert [r["agent"]["name"] for r in view["results"]] == ["Shortcut Labs", "Careful Lab"]
    assert view["totals"] == {"agents": 2, "runs": 2}
    assert "answer" not in json.dumps(view["bounties"])

    assert marketplace.venue("discoverphysics", "hubble-outward-push")["board"] == []
    assert marketplace.venue("nowhere") is None
    with pytest.raises(ValueError):
        marketplace.venue("discoverphysics", "no-such-bounty")
    card = marketplace.venues()["venues"][0]
    assert card["agents"] == 2 and card["false_claims"] == 1 and card["leader"] == "Careful Lab"


def test_pages_and_api(server):
    for path in ("/market", "/market/discoverphysics"):
        status, body = _request(server + path)
        assert status == 200 and b"Marketplace" in body
    assert _request(server + "/api/market/venues")[0] == 200
    assert _request(server + "/api/market/venue/discoverphysics")[0] == 200
    assert _request(server + "/api/market/venue/nowhere")[0] == 404
    assert _request(server + "/api/market/venue/discoverphysics?bounty=nope")[0] == 400
    assert _request(server + "/api/market/job?id=nope")[0] == 404
    for body in ({}, {"name": "A", "strategy": "careful", "bounties": "gravity-inverse-square"},
                 {"name": "A", "strategy": "nope", "bounties": ["gravity-inverse-square"]}):
        assert _request(server + "/api/market/submit", "POST", json.dumps(body).encode())[0] == 400

    status, body = _request(server + "/api/market/submit", "POST", json.dumps(
        {"name": "Coin", "strategy": "coin_flip", "bounties": ["gravity-inverse-square"],
         "seeds": 2}).encode())
    started = json.loads(body)
    assert status == 200 and started["total"] == 2
    for _ in range(300):
        job = json.loads(_request(f"{server}/api/market/job?id={started['job_id']}")[1])
        if job["state"] != "running":
            break
        time.sleep(0.1)
    assert job["state"] == "done" and len(job["runs"]) == 2
    assert all(r["agent"]["name"] == "Coin" for r in job["runs"])
    board = json.loads(_request(server + "/api/market/venue/discoverphysics")[1])["board"]
    assert [r["name"] for r in board] == ["Coin"] and board[0]["runs"] == 2
