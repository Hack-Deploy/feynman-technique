"""Marketplace: submission checks, example agent repos judged and settled, boards, HTTP routes."""

import json
import threading
import time
import urllib.error
import urllib.request

import pytest
from types import SimpleNamespace

import app
import marketplace
from dm import repo_agent
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
    monkeypatch.setattr(repo_agent, "CHECKOUTS", tmp_path / "agent_repos")


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


CAREFUL, SHORTCUT, YES = "examples/careful-lab", "examples/shortcut-labs", "examples/yes-man"
GRAVITY = "gravity-inverse-square"


def _play(p, bounty, seed):
    return marketplace.run_one(p, repo_agent.fetch(p["repo"]), bounty, seed)


@pytest.mark.parametrize("args, message", [
    (("", CAREFUL, [GRAVITY], 1), "name"),
    (("x" * 41, CAREFUL, [GRAVITY], 1), "name"),
    (("---", CAREFUL, [GRAVITY], 1), "letter"),
    (("A", "examples/oracle", [GRAVITY], 1), "example"),
    (("A", "file:///etc", [GRAVITY], 1), "https"),
    (("A", "git@github.com:a/b.git", [GRAVITY], 1), "https"),
    (("A", "https://github.com/a/b;rm -rf", [GRAVITY], 1), "https"),
    (("A", CAREFUL, ["no-such-bounty"], 1), "bounty"),
    (("A", CAREFUL, [], 1), "bounty"),
    (("A", CAREFUL, [GRAVITY], 0), "seeds"),
    (("A", CAREFUL, [GRAVITY], 4), "seeds"),
    (("A", CAREFUL, [GRAVITY], True), "seeds"),
])
def test_plan_rejects_bad_submissions(args, message):
    with pytest.raises(ValueError, match=message):
        marketplace.plan(*args)


def test_plan_orders_runs_and_takes_next_free_seed():
    p = marketplace.plan("  Yes   Man ", YES, ["hubble-outward-push", GRAVITY], 2)
    assert p["name"] == "Yes Man" and p["solver"] == "agent:yes-man" and p["repo"] == YES
    assert p["runs"] == [{"bounty": GRAVITY, "seed": 0}, {"bounty": GRAVITY, "seed": 1},
                         {"bounty": "hubble-outward-push", "seed": 0},
                         {"bounty": "hubble-outward-push", "seed": 1}]
    _play(p, GRAVITY, 0)
    again = marketplace.plan("Yes Man", YES + "/", [GRAVITY], 1)
    assert again["runs"] == [{"bounty": GRAVITY, "seed": 1}]
    with pytest.raises(ValueError, match="another repo"):
        marketplace.plan("yes man", CAREFUL, [GRAVITY], 1)
    assert marketplace.plan("B", "github.com/you/agent", [GRAVITY], 1)["repo"] == \
        "https://github.com/you/agent"


def test_yes_man_tops_naive_board_and_falls_on_market_board():
    g = _play(marketplace.plan("Careful Lab", CAREFUL, [GRAVITY], 1), GRAVITY, 0)
    b = _play(marketplace.plan("Yes Man", YES, [GRAVITY], 1), GRAVITY, 0)
    assert g.passed and g.extra["agent_name"] == "Careful Lab" and g.extra["repo"] == CAREFUL
    assert len(g.extra["commit"]) == 12 and g.extra["agent_errors"] == []
    assert b.verdict["outcome"] == "false_claim" and b.experiments == 0
    assert b.extra["settlements"]["naive"]["paid"] and not b.extra["settlements"]["market"]["paid"]
    assert b.extra["settlements"]["market"]["profit"] < 0

    view = marketplace.venue("discoverphysics", GRAVITY)
    rows = {r["name"]: r for r in view["board"]}
    assert rows["Careful Lab"]["market_rank"] == 1 and rows["Yes Man"]["market_rank"] == 2
    assert rows["Yes Man"]["naive_rank"] == 1  # a guess costs nothing; naive pays it most
    assert rows["Yes Man"]["repo"] == YES and rows["Yes Man"]["strategy_label"].startswith(YES + " @ ")
    assert [r["agent"]["name"] for r in view["results"]] == ["Yes Man", "Careful Lab"]
    assert view["totals"] == {"agents": 2, "runs": 2}
    assert [x["repo"] for x in view["examples"]] == [CAREFUL, SHORTCUT, YES]
    assert "answer" not in json.dumps(view["bounties"])

    assert marketplace.venue("discoverphysics", "hubble-outward-push")["board"] == []
    assert marketplace.venue("nowhere") is None
    with pytest.raises(ValueError):
        marketplace.venue("discoverphysics", "no-such-bounty")
    card = marketplace.venues()["venues"][0]
    assert card["agents"] == 2 and card["false_claims"] == 1 and card["leader"] == "Careful Lab"


def test_shortcut_labs_p_hacks_and_the_checker_flags_it():
    r = _play(marketplace.plan("Shortcut Labs", SHORTCUT, [GRAVITY], 1), GRAVITY, 0)
    assert r.verdict["outcome"] == "false_claim" and r.extra["agent_verdict"] == "supported"
    assert {"rerun", "off_plan"} <= {f["flag"] for f in r.verdict["ruling"]["flags"]}
    st = r.extra["settlements"]
    assert st["naive"]["profit"] > 0 > st["market"]["profit"]


def test_a_crashing_repo_is_withdrawn_and_its_error_shown(tmp_path):
    repo = tmp_path / "crash"
    repo.mkdir()
    (repo / "agent.py").write_text("import poc.truth\n")
    p = marketplace.plan("Cheater", YES, [GRAVITY], 1)
    agent_repo = repo_agent.AgentRepo("https://example.org/cheater", repo, "abc123def456")
    r = marketplace.run_one(p, agent_repo, GRAVITY, 0)
    assert r.verdict["outcome"] in ("walked_away", "declined") and not r.passed
    assert "not allowed" in r.extra["agent_errors"][0]
    shown = marketplace.venue("discoverphysics")["results"][0]
    assert shown["agent_errors"] == r.extra["agent_errors"]


def test_records_from_before_repos_map_to_the_example_repos():
    old = SimpleNamespace(extra={"strategy": "p_hacker"})
    assert marketplace._repo_of(old) == SHORTCUT
    assert marketplace._repo_of(SimpleNamespace(extra={"repo": YES, "strategy": "careful"})) == YES


def test_pages_and_api(server):
    for path in ("/market", "/market/discoverphysics"):
        status, body = _request(server + path)
        assert status == 200 and b"Marketplace" in body
    assert _request(server + "/api/market/venues")[0] == 200
    assert _request(server + "/api/market/venue/discoverphysics")[0] == 200
    assert _request(server + "/api/market/venue/nowhere")[0] == 404
    assert _request(server + "/api/market/venue/discoverphysics?bounty=nope")[0] == 400
    assert _request(server + "/api/market/job?id=nope")[0] == 404
    for body in ({}, {"name": "A", "repo": YES, "bounties": GRAVITY},
                 {"name": "A", "repo": "file:///etc/passwd", "bounties": [GRAVITY]},
                 {"name": "A", "strategy": "careful", "bounties": [GRAVITY]}):
        assert _request(server + "/api/market/submit", "POST", json.dumps(body).encode())[0] == 400

    status, body = _request(server + "/api/market/submit", "POST", json.dumps(
        {"name": "Yes", "repo": YES, "bounties": [GRAVITY], "seeds": 2}).encode())
    started = json.loads(body)
    assert status == 200 and started["total"] == 2 and started["agent"]["repo"] == YES
    for _ in range(300):
        job = json.loads(_request(f"{server}/api/market/job?id={started['job_id']}")[1])
        if job["state"] != "running":
            break
        time.sleep(0.1)
    assert job["state"] == "done" and len(job["runs"]) == 2 and job["phase"] == "running"
    assert job["commit"] and all(r["agent"]["name"] == "Yes" for r in job["runs"])
    board = json.loads(_request(server + "/api/market/venue/discoverphysics")[1])["board"]
    assert [r["name"] for r in board] == ["Yes"] and board[0]["runs"] == 2


def test_a_repo_that_cannot_be_fetched_ends_the_job_with_an_error(monkeypatch):
    def fail(args):
        raise ValueError("Could not fetch the repo: not found")
    monkeypatch.setattr(repo_agent, "_git", fail)
    started = marketplace.submit("Ghost", "https://github.com/nobody/nothing", [GRAVITY], 1)
    for _ in range(100):
        job = marketplace.job(started["job_id"])
        if job["state"] != "running":
            break
        time.sleep(0.05)
    assert job["state"] == "error" and "not found" in job["error"] and job["runs"] == []
    assert not marketplace.SUBMISSIONS_PATH.exists()


def test_house_agents_and_bounty_cards_come_from_the_bench_store():
    from dm.store import AttemptStore
    from poc import baselines, bench
    from poc.attempt import run_attempt

    cfg = C.load()
    hyp = cfg.hypothesis(GRAVITY)
    model = baselines.PREFIX + "always_supported"
    s = run_attempt(model, hyp.id, 0, [], cfg=cfg, complete=baselines.make(model, cfg, hyp, 0))
    AttemptStore(C.ATTEMPTS_PATH).append([bench.resolve(hyp, s, cfg)])
    _play(marketplace.plan("Careful Lab", CAREFUL, [hyp.id], 1), hyp.id, 0)

    view = marketplace.venue("discoverphysics")
    house = next(r for r in view["board"] if r["kind"] == "house")
    assert house["name"] == "Always yes" and house["naive_rank"] == 1 and house["market_rank"] == 2
    card = next(b for b in view["bounties"] if b["id"] == hyp.id)
    assert card["title"] == "Does gravity fall off as 1/r²?" and card["glyph"] == "1/r²"
    assert card["agents"] == 2 and card["claims"] == {"supported": 1, "refuted": 1}
    assert card["confidence"] == 0.9
    assert view["results"][0]["agent"]["name"] == "Careful Lab"  # submissions lead the feed
