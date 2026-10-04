"""Repo agents: URL checks, fetching (git faked, no network), and the per-round sandbox."""

import json
import subprocess
from pathlib import Path

import pytest

from dm import repo_agent

REQUEST = {"protocol": "discoverphysics_market", "seed": 0, "bounty": {"id": "b"}, "prices": {}}


def _agent(tmp_path: Path, code: str) -> repo_agent.RepoAgent:
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    (repo / "agent.py").write_text(code)
    return repo_agent.RepoAgent(repo_agent.AgentRepo("https://example.org/a", repo, "0" * 12), REQUEST)


def _say(agent: repo_agent.RepoAgent) -> str:
    return agent("agent:x", [{"role": "user", "content": "go"}], system="sys")


@pytest.mark.parametrize("repo, expected", [
    ("https://github.com/you/agent", "https://github.com/you/agent"),
    ("https://github.com/you/agent.git/", "https://github.com/you/agent.git"),
    ("github.com/you/agent", "https://github.com/you/agent"),
    ("examples/yes-man", "examples/yes-man"),
])
def test_normalise_accepts_https_and_examples(repo, expected):
    assert repo_agent.normalise(repo) == expected


@pytest.mark.parametrize("repo", ["", "http://github.com/a/b", "file:///etc", "ssh://h/a",
                                  "git@github.com:a/b", "https://h/a b", "https://h/--upload-pack=x;y",
                                  "examples/../poc", "examples/nope", "/etc/passwd", "-c x"])
def test_normalise_rejects_everything_else(repo):
    with pytest.raises(ValueError):
        repo_agent.normalise(repo)


def test_fetch_clones_once_and_pins_the_commit(monkeypatch, tmp_path):
    monkeypatch.setattr(repo_agent, "CHECKOUTS", tmp_path)
    calls = []

    def git(args):
        calls.append(args)
        if args[0] == "clone":
            dest = Path(args[-1])
            dest.mkdir()
            (dest / "agent.py").write_text("print('<withdraw>x</withdraw>')\n")
            return ""
        return "0123456789abcdef\n"

    monkeypatch.setattr(repo_agent, "_git", git)
    got = repo_agent.fetch("github.com/you/agent")
    assert got.source == "https://github.com/you/agent" and got.commit == "0123456789ab"
    assert got.path == tmp_path / "github-com-you-agent-0123456789ab" and (got.path / "agent.py").is_file()
    assert calls[0][:5] == ["clone", "--depth", "1", "--quiet", "--"]
    assert repo_agent.fetch("https://github.com/you/agent").path == got.path
    assert [p.name for p in tmp_path.iterdir()] == [got.path.name]  # temp clones cleaned up


def test_fetch_needs_agent_py_and_reports_git_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(repo_agent, "CHECKOUTS", tmp_path)
    monkeypatch.setattr(repo_agent, "_git", lambda a: Path(a[-1]).mkdir() if a[0] == "clone" else "ab")
    with pytest.raises(ValueError, match="agent.py"):
        repo_agent.fetch("https://github.com/you/empty")
    assert list(tmp_path.iterdir()) == []

    def timeout(args):
        raise subprocess.TimeoutExpired("git", 1)
    monkeypatch.setattr(repo_agent, "_git", timeout)
    with pytest.raises(ValueError, match="longer than"):
        repo_agent.fetch("https://github.com/you/slow")


def test_example_repos_are_pinned_by_content():
    a = repo_agent.fetch("examples/yes-man")
    assert a.path == repo_agent.EXAMPLES / "yes-man" and len(a.commit) == 12
    assert repo_agent.fetch("examples/yes-man").commit == a.commit


def test_agent_reads_the_round_on_stdin_and_replies_on_stdout(tmp_path):
    agent = _agent(tmp_path, "import json, sys, numpy, scipy.optimize\n"
                             "from poc import estimate, reference, pricing\n"
                             "r = json.load(sys.stdin)\n"
                             "open('notes.txt', 'w').write('ok')\n"  # its own repo is writable
                             "print(json.dumps([r['call'], r['system'], r['messages'][0]['content'], r['seed']]))\n")
    assert json.loads(_say(agent)) == [1, "sys", "go", 0]
    assert json.loads(_say(agent))[0] == 2 and agent.errors == []
    assert (agent.repo.path / "notes.txt").read_text() == "ok"


@pytest.mark.parametrize("code, error", [
    ("import poc.truth", "not allowed"),
    ("import poc.checker", "not allowed"),
    ("from dm.oracle import judge", "not allowed"),
    ("import anthropic", "not allowed"),
    ("open(r'{root}/poc/config.yaml').read()", "not allowed"),
    ("open(r'{root}/attempts/poc_dp_bench.jsonl').read()", "not allowed"),
    ("import os; os.listdir('.'); open(r'{root}/vendor/x')", "not allowed"),
    ("open(r'{root}/dm/hacked.py', 'w')", "not allowed"),
    ("import os; os.remove(r'{root}/README.md')", "not allowed"),
    ("import subprocess; subprocess.run(['ls'])", "not allowed"),
    ("import os; os.system('ls')", "not allowed"),
    ("import socket; socket.create_connection(('example.org', 80))", "not allowed"),
    ("raise SystemExit(3)", "agent error"),
])
def test_sandbox_withdraws_agents_that_break_the_rules(tmp_path, code, error):
    agent = _agent(tmp_path, code.format(root=repo_agent.ROOT) + "\nprint('<withdraw>fine</withdraw>')\n")
    reply = _say(agent)
    assert "<withdraw>" in reply and "fine" not in reply
    assert len(agent.errors) == 1 and error in agent.errors[0]


def test_an_agent_that_prints_nothing_is_withdrawn(tmp_path):
    agent = _agent(tmp_path, "pass\n")
    assert "<withdraw>" in _say(agent) and "no reply" in agent.errors[0]


def test_a_slow_round_times_out(monkeypatch, tmp_path):
    monkeypatch.setattr(repo_agent, "ROUND_TIMEOUT", 1)
    agent = _agent(tmp_path, "import time\ntime.sleep(10)\n")
    assert "<withdraw>" in _say(agent) and "timed out" in agent.errors[0]


def test_secrets_never_reach_the_agent(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-secret")
    agent = _agent(tmp_path, "import os\nprint(repr(sorted(os.environ)))\n")
    keys = _say(agent)
    assert "ANTHROPIC" not in keys and "sk-test" not in keys
