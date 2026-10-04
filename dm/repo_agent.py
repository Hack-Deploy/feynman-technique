"""Agents submitted as repositories: fetch the repo, then play its agent.py one round at a time.

Contract (see agents/README.md): the repo has ``agent.py`` at its root. Each round the market runs
it with one JSON object on stdin (the bounty, prices, seed, system prompt and the conversation so
far) and reads its reply, the tagged protocol text, from stdout. Each round is a fresh process, so
an agent keeps no memory except what the conversation holds.

Isolation is best effort and not a security boundary (dm/_repo_agent_child.py): an audit hook
refuses the checker, the truth, the simulator, the answer files, network access, subprocesses and
writes outside the repo; the environment carries no API keys; every round has a time limit. Run
untrusted repos in a real sandbox (a container or VM) before opening this beyond a local demo.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "agents" / "examples"
CHECKOUTS = ROOT / "attempts" / "agent_repos"
CHILD = Path(__file__).resolve().parent / "_repo_agent_child.py"
ROUND_TIMEOUT = 60
CLONE_TIMEOUT = 90
MAX_REPLY_CHARS = 20_000
_URL = re.compile(r"^https://[A-Za-z0-9.-]+(:\d+)?(/[A-Za-z0-9._~-]+)+$")
_BARE_HOST = re.compile(r"^[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)+/")


@dataclass(frozen=True)
class AgentRepo:
    source: str  # https URL or examples/<name>
    path: Path
    commit: str


def normalise(repo: str) -> str:
    """An https git URL (``github.com/you/agent`` is read as https) or ``examples/<name>``."""
    repo = str(repo).strip().rstrip("/")
    if repo.startswith("examples/"):
        name = repo.removeprefix("examples/")
        if not re.fullmatch(r"[a-z0-9-]+", name) or not (EXAMPLES / name / "agent.py").is_file():
            raise ValueError(f"No example agent called {name!r}.")
        return repo
    if _BARE_HOST.match(repo):
        repo = "https://" + repo
    if not _URL.match(repo):
        raise ValueError("Give an https git URL, e.g. https://github.com/you/your-agent.")
    return repo


def _git(args: list[str]) -> str:
    env = {"PATH": os.environ.get("PATH", ""), "GIT_TERMINAL_PROMPT": "0",
           "GIT_CONFIG_NOSYSTEM": "1", "HOME": str(CHECKOUTS)}
    proc = subprocess.run(
        ["git", "-c", "protocol.allow=never", "-c", "protocol.https.allow=always", *args],
        capture_output=True, text=True, timeout=CLONE_TIMEOUT, env=env)
    if proc.returncode != 0:
        lines = (proc.stderr or "").strip().splitlines()
        raise ValueError("Could not fetch the repo" + (f": {lines[-1]}" if lines else "."))
    return proc.stdout


def _slug(url: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", url.lower().removeprefix("https://")).strip("-")[:60]


def fetch(repo: str) -> AgentRepo:
    """A checkout of the repo's default branch (shallow), pinned to its commit."""
    source = normalise(repo)
    if source.startswith("examples/"):
        path = EXAMPLES / source.removeprefix("examples/")
        return AgentRepo(source, path, hashlib.sha256((path / "agent.py").read_bytes()).hexdigest()[:12])
    CHECKOUTS.mkdir(parents=True, exist_ok=True)
    tmp = CHECKOUTS / f".clone-{uuid.uuid4().hex}"
    try:
        _git(["clone", "--depth", "1", "--quiet", "--", source, str(tmp)])
        commit = _git(["-C", str(tmp), "rev-parse", "HEAD"]).strip()[:12]
        if not (tmp / "agent.py").is_file():
            raise ValueError("The repo has no agent.py at its root.")
        dest = CHECKOUTS / f"{_slug(source)}-{commit}"
        if not dest.exists():
            tmp.rename(dest)
        return AgentRepo(source, dest, commit)
    except subprocess.TimeoutExpired:
        raise ValueError(f"Fetching the repo took longer than {CLONE_TIMEOUT} s.") from None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _withdraw(reason: str) -> str:
    return ("<assessment>The agent could not reply.</assessment><p_success>0</p_success>"
            f"<planned_cost>0</planned_cost><withdraw>{reason[:300]}</withdraw>")


class RepoAgent:
    """``complete(model, messages, system, max_tokens) -> str`` backed by a repo's agent.py."""

    def __init__(self, repo: AgentRepo, request: dict):
        self.repo, self.request = repo, request
        self.calls = 0
        self.errors: list[str] = []

    def __call__(self, model: str, messages: list[dict], system: str | None = None,
                 max_tokens: int = 4096) -> str:
        self.calls += 1
        payload = {**self.request, "call": self.calls, "system": system, "messages": messages}
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0",
               "HOME": str(self.repo.path), "DM_AGENT": "1"}
        try:
            proc = subprocess.run([sys.executable, "-I", str(CHILD), str(self.repo.path), str(ROOT)],
                                  input=json.dumps(payload), capture_output=True, text=True,
                                  timeout=ROUND_TIMEOUT, cwd=self.repo.path, env=env)
        except subprocess.TimeoutExpired:
            return self._fail(f"agent timed out after {ROUND_TIMEOUT} s")
        if proc.returncode != 0 or not proc.stdout.strip():
            lines = (proc.stderr or "").strip().splitlines()
            return self._fail("agent error: " + (lines[-1] if lines else "no reply on stdout"))
        return proc.stdout[:MAX_REPLY_CHARS]

    def _fail(self, reason: str) -> str:
        self.errors.append(reason)
        return _withdraw(reason)
