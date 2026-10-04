"""Child process for one round of a repo agent: ``python -I _repo_agent_child.py <repo> <root>``.

Runs ``<repo>/agent.py`` as ``__main__`` with the round's JSON on stdin; whatever it prints is the
reply. Isolation (best effort; not a security boundary):
* a few libraries an agent may use are loaded first (numpy, scipy, and the open measurement
  helpers poc.estimate, poc.reference and poc.pricing);
* an audit hook, which Python code cannot remove, then refuses imports of the checker, the truth,
  the oracle and the simulator; reading the answer files, the attempt store, the vendor tree or
  /proc; writing or deleting outside the repo; starting processes; and network access;
* the parent passes no API keys and enforces a time limit.
"""

from __future__ import annotations

import importlib
import os
import runpy
import sys

BLOCKED_MODULES = ("poc.truth", "poc.checker", "poc.calibrate", "poc.ledger", "dm.oracle",
                   "scienceagent", "physchool", "jax", "jaxlib", "anthropic")
BLOCKED_EVENTS = ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.fork",
                  "os.forkpty", "pty.spawn", "socket.connect", "socket.bind", "socket.getaddrinfo",
                  "socket.sendto", "ctypes.dlopen", "webbrowser.open")
WRITE_EVENTS = ("os.remove", "os.rename", "os.rmdir", "os.mkdir", "os.chmod", "os.symlink",
                "os.link", "os.truncate", "shutil.rmtree", "shutil.move", "shutil.copyfile")
PRELOAD = ("json", "math", "re", "numpy", "scipy.optimize", "scipy.special", "poc.estimate",
           "poc.reference", "poc.pricing")


def _blocked_paths(root: str) -> tuple[str, ...]:
    j = lambda *p: os.path.join(root, *p)  # noqa: E731
    return (j("attempts"), j("vendor"), j("output"), j("poc", "config.yaml"), j("poc", "truth.py"),
            j("poc", "checker.py"), j("poc", "calibrate.py"), j(".env"), j("poc", ".env"), "/proc")


def _install(repo: str, root: str) -> None:
    blocked = _blocked_paths(root)
    inside = lambda p: p == repo or p.startswith(repo + os.sep)  # noqa: E731

    def real(arg) -> str | None:
        if isinstance(arg, (str, bytes, os.PathLike)):
            return os.path.realpath(os.fsdecode(arg))
        return None

    def hook(event: str, args: tuple) -> None:
        if event == "import" and args:
            name = str(args[0])
            if any(name == m or name.startswith(m + ".") for m in BLOCKED_MODULES):
                raise ImportError(f"import of {name} is not allowed for market agents")
        elif event == "open" and args:
            path = real(args[0])
            if path is None or inside(path):
                return
            if path.startswith(blocked):
                raise PermissionError(f"reading {path} is not allowed for market agents")
            mode = args[1] if len(args) > 1 and isinstance(args[1], str) else "r"
            flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
            if any(c in mode for c in "wax+") or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
                raise PermissionError(f"writing {path} is not allowed for market agents")
        elif event in WRITE_EVENTS and args:
            path = real(args[0])
            if path is not None and not inside(path):
                raise PermissionError(f"{event} outside the repo is not allowed for market agents")
        elif event in BLOCKED_EVENTS:
            raise PermissionError(f"{event} is not allowed for market agents")

    sys.addaudithook(hook)


def main() -> None:
    repo, root = (os.path.realpath(a) for a in sys.argv[1:3])
    sys.dont_write_bytecode = True  # -I ignores PYTHONDONTWRITEBYTECODE
    sys.path[:0] = [repo, root]
    for name in PRELOAD:
        try:
            importlib.import_module(name)
        except ImportError:
            pass
    _install(repo, root)
    sys.argv = [os.path.join(repo, "agent.py")]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
