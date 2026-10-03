"""Sandbox child: run a submitted law on the hidden cases' initial conditions only.

Launched by the trusted scoring worker (``dm.oracle._worker``). It never sees the
ground truth: the vendor evaluator runs against a stand-in executor whose
trajectories are all zeros, and only the law's predictions are sent back. The
trusted worker computes the errors against the real ground truth.

Isolation (best effort; not a security boundary):
* the real simulator is never imported here: ``scienceagent.evaluator`` is
  loaded under a stand-in ``scienceagent.executor`` module, bypassing the
  package ``__init__`` (which would import the simulator);
* an audit hook, which Python code cannot remove, then refuses imports of
  ``scienceagent``, ``physchool`` and ``jax``, reading files under the vendor
  tree or ``/proc``, starting processes, and opening network connections;
* the environment carries no API keys or oracle secret.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
import types

import numpy as np

EXECUTOR_NAMES = ("SimulationExecutor", "SpeciesExecutor", "ThreeSpeciesExecutor",
                  "DarkMatterExecutor", "NBodyEtherExecutor", "NBodyHubbleExecutor")
BLOCKED_MODULES = ("scienceagent", "physchool", "jax", "jaxlib")
BLOCKED_EVENTS = ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.fork",
                  "os.forkpty", "pty.spawn", "socket.connect", "socket.getaddrinfo",
                  "socket.sendto", "ctypes.dlopen")
# Mirrors _FIT_WORLDS in vendor ScienceAgent/run_discovery.py.
FIT_WORLDS = {"gravity", "yukawa", "fractional", "oscillator", "extra_dimensions",
              "circle", "ether", "hubble"}


def _load_vendor_evaluator():
    """Import scienceagent.evaluator without the simulator behind it."""
    spec = importlib.util.find_spec("scienceagent")
    pkg = types.ModuleType("scienceagent")
    pkg.__path__ = list(spec.submodule_search_locations)
    sys.modules["scienceagent"] = pkg
    stub = types.ModuleType("scienceagent.executor")
    for name in EXECUTOR_NAMES:
        setattr(stub, name, type(name, (), {}))
    sys.modules["scienceagent.executor"] = stub
    import scienceagent.evaluator as E  # noqa: E402
    vendor_root = os.path.realpath(os.path.join(pkg.__path__[0], "..", ".."))
    return E, vendor_root


class StandInExecutor:
    """Answers the evaluator's calls with zero trajectories and agent-visible attributes."""

    def __init__(self, outputs: list[dict], full_outputs: list[dict] | None, attrs: dict):
        self._outputs = outputs
        self._full = full_outputs
        for k, v in attrs.items():
            setattr(self, k, np.asarray(v) if isinstance(v, list) else v)

    @contextlib.contextmanager
    def noise_disabled(self):
        yield

    def run(self, experiments):
        if len(experiments) == len(self._outputs):
            return self._outputs
        # Extra calls (e.g. CircleEvaluator's unused t=0 probe): same shapes, zeros.
        return [self._outputs[0] for _ in experiments]

    def run_full(self, experiments):
        return self._full or []


def _install_audit_hook(vendor_root: str) -> None:
    def hook(event, args):
        if event == "import" and args and str(args[0]).split(".")[0] in BLOCKED_MODULES:
            raise ImportError(f"import of {args[0]} is not allowed in the oracle sandbox")
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            path = os.path.realpath(os.fsdecode(args[0]))
            if path.startswith((vendor_root, "/proc")):
                raise PermissionError(f"reading {path} is not allowed in the oracle sandbox")
        if event in BLOCKED_EVENTS:
            raise PermissionError(f"{event} is not allowed in the oracle sandbox")
    sys.addaudithook(hook)


def _evaluator(E, world: str, executor, cases):
    cls = {
        "circle": E.CircleEvaluator,
        "three_species": E.ThreeSpeciesEvaluator,
        "dark_matter": E.DarkMatterEvaluator,
        "ether": E.EtherEvaluator,
        "hubble": E.HubbleEvaluator,
    }.get(world, E.Evaluator)
    return cls(executor, test_cases=cases)


def _plain(x):
    """JSON-safe predictions (numpy → lists; non-finite floats kept, parent checks)."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    return x


def main() -> None:
    job = json.loads(sys.stdin.read())
    out = sys.stdout
    E, vendor_root = _load_vendor_evaluator()
    executor = StandInExecutor(job["outputs"], job.get("full_outputs"), job.get("attrs", {}))
    evaluator = _evaluator(E, job["world"], executor, job["cases"])
    # verbose=True makes the vendor evaluator print per-case exceptions (it swallows
    # them otherwise); stdout is captured below and only ERROR lines are kept.
    kwargs = {"verbose": True}
    if job["world"] in FIT_WORLDS and job.get("training"):
        kwargs["training_trajectories"] = job["training"]

    _install_audit_hook(vendor_root)
    result = {"ok": True, "reason": None}
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            ev = evaluator.evaluate(job["law_source"], **kwargs)
        result["preds"] = [_plain(t.get("pred")) for t in ev.get("trajectories", [])]
        result["fit"] = _plain(ev.get("fit"))
        # The vendor evaluator prints per-case exceptions instead of raising them.
        errors = [ln.strip() for ln in log.getvalue().splitlines() if "ERROR" in ln]
        if errors:
            result["reason"] = "; ".join(dict.fromkeys(errors))[:500]
    except Exception as e:  # compile errors, missing discovered_law, failed fit, ...
        result.update({"ok": False, "preds": None, "reason": f"{type(e).__name__}: {e}"[:500]})
    out.write(json.dumps(result, default=str))
    out.flush()


if __name__ == "__main__":
    main()
