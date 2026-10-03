"""Subprocess entry point: score one submitted law against hidden cases.

Reads a JSON job on stdin and writes one JSON result on stdout. Launched by
``dm.oracle.score`` with a wall-clock timeout and a scrubbed environment (no API
keys). Network access is disabled in-process before the law is compiled. This is
a guard rail, not a security boundary.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import socket
import sys

# Worlds whose evaluators fit ``fit_parameters()`` on the solver's own training
# data (mirrors ``_FIT_WORLDS`` in vendor ScienceAgent/run_discovery.py).
FIT_WORLDS = {"gravity", "yukawa", "fractional", "oscillator", "extra_dimensions",
              "circle", "ether", "hubble"}


def _no_network(*_a, **_k):
    raise OSError("network access is disabled inside the oracle")


def _block_network() -> None:
    socket.socket.connect = _no_network          # type: ignore[assignment]
    socket.socket.connect_ex = _no_network       # type: ignore[assignment]
    socket.create_connection = _no_network       # type: ignore[assignment]
    socket.getaddrinfo = _no_network             # type: ignore[assignment]


def _evaluator_for(world: str, executor, cases):
    import scienceagent.evaluator as E
    cls = {
        "circle": E.CircleEvaluator,
        "three_species": E.ThreeSpeciesEvaluator,
        "dark_matter": E.DarkMatterEvaluator,
        "ether": E.EtherEvaluator,
        "hubble": E.HubbleEvaluator,
    }.get(world, E.Evaluator)
    return cls(executor, test_cases=cases)


def _finite(x):
    return float(x) if x is not None and math.isfinite(float(x)) else None


def main() -> None:
    job = json.loads(sys.stdin.read())
    world = job["world"]
    from dm.oracle.cases import build_world  # noqa: E402  (after reading the job)

    executor = build_world(world)["executor"]
    evaluator = _evaluator_for(world, executor, job["test_cases"])
    kwargs = {"verbose": False}
    if world in FIT_WORLDS and job.get("training"):
        kwargs["training_trajectories"] = job["training"]

    _block_network()
    result = {"ok": True, "reason": None}
    try:
        # Laws may print; keep stdout clean for the JSON result.
        with contextlib.redirect_stdout(io.StringIO()):
            ev = evaluator.evaluate(job["law_source"], **kwargs)
        mse = _finite(ev.get("mean_pos_error"))
        result.update({
            "mean_pos_error": mse,
            "per_case": [_finite(x) for x in ev.get("per_case", [])],
            "fit": ev.get("fit"),
        })
        if mse is None:
            result["reason"] = "non-finite error (law raised or diverged on a test case)"
    except Exception as e:  # compile errors, missing discovered_law, ...
        result.update({"ok": False, "mean_pos_error": None, "per_case": [],
                       "reason": f"{type(e).__name__}: {e}"[:500]})
    sys.stdout.write(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
