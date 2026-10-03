"""The oracle: preregisters hidden tests, then scores submitted laws against them.

Independent of solvers by construction: only ``dm.settle`` may import this
package (enforced by a test), and submitted code runs only in a subprocess with
a wall-clock timeout, no network and no API keys in its environment.

Settlement is numeric: normalised MSE < threshold on the hidden cases, where the
MSE is the vendor evaluator's ``mean_pos_error`` and the normaliser is frozen in
the preregistration. The explanation score is optional and never settles.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from dm.oracle.cases import TWO_PARTICLE_WORLDS, hidden_cases, jsonable
from dm.types import Preregistration

ORACLE_VERSION = "dm-oracle-1.0 (discovery-agents@450818fa)"
DEFAULT_TIMEOUT_S = 120.0
VENUES = {"discoverphysics", "forcebench"}
ROOT = Path(__file__).resolve().parent.parent.parent

# Environment variables never passed to the scoring subprocess.
_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


@lru_cache(maxsize=64)
def make_prereg(venue: str, world: str, test_seed: int) -> Preregistration:
    """Generate the hidden test set for (venue, world, test_seed) and freeze it."""
    if venue not in VENUES:
        raise ValueError(f"unknown venue {venue!r}")
    if venue == "forcebench" and world not in TWO_PARTICLE_WORLDS:
        raise ValueError(f"ForceBench has only two-particle worlds, not {world!r}")
    cases, norm_var, public = hidden_cases(world, test_seed)
    return Preregistration(
        question_id=f"{venue}/{world}",
        venue=venue,
        world=world,
        test_seed=test_seed,
        test_cases=jsonable(cases),
        norm_variance=float(norm_var),
        oracle_version=ORACLE_VERSION,
        public_tests=public,
    )


def _scrubbed_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if not any(m in k.upper() for m in _SECRET_MARKERS)}
    env["PYTHONHASHSEED"] = "0"
    env["JAX_PLATFORMS"] = env.get("JAX_PLATFORMS", "cpu")
    return env


def score(prereg: Preregistration, law_source: str | None,
          training: list | None = None,
          timeout_s: float = DEFAULT_TIMEOUT_S) -> dict:
    """Score a law on the preregistered hidden cases. Any failure is a fail."""
    verdict = {
        "normalised_mse": None,
        "mean_pos_error": None,
        "passed": False,
        "prereg_commitment": prereg.commitment(),
        "threshold": prereg.threshold,
        "metric": prereg.metric,
        "public_tests": prereg.public_tests,
        "oracle_version": prereg.oracle_version,
        "explanation_score": None,
        "reason": None,
    }
    if not law_source or not law_source.strip():
        verdict["reason"] = "no law submitted"
        return verdict

    job = {"world": prereg.world, "test_cases": prereg.test_cases,
           "law_source": law_source, "training": training or []}
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "dm.oracle._worker"],
            input=json.dumps(job), capture_output=True, text=True,
            timeout=timeout_s, cwd=ROOT, env=_scrubbed_env(),
        )
    except subprocess.TimeoutExpired:
        verdict["reason"] = f"timeout after {timeout_s:g}s"
        return verdict

    try:
        result = json.loads(proc.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        verdict["reason"] = f"oracle worker crashed (exit {proc.returncode}): {proc.stderr[-400:]}"
        return verdict

    mse = result.get("mean_pos_error")
    verdict["mean_pos_error"] = mse
    verdict["per_case"] = result.get("per_case")
    verdict["reason"] = result.get("reason")
    if mse is not None and math.isfinite(mse):
        nmse = mse / prereg.norm_variance
        verdict["normalised_mse"] = nmse
        verdict["passed"] = bool(nmse < prereg.threshold)
    return verdict


def explain_score(world: str, explanation: str | None,
                  judge_model: str = "claude-opus-4-6") -> float | None:
    """Optional LLM-judged explanation score. Never settles; off unless ENABLE_LIVE=1."""
    if os.environ.get("ENABLE_LIVE") != "1" or not os.environ.get("DM_MAX_USD"):
        return None
    if not explanation:
        return None
    from scienceagent.evaluator import ExplanationJudge
    from dm.oracle.cases import build_world
    w = build_world(world)
    out = ExplanationJudge(judge_model=judge_model).score(
        agent_explanation=explanation,
        optimal_explanation=w["optimal_explanation"],
        rubric=w["explanation_rubric"],
        verbose=False,
    )
    return out.get("score")
