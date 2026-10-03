"""Core records shared by the oracle, venues, solvers and markets."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, no NaN/Infinity."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


class InsufficientCredits(RuntimeError):
    """Raised by a venue when a wallet cannot pay for the requested experiments.

    The vendor agent loop catches exceptions from ``executor.run`` and shows them to
    the solver as an error message, so the solver sees this and must submit.
    """


@dataclass(frozen=True)
class Preregistration:
    """What the oracle will score against, fixed when the prize is posted.

    Only ``commitment()`` is published at posting time; the full record (including
    the hidden ``test_cases``) is revealed when the prize closes.
    """

    question_id: str  # e.g. "discoverphysics/gravity" or "forcebench/yukawa"
    venue: str
    world: str
    test_seed: int
    test_cases: list[dict]  # hidden until reveal
    norm_variance: float
    oracle_version: str
    metric: str = "normalised_mse"
    threshold: float = 0.1
    public_tests: bool = False  # True if the world's default (public) cases had to be used

    def __post_init__(self) -> None:
        # Freeze the hidden cases (a deep copy) and the commitment at construction, so
        # nothing done to the caller's list later can change what was committed.
        frozen = tuple(json.loads(canonical_json(list(self.test_cases))))
        object.__setattr__(self, "test_cases", frozen)
        object.__setattr__(self, "_commitment", hashlib.sha256(
            canonical_json(self.to_dict()).encode()).hexdigest())

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["test_cases"] = json.loads(canonical_json(list(self.test_cases)))
        return d

    def commitment(self) -> str:
        """sha256 over the canonical JSON of every field (fixed at construction)."""
        return self._commitment

    def public(self) -> dict[str, Any]:
        """Everything except the hidden test cases, plus the commitment."""
        d = {k: v for k, v in self.to_dict().items() if k != "test_cases"}
        d["commitment"] = self.commitment()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Preregistration:
        return cls(**d)


@dataclass(frozen=True)
class AttemptRecord:
    """One attempt at one world by one solver: the unit the market replays."""

    attempt_id: str
    source: str  # live | published_replay | sim
    protocol: str  # discoverphysics_native | forcebench_menu | ara_harness
    venue: str
    world: str
    solver: str  # e.g. "claude-opus-4-7", "bayes_lite"
    seed: int
    stated_p_success: float | None
    rounds: int
    experiments: int
    lab_cost: float  # credits paid to the lab
    llm_usage: dict = field(default_factory=dict)  # tokens in/out, usd, estimated: bool
    submitted_law: str | None = None
    verdict: dict = field(default_factory=dict)  # normalised_mse, passed, prereg_commitment, explanation_score|None
    transcript_path: str | None = None
    created_at: str = ""  # ISO time, metadata only, never used in outcomes
    extra: dict = field(default_factory=dict)  # provenance, source-specific fields

    def __post_init__(self) -> None:
        for name in ("rounds", "experiments", "seed"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool):
                raise TypeError(f"{self.attempt_id}: {name} must be int, got {v!r}")
        if self.rounds < 0 or self.experiments < 0:
            raise ValueError(f"{self.attempt_id}: negative rounds/experiments")
        if not (isinstance(self.lab_cost, (int, float)) and math.isfinite(self.lab_cost)
                and self.lab_cost >= 0):
            raise ValueError(f"{self.attempt_id}: lab_cost must be finite and >= 0")
        p = self.stated_p_success
        if p is not None and not 0.0 <= p <= 1.0:
            raise ValueError(f"{self.attempt_id}: stated_p_success outside [0, 1]")
        if "passed" in self.verdict and not isinstance(self.verdict["passed"], bool):
            raise TypeError(f"{self.attempt_id}: verdict.passed must be bool")

    @property
    def passed(self) -> bool:
        return self.verdict.get("passed") is True

    @property
    def settled(self) -> bool:
        return isinstance(self.verdict.get("passed"), bool)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AttemptRecord:
        """Unknown keys (from newer writers) are kept under ``extra["_unknown"]`` so a
        load → append round trip never drops data."""
        known = {f.name for f in fields(cls)}
        kw = {k: v for k, v in d.items() if k in known}
        unknown = {k: v for k, v in d.items() if k not in known}
        if unknown:
            kw["extra"] = {**kw.get("extra", {}),
                           "_unknown": {**kw.get("extra", {}).get("_unknown", {}), **unknown}}
        return cls(**kw)
