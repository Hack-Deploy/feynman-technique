"""Core records shared by the oracle, venues, solvers and markets."""

from __future__ import annotations

import hashlib
import json
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def commitment(self) -> str:
        """sha256 over the canonical JSON of every field."""
        return hashlib.sha256(canonical_json(self.to_dict()).encode()).hexdigest()

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

    @property
    def passed(self) -> bool:
        return bool(self.verdict.get("passed", False))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AttemptRecord:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass(frozen=True)
class SubmittedAttempt:
    """What a venue hands to ``dm.settle``: everything except the verdict."""

    source: str
    protocol: str
    venue: str
    world: str
    solver: str
    seed: int
    stated_p_success: float | None
    rounds: int
    experiments: int
    lab_cost: float
    llm_usage: dict = field(default_factory=dict)
    submitted_law: str | None = None
    explanation: str | None = None
    training: list = field(default_factory=list)  # experiments the solver paid for
    transcript_path: str | None = None
    created_at: str = ""
    extra: dict = field(default_factory=dict)
