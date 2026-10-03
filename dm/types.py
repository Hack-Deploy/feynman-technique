"""Core records shared by the oracle, venues, solvers and markets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any


def _json_default(o: Any) -> Any:
    # numpy scalars/arrays (oracle test cases are built with numpy RNGs)
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, no NaN/Infinity.

    numpy scalars and arrays are converted with ``tolist()`` so they hash like the
    equivalent Python values (float32 widens to the float64 of the same value)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False,
                      default=_json_default)


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
    # >=128-bit secret nonce chosen by the oracle; without it a commitment over cases
    # derived from a small seed can be brute-forced. Hidden until reveal, like test_seed.
    salt: str = ""

    HIDDEN = ("test_cases", "test_seed", "salt")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def commitment(self) -> str:
        """sha256 over the canonical JSON of every field."""
        return hashlib.sha256(canonical_json(self.to_dict()).encode()).hexdigest()

    def public(self) -> dict[str, Any]:
        """Everything except the hidden fields, plus the commitment."""
        d = {k: v for k, v in self.to_dict().items() if k not in self.HIDDEN}
        d["commitment"] = self.commitment()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Preregistration:
        return cls(**d)

    @classmethod
    def verify(cls, revealed: dict[str, Any], commitment: str) -> Preregistration:
        """Rebuild a revealed preregistration and check it against the posted commitment."""
        p = cls.from_dict(revealed)
        if p.commitment() != commitment:
            raise ValueError("revealed preregistration does not match its commitment")
        return p


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
