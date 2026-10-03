"""Core records shared by the oracle, venues, solvers and markets."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields
from numbers import Real
from typing import Any

import numpy as np


# Canonical fields collected for per-attempt LLM usage.
LLM_USAGE_KEYS = ("calls", "input_tokens", "output_tokens", "usd", "estimated")


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


def _to_builtin(obj: Any) -> Any:
    """Convert numpy values and tuples to JSON-compatible builtin values."""
    if isinstance(obj, np.generic):
        return _to_builtin(obj.item())
    if isinstance(obj, np.ndarray):
        return _to_builtin(obj.tolist())
    if isinstance(obj, tuple):
        return [_to_builtin(item) for item in obj]
    if isinstance(obj, list):
        return [_to_builtin(item) for item in obj]
    if isinstance(obj, dict):
        if any(not isinstance(key, str) for key in obj):
            raise ValueError("dictionary keys must be strings")
        return {key: _to_builtin(value) for key, value in obj.items()}
    return obj


def _validate_attempt_values(
    stated_p_success: float | None, rounds: int, experiments: int, lab_cost: float
) -> None:
    if stated_p_success is not None:
        if (
            isinstance(stated_p_success, bool)
            or not isinstance(stated_p_success, Real)
            or not math.isfinite(stated_p_success)
            or not 0 <= stated_p_success <= 1
        ):
            raise ValueError("stated_p_success must be None or a finite number in [0, 1]")
    for name, value in (("rounds", rounds), ("experiments", experiments)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be an integer >= 0")
    if (
        isinstance(lab_cost, bool)
        or not isinstance(lab_cost, Real)
        or not math.isfinite(lab_cost)
        or lab_cost < 0
    ):
        raise ValueError("lab_cost must be finite and >= 0")


class InsufficientCredits(RuntimeError):
    """Raised by a venue when a wallet cannot pay for the requested experiments.

    The vendor agent loop catches exceptions from ``executor.run`` and shows them to
    the solver as an error message, so the solver sees this and must submit.
    """

    def __init__(self, needed: float, balance: float, count: int, price: float):
        self.needed = needed
        self.balance = balance
        self.count = count
        self.price = price
        super().__init__(
            f"insufficient credits: {count} experiment(s) x {price} = {needed} credits needed, "
            f"balance {balance}. Submit your <final_law> or run fewer experiments."
        )

    def __reduce__(self):
        return (type(self), (self.needed, self.balance, self.count, self.price))


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

    def __post_init__(self) -> None:
        _validate_attempt_values(
            self.stated_p_success, self.rounds, self.experiments, self.lab_cost
        )
        for name in ("llm_usage", "verdict", "extra"):
            value = json.loads(canonical_json(_to_builtin(getattr(self, name))))
            object.__setattr__(self, name, value)

    @property
    def passed(self) -> bool:
        return bool(self.verdict.get("passed", False))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AttemptRecord:
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"unknown AttemptRecord keys: {', '.join(unknown)}")
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
