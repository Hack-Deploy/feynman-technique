"""Pricing, spend projection, and the append-only live spend ledger."""

from __future__ import annotations

import json
import math
import os
import threading
import uuid
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import yaml

from poc.config import ROOT

LIVE_MODELS_PATH = Path(__file__).with_name("live_models.yaml")
LEDGER_PATH = ROOT / "attempts" / "live_spend.jsonl"
_FALLBACK_LOCKS: dict[str, threading.Lock] = {}
_FALLBACK_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True)
class ModelPrice:
    label: str
    id: str
    input: float
    output: float


@dataclass(frozen=True)
class LiveSettings:
    models: tuple[ModelPrice, ...]
    max_usd: float
    max_rounds: int
    max_tokens: int
    chars_per_token: float
    data_chars_per_round: int
    hypotheses: tuple[str, ...]
    seeds: tuple[int, ...]
    source: dict


def _finite_positive(value, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1")
    return value


def load_settings(path: Path = LIVE_MODELS_PATH) -> LiveSettings:
    data = yaml.safe_load(Path(path).read_text())
    models = []
    ids = set()
    for row in data.get("models", []):
        model_id = str(row.get("id", "")).strip()
        if not model_id or model_id in ids:
            raise ValueError(f"model ids must be non-empty and unique: {model_id!r}")
        ids.add(model_id)
        models.append(ModelPrice(
            label=str(row.get("label", "")).strip(),
            id=model_id,
            input=_finite_positive(row.get("input"), f"{model_id}.input"),
            output=_finite_positive(row.get("output"), f"{model_id}.output"),
        ))
    if not models:
        raise ValueError("live_models.yaml must define at least one model")

    live = data.get("live") or {}
    projection = data.get("projection") or {}
    demo = data.get("demo_grid") or {}
    hypotheses = tuple(str(item) for item in demo.get("hypotheses", []))
    seeds = tuple(demo.get("seeds", []))
    if not hypotheses:
        raise ValueError("demo_grid.hypotheses must not be empty")
    if not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("demo_grid.seeds must contain integers")
    return LiveSettings(
        models=tuple(models),
        max_usd=_finite_positive(live.get("max_usd"), "live.max_usd"),
        max_rounds=_positive_int(live.get("max_rounds"), "live.max_rounds"),
        max_tokens=_positive_int(live.get("max_tokens"), "live.max_tokens"),
        chars_per_token=_finite_positive(
            projection.get("chars_per_token"), "projection.chars_per_token"
        ),
        data_chars_per_round=_positive_int(
            projection.get("data_chars_per_round"), "projection.data_chars_per_round"
        ),
        hypotheses=hypotheses,
        seeds=seeds,
        source=dict(data.get("source") or {}),
    )


def price(settings: LiveSettings, model_id: str) -> ModelPrice:
    for model in settings.models:
        if model.id == model_id:
            return model
    raise KeyError(f"no price for model {model_id} in poc/live_models.yaml")


def usd_for_usage(model_price: ModelPrice, usage: dict) -> float:
    input_tokens = usage.get("input_tokens", 0) or 0
    output_tokens = usage.get("output_tokens", 0) or 0
    cache_creation = usage.get("cache_creation_input_tokens", 0) or 0
    cache_read = usage.get("cache_read_input_tokens", 0) or 0
    usd = (
        input_tokens * model_price.input
        + output_tokens * model_price.output
        + cache_creation * 1.25 * model_price.input
        + cache_read * model_price.input
    ) / 1_000_000
    return round(usd, 6)


def project_run_usd(
    model_price: ModelPrice,
    base_chars: int,
    max_rounds: int,
    max_tokens: int,
    chars_per_token: float,
    data_chars_per_round: int,
) -> dict:
    calls = 2 * max_rounds + 1
    base_tokens = math.ceil(base_chars / chars_per_token)
    growth = max_tokens + math.ceil(data_chars_per_round / chars_per_token)
    input_tokens = sum(base_tokens + call * growth for call in range(calls))
    output_tokens = calls * max_tokens
    usd = round(
        (input_tokens * model_price.input + output_tokens * model_price.output) / 1_000_000,
        6,
    )
    return {
        "calls": calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "usd": usd,
    }


def effective_cap(settings: LiveSettings) -> float | None:
    raw = os.environ.get("DM_MAX_USD")
    if raw is None:
        return None
    try:
        env_cap = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(env_cap) or env_cap <= 0:
        return None
    return min(env_cap, settings.max_usd)


def cap_note(settings: LiveSettings) -> str:
    raw = os.environ.get("DM_MAX_USD")
    try:
        env_cap = float(raw)
    except (TypeError, ValueError):
        return ""
    if math.isfinite(env_cap) and env_cap > settings.max_usd:
        return (
            f"DM_MAX_USD (${env_cap:g}) exceeds the configured hard cap "
            f"(${settings.max_usd:g}); the hard cap is applied."
        )
    return ""


class CapReached(RuntimeError):
    """The cumulative spend cap would be exceeded."""


def _canonical_json(entry: dict) -> str:
    return json.dumps(entry, sort_keys=True, separators=(",", ":"), allow_nan=False)


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    lock_path = Path(f"{path}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError:
        key = str(lock_path.resolve())
        with _FALLBACK_LOCKS_GUARD:
            lock = _FALLBACK_LOCKS.setdefault(key, threading.Lock())
        with lock:
            yield
        return

    with lock_path.open("a") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class SpendLedger:
    def __init__(self, path: Path | None = None, cap: float | None = None):
        self.path = Path(path) if path is not None else LEDGER_PATH
        self.cap = cap

    def _read_unlocked(self) -> list[dict]:
        if not self.path.exists():
            return []
        raw = self.path.read_bytes()
        lines = raw.splitlines(keepends=True)
        entries = []
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                if index == len(lines) - 1 and not line.endswith(b"\n"):
                    warnings.warn(
                        f"ignoring truncated final spend-ledger line in {self.path}",
                        RuntimeWarning,
                    )
                    break
                raise
        return entries

    def _totals_unlocked(self) -> dict:
        reservations: dict[str, float] = {}
        actual = 0.0
        calls = 0
        for entry in self._read_unlocked():
            kind = entry.get("type")
            if kind == "reserve":
                reservations[entry["id"]] = float(entry["upper_usd"])
            elif kind == "call":
                actual += float(entry.get("usd", 0.0))
                calls += 1
                reservations.pop(entry["id"], None)
            elif kind == "void":
                reservations.pop(entry["id"], None)
        actual = round(actual, 6)
        opened = round(sum(reservations.values()), 6)
        return {
            "actual_usd": actual,
            "open_usd": opened,
            "committed_usd": round(actual + opened, 6),
            "calls": calls,
        }

    def totals(self) -> dict:
        with _locked(self.path):
            return self._totals_unlocked()

    def _append_unlocked(self, entry: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(_canonical_json(entry) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    def reserve(self, run: str, model: str, upper_usd: float) -> str:
        upper_usd = float(upper_usd)
        if not math.isfinite(upper_usd) or upper_usd < 0:
            raise ValueError("reservation must be finite and >= 0")
        with _locked(self.path):
            totals = self._totals_unlocked()
            if self.cap is not None and totals["committed_usd"] + upper_usd > self.cap + 1e-9:
                raise CapReached(
                    f"spend cap reached: spent ${totals['committed_usd']:.6f}, "
                    f"cap ${self.cap:.6f}, needed ${upper_usd:.6f}"
                )
            reservation_id = uuid.uuid4().hex
            self._append_unlocked({
                "type": "reserve",
                "id": reservation_id,
                "run": run,
                "model": model,
                "upper_usd": round(upper_usd, 6),
                "created_at": self._timestamp(),
            })
            return reservation_id

    def settle(
        self,
        reservation_id: str,
        run: str,
        model: str,
        usage: dict,
        usd: float,
    ) -> None:
        with _locked(self.path):
            self._append_unlocked({
                "type": "call",
                "id": reservation_id,
                "run": run,
                "model": model,
                "usage": {
                    key: int(usage.get(key, 0) or 0)
                    for key in (
                        "input_tokens",
                        "output_tokens",
                        "cache_creation_input_tokens",
                        "cache_read_input_tokens",
                    )
                },
                "usd": round(float(usd), 6),
                "created_at": self._timestamp(),
            })

    def void(self, reservation_id: str, reason: str) -> None:
        with _locked(self.path):
            self._append_unlocked({
                "type": "void",
                "id": reservation_id,
                "reason": reason,
            })

    def admit_run(self, projected_usd: float) -> None:
        projected_usd = float(projected_usd)
        if not math.isfinite(projected_usd) or projected_usd < 0:
            raise ValueError("projected spend must be finite and >= 0")
        totals = self.totals()
        if self.cap is not None and totals["committed_usd"] + projected_usd > self.cap + 1e-9:
            raise CapReached(
                f"spend cap reached: spent ${totals['committed_usd']:.6f}, "
                f"cap ${self.cap:.6f}, needed ${projected_usd:.6f}"
            )
