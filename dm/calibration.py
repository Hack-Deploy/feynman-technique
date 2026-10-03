"""Calibration metrics for solver-stated probabilities."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

from dm.types import AttemptRecord

Pair = tuple[float, bool]


def pairs_from_records(records: Iterable[AttemptRecord]) -> tuple[list[Pair], int]:
    pairs: list[Pair] = []
    excluded = 0
    for record in records:
        if record.stated_p_success is None:
            excluded += 1
        else:
            pairs.append((float(record.stated_p_success), record.passed))
    return pairs, excluded


def _validate_pairs(pairs: Iterable[Pair]) -> list[Pair]:
    values = list(pairs)
    for probability, outcome in values:
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("probabilities must be finite and in [0, 1]")
        if not isinstance(outcome, bool):
            raise TypeError("outcomes must be bool")
    return values


def brier_score(pairs: Iterable[Pair]) -> float | None:
    values = _validate_pairs(pairs)
    if not values:
        return None
    return sum((p - float(y)) ** 2 for p, y in values) / len(values)


def calibration_in_the_large(pairs: Iterable[Pair]) -> float | None:
    values = _validate_pairs(pairs)
    if not values:
        return None
    return sum(p for p, _ in values) / len(values) - sum(y for _, y in values) / len(values)


def reliability_table(pairs: Iterable[Pair], n_bins: int = 10) -> list[dict]:
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be an integer >= 1")
    values = _validate_pairs(pairs)
    bins: list[list[Pair]] = [[] for _ in range(n_bins)]
    for probability, outcome in values:
        index = min(int(math.floor(probability * n_bins)), n_bins - 1)
        bins[index].append((probability, outcome))
    return [
        {
            "lo": index / n_bins,
            "hi": (index + 1) / n_bins,
            "count": len(bucket),
            "mean_p": (sum(p for p, _ in bucket) / len(bucket) if bucket else None),
            "pass_rate": (sum(y for _, y in bucket) / len(bucket) if bucket else None),
        }
        for index, bucket in enumerate(bins)
    ]


def _summary(pairs: list[Pair], excluded: int) -> dict:
    n = len(pairs)
    return {
        "n": n,
        "n_excluded_no_stated_p": excluded,
        "brier": brier_score(pairs),
        "calibration_in_the_large": calibration_in_the_large(pairs),
        "mean_stated_p": (sum(p for p, _ in pairs) / n if n else None),
        "pass_rate": (sum(y for _, y in pairs) / n if n else None),
        "reliability": reliability_table(pairs),
    }


def calibration_summary(records: Iterable[AttemptRecord]) -> dict:
    by_solver: dict[str, list[AttemptRecord]] = defaultdict(list)
    all_records = list(records)
    for record in all_records:
        by_solver[record.solver].append(record)
    pairs, excluded = pairs_from_records(all_records)
    return {
        **_summary(pairs, excluded),
        "by_solver": {
            solver: _summary(*pairs_from_records(solver_records))
            for solver, solver_records in sorted(by_solver.items())
        },
    }
