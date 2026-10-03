"""Append-only attempt store: one JSON object per line under ``attempts/``.

Records are never rewritten. A corrected record is appended with the same
``attempt_id`` and supersedes the earlier one when loaded (last one wins).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

from dm.types import AttemptRecord

ROOT = Path(__file__).resolve().parent.parent
ATTEMPTS_DIR = ROOT / "attempts"
FIXTURES_DIR = ATTEMPTS_DIR / "fixtures"


class AttemptStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    def append(self, records: Iterable[AttemptRecord]) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with self.path.open("a") as f:
            for r in records:
                f.write(json.dumps(r.to_dict(), sort_keys=True) + "\n")
                n += 1
        return n

    def load(self, where: Callable[[AttemptRecord], bool] | None = None) -> list[AttemptRecord]:
        """All current records (later lines supersede earlier ones with the same id),
        in first-seen order."""
        if not self.path.exists():
            return []
        latest: dict[str, AttemptRecord] = {}
        with self.path.open() as f:
            for line in f:
                if line.strip():
                    r = AttemptRecord.from_dict(json.loads(line))
                    latest[r.attempt_id] = r
        records = list(latest.values())
        return [r for r in records if where(r)] if where else records

    def has(self, venue: str, world: str, solver: str, seed: int) -> bool:
        """Used by resumable live grids: is this cell already in the store?"""
        return any(
            r.venue == venue and r.world == world and r.solver == solver and r.seed == seed
            for r in self.load()
        )


def load_records(paths: Iterable[Path]) -> list[AttemptRecord]:
    out: list[AttemptRecord] = []
    for p in paths:
        out.extend(AttemptStore(p).load())
    return out
