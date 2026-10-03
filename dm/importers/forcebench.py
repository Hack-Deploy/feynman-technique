"""Import the committed ForceBench settle output as replayable attempts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dm.store import ATTEMPTS_DIR
from dm.types import AttemptRecord

ROOT = Path(__file__).resolve().parents[2]
SETTLE_PATHS = (
    ROOT / "output" / "forcebench_settle.json",
    ROOT / "attempts" / "fixtures" / "demo" / "forcebench_settle.json",
)


def default_settle_path() -> Path:
    return next((path for path in SETTLE_PATHS if path.exists()), SETTLE_PATHS[-1])


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def records_from_settle(path: Path) -> list[AttemptRecord]:
    settle_path = Path(path)
    data: dict[str, Any] = json.loads(settle_path.read_text())
    top_price = data.get("price")
    out = []
    for row in data["results"]:
        experiments = row["experiments"]
        price = row.get("price", top_price if top_price is not None else 1.0)
        out.append(AttemptRecord(
            attempt_id=row["attempt_id"],
            source="live",
            protocol="forcebench_menu",
            venue="forcebench",
            world=row["world"],
            solver=row["solver"],
            seed=row["seed"],
            stated_p_success=row.get("stated_p"),
            rounds=experiments,
            experiments=experiments,
            lab_cost=experiments * price,
            verdict={
                "passed": row["passed"],
                "normalised_mse": row.get("nmse"),
                "prereg_commitment": row.get("prereg_commitment"),
                "public_tests": row.get("public_tests"),
                "reason": row.get("reason"),
                "explanation_score": None,
            },
            extra={
                "price": price,
                "settle_file": _repo_relative(settle_path),
                "head": data.get("head"),
                "test_seed": data.get("test_seed"),
                "identified": row.get("identified"),
                "top_model": row.get("top_model"),
                "top_params": row.get("top_params"),
                "baseline_passed": row.get("baseline_passed"),
                "baseline_nmse": row.get("baseline_nmse"),
                "stopped_reason": row.get("stopped_reason"),
                "free_seed_launch": True,
            },
        ))
    return out
