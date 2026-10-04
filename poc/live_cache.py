"""Append-only cache for real and scripted live-market runs."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from dm.types import AttemptRecord
from poc import report
from poc.config import ROOT

LIVE_DIR = ROOT / "attempts" / "fixtures" / "live"
RUNS_PATH = LIVE_DIR / "runs.jsonl"
SCRIPTED_PATH = LIVE_DIR / "scripted_demo.jsonl"


def summary_path(cache_path: Path) -> Path:
    return Path(cache_path).with_suffix(".summary.json")


def compact_round(entry: dict, usd_so_far: float | None = None) -> dict:
    experiments = entry.get("experiment_input")
    reply = entry.get("llm_reply")
    mse_fit = entry.get("mse_fit_output")
    return {
        "round": entry.get("round"),
        "action": entry.get("action"),
        "assessment": entry.get("assessment"),
        "p_success": entry.get("p_success"),
        "estimates": entry.get("estimates") or {},
        "experiment_input": (
            [_cap_experiment(item) for item in experiments]
            if isinstance(experiments, list) else None
        ),
        "n_experiments": len(experiments) if isinstance(experiments, list) else 0,
        "experiment_error": entry.get("experiment_error"),
        "mse_fit": str(mse_fit)[:800] if mse_fit is not None else None,
        "experiments_cost": entry.get("experiments_cost", 0.0),
        "round_fee": entry.get("round_fee", 0.0),
        "spent_so_far": entry.get("spent_so_far"),
        "verdict": entry.get("verdict"),
        "evidence": entry.get("evidence"),
        "withdraw_reason": entry.get("withdraw_reason"),
        "reply": str(reply)[:4000] if reply is not None else None,
        "usd_so_far": usd_so_far if usd_so_far is not None else entry.get("usd_so_far"),
    }


def _cap_experiment(experiment):
    if not isinstance(experiment, dict):
        return experiment
    encoded = json.dumps(experiment, separators=(",", ":"), ensure_ascii=False, default=str)
    if len(encoded) > 600:
        return {"truncated": encoded[:600]}
    return experiment


def append(path: Path, entry: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(entry, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(encoded + "\n")


def load(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    entries = {}
    lines = raw.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not line.endswith(b"\n"):
                warnings.warn(
                    f"ignoring truncated final live-cache line in {path}",
                    RuntimeWarning,
                )
                break
            raise
        entries[entry["record"]["attempt_id"]] = entry
    return list(entries.values())


def done_keys(entries: list[dict]) -> set[tuple[str, str, int]]:
    return {
        (
            entry["key"]["model"],
            entry["key"]["hypothesis_id"],
            entry["key"]["seed"],
        )
        for entry in entries
    }


def records(entries: list[dict]) -> list[AttemptRecord]:
    return [AttemptRecord.from_dict(entry["record"]) for entry in entries]


def comparison(entries: list[dict], model_order: list) -> list[dict]:
    from poc.report import summarise

    models_by_id = {model.id: model for model in model_order}
    order = [model.id for model in model_order]
    order.extend(
        model for model in dict.fromkeys(entry["key"]["model"] for entry in entries)
        if model not in models_by_id
    )
    rows = []
    for model_id in order:
        model_entries = [entry for entry in entries if entry["key"]["model"] == model_id]
        model_records = [AttemptRecord.from_dict(entry["record"]) for entry in model_entries]
        summary = summarise(model_records)
        record_model = model_records[0].solver if model_records else None
        metrics = summary["models"].get(record_model, {})
        wins = sum(record.passed for record in model_records)
        credits_spent = round(sum(record.lab_cost for record in model_records), 6)
        prize_paid = round(
            sum(record.extra.get("prize_paid", 0.0) for record in model_records), 6
        )
        rows.append({
            "model": model_id,
            "label": (
                model_entries[0].get("model_label")
                if model_entries else getattr(models_by_id.get(model_id), "label", model_id)
            ),
            "runs": len(model_entries),
            "wins": wins,
            "win_rate": round(wins / len(model_entries), 4) if model_entries else None,
            "profit": round(prize_paid - credits_spent, 6),
            "credits_spent": credits_spent,
            "prize_paid": prize_paid,
            "usd_spent": round(sum(float(entry.get("usd", 0.0)) for entry in model_entries), 6),
            "brier": metrics.get("brier_final_p"),
            "points": [
                {
                    "p": record.extra.get("final_p"),
                    "passed": record.passed,
                    "hypothesis_id": record.extra.get("hypothesis_id"),
                    "attempt_id": record.attempt_id,
                }
                for record in model_records
                if record.extra.get("outcome") == "verdict"
                and record.extra.get("final_p") is not None
            ],
        })
    return rows


def write_summary(path: Path, entries: list[dict], settings) -> None:
    summary = {
        "generated_from": str(path),
        "source": sorted({entry.get("source") for entry in entries}),
        "models_source": settings.source,
        "models_checked": settings.source.get("checked"),
        "settings": {
            "max_usd": settings.max_usd,
            "max_rounds": settings.max_rounds,
            "max_tokens": settings.max_tokens,
            "chars_per_token": settings.chars_per_token,
            "data_chars_per_round": settings.data_chars_per_round,
            "hypotheses": list(settings.hypotheses),
            "seeds": list(settings.seeds),
        },
        "comparison": comparison(entries, settings.models),
    }
    out = summary_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
