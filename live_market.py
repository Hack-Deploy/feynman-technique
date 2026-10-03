"""Callback-backed live and scripted runs for the local discovery market app."""

from __future__ import annotations

import json
import math
import os
import threading
import uuid
from pathlib import Path

from dm.store import AttemptStore
from dm.types import AttemptRecord
from poc import bench, config as C
from poc import report as poc_report
from poc.attempt import run_attempt

DEMO_PATH = C.ROOT / "attempts" / "poc_dp_demo.jsonl"
_LOCK = threading.RLock()
_JOBS: dict[str, dict] = {}
_ACTIVE_JOB: str | None = None
_PROJECTED_USD_SPENT = 0.0

_TWO_PARTICLE_WORLDS = {
    "gravity", "yukawa", "coulomb_easy", "oscillator", "fractional", "extra_dimensions",
}


def _models() -> list[str]:
    raw = os.environ.get("DM_LIVE_MODELS", "")
    if not raw.strip():
        return ["claude-sonnet-4-6"]
    return list(dict.fromkeys(model.strip() for model in raw.split(",") if model.strip()))


def _positive_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def _usd_per_call() -> float:
    return _positive_float(os.environ.get("DM_USD_PER_CALL")) or 0.05


def info() -> dict:
    bench.load_env()
    cfg = C.load()
    api_key_present = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
    enabled_flag = os.environ.get("ENABLE_LIVE") == "1"
    max_usd = _positive_float(os.environ.get("DM_MAX_USD"))
    reasons = []
    if not api_key_present:
        reasons.append("ANTHROPIC_API_KEY is not set.")
    if not enabled_flag:
        reasons.append("Set ENABLE_LIVE=1 to allow paid runs.")
    if max_usd is None:
        reasons.append("DM_MAX_USD must be set to a positive amount.")
    usd_per_call = _usd_per_call()
    projected = (2 * cfg.max_rounds + 1) * usd_per_call
    with _LOCK:
        projected_spent = _PROJECTED_USD_SPENT
    return {
        "hypotheses": [
            {
                "id": hyp.id,
                "world": hyp.world,
                "hypothesis": hyp.hypothesis,
                "resolution_criteria": hyp.resolution_criteria,
                "prize": hyp.prize,
            }
            for hyp in cfg.hypotheses
        ],
        "round_fee": cfg.round_fee,
        "experiment_costs": dict(cfg.experiment_costs),
        "max_rounds": cfg.max_rounds,
        "budget": cfg.budget,
        "payout_rule": cfg.payout_rule,
        "models": _models(),
        "live": {
            "enabled": api_key_present and enabled_flag and max_usd is not None,
            "reasons": reasons,
            "max_usd": max_usd,
            "usd_per_call": usd_per_call,
            "projected_usd_per_run": projected,
            "projected_usd_spent": projected_spent,
        },
    }


def _records(path: Path) -> list[AttemptRecord]:
    return AttemptStore(path).load(lambda record: record.protocol == C.PROTOCOL)


def _run_row(record: AttemptRecord) -> dict:
    extra = record.extra or {}
    prize_paid = extra.get("prize_paid", 0.0)
    return {
        "attempt_id": record.attempt_id,
        "hypothesis_id": extra.get("hypothesis_id"),
        "world": record.world,
        "model": record.solver,
        "seed": record.seed,
        "outcome": extra.get("outcome"),
        "agent_verdict": extra.get("agent_verdict", record.verdict.get("agent_verdict")),
        "answer": record.verdict.get("answer"),
        "passed": record.passed,
        "rounds": record.rounds,
        "experiments": record.experiments,
        "spent": record.lab_cost,
        "prize": extra.get("prize"),
        "prize_paid": prize_paid,
        "profit": prize_paid - record.lab_cost,
        "first_p": extra.get("first_p"),
        "final_p": extra.get("final_p"),
        "withdraw_reason": extra.get("withdraw_reason"),
        "evidence": extra.get("evidence"),
        "ledger_seen": len(extra.get("ledger_seen") or []),
        "round_log": extra.get("round_log") or [],
        "created_at": record.created_at,
    }


def runs() -> dict:
    live_records = _records(C.ATTEMPTS_PATH)
    demo_records = _records(DEMO_PATH)
    return {
        "live": [_run_row(record) for record in live_records],
        "demo": [_run_row(record) for record in demo_records],
        "summary": poc_report.summarise(live_records) or {},
    }


def _scripted_experiment(hyp: C.Hypothesis) -> dict:
    if hyp.world in _TWO_PARTICLE_WORLDS:
        return {
            "p1": 1,
            "p2": 1,
            "pos2": [3, 0],
            "velocity2": [0, 0],
            "measurement_times": [0.5, 1, 2],
        }

    from scienceagent.worlds import get_world

    cfg = C.load()
    world_spec = get_world(hyp.world, engine=C.ENGINE, noise_std=cfg.noise_std, noise_seed=0)
    formatted = world_spec["experiment_format"]
    start = formatted.find("[", formatted.find("<run_experiment>"))
    if start < 0:
        raise ValueError(f"no scripted experiment format for {hyp.world}")
    experiments, _ = json.JSONDecoder().raw_decode(formatted[start:])
    if not experiments:
        raise ValueError(f"no scripted experiment example for {hyp.world}")
    experiment = experiments[0]
    experiment["measurement_times"] = [1.0, 2.0]
    return experiment


def scripted_llm(hyp: C.Hypothesis):
    from poc.fake_llm import ScriptedLLM

    experiment = _scripted_experiment(hyp)
    return ScriptedLLM([
        "<assessment>The first measurement can reveal whether the hypothesis fits the observed motion.</assessment>"
        "<p_success>0.6</p_success>"
        f"<run_experiment>{json.dumps([experiment])}</run_experiment>",
        "<assessment>The measured trajectory is consistent with the proposed relationship.</assessment>"
        "<p_success>0.7</p_success><verdict>supported</verdict>"
        "<evidence>The measured trajectory is consistent with the proposed relationship.</evidence>",
    ])


def _cap_experiment(experiment):
    if not isinstance(experiment, dict):
        return experiment
    encoded = json.dumps(experiment, separators=(",", ":"), ensure_ascii=False, default=str)
    if len(encoded) > 600:
        return {"truncated": encoded[:600]}
    return experiment


def _compact_round(entry: dict) -> dict:
    experiments = entry.get("experiment_input")
    reply = entry.get("llm_reply")
    mse_fit = entry.get("mse_fit_output")
    return {
        "round": entry.get("round"),
        "action": entry.get("action"),
        "assessment": entry.get("assessment"),
        "p_success": entry.get("p_success"),
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
    }


def _job_round_callback(job_id: str):
    def callback(entry: dict) -> None:
        compact = _compact_round(entry)
        with _LOCK:
            _JOBS[job_id]["rounds"].append(compact)

    return callback


def _safe_error(exc: Exception) -> str:
    message = str(exc)
    secret = os.environ.get("ANTHROPIC_API_KEY")
    if secret:
        message = message.replace(secret, "[redacted]")
    return message[:1000] or type(exc).__name__


def _run_job(job_id: str, hyp: C.Hypothesis, model: str, seed: int, scripted: bool,
             cfg: C.Config, ledger: list[dict], store_path: Path) -> None:
    global _ACTIVE_JOB
    try:
        submitted = run_attempt(
            model,
            hyp.id,
            seed,
            ledger,
            cfg=cfg,
            complete=scripted_llm(hyp) if scripted else None,
            on_round=_job_round_callback(job_id),
        )
        record = bench.resolve(hyp, submitted)
        AttemptStore(store_path).append([record])
        with _LOCK:
            _JOBS[job_id]["run"] = _run_row(record)
            _JOBS[job_id]["state"] = "done"
    except Exception as exc:
        with _LOCK:
            _JOBS[job_id]["error"] = _safe_error(exc)
            _JOBS[job_id]["state"] = "error"
    finally:
        with _LOCK:
            if _ACTIVE_JOB == job_id:
                _ACTIVE_JOB = None


def start(hypothesis_id: str, model: str, scripted: bool) -> dict:
    global _ACTIVE_JOB, _PROJECTED_USD_SPENT
    with _LOCK:
        if _ACTIVE_JOB is not None:
            raise RuntimeError("Another live-market run is already in progress.")
        if not isinstance(scripted, bool):
            raise ValueError("scripted must be a boolean")

        cfg = C.load()
        try:
            hyp = cfg.hypothesis(hypothesis_id)
        except KeyError:
            raise ValueError(f"unknown hypothesis: {hypothesis_id}") from None

        projection = 0.0
        if scripted:
            model = "scripted-demo"
            store_path = DEMO_PATH
        else:
            settings = info()
            if model not in settings["models"]:
                raise ValueError(f"model is not in the live allowlist: {model}")
            live_settings = settings["live"]
            if not live_settings["enabled"]:
                raise PermissionError("; ".join(live_settings["reasons"]))
            projection = live_settings["projected_usd_per_run"]
            if _PROJECTED_USD_SPENT + projection > live_settings["max_usd"]:
                raise PermissionError("Projected run spend exceeds DM_MAX_USD.")
            store_path = C.ATTEMPTS_PATH

        records = _records(store_path)
        used_seeds = {
            record.seed for record in records
            if record.extra.get("hypothesis_id") == hyp.id and record.solver == model
        }
        seed = 0
        while seed in used_seeds:
            seed += 1

        job_id = uuid.uuid4().hex
        _JOBS[job_id] = {
            "state": "running",
            "hypothesis_id": hyp.id,
            "model": model,
            "seed": seed,
            "scripted": scripted,
            "rounds": [],
            "run": None,
            "error": None,
        }
        _ACTIVE_JOB = job_id
        _PROJECTED_USD_SPENT += projection
        thread = threading.Thread(
            target=_run_job,
            args=(job_id, hyp, model, seed, scripted, cfg, bench.public_record(records, hyp, cfg),
                  store_path),
            daemon=True,
        )
        try:
            thread.start()
        except Exception:
            _ACTIVE_JOB = None
            _PROJECTED_USD_SPENT -= projection
            _JOBS[job_id]["state"] = "error"
            raise RuntimeError("Could not start the live-market worker.") from None

    return {
        "ok": True,
        "job_id": job_id,
        "seed": seed,
        "scripted": scripted,
        "projected_usd": projection,
    }


def job(job_id: str) -> dict | None:
    with _LOCK:
        stored = _JOBS.get(job_id)
        if stored is None:
            return None
        return {
            "state": stored["state"],
            "hypothesis_id": stored["hypothesis_id"],
            "model": stored["model"],
            "seed": stored["seed"],
            "scripted": stored["scripted"],
            "rounds": list(stored["rounds"]),
            "run": stored["run"] if stored["state"] == "done" else None,
            "error": stored["error"],
        }
