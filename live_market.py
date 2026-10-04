"""Callback-backed live and scripted runs for the local discovery market app."""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import replace
from pathlib import Path

from dm.store import AttemptStore
from dm.types import AttemptRecord
from poc import bench, config as C, live_cache, spend
from poc import fake_llm, llm as poc_llm
from poc import report as poc_report
from poc.attempt import run_attempt
from poc.llm import MeteredLLM
from poc.spend import CapReached, SpendLedger

DEMO_PATH = C.ROOT / "attempts" / "poc_dp_demo.jsonl"
_LOCK = threading.RLock()
_JOBS: dict[str, dict] = {}
_ACTIVE_JOB: str | None = None
_PROMPT_CHARS_CACHE: dict[tuple[str, tuple[str, ...]], int] = {}

_TWO_PARTICLE_WORLDS = {
    "gravity", "yukawa", "coulomb_easy", "oscillator", "fractional", "extra_dimensions",
}


def _models() -> list[str]:
    raw = os.environ.get("DM_LIVE_MODELS", "")
    if not raw.strip():
        return [model.id for model in spend.load_settings().models]
    return list(dict.fromkeys(model.strip() for model in raw.split(",") if model.strip()))


def _prompt_size(hyp: C.Hypothesis, cfg: C.Config, ledger_entries: list[dict]) -> int:
    from poc.attempt import prompt_chars

    key = (hyp.id, tuple(entry["id"] for entry in ledger_entries))
    if key not in _PROMPT_CHARS_CACHE:
        _PROMPT_CHARS_CACHE[key] = prompt_chars(hyp.id, cfg, ledger_entries)
    return _PROMPT_CHARS_CACHE[key]


def info() -> dict:
    bench.load_env()
    cfg = C.load()
    settings = spend.load_settings()
    api_key_present = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
    enabled_flag = os.environ.get("ENABLE_LIVE") == "1"
    max_usd = spend.effective_cap(settings)
    reasons = []
    if not api_key_present:
        reasons.append("ANTHROPIC_API_KEY is not set.")
    if not enabled_flag:
        reasons.append("Set ENABLE_LIVE=1 to allow paid runs.")
    if max_usd is None:
        reasons.append("DM_MAX_USD must be set to a positive amount.")
    run_cfg = replace(cfg, max_rounds=settings.max_rounds)
    attempt_records = _records(C.ATTEMPTS_PATH)
    solved_records = {
        hyp.id: bench.solved_by(attempt_records, hyp.id)
        for hyp in cfg.hypotheses
    }
    projected = {}
    for model in settings.models:
        per_hypothesis = []
        for hyp in cfg.hypotheses:
            public_entries = bench.public_record(attempt_records, hyp, cfg)
            projection = spend.project_run_usd(
                model,
                _prompt_size(hyp, run_cfg, public_entries),
                settings.max_rounds,
                settings.max_tokens,
                settings.chars_per_token,
                settings.data_chars_per_round,
            )
            per_hypothesis.append(projection["usd"])
        projected[model.id] = max(per_hypothesis, default=0.0)
    totals = SpendLedger(cap=max_usd).totals()
    return {
        "hypotheses": [
            {
                "id": hyp.id,
                "world": hyp.world,
                "hypothesis": hyp.hypothesis,
                "resolution_criteria": hyp.resolution_criteria,
                "prize": hyp.prize,
                "solved_by": (
                    solved_records[hyp.id].solver
                    if solved_records[hyp.id] is not None else None
                ),
            }
            for hyp in cfg.hypotheses
        ],
        "ledger_read_fee": cfg.ledger_read_fee,
        "round_fee": cfg.round_fee,
        "noise_std": cfg.noise_std,
        "velocity_noise_std": cfg.velocity_noise_std,
        "experiment_costs": dict(cfg.experiment_costs),
        "max_rounds": cfg.max_rounds,
        "budget": cfg.budget,
        "payout_rule": cfg.payout_rule,
        "models": _models(),
        "model_table": [
            {
                "id": model.id,
                "label": model.label,
                "input": model.input,
                "output": model.output,
            }
            for model in settings.models
        ],
        "prices_source": settings.source.get("pricing"),
        "prices_checked": settings.source.get("checked"),
        "live": {
            "enabled": api_key_present and enabled_flag and max_usd is not None,
            "reasons": reasons,
            "max_usd": max_usd,
            "hard_cap_usd": settings.max_usd,
            "cap_note": spend.cap_note(settings),
            "spent_usd": totals["committed_usd"],
            "actual_usd": totals["actual_usd"],
            "remaining_usd": (
                round(max(0.0, max_usd - totals["committed_usd"]), 6)
                if max_usd is not None else None
            ),
            "max_rounds": settings.max_rounds,
            "max_tokens": settings.max_tokens,
            "projected_usd_per_run": projected,
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
    return fake_llm.scripted_experiment(hyp, C.load())


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


_cap_experiment = live_cache._cap_experiment
_compact_round = live_cache.compact_round


def _job_round_callback(job_id: str, metered: MeteredLLM | None = None):
    def callback(entry: dict) -> None:
        compact = live_cache.compact_round(
            entry,
            usd_so_far=metered.usd if metered is not None else None,
        )
        if metered is None:
            compact["cut_off"] = False
        else:
            compact["cut_off"] = metered.take_cut_off()
        with _LOCK:
            _JOBS[job_id]["rounds"].append(compact)

    return callback


def _safe_error(exc: Exception) -> str:
    return poc_llm.redact(str(exc)) or type(exc).__name__


def _run_job(job_id: str, hyp: C.Hypothesis, model: str, seed: int, scripted: bool,
             cfg: C.Config, ledger: list[dict], store_path: Path,
             spend_ledger: SpendLedger | None = None, model_price=None,
             live_settings=None, projected_usd: float = 0.0) -> None:
    global _ACTIVE_JOB
    try:
        metered = None
        if not scripted:
            metered = MeteredLLM(
                model,
                model_price,
                spend_ledger,
                f"{model}:{hyp.id}:{seed}",
            )
        submitted = run_attempt(
            model,
            hyp.id,
            seed,
            ledger,
            cfg=cfg,
            complete=scripted_llm(hyp) if scripted else metered,
            on_round=_job_round_callback(job_id, metered),
            max_tokens=live_settings.max_tokens if live_settings else C.MAX_TOKENS,
        )
        record = bench.resolve(
            hyp,
            submitted,
            llm_usage=metered.usage if metered is not None else {},
        )
        AttemptStore(store_path).append([record])
        if metered is not None:
            with _LOCK:
                rounds = list(_JOBS[job_id]["rounds"])
            live_cache.append(live_cache.RUNS_PATH, {
                "schema": 1,
                "source": "real",
                "key": {"model": model, "hypothesis_id": hyp.id, "seed": seed},
                "model_label": model_price.label,
                "settings": {
                    "max_rounds": live_settings.max_rounds,
                    "max_tokens": live_settings.max_tokens,
                    "prices": {
                        "input": model_price.input,
                        "output": model_price.output,
                    },
                },
                "record": record.to_dict(),
                "rounds": rounds,
                "usd": metered.usd,
            })
            live_cache.write_summary(
                live_cache.RUNS_PATH,
                live_cache.load(live_cache.RUNS_PATH),
                live_settings,
            )
        with _LOCK:
            _JOBS[job_id]["run"] = _run_row(record)
            _JOBS[job_id]["state"] = "done"
    except Exception as exc:
        with _LOCK:
            if isinstance(exc, CapReached):
                _JOBS[job_id]["error"] = f"Spend cap reached: {_safe_error(exc)}"
            else:
                _JOBS[job_id]["error"] = _safe_error(exc)
            _JOBS[job_id]["state"] = "error"
    finally:
        with _LOCK:
            if _ACTIVE_JOB == job_id:
                _ACTIVE_JOB = None


def start(hypothesis_id: str, model: str, scripted: bool) -> dict:
    global _ACTIVE_JOB
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

        if not scripted:
            solver = bench.solved_by(_records(C.ATTEMPTS_PATH), hyp.id)
            if solver is not None:
                raise ValueError(f"{hyp.id} was solved by {solver.solver}; it is off the market.")

        projection = 0.0
        live_settings = None
        model_price = None
        spend_ledger = None
        if scripted:
            model = "scripted-demo"
            store_path = DEMO_PATH
            run_cfg = cfg
        else:
            settings = info()
            if model not in settings["models"]:
                raise ValueError(f"model is not in the live allowlist: {model}")
            live_settings = settings["live"]
            if not live_settings["enabled"]:
                raise PermissionError("; ".join(live_settings["reasons"]))
            price_settings = spend.load_settings()
            try:
                model_price = spend.price(price_settings, model)
            except KeyError as exc:
                raise ValueError(str(exc)) from None
            run_cfg = replace(cfg, max_rounds=price_settings.max_rounds)
            records = _records(C.ATTEMPTS_PATH)
            public_entries = bench.public_record(records, hyp, cfg)
            projection = spend.project_run_usd(
                model_price,
                _prompt_size(hyp, run_cfg, public_entries),
                price_settings.max_rounds,
                price_settings.max_tokens,
                price_settings.chars_per_token,
                price_settings.data_chars_per_round,
            )["usd"]
            spend_ledger = SpendLedger(cap=live_settings["max_usd"])
            try:
                spend_ledger.admit_run(projection)
            except CapReached as exc:
                raise PermissionError(
                    "Projected run spend exceeds the spend cap: "
                    f"{exc}; check DM_MAX_USD."
                ) from None
            store_path = C.ATTEMPTS_PATH

        records = _records(store_path)
        used_seeds = {
            record.seed for record in records
            if record.extra.get("hypothesis_id") == hyp.id and record.solver == model
        }
        if not scripted:
            used_seeds.update(
                entry["key"]["seed"]
                for entry in live_cache.load(live_cache.RUNS_PATH)
                if entry["key"]["hypothesis_id"] == hyp.id
                and entry["key"]["model"] == model
            )
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
        thread = threading.Thread(
            target=_run_job,
            args=(
                job_id,
                hyp,
                model,
                seed,
                scripted,
                run_cfg,
                bench.public_record(records, hyp, cfg),
                store_path,
                spend_ledger,
                model_price,
                spend.load_settings() if not scripted else None,
                projection,
            ),
            daemon=True,
        )
        try:
            thread.start()
        except Exception:
            _ACTIVE_JOB = None
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


def recorded() -> dict:
    settings = spend.load_settings()
    real_entries = live_cache.load(live_cache.RUNS_PATH)
    scripted_entries = live_cache.load(live_cache.SCRIPTED_PATH)
    shown = real_entries if real_entries else scripted_entries
    real_count = len(real_entries)
    scripted_count = len(scripted_entries)
    source = "real" if real_entries else "scripted" if scripted_entries else "none"
    rows = []
    for entry in shown:
        record = AttemptRecord.from_dict(entry["record"])
        row = {
            key: value for key, value in _run_row(record).items()
            if key != "round_log"
        }
        row.update({
            "usd": entry.get("usd", record.llm_usage.get("usd", 0.0)),
            "model_label": entry.get("model_label"),
            "source": entry.get("source"),
            "calls": record.llm_usage.get("calls", 0),
        })
        rows.append(row)
    return {
        "source": source,
        "real_count": real_count,
        "scripted_count": scripted_count,
        "prices_source": settings.source.get("pricing"),
        "prices_checked": settings.source.get("checked"),
        "runs": rows,
        "comparison": live_cache.comparison(shown, settings.models),
    }


def recorded_run(attempt_id: str) -> dict | None:
    for path in (live_cache.RUNS_PATH, live_cache.SCRIPTED_PATH):
        for entry in live_cache.load(path):
            record = AttemptRecord.from_dict(entry["record"])
            if record.attempt_id != attempt_id:
                continue
            row = {
                key: value for key, value in _run_row(record).items()
                if key != "round_log"
            }
            row.update({
                "usd": entry.get("usd", record.llm_usage.get("usd", 0.0)),
                "model_label": entry.get("model_label"),
                "source": entry.get("source"),
                "calls": record.llm_usage.get("calls", 0),
            })
            return {
                "run": row,
                "rounds": [
                    {**round_entry, "cut_off": round_entry.get("cut_off", False)}
                    for round_entry in entry.get("rounds", [])
                ],
                "settings": entry.get("settings", {}),
                "source": entry.get("source"),
            }
    return None
