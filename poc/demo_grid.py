"""Run or preview the configured multi-model live demo grid."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from dm.types import AttemptRecord
from poc import bench, config as C, fake_llm, live_cache, spend
from poc.attempt import prompt_chars, run_attempt
from poc.llm import MeteredLLM, redact
from poc.spend import CapReached, LiveSettings, ModelPrice, SpendLedger


def _plan(settings: LiveSettings, cache_path: Path) -> tuple[list[dict], list[dict], C.Config]:
    cfg = replace(C.load(), max_rounds=settings.max_rounds)
    entries = live_cache.load(cache_path)
    cached_keys = live_cache.done_keys(entries)
    cached_records = live_cache.records(entries)
    todo = []
    prompt_cache = {}
    for hypothesis_id in settings.hypotheses:
        hyp = cfg.hypothesis(hypothesis_id)
        public_entries = bench.public_record(cached_records, hyp, cfg)
        prompt_key = (hypothesis_id, tuple(entry["id"] for entry in public_entries))
        if prompt_key not in prompt_cache:
            prompt_cache[prompt_key] = prompt_chars(hypothesis_id, cfg, public_entries)
        for model in settings.models:
            for seed in settings.seeds:
                key = (model.id, hypothesis_id, seed)
                if key in cached_keys:
                    continue
                projection = spend.project_run_usd(
                    model,
                    prompt_cache[prompt_key],
                    settings.max_rounds,
                    settings.max_tokens,
                    settings.chars_per_token,
                    settings.data_chars_per_round,
                )
                todo.append({
                    "hyp": hyp,
                    "model": model,
                    "seed": seed,
                    "public_entries": public_entries,
                    "projection": projection,
                })
    return todo, entries, cfg


def _print_preflight(
    settings: LiveSettings,
    todo: list[dict],
    ledger: SpendLedger | None,
    out=print,
) -> dict:
    rows = []
    for model in settings.models:
        runs = [run for run in todo if run["model"].id == model.id]
        worst = max((run["projection"]["usd"] for run in runs), default=0.0)
        rows.append({
            "model": model,
            "runs": len(runs),
            "worst_per_run": worst,
            "worst_total": round(worst * len(runs), 6),
        })
    total = round(sum(row["worst_total"] for row in rows), 6)
    committed = ledger.totals()["committed_usd"] if ledger is not None else 0.0
    cap = spend.effective_cap(settings)
    cap_text = f"${cap:.6f}" if cap is not None else "not enabled (DM_MAX_USD missing/invalid)"

    out(
        f"{'Model':24s} {'API id':34s} {'$/MTok in/out':>18s} "
        f"{'runs':>4s} {'worst/run':>12s} {'worst total':>12s}"
    )
    for row in rows:
        model = row["model"]
        prices = f"{model.input:g}/{model.output:g}"
        out(
            f"{model.label:24.24s} {model.id:34.34s} "
            f"{prices:>18s} "
            f"{row['runs']:4d} ${row['worst_per_run']:11.6f} ${row['worst_total']:11.6f}"
        )
    out(
        f"Total worst-case: ${total:.6f}; ledger committed so far: ${committed:.6f}; "
        f"effective cap: {cap_text}; configured hard cap: ${settings.max_usd:.2f}"
    )
    out(
        "Runs start only while spent + that run's worst case <= cap; every call is also "
        "checked against the cap before it is sent."
    )
    out(
        f"Assumptions: {settings.chars_per_token:g} chars/token, "
        f"{settings.data_chars_per_round} data chars/round, max_tokens={settings.max_tokens}, "
        f"max_rounds={settings.max_rounds}."
    )
    return {"rows": rows, "total": total, "committed": committed, "cap": cap}


def _run_settings(settings: LiveSettings, model: ModelPrice) -> dict:
    return {
        "max_rounds": settings.max_rounds,
        "max_tokens": settings.max_tokens,
        "prices": {"input": model.input, "output": model.output},
    }


def _confirm(confirmation) -> bool:
    if callable(confirmation):
        return bool(confirmation())
    return bool(confirmation)


def _print_run(model: str, hyp: str, record: AttemptRecord, usd: float, out) -> None:
    extra = record.extra
    out(
        f"{model} {hyp} {extra.get('outcome')} {extra.get('agent_verdict')} "
        f"passed={record.passed} credits={record.lab_cost:g} usd=${usd:.6f}"
    )


def run_grid(
    settings: LiveSettings,
    cache_path: Path,
    ledger: SpendLedger,
    transport_factory,
    confirm,
    out=print,
) -> dict:
    cache_path = Path(cache_path)
    todo, entries, cfg = _plan(settings, cache_path)
    _print_preflight(settings, todo, ledger, out)
    total_runs = len(settings.hypotheses) * len(settings.models) * len(settings.seeds)
    result = {
        "done": 0,
        "skipped": total_runs - len(todo),
        "stopped_reason": None,
    }
    if not todo:
        return result
    if not _confirm(confirm):
        result["stopped_reason"] = "declined"
        return result

    cached_records = live_cache.records(entries)
    for index, item in enumerate(todo):
        model = item["model"]
        hyp = item["hyp"]
        seed = item["seed"]
        projection = item["projection"]["usd"]
        try:
            ledger.admit_run(projection)
        except CapReached:
            remaining = len(todo) - index
            out(
                f"spend cap reached: {remaining} run(s) left; nothing more was spent"
            )
            result["stopped_reason"] = "cap"
            break

        run_key = f"{model.id}:{hyp.id}:{seed}"
        metered = MeteredLLM(
            model.id,
            model,
            ledger,
            run_key,
            transport=transport_factory(model.id),
        )
        rounds = []

        def on_round(entry):
            rounds.append(live_cache.compact_round(entry, usd_so_far=metered.usd))

        try:
            public_entries = bench.public_record(cached_records, hyp, cfg)
            submitted = run_attempt(
                model.id,
                hyp.id,
                seed,
                public_entries,
                cfg=cfg,
                complete=metered,
                on_round=on_round,
                max_tokens=settings.max_tokens,
            )
            record = bench.resolve(hyp, submitted, llm_usage=metered.usage)
            entry = {
                "schema": 1,
                "source": "real",
                "key": {
                    "model": model.id,
                    "hypothesis_id": hyp.id,
                    "seed": seed,
                },
                "model_label": model.label,
                "settings": _run_settings(settings, model),
                "record": record.to_dict(),
                "rounds": rounds,
                "usd": metered.usd,
            }
            live_cache.append(cache_path, entry)
            entries.append(entry)
            cached_records.append(record)
            live_cache.write_summary(cache_path, entries, settings)
            result["done"] += 1
            _print_run(model.id, hyp.id, record, metered.usd, out)
        except CapReached as exc:
            out(f"Spend cap reached: {redact(str(exc))}")
            result["stopped_reason"] = "cap"
            break
        except KeyboardInterrupt:
            out("interrupted; completed runs are cached, rerun to resume")
            result["stopped_reason"] = "interrupted"
            break
        except Exception as exc:
            out(f"live run stopped: {redact(str(exc))}")
            result["stopped_reason"] = "error"
            break
    return result


def _persona_replies(slot: int, hyp, cfg: C.Config) -> list[str]:
    experiment = fake_llm.scripted_experiment(hyp, cfg)
    p_values = ((0.6, 0.7), (0.55, 0.65), (0.5, 0.7, 0.8), (0.4, 0.3))
    verdicts = ("supported", "refuted", "supported", "inconclusive")
    values = p_values[slot]
    replies = []
    for i, value in enumerate(values[:-1]):
        replies.append(
            f"<assessment>Scripted experiment for {hyp.id}.</assessment>"
            f"<p_success>{value}</p_success>"
            + ("<planned_cost>10</planned_cost>" if i == 0 else "")
            + f"<run_experiment>{json.dumps([experiment])}</run_experiment>"
        )
    replies.append(
        f"<assessment>Scripted verdict for {hyp.id}.</assessment>"
        f"<p_success>{values[-1]}</p_success>"
        f"<verdict>{verdicts[slot]}</verdict>"
        + fake_llm.estimate_block(hyp)
        + "<evidence>Scripted demonstration only.</evidence>"
    )
    return replies


def _run_fake_grid(settings: LiveSettings, cache_path: Path, out=print) -> dict:
    todo, entries, cfg = _plan(settings, cache_path)
    _print_preflight(settings, todo, None, out)
    total_runs = len(settings.hypotheses) * len(settings.models) * len(settings.seeds)
    result = {
        "done": 0,
        "skipped": total_runs - len(todo),
        "stopped_reason": None,
    }
    for item in todo:
        model = item["model"]
        hyp = item["hyp"]
        seed = item["seed"]
        slot = settings.models.index(model)
        fake = fake_llm.ScriptedLLM(_persona_replies(slot, hyp, cfg))
        rounds = []
        submitted = run_attempt(
            model.id,
            hyp.id,
            seed,
            bench.public_record(live_cache.records(entries), hyp, cfg),
            cfg=cfg,
            complete=fake,
            on_round=lambda row: rounds.append(live_cache.compact_round(row, 0.0)),
            max_tokens=settings.max_tokens,
        )
        usage = {
            "calls": len(fake.calls),
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "usd": 0.0,
            "scripted": True,
        }
        record = bench.resolve(hyp, submitted, llm_usage=usage)
        record = replace(record, source="scripted", solver=f"scripted:{model.id}")
        entry = {
            "schema": 1,
            "source": "scripted",
            "key": {"model": model.id, "hypothesis_id": hyp.id, "seed": seed},
            "model_label": f"Scripted stand-in for {model.label}",
            "settings": _run_settings(settings, model),
            "record": record.to_dict(),
            "rounds": rounds,
            "usd": 0.0,
        }
        live_cache.append(cache_path, entry)
        entries.append(entry)
        live_cache.write_summary(cache_path, entries, settings)
        result["done"] += 1
        _print_run(model.id, hyp.id, record, 0.0, out)
    return result


def main(argv: list[str] | None = None) -> None:
    bench.load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--ledger", type=Path)
    args = parser.parse_args(argv)

    settings = spend.load_settings()
    cache_path = args.cache or (
        live_cache.SCRIPTED_PATH if args.fake else live_cache.RUNS_PATH
    )
    cap = spend.effective_cap(settings)
    ledger = SpendLedger(args.ledger, cap=cap)
    if args.preflight:
        todo, _, _ = _plan(settings, cache_path)
        _print_preflight(settings, todo, ledger)
        return
    if args.fake:
        result = _run_fake_grid(settings, cache_path)
        if result["stopped_reason"]:
            raise SystemExit(1)
        return

    if os.environ.get("ENABLE_LIVE") != "1":
        raise SystemExit("refusing paid calls: set ENABLE_LIVE=1")
    if cap is None:
        raise SystemExit("refusing paid calls: DM_MAX_USD must be set to a positive amount")
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set: put it in poc/.env (see poc/.env.example)"
        )
    def ask_confirmation():
        try:
            return input("Start live runs? [y/N] ").strip().lower() in ("y", "yes")
        except EOFError:
            return False

    result = run_grid(
        settings,
        cache_path,
        ledger,
        transport_factory=lambda _model: None,
        confirm=True if args.yes else ask_confirmation,
    )
    if result["stopped_reason"] == "interrupted":
        raise SystemExit(130)
    if result["stopped_reason"] == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
