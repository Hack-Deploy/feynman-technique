"""Run or preview the configured multi-model live demo grid."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from dm.store import AttemptStore
from dm.types import AttemptRecord
from poc import bench, config as C, fake_llm, live_cache, spend
from poc.attempt import prompt_chars, run_attempt
from poc.llm import MeteredLLM, redact
from poc.spend import CapReached, LiveSettings, ModelPrice, SpendLedger


def _plan(
    settings: LiveSettings,
    cache_path: Path,
    order: list[tuple[str, str, int]] | None = None,
    ignore_cached: bool = False,
) -> tuple[list[dict], list[dict], C.Config]:
    cfg = replace(C.load(), max_rounds=settings.max_rounds)
    entries = [] if ignore_cached else live_cache.load(cache_path)
    cached_keys = live_cache.done_keys(entries)
    cached_records = live_cache.records(entries)
    todo = []
    prompt_cache = {}
    planned = order if order is not None else [
        (hypothesis_id, model.id, seed)
        for hypothesis_id in settings.hypotheses
        for model in settings.models
        for seed in settings.seeds
    ]
    models = {model.id: model for model in settings.models}
    solved_claims = {
        hypothesis_id
        for hypothesis_id in settings.hypotheses
        if bench.solved_by(cached_records, hypothesis_id) is not None
    }
    for hypothesis_id, model_id, seed in planned:
        if hypothesis_id in solved_claims:
            continue
        hyp = cfg.hypothesis(hypothesis_id)
        public_entries = bench.public_record(cached_records, hyp, cfg)
        prompt_key = (hypothesis_id, tuple(entry["id"] for entry in public_entries))
        if prompt_key not in prompt_cache:
            prompt_cache[prompt_key] = prompt_chars(hypothesis_id, cfg, public_entries)
        model = models[model_id]
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
    open_claims: int | None = None,
    total_claims: int | None = None,
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
    if note := spend.cap_note(settings):
        out(note)
    if open_claims is not None:
        total_claims = total_claims if total_claims is not None else len(settings.hypotheses)
        out(
            f"Open claims: {open_claims}/{total_claims}; the projection above is the "
            "worst case if none of these claims is solved."
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


def _reconcile_store(entries: list[dict], store_path: Path, out) -> None:
    store = AttemptStore(store_path)
    stored_ids = {record.attempt_id for record in store.load()}
    for entry in entries:
        if entry.get("source") != "real":
            continue
        record = AttemptRecord.from_dict(entry["record"])
        if record.attempt_id in stored_ids:
            continue
        store.append([record])
        stored_ids.add(record.attempt_id)
        out(f"Repaired attempt store with cached record {record.attempt_id}")


def run_grid(
    settings: LiveSettings,
    cache_path: Path,
    ledger: SpendLedger,
    transport_factory,
    confirm,
    out=print,
    order: list[tuple[str, str, int]] | None = None,
    extra_entry_fields: dict[tuple[str, str, int], dict] | None = None,
    store_path: Path | None = None,
    summary_fields: dict | None = None,
    show_preflight: bool = True,
) -> dict:
    cache_path = Path(cache_path)
    todo, entries, cfg = _plan(settings, cache_path, order=order)
    if store_path is not None:
        _reconcile_store(entries, Path(store_path), out)
    if show_preflight:
        _print_preflight(settings, todo, ledger, out)
    planned = order if order is not None else [
        (hypothesis_id, model.id, seed)
        for hypothesis_id in settings.hypotheses
        for model in settings.models
        for seed in settings.seeds
    ]
    cached_keys = live_cache.done_keys(entries)
    cached_records = live_cache.records(entries)
    solved_claims = {
        hypothesis_id
        for hypothesis_id in settings.hypotheses
        if bench.solved_by(cached_records, hypothesis_id) is not None
    }
    closed = sum(
        hypothesis_id in solved_claims
        and (model_id, hypothesis_id, seed) not in cached_keys
        for hypothesis_id, model_id, seed in planned
    )
    skipped = sum(
        (model_id, hypothesis_id, seed) in cached_keys
        for hypothesis_id, model_id, seed in planned
    )
    result = {
        "done": 0,
        "skipped": skipped,
        "closed": closed,
        "stopped_reason": None,
    }
    if not todo:
        return result
    if not _confirm(confirm):
        result["stopped_reason"] = "declined"
        return result

    for index, item in enumerate(todo):
        model = item["model"]
        hyp = item["hyp"]
        seed = item["seed"]
        solver = bench.solved_by(cached_records, hyp.id)
        if solver is not None:
            out(
                f"{hyp.id}: solved by {solver.solver}; off the market, skipping {model.id}"
            )
            result["closed"] += 1
            continue
        projection = item["projection"]["usd"]
        try:
            ledger.admit_run(projection)
        except CapReached:
            remaining = sum(
                bench.solved_by(cached_records, pending["hyp"].id) is None
                for pending in todo[index:]
            )
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
            compact = live_cache.compact_round(entry, usd_so_far=metered.usd)
            compact["cut_off"] = metered.take_cut_off()
            rounds.append(compact)

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
            cache_key = (hyp.id, model.id, seed)
            entry.update((extra_entry_fields or {}).get(cache_key, {}))
            live_cache.append(cache_path, entry)
            if store_path is not None:
                AttemptStore(store_path).append([record])
            entries.append(entry)
            cached_records.append(record)
            live_cache.write_summary(
                cache_path,
                entries,
                settings,
                extra_fields=summary_fields,
            )
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
    for value in values[:-1]:
        replies.append(
            f"<assessment>Scripted experiment for {hyp.id}.</assessment>"
            f"<p_success>{value}</p_success>"
            f"<run_experiment>{json.dumps([experiment])}</run_experiment>"
        )
    replies.append(
        f"<assessment>Scripted verdict for {hyp.id}.</assessment>"
        f"<p_success>{values[-1]}</p_success>"
        f"<verdict>{verdicts[slot]}</verdict>"
        "<evidence>Scripted demonstration only.</evidence>"
    )
    return replies


def _run_fake_grid(
    settings: LiveSettings,
    cache_path: Path,
    out=print,
    order: list[tuple[str, str, int]] | None = None,
    extra_entry_fields: dict[tuple[str, str, int], dict] | None = None,
    store_path: Path | None = None,
    summary_fields: dict | None = None,
    show_preflight: bool = True,
) -> dict:
    todo, entries, cfg = _plan(settings, cache_path, order=order)
    if show_preflight:
        _print_preflight(settings, todo, None, out)
    planned = order if order is not None else [
        (hypothesis_id, model.id, seed)
        for hypothesis_id in settings.hypotheses
        for model in settings.models
        for seed in settings.seeds
    ]
    cached_keys = live_cache.done_keys(entries)
    cached_records = live_cache.records(entries)
    solved_claims = {
        hypothesis_id
        for hypothesis_id in settings.hypotheses
        if bench.solved_by(cached_records, hypothesis_id) is not None
    }
    closed = sum(
        hypothesis_id in solved_claims
        and (model_id, hypothesis_id, seed) not in cached_keys
        for hypothesis_id, model_id, seed in planned
    )
    skipped = sum(
        (model_id, hypothesis_id, seed) in cached_keys
        for hypothesis_id, model_id, seed in planned
    )
    result = {
        "done": 0,
        "skipped": skipped,
        "closed": closed,
        "stopped_reason": None,
    }
    for item in todo:
        model = item["model"]
        hyp = item["hyp"]
        seed = item["seed"]
        solver = bench.solved_by(cached_records, hyp.id)
        if solver is not None:
            out(
                f"{hyp.id}: solved by {solver.solver}; off the market, skipping {model.id}"
            )
            result["closed"] += 1
            continue
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
        cache_key = (hyp.id, model.id, seed)
        entry.update((extra_entry_fields or {}).get(cache_key, {}))
        live_cache.append(cache_path, entry)
        if store_path is not None:
            AttemptStore(store_path).append([record])
        entries.append(entry)
        cached_records.append(record)
        live_cache.write_summary(
            cache_path,
            entries,
            settings,
            extra_fields=summary_fields,
        )
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
