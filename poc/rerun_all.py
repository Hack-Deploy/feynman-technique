"""Rerun every live-market hypothesis against every priced model in randomized order."""

from __future__ import annotations

import argparse
import os
import random
import secrets
from dataclasses import replace
from pathlib import Path

from poc import archive, bench, config as C, demo_grid, live_cache, spend
from poc.spend import SpendLedger


def _order(
    settings: spend.LiveSettings,
    cfg: C.Config,
    order_seed: int,
) -> tuple[list[tuple[str, str, int]], dict[str, list[str]], dict]:
    rng = random.Random(order_seed)
    models = list(settings.models)
    claim_order = {
        hyp.id: [model.id for model in rng.sample(models, len(models))]
        for hyp in cfg.hypotheses
    }
    runs = [
        (hyp.id, claim_order[hyp.id][position], 0)
        for position in range(len(models))
        for hyp in cfg.hypotheses
    ]
    extra_fields = {
        (hyp.id, model_id, 0): {
            "order_seed": order_seed,
            "order_position": position,
        }
        for hyp in cfg.hypotheses
        for position, model_id in enumerate(claim_order[hyp.id])
    }
    return runs, claim_order, extra_fields


def _cached_order_seed(entries: list[dict]) -> int | None:
    seeds = {
        entry.get("order_seed")
        for entry in entries
        if type(entry.get("order_seed")) is int
    }
    if len(seeds) > 1:
        raise ValueError(f"cache contains conflicting order seeds: {sorted(seeds)}")
    return next(iter(seeds), None)


def _resolve_order_seed(
    entries: list[dict],
    requested: int | None,
    reuse_cached: bool = True,
) -> int:
    cached = _cached_order_seed(entries) if reuse_cached else None
    if requested is not None:
        if cached is not None and requested != cached:
            raise ValueError(
                f"cache was run with order seed {cached}; use --purge to start a new order "
                "or omit --order-seed"
            )
        return requested
    return cached if cached is not None else secrets.randbelow(2**31)


def _print_order(order_seed: int, claim_order: dict[str, list[str]], out=print) -> None:
    out(f"Order seed: {order_seed}")
    out("Claim → model order:")
    for claim, models in claim_order.items():
        out(f"{claim}: {' → '.join(models)}")


def _archive(
    cache_path: Path,
    store_path: Path,
    transcripts_path: Path,
    out=print,
) -> list[tuple[Path, Path]]:
    sources = [
        Path(cache_path),
        live_cache.summary_path(Path(cache_path)),
        Path(store_path),
        Path(transcripts_path),
    ]
    protected = [
        Path(live_cache.SCRIPTED_PATH),
        live_cache.summary_path(Path(live_cache.SCRIPTED_PATH)),
        Path(spend.LEDGER_PATH),
        Path(f"{spend.LEDGER_PATH}.lock"),
    ]
    return archive.archive_paths(sources, Path(C.ROOT), out=out, protected_paths=protected)


def _check_live_gates(cap: float | None) -> None:
    if os.environ.get("ENABLE_LIVE") != "1":
        raise SystemExit("refusing paid calls: set ENABLE_LIVE=1")
    if cap is None:
        raise SystemExit("refusing paid calls: DM_MAX_USD must be set to a positive amount")
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set: put it in poc/.env (see poc/.env.example)"
        )


def _confirm(args) -> bool:
    if args.yes:
        return True
    prompt = (
        "Archive existing live data and start all reruns? [y/N] "
        if args.purge else "Start all live reruns? [y/N] "
    )
    try:
        return input(prompt).strip().lower() in ("y", "yes")
    except EOFError:
        return False


def main(argv: list[str] | None = None) -> None:
    bench.load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--purge", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--order-seed", type=int)
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--store", type=Path)
    parser.add_argument("--transcripts", type=Path)
    args = parser.parse_args(argv)
    if args.order_seed is not None and args.order_seed < 0:
        parser.error("--order-seed must be non-negative")

    cfg = C.load()
    settings = replace(
        spend.load_settings(),
        hypotheses=tuple(hyp.id for hyp in cfg.hypotheses),
        seeds=(0,),
    )
    fake_dir = Path(C.ROOT) / "output" / "rerun_all_fake"
    cache_path = args.cache or (
        fake_dir / "runs.jsonl" if args.fake else live_cache.RUNS_PATH
    )
    store_path = args.store or (
        fake_dir / "attempts.jsonl" if args.fake else C.ATTEMPTS_PATH
    )
    transcripts_path = args.transcripts or (
        fake_dir / "transcripts" if args.fake else C.TRANSCRIPTS_DIR
    )

    cached_entries = live_cache.load(cache_path)
    try:
        order_seed = _resolve_order_seed(
            cached_entries,
            args.order_seed,
            reuse_cached=not args.purge,
        )
    except ValueError as exc:
        parser.error(str(exc))
    order, claim_order, extra_fields = _order(settings, cfg, order_seed)
    summary_fields = {"order_seed": order_seed, "claim_order": claim_order}
    _print_order(order_seed, claim_order)

    cap = spend.effective_cap(settings)
    ledger = SpendLedger(spend.LEDGER_PATH, cap=cap)
    todo, _, _ = demo_grid._plan(
        settings,
        cache_path,
        order=order,
        ignore_cached=args.purge,
    )
    demo_grid._print_preflight(settings, todo, ledger)
    if args.preflight:
        return

    if not args.fake:
        _check_live_gates(cap)
    if not todo and not args.purge:
        print("No runs to do; the cache already contains every requested run.")
        return
    if (not args.fake or args.purge) and not _confirm(args):
        print("Declined; no data was moved and no runs were started.")
        return
    if args.purge:
        _archive(cache_path, store_path, transcripts_path)

    previous_transcripts = C.TRANSCRIPTS_DIR
    previous_trajectories = C.TRAJECTORIES_DIR
    C.TRANSCRIPTS_DIR = Path(transcripts_path)
    if Path(transcripts_path) != Path(previous_transcripts):
        C.TRAJECTORIES_DIR = Path(transcripts_path).parent / "poc_trajectories"
    try:
        if args.fake:
            result = demo_grid._run_fake_grid(
                settings,
                cache_path,
                order=order,
                extra_entry_fields=extra_fields,
                store_path=store_path,
                summary_fields=summary_fields,
                show_preflight=False,
            )
        else:
            result = demo_grid.run_grid(
                settings,
                cache_path,
                ledger,
                transport_factory=lambda _model: None,
                confirm=True,
                order=order,
                extra_entry_fields=extra_fields,
                store_path=store_path,
                summary_fields=summary_fields,
                show_preflight=False,
            )
    finally:
        C.TRANSCRIPTS_DIR = previous_transcripts
        C.TRAJECTORIES_DIR = previous_trajectories

    if result["stopped_reason"] == "interrupted":
        raise SystemExit(130)
    if result["stopped_reason"] == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
