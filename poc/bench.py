"""Run the bounty benchmark: every hypothesis in config.yaml × models × seeds.

Runs on a hypothesis go in a fixed order (models, then seeds); each sees the public record of
the failed runs before it. Every run happens even after a hypothesis is settled correctly;
successes are never shown. Results are appended to attempts/poc_dp_bench.jsonl (resumable).

    uv run python -m poc.bench --fake                         # scripted LLM, no API calls
    ENABLE_LIVE=1 DM_MAX_USD=5 uv run python -m poc.bench --models claude-sonnet-5-5 --seeds 0
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from dataclasses import replace

from dm.store import AttemptStore
from dm.types import AttemptRecord, SubmittedAttempt
from poc import config as C, spend
from poc import protocol
from poc.attempt import prompt_chars, run_attempt
from poc.llm import MeteredLLM
from poc.spend import CapReached, SpendLedger


def resolve(
    hyp: C.Hypothesis, s: SubmittedAttempt, llm_usage: dict | None = None
) -> AttemptRecord:
    """Judge a run against the hidden answer. A clear verdict that matches it wins the prize."""
    agent_verdict = s.extra.get("agent_verdict")
    passed = s.extra.get("outcome") == "verdict" and agent_verdict == hyp.answer
    return AttemptRecord(
        attempt_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{C.PROTOCOL}:{hyp.id}:{s.solver}:{s.seed}")),
        source=s.source, protocol=s.protocol, venue=s.venue, world=s.world, solver=s.solver,
        seed=s.seed, stated_p_success=s.stated_p_success, rounds=s.rounds,
        experiments=s.experiments, lab_cost=s.lab_cost, llm_usage=llm_usage or {},
        submitted_law=None,
        verdict={"passed": passed, "agent_verdict": agent_verdict, "answer": hyp.answer,
                 "resolved_by": "answer_key"},
        transcript_path=s.transcript_path, created_at=s.created_at,
        extra={**s.extra, "prize_paid": hyp.prize if passed else 0.0},
    )


def public_record(records: list[AttemptRecord], hyp: C.Hypothesis, cfg: C.Config) -> list[dict]:
    failed = [r.to_dict() for r in records
              if r.extra.get("hypothesis_id") == hyp.id and not r.passed]
    return [protocol.ledger_entry(r, cfg.ledger_max_data_chars)
            for r in failed[-cfg.ledger_max_entries:]]


def _done(records: list[AttemptRecord], hid: str, model: str, seed: int) -> bool:
    return any(r.extra.get("hypothesis_id") == hid and r.solver == model and r.seed == seed
               for r in records)


def load_env(path=C.ENV_PATH) -> None:
    """Read KEY=value lines from poc/.env (git-ignored) into the environment. Values already
    set in the shell win. Keys are never printed or written anywhere."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key.strip(), value)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["fake"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--hypotheses", nargs="+", help="ids from config.yaml (default: all)")
    ap.add_argument("--fake", action="store_true", help="scripted LLM, no API calls")
    ap.add_argument("--usd-per-call", type=float)
    ap.add_argument("--store", default=str(C.ATTEMPTS_PATH))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    if args.usd_per_call is not None:
        print(
            "deprecated: --usd-per-call is accepted but ignored; live spend uses "
            "poc/live_models.yaml"
        )

    cfg = C.load()
    hyps = [cfg.hypothesis(h) for h in args.hypotheses] if args.hypotheses else list(cfg.hypotheses)
    store = AttemptStore(args.store)
    records = store.load()
    todo = [(h, m, s) for h in hyps for m in args.models for s in args.seeds
            if not _done(records, h.id, m, s)]
    print(f"{len(todo)} run(s) to do, {len(hyps) * len(args.models) * len(args.seeds) - len(todo)} "
          f"already in {args.store}")
    if not todo:
        return

    complete = None
    run_cfg = cfg
    live_settings = None
    spend_ledger = None
    if args.fake:
        from poc.fake_llm import ScriptedLLM
    else:
        load_env()
        if os.environ.get("ENABLE_LIVE") != "1":
            sys.exit("refusing paid calls: set ENABLE_LIVE=1")
        live_settings = spend.load_settings()
        cap = spend.effective_cap(live_settings)
        if cap is None:
            sys.exit("refusing paid calls: DM_MAX_USD must be set to a positive amount")
        unknown_models = sorted(set(args.models) - {model.id for model in live_settings.models})
        if unknown_models:
            sys.exit(
                "refusing paid calls: model(s) not in poc/live_models.yaml: "
                + ", ".join(unknown_models)
            )
        if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
            sys.exit(
                "ANTHROPIC_API_KEY is not set: put it in poc/.env "
                "(see poc/.env.example)"
            )
        run_cfg = replace(cfg, max_rounds=live_settings.max_rounds)
        spend_ledger = SpendLedger(cap=cap)

    for hyp, model, seed in todo:
        if args.fake:
            complete = ScriptedLLM.default()
            max_tokens = C.MAX_TOKENS
            run_ledger = public_record(records, hyp, run_cfg)
        else:
            model_price = spend.price(live_settings, model)
            run_ledger = public_record(records, hyp, run_cfg)
            projection = spend.project_run_usd(
                model_price,
                prompt_chars(hyp.id, run_cfg, run_ledger),
                live_settings.max_rounds,
                live_settings.max_tokens,
                live_settings.chars_per_token,
                live_settings.data_chars_per_round,
            )
            print(
                f"projected worst case for {hyp.id}/{model}: "
                f"${projection['usd']:.6f} (effective cap ${spend_ledger.cap:.6f})"
            )
            try:
                spend_ledger.admit_run(projection["usd"])
            except CapReached as exc:
                print(f"spend cap reached; stopping live grid: {exc}")
                break
            complete = MeteredLLM(
                model,
                model_price,
                spend_ledger,
                f"bench:{hyp.id}:{model}:{seed}",
            )
            max_tokens = live_settings.max_tokens
        try:
            submitted = run_attempt(
                model,
                hyp.id,
                seed,
                run_ledger,
                cfg=run_cfg,
                complete=complete,
                verbose=args.verbose,
                max_tokens=max_tokens,
            )
        except CapReached as exc:
            print(f"spend cap reached; stopping live grid: {exc}")
            break
        record = resolve(
            hyp,
            submitted,
            llm_usage=complete.usage if not args.fake else None,
        )
        store.append([record])
        records.append(record)
        print(f"{hyp.id:32s} {model:24s} seed {seed}: {record.extra['outcome']:13s} "
              f"verdict {record.verdict['agent_verdict']!s:12s} passed {record.passed!s:5s} "
              f"spent {record.lab_cost:g}  p {record.extra.get('final_p')}")


if __name__ == "__main__":
    main()
