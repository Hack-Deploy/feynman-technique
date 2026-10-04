"""Run the bounty benchmark: every hypothesis in config.yaml × models × seeds.

Runs on a hypothesis go in a fixed order (models, then seeds); each sees the public record of
the failed runs before it. Every run happens even after a hypothesis is settled correctly;
successes are never shown. Results are appended to attempts/poc_dp_bench.jsonl (resumable).

    uv run python -m poc.bench --fake                         # scripted LLM, no API calls
    uv run python -m poc.bench --baselines --seeds 0 1 2      # scripted agents, no API calls
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


def resolve(hyp: C.Hypothesis, s: SubmittedAttempt, cfg: C.Config | None = None,
            llm_usage: dict | None = None) -> AttemptRecord:
    """Judge a run with the independent checker and settle it under both reward rules."""
    from poc.checker import judge
    from poc.ledger import settle

    cfg = cfg or C.load()
    x = s.extra
    ruling = judge(hyp, x, s.lab_cost)
    told = x.get("rule", "market")
    settlements = {rule: settle(rule, cfg, hyp.prize, x.get("account_events") or [],
                                x.get("outcome"), x.get("agent_verdict"), ruling.confirmed,
                                x.get("bid_p"))
                   for rule in C.RULES}
    actual = settlements[told]
    key = f"{C.PROTOCOL}:{hyp.id}:{s.solver}:{told}:{int(x.get('experiments_enabled', True))}:{s.seed}"
    return AttemptRecord(
        attempt_id=str(uuid.uuid5(uuid.NAMESPACE_URL, key)),
        source=s.source, protocol=s.protocol, venue=s.venue, world=s.world, solver=s.solver,
        seed=s.seed, stated_p_success=s.stated_p_success, rounds=s.rounds,
        experiments=s.experiments, lab_cost=s.lab_cost, llm_usage=llm_usage or {},
        submitted_law=None,
        verdict={"passed": ruling.confirmed, "outcome": ruling.outcome,
                 "agent_verdict": x.get("agent_verdict"), "answer": hyp.answer,
                 "verdict_matches_answer": x.get("agent_verdict") == hyp.answer,
                 "ruling": ruling.to_dict(), "resolved_by": "poc.checker"},
        transcript_path=s.transcript_path, created_at=s.created_at,
        extra={**x, "settlements": settlements, "prize_paid": actual["prize_paid"],
               "profit": actual["profit"]},
    )


def public_record(records: list[AttemptRecord], hyp: C.Hypothesis, cfg: C.Config) -> list[dict]:
    failed = [r.to_dict() for r in records
              if r.extra.get("hypothesis_id") == hyp.id and not r.passed]
    return [protocol.ledger_entry(r, cfg.ledger_max_data_chars)
            for r in failed[-cfg.ledger_max_entries:]]


def _done(records: list[AttemptRecord], hid: str, model: str, seed: int, rule: str,
          experiments: bool) -> bool:
    return any(r.extra.get("hypothesis_id") == hid and r.solver == model and r.seed == seed
               and r.extra.get("rule", "market") == rule
               and r.extra.get("experiments_enabled", True) == experiments for r in records)


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
    ap.add_argument("--baselines", action="store_true",
                    help="add every scripted baseline agent (no API calls)")
    ap.add_argument("--rule", choices=C.RULES, default="market",
                    help="the reward rule the agent is told (both are always scored)")
    ap.add_argument("--prior-only", action="store_true",
                    help="no lab: one reply from prior knowledge (the experimenting control)")
    ap.add_argument("--record", action="store_true",
                    help="show the public record of earlier failed runs (default: blind)")
    ap.add_argument("--max-rounds", type=int,
                    help="live runs only: override live.max_rounds in poc/live_models.yaml")
    ap.add_argument("--max-usd", type=float,
                    help="live runs only: override live.max_usd (DM_MAX_USD still applies)")
    ap.add_argument("--usd-per-call", type=float)
    ap.add_argument("--store", default=str(C.ATTEMPTS_PATH))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    from poc import baselines

    if args.usd_per_call is not None:
        print("deprecated: --usd-per-call is accepted but ignored; live spend uses "
              "poc/live_models.yaml")

    cfg = C.load()
    hyps = [cfg.hypothesis(h) for h in args.hypotheses] if args.hypotheses else list(cfg.hypotheses)
    models = list(args.models)
    if args.baselines:
        models = [m for m in models if m != "fake"] + [baselines.PREFIX + n for n in baselines.NAMES]
    experiments = not args.prior_only
    store = AttemptStore(args.store)
    records = store.load()
    todo = [(h, m, s) for h in hyps for m in models for s in args.seeds
            if not _done(records, h.id, m, s, args.rule, experiments)]
    print(f"{len(todo)} run(s) to do, {len(hyps) * len(models) * len(args.seeds) - len(todo)} "
          f"already in {args.store}")
    if not todo:
        return

    live = sorted({m for _, m, _ in todo if not baselines.is_baseline(m) and not args.fake})
    run_cfg = cfg
    live_settings = None
    spend_ledger = None
    if args.fake:
        from poc.fake_llm import ScriptedLLM
    if live:
        load_env()
        if os.environ.get("ENABLE_LIVE") != "1":
            sys.exit("refusing paid calls: set ENABLE_LIVE=1")
        live_settings = spend.load_settings()
        overrides = {k: v for k, v in (("max_rounds", args.max_rounds),
                                       ("max_usd", args.max_usd)) if v is not None}
        live_settings = replace(live_settings, **overrides)
        cap = spend.effective_cap(live_settings)
        if cap is None:
            sys.exit("refusing paid calls: DM_MAX_USD must be set to a positive amount")
        unknown_models = sorted(set(live) - {model.id for model in live_settings.models})
        if unknown_models:
            sys.exit("refusing paid calls: model(s) not in poc/live_models.yaml: "
                     + ", ".join(unknown_models))
        if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
            sys.exit("ANTHROPIC_API_KEY is not set: put it in poc/.env (see poc/.env.example)")
        run_cfg = replace(cfg, max_rounds=live_settings.max_rounds)
        spend_ledger = SpendLedger(cap=cap)

    for hyp, model, seed in todo:
        ledger = public_record(records, hyp, run_cfg) if args.record else []
        max_tokens = C.MAX_TOKENS
        usage = None
        if baselines.is_baseline(model):
            complete = baselines.make(model, cfg, hyp, seed)
        elif args.fake:
            complete = ScriptedLLM.default()
        else:
            model_price = spend.price(live_settings, model)
            projection = spend.project_run_usd(
                model_price, prompt_chars(hyp.id, run_cfg, ledger), live_settings.max_rounds,
                live_settings.max_tokens, live_settings.chars_per_token,
                live_settings.data_chars_per_round)
            print(f"projected worst case for {hyp.id}/{model}: "
                  f"${projection['usd']:.6f} (effective cap ${spend_ledger.cap:.6f})")
            try:
                spend_ledger.admit_run(projection["usd"])
            except CapReached as exc:
                print(f"spend cap reached; stopping live grid: {exc}")
                break
            complete = MeteredLLM(model, model_price, spend_ledger,
                                  f"bench:{hyp.id}:{model}:{args.rule}:{seed}")
            max_tokens = live_settings.max_tokens
        cfg_for_run = cfg if baselines.is_baseline(model) or args.fake else run_cfg
        try:
            submitted = run_attempt(model, hyp.id, seed, ledger, cfg=cfg_for_run,
                                    complete=complete, verbose=args.verbose,
                                    max_tokens=max_tokens, rule=args.rule,
                                    experiments=experiments)
        except CapReached as exc:
            print(f"spend cap reached; stopping live grid: {exc}")
            break
        if isinstance(complete, MeteredLLM):
            usage = complete.usage
        record = resolve(hyp, submitted, cfg_for_run, llm_usage=usage)
        store.append([record])
        records.append(record)
        st = record.extra["settlements"]
        flags = ",".join(f["flag"] for f in record.verdict["ruling"]["flags"]) or "-"
        print(f"{hyp.id:26s} {model:28s} s{seed} {record.verdict['outcome']:12s} "
              f"verdict {record.verdict['agent_verdict']!s:12s} spent {record.lab_cost:6g} "
              f"market {st['market']['profit']:+8.1f} naive {st['naive']['profit']:+8.1f} "
              f"flags {flags}")

if __name__ == "__main__":
    main()
