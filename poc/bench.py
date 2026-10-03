"""Run the bounty benchmark: every hypothesis in config.yaml × models × seeds.

Runs on a hypothesis go in a fixed order (models, then seeds); each sees the public record of
the failed runs before it. Every run happens even after a hypothesis is settled correctly;
successes are never shown. Results are appended to attempts/poc_dp_bench.jsonl (resumable).

    uv run python -m poc.bench --fake                         # scripted LLM, no API calls
    ENABLE_LIVE=1 DM_MAX_USD=5 uv run python -m poc.demo_grid --preflight
    ENABLE_LIVE=1 DM_MAX_USD=5 uv run python -m poc.demo_grid
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid

from dm.store import AttemptStore
from dm.types import AttemptRecord, SubmittedAttempt
from poc import config as C
from poc import protocol
from poc.attempt import run_attempt


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


def _check_spend(n_runs: int, cfg: C.Config, usd_per_call: float | None) -> None:
    """CLAUDE.md rule 1: no paid calls without ENABLE_LIVE=1 and DM_MAX_USD; print projection."""
    if os.environ.get("ENABLE_LIVE") != "1" or not os.environ.get("DM_MAX_USD"):
        sys.exit("refusing paid calls: set ENABLE_LIVE=1 and DM_MAX_USD (or use --fake)")
    if usd_per_call is None:
        sys.exit("pass --usd-per-call (your estimate) so the projected spend can be checked")
    max_calls = n_runs * (2 * cfg.max_rounds + 1)  # each round may need one re-prompt
    projected = max_calls * usd_per_call
    limit = float(os.environ["DM_MAX_USD"])
    print(f"projected worst case: {n_runs} runs, {max_calls} calls, ${projected:.2f} "
          f"(limit ${limit:.2f})")
    if projected > limit:
        sys.exit("projected spend exceeds DM_MAX_USD; nothing was run")


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
    if args.fake:
        from poc.fake_llm import ScriptedLLM
    else:
        load_env()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("ANTHROPIC_API_KEY is not set: put it in poc/.env (see poc/.env.example)")
        _check_spend(len(todo), cfg, args.usd_per_call)

    for hyp, model, seed in todo:
        if args.fake:
            complete = ScriptedLLM.default()
        ledger = public_record(records, hyp, cfg)
        submitted = run_attempt(model, hyp.id, seed, ledger, cfg=cfg, complete=complete,
                                verbose=args.verbose)
        record = resolve(hyp, submitted)
        store.append([record])
        records.append(record)
        print(f"{hyp.id:32s} {model:24s} seed {seed}: {record.extra['outcome']:13s} "
              f"verdict {record.verdict['agent_verdict']!s:12s} passed {record.passed!s:5s} "
              f"spent {record.lab_cost:g}  p {record.extra.get('final_p')}")


if __name__ == "__main__":
    main()
