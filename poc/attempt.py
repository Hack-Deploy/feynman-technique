"""One run: one AI scientist, one posted hypothesis, a fresh conversation, with an optional
purchase of the public record of failed runs. Returns a ``SubmittedAttempt``; ``poc.bench``
resolves it against the hidden answer, so this module never sees the answer.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import re
from typing import Callable

from dm.types import SubmittedAttempt
from poc import config as C, protocol
from poc.agent import Complete, MarketAgent
from poc.pricing import Account


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-") or "x"


def _abs_vendor(rel: str | None) -> str | None:
    # Vendor prompt paths are relative to the vendor repo root; os.path.join keeps absolute paths.
    return None if rel is None else str(C.VENDOR_ROOT / rel)


def round_log(conversation_log: list[dict]) -> list[dict]:
    """The per-round summary kept in the record."""
    out = []
    for e in conversation_log:
        ran = e.get("experiment_input") if e.get("experiment_output") is not None else None
        out.append({
            "round": e["round"], "action": e["action"], "assessment": e.get("assessment"),
            "p_success": e.get("p_success"),
            "record_bought": e.get("record_bought", False),
            "experiments": len(ran) if isinstance(ran, list) else 0,
            "experiments_cost": e.get("experiments_cost", 0.0), "round_fee": e.get("round_fee", 0.0),
            "spent_so_far": e.get("spent_so_far"),
        })
    return out


def _finite(obj):
    """Simulator output can contain nan/inf; records are strict JSON, so those become None."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if hasattr(obj, "tolist"):
        return _finite(obj.tolist())
    return obj


def runs(conversation_log: list[dict]) -> list[dict]:
    """Every experiment that ran, with its data: what goes into the public record on failure."""
    out = []
    for e in conversation_log:
        if e.get("experiment_output") is None:
            continue
        for inp, res in zip(e["experiment_input"], e["experiment_output"]):
            out.append({"round": e["round"], "input": _finite(inp), "output": _finite(res)})
    return out


def run_attempt(model: str, hypothesis_id: str, seed: int, ledger_entries: list[dict],
                cfg: C.Config | None = None, complete: Complete | None = None,
                verbose: bool = False, world_spec: dict | None = None,
                on_round: Callable[[dict], None] | None = None,
                max_tokens: int = C.MAX_TOKENS) -> SubmittedAttempt:
    cfg = cfg or C.load()
    hyp = cfg.hypothesis(hypothesis_id)
    if world_spec is None:
        from scienceagent.worlds import get_world
        world_spec = get_world(hyp.world, engine=C.ENGINE, noise_std=cfg.noise_std,
                               noise_seed=seed)
    executor = world_spec["executor"]

    from scienceagent.trajectory_logger import TrajectoryLogger, make_run_id

    tag = f"{_slug(model)}_seed{seed}"
    csv_path = C.TRAJECTORIES_DIR / hyp.id / f"{tag}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.unlink(missing_ok=True)
    logger = TrajectoryLogger(world=hyp.world, executor=executor, csv_path=csv_path,
                              run_id=make_run_id(model, when=_dt.datetime(2000, 1, 1)))

    account = Account(agent=model, hypothesis=hyp.id, budget=cfg.budget)
    agent = MarketAgent(
        cfg=cfg, hyp=hyp, account=account, ledger_entries=ledger_entries, complete=complete,
        on_round=on_round,
        model=model, executor=executor, mission=world_spec["mission"],
        max_tokens=max_tokens, verbose=verbose, noise_seed=seed,
        system_prompt_path=_abs_vendor(world_spec["system_prompt"]),
        instructions_path=_abs_vendor(world_spec["instructions"]),
        law_stub=world_spec["law_stub"], experiment_format=world_spec["experiment_format"],
        trajectory_logger=logger,
    )
    try:
        agent.run()
    finally:
        csv_path.unlink(missing_ok=True)

    log = round_log(agent.conversation_log)
    last = log[-1] if log else {}
    transcript = C.TRANSCRIPTS_DIR / hyp.id / f"{tag}.json"
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text(json.dumps({
        "model": model, "hypothesis_id": hyp.id, "world": hyp.world, "seed": seed,
        "system_prompt": agent._system, "outcome": agent.outcome, "verdict": agent.verdict,
        "account_events": account.events, "rounds": agent.conversation_log,
    }, indent=2, default=str))
    try:
        transcript_record_path = str(transcript.relative_to(C.ROOT))
    except ValueError:
        transcript_record_path = str(transcript)

    return SubmittedAttempt(
        source="live",
        protocol=C.PROTOCOL,
        venue=C.VENUE,
        world=hyp.world,
        solver=model,
        seed=seed,
        # Calibration uses the p stated with the verdict; other outcomes keep theirs in extra.
        stated_p_success=last.get("p_success") if agent.outcome == "verdict" else None,
        rounds=len(log),
        experiments=agent.executor.experiments,
        lab_cost=account.spent,
        submitted_law=None,
        explanation=agent.evidence,
        transcript_path=transcript_record_path,
        extra={
            "hypothesis_id": hyp.id,
            "hypothesis": hyp.hypothesis,
            "resolution_criteria": hyp.resolution_criteria,
            "prize": hyp.prize,
            "outcome": agent.outcome,
            "agent_verdict": agent.verdict,
            "evidence": agent.evidence,
            "withdraw_reason": agent.withdraw_reason,
            "final_assessment": last.get("assessment"),
            "final_p": last.get("p_success"),
            "first_p": log[0]["p_success"] if log else None,
            "budget": cfg.budget,
            "round_fee": cfg.round_fee,
            "experiment_costs": cfg.experiment_costs,
            "max_rounds": cfg.max_rounds,
            "noise_std": cfg.noise_std,
            "velocity_noise_std": cfg.velocity_noise_std,
            "round_log": log,
            "runs": runs(agent.conversation_log),
            "ledger_seen": [e["id"] for e in ledger_entries] if agent.record_bought else [],
            "record_bought": agent.record_bought,
            "record_fee": agent.record_fee,
            "account_events": account.events,
        },
    )


def prompt_chars(hyp_id: str, cfg: C.Config, ledger_entries: list[dict]) -> int:
    """Prompt-size basis for a run, without making an LLM call or creating a trajectory log."""
    from scienceagent.worlds import get_world

    hyp = cfg.hypothesis(hyp_id)
    world_spec = get_world(
        hyp.world,
        engine=C.ENGINE,
        noise_std=cfg.noise_std,
        noise_seed=0,
    )
    account = Account(agent="projection", hypothesis=hyp.id, budget=cfg.budget)
    agent = MarketAgent(
        cfg=cfg,
        hyp=hyp,
        account=account,
        ledger_entries=ledger_entries,
        complete=lambda **kwargs: "",
        model="projection",
        executor=world_spec["executor"],
        mission=world_spec["mission"],
        max_tokens=C.MAX_TOKENS,
        verbose=False,
        system_prompt_path=_abs_vendor(world_spec["system_prompt"]),
        instructions_path=_abs_vendor(world_spec["instructions"]),
        law_stub=world_spec["law_stub"],
        experiment_format=world_spec["experiment_format"],
        trajectory_logger=None,
        noise_seed=0,
    )
    return (
        len(agent._system)
        + len(world_spec["mission"])
        + len(protocol.ledger_block(ledger_entries))
    )
