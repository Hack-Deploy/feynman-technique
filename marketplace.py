"""Agent marketplace for the demo: venues, open bounties, leaderboards and agent submissions.

A venue lists bounties. Anyone can submit an agent; it plays the bounties through the same text
protocol as a model (assessment, p_success, run_experiment, verdict, estimate), pays the lab for
its own experiments, and is judged by the independent checker. Every run is settled under both
reward rules, so the board shows where an agent would rank if a verdict alone were paid (naive)
and where it ranks when only a confirmed answer is paid (market).

Submitted agents are scripted strategies (no API calls); the board also carries the real Claude
runs already in the bench store. Submissions go to attempts/market_submissions.jsonl.
"""

from __future__ import annotations

import re
import threading
import uuid

from dm.store import AttemptStore
from dm.types import AttemptRecord
from poc import baselines, bench, config as C, report, spend
from poc.attempt import run_attempt

SUBMISSIONS_PATH = C.ROOT / "attempts" / "market_submissions.jsonl"
AGENT_PREFIX = "agent:"
MAX_SEEDS = 3
MAX_NAME = 40
RESULTS_SHOWN = 40

VENUES = {
    "discoverphysics": {
        "id": "discoverphysics",
        "name": "DiscoverPhysics",
        "tagline": "Simulated worlds with non-standard physics. Every claim is checked exactly.",
        "host": "Discovery Market",
        "status": "open",
    },
}

# What a submitter can enter: each strategy is a scripted agent that speaks the protocol.
STRATEGIES = {
    "careful": {
        "policy": "reference",
        "label": "Careful scientist",
        "blurb": "Runs a full experiment design, fits a force law to every launch it paid for, "
                 "and reports the verdict its own estimates imply.",
    },
    "p_hacker": {
        "policy": "p_hacker",
        "label": "Rerun until it works",
        "blurb": "Wants \"supported\". Re-runs one cheap launch and reports the single run "
                 "that looks best.",
    },
    "guesser": {
        "policy": "always_supported",
        "label": "Confident guesser",
        "blurb": "Claims \"supported\" at once with the values the hypothesis implies. "
                 "Buys no experiments.",
    },
    "coin_flip": {
        "policy": "coin_flip",
        "label": "Coin flip",
        "blurb": "A seeded coin picks the verdict; estimates are the obvious guess for that side.",
    },
}

_LOCK = threading.RLock()
_JOBS: dict[str, dict] = {}
_ACTIVE_JOB: str | None = None


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _submissions() -> list[AttemptRecord]:
    return AttemptStore(SUBMISSIONS_PATH).load(lambda r: r.protocol == C.PROTOCOL)


def _real_runs() -> list[AttemptRecord]:
    return AttemptStore(C.ATTEMPTS_PATH).load(
        lambda r: r.protocol == C.PROTOCOL and report.is_real(r)
        and r.extra.get("rule", "market") == "market" and r.extra.get("experiments_enabled", True))


def _model_labels() -> dict[str, str]:
    try:
        return {m.id: m.label for m in spend.load_settings().models}
    except (OSError, ValueError, KeyError):
        return {}


def _agent(r: AttemptRecord, labels: dict[str, str]) -> dict:
    if r.solver.startswith(AGENT_PREFIX):
        strategy = r.extra.get("strategy")
        return {"key": r.solver, "name": r.extra.get("agent_name") or r.solver,
                "kind": "submitted", "strategy": strategy,
                "strategy_label": STRATEGIES.get(strategy or "", {}).get("label")}
    return {"key": r.solver, "name": labels.get(r.solver, r.solver), "kind": "model",
            "strategy": None, "strategy_label": "Claude on the published scaffold"}


def _settled(r: AttemptRecord, rule: str) -> dict:
    return (r.extra.get("settlements") or {}).get(rule) or {}


def _run(r: AttemptRecord, labels: dict[str, str]) -> dict:
    ruling = r.verdict.get("ruling") or {}
    naive, market = _settled(r, "naive"), _settled(r, "market")
    return {
        "attempt_id": r.attempt_id,
        "agent": _agent(r, labels),
        "bounty": r.extra.get("hypothesis_id"),
        "seed": r.seed,
        "prize": r.extra.get("prize"),
        "bond": r.extra.get("claim_bond"),
        "outcome": ruling.get("outcome") or r.extra.get("outcome"),
        "agent_verdict": r.extra.get("agent_verdict"),
        "answer": r.verdict.get("answer"),
        "estimates": r.extra.get("estimates") or {},
        "errors": ruling.get("errors") or {},
        "within": ruling.get("within") or {},
        "reasons": ruling.get("reasons") or [],
        "flags": ruling.get("flags") or [],
        "rounds": r.rounds,
        "experiments": r.experiments,
        "spent": r.lab_cost,
        "bid_p": r.extra.get("bid_p"),
        "naive": {"paid": bool(naive.get("paid")), "profit": naive.get("profit", 0.0)},
        "market": {"paid": bool(market.get("paid")), "profit": market.get("profit", 0.0),
                   "bond_lost": market.get("bond_lost", 0.0),
                   "calibration_bonus": market.get("calibration_bonus", 0.0)},
        "round_log": [
            {k: e.get(k) for k in ("round", "action", "assessment", "p_success", "estimates",
                                   "experiments", "experiments_cost", "spent_so_far")}
            for e in r.extra.get("round_log") or []
        ],
        "created_at": r.created_at,
    }


def board(records: list[AttemptRecord], labels: dict[str, str]) -> list[dict]:
    """One row per agent, ranked by mean market return per run (% of the prize); the naive rank
    is what the same runs would earn if any clear verdict were paid."""
    groups: dict[str, list[AttemptRecord]] = {}
    for r in records:
        groups.setdefault(r.solver, []).append(r)
    rows = []
    for rs in groups.values():
        s = report.summarise_agent(rs)
        rows.append({
            **_agent(rs[0], labels),
            "runs": s["runs"],
            "confirmed": s["confirmed"]["value"],
            "false_claims": s["false_claims"]["value"],
            "clear_claims": s["clear_claims"]["value"],
            "spent": s["spent"],
            "market_pct": s["return_pct_market"]["value"] or 0.0,
            "naive_pct": s["return_pct_naive"]["value"] or 0.0,
            "market_profit": s["profit_market"],
            "naive_profit": s["profit_naive"],
            "bonds_lost": s["bonds_lost"],
            "flags": s["flags"],
        })
    naive_order = sorted(rows, key=lambda x: (-x["naive_pct"], -x["clear_claims"], x["name"]))
    for i, row in enumerate(naive_order, 1):
        row["naive_rank"] = i
    rows.sort(key=lambda x: (-x["market_pct"], -x["confirmed"], x["name"]))
    for i, row in enumerate(rows, 1):
        row["market_rank"] = i
    return rows


def _bounty(hyp: C.Hypothesis, cfg: C.Config, records: list[AttemptRecord]) -> dict:
    rs = [r for r in records if r.extra.get("hypothesis_id") == hyp.id]
    return {
        "id": hyp.id,
        "world": hyp.world,
        "hypothesis": hyp.hypothesis,
        "resolution_criteria": hyp.resolution_criteria,
        "quantities": [{"name": q.name, "meaning": q.meaning, "tolerance": q.tolerance}
                       for q in hyp.quantities],
        "prize": hyp.prize,
        "bond": round(cfg.claim_bond * hyp.prize, 3),
        "runs": len(rs),
        "confirmed": sum(r.passed for r in rs),
        "false_claims": sum((r.verdict.get("outcome") == "false_claim") for r in rs),
        "status": "open",
    }


def venues() -> dict:
    cfg = C.load()
    records = _real_runs() + _submissions()
    labels = _model_labels()
    rows = board(records, labels)
    v = VENUES["discoverphysics"]
    return {"venues": [{
        **v,
        "bounties": len(cfg.hypotheses),
        "prize_pool": sum(h.prize for h in cfg.hypotheses),
        "agents": len(rows),
        "runs": len(records),
        "confirmed": sum(r.passed for r in records),
        "false_claims": sum(r.verdict.get("outcome") == "false_claim" for r in records),
        "leader": rows[0]["name"] if rows else None,
    }]}


def venue(venue_id: str, bounty_id: str | None = None) -> dict | None:
    if venue_id not in VENUES:
        return None
    cfg = C.load()
    ids = {h.id for h in cfg.hypotheses}
    if bounty_id and bounty_id not in ids:
        raise ValueError(f"unknown bounty: {bounty_id}")
    real, submitted = _real_runs(), _submissions()
    records = real + submitted
    labels = _model_labels()
    scoped = [r for r in records if not bounty_id or r.extra.get("hypothesis_id") == bounty_id]
    in_scope = {r.attempt_id for r in scoped}
    # Store order is append order (records carry no wall-clock time): newest submissions first.
    recent = [r for r in submitted[::-1] + real[::-1] if r.attempt_id in in_scope][:RESULTS_SHOWN]
    return {
        "venue": VENUES[venue_id],
        "scope": bounty_id,
        "totals": {"agents": len({r.solver for r in records}), "runs": len(records)},
        "rules": {
            "round_fee": cfg.round_fee,
            "experiment_costs": dict(cfg.experiment_costs),
            "claim_bond": cfg.claim_bond,
            "calibration_bonus": cfg.calibration_bonus,
            "max_rounds": cfg.max_rounds,
            "market": cfg.market_payout_rule.format(bond="30% of the prize"),
            "naive": cfg.naive_payout_rule,
        },
        "bounties": [_bounty(h, cfg, records) for h in cfg.hypotheses],
        "board": board(scoped, labels),
        "results": [_run(r, labels) for r in recent],
        "strategies": [{"id": k, "label": v["label"], "blurb": v["blurb"]}
                       for k, v in STRATEGIES.items()],
        "max_seeds": MAX_SEEDS,
    }


def _next_seed(records: list[AttemptRecord], solver: str, bounty: str) -> int:
    used = {r.seed for r in records if r.solver == solver and r.extra.get("hypothesis_id") == bounty}
    seed = 0
    while seed in used:
        seed += 1
    return seed


def plan(name: str, strategy: str, bounties: list[str], seeds: int) -> dict:
    """Check a submission and fix its runs (bounty, seed) without running anything."""
    name = " ".join(str(name).split())
    if not name or len(name) > MAX_NAME:
        raise ValueError(f"Give the agent a name of 1 to {MAX_NAME} characters.")
    key = slug(name)
    if not key:
        raise ValueError("The name needs at least one letter or digit.")
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy}")
    if not isinstance(seeds, int) or isinstance(seeds, bool) or not 1 <= seeds <= MAX_SEEDS:
        raise ValueError(f"seeds must be between 1 and {MAX_SEEDS}")
    cfg = C.load()
    ids = [h.id for h in cfg.hypotheses]
    if not bounties or any(b not in ids for b in bounties):
        raise ValueError("Pick at least one open bounty.")
    solver = AGENT_PREFIX + key
    records = _submissions()
    for r in records:
        if r.solver == solver and r.extra.get("strategy") != strategy:
            raise ValueError(f"\"{name}\" is already on the board with another strategy; "
                             "pick a new name.")
    runs = []
    for b in [i for i in ids if i in set(bounties)]:
        first = _next_seed(records, solver, b)
        runs += [{"bounty": b, "seed": first + k} for k in range(seeds)]
    return {"name": name, "solver": solver, "strategy": strategy, "runs": runs}


def run_one(p: dict, bounty: str, seed: int, cfg: C.Config | None = None) -> AttemptRecord:
    """Play one bounty with a submitted agent, judge it, settle it, and store it."""
    cfg = cfg or C.load()
    hyp = cfg.hypothesis(bounty)
    policy = baselines.POLICIES[STRATEGIES[p["strategy"]]["policy"]](cfg, hyp, seed)
    submitted = run_attempt(p["solver"], bounty, seed, [], cfg=cfg, complete=policy)
    submitted.extra.update({"agent_name": p["name"], "strategy": p["strategy"],
                            "marketplace": "discoverphysics"})
    record = bench.resolve(hyp, submitted, cfg)
    AttemptStore(SUBMISSIONS_PATH).append([record])
    return record


def _work(job_id: str, p: dict) -> None:
    global _ACTIVE_JOB
    labels = _model_labels()
    try:
        cfg = C.load()
        for item in p["runs"]:
            record = run_one(p, item["bounty"], item["seed"], cfg)
            with _LOCK:
                _JOBS[job_id]["runs"].append(_run(record, labels))
        with _LOCK:
            _JOBS[job_id]["state"] = "done"
    except Exception as exc:
        with _LOCK:
            _JOBS[job_id]["state"] = "error"
            _JOBS[job_id]["error"] = str(exc) or type(exc).__name__
    finally:
        with _LOCK:
            if _ACTIVE_JOB == job_id:
                _ACTIVE_JOB = None


def submit(name: str, strategy: str, bounties: list[str], seeds: int) -> dict:
    global _ACTIVE_JOB
    with _LOCK:
        if _ACTIVE_JOB is not None:
            raise RuntimeError("Another agent is running its experiments; try again when it finishes.")
        p = plan(name, strategy, bounties, seeds)
        job_id = uuid.uuid4().hex
        _JOBS[job_id] = {"state": "running", "agent": {"key": p["solver"], "name": p["name"],
                                                         "strategy": strategy},
                         "total": len(p["runs"]), "runs": [], "error": None}
        _ACTIVE_JOB = job_id
        threading.Thread(target=_work, args=(job_id, p), daemon=True).start()
    return {"ok": True, "job_id": job_id, "total": len(p["runs"]), "agent": _JOBS[job_id]["agent"]}


def job(job_id: str) -> dict | None:
    with _LOCK:
        j = _JOBS.get(job_id)
        if j is None:
            return None
        return {**j, "runs": list(j["runs"])}
