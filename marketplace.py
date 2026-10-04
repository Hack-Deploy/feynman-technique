"""Agent marketplace for the demo: venues, open bounties, leaderboards and agent submissions.

A venue lists bounties. Anyone can submit an agent as a git repo with agent.py at its root
(dm.repo_agent; format in agents/README.md). It plays the bounties through the same text protocol
as a model (assessment, p_success, run_experiment, verdict, estimate), pays the lab for its own
experiments, and is judged by the independent checker. Every run is settled under both
reward rules, so the board shows where an agent would rank if a verdict alone were paid (naive)
and where it ranks when only a confirmed answer is paid (market).

Submitted agents make no API calls through the market. The board also carries the runs already in
the bench store: real Claude runs and the house baselines (poc.baselines), all run through the
same simulator and checker. Submissions go to attempts/market_submissions.jsonl.
"""

from __future__ import annotations

import re
import threading
import uuid
from itertools import zip_longest

from dm.store import AttemptStore
from dm.types import AttemptRecord
from dm import repo_agent
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

# Example repos offered on the submit panel (agents/examples).
EXAMPLES = [
    {"repo": "examples/careful-lab", "name": "Careful Lab",
     "blurb": "Runs a full experiment design and claims what its own fit implies."},
    {"repo": "examples/shortcut-labs", "name": "Shortcut Labs",
     "blurb": "Wants \"supported\": reruns one cheap launch and reports the best-looking run."},
    {"repo": "examples/yes-man", "name": "Yes Man",
     "blurb": "No experiments. Says \"supported\" with numbers that fit the rule."},
]
# Submissions made before agents were repos named a scripted strategy; this is its repo now.
LEGACY_REPOS = {"careful": "examples/careful-lab", "p_hacker": "examples/shortcut-labs",
                "guesser": "examples/yes-man"}

# Short card titles and icon glyphs for the bounties (display only).
BOUNTY_CARDS = {
    "gravity-inverse-square": ("Does gravity fall off as 1/r²?", "1/r²"),
    "fractional-2d-gravity": ("Does the pull fall off as 1/r?", "1/r"),
    "yukawa-screened": ("Is the force screened at long range?", "e⁻ʳ"),
    "oscillator-time-varying": ("Does the pull change over time?", "∿"),
    "dark-matter-unseen-pull": ("Is unseen matter pulling the probes?", "◌"),
    "circle-ordinary-gravity": ("Does the ring obey 2D gravity?", "◯"),
    "ether-outward-push": ("Is space pushing particles apart?", "⇱"),
    "hubble-outward-push": ("Is space pushing particles apart?", "⇲"),
}
HOUSE_NAMES = {
    "reference": "Reference design",
    "p_hacker": "P-hacker",
    "always_supported": "Always yes",
    "coin_flip": "Coin flipper",
    "abstain": "Sits it out",
}

_LOCK = threading.RLock()
_JOBS: dict[str, dict] = {}
_ACTIVE_JOB: str | None = None


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _submissions() -> list[AttemptRecord]:
    return AttemptStore(SUBMISSIONS_PATH).load(lambda r: r.protocol == C.PROTOCOL)


def _bench_runs() -> list[AttemptRecord]:
    """Real Claude runs, then house baselines, as told the market rule with experiments on."""
    rs = AttemptStore(C.ATTEMPTS_PATH).load(
        lambda r: r.protocol == C.PROTOCOL
        and (report.is_real(r) or r.solver.startswith(baselines.PREFIX))
        and r.extra.get("rule", "market") == "market" and r.extra.get("experiments_enabled", True))
    house: dict[str, list[AttemptRecord]] = {}
    for r in rs:
        if not report.is_real(r):
            house.setdefault(r.solver, []).append(r)
    # Interleave the house agents so a feed of recent runs is not one agent at a time.
    mixed = [r for group in zip_longest(*house.values()) for r in group if r is not None]
    return [r for r in rs if report.is_real(r)] + mixed


def _model_labels() -> dict[str, str]:
    try:
        return {m.id: m.label for m in spend.load_settings().models}
    except (OSError, ValueError, KeyError):
        return {}


def _repo_of(r: AttemptRecord) -> str | None:
    return r.extra.get("repo") or LEGACY_REPOS.get(r.extra.get("strategy") or "")


def _agent(r: AttemptRecord, labels: dict[str, str]) -> dict:
    if r.solver.startswith(AGENT_PREFIX):
        repo, commit = _repo_of(r), r.extra.get("commit")
        shown = (repo or "").removeprefix("https://")
        return {"key": r.solver, "name": r.extra.get("agent_name") or r.solver,
                "kind": "submitted", "strategy": None, "repo": repo, "commit": commit,
                "strategy_label": shown + (f" @ {commit[:7]}" if commit else "")}
    if r.solver.startswith(baselines.PREFIX):
        policy = r.solver[len(baselines.PREFIX):]
        return {"key": r.solver, "name": HOUSE_NAMES.get(policy, policy), "kind": "house",
                "strategy": policy, "strategy_label": "House agent · scripted"}
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
        "agent_errors": r.extra.get("agent_errors") or [],
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
    bids = [r.extra["bid_p"] for r in rs if r.extra.get("bid_p")]
    claims = [r.extra.get("agent_verdict") for r in rs
              if r.extra.get("outcome") == "verdict" and r.extra.get("agent_verdict") in C.ANSWERS]
    title, glyph = BOUNTY_CARDS.get(hyp.id, (hyp.hypothesis, "?"))
    return {
        "id": hyp.id,
        "title": title,
        "glyph": glyph,
        "world": hyp.world,
        "agents": len({r.solver for r in rs}),
        "confidence": round(sum(bids) / len(bids), 3) if bids else None,
        "claims": {a: claims.count(a) for a in C.ANSWERS},
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
    records = _bench_runs() + _submissions()
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
    real, submitted = _bench_runs(), _submissions()
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
        "examples": EXAMPLES,
        "max_seeds": MAX_SEEDS,
    }


def _next_seed(records: list[AttemptRecord], solver: str, bounty: str) -> int:
    used = {r.seed for r in records if r.solver == solver and r.extra.get("hypothesis_id") == bounty}
    seed = 0
    while seed in used:
        seed += 1
    return seed


def plan(name: str, repo: str, bounties: list[str], seeds: int) -> dict:
    """Check a submission and fix its runs (bounty, seed) without running anything."""
    name = " ".join(str(name).split())
    if not name or len(name) > MAX_NAME:
        raise ValueError(f"Give the agent a name of 1 to {MAX_NAME} characters.")
    key = slug(name)
    if not key:
        raise ValueError("The name needs at least one letter or digit.")
    source = repo_agent.normalise(repo)
    if not isinstance(seeds, int) or isinstance(seeds, bool) or not 1 <= seeds <= MAX_SEEDS:
        raise ValueError(f"seeds must be between 1 and {MAX_SEEDS}")
    cfg = C.load()
    ids = [h.id for h in cfg.hypotheses]
    if not bounties or any(b not in ids for b in bounties):
        raise ValueError("Pick at least one open bounty.")
    solver = AGENT_PREFIX + key
    records = _submissions()
    for r in records:
        if r.solver == solver and _repo_of(r) != source:
            raise ValueError(f"\"{name}\" is already on the board with another repo; "
                             "pick a new name.")
    runs = []
    for b in [i for i in ids if i in set(bounties)]:
        first = _next_seed(records, solver, b)
        runs += [{"bounty": b, "seed": first + k} for k in range(seeds)]
    return {"name": name, "solver": solver, "repo": source, "runs": runs}


def request(hyp: C.Hypothesis, cfg: C.Config, seed: int) -> dict:
    """What a repo agent reads each round besides the conversation: public posting data only."""
    rule = hyp.supported_if
    return {
        "protocol": C.PROTOCOL,
        "seed": seed,
        "bounty": {
            "id": hyp.id, "world": hyp.world, "hypothesis": hyp.hypothesis,
            "resolution_criteria": hyp.resolution_criteria, "prize": hyp.prize,
            "bond": round(cfg.claim_bond * hyp.prize, 3), "particles": hyp.particles,
            "quantities": [{"name": q.name, "meaning": q.meaning, "tolerance": q.tolerance}
                           for q in hyp.quantities],
            "supported_if": {"quantity": rule.quantity, "kind": rule.kind,
                             "bounds": list(rule.bounds)},
        },
        "prices": {"round_fee": cfg.round_fee, "experiment_costs": dict(cfg.experiment_costs)},
        "max_rounds": cfg.max_rounds,
    }


def run_one(p: dict, repo: repo_agent.AgentRepo, bounty: str, seed: int,
            cfg: C.Config | None = None) -> AttemptRecord:
    """Play one bounty with a submitted agent's repo, judge it, settle it, and store it."""
    cfg = cfg or C.load()
    hyp = cfg.hypothesis(bounty)
    agent = repo_agent.RepoAgent(repo, request(hyp, cfg, seed))
    submitted = run_attempt(p["solver"], bounty, seed, [], cfg=cfg, complete=agent)
    submitted.extra.update({"agent_name": p["name"], "repo": repo.source, "commit": repo.commit,
                            "agent_errors": agent.errors, "marketplace": "discoverphysics"})
    record = bench.resolve(hyp, submitted, cfg)
    AttemptStore(SUBMISSIONS_PATH).append([record])
    return record


def _work(job_id: str, p: dict) -> None:
    global _ACTIVE_JOB
    labels = _model_labels()
    try:
        repo = repo_agent.fetch(p["repo"])
        with _LOCK:
            _JOBS[job_id].update(phase="running", commit=repo.commit)
        cfg = C.load()
        for item in p["runs"]:
            record = run_one(p, repo, item["bounty"], item["seed"], cfg)
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


def submit(name: str, repo: str, bounties: list[str], seeds: int) -> dict:
    global _ACTIVE_JOB
    with _LOCK:
        if _ACTIVE_JOB is not None:
            raise RuntimeError("Another agent is running its experiments; try again when it finishes.")
        p = plan(name, repo, bounties, seeds)
        job_id = uuid.uuid4().hex
        _JOBS[job_id] = {"state": "running", "agent": {"key": p["solver"], "name": p["name"],
                                                         "repo": p["repo"]},
                         "phase": "fetching",
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
