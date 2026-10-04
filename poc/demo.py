"""Data for /demo: a good run and a bad run on the same question, side by side, then the eval's
design numbers and the naive and market leaderboards.

The bad run is the hook (a real model's false claim the naive rule paid, see poc.animate). The
good run is a confirmed real-model run on the same hypothesis and seed, under the same round cap
and the same rule the agent was told, so the two differ only in the model. For each: its bid, its
rounds in its own words, the separation of probe and source in every launch it bought (noisy
snapshots and the noise-free path), its claim, the checker's ruling and re-fit, and its payout
under each rule.

This is presentation code on the judge's side: it reads ``poc.truth`` after the runs are over.

    uv run python -m poc.calibrate   # once: writes output/poc_calibration.json
    uv run python -m poc.demo        # writes web/data/demo.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dm.store import AttemptStore
from poc import animate, report, truth
from poc import config as C
from poc.calibrate import OUT as CALIBRATION_PATH

OUT = C.ROOT / "web" / "data" / "demo.json"
PLAIN = {
    "baseline:always_supported": "Always says “supported”",
    "baseline:coin_flip": "Flips a coin",
    "baseline:p_hacker": "P-hacker",
    "baseline:reference": "Reference scientist",
    "baseline:abstain": "Always abstains",
    "fake": "Scripted stand-in LLM",
}


def _label(agent: str) -> str:
    return PLAIN.get(agent) or animate.LABELS.get(agent, agent)


def matched_good(records: list, bad) -> object | None:
    """A confirmed real-model run on the bad run's hypothesis and seed, with the same round cap
    and the same rule told; the highest market profit if there are several."""
    x = bad.extra
    cands = [r for r in records if report.is_real(r) and r.passed and r.solver != bad.solver
             and r.seed == bad.seed and r.extra.get("hypothesis_id") == x["hypothesis_id"]
             and r.extra.get("max_rounds") == x.get("max_rounds")
             and r.extra.get("rule") == x.get("rule")
             and bool(r.extra.get("ledger_seen")) == bool(x.get("ledger_seen"))]
    return max(cands, key=lambda r: report._settled(r, "market").get("profit", 0.0), default=None)


def _separation(path) -> np.ndarray:
    a = np.asarray(path, float)
    return np.hypot(*(a[:, 1, :] - a[:, 0, :]).T)


def _drop(launch: dict) -> dict:
    """How far the probe has fallen toward the source: observed snapshots and the true path."""
    obs = _separation(launch["obs"])
    r0 = float(np.hypot(*np.subtract(launch["input"]["pos2"], launch["input"].get("pos1") or [0, 0])))
    out = {"r0": round(r0, 3), "t": launch["obs_t"], "fallen": animate._r(r0 - obs, 3)}
    if launch.get("true") is not None:
        out["true_t"] = launch["true_t"]
        out["true_fallen"] = animate._r(r0 - _separation(launch["true"]), 3)
    return out


def _spend_by_round(record) -> dict[int, float]:
    spent: dict[int, float] = {}
    for e in (record.extra.get("settlements") or {}).get("market", {}).get("events", []):
        if e.get("from") == "agent" and e.get("to") == "lab" and e.get("round") is not None:
            spent[e["round"]] = spent.get(e["round"], 0.0) + e["amount"]
    return spent


def side(record, cfg: C.Config) -> dict:
    x = record.extra
    hyp = cfg.hypothesis(x["hypothesis_id"])
    ruling = record.verdict.get("ruling") or {}
    est = x.get("estimates") or {}
    claimed = {k: (est.get(k) or {}).get("value") for k in ("n", "a3")}
    spend, total = _spend_by_round(record), 0.0
    rounds = []
    for e in x.get("round_log") or []:
        total += spend.get(e.get("round"), 0.0)
        exps = e.get("experiment_input")
        rounds.append({"round": e.get("round"), "action": e.get("action"),
                       "p_success": e.get("p_success"), "assessment": e.get("assessment"),
                       "experiments": len(exps) if isinstance(exps, list) else 0,
                       "spent_so_far": round(total, 2)})
    plan = (x.get("bid") or {}).get("plan") or {}
    st = x.get("settlements") or {}
    pay = {rule: {k: (st.get(rule) or {}).get(k)
                  for k in ("prize_paid", "lab_revenue", "bond_lost", "calibration_bonus", "profit")}
           for rule in C.RULES}
    launches = [animate._launch(hyp.world, record.seed, run, claimed) for run in x.get("runs") or []]
    return {
        "id": record.attempt_id, "agent": _label(record.solver), "seed": record.seed,
        "max_rounds": x.get("max_rounds"), "rule": x.get("rule"),
        "blind": not x.get("ledger_seen"),
        "bid": {"p": x.get("bid_p"), "planned_cost": x.get("planned_cost"),
                "experiments": len(plan.get("experiments") or []),
                "analysis": plan.get("analysis")},
        "rounds": rounds,
        "launches": [_drop(l) for l in launches if l],
        "claim": {"verdict": x.get("agent_verdict"), "final_p": x.get("final_p"),
                  "estimates": est, "evidence": x.get("evidence")},
        "ruling": {"outcome": record.verdict.get("outcome"), "reasons": ruling.get("reasons") or [],
                   "flags": ruling.get("flags") or [], "within": ruling.get("within") or {},
                   "reanalysis": ruling.get("reanalysis")},
        "spent": record.lab_cost, "pay": pay,
        "usd": round((record.llm_usage or {}).get("usd", 0.0), 4),
    }


def question(hyp, cfg: C.Config) -> dict:
    tv = truth.true_values(hyp)
    rule = hyp.supported_if
    return {
        "id": hyp.id, "hypothesis": hyp.hypothesis, "criteria": hyp.resolution_criteria,
        "prize": hyp.prize, "bond": round(cfg.claim_bond * hyp.prize, 2),
        "round_fee": cfg.round_fee, "noise_std": cfg.noise_std, "answer": hyp.answer,
        "supported_if": {"quantity": rule.quantity, "kind": rule.kind, "bounds": list(rule.bounds)},
        "quantities": [{"name": q.name, "meaning": q.meaning, "tolerance": q.tolerance,
                        "true": round(tv[q.name], 5)} for q in hyp.quantities],
    }


def calibration_rows() -> list[dict]:
    if not CALIBRATION_PATH.exists():
        raise SystemExit(f"{CALIBRATION_PATH} is missing: run `uv run python -m poc.calibrate`")
    return [{"hypothesis": c["hypothesis"], "seeds": len(c["seeds"]), "pass_rate": c["pass_rate"],
             "reference_cost": c["reference_cost"], "prize": c["prize"], "solvable": c["solvable"],
             "tolerances": {k: q["tolerance"] for k, q in c["quantities"].items()}}
            for c in json.loads(CALIBRATION_PATH.read_text())]


def boards(records: list) -> dict:
    s = report.summarise(records)
    agents = s["agents"]
    lb = s["leaderboards"]

    def row(e):
        a = agents[e["agent"]]
        return {"agent": _label(e["agent"]), "real": not e["agent"].startswith("baseline:")
                and e["agent"] != "fake", "return_pct": e["return_pct"], "runs": e["runs"],
                "confirmed": a["confirmed"]["value"], "false_claims": a["false_claims"]["value"],
                "coverage": a["coverage"]["value"], "flags": a["flags"]}
    return {"naive": [row(e) for e in lb["naive"]], "market": [row(e) for e in lb["market"]],
            "spearman": lb["spearman_with_confirmed"], "n_runs": s["n_runs"],
            "n_real_runs": s["n_real_runs"]}


def listing(records: list, hyp_id: str) -> dict:
    """Every real-model run on this hypothesis: what a market listing would show as its record."""
    rs = [r for r in records if report.is_real(r) and r.extra.get("hypothesis_id") == hyp_id]
    return {"runs": len(rs),
            "outcomes": {o: sum(report._outcome(r) == o for r in rs)
                         for o in sorted({report._outcome(r) for r in rs})},
            "lab_revenue": round(sum(r.lab_cost for r in rs), 2)}


def build(records: list, cfg: C.Config) -> dict:
    bad = animate.hook_record(records, cfg)
    if bad is None:
        raise SystemExit("no real-model false claim in a power-law world yet")
    good = matched_good(records, bad)
    if good is None:
        raise SystemExit(f"no confirmed real-model run matches {bad.attempt_id}")
    hyp = cfg.hypothesis(bad.extra["hypothesis_id"])
    return {"question": question(hyp, cfg), "good": side(good, cfg), "bad": side(bad, cfg),
            "calibration": calibration_rows(), "boards": boards(records),
            "listing": listing(records, hyp.id),
            "design": {"claim_bond": cfg.claim_bond, "calibration_bonus": cfg.calibration_bonus,
                       "round_fee": cfg.round_fee, "noise_std": cfg.noise_std,
                       "hypotheses": len(cfg.hypotheses)}}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=str(C.ATTEMPTS_PATH))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    cfg = C.load()
    records = AttemptStore(args.store).load(lambda r: r.protocol == C.PROTOCOL)
    data = build(records, cfg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False))
    print(f"good {data['good']['agent']} vs bad {data['bad']['agent']} on "
          f"{data['question']['id']} seed {data['bad']['seed']}, "
          f"{out.stat().st_size / 1e3:.0f} kB -> {out}")


if __name__ == "__main__":
    main()
