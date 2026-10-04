"""Report over the bounty runs: the naive and market leaderboards side by side.

    uv run python -m poc.report               # output/poc_report.json + both leaderboards
    uv run python -m poc.report --dashboard   # also output/poc_dashboard.html

Per agent: runs offered, coverage (bids accepted), clear claims (what the naive rule pays),
confirmed and false claims, confirmed per 100 credits, profit under each rule, Brier scores for
the bid-time and final p against "confirmed", planned vs actual cost, and checker flags. Every
number carries its n. The hook is picked only from real-model runs that the naive rule paid and
the checker rejected.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from statistics import mean

from dm.store import AttemptStore
from poc import config as C

BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
BASELINE_PREFIX = "baseline:"


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(mean(xs), 4) if xs else None


def brier(pairs: list[tuple[float, bool]]) -> float | None:
    return round(mean((p - float(y)) ** 2 for p, y in pairs), 4) if pairs else None


def reliability(pairs: list[tuple[float, bool]]) -> list[dict]:
    rows = []
    for lo, hi in zip(BINS, BINS[1:]):
        inside = [(p, y) for p, y in pairs if lo <= p < hi or (hi == 1.0 and p == 1.0)]
        rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": len(inside),
                     "mean_p": _mean([p for p, _ in inside]),
                     "success_rate": _mean([float(y) for _, y in inside])})
    return rows


def agent_label(r) -> str:
    x = r.extra
    label = r.solver
    if not x.get("experiments_enabled", True):
        label += " (prior only)"
    if x.get("rule", "market") == "naive":
        label += " [told naive]"
    return label


def is_real(r) -> bool:
    return not r.solver.startswith(BASELINE_PREFIX) and r.source == "live" and r.solver != "fake"


def _outcome(r) -> str:
    return r.verdict.get("outcome") or r.extra.get("outcome") or "unknown"


def _settled(r, rule: str) -> dict:
    return (r.extra.get("settlements") or {}).get(rule) or {}


def _rank(values: dict[str, float]) -> dict[str, float]:
    """Average ranks, 1 = highest."""
    order = sorted(values, key=lambda k: -values[k])
    ranks: dict[str, float] = {}
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in order[i:j + 1]:
            ranks[k] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a: dict[str, float], b: dict[str, float]) -> float | None:
    keys = sorted(set(a) & set(b))
    if len(keys) < 3:
        return None
    ra, rb = _rank({k: a[k] for k in keys}), _rank({k: b[k] for k in keys})
    ma, mb = mean(ra.values()), mean(rb.values())
    cov = sum((ra[k] - ma) * (rb[k] - mb) for k in keys)
    va = sum((ra[k] - ma) ** 2 for k in keys) ** 0.5
    vb = sum((rb[k] - mb) ** 2 for k in keys) ** 0.5
    return round(cov / (va * vb), 3) if va and vb else None


def _return_pct(rs: list, rule: str) -> float | None:
    """Mean profit per run as a percentage of that run's prize: comparable across agents with
    different numbers of runs and across prizes of different sizes."""
    xs = [100 * _settled(r, rule).get("profit", 0.0) / r.extra["prize"]
          for r in rs if r.extra.get("prize")]
    return round(mean(xs), 1) if xs else None


def summarise_agent(rs: list) -> dict:
    outcomes = Counter(_outcome(r) for r in rs)
    n = len(rs)
    bid = [r for r in rs if _outcome(r) not in ("walked_away", "declined")]
    clear = [r for r in rs if r.extra.get("outcome") == "verdict"
             and r.extra.get("agent_verdict") in C.ANSWERS]
    confirmed = sum(r.passed for r in rs)
    spent = round(sum(r.lab_cost for r in rs), 3)
    bid_pairs = [(r.extra["bid_p"], r.passed) for r in bid if r.extra.get("bid_p") is not None]
    final_pairs = [(r.extra["final_p"], r.passed) for r in clear
                   if r.extra.get("final_p") is not None]
    cost_pairs = [(r.extra["planned_cost"], r.lab_cost) for r in bid
                  if r.extra.get("planned_cost")]
    flags = Counter(f["flag"] for r in rs for f in (r.verdict.get("ruling") or {}).get("flags", []))
    usd = sum((r.llm_usage or {}).get("usd", 0.0) for r in rs)
    return {
        "runs": n,
        "outcomes": dict(outcomes),
        "coverage": {"value": round(len(bid) / n, 4) if n else None, "n": n},
        "clear_claims": {"value": len(clear), "n": n},
        "confirmed": {"value": confirmed, "n": n},
        "false_claims": {"value": outcomes.get("false_claim", 0), "n": n},
        "stopped": {"value": outcomes.get("stopped", 0), "n": n},
        "verdict_matches_answer": {"value": sum(r.verdict.get("verdict_matches_answer", False)
                                                for r in clear), "n": len(clear)},
        "confirmed_per_100_credits": {"value": round(100 * confirmed / spent, 3) if spent else None,
                                      "n": n},
        "spent": spent,
        "profit_market": round(sum(_settled(r, "market").get("profit", 0.0) for r in rs), 3),
        "profit_naive": round(sum(_settled(r, "naive").get("profit", 0.0) for r in rs), 3),
        "return_pct_market": {"value": _return_pct(rs, "market"), "n": n},
        "return_pct_naive": {"value": _return_pct(rs, "naive"), "n": n},
        "bonds_lost": round(sum(_settled(r, "market").get("bond_lost", 0.0) for r in rs), 3),
        "calibration_bonus": round(sum(_settled(r, "market").get("calibration_bonus", 0.0)
                                       for r in rs), 3),
        "brier_bid_p": {"value": brier(bid_pairs), "n": len(bid_pairs)},
        "brier_final_p": {"value": brier(final_pairs), "n": len(final_pairs)},
        "mean_bid_p": {"value": _mean([p for p, _ in bid_pairs]), "n": len(bid_pairs)},
        "reliability_bid_p": reliability(bid_pairs),
        "cost_ratio_actual_over_planned": {
            "value": _mean([a / p for p, a in cost_pairs]), "n": len(cost_pairs)},
        "flags": dict(flags),
        "llm_usd": round(usd, 4),
    }


def boards(agents: dict[str, dict]) -> dict:
    """Ranked by mean return per run (% of the prize), so agents with different run counts
    compare fairly; totals are kept alongside."""
    def ret(a, rule):
        v = agents[a][f"return_pct_{rule}"]["value"]
        return v if v is not None else 0.0
    naive = sorted(agents, key=lambda a: (-ret(a, "naive"), -agents[a]["clear_claims"]["value"]))
    market = sorted(agents, key=lambda a: -ret(a, "market"))
    rate = {a: v["confirmed"]["value"] / v["runs"] for a, v in agents.items() if v["runs"]}
    n = len(agents)
    return {
        "naive": [{"agent": a, "return_pct": ret(a, "naive"), "profit": agents[a]["profit_naive"],
                   "runs": agents[a]["runs"], "clear_claims": agents[a]["clear_claims"]}
                  for a in naive],
        "market": [{"agent": a, "return_pct": ret(a, "market"),
                    "profit": agents[a]["profit_market"], "runs": agents[a]["runs"],
                    "confirmed": agents[a]["confirmed"]} for a in market],
        "spearman_with_confirmed": {
            "naive": spearman({a: ret(a, "naive") for a in agents}, rate),
            "market": spearman({a: ret(a, "market") for a in agents}, rate),
            "label": f"directional, n = {n} agents",
        },
    }


def hook(records: list) -> dict | None:
    """A real-model run the naive rule paid and the checker rejected; the largest naive gain."""
    cands = [r for r in records if is_real(r) and _settled(r, "naive").get("paid")
             and _outcome(r) == "false_claim"]
    if not cands:
        return None
    r = max(cands, key=lambda r: (_settled(r, "naive").get("profit", 0.0), r.extra.get("final_p") or 0))
    ruling = r.verdict.get("ruling") or {}
    return {"attempt_id": r.attempt_id, "agent": agent_label(r),
            "hypothesis_id": r.extra.get("hypothesis_id"), "seed": r.seed,
            "agent_verdict": r.extra.get("agent_verdict"), "estimates": r.extra.get("estimates"),
            "bid_p": r.extra.get("bid_p"), "final_p": r.extra.get("final_p"),
            "reasons": ruling.get("reasons"), "flags": ruling.get("flags"),
            "naive_profit": _settled(r, "naive").get("profit"),
            "market_profit": _settled(r, "market").get("profit"),
            "transcript_path": r.transcript_path}


def by_solver(records: list) -> dict[str, dict]:
    """Plain per-solver numbers, in the shape the /live page and the live cache read."""
    groups = defaultdict(list)
    for r in records:
        groups[r.solver].append(r)
    out = {}
    for solver, rs in groups.items():
        final = [(r.extra["final_p"], r.passed) for r in rs
                 if r.extra.get("outcome") == "verdict" and r.extra.get("final_p") is not None]
        out[solver] = {"runs": len(rs), "correct_verdicts": sum(r.passed for r in rs),
                       "brier_final_p": brier(final),
                       "spent": round(sum(r.lab_cost for r in rs), 3),
                       "profit": round(sum(r.extra.get("profit", 0.0) for r in rs), 3)}
    return out


def summarise(records: list) -> dict:
    by_agent = defaultdict(list)
    for r in records:
        by_agent[agent_label(r)].append(r)
    agents = {a: summarise_agent(rs) for a, rs in sorted(by_agent.items())}
    by_hyp = defaultdict(list)
    for r in records:
        by_hyp[r.extra.get("hypothesis_id")].append(r)
    found = hook(records)
    return {
        "agents": agents,
        "models": by_solver(records),
        "leaderboards": boards(agents),
        "hook": found,
        "hook_note": None if found else
        "No real-model run was paid by the naive rule and rejected by the checker.",
        "hypotheses": {k: {"runs": len(v), "confirmed": sum(r.passed for r in v),
                           "false_claims": sum(_outcome(r) == "false_claim" for r in v),
                           "mean_spent": _mean([r.lab_cost for r in v])}
                       for k, v in sorted(by_hyp.items())},
        "n_runs": len(records),
        "n_real_runs": sum(is_real(r) for r in records),
    }


def print_boards(report: dict) -> None:
    a = report["agents"]
    lb = report["leaderboards"]
    print("\nMean profit per run, % of the prize (runs)")
    print(f"{'NAIVE: paid for any clear verdict':48s}   MARKET: paid only if confirmed")
    for i, (nv, mk) in enumerate(zip(lb["naive"], lb["market"]), 1):
        print(f"{i:2d}. {nv['agent'][:30]:30s} {nv['return_pct']:+7.1f}% ({nv['runs']:2d})    "
              f"{i:2d}. {mk['agent'][:30]:30s} {mk['return_pct']:+7.1f}% ({mk['runs']:2d})")
    sp = lb["spearman_with_confirmed"]
    print(f"Spearman with confirmed answers ({sp['label']}): naive {sp['naive']}, market {sp['market']}")
    print(f"\n{'agent':32s} {'runs':>4s} {'cover':>5s} {'claims':>6s} {'conf':>4s} {'false':>5s} "
          f"{'stop':>4s} {'brier_bid':>9s} {'cost×':>6s} flags")
    for name, s in a.items():
        print(f"{name[:32]:32s} {s['runs']:4d} {s['coverage']['value'] or 0:5.2f} "
              f"{s['clear_claims']['value']:6d} {s['confirmed']['value']:4d} "
              f"{s['false_claims']['value']:5d} {s['stopped']['value']:4d} "
              f"{s['brier_bid_p']['value']!s:>9s} "
              f"{s['cost_ratio_actual_over_planned']['value']!s:>6s} "
              f"{','.join(f'{k}:{v}' for k, v in s['flags'].items()) or '-'}")
    print("\nhook:", json.dumps(report["hook"])[:400] if report["hook"] else report["hook_note"])


DASHBOARD_TEMPLATE = C.ROOT / "poc" / "dashboard.html"
DASHBOARD_OUT = C.ROOT / "output" / "poc_dashboard.html"
_PLACEHOLDER = '<script id="runs-data" type="application/json">null</script>'


def dashboard_data(records: list, cfg: C.Config) -> dict:
    """Runs in the shape poc/dashboard.html reads."""
    runs = []
    for r in records:
        x = r.extra
        log = x.get("round_log") or []
        outcome = _outcome(r)
        result = {"confirmed": "correct", "false_claim": "wrong", "stopped": "inconclusive"}.get(outcome)
        runs.append({
            "hypothesis_id": x.get("hypothesis_id"), "model": agent_label(r), "seed": r.seed,
            "outcome": x.get("outcome"), "verdict_result": result, "passed": r.passed,
            "rounds": r.rounds, "experiments": r.experiments, "spent": r.lab_cost,
            "prize": x.get("prize"), "prize_paid": x.get("prize_paid", 0.0),
            "first_p": x.get("first_p"), "final_p": x.get("final_p"),
            "round_log": [{**e, "charged": (e.get("experiments_cost") or 0) + (e.get("round_fee") or 0)}
                          for e in log],
        })
    return {"dummy": False,
            "hypotheses": [{"id": h.id, "hypothesis": h.hypothesis, "prize": h.prize}
                           for h in cfg.hypotheses],
            "runs": runs}


def write_dashboard(records: list, cfg: C.Config, out=DASHBOARD_OUT) -> None:
    page = DASHBOARD_TEMPLATE.read_text()
    if _PLACEHOLDER not in page:
        raise RuntimeError("dashboard.html no longer has the runs-data placeholder")
    data = json.dumps(dashboard_data(records, cfg)).replace("</", "<\\/")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page.replace(_PLACEHOLDER, _PLACEHOLDER.replace(">null<", f">{data}<")))
    print(f"dashboard written to {out}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=str(C.ATTEMPTS_PATH))
    ap.add_argument("--out", default=str(C.REPORT_PATH))
    ap.add_argument("--dashboard", action="store_true",
                    help=f"also write {DASHBOARD_OUT.relative_to(C.ROOT)} with these runs")
    args = ap.parse_args(argv)
    records = AttemptStore(args.store).load(lambda r: r.protocol == C.PROTOCOL)
    report = summarise(records)
    C.REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print_boards(report)
    print(f"written to {args.out}")
    if args.dashboard:
        write_dashboard(records, C.load())


if __name__ == "__main__":
    main()
