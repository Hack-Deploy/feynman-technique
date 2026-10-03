"""Calibration and cost report over attempts/poc_dp_bench.jsonl.

    uv run python -m poc.report               # JSON summary + table
    uv run python -m poc.report --dashboard   # also output/poc_dashboard.html
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from statistics import mean

from dm.store import AttemptStore
from poc import config as C

BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


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


def summarise(records: list) -> dict:
    by_model = defaultdict(list)
    for r in records:
        by_model[r.solver].append(r)
    models = {}
    for model, rs in by_model.items():
        outcomes = defaultdict(int)
        for r in rs:
            outcomes[r.extra.get("outcome")] += 1
        verdict_runs = [r for r in rs if r.extra.get("outcome") == "verdict"]
        final = [(r.extra["final_p"], r.passed) for r in verdict_runs
                 if r.extra.get("final_p") is not None]
        # Bid-time estimate: every run that went ahead (walking away has no outcome to score).
        first = [(r.extra["first_p"], r.passed) for r in rs
                 if r.extra.get("outcome") != "walked_away" and r.extra.get("first_p") is not None]
        passed = sum(r.passed for r in rs)
        spent = sum(r.lab_cost for r in rs)
        paid = sum(r.extra.get("prize_paid", 0.0) for r in rs)
        models[model] = {
            "runs": len(rs),
            "outcomes": dict(outcomes),
            "correct_verdicts": passed,
            "clear_answer_rate": round(passed / len(rs), 4),
            "brier_final_p": brier(final),
            "brier_bid_p": brier(first),
            "mean_final_p": _mean([p for p, _ in final]),
            "reliability_final_p": reliability(final),
            "mean_experiments": _mean([r.experiments for r in rs]),
            "mean_rounds": _mean([r.rounds for r in rs]),
            "mean_spent": _mean([r.lab_cost for r in rs]),
            "cost_per_correct": round(spent / passed, 2) if passed else None,
            "prize_paid": paid,
            "spent": spent,
            "profit": round(paid - spent, 2),
        }
    by_seen = defaultdict(list)
    for r in records:
        n = len(r.extra.get("ledger_seen") or [])
        by_seen["0" if n == 0 else "1-2" if n <= 2 else "3+"].append(r)
    by_hyp = defaultdict(list)
    for r in records:
        by_hyp[r.extra.get("hypothesis_id")].append(r)
    return {
        "models": models,
        "by_failed_runs_seen": {k: {"runs": len(v), "clear_answer_rate": _mean([float(r.passed) for r in v]),
                                    "mean_spent": _mean([r.lab_cost for r in v])}
                                for k, v in sorted(by_seen.items())},
        "hypotheses": {k: {"runs": len(v), "correct": sum(r.passed for r in v),
                           "walked_away": sum(r.extra.get("outcome") == "walked_away" for r in v),
                           "mean_spent": _mean([r.lab_cost for r in v])}
                       for k, v in by_hyp.items()},
    }


DASHBOARD_TEMPLATE = C.ROOT / "poc" / "dashboard.html"
DASHBOARD_OUT = C.ROOT / "output" / "poc_dashboard.html"
_PLACEHOLDER = '<script id="runs-data" type="application/json">null</script>'


def dashboard_data(records: list, cfg: C.Config) -> dict:
    """Runs in the shape poc/dashboard.html reads."""
    runs = []
    for r in records:
        x = r.extra
        log = x.get("round_log") or []
        if r.passed:
            result = "correct"
        elif x.get("outcome") == "verdict":
            result = "inconclusive" if r.verdict.get("agent_verdict") == "inconclusive" else "wrong"
        else:
            result = None
        runs.append({
            "hypothesis_id": x.get("hypothesis_id"), "model": r.solver, "seed": r.seed,
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
    print(f"{'model':24s} {'runs':>4s} {'correct':>7s} {'brier':>6s} {'spent':>8s} {'profit':>8s}")
    for m, s in report["models"].items():
        print(f"{m:24s} {s['runs']:4d} {s['correct_verdicts']:7d} {s['brier_final_p']!s:>6s} "
              f"{s['spent']:8g} {s['profit']:8g}")
    print(f"written to {args.out}")
    if args.dashboard:
        write_dashboard(records, C.load())


if __name__ == "__main__":
    main()
