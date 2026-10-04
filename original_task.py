"""The original DiscoverPhysics task, for the /live replay: no hypothesis, find the law.

A model gets a world with a hidden force law and must find the whole law from scratch, as in
the vendor benchmark (16 rounds, noise σ 0.075, free mid-round MSE fits), then submit it as
Python. The oracle scores it on hidden noise-free test cases (pass: normalised MSE < 0.1).

    ENABLE_LIVE=1 DM_MAX_USD=8 uv run python original_task.py run --world ether --seed 1
    uv run python original_task.py build          # writes web/data/original.json

``run`` goes through ``dm.venues.discoverphysics.run_attempt`` unchanged, without the market
note (the original task), and appends one summary line to ``attempts/original_runs.jsonl``.
``build`` is judge-side: it scores the law the model held after each round (its mid-round MSE
fits, with the parameters the fit found, then its final law) on the same hidden cases.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from dm.store import ATTEMPTS_DIR

ROOT = Path(__file__).resolve().parent

MODEL = "claude-opus-5-5"
STORE = ATTEMPTS_DIR / "original_runs.jsonl"
TRANSCRIPTS = ATTEMPTS_DIR / "transcripts" / "original"
OUT = ROOT / "web" / "data" / "original.json"
TEST_SEED = 0
LABELS = {"claude-opus-5-5": "Claude Opus 5.5"}


def run(world: str, seed: int, max_rounds: int) -> dict:
    from dm.venues.discoverphysics import run_attempt
    from dm.wallet import Wallet
    from poc.bench import load_env

    load_env()
    attempt = run_attempt(MODEL, world, seed, Wallet(MODEL, 10_000.0), 1.0, market_aware=False,
                          max_rounds=max_rounds, transcript_dir=TRANSCRIPTS)
    row = {"solver": MODEL, "world": world, "seed": seed, "rounds": attempt.rounds,
           "experiments": attempt.experiments, "stated_p": attempt.stated_p_success,
           "llm_usage": attempt.llm_usage, "transcript": attempt.transcript_path,
           "submitted": attempt.submitted_law is not None}
    STORE.parent.mkdir(parents=True, exist_ok=True)
    with STORE.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def _with_params(source: str, params: dict) -> str:
    """A mid-round fit's law with the parameters the fit found baked in."""
    if not params:
        return source
    return (source + "\n\n_fitted = " + json.dumps(params) + "\n_law = discovered_law\n"
            "def discovered_law(*args, **kwargs):  # any world's signature\n"
            "    return _law(*args, **{**_fitted, **kwargs})\n")


def _nmse(prereg, source: str | None) -> float | None:
    from dm.oracle import score

    v = score(prereg, source)
    return None if v["normalised_mse"] is None else round(float(v["normalised_mse"]), 6)


CASE = 0  # the hidden test case drawn on the page


def _truth(prereg) -> dict:
    """The drawn hidden case without noise: every particle's path, and which are probes."""
    from dm.oracle._worker import PROBE_SLICES
    from dm.oracle.cases import build_world

    ex = build_world(prereg.world)["executor"]
    with ex.noise_disabled():
        gt = ex.run([prereg.test_cases[CASE]])[0]
    pos = np.asarray(gt["positions"], float)                       # (T, N, 2)
    sl = PROBE_SLICES.get(prereg.world, slice(None))
    probes = list(range(pos.shape[1]))[sl]
    return {"times": prereg.test_cases[CASE]["measurement_times"], "positions": _r(pos),
            "probes": probes}


def _predicted(prereg, source: str) -> list | None:
    """The drawn case as the law predicts it, run in the oracle's sandbox."""
    from dm.oracle._worker import _attrs, _run_sandbox, _zeroed
    from dm.oracle.cases import build_world

    ex = build_world(prereg.world)["executor"]
    case = prereg.test_cases[CASE]
    with ex.noise_disabled():
        gt = ex.run([case])[0]
    child = _run_sandbox({"world": prereg.world, "cases": [case], "law_source": source,
                          "training": [], "outputs": [_zeroed(gt)], "full_outputs": None,
                          "attrs": _attrs(ex)}, 60.0)
    preds = child.get("preds") or []
    try:
        p = np.asarray(preds[0], float).reshape(np.asarray(gt["positions"]).shape)
    except (IndexError, TypeError, ValueError):
        return None
    return _r(p) if np.all(np.isfinite(p)) else None


NO_FORCE_LAW = (  # the floor: every particle keeps its starting velocity
    "def discovered_law(positions, velocities, masses, duration):\n"
    "    import numpy as np\n"
    "    return (np.asarray(positions) + np.asarray(velocities) * duration).tolist()\n")
_TAGS = re.compile(r"<(run_experiment|run_mse_fit|final_law|explanation)>.*?</\1>", re.S)


def _thinking(reply: str) -> str:
    """Its words for the round, without the code and JSON blocks."""
    text = re.sub(r"\n{3,}", "\n\n", _TAGS.sub("", reply)).strip()
    return text[:700] + ("…" if len(text) > 700 else "")


def _ara(world: str) -> dict | None:
    """How the eight ARA frontier runs did on this world (CC BY 4.0, ARA Labs)."""
    try:
        import real_data

        ara = real_data.build_real_data()["ara"]
        rows = [a for a in ara.get("attempts") or [] if a.get("world") == world]
    except Exception:
        return None
    return {"runs": len(rows), "passed": [a.get("model") or a.get("solver") for a in rows
                                          if a.get("passed")],
            "credit": ara.get("attribution"), "url": ara.get("attribution_url")}


def _r(x) -> list:
    return np.round(np.asarray(x, float), 3).tolist()


def build(transcript: Path) -> dict:
    from dm.oracle import make_prereg

    t = json.loads(transcript.read_text())
    prereg = make_prereg("discoverphysics", t["world"], TEST_SEED)
    truth = _truth(prereg)
    rounds, held, held_nmse, held_path = [], None, None, None
    for e in t["rounds"]:
        fit = e.get("mse_fit_output") or {}
        law = None
        if e.get("final_law"):
            law = e["final_law"]
        elif e.get("mse_fit_input") and not fit.get("error"):
            law = _with_params(e["mse_fit_input"], fit.get("fitted_params") or {})
        if law is not None:
            held, held_nmse, held_path = law, _nmse(prereg, law), _predicted(prereg, law)
        reply = e.get("llm_reply") or ""
        rounds.append({
            "round": e.get("round"), "action": e.get("action"),
            "experiments": len(e.get("experiment_input") or []),
            "error": e.get("experiment_error"),
            "fit_loss": fit.get("loss_after"), "fit_error": fit.get("error"),
            "new_law": law is not None, "law": held, "nmse": held_nmse, "path": held_path,
            "thinking": _thinking(reply),
        })
    usage = t.get("llm_usage") or {}
    final_nmse = rounds[-1]["nmse"] if rounds and t.get("final_law") else None
    return {
        "agent": LABELS.get(t["solver"], t["solver"]), "world": t["world"], "seed": t["seed"],
        "mission": t["mission"], "threshold": prereg.threshold, "max_rounds": t["terms"]["max_rounds"],
        "truth": truth, "no_force_nmse": _nmse(prereg, NO_FORCE_LAW), "ara": _ara(t["world"]),
        "rounds": rounds, "final_law": t.get("final_law"), "explanation": t.get("explanation"),
        "final_nmse": final_nmse, "passed": final_nmse is not None and final_nmse < prereg.threshold,
        "stated_p": (t.get("stated_p") or {}).get("reply"), "usd": usage.get("usd"),
        "experiments": sum(r["experiments"] for r in rounds),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--world", default="ether")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--max-rounds", type=int, default=16)
    b = sub.add_parser("build")
    b.add_argument("--transcript", help="default: the last run in attempts/original_runs.jsonl")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        row = run(args.world, args.seed, args.max_rounds)
        print(f"{row['world']} seed {row['seed']}: {row['rounds']} rounds, "
              f"{row['experiments']} experiments, ${(row['llm_usage'] or {}).get('usd', 0):.2f}")
        return
    path = args.transcript
    if path is None:
        rows = [json.loads(x) for x in STORE.read_text().splitlines() if x.strip()]
        path = rows[-1]["transcript"]
    data = build(ROOT / path)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False))
    print(f"{data['world']}: final nmse {data['final_nmse']} passed {data['passed']} -> {OUT}")


if __name__ == "__main__":
    main()
