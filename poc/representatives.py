"""Build the paired /live replays from an archived Hugging Face snapshot.

Run: uv run python -m poc.representatives
The source preserves the original records. Numeric estimates come only from
recorded fit output or explicit final evidence; missing estimates stay missing.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

from poc import animate, config, truth
from poc.baselines import SUPPORT_GUESS

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "attempts/fixtures/demo/hf_representatives.jsonl"
URL = "https://huggingface.co/datasets/arushisinha98/discovery-market-live/blob/main/attempts/fixtures/live/runs.jsonl"


def build(entry: dict, cfg: config.Config) -> dict:
    record = entry["record"]
    x = record["extra"]
    hyp = cfg.hypothesis(x["hypothesis_id"])
    rounds = []
    estimates = {}
    for rd in entry["rounds"]:
        if rd.get("mse_fit"):
            fit = ast.literal_eval(rd["mse_fit"]) if isinstance(rd["mse_fit"], str) else rd["mse_fit"]
            params = fit.get("fitted_params") or {}
            if "n" in params and "k" in params:
                estimates = {"n": {"value": params["n"]},
                             "a3": {"value": params["k"] / 3 ** params["n"]}}
        if rd["action"] == "verdict" and entry["key"]["model"] == "claude-sonnet-5-5":
            # Explicitly stated in this run's final evidence, not a fitted value.
            assert "n ≈ 1" in rd["evidence"] and "about 0.15" in rd["evidence"]
            estimates = {"n": {"value": 1.0, "sigma": 0.15}}
        rounds.append({"round": rd["round"], "action": rd["action"],
                       "assessment": rd["assessment"], "p_success": rd["p_success"],
                       "experiments": rd["n_experiments"], "spent_so_far": rd["spent_so_far"],
                       "estimates": estimates.copy(),
                       "points": [p for run in x["runs"] if run["round"] == rd["round"]
                                  if (p := animate._reading(run)) is not None]})
    passed = record["verdict"]["passed"]
    return {"id": record["attempt_id"], "agent": entry["model_label"],
            "model": entry["key"]["model"], "world": record["world"],
            "resolution_criteria": x["resolution_criteria"], "round_fee": x["round_fee"],
            "hypothesis_id": hyp.id, "hypothesis": x["hypothesis"], "seed": record["seed"],
            "answer": record["verdict"]["answer"], "verdict": x["agent_verdict"],
            "outcome": "confirmed" if passed else "false_claim", "judge": "answer_key",
            "prize": x["prize"], "prize_paid": x["prize_paid"],
            "expected": {k: SUPPORT_GUESS[hyp.id][k] for k in ("n", "a3")},
            "true": truth.true_values(hyp), "tolerance": {}, "rounds": rounds,
            "max_rounds": x["max_rounds"], "spent": record["lab_cost"], "usd": entry["usd"],
            "profit": x["prize_paid"] - record["lab_cost"] - x["record_fee"],
            "source_url": URL,
            "caveat": "The dataset scores the final verdict against an answer key. "
                      "This run reports uncertainty about 0.15, above the requested 0.1 target."
                      if passed else "The recorded fit stayed near its initial n = 2. "
                      "A lower fitting loss did not make the final verdict correct."}


def main() -> None:
    cfg = config.load()
    runs = [build(json.loads(line), cfg) for line in SOURCE.read_text().splitlines()]
    (ROOT / "web/data/learning.json").write_text(json.dumps({"runs": runs}, separators=(",", ":"), allow_nan=False))
    print("Built successful and failed gravity replays from the archived dataset records.")


if __name__ == "__main__":
    main()
