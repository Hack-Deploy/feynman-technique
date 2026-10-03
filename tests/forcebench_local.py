"""Local, non-oracle scoring and grid runner for ForceBench development."""

from __future__ import annotations

import ast
import json
import math
import multiprocessing
from pathlib import Path
import time
from typing import Iterable

from scienceagent.evaluator import Evaluator
from scienceagent.worlds import get_world

from dm.venues.forcebench import WORLDS, SubmittedAttempt, run_attempt
from dm.wallet import Wallet

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "output" / "forcebench_grid.json"
SOLVERS = ("bayes_lite", "random_menu")
SEEDS = tuple(range(5))

BASELINE_INV_R_LAW = """import math

def fit_parameters():
    return {"k": {"init": 0.1, "bounds": [-5.0, 5.0]}}

def discovered_law(pos1, pos2, p1, p2, velocity2, duration, **params):
    k = params.get("k", 0.1)
    eps2 = 0.05 ** 2
    n = max(1, int(round(duration / 0.01)))
    h = duration / n
    x, y = float(pos2[0]), float(pos2[1])
    vx, vy = float(velocity2[0]), float(velocity2[1])
    g = p1 / p2

    def acc(x, y):
        r2 = x * x + y * y + eps2
        s = -k * g / r2
        return s * x, s * y

    ax, ay = acc(x, y)
    for _ in range(n):
        vx += 0.5 * h * ax
        vy += 0.5 * h * ay
        x += h * vx
        y += h * vy
        ax, ay = acc(x, y)
        vx += 0.5 * h * ax
        vy += 0.5 * h * ay
    return [x, y], [vx, vy]
"""


def vendor_world_vars() -> dict[str, float]:
    """Read benchmark normalisers without importing its command-line script."""
    path = ROOT / "vendor" / "discovery-agents" / "scripts" / "run_benchmark.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "_WORLD_VARS":
                return {key: float(value) for key, value in ast.literal_eval(node.value).items()}
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_WORLD_VARS"
            for target in node.targets
        ):
            return {key: float(value) for key, value in ast.literal_eval(node.value).items()}
    raise ValueError(f"_WORLD_VARS not found in {path}")


def _evaluate(
    law_source: str,
    world: str,
    training: list | None = None,
) -> dict:
    try:
        executor = get_world(
            world, engine="nbody", noise_std=0.03, noise_seed=0
        )["executor"]
        evaluator = Evaluator(executor)
        kwargs = {"verbose": False}
        if training is not None:
            kwargs["training_trajectories"] = training
        result = evaluator.evaluate(law_source, **kwargs)
        mean_pos_error = float(result["mean_pos_error"])
        normaliser = vendor_world_vars()[world]
        nmse = mean_pos_error / normaliser
        if not math.isfinite(nmse):
            return {
                "nmse": float("inf"),
                "passed": False,
                "mean_pos_error": mean_pos_error,
                "reason": "non-finite normalised MSE",
                "fit": result.get("fit"),
            }
        return {
            "nmse": nmse,
            "passed": nmse < 0.1,
            "mean_pos_error": mean_pos_error,
            "fit": result.get("fit"),
        }
    except Exception as error:
        return {
            "nmse": float("inf"),
            "passed": False,
            "mean_pos_error": float("inf"),
            "reason": f"{type(error).__name__}: {error}",
            "fit": None,
        }


def score_local(submitted: SubmittedAttempt, world: str) -> dict:
    """Score a submitted law against the vendor's default cases locally."""
    if submitted.submitted_law is None:
        return {"nmse": float("inf"), "passed": False, "reason": "attempt has no law"}
    return _evaluate(submitted.submitted_law, world)


def score_baseline(submitted: SubmittedAttempt, world: str) -> dict:
    """Fit the fixed 1/r baseline to this attempt's paid training data and score it."""
    result = _evaluate(BASELINE_INV_R_LAW, world, submitted.training)
    fit = result.get("fit") or {}
    fitted = fit.get("fitted_params") or {}
    result["fitted_k"] = fitted.get("k")
    return result


def identify_model(world: str, top_model: dict) -> tuple[bool, str | None]:
    """Classify a fitted top model against the known world, for local reporting only."""
    family = top_model.get("family")
    role = top_model.get("role")
    params = [float(value) for value in top_model.get("params", [])]
    k = params[0] if params else None
    positive = k is not None and k > 0
    negative = k is not None and k < 0
    if world == "gravity":
        identified = (
            role == "ratio"
            and positive
            and (
                family == "log"
                or (
                    family == "power"
                    and len(params) > 1
                    and abs(params[1] - 1.0) < 0.1
                )
            )
        )
    elif world == "fractional":
        identified = (
            family == "power"
            and role == "ratio"
            and positive
            and len(params) > 1
            and abs(params[1] - 2.0) < 0.1
        )
    elif world == "yukawa":
        identified = family == "yukawa" and role == "ratio" and positive
    elif world == "oscillator":
        identified = family == "timemod" and role == "ratio"
    elif world == "coulomb_easy":
        identified = (
            family == "power"
            and role == "p1"
            and negative
            and len(params) > 1
            and abs(params[1] - 2.0) < 0.1
        )
    elif world == "extra_dimensions":
        if family == "crossover" and role == "ratio":
            return False, "closest: crossover/ratio"
        return False, None
    else:
        return False, None
    return bool(identified), "exact" if identified else None


def _top_model(attempt: SubmittedAttempt) -> dict:
    transcript = Path(attempt.transcript_path or "")
    if not transcript.is_absolute():
        transcript = ROOT / transcript
    if transcript.exists():
        return json.loads(transcript.read_text())["submission"]["top_model"] or {}
    return {}


def _run_grid_cell(cell: tuple[str, str, int, float]) -> dict:
    solver_name, world, seed, price = cell
    wallet = Wallet(owner=f"{solver_name}-{world}-{seed}", balance=100.0)
    initial_total = wallet.balance + wallet.lab_revenue
    run_started = time.perf_counter()
    attempt = run_attempt(solver_name, world, seed, wallet, price)
    run_seconds = time.perf_counter() - run_started
    score_started = time.perf_counter()
    score = score_local(attempt, world)
    score_seconds = time.perf_counter() - score_started
    baseline_started = time.perf_counter()
    baseline = score_baseline(attempt, world)
    baseline_seconds = time.perf_counter() - baseline_started
    top_model = _top_model(attempt)
    identified, identification = identify_model(world, top_model)
    return {
        "solver": solver_name,
        "world": world,
        "seed": seed,
        "experiments": attempt.experiments,
        "rounds": attempt.rounds,
        "lab_cost": attempt.lab_cost,
        "stated_p": attempt.stated_p_success,
        "top_model": top_model,
        "identified": identified,
        "identification": identification,
        "nmse": score["nmse"],
        "passed": score["passed"],
        "score_reason": score.get("reason"),
        "baseline": baseline,
        "run_seconds": run_seconds,
        "score_seconds": score_seconds,
        "baseline_seconds": baseline_seconds,
        "wallet_conserved": abs(
            wallet.balance + wallet.lab_revenue - initial_total
        ) < 1e-12,
        "charged_events": sum(
            event["type"] == "experiment_charged" for event in wallet.events
        ),
        "attempt": attempt,
    }


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else float("nan")


def markdown_table(results: list[dict]) -> str:
    rows = []
    paper_pass = {
        "gravity": 100,
        "yukawa": 100,
        "coulomb_easy": 56,
        "oscillator": 100,
        "fractional": 100,
        "extra_dimensions": 100,
    }
    menu_llm = {
        "gravity": 22,
        "yukawa": 33,
        "coulomb_easy": 22,
        "oscillator": 33,
        "fractional": 56,
        "extra_dimensions": 22,
    }
    for world in WORLDS:
        row = [world, f"{paper_pass[world]}%", f"{menu_llm[world]}%"]
        for solver_name in SOLVERS:
            selected = [
                item
                for item in results
                if item["world"] == world and item["solver"] == solver_name
            ]
            identified_count = sum(item["identified"] for item in selected)
            identified = f"{identified_count}/{len(selected)}"
            if world == "extra_dimensions":
                closest_count = sum(
                    item["identification"] == "closest: crossover/ratio"
                    for item in selected
                )
                identified += (
                    f" (closest crossover/ratio: {closest_count}/{len(selected)})"
                )
            row.extend(
                [
                    f"{100 * _mean(item['passed'] for item in selected):.0f}%",
                    identified,
                    f"{_mean(item['experiments'] for item in selected):.2f}",
                    f"{_mean(item['stated_p'] for item in selected):.3f}",
                    f"{100 * _mean(item['baseline']['passed'] for item in selected):.0f}%",
                    f"{_mean(item['baseline']['nmse'] for item in selected):.4f}",
                ]
            )
        rows.append(row)
    headers = [
        "world",
        "MDA Table 8",
        "MDA menu LLM",
        "bayes_lite pass %",
        "bayes_lite identified seeds",
        "bayes_lite mean exp.",
        "bayes_lite mean stated p",
        "1/r+fit pass % (bayes_lite data)",
        "mean baseline nMSE (bayes_lite)",
        "random_menu pass %",
        "random_menu identified seeds",
        "random_menu mean exp.",
        "random_menu mean stated p",
        "1/r+fit pass % (random_menu data)",
        "mean baseline nMSE (random_menu)",
    ]
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(row) + " |" for row in rows),
        ]
    )


def run_grid(
    *,
    solvers: tuple[str, ...] = SOLVERS,
    worlds: tuple[str, ...] = WORLDS,
    seeds: tuple[int, ...] = SEEDS,
    price: float = 1.0,
    output_path: Path | str | None = None,
    processes: int | None = None,
) -> list[dict]:
    """Run independent grid cells in a spawn-based process pool."""
    cells = [
        (solver_name, world, seed, price)
        for solver_name in solvers
        for world in worlds
        for seed in seeds
    ]
    if len(cells) == 1:
        results = [_run_grid_cell(cells[0])]
    else:
        context = multiprocessing.get_context("spawn")
        worker_count = processes or min(6, len(cells))
        with context.Pool(processes=worker_count) as pool:
            results = pool.map(_run_grid_cell, cells)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = [
            {key: value for key, value in record.items() if key != "attempt"}
            for record in results
        ]
        identification_summary = []
        for solver_name in solvers:
            for world in worlds:
                selected = [
                    item
                    for item in serialized
                    if item["solver"] == solver_name and item["world"] == world
                ]
                identification_summary.append(
                    {
                        "solver": solver_name,
                        "world": world,
                        "identified": sum(item["identified"] for item in selected),
                        "seeds": len(selected),
                        "closest_crossover_ratio": sum(
                            item["identification"] == "closest: crossover/ratio"
                            for item in selected
                        ),
                    }
                )
        payload = {
            "results": serialized,
            "identification_summary": identification_summary,
        }
        path.write_text(json.dumps(payload, sort_keys=True, indent=1) + "\n")
    return results


def main() -> None:
    results = run_grid(output_path=OUTPUT_PATH)
    print(markdown_table(results))
    print("\nAttempt wall times (run / solver score / 1/r baseline score, seconds):")
    for item in results:
        print(
            f"{item['solver']} {item['world']} seed={item['seed']}: "
            f"{item['run_seconds']:.3f} / {item['score_seconds']:.3f} / "
            f"{item['baseline_seconds']:.3f}"
        )


if __name__ == "__main__":
    main()
