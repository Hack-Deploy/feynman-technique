"""Import the published ARA DiscoverPhysics runs as ``AttemptRecord``s.

Data: ARA Labs (AgentNativeResearchLab), CC BY 4.0,
https://huggingface.co/AgentNativeResearchLab. One run per (model, world): seed 0,
≤16 rounds, relative noise σ = 0.075·√Var(world), run through an agentic coding
harness (protocol ``ara_harness``), scored on the vendor's public default cases.

Files are cached under ``attempts/cache/ara/<model>/`` (git-ignored); once the cache
is populated the importer works offline.

    uv run python -m dm.importers.ara            # download (if needed) + import
    uv run python -m dm.importers.ara --offline  # cache only
"""

from __future__ import annotations

import argparse
import json
import math
import re
import uuid
from pathlib import Path
from typing import Any

from dm.store import ATTEMPTS_DIR, AttemptStore
from dm.types import AttemptRecord

MODELS = ["fable", "gpt5.6-sol", "gpt5.5", "kimi-k2.7", "gemini3.1-pro",
          "opus4.8-max", "glm5.2", "grok4.5"]
DATASET = "AgentNativeResearchLab/discoverphysics-{model}-ara"
HF_URL = "https://huggingface.co/datasets/" + DATASET
TREE_URL = "https://huggingface.co/api/datasets/" + DATASET + "/tree/main"
WORLD_FILES = ("meta.json", "result.json", "episode.json", "posthoc_salvage.json")

CACHE_DIR = ATTEMPTS_DIR / "cache" / "ara"
STORE_PATH = ATTEMPTS_DIR / "ara.jsonl"

PROTOCOL = "ara_harness"
VENUE = "discoverphysics"
PASS_THRESHOLD = 0.1          # numeric-only rule: nMSE < 0.1
ARA_EXPL_THRESHOLD = 0.75     # ARA's own rule also needs explanation ≥ 0.75
ATTRIBUTION = "ARA Labs (AgentNativeResearchLab), CC BY 4.0"
ATTRIBUTION_URL = "https://huggingface.co/AgentNativeResearchLab"
COST_NOTE = ("cost/total_cost_usd covers only the judge call for subscription "
             "harnesses (the solver model itself ran on a subscription)")

# What meta.total_cost_usd covers, per model, from each SCOREBOARD.md header.
COST_COVERAGE: dict[str, tuple[str, str]] = {
    "fable": ("full_api", "API-billed run (CC+ARA bridge); same basis as opus4.8-max"),
    "opus4.8-max": ("full_api", "本组为唯一美元计费组: the only dollar-billed sibling group"),
    "gpt5.5": ("judge_only", "cost 仅 judge 开销, gpt-5.5 本体走订阅"),
    "gpt5.6-sol": ("judge_only", "cost 列仅含 judge 开销 — codex CLI 不上报 API 费用"),
    "kimi-k2.7": ("judge_only", "subscription CLI (~$0.05/world); opus4.8-max is the only "
                                "dollar-billed group"),
    "gemini3.1-pro": ("judge_only", "subscription CLI (~$0.05/world); agy CLI reports no "
                                    "token usage, tokens are estimates"),
    "glm5.2": ("proxy_billing", "total_cost_usd 为代理侧计费, 非官方 API 口径, 仅供参考"),
    "grok4.5": ("none", "grok CLI 无美元计费 (订阅代理); judge cost in meta.judge_cost_usd"),
}

# Vendor normalising variances, copied from
# vendor/discovery-agents/scripts/run_benchmark.py::_WORLD_VARS (that module exits
# on import without PyYAML and parses argv, so it is not imported here).
WORLD_VARS: dict[str, float] = {
    "circle": 6.596, "coulomb_easy": 11.465, "dark_matter": 63.303,
    "ether": 21.259, "extra_dimensions": 4.248, "fractional": 5.721,
    "gravity": 4.283, "hubble": 41.189, "oscillator": 6.332,
    "three_species": 28.717, "yukawa": 5.677,
}


class AraUnreachable(RuntimeError):
    pass


# --------------------------------------------------------------- download

def _get(session, url: str, timeout: float, retries: int = 4):
    """GET with a timeout; backs off on HTTP 429 (Hugging Face rate limit)."""
    import time

    import requests
    try:
        for attempt in range(retries + 1):
            r = session.get(url, timeout=timeout)
            if r.status_code != 429 or attempt == retries:
                return r
            # HF sends `ratelimit: "resolvers";r=0;t=<seconds until the window resets>`
            m = re.search(r"t=(\d+)", r.headers.get("ratelimit", ""))
            wait = float(r.headers.get("Retry-After") or (m.group(1) if m else 0) or 2 ** attempt)
            time.sleep(min(wait + 1, 310))
        return r
    except requests.RequestException as e:
        raise AraUnreachable(
            f"Hugging Face unreachable ({e.__class__.__name__}: {url}). Download "
            f"SCOREBOARD.md and <world>/{{meta,result,episode}}.json for each model "
            f"from {HF_URL.format(model='<model>')} into {CACHE_DIR}/<model>/ and "
            f"rerun with --offline.") from e


def download(models: list[str] | None = None, cache_dir: Path = CACHE_DIR,
             timeout: float = 30.0, force: bool = False) -> dict[str, list[str]]:
    """Fetch every model's scoreboard and per-world files into the cache.

    Files already cached are not fetched again (unless ``force``). Returns the
    files present per model. A model whose scoreboard 404s is skipped.
    """
    import requests
    session = requests.Session()
    have: dict[str, list[str]] = {}
    for model in models or MODELS:
        mdir = cache_dir / model
        files: list[str] = []

        def fetch(rel: str, url: str) -> bool:
            dest = mdir / rel
            if dest.exists() and not force:
                files.append(rel)
                return True
            r = _get(session, url, timeout)
            if r.status_code == 404:
                return False
            r.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(r.content)
            files.append(rel)
            return True

        base = HF_URL.format(model=model) + "/resolve/main/"
        if not fetch("SCOREBOARD.md", base + "SCOREBOARD.md"):
            continue
        fetch("tree.json", TREE_URL.format(model=model))
        tree = json.loads((mdir / "tree.json").read_text()) if (mdir / "tree.json").exists() else []
        worlds = sorted(e["path"] for e in tree if e.get("type") == "directory")
        for world in worlds:
            fetch(f"{world}/tree.json", TREE_URL.format(model=model) + "/" + world)
            wtree = json.loads((mdir / world / "tree.json").read_text())
            present = {Path(e["path"]).name for e in wtree if e.get("type") == "file"}
            for name in WORLD_FILES:
                if name in present:
                    fetch(f"{world}/{name}", base + f"{world}/{name}")
        have[model] = files
    return have


# --------------------------------------------------------------- parsing

def parse_scoreboard(text: str) -> dict[str, dict[str, str]]:
    """Rows of the first markdown table with a ``world`` column, keyed by world.

    Columns are matched by header name (lower-cased, stripped), never position.
    """
    rows: dict[str, dict[str, str]] = {}
    header: list[str] | None = None
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("|"):
            if header is not None and rows:
                break
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if header is None:
            if "world" in [c.lower() for c in cells]:
                header = [c.lower() for c in cells]
            continue
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
            continue
        row = dict(zip(header, cells))
        if row.get("world"):
            rows[row["world"]] = row
    return rows


_SUFFIX = {"k": 1e3, "m": 1e6, "b": 1e9}


def parse_number(s: str | None) -> float | None:
    """'469', '7.03e-05', '$0.07', '169k', '6.1M (5.7M)', '30k(est)', 'inf' → float.

    Returns None for blanks/dashes; non-finite strings return inf/nan."""
    if s is None:
        return None
    t = s.strip().replace("$", "").replace(",", "")
    m = re.match(r"^([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([kKmMbB])?", t)
    if m:
        v = float(m.group(1))
        return v * _SUFFIX[m.group(2).lower()] if m.group(2) else v
    low = t.lower()
    if low in {"inf", "+inf", "infinity", "∞"}:
        return math.inf
    if low in {"nan", "-inf"}:
        return math.nan if low == "nan" else -math.inf
    return None


def finite_or_none(x: Any) -> float | None:
    ok = isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
    return x if ok else None


def sig_digits(s: str) -> int:
    """Significant digits shown in a numeric string ('0.0105' → 3, '469' → 3, '0.15' → 2)."""
    mant = re.split(r"[eE]", s.strip())[0].lstrip("+-").replace(".", "").lstrip("0")
    return max(len(mant), 1)


def round_sig(x: float, n: int) -> float:
    return 0.0 if x == 0 else round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))


def _json_safe(obj: Any) -> Any:
    """Replace non-finite floats with "inf"/"-inf"/"nan" so records stay strict JSON.

    Failed ARA runs can report an infinite position error; the string keeps that
    information without breaking AttemptRecord's no-NaN/Infinity rule."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return "nan" if math.isnan(obj) else ("inf" if obj > 0 else "-inf")
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def nmse_agrees(shown: str, computed: float, max_sig: int = 3) -> bool:
    """Scoreboard value vs mean_pos_error/Var(world), compared at
    min(3, digits shown) significant figures."""
    v = parse_number(shown)
    if v is None or not math.isfinite(v) or not math.isfinite(computed):
        return False
    n = min(max_sig, sig_digits(shown))
    return math.isclose(round_sig(v, n), round_sig(computed, n), rel_tol=1e-9)


def count_experiments(episode: dict) -> int:
    return sum(len(r["experiment_input"]) for r in episode.get("rounds", [])
               if isinstance(r.get("experiment_input"), list))


def attempt_id(model: str, world: str, protocol: str = PROTOCOL) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"ara:{model}:{world}:{protocol}"))


def _load_json(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


# --------------------------------------------------------------- records

def build_record(model: str, world: str, row: dict[str, str],
                 mdir: Path) -> tuple[AttemptRecord, list[str]]:
    """One record from a scoreboard row and the world's files (if cached).

    Returns the record and a list of human-readable data issues."""
    issues: list[str] = []
    wdir = mdir / world
    meta = _load_json(wdir / "meta.json")
    result = _load_json(wdir / "result.json")
    episode = _load_json(wdir / "episode.json")
    salvage = _load_json(wdir / "posthoc_salvage.json")
    var = WORLD_VARS.get(world)

    # rounds: scoreboard, falling back to meta.json; episode.json for reference.
    r_sb = parse_number(row.get("rounds"))
    r_meta = meta.get("rounds") if meta else None
    r_ep = len(episode["rounds"]) if episode and "rounds" in episode else None
    rounds = int(r_sb) if r_sb is not None else (int(r_meta) if r_meta is not None else 0)
    rounds_sources = {"scoreboard": None if r_sb is None else int(r_sb),
                      "meta": r_meta, "episode": r_ep}
    distinct = {v for v in rounds_sources.values() if v is not None}
    if len(distinct) > 1:
        issues.append(f"{model}/{world}: rounds disagree {rounds_sources} "
                      f"(used scoreboard={rounds})")

    experiments = count_experiments(episode) if episode else 0
    exp_source = "episode.json" if episode else "unavailable: no episode.json, recorded as 0"

    # nMSE: scoreboard value; cross-check with mean_pos_error / Var(world).
    # Post-hoc salvaged runs (fable ether/extra_dimensions) are scored on the
    # salvaged law; their meta.json keeps the original (failed) mean_pos_error.
    nmse_raw = row.get("norm_mse")
    nmse_val = parse_number(nmse_raw)
    meta_mpe = meta.get("mean_pos_error") if meta else None
    salvage_mpe = salvage.get("mean_pos_error") if salvage else None
    mpe, mpe_from = ((salvage_mpe, "posthoc_salvage.json") if salvage_mpe is not None
                     else (meta_mpe, "meta.json"))
    computed = (mpe / var) if (isinstance(mpe, (int, float)) and var) else None
    nmse_check: dict[str, Any] = {"scoreboard": nmse_raw, "computed": computed,
                                  "mean_pos_error": mpe, "mean_pos_error_from": mpe_from,
                                  "meta_mean_pos_error": meta_mpe, "world_var": var}
    if computed is None:
        nmse_check["agrees"] = None
        issues.append(f"{model}/{world}: nMSE not cross-checkable (no mean_pos_error)")
    elif nmse_val is not None and not math.isfinite(nmse_val) and not math.isfinite(computed):
        nmse_check["agrees"] = True  # both non-finite
    else:
        nmse_check["agrees"] = nmse_agrees(nmse_raw or "", computed)
        if not nmse_check["agrees"]:
            finite = nmse_val is not None and math.isfinite(nmse_val) and nmse_val > 0
            rel = abs(computed - nmse_val) / nmse_val if finite else math.inf
            nmse_check["kind"] = "last_digit" if rel < 0.01 else "systematic"
            if finite and mpe:
                nmse_check["implied_var"] = mpe / nmse_val
            issues.append(f"{model}/{world}: scoreboard nMSE {nmse_raw} vs "
                          f"mean_pos_error/Var = {computed:.4g} ({nmse_check['kind']})")

    finite = nmse_val is not None and math.isfinite(nmse_val)
    if not finite:
        issues.append(f"{model}/{world}: non-finite or missing nMSE {nmse_raw!r} "
                      f"→ normalised_mse=None, passed=False")
    normalised_mse = nmse_val if finite else None
    passed = bool(finite and nmse_val < PASS_THRESHOLD)

    expl = parse_number(row.get("expl"))
    if expl is None and meta:
        expl = meta.get("explanation_score")
    ara_verdict = row.get("verdict")
    ara_passed = (ara_verdict or "").strip().upper() == "PASS"

    # Token keys differ by harness and are kept as published. The gemini (agy) CLI
    # reports no usage, so its est_* counts are estimates.
    tokens = dict((meta or {}).get("tokens") or {})
    llm_usage: dict[str, Any] = {
        **tokens,
        "usd": (meta or {}).get("total_cost_usd"),
        "estimated": any(k.startswith("est_") for k in tokens),
    }
    if meta and meta.get("judge_cost_usd") is not None:
        llm_usage["judge_usd"] = meta["judge_cost_usd"]
    if not meta:
        issues.append(f"{model}/{world}: no meta.json; llm_usage empty")
    coverage, coverage_why = COST_COVERAGE.get(model, ("unknown", ""))
    if rounds > 16:
        issues.append(f"{model}/{world}: {rounds} rounds exceeds the stated 16-round cap")

    src = HF_URL.format(model=model) + "/resolve/main/"
    files = ["SCOREBOARD.md"] + [f"{world}/{n}" for n in WORLD_FILES if (wdir / n).exists()]
    extra: dict[str, Any] = {
        "ara_model": model,
        "ara_verdict": ara_verdict,
        "ara_passed": ara_passed,
        "ara_rule": "norm_MSE < 0.1 AND explanation >= 0.75",
        "numeric_rule": "norm_MSE < 0.1",
        "nmse_raw": nmse_raw,
        "nmse_check": nmse_check,
        "rounds_sources": rounds_sources,
        "rounds_rule": "scoreboard, else meta.json",
        "experiments_source": exp_source,
        "noise_std": meta.get("noise_std") if meta else None,
        "noise_seed": meta.get("noise_seed") if meta else None,
        "noise_rule": "sigma = 0.075 * sqrt(Var(world))",
        "harness": meta.get("harness") if meta else None,
        "harness_model": meta.get("harness_model") if meta else None,
        "judge_model": meta.get("judge_model") if meta else None,
        "cost_coverage": coverage,
        "cost_coverage_source": coverage_why,
        "cost_note": COST_NOTE,
        "scoreboard_row": row,
        "salvage": salvage,
        "attribution": ATTRIBUTION,
        "attribution_url": ATTRIBUTION_URL,
        "dataset_url": HF_URL.format(model=model),
        "source_files": [src + f for f in files],
        "cache_files": [f"attempts/cache/ara/{model}/{f}" for f in files],
    }
    rec = AttemptRecord(
        attempt_id=attempt_id(model, world),
        source="published_replay", protocol=PROTOCOL, venue=VENUE, world=world,
        solver=f"ara:{model}", seed=0, stated_p_success=None,
        rounds=rounds, experiments=experiments, lab_cost=float(rounds) * 1.0,
        llm_usage=_json_safe(llm_usage),
        submitted_law=result.get("law") if result else None,
        verdict={"normalised_mse": normalised_mse, "passed": passed,
                 "prereg_commitment": None, "public_tests": True,
                 "explanation_score": finite_or_none(expl)},
        transcript_path=f"attempts/cache/ara/{model}/{world}/episode.json" if episode else None,
        created_at=(meta or {}).get("archived_at", ""),
        extra=_json_safe(extra),
    )
    return rec, issues


def import_cache(cache_dir: Path = CACHE_DIR,
                 models: list[str] | None = None) -> tuple[list[AttemptRecord], dict]:
    """Records for every scoreboard row in the cache, plus an import report."""
    records: list[AttemptRecord] = []
    report: dict[str, Any] = {"counts": {}, "missing_models": [], "missing_worlds": {},
                              "issues": [], "verdict_disagreements": []}
    for model in models or MODELS:
        mdir = cache_dir / model
        sb = mdir / "SCOREBOARD.md"
        if not sb.exists():
            report["missing_models"].append(model)
            report["counts"][model] = 0
            continue
        text = sb.read_text()
        rows = parse_scoreboard(text)
        for world in sorted(rows):
            rec, issues = build_record(model, world, rows[world], mdir)
            records.append(rec)
            report["issues"].extend(issues)
            if rec.extra["ara_passed"] != rec.passed:
                report["verdict_disagreements"].append(
                    f"{model}/{world}: ARA {rec.extra['ara_verdict']}, numeric-only "
                    f"{'PASS' if rec.passed else 'FAIL'} (nMSE {rec.extra['nmse_raw']}, "
                    f"expl {rec.verdict['explanation_score']})")
        report["counts"][model] = len(rows)
        missing = sorted(set(WORLD_VARS) - set(rows))
        if missing:
            report["missing_worlds"][model] = missing
    return records, report


def write_store(records: list[AttemptRecord], path: Path = STORE_PATH) -> int:
    """Append records whose content differs from the stored version (append-only;
    a re-import with changed data supersedes by attempt_id)."""
    store = AttemptStore(path)
    current = {r.attempt_id: r.to_dict() for r in store.load()}
    new = [r for r in records if current.get(r.attempt_id) != json.loads(
        json.dumps(r.to_dict(), sort_keys=True))]
    return store.append(new) if new else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--offline", action="store_true", help="use the cache only")
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--out", type=Path, default=STORE_PATH)
    ap.add_argument("--report", type=Path, default=None,
                    help="write the import report as JSON here")
    args = ap.parse_args(argv)
    if not args.offline:
        try:
            download(cache_dir=args.cache)
        except AraUnreachable as e:
            print(f"error: {e}")
            return 2
    records, report = import_cache(args.cache)
    n = write_store(records, args.out)
    print(f"Data: {ATTRIBUTION} — {ATTRIBUTION_URL}")
    for model, c in report["counts"].items():
        print(f"  {model:<14} {c:>2} records")
    print(f"{len(records)} records, {n} appended to {args.out}")
    print(f"{len(report['issues'])} data issues, "
          f"{len(report['verdict_disagreements'])} ARA vs numeric-only verdict disagreements")
    for line in report["issues"]:
        print("  issue:", line)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
