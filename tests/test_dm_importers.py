"""Phase 4: ARA and vendor-run importers. Offline only (attempts/fixtures/ara_sample)."""

import json
import math
import shutil

import pytest
import requests

from dm.importers import ara, vendor_runs
from dm.store import ATTEMPTS_DIR, FIXTURES_DIR, AttemptStore

SAMPLE = FIXTURES_DIR / "ara_sample"
SAMPLE_MODELS = ["fable", "gpt5.5", "opus4.8-max"]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("tests must not use the network")
    monkeypatch.setattr(requests.Session, "get", boom)
    monkeypatch.setattr(requests, "get", boom)


@pytest.fixture(scope="module")
def imported():
    recs, report = ara.import_cache(SAMPLE, SAMPLE_MODELS)
    return {(r.solver, r.world): r for r in recs}, report


def _rec(imported, model, world):
    return imported[0][(f"ara:{model}", world)]


# ------------------------------------------------------------- parsing

def test_scoreboard_parsed_by_header_name():
    text = ("intro\n| verdict | extra | world | norm_MSE |\n|---|---|---|---|\n"
            "| PASS | x | gravity | 0.001 |\n| FAIL | y | yukawa | inf |\n\ntrailer | no\n")
    rows = ara.parse_scoreboard(text)
    assert rows["gravity"] == {"verdict": "PASS", "extra": "x", "world": "gravity",
                               "norm_mse": "0.001"}
    assert rows["yukawa"]["norm_mse"] == "inf"
    assert "rounds" not in rows["gravity"]  # missing columns tolerated


def test_sample_scoreboards_have_eleven_worlds():
    for m in SAMPLE_MODELS:
        rows = ara.parse_scoreboard((SAMPLE / m / "SCOREBOARD.md").read_text())
        assert sorted(rows) == sorted(ara.WORLD_VARS)


@pytest.mark.parametrize("s,v", [("469", 469.0), ("7.03e-05", 7.03e-05), ("$0.07", 0.07),
                                 ("169k", 169e3), ("6.1M (5.7M)", 6.1e6),
                                 ("30k(est)", 30e3), ("1.03e+03", 1030.0), ("", None),
                                 ("—", None)])
def test_parse_number(s, v):
    assert ara.parse_number(s) == (pytest.approx(v) if v is not None else None)


def test_parse_number_non_finite():
    assert ara.parse_number("inf") == math.inf
    assert math.isnan(ara.parse_number("nan"))


def test_nmse_agreement_rule():
    assert ara.nmse_agrees("0.000115", 0.000492266866 / 4.283)
    assert ara.nmse_agrees("0.15", 0.1504)          # 2 digits shown
    assert not ara.nmse_agrees("0.104", 0.03833)
    assert not ara.nmse_agrees("inf", 1.0)


# ------------------------------------------------------------- records

def test_counts_and_missing_models():
    recs, report = ara.import_cache(SAMPLE)  # all eight models; sample has three
    assert {m: report["counts"][m] for m in SAMPLE_MODELS} == dict.fromkeys(SAMPLE_MODELS, 11)
    assert sorted(report["missing_models"]) == sorted(set(ara.MODELS) - set(SAMPLE_MODELS))
    assert len(recs) == 33


def test_gpt55_gravity_matches_verified_facts(imported):
    r = _rec(imported, "gpt5.5", "gravity")
    assert (r.source, r.protocol, r.venue, r.solver, r.seed) == (
        "published_replay", "ara_harness", "discoverphysics", "ara:gpt5.5", 0)
    assert r.stated_p_success is None
    assert (r.rounds, r.experiments, r.lab_cost) == (14, 36, 14.0)
    assert r.verdict == {"normalised_mse": 0.000115, "passed": True,
                         "prereg_commitment": None, "public_tests": True,
                         "explanation_score": 0.9}
    assert r.extra["nmse_check"]["agrees"] is True
    assert r.extra["noise_std"] == pytest.approx(0.155216)
    assert r.extra["ara_verdict"] == "PASS" and r.extra["ara_passed"] is True
    assert r.extra["cost_coverage"] == "judge_only"
    assert r.llm_usage["output_tokens"] == 140875 and r.llm_usage["estimated"] is False
    assert r.llm_usage["usd"] == pytest.approx(0.0658)
    assert r.submitted_law.startswith("def discovered_law")
    assert "AgentNativeResearchLab" in r.extra["attribution"]
    assert r.extra["attribution_url"].startswith("https://huggingface.co/")
    assert all(u.startswith(ara.HF_URL.format(model="gpt5.5")) for u in r.extra["source_files"])


def test_attempt_id_deterministic(imported):
    r = _rec(imported, "gpt5.5", "gravity")
    assert r.attempt_id == ara.attempt_id("gpt5.5", "gravity", "ara_harness")
    again, _ = ara.import_cache(SAMPLE, ["gpt5.5"])
    assert [x.to_dict() for x in again] == [
        x.to_dict() for (s, _w), x in sorted(imported[0].items()) if s == "ara:gpt5.5"]


def test_salvaged_run_cross_checked_against_salvage(imported):
    r = _rec(imported, "fable", "ether")
    c = r.extra["nmse_check"]
    assert c["meta_mean_pos_error"] == "inf"  # stored as a string: records are strict JSON
    assert c["mean_pos_error_from"] == "posthoc_salvage.json" and c["agrees"] is True
    assert r.passed and r.extra["ara_passed"]


def test_systematic_nmse_disagreement_recorded(imported):
    r = _rec(imported, "fable", "coulomb_easy")
    c = r.extra["nmse_check"]
    assert c["agrees"] is False and c["kind"] == "systematic"
    assert c["implied_var"] == pytest.approx(4.23, abs=0.02)  # vs vendor 11.465
    assert r.verdict["normalised_mse"] == 0.104 and not r.passed  # scoreboard value used
    assert any("fable/coulomb_easy" in i for i in imported[1]["issues"])


def test_rounds_tie_break_scoreboard_then_meta(imported):
    r = _rec(imported, "opus4.8-max", "ether")
    assert r.rounds == 6
    assert r.extra["rounds_sources"] == {"scoreboard": 6, "meta": 6, "episode": 4}
    assert any("opus4.8-max/ether: rounds disagree" in i for i in imported[1]["issues"])


def test_world_without_files(imported):
    r = _rec(imported, "gpt5.5", "yukawa")
    assert r.rounds == 12 and r.experiments == 0
    assert r.extra["experiments_source"].startswith("unavailable")
    assert r.submitted_law is None and r.verdict["normalised_mse"] == 0.004


def test_verdict_disagreements_kept(imported):
    r = _rec(imported, "gpt5.5", "circle")  # nMSE 0.0105 but explanation 0.5
    assert r.passed and not r.extra["ara_passed"]
    assert any(d.startswith("gpt5.5/circle") for d in imported[1]["verdict_disagreements"])


def test_non_finite_nmse_written_as_none(tmp_path):
    cache = tmp_path / "ara"
    shutil.copytree(SAMPLE / "gpt5.5", cache / "gpt5.5")
    sb = cache / "gpt5.5" / "SCOREBOARD.md"
    sb.write_text(sb.read_text().replace("| gravity | 14 | 0.000115 |", "| gravity | 14 | inf |"))
    recs, report = ara.import_cache(cache, ["gpt5.5"])
    r = next(x for x in recs if x.world == "gravity")
    assert r.verdict["normalised_mse"] is None and r.passed is False
    assert r.extra["nmse_raw"] == "inf"
    assert any("non-finite" in i for i in report["issues"])
    out = tmp_path / "ara.jsonl"
    ara.write_store(recs, out)

    def strict(c):
        raise ValueError(f"non-standard JSON constant {c}")
    for line in out.read_text().splitlines():
        json.loads(line, parse_constant=strict)


def test_store_append_only_and_idempotent(tmp_path, imported):
    out = tmp_path / "ara.jsonl"
    recs = list(imported[0].values())
    assert ara.write_store(recs, out) == len(recs)
    assert ara.write_store(recs, out) == 0
    import dataclasses
    changed = dataclasses.replace(recs[0], rounds=recs[0].rounds + 1)
    assert ara.write_store([changed], out) == 1
    assert len(out.read_text().splitlines()) == len(recs) + 1
    loaded = {r.attempt_id: r for r in AttemptStore(out).load()}
    assert len(loaded) == len(recs) and loaded[changed.attempt_id].rounds == changed.rounds


def test_download_reports_unreachable(tmp_path, monkeypatch):
    def down(*a, **k):
        raise requests.ConnectionError("offline")
    monkeypatch.setattr(requests.Session, "get", down)
    with pytest.raises(ara.AraUnreachable, match="--offline"):
        ara.download(["gpt5.5"], cache_dir=tmp_path)


def test_download_skips_cached_files(tmp_path):
    cache = tmp_path / "ara"
    shutil.copytree(SAMPLE / "gpt5.5", cache / "gpt5.5")
    (cache / "gpt5.5" / "tree.json").write_text("[]")
    have = ara.download(["gpt5.5"], cache_dir=cache)  # network would raise
    assert set(have["gpt5.5"]) == {"SCOREBOARD.md", "tree.json"}


REAL = ATTEMPTS_DIR / "cache" / "ara"


@pytest.mark.skipif(not all((REAL / m / "SCOREBOARD.md").exists() for m in ara.MODELS),
                    reason="full ARA cache not downloaded")
def test_full_cache_counts_and_known_disagreements():
    recs, report = ara.import_cache(REAL)
    assert report["counts"] == dict.fromkeys(ara.MODELS, 11) and len(recs) == 88
    bad = [r for r in recs if r.extra["nmse_check"]["agrees"] is False]
    # coulomb_easy: ARA's Var ≈ 4.24 for every model; two last-digit differences.
    assert {r.world for r in bad if r.extra["nmse_check"]["kind"] == "systematic"} == {
        "coulomb_easy"}
    assert sum(r.extra["nmse_check"]["kind"] == "last_digit" for r in bad) == 2


# ------------------------------------------------------------- vendor runs

def _vendor_fixture(tmp_path):
    d = tmp_path / "results"
    (d / "a").mkdir(parents=True)
    (d / "a" / "gravity.json").write_text(json.dumps({
        "world": "gravity", "model": "m1", "law": "def discovered_law(): pass",
        "rounds": [{"experiment_input": [{}, {}]}, {"experiment_input": None},
                   {"experiment_input": [{}]}],
        "evaluation": {"mean_pos_error": 0.04283, "explanation": {"score": 0.8}}}))
    (d / "a" / "yukawa.json").write_text(json.dumps({
        "world": "yukawa", "model": "m1", "law": "x", "rounds": 5,
        "evaluation": {"mean_pos_error": 5.677}, "explanation": {"score": 0.4}}))
    (d / "a" / "config.json").write_text(json.dumps({"not": "a run"}))
    (d / "a" / "diverged.json").write_text(
        '{"world": "gravity", "model": "m2", "law": "x", "rounds": 2, '
        '"evaluation": {"mean_pos_error": Infinity}}')
    return d


def test_vendor_runs_import(tmp_path):
    recs = {(r.solver, r.world): r for r in
            vendor_runs.import_paths([_vendor_fixture(tmp_path)], root=tmp_path)}
    assert len(recs) == 3
    g = recs[("m1", "gravity")]
    assert g.protocol == "discoverphysics_native" and g.venue == "discoverphysics"
    assert (g.rounds, g.experiments) == (3, 3)
    assert g.verdict["normalised_mse"] == pytest.approx(0.01) and g.passed
    assert g.verdict["explanation_score"] == 0.8
    y = recs[("m1", "yukawa")]
    assert y.verdict["normalised_mse"] == pytest.approx(1.0) and not y.passed
    assert y.experiments == 0 and y.extra["experiments_source"].startswith("unavailable")
    assert y.verdict["explanation_score"] == 0.4
    dv = recs[("m2", "gravity")]
    assert dv.verdict["normalised_mse"] is None and not dv.passed
    assert dv.extra["mean_pos_error"] == "inf" and dv.extra["nmse_raw"] == "inf"


def test_vendor_runs_store_roundtrip(tmp_path):
    recs = vendor_runs.import_paths([_vendor_fixture(tmp_path)], root=tmp_path)
    out = tmp_path / "v.jsonl"
    assert ara.write_store(recs, out) == 3
    assert ara.write_store(recs, out) == 0
    ids = {r.attempt_id for r in AttemptStore(out).load()}
    assert ids == {r.attempt_id for r in recs}


def test_json_safe_and_finite_or_none():
    assert ara._json_safe({"a": [math.inf, {"b": -math.inf}], "c": math.nan, "d": 1.5}) == {
        "a": ["inf", {"b": "-inf"}], "c": "nan", "d": 1.5}
    assert ara.finite_or_none(math.inf) is None and ara.finite_or_none(True) is None
    assert ara.finite_or_none(0.5) == 0.5


def test_records_round_trip_strict_json(imported):
    from dm.types import AttemptRecord, canonical_json
    for r in imported[0].values():
        assert AttemptRecord.from_dict(json.loads(canonical_json(r.to_dict()))) == r
