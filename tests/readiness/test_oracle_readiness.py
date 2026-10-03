"""Oracle & settlement readiness (Phase 2 scaffolding).

Tests the pieces the oracle will build on: ``Preregistration`` commitments, the
engine's commit/reveal flow (``MarketRun.preregs``), hidden-case secrecy before
the reveal, and the oracle import-isolation rule. ``xfail(strict=True)`` marks a
known gap found during readiness testing; it turns into a failure once fixed so
the marker gets removed.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dm.outcomes import ReplayPool, experiments_cost
from dm.store import FIXTURES_DIR, AttemptStore
from dm.types import AttemptRecord, Preregistration, canonical_json
from market import MarketRun, TrackACostModel, run_market

ROOT = Path(__file__).resolve().parents[2]
SENTINEL = 7.123456789012345  # a value that only ever appears inside hidden test cases


def _cases(scale: float = 1.0) -> list[dict]:
    return [
        {"p1": 3.0, "p2": 5.0, "pos2": [SENTINEL * scale, 0.0], "velocity2": [0.0, 0.3],
         "measurement_times": [0.5, 1.0, 9.75]},
        {"p1": 4.0, "p2": 3.0, "pos2": [3.25, 0.0], "velocity2": [0.0, 0.4],
         "measurement_times": [0.25, 2.5]},
    ]


def _prereg(**kw) -> Preregistration:
    base = dict(question_id="dp:gravity:7", venue="discoverphysics", world="gravity",
                test_seed=7, test_cases=_cases(), norm_variance=1.702,
                oracle_version="0.1.0")
    base.update(kw)
    return Preregistration(**base)


def _sha(obj) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


# ----------------------------------------------------------------- commitment

class TestCommitment:

    def test_is_sha256_of_canonical_json(self):
        p = _prereg()
        assert p.commitment() == _sha(p.to_dict())
        assert len(p.commitment()) == 64

    def test_deterministic_across_processes_and_hash_seeds(self):
        code = ("import json, sys; from dm.types import Preregistration;"
                "print(Preregistration.from_dict(json.loads(sys.stdin.read())).commitment())")
        payload = json.dumps(_prereg().to_dict())
        outs = set()
        for hs in ("0", "1", "random"):
            env = {**os.environ, "PYTHONHASHSEED": hs}
            outs.add(subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                                    input=payload, capture_output=True, text=True,
                                    check=True).stdout.strip())
        assert outs == {_prereg().commitment()}

    def test_insensitive_to_dict_key_order(self):
        reordered = [dict(reversed(list(c.items()))) for c in _cases()]
        assert _prereg(test_cases=reordered).commitment() == _prereg().commitment()

    @pytest.mark.parametrize("field,value", [
        ("question_id", "dp:gravity:8"), ("venue", "forcebench"), ("world", "yukawa"),
        ("test_seed", 8), ("norm_variance", 1.7020000000000002),
        ("oracle_version", "0.1.1"), ("metric", "mse"), ("threshold", 0.1000001),
        ("public_tests", True),
    ])
    def test_every_field_changes_commitment(self, field, value):
        assert _prereg(**{field: value}).commitment() != _prereg().commitment()

    def test_all_fields_are_covered_by_the_parametrisation(self):
        tested = {"question_id", "venue", "world", "test_seed", "norm_variance",
                  "oracle_version", "metric", "threshold", "public_tests", "test_cases"}
        assert {f.name for f in dataclasses.fields(Preregistration)} == tested

    @pytest.mark.parametrize("mutate", [
        lambda c: c[0]["pos2"].__setitem__(0, SENTINEL + 1e-12),
        lambda c: c[1]["measurement_times"].append(5.0),
        lambda c: c[0].__setitem__("p1", 3),            # int vs float is visible
        lambda c: c.reverse(),                          # case order is committed
        lambda c: c.pop(),
        lambda c: c[0].__setitem__("extra", None),
    ])
    def test_test_case_content_changes_commitment(self, mutate):
        cases = _cases()
        mutate(cases)
        assert _prereg(test_cases=cases).commitment() != _prereg().commitment()

    def test_round_trip_to_dict(self):
        p = _prereg()
        q = Preregistration.from_dict(p.to_dict())
        assert q == p and q.commitment() == p.commitment()

    def test_round_trip_through_canonical_json(self):
        # What the reveal event carries is JSON; it must hash back to the commitment.
        p = _prereg()
        q = Preregistration.from_dict(json.loads(canonical_json(p.to_dict())))
        assert q.commitment() == p.commitment()

    def test_close_floats_round_trip_exactly(self):
        x = float(np.nextafter(1.702, 2.0))
        p, q = _prereg(norm_variance=1.702), _prereg(norm_variance=x)
        assert p.commitment() != q.commitment()
        back = Preregistration.from_dict(json.loads(canonical_json(q.to_dict())))
        assert back.norm_variance == x and back.commitment() == q.commitment()

    def test_negative_zero_is_distinct_but_stable(self):
        a, b = _prereg(norm_variance=0.0), _prereg(norm_variance=-0.0)
        assert a.commitment() != b.commitment()
        back = Preregistration.from_dict(json.loads(canonical_json(b.to_dict())))
        assert math.copysign(1, back.norm_variance) == -1
        assert back.commitment() == b.commitment()

    def test_tuples_hash_like_lists(self):
        cases = _cases()
        cases[0]["pos2"] = tuple(cases[0]["pos2"])
        assert _prereg(test_cases=cases).commitment() == _prereg().commitment()

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
    def test_non_finite_values_are_rejected(self, bad):
        with pytest.raises(ValueError):
            _prereg(norm_variance=bad).commitment()
        cases = _cases()
        cases[1]["pos2"][0] = bad
        with pytest.raises(ValueError):
            _prereg(test_cases=cases).commitment()

    def test_numpy_float64_hashes_like_float(self):
        cases = _cases()
        cases[0]["pos2"] = [np.float64(SENTINEL), np.float64(0.0)]
        assert _prereg(test_cases=cases, norm_variance=np.float64(1.702)).commitment() \
            == _prereg().commitment()

    @pytest.mark.xfail(strict=True, raises=TypeError,
                       reason="gap: canonical_json rejects numpy int64/float32/ndarray; "
                              "make_prereg will build cases with numpy RNG")
    @pytest.mark.parametrize("value", [np.int64(7), np.float32(0.5), np.array([1.0, 2.0])])
    def test_numpy_scalars_and_arrays_are_normalised(self, value):
        cases = _cases()
        cases[0]["p1"] = value
        _prereg(test_cases=cases).commitment()

    def test_public_view_has_no_test_cases(self):
        p = _prereg()
        pub = p.public()
        assert "test_cases" not in pub and pub["commitment"] == p.commitment()
        assert repr(SENTINEL) not in canonical_json(pub)

    @pytest.mark.xfail(strict=True,
                       reason="gap: public() publishes test_seed and the commitment has no "
                              "salt, so hidden cases derived from the seed are recoverable")
    def test_hidden_cases_not_recoverable_from_public_view(self):
        # Model of PLAN's make_prereg(venue, world, test_seed): cases are a pure
        # function of the seed. Anyone holding public() + the code can rebuild them.
        def make(seed: int) -> Preregistration:
            rng = np.random.default_rng(seed)
            cases = [{"p1": float(rng.choice([3, 4, 5])), "p2": float(rng.choice([3, 5])),
                      "pos2": [float(rng.uniform(2.5, 5.0)), 0.0], "velocity2": [0.0, 0.3],
                      "measurement_times": [1.0, 5.0]}]
            return _prereg(test_seed=seed, test_cases=cases)

        posted = make(4242)
        pub = posted.public()
        recovered = None
        for seed in range(10_000):  # even without test_seed, small seeds brute-force
            if make(seed).commitment() == pub["commitment"]:
                recovered = make(seed)
                break
        assert recovered is None, "hidden test cases recovered from the public commitment"


# ------------------------------------------------------- engine commit/reveal

def _bernoulli_cfg(preregs, worlds=("gravity", "yukawa"), probs=(0.6, 0.0), ticks=40,
                   seed=0):
    worlds = list(worlds)
    return MarketRun(
        run_id="prereg", seed=seed, ticks=ticks, worlds=worlds, agents=["a", "b"],
        prizes={w: 40.0 for w in worlds}, starting_credits=100,
        cost_model=TrackACostModel(),
        true_probs={a: dict(zip(worlds, probs)) for a in ["a", "b"]},
        initial_beliefs={a: {w: 0.6 for w in worlds} for a in ["a", "b"]},
        belief_weight=2, track="A", probability_source="t", preregs=preregs)


def _replay_cfg(preregs, seed=0, ticks=50):
    recs = AttemptStore(FIXTURES_DIR / "replay_pool.jsonl").load()
    pool = ReplayPool(recs, experiments_cost(0.5))
    worlds = ["gravity", "yukawa", "coulomb_easy"]
    return MarketRun(
        run_id="prereg_replay", seed=seed, ticks=ticks, worlds=worlds,
        agents=["fast", "slow"], prizes={w: 500.0 for w in worlds},
        starting_credits=100, cost_model=pool, true_probs=None,
        initial_beliefs={a: {w: 0.5 for w in worlds} for a in ["fast", "slow"]},
        belief_weight=2, track="replay", probability_source="replay:fixture",
        outcome_source=pool, state_confidence=True, preregs=preregs)


WORLD_SCALE = {"gravity": 1.0, "yukawa": 1.5, "coulomb_easy": 2.0}


def _preregs(worlds):
    # Distinct hidden cases per world: shared cases would leak through an earlier
    # world's reveal (per-question test seeds).
    return {w: _prereg(world=w, question_id=f"dp:{w}:7", test_cases=_cases(WORLD_SCALE[w]))
            for w in worlds}


def _run(cfg):
    events, ledger = run_market(cfg)
    return [e.to_dict() for e in events], [r.to_dict() for r in ledger]


CFGS = {
    "bernoulli": lambda s: _bernoulli_cfg(_preregs(["gravity", "yukawa"]), seed=s),
    "replay": lambda s: _replay_cfg(_preregs(["gravity", "yukawa", "coulomb_easy"]), seed=s),
}


@pytest.mark.parametrize("kind", sorted(CFGS))
@pytest.mark.parametrize("seed", range(4))
class TestEnginePreregFlow:

    def test_posting_carries_only_commitment(self, kind, seed):
        cfg = CFGS[kind](seed)
        events, _ = _run(cfg)
        for w, p in cfg.preregs.items():
            posted = [e for e in events if e["type"] == "prize_posted" and e["world"] == w]
            assert len(posted) == 1
            e = posted[0]
            assert e["commitment"] == p.commitment() and "detail" not in e
            nxt = events[events.index(e) + 1]
            assert nxt["type"] == "prereg_committed" and nxt["world"] == w
            assert nxt["commitment"] == p.commitment() and "detail" not in nxt

    def test_reveal_once_at_close_or_refund_and_hashes_to_commitment(self, kind, seed):
        cfg = CFGS[kind](seed)
        events, _ = _run(cfg)
        for w, p in cfg.preregs.items():
            reveals = [i for i, e in enumerate(events)
                       if e["type"] == "prereg_revealed" and e["world"] == w]
            assert len(reveals) == 1
            i = reveals[0]
            prev = events[i - 1]
            assert prev["type"] in ("prize_paid", "prize_refunded") and prev["world"] == w
            assert events[i]["tick"] == prev["tick"]
            detail = events[i]["detail"]
            assert _sha(detail) == events[i]["commitment"] == p.commitment()
            assert Preregistration.from_dict(json.loads(json.dumps(detail))) == p

    def test_no_hidden_case_content_before_reveal(self, kind, seed):
        cfg = CFGS[kind](seed)
        events, ledger = _run(cfg)
        generic = ['"test_cases"', "measurement_times", "norm_variance"]
        reveal_at = {e["world"]: i for i, e in enumerate(events)
                     if e["type"] == "prereg_revealed"}
        first = min(reveal_at.values())
        for n in generic:
            assert n not in json.dumps(events[:first]), n
        for w in cfg.preregs:
            own = repr(SENTINEL * WORLD_SCALE[w])
            assert own not in json.dumps(events[:reveal_at[w]]), w
        # The ledger never carries the preregistration at all.
        assert not any(n in json.dumps(ledger) for n in generic + [repr(SENTINEL)])

    def test_committed_before_any_bid(self, kind, seed):
        events, _ = _run(CFGS[kind](seed))
        first_bid = next(i for i, e in enumerate(events) if e["type"] == "bid_placed")
        assert all(i < first_bid for i, e in enumerate(events)
                   if e["type"] == "prereg_committed")


def test_unsolved_world_reveals_at_refund():
    cfg = _bernoulli_cfg(_preregs(["gravity", "yukawa"]), probs=(0.0, 0.0), ticks=5)
    events, _ = _run(cfg)
    r = [e for e in events if e["type"] == "prereg_revealed"]
    assert {e["world"] for e in r} == {"gravity", "yukawa"} and {e["tick"] for e in r} == {5}


def test_worlds_without_prereg_emit_no_prereg_events():
    cfg = _bernoulli_cfg(_preregs(["gravity"]))
    events, _ = _run(cfg)
    assert not [e for e in events if e.get("world") == "yukawa"
                and (e["type"].startswith("prereg_") or "commitment" in e)]


def test_preregs_do_not_change_settlement_or_balances():
    a = _run(_bernoulli_cfg(None))[0]
    b = _run(_bernoulli_cfg(_preregs(["gravity", "yukawa"])))[0]
    strip = lambda ev: [(e["type"], e.get("world"), e.get("agent"), e.get("amount"))
                        for e in ev if not e["type"].startswith("prereg_")]
    assert strip(a) == strip(b)


@pytest.mark.xfail(strict=True,
                   reason="gap: verdict_issued.commitment comes from the replayed record and "
                          "is never checked against the prereg posted for the world")
def test_verdicts_settle_against_the_posted_commitment():
    cfg = _replay_cfg(_preregs(["gravity", "yukawa", "coulomb_easy"]))
    events, _ = _run(cfg)
    posted = {e["world"]: e["commitment"] for e in events if e["type"] == "prize_posted"}
    verdicts = [e for e in events if e["type"] == "verdict_issued"]
    assert verdicts
    for e in verdicts:
        assert e.get("commitment") == posted[e["world"]], e


def test_store_round_trips_prereg_commitment_and_last_line_wins(tmp_path):
    p = _prereg()
    rec = AttemptRecord(
        attempt_id="x", source="live", protocol="dp-v1", venue="discoverphysics",
        world="gravity", solver="s", seed=0, stated_p_success=0.5, rounds=1,
        experiments=2, lab_cost=1.0, llm_usage={}, submitted_law="def f(): pass",
        verdict={"passed": False, "normalised_mse": 0.5, "explanation_score": None,
                 "prereg_commitment": p.commitment()},
        transcript_path=None, created_at="2026-10-03T00:00:00", extra={})
    store = AttemptStore(tmp_path / "a.jsonl")
    store.append([rec])
    assert store.load()[0].verdict["prereg_commitment"] == p.commitment()
    # Append-only, but a later line silently replaces the verdict: settlement must
    # therefore never re-read verdicts from the store after a prize is paid.
    store.append([dataclasses.replace(rec, verdict={**rec.verdict, "passed": True})])
    assert store.load()[0].passed is True
    assert len((tmp_path / "a.jsonl").read_text().splitlines()) == 2


# ----------------------------------------------------------- import isolation

def _module_name(path: Path, root: Path = ROOT) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = rel.parts[:-1] if rel.name == "__init__" else rel.parts
    return ".".join(parts)


def _imports(path: Path, root: Path = ROOT) -> set[str]:
    """Absolute names of dm.* modules imported anywhere in the file (incl. lazily)."""
    mod = _module_name(path, root)
    pkg = mod if path.name == "__init__.py" else mod.rpartition(".")[0]
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = pkg.split(".")[: len(pkg.split(".")) - (node.level - 1)]
                base = ".".join(anchor + ([base] if base else []))
            out.add(base)
            out |= {f"{base}.{a.name}" for a in node.names}
        elif (isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", ""))
              in ("import_module", "__import__") and node.args
              and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            out.add(node.args[0].value)
    return {m for m in out if m == "dm" or m.startswith("dm.")}


def _graph() -> dict[str, set[str]]:
    files = {_module_name(p): p for p in (ROOT / "dm").rglob("*.py")}
    graph = {}
    for name, path in files.items():
        deps = set()
        for imp in _imports(path):
            # importing a.b.c also executes a and a.b
            parts = imp.split(".")
            deps |= {".".join(parts[:i]) for i in range(1, len(parts) + 1)}
        graph[name] = {d for d in deps if d in files and d != name}
    return graph


def _reaches(graph, start, target) -> bool:
    seen, stack = set(), [start]
    while stack:
        m = stack.pop()
        if m == target or m.startswith(target + "."):
            return True
        if m not in seen:
            seen.add(m)
            stack.extend(graph.get(m, ()))
    return False


def test_import_graph_sees_lazy_and_relative_imports(tmp_path):
    src = ("import dm.store\nfrom . import types\nfrom ..oracle import score\n"
           "def f():\n    import importlib; importlib.import_module('dm.oracle.worker')\n")
    f = tmp_path / "dm" / "_probe_pkg" / "x.py"
    f.parent.mkdir(parents=True)
    (f.parent / "__init__.py").write_text("")
    f.write_text(src)
    assert {"dm.store", "dm._probe_pkg.types", "dm.oracle", "dm.oracle.score",
            "dm.oracle.worker"} <= _imports(f, root=tmp_path)


def test_solvers_and_venues_never_reach_the_oracle():
    graph = _graph()
    offenders = [m for m in graph
                 if (m.startswith("dm.solvers") or m.startswith("dm.venues"))
                 and _reaches(graph, m, "dm.oracle")]
    assert not offenders, offenders


def test_only_settle_imports_the_oracle():
    graph = _graph()
    direct = {m for m, deps in graph.items()
              if not m.startswith("dm.oracle") and any(d.startswith("dm.oracle") for d in deps)}
    assert direct <= {"dm.settle"}, direct


def test_shared_modules_do_not_reach_the_oracle():
    # Solvers import dm.types / dm.store / dm.outcomes; if any of those ever pulls in
    # the oracle, every solver transitively can read hidden cases.
    graph = _graph()
    for m in ("dm", "dm.types", "dm.store", "dm.outcomes", "dm.testing"):
        if m in graph:
            assert not _reaches(graph, m, "dm.oracle"), m
