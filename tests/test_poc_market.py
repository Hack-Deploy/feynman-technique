"""Bounty market v2 (poc/): claims with checked estimates, the bid rule, the checker and its
flags, conservation under both reward rules, true values, independence, and the baselines.

No API calls. Tests marked ``slow`` run the vendor simulator."""

from __future__ import annotations

import ast
import dataclasses
import re
from pathlib import Path

import pytest

from poc import config as C
from poc import protocol
from poc.ledger import ConservationError, Ledger, settle

CFG = C.load()
HYP = CFG.hypothesis("gravity-inverse-square")


# ------------------------------------------------------------------ parsing and the bid rule

def test_parse_estimates_lines_sigma_and_fences():
    text = "```xml\n<estimate>\nn = 1.02 ± 0.05\na3 = 5.3e-2 +/- 0.002\n</estimate>\n```"
    est = protocol.parse_estimates(text)
    assert est == {"n": {"value": 1.02, "sigma": 0.05}, "a3": {"value": 0.053, "sigma": 0.002}}
    assert protocol.parse_estimates("<estimate>n: -1</estimate>") == {"n": {"value": -1.0, "sigma": None}}
    assert protocol.parse_estimates("<estimate>n is about two</estimate>") == {}


def test_bid_rule_uses_the_agents_own_numbers():
    assert protocol.bids(0.5, 40, 100)
    assert not protocol.bids(0.3, 40, 100)
    assert not protocol.bids(None, 1, 100) and not protocol.bids(0.9, None, 100)


def test_rule_verdicts():
    r = C.Rule("x", "outside", (0.9, 1.1))
    assert r.verdict({"x": -1}) == "supported" and r.verdict({"x": 1.0}) == "refuted"
    assert C.Rule("x", "above", (2.0,)).verdict({"x": 2.5}) == "supported"
    assert C.Rule("x", "between", (1.8, 2.2)).verdict({"x": 1.0}) == "refuted"
    assert C.Rule("x", "above", (2.0,)).verdict({}) is None


def test_prompt_lists_quantities_tolerances_bond_and_hides_truth():
    block = protocol.market_block(CFG, HYP, "market")
    for q in HYP.quantities:
        assert f"`{q.name}`" in block and f"±{q.tolerance:g}" in block
    assert f"{CFG.claim_bond * HYP.prize:g} credits" in block
    assert "p_success × prize > planned_cost" in block
    naive = protocol.market_block(CFG, HYP, "naive")
    assert "not checked" in naive and "bond" not in naive.split("**Costs**")[0]


def test_unusable_fit_covariance_gives_no_sigma_not_nan():
    import json
    import numpy as np
    from poc.estimate import _fit
    # rank-deficient: only p0 + p1 is determined, so the covariance cannot be computed
    est = _fit(lambda p: np.array([p[0] + p[1] - 1.0, 0.0, 0.0]), [np.zeros(2)],
               lambda p: {"s": p[0] + p[1]}, 3)
    assert est.values["s"] == pytest.approx(1.0) and est.sigmas == {"s": None}
    json.dumps(est.sigmas, allow_nan=False)


# ------------------------------------------------------------------ agent loop: bid and claim

def _agent(replies, cfg=CFG, hyp=HYP, **kw):
    from tests.test_poc import StubExecutor
    from poc.agent import MarketAgent
    from poc.fake_llm import ScriptedLLM
    from poc.pricing import Account

    vendor = C.VENDOR_ROOT / "PhysicsSchool" / "prompts"
    account = Account(agent="m", hypothesis=hyp.id, budget=cfg.budget)
    llm = ScriptedLLM(replies)
    agent = MarketAgent(cfg=cfg, hyp=hyp, account=account, ledger_entries=[], complete=llm,
                        model="m", executor=StubExecutor(), mission="mission", verbose=False,
                        system_prompt_path=str(vendor / "_template_interactive.md"),
                        instructions_path=str(vendor / "2particle_instructions.md"),
                        law_stub="def discovered_law(...):\n    pass\n",
                        experiment_format="<run_experiment>[]</run_experiment>",
                        trajectory_logger=None, **kw)
    return agent, account, llm


CLAIM = ("<assessment>a</assessment><p_success>0.7</p_success><verdict>refuted</verdict>"
         "<estimate>n = 1.0 ± 0.1\na3 = 0.053</estimate><evidence>e</evidence>")


def test_bid_below_break_even_is_declined_and_free():
    agent, account, llm = _agent(["<assessment>a</assessment><p_success>0.01</p_success>"
                                  "<planned_cost>50</planned_cost><run_experiment>[]</run_experiment>"])
    assert agent.run() is None
    assert agent.outcome == "declined" and account.spent == 0 and len(llm.calls) == 1
    assert agent.bid["accepted"] is False


def test_missing_planned_cost_reprompts_once():
    agent, _, llm = _agent(["<assessment>a</assessment><p_success>0.7</p_success>",
                            "<planned_cost>10</planned_cost><plan>{\"experiments\": []}</plan>"
                            + CLAIM])
    agent.run()
    assert agent.bid["planned_cost"] == 10 and agent.bid["plan"] == {"experiments": []}
    assert agent.outcome == "verdict" and agent.estimates["n"]["value"] == 1.0


def test_missing_estimates_reprompt_once():
    claim = CLAIM.replace("<estimate>n = 1.0 ± 0.1\na3 = 0.053</estimate>", "<estimate>n = 1</estimate>")
    agent, _, llm = _agent(["<p_success>0.7</p_success><planned_cost>5</planned_cost>" + claim,
                            "<estimate>a3 = 0.05</estimate>"])
    agent.run()
    assert set(agent.estimates) == {"n", "a3"} and agent.estimates["n"]["value"] == 1.0


def test_prior_only_mode_has_one_round_and_no_lab():
    agent, account, llm = _agent(["<p_success>0.7</p_success><planned_cost>0</planned_cost>"
                                  "<run_experiment>[{}]</run_experiment>"], experiments=False)
    agent.run()
    assert agent.max_rounds == 1 and agent.executor.inner.calls == []
    assert "cannot run experiments" in llm.calls[0]["system"]


# ------------------------------------------------------------------ ledger

def _charges(*amounts):
    return [{"type": "round_charged", "amount": a, "round": i} for i, a in enumerate(amounts, 1)]


@pytest.mark.parametrize("rule", C.RULES)
@pytest.mark.parametrize("outcome,verdict,confirmed", [
    ("verdict", "supported", True), ("verdict", "refuted", False), ("verdict", "inconclusive", False),
    ("withdrawn", None, False), ("walked_away", None, False), ("declined", None, False)])
def test_settlement_conserves_credits_and_empties_escrow(rule, outcome, verdict, confirmed):
    s = settle(rule, CFG, 100.0, _charges(2, 2.5, 13.75), outcome, verdict, confirmed, 0.8)
    net = sum(e["amount"] * (1 if e["to"] == "agent" else -1)
              for e in s["events"] if "agent" in (e["to"], e["from"]))
    assert s["profit"] == pytest.approx(net)


def test_market_vs_naive_payouts():
    false_claim = dict(outcome="verdict", agent_verdict="supported", confirmed=False, bid_p=0.9)
    m = settle("market", CFG, 100.0, _charges(2), **false_claim)
    n = settle("naive", CFG, 100.0, _charges(2), **false_claim)
    bonus = CFG.calibration_bonus * 100 * (1 - 4 * 0.81)
    assert n["profit"] == pytest.approx(98)
    assert m["profit"] == pytest.approx(-2 - CFG.claim_bond * 100 + bonus)
    stop = settle("market", CFG, 100.0, _charges(2), "verdict", "inconclusive", False, 0.5)
    assert stop["profit"] == pytest.approx(-2)  # honest stopping loses only its costs
    assert stop["profit"] > m["profit"]


def test_ledger_raises_when_credits_leak():
    led = Ledger({"researcher": 10})
    led.transfer("prize_posted", "researcher", "escrow", 10)
    led.balances["escrow"] -= 1
    with pytest.raises(ConservationError):
        led.close()


# ------------------------------------------------------------------ independence

HIDDEN = ("poc.truth", "poc.checker")
PLAYERS = ("agent.py", "protocol.py", "attempt.py", "baselines.py", "estimate.py",
           "reference.py", "lab.py", "fake_llm.py", "pricing.py")


def test_agents_cannot_import_the_truth_or_the_checker():
    for name in PLAYERS:
        src = (C.ROOT / "poc" / name).read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [f"{node.module}.{a.name}" for a in node.names] + [node.module or ""]
            else:
                continue
            assert not any(m.startswith(HIDDEN) for m in mods), (name, mods)
        # dynamic imports: importlib, __import__, or the module names as strings
        assert not re.search(r"importlib|__import__", src), name
        assert not re.search(r"""["'](poc\.)?(truth|checker)["']""", src), name


# ------------------------------------------------------------------ simulator-backed

@pytest.mark.slow
def test_every_true_value_implies_the_hidden_answer_with_margin():
    from poc import truth
    for h in CFG.hypotheses:
        values = truth.true_values(h)
        assert set(values) == {q.name for q in h.quantities}
        assert h.supported_if.verdict(values) == h.answer, h.id
        q = h.quantity(h.supported_if.quantity)
        assert h.supported_if.margin(values[q.name]) > q.tolerance, h.id


@pytest.mark.slow
def test_free_prior_guesses_are_never_within_tolerance():
    """A guess made without experiments must miss on at least one quantity, or the market
    would pay for guessing."""
    from poc import truth
    from poc.baselines import REFUTE_GUESS, SUPPORT_GUESS
    for h in CFG.hypotheses:
        values = truth.true_values(h)
        for guess in (SUPPORT_GUESS[h.id], REFUTE_GUESS[h.id]):
            assert any(abs(guess[q.name] - values[q.name]) > q.tolerance for q in h.quantities), \
                (h.id, guess, values)


@pytest.mark.slow
def test_lab_returns_positions_only():
    from scienceagent.worlds import get_world
    from poc.lab import PositionsOnlyExecutor
    for world, exp in [("gravity", {"p1": 1, "p2": 1, "pos2": [3, 0], "velocity2": [0, 0],
                                    "measurement_times": [1.0]}),
                       ("hubble", {"probe_positions": [[5, 0]] * 5,
                                   "probe_velocities": [[0, 0]] * 5, "measurement_times": [1.0]})]:
        ex = PositionsOnlyExecutor(get_world(world, engine="nbody", noise_std=0.075, noise_seed=0)["executor"])
        out = ex.run([exp])[0]
        assert not any("veloc" in k for k in out), out.keys()


@pytest.mark.slow
@pytest.mark.parametrize("name", ["abstain", "always_supported", "coin_flip", "p_hacker", "reference"])
def test_baselines_run_end_to_end_and_conserve(name):
    from poc import baselines
    from poc.attempt import run_attempt
    from poc.bench import resolve

    hyp = CFG.hypothesis("ether-outward-push")  # refuted: the p-hacker wants "supported"
    s = run_attempt(baselines.PREFIX + name, hyp.id, 0, [], cfg=CFG,
                    complete=baselines.make(baselines.PREFIX + name, CFG, hyp, 0))
    r = resolve(hyp, s, CFG)
    outcome = r.verdict["outcome"]
    expect = {"abstain": {"walked_away"}, "always_supported": {"false_claim"},
              "coin_flip": {"false_claim"}, "p_hacker": {"false_claim"},
              "reference": {"confirmed"}}[name]
    assert outcome in expect, (name, outcome, r.verdict["ruling"])
    if name == "p_hacker":
        assert {"rerun"} <= {f["flag"] for f in r.verdict["ruling"]["flags"]}
    if name == "always_supported":
        assert r.extra["settlements"]["naive"]["profit"] > 0 > r.extra["settlements"]["market"]["profit"]


@pytest.mark.slow
def test_animation_scene_has_true_paths_and_no_nan():
    import json
    from poc import animate, baselines
    from poc.attempt import run_attempt
    from poc.bench import resolve

    hyp = CFG.hypothesis("gravity-inverse-square")
    s = run_attempt(baselines.PREFIX + "p_hacker", hyp.id, 0, [], cfg=CFG,
                    complete=baselines.make(baselines.PREFIX + "p_hacker", CFG, hyp, 0))
    scene = animate.scene(resolve(hyp, s, CFG), CFG)
    json.dumps(scene, allow_nan=False)
    assert scene["outcome"] == "false_claim" and scene["launches"]
    first = scene["launches"][0]
    assert first["true"] and len(first["true"]) == animate.FRAMES + 1
    assert first["claimed"] and first["single_estimate"]
