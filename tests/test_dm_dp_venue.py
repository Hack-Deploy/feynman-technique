"""Phase 3: DiscoverPhysics venue (metered vendor loop, usage, stated p, live guard).

Every test uses the scripted fake LLM; nothing calls an API.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from dm.llm import (BudgetExceeded, LiveDisabled, LLMReply, SpendCap, UsageMeter,
                    live_llm, usd_for)
from dm.testing.fake_llm import (EXPERIMENT, ScriptedLLM, fake_llm,
                                 one_experiment_then_law)
from dm.types import InsufficientCredits
from dm.venues.discoverphysics import (STATED_P_PROMPT, MeteredExecutor, market_note,
                                       parse_p, run_attempt)
from dm.wallet import Wallet

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.filterwarnings("ignore:DM_ORACLE_SECRET is not set")


def _executor(seed=0, world="gravity"):
    from scienceagent.worlds import get_world
    return get_world(world, engine="nbody", noise_std=0.075, noise_seed=seed)["executor"]


def _run(tmp_path, llm=None, balance=10.0, price=0.5, **kw):
    wallet = Wallet("fake", balance)
    llm = llm if llm is not None else fake_llm()
    kw.setdefault("world", "gravity")
    kw.setdefault("seed", 0)
    a = run_attempt("fake-1overr", kw.pop("world"), kw.pop("seed"), wallet, price,
                    llm=llm, transcript_dir=tmp_path, **kw)
    return a, wallet, llm


# --------------------------------------------------------------------- MeteredExecutor

class TestMeteredExecutor:

    def test_charges_before_running(self):
        wallet = Wallet("a", 10)
        seen = []

        class Spy:
            def run(self, exps):
                seen.append(wallet.balance)
                return [{} for _ in exps]

        MeteredExecutor(Spy(), wallet, 0.5, "gravity").run([EXPERIMENT, EXPERIMENT])
        assert seen == [9.0]

    def test_over_balance_never_runs(self):
        wallet = Wallet("a", 1)
        m = MeteredExecutor(_executor(), wallet, 0.5, "gravity")
        with pytest.raises(InsufficientCredits):
            m.run([EXPERIMENT] * 3)
        assert wallet.balance == 1.0 and wallet.lab_revenue == 0.0
        assert wallet.events[-1]["type"] == "insufficient_credits"

    def test_fractional_price_exact_balance(self):
        wallet = Wallet("a", 0.3)
        MeteredExecutor(_executor(), wallet, 0.1, "gravity").run([EXPERIMENT] * 3)
        assert wallet.balance == 0.0 and wallet.lab_revenue == 0.3

    def test_non_list_refused_uncharged(self):
        wallet = Wallet("a", 10)
        m = MeteredExecutor(_executor(), wallet, 1.0, "gravity")
        for bad in (dict(EXPERIMENT), [EXPERIMENT, 3], "x"):
            with pytest.raises(ValueError):
                m.run(bad)
        assert wallet.lab_revenue == 0.0 and wallet.events == []

    def test_simulator_failure_refunded(self):
        wallet = Wallet("a", 10)
        m = MeteredExecutor(_executor(), wallet, 1.0, "gravity")
        with pytest.raises(KeyError):
            m.run([EXPERIMENT, {"p1": 1.0}])
        assert wallet.balance == 10.0 and wallet.lab_revenue == 0.0
        assert [e["type"] for e in wallet.events] == ["experiment_charged",
                                                      "experiment_refunded"]

    def test_forwards_noise_controls(self):
        m = MeteredExecutor(_executor(), Wallet("a", 1), 1.0, "gravity")
        assert m.noise_std == 0.075
        with m.noise_disabled():
            assert m.noise_std == 0.0


# --------------------------------------------------------------------- run_attempt

class TestRunAttempt:

    def test_fake_attempt_end_to_end(self, tmp_path):
        a, wallet, llm = _run(tmp_path)
        assert (a.venue, a.protocol, a.source) == ("discoverphysics",
                                                    "discoverphysics_native", "live")
        assert a.rounds == 2 and a.experiments == 1 and a.lab_cost == 0.5
        assert wallet.balance == 9.5 and wallet.lab_revenue == 0.5
        assert "discovered_law" in a.submitted_law
        assert a.explanation and a.stated_p_success == 0.6
        assert len(a.training) == 1 and set(a.training[0]) == {"input", "output"}
        assert a.llm_usage["calls"] == 3 and a.llm_usage["estimated"] is True
        assert len(llm.calls) == 3 and llm.calls[-1]["messages"][-1]["content"] == STATED_P_PROMPT

    def test_settles_through_oracle(self, tmp_path):
        from dm.settle import prereg_for, settle
        a, _, _ = _run(tmp_path)
        rec = settle(prereg_for("discoverphysics", "gravity", 0), a)
        assert rec.passed and rec.verdict["normalised_mse"] < 1e-4
        assert rec.lab_cost == 0.5 and rec.llm_usage == a.llm_usage

    def test_one_over_r_fails_yukawa(self, tmp_path):
        from dm.settle import prereg_for, settle
        a, _, _ = _run(tmp_path, world="yukawa")
        assert not settle(prereg_for("discoverphysics", "yukawa", 0), a).passed

    def test_stated_p_asked_in_same_conversation(self, tmp_path):
        _, _, llm = _run(tmp_path)
        law_call, p_call = llm.calls[1], llm.calls[2]
        assert p_call["system"] == law_call["system"]
        assert p_call["messages"][:-2] == law_call["messages"]
        assert "<final_law>" in p_call["messages"][-2]["content"]

    def test_stated_p_unparseable_is_none(self, tmp_path):
        llm = ScriptedLLM(one_experiment_then_law(), p_reply="about 70%")
        assert _run(tmp_path, llm=llm)[0].stated_p_success is None
        llm = ScriptedLLM(one_experiment_then_law(), p_reply="<p_success>1.7</p_success>")
        assert _run(tmp_path, llm=llm)[0].stated_p_success is None

    def test_no_submission_gives_no_law_and_no_p_call(self, tmp_path):
        llm = ScriptedLLM([f"<run_experiment>{json.dumps([EXPERIMENT])}</run_experiment>"])
        a, wallet, _ = _run(tmp_path, llm=llm, max_rounds=4)
        assert a.submitted_law is None and a.stated_p_success is None
        assert a.rounds == 4 and a.llm_usage["calls"] == 4
        assert a.experiments == 4 and wallet.lab_revenue == 2.0
        assert a.extra["stopped_reason"] == "max_rounds"

    def test_no_submission_settles_as_fail(self, tmp_path):
        from dm.settle import prereg_for, settle
        llm = ScriptedLLM([f"<run_experiment>{json.dumps([EXPERIMENT])}</run_experiment>"])
        a, _, _ = _run(tmp_path, llm=llm, max_rounds=2)
        rec = settle(prereg_for("discoverphysics", "gravity", 0), a)
        assert rec.settled and not rec.passed

    def test_insufficient_credits_shown_and_attempt_still_submits(self, tmp_path):
        llm = ScriptedLLM(one_experiment_then_law([EXPERIMENT] * 3))
        a, wallet, llm = _run(tmp_path, llm=llm, balance=1.0)
        assert a.submitted_law is not None and a.experiments == 0 and a.lab_cost == 0
        assert wallet.balance == 1.0
        shown = llm.calls[1]["messages"][-1]["content"]
        assert "insufficient credits: 3 experiment(s) x 0.5 = 1.5" in shown
        assert a.extra["insufficient_credit_refusals"] == 1

    def test_malformed_batch_shown_to_solver_and_refunded(self, tmp_path):
        llm = ScriptedLLM(one_experiment_then_law([EXPERIMENT, {"p1": 1.0}]))
        a, wallet, llm = _run(tmp_path, llm=llm)
        assert wallet.balance == 10.0 and a.experiments == 0 and a.lab_cost == 0
        assert "Error running experiment" in llm.calls[1]["messages"][-1]["content"]

    def test_market_aware_changes_only_the_mission(self, tmp_path):
        _, _, aware = _run(tmp_path, prize=60)
        _, _, plain = _run(tmp_path, market_aware=False)
        assert aware.calls[0]["system"] == plain.calls[0]["system"]
        note = market_note(0.5, 10.0, 60)
        assert aware.calls[0]["messages"][0]["content"] == (
            plain.calls[0]["messages"][0]["content"] + note)
        assert "0.5 credits" in note and "balance is 10 credits" in note and "60" in note

    def test_charge_events_carry_agent_round(self, tmp_path):
        replies = [f"<run_experiment>{json.dumps([EXPERIMENT])}</run_experiment>",
                   "no tags here",
                   f"<run_experiment>{json.dumps([EXPERIMENT] * 2)}</run_experiment>",
                   one_experiment_then_law()[1]]
        a, wallet, _ = _run(tmp_path, llm=ScriptedLLM(replies))
        charges = [(e["round"], e["count"]) for e in wallet.events]
        assert charges == [(1, 1), (3, 2)] and a.experiments == 3

    def test_reported_usage_from_provider(self, tmp_path):
        a, _, _ = _run(tmp_path, llm=fake_llm(with_usage=True))
        assert a.llm_usage["estimated"] is False and a.llm_usage["calls"] == 3

    def test_transcript_written_strict_json(self, tmp_path):
        a, _, _ = _run(tmp_path)
        t = json.loads(Path(a.transcript_path).read_text() if Path(a.transcript_path).is_absolute()
                       else (ROOT / a.transcript_path).read_text())
        assert t["rounds"][0]["action"] == "experiment"
        assert t["stated_p"]["reply"] == "<p_success>0.6</p_success>"
        assert t["wallet_events"][0]["type"] == "experiment_charged"

    def test_deterministic(self, tmp_path):
        a, _, _ = _run(tmp_path / "a", seed=3)
        b, _, _ = _run(tmp_path / "b", seed=3)
        c, _, _ = _run(tmp_path / "c", seed=4)
        strip = lambda x: {**x.__dict__, "transcript_path": None}
        assert strip(a) == strip(b)
        assert a.training != c.training

    def test_restores_vendor_llm_client(self, tmp_path):
        from scienceagent import llm_client
        before = llm_client.complete
        _run(tmp_path)
        assert llm_client.complete is before

    def test_restores_vendor_llm_client_on_error(self, tmp_path):
        from scienceagent import llm_client
        before = llm_client.complete

        def boom(**_):
            raise RuntimeError("provider down")

        with pytest.raises(RuntimeError, match="provider down"):
            _run(tmp_path, llm=boom)
        assert llm_client.complete is before

    def test_rejects_unknown_world_and_bad_seed(self, tmp_path):
        with pytest.raises(ValueError):
            _run(tmp_path, world="ether2")
        with pytest.raises(TypeError):
            _run(tmp_path, seed=None)

    def test_multi_particle_world_runs(self, tmp_path):
        llm = ScriptedLLM(["no tags"], p_reply="")
        a, wallet, _ = _run(tmp_path, llm=llm, world="circle", max_rounds=2)
        assert a.submitted_law is None and a.lab_cost == 0 and wallet.balance == 10.0

    def test_vendor_tree_clean(self, tmp_path):
        _run(tmp_path)
        out = subprocess.run(["git", "-C", str(ROOT / "vendor" / "discovery-agents"),
                              "status", "--porcelain"], capture_output=True, text=True,
                             check=True).stdout
        assert out.strip() == ""


# --------------------------------------------------------------------- usage / live guard

class TestUsageAndLiveGuard:

    def test_parse_p(self):
        assert parse_p("<p_success> 0.25 </p_success>") == 0.25
        assert parse_p("<p_success>1</p_success>") == 1.0
        assert parse_p(None) is None and parse_p("<p_success>2</p_success>") is None

    def test_usd(self):
        assert usd_for("claude-opus-5-5", 1_000_000, 1_000_000) == 24.0
        assert usd_for("claude-haiku-4-5", 2_000_000, 0) == 2.0
        assert usd_for("fake-model", 10, 10) is None

    def test_meter_prices_reported_usage(self):
        m = UsageMeter(lambda **_: LLMReply("hi", 1000, 100))
        m(model="claude-sonnet-5-5", messages=[{"role": "user", "content": "x"}])
        assert m.usage() == {"calls": 1, "input_tokens": 1000, "output_tokens": 100,
                             "usd": pytest.approx(0.003), "estimated": False}

    def test_cap_refuses_before_calling(self):
        called = []
        cap = SpendCap(max_usd=0.01)
        m = UsageMeter(lambda **_: called.append(1) or "x", cap=cap)
        with pytest.raises(BudgetExceeded):
            m(model="claude-opus-5-5", messages=[{"role": "user", "content": "x"}],
              max_tokens=8192)  # worst case 8192 × $20/M = $0.16
        assert called == []

    def test_cap_refuses_unpriced_model(self):
        m = UsageMeter(lambda **_: "x", cap=SpendCap(max_usd=100))
        with pytest.raises(BudgetExceeded, match="no price"):
            m(model="mystery", messages=[{"role": "user", "content": "x"}])

    def test_cap_accumulates_across_meters(self):
        cap = SpendCap(max_usd=1.0)
        for _ in range(2):
            UsageMeter(lambda **_: LLMReply("x", 10_000, 1000), cap=cap)(
                model="claude-haiku-4-5", messages=[{"role": "user", "content": "x"}],
                max_tokens=1000)
        assert cap.spent_usd == pytest.approx(2 * 0.015)

    @pytest.mark.parametrize("env", [{}, {"ENABLE_LIVE": "1"}, {"DM_MAX_USD": "5"},
                                     {"ENABLE_LIVE": "1", "DM_MAX_USD": "lots"}])
    def test_live_needs_both_variables(self, monkeypatch, env, tmp_path):
        monkeypatch.delenv("ENABLE_LIVE", raising=False)
        monkeypatch.delenv("DM_MAX_USD", raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        with pytest.raises(LiveDisabled):
            live_llm("claude-haiku-4-5")
        with pytest.raises(LiveDisabled):
            run_attempt("claude-haiku-4-5", "gravity", 0, Wallet("a", 10), 0.5,
                        transcript_dir=tmp_path)

    def test_live_on_selects_clients_without_calling(self, monkeypatch):
        from dm import llm as dmllm
        monkeypatch.setenv("ENABLE_LIVE", "1")
        monkeypatch.setenv("DM_MAX_USD", "1")
        assert live_llm("claude-haiku-4-5") is dmllm.anthropic_llm
        from scienceagent import llm_client
        assert live_llm("openrouter/x") is llm_client.complete

    def test_venue_module_imports_no_api_client(self):
        code = ("import sys, dm.venues.discoverphysics, dm.llm;"
                "print('anthropic' in sys.modules)")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             cwd=ROOT, check=True).stdout.strip()
        assert out == "False"


def test_every_world_gets_its_own_prompt_not_the_generic_fallback(tmp_path):
    """From the repo root the vendor loader cannot find relative prompt paths and falls back
    to a one-line prompt; the wrapper must hand it absolute paths."""
    from scienceagent.agent import _load_system_prompt
    from scienceagent.worlds import get_world

    from dm.venues.discoverphysics import WORLDS, prompt_path

    for world in WORLDS:
        w = get_world(world, engine="nbody", noise_std=0.0, noise_seed=0)
        text = _load_system_prompt(prompt_path(w["system_prompt"]), prompt_path(w["instructions"]))
        assert "{{world_instructions}}" not in text
        assert not text.startswith("You are a scientific discovery agent. Design experiments")
        assert len(text) > 3000, world

    a, _, _ = _run(tmp_path, world="ether", seed=0)
    system = json.loads((tmp_path / "fake-1overr_ether_s0.json").read_text())["system"]
    assert "probe_positions" in system  # ether's experiment format reached the model
