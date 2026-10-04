"""Bounty benchmark (poc/): pricing, reply parsing, the agent loop, resolution, public record.

No API calls (scripted LLM) and no simulator (stub executor)."""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from poc import config as C
from poc import protocol
from poc.fake_llm import ScriptedLLM
from poc.pricing import Account, experiment_price

CFG = C.load()
HYP = CFG.hypothesis("gravity-inverse-square")
EXP = {"p1": 2.0, "p2": 1.0, "pos2": [3.0, 0.0], "velocity2": [0.0, 0.0],
       "measurement_times": [1.0, 2.0, 3.0], "duration": 5.0}


def conf(p="0.6"):
    return (f"<assessment>looks like 1/r</assessment><p_success>{p}</p_success>"
            "<planned_cost>20</planned_cost>")


def run_exp(exp=EXP):
    import json
    return conf() + "<run_experiment>" + json.dumps([exp]) + "</run_experiment>"


VERDICT = (conf("0.8") + "<verdict>refuted</verdict><estimate>n = 1.0 ± 0.05\na3 = 0.053</estimate>"
           "<evidence>n = 1.0 ± 0.05</evidence>")


class StubExecutor:
    def __init__(self):
        self.calls = []

    def run(self, exps):
        self.calls.append(exps)
        return [{"pos2": [[3.0, 0.0]] * len(e["measurement_times"])} for e in exps]


def make_agent(replies, cfg=CFG, ledger=(), on_round=None):
    from poc.agent import MarketAgent
    vendor = C.VENDOR_ROOT / "PhysicsSchool" / "prompts"
    llm = ScriptedLLM(replies)
    account = Account(agent="m", hypothesis=HYP.id, budget=cfg.budget)
    agent = MarketAgent(
        cfg=cfg, hyp=HYP, account=account, ledger_entries=list(ledger), complete=llm,
        on_round=on_round,
        model="m", executor=StubExecutor(), mission="mission", verbose=False,
        system_prompt_path=str(vendor / "_template_interactive.md"),
        instructions_path=str(vendor / "2particle_instructions.md"),
        law_stub="def discovered_law(...):\n    pass\n", experiment_format="<run_experiment>[]</run_experiment>",
        trajectory_logger=None,
    )
    return agent, account, llm


# ------------------------------------------------------------------ config and pricing

def test_config_every_hypothesis_has_criteria_answer_prize_quantities_and_rule():
    assert len(CFG.hypotheses) == 8
    for h in CFG.hypotheses:
        assert h.resolution_criteria and h.answer in C.ANSWERS and h.prize > 0
        assert h.quantities and all(q.tolerance > 0 for q in h.quantities)
        assert h.supported_if.quantity in {q.name for q in h.quantities}
    answers = [h.answer for h in CFG.hypotheses]
    assert answers.count("supported") == answers.count("refuted")


def test_experiment_price_itemised():
    price, b = experiment_price(EXP, CFG, HYP)
    c = CFG.experiment_costs
    expected = (c["per_experiment"] + 3 * c["per_measurement"] + 5 * c["per_time_unit"]
                + 1 * c["per_particle"] + 1 * c["per_custom_property"])
    assert price == pytest.approx(expected)
    assert b["units"]["per_custom_property"] == 1  # p1 = 2, p2 = default


def test_price_counts_probes_and_uses_last_time_without_duration():
    exp = {"probe_positions": [[0, 1]] * 5, "probe_velocities": [[0, 0]] * 5,
           "probe_masses": [1, 2, 1, 1, 4], "measurement_times": [2.0, 7.0]}
    _, b = experiment_price(exp, CFG, CFG.hypothesis("ether-outward-push"))
    assert b["units"]["per_particle"] == 5
    assert b["units"]["per_time_unit"] == 7.0
    assert b["units"]["per_custom_property"] == 2


# ------------------------------------------------------------------ parsing

@pytest.mark.parametrize("raw,want", [("0.7", 0.7), ("70%", 0.7), (" 1 ", 1.0), ("1.5", None),
                                      ("likely", None), ("-0.1", None)])
def test_parse_p(raw, want):
    assert protocol.parse_p(f"<p_success>{raw}</p_success>") == want


def test_parse_verdict_and_withdraw():
    assert protocol.parse_verdict("<verdict> Refuted </verdict>") == "refuted"
    assert protocol.parse_verdict("<verdict>probably</verdict>") == "inconclusive"
    assert protocol.parse_verdict("no tag") is None
    assert protocol.parse_withdraw("<withdraw/>") == ""
    assert protocol.parse_withdraw("<withdraw>too costly</withdraw>") == "too costly"
    assert protocol.parse_withdraw("nothing") is None


def test_prompt_shows_hypothesis_criteria_prize_costs_not_answer():
    block = protocol.market_block(CFG, HYP)
    assert HYP.hypothesis in block and HYP.resolution_criteria in block
    assert f"{HYP.prize:g} credits" in block and "per measurement time" in block
    assert "answer" not in block.lower()


# ------------------------------------------------------------------ agent loop

def test_experiment_then_verdict_charges_rounds_and_experiments():
    agent, account, _ = make_agent([run_exp(), VERDICT])
    assert agent.run() == "refuted"
    price, _ = experiment_price(EXP, CFG, HYP)
    assert account.spent == pytest.approx(2 * CFG.round_fee + price)
    assert account.lab_revenue() == pytest.approx(account.spent)  # conservation
    assert agent.outcome == "verdict" and agent.evidence == "n = 1.0 ± 0.05"
    assert [e["p_success"] for e in agent.conversation_log] == [0.6, 0.8]
    assert agent.executor.experiments == 1


def test_running_estimates_are_recorded_every_round():
    from poc.attempt import round_log
    agent, _, _ = make_agent([run_exp() + "<estimate>n = 1.6 ± 0.5</estimate>", run_exp(), VERDICT])
    agent.run()
    log = round_log(agent.conversation_log)
    assert [e["estimates"].get("n", {}).get("value") for e in log] == [1.6, None, 1.0]
    assert log[2]["estimates"]["a3"]["value"] == pytest.approx(0.053)


def test_round_callback_fires_once_after_each_round():
    rounds = []
    agent, _, _ = make_agent([run_exp(), VERDICT], on_round=lambda entry: rounds.append(entry["round"]))

    agent.run()

    assert rounds == [1, 2]


def test_round_callback_errors_do_not_change_the_run(capsys):
    def fail(_entry):
        raise RuntimeError("callback failed")

    agent, _, _ = make_agent([run_exp(), VERDICT], on_round=fail)

    assert agent.run() == "refuted"
    assert capsys.readouterr().err.count("on_round callback failed") == 2


def test_walking_away_in_first_reply_is_free():
    agent, account, _ = make_agent([conf("0.1") + "<withdraw>break-even is 0.3</withdraw>"])
    assert agent.run() is None
    assert agent.outcome == "walked_away" and account.spent == 0
    assert agent.withdraw_reason == "break-even is 0.3"


def test_later_withdrawal_pays_its_round():
    agent, account, _ = make_agent([run_exp(), conf("0.1") + "<withdraw>too costly</withdraw>"])
    agent.run()
    price, _ = experiment_price(EXP, CFG, HYP)
    assert agent.outcome == "withdrawn"
    assert account.spent == pytest.approx(2 * CFG.round_fee + price)


def test_missing_p_reprompts_once_then_records_none():
    agent, _, llm = make_agent([run_exp(), "<run_mse_fit>x</run_mse_fit>", "still nothing", VERDICT])
    agent.run()
    second = agent.conversation_log[1]
    assert "confidence_reprompt" in second and second["p_success"] is None
    assert agent.conversation_log[2]["p_success"] == 0.8
    assert len(llm.calls) == 4  # round 1, round 2, its re-prompt, round 3


def test_over_budget_batch_is_refused_and_not_charged():
    cfg = dataclasses.replace(CFG, budget=CFG.round_fee + 1)
    agent, account, _ = make_agent([run_exp(), VERDICT], cfg=cfg)
    agent.run()
    assert agent.executor.inner.calls == []
    assert "refused" in agent.conversation_log[0]["experiment_error"]
    assert any(e["type"] == "insufficient_budget" for e in account.events)


def test_out_of_rounds():
    cfg = dataclasses.replace(CFG, max_rounds=2)
    agent, account, _ = make_agent([run_exp()], cfg=cfg)
    agent.run()
    assert agent.outcome == "out_of_rounds" and len(agent.conversation_log) == 2


def test_public_record_is_in_the_prompt():
    entry = {"id": "x", "model": "other", "outcome": "verdict", "rounds": 3, "experiments": 2,
             "spent": 70, "p_success": 0.9, "withdraw_reason": None, "data": "[...]"}
    agent, _, llm = make_agent([VERDICT], ledger=[entry])
    agent.run()
    assert "other: no clear result after 3 round(s)" in llm.calls[0]["system"]


# ------------------------------------------------------------------ resolution and record

def _record(outcome, verdict, passed, hid=HYP.id):
    return {"attempt_id": "a", "solver": "m", "rounds": 2, "experiments": 1, "lab_cost": 30,
            "verdict": {"passed": passed, "agent_verdict": verdict, "answer": "refuted"},
            "extra": {"hypothesis_id": hid, "outcome": outcome, "agent_verdict": verdict,
                      "evidence": "secret reasoning", "final_assessment": "it is refuted",
                      "final_p": 0.7, "withdraw_reason": None,
                      "runs": [{"input": EXP, "output": {"pos2": [[1, 2]]}}]}}


def test_ledger_entry_hides_conclusions_and_answer():
    e = protocol.ledger_entry(_record("verdict", "supported", False), 6000)
    text = str(e) + protocol.ledger_block([e])
    for leak in ("supported", "refuted", "secret reasoning", "it is refuted"):
        assert leak not in text
    assert '"pos2"' in e["data"]


def test_resolve_pays_only_checked_claims_but_naive_pays_any_clear_verdict():
    from dm.types import SubmittedAttempt
    from poc import truth
    from poc.bench import resolve

    good = {k: {"value": v, "sigma": None} for k, v in truth.true_values(HYP).items()}
    wrong = {k: {"value": v * 3, "sigma": None} for k, v in truth.true_values(HYP).items()}

    def sub(outcome, verdict, est=None):
        events = [{"type": "round_charged", "amount": 10, "round": 1}]
        return SubmittedAttempt(source="live", protocol=C.PROTOCOL, venue=C.VENUE, world=HYP.world,
                                solver="m", seed=0, stated_p_success=0.5, rounds=1, experiments=0,
                                lab_cost=10, extra={"hypothesis_id": HYP.id, "outcome": outcome,
                                                    "agent_verdict": verdict, "estimates": est or {},
                                                    "account_events": events, "bid_p": 0.5})
    ok = resolve(HYP, sub("verdict", HYP.answer, good))
    assert ok.passed and ok.extra["prize_paid"] == HYP.prize
    assert ok.extra["settlements"]["market"]["profit"] == pytest.approx(HYP.prize - 10)
    bad = resolve(HYP, sub("verdict", HYP.answer, wrong))
    assert not bad.passed and bad.verdict["outcome"] == "false_claim"
    assert bad.extra["settlements"]["naive"]["profit"] == pytest.approx(HYP.prize - 10)
    assert bad.extra["settlements"]["market"]["profit"] == pytest.approx(
        -10 - CFG.claim_bond * HYP.prize)
    assert not resolve(HYP, sub("verdict", "inconclusive")).passed
    assert not resolve(HYP, sub("walked_away", None)).passed


def test_public_record_lists_only_failures_of_that_hypothesis():
    from dm.types import AttemptRecord
    from poc.bench import public_record

    def rec(i, passed, hid):
        return AttemptRecord(attempt_id=str(i), source="live", protocol=C.PROTOCOL, venue=C.VENUE,
                             world="w", solver="m", seed=i, stated_p_success=None, rounds=1,
                             experiments=0, lab_cost=10, verdict={"passed": passed},
                             extra={"hypothesis_id": hid, "outcome": "verdict"})
    rs = [rec(0, False, HYP.id), rec(1, True, HYP.id), rec(2, False, "other")]
    assert [e["id"] for e in public_record(rs, HYP, CFG)] == ["0"]


def test_poc_never_imports_the_oracle():
    for path in Path(C.ROOT / "poc").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            assert not any(n.startswith(("dm.oracle", "dm.settle")) for n in names), path


def test_learning_reading_recovers_pull_from_a_drop_at_rest():
    from poc.animate import _reading
    t = [0.5, 1.0, 1.5, 2.0]
    run = {"input": {"p1": 2.0, "p2": 1.0, "pos2": [0.0, 4.0], "velocity2": [0.0, 0.0]},
           "output": {"measurement_times": t, "pos2": [[0.0, 4.0 - 0.5 * 0.06 * s * s] for s in t]}}
    assert _reading(run) == {"r": 4.0, "a": pytest.approx(0.03)}  # 0.06 pull at p1/p2 = 2
    moving = {**run, "input": {**run["input"], "velocity2": [0.1, 0.0]}}
    assert _reading(moving) is None
