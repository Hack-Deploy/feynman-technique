"""Bounty benchmark (poc/): pricing, reply parsing, the agent loop, resolution, public record.

No API calls (scripted LLM) and no simulator (stub executor)."""

from __future__ import annotations

import ast
from copy import deepcopy
import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dm.types import SubmittedAttempt
from poc import config as C
from poc import bench
from poc import protocol
from poc.fake_llm import ScriptedLLM
from poc.pricing import Account, experiment_price

CFG = C.load()
HYP = CFG.hypothesis("gravity-inverse-square")
EXP = {"p1": 2.0, "p2": 1.0, "pos2": [3.0, 0.0], "velocity2": [0.0, 0.0],
       "measurement_times": [1.0, 2.0, 3.0], "duration": 5.0}


def conf(p="0.6"):
    return f"<assessment>looks like 1/r</assessment><p_success>{p}</p_success>"


def run_exp(exp=EXP):
    import json
    return conf() + "<run_experiment>" + json.dumps([exp]) + "</run_experiment>"


VERDICT = conf("0.8") + "<verdict>refuted</verdict><evidence>n = 1.0 ± 0.05</evidence>"


class StubExecutor:
    def __init__(self):
        self.calls = []

    def run(self, exps):
        self.calls.append(exps)
        return [{"pos2": [[3.0, 0.0]] * len(e["measurement_times"])} for e in exps]


class FixedResultExecutor:
    def __init__(self, output):
        self.output = output

    def run(self, exp_input):
        return deepcopy(self.output)


def make_metered_executor(inner, cfg=CFG, noise_seed=0):
    from poc.agent import MeteredExecutor

    account = Account(agent="m", hypothesis=HYP.id, budget=cfg.budget)
    executor = MeteredExecutor(
        inner, cfg, HYP, account, cfg.velocity_noise_std, noise_seed
    )
    return executor, account


def make_agent(replies, cfg=CFG, ledger=(), on_round=None, buy_record_first=False):
    from poc.agent import MarketAgent
    vendor = C.VENDOR_ROOT / "PhysicsSchool" / "prompts"
    llm = ScriptedLLM(replies, buy_record_first=buy_record_first)
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

def test_config_every_hypothesis_has_criteria_answer_and_prize():
    assert len(CFG.hypotheses) == 11
    for h in CFG.hypotheses:
        assert h.resolution_criteria and h.answer in C.ANSWERS and h.prize > 0


def test_ledger_read_fee_defaults_and_rejects_negative(tmp_path):
    assert CFG.ledger_read_fee == 30
    original = C.CONFIG_PATH.read_text()
    default_path = tmp_path / "default-fee.yaml"
    default_path.write_text("\n".join(
        line for line in original.splitlines() if not line.strip().startswith("read_fee:")
    ))
    assert C.load(default_path).ledger_read_fee == 30
    path = tmp_path / "config.yaml"
    path.write_text(original.replace("read_fee: 30", "read_fee: -1"))
    with pytest.raises(ValueError, match="ledger.read_fee must be a finite number >= 0"):
        C.load(path)


def test_velocity_noise_config_defaults_to_zero_and_loads_from_real_config(tmp_path):
    assert CFG.velocity_noise_std == pytest.approx(0.05)
    original = C.CONFIG_PATH.read_text()
    path = tmp_path / "without-velocity-noise.yaml"
    path.write_text("\n".join(
        line for line in original.splitlines()
        if not line.strip().startswith("velocity_noise_std:")
    ))
    assert C.load(path).velocity_noise_std == 0.0


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


@pytest.mark.parametrize("raw", ["<buy_record/>", "<buy_record></buy_record>"])
def test_parse_buy_record(raw):
    assert protocol.parse_buy_record(raw)
    assert not protocol.parse_buy_record("<run_experiment/>")


def test_prompt_shows_hypothesis_criteria_prize_costs_not_answer():
    block = protocol.market_block(CFG, HYP)
    assert HYP.hypothesis in block and HYP.resolution_criteria in block
    assert f"{HYP.prize:g} credits" in block and "per measurement time" in block
    assert "public-record fee also counts as cost" in block
    assert "<buy_record/>" in block
    assert "each observed position has independent Gaussian noise σ = 0.075" in block
    assert "each observed velocity has independent Gaussian noise σ = 0.05" in block
    assert (
        "Results can be inconclusive. You may repeat any experiment (identical input, "
        "in the same or a later round) to get a fresh, independent noisy reading; "
        "each repeat is charged at the full price."
    ) in block
    assert "answer" not in block.lower()


def test_prompt_omits_zero_velocity_noise_and_repeat_option_without_noise():
    position_only = protocol.market_block(
        dataclasses.replace(CFG, velocity_noise_std=0.0), HYP
    )
    assert "each observed position has independent Gaussian noise σ = 0.075" in position_only
    assert "each observed velocity has independent Gaussian noise" not in position_only
    assert "You may repeat any experiment" in position_only

    no_noise = protocol.market_block(
        dataclasses.replace(CFG, noise_std=0.0, velocity_noise_std=0.0), HYP
    )
    assert "You may repeat any experiment" not in no_noise


def test_metered_executor_zero_velocity_noise_returns_results_unchanged():
    expected = [
        {
            "pos1": [[1.0, 2.0]],
            "pos2": [[3.0, 4.0]],
            "velocity1": [[0.1, 0.2]],
            "velocity2": [[0.3, 0.4]],
            "velocities": [[[0.5, 0.6]]],
            "background_initial_velocities": [[7.0, 8.0]],
        },
        {"error": "experiment failed"},
        "not a result object",
    ]
    cfg = dataclasses.replace(CFG, velocity_noise_std=0.0)
    inner = FixedResultExecutor(expected)
    metered, _ = make_metered_executor(inner, cfg)
    rng_state = deepcopy(metered._vel_rng.bit_generator.state)

    actual = metered.run([EXP])

    assert actual == expected
    assert inner.output == expected
    assert metered._vel_rng.bit_generator.state == rng_state


def test_metered_executor_velocity_noise_is_seeded_and_leaves_other_fields_alone():
    expected = [
        {
            "pos1": [[1.0, 2.0]],
            "pos2": [[3.0, 4.0]],
            "positions": [[[5.0, 6.0]]],
            "velocity1": [[0.1, 0.2]],
            "velocity2": [[0.3, 0.4]],
            "velocities": [[[0.5, 0.6]]],
            "background_initial_velocities": [[7.0, 8.0]],
            "error": None,
        },
        {"error": "experiment failed"},
        "not a result object",
    ]

    def run(seed):
        metered, _ = make_metered_executor(FixedResultExecutor(expected), noise_seed=seed)
        return metered.run([EXP])

    first = run(42)
    repeated = run(42)
    different_seed = run(43)

    assert first == repeated
    assert first != different_seed
    for key in ("velocity1", "velocity2", "velocities"):
        assert first[0][key] != expected[0][key]
    for key in ("pos1", "pos2", "positions", "background_initial_velocities", "error"):
        assert first[0][key] == expected[0][key]
    assert first[1:] == expected[1:]


def test_repeated_experiments_in_one_batch_get_fresh_readings_and_full_price():
    from poc.agent import MeteredExecutor
    from scienceagent.worlds import get_world

    seed = 18
    world = get_world(
        HYP.world, engine=C.ENGINE, noise_std=CFG.noise_std, noise_seed=seed
    )
    account = Account(agent="m", hypothesis=HYP.id, budget=CFG.budget)
    metered = MeteredExecutor(
        world["executor"], CFG, HYP, account, CFG.velocity_noise_std, seed
    )

    results = metered.run([deepcopy(EXP), deepcopy(EXP)])

    assert not np.array_equal(results[0]["pos2"], results[1]["pos2"])
    assert not np.array_equal(results[0]["velocity2"], results[1]["velocity2"])
    price, _ = experiment_price(EXP, CFG, HYP)
    assert account.spent == pytest.approx(2 * price)


def test_repeated_experiment_across_calls_gets_fresh_readings_and_full_price():
    from poc.agent import MeteredExecutor
    from scienceagent.worlds import get_world

    seed = 19
    world = get_world(
        HYP.world, engine=C.ENGINE, noise_std=CFG.noise_std, noise_seed=seed
    )
    account = Account(agent="m", hypothesis=HYP.id, budget=CFG.budget)
    metered = MeteredExecutor(
        world["executor"], CFG, HYP, account, CFG.velocity_noise_std, seed
    )

    first = metered.run([deepcopy(EXP)])[0]
    second = metered.run([deepcopy(EXP)])[0]

    assert not np.array_equal(first["pos2"], second["pos2"])
    assert not np.array_equal(first["velocity2"], second["velocity2"])
    price, _ = experiment_price(EXP, CFG, HYP)
    assert account.spent == pytest.approx(2 * price)


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


def test_round_callback_fires_once_after_each_round():
    rounds = []
    agent, _, _ = make_agent([run_exp(), VERDICT], on_round=lambda entry: rounds.append(entry["round"]))

    agent.run()

    assert rounds == [1, 2]


def test_penultimate_round_warning_allows_experiments():
    cfg = dataclasses.replace(CFG, max_rounds=3)
    agent, _, _ = make_agent([run_exp(), run_exp(), VERDICT], cfg=cfg)

    agent.run()

    assert "You may still run an experiment" in agent.conversation_log[1]["system_message"]


def test_empty_no_tag_reply_gets_cutoff_feedback():
    agent, _, _ = make_agent(["", VERDICT])

    agent.run()

    assert agent.conversation_log[0]["system_message"].startswith(
        "ERROR: your reply was empty"
    )


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
    agent, _, llm = make_agent(["<run_mse_fit>x</run_mse_fit>", "still nothing", VERDICT])
    agent.run()
    first = agent.conversation_log[0]
    assert "confidence_reprompt" in first and first["p_success"] is None
    assert agent.conversation_log[1]["p_success"] == 0.8
    assert len(llm.calls) == 3  # round 1, its re-prompt, round 2


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


def test_prompt_offers_record_without_showing_its_data():
    entry = {"id": "x", "model": "other", "outcome": "verdict", "rounds": 3, "experiments": 2,
             "spent": 70, "p_success": 0.9, "withdraw_reason": None,
             "data": "[RAW_DATA_SENTINEL]"}
    agent, _, llm = make_agent([VERDICT], ledger=[entry])
    agent.run()
    system = llm.calls[0]["system"]
    assert "PUBLIC RECORD: 1 earlier run" in system
    assert "30 credits" in system
    assert "other: no clear result" not in system
    assert "RAW_DATA_SENTINEL" not in system


def _ledger_entry():
    return {
        "id": "earlier-run",
        "model": "other",
        "outcome": "withdrawn",
        "rounds": 2,
        "experiments": 1,
        "spent": 15,
        "p_success": 0.4,
        "withdraw_reason": "not enough evidence",
        "data": '[{"output":"RAW_DATA_SENTINEL"}]',
    }


def test_buy_record_is_paid_once_and_revealed_without_using_a_round():
    ledger = [_ledger_entry()]
    agent, account, llm = make_agent(
        [VERDICT], ledger=ledger, buy_record_first=True
    )

    assert agent.run() == "refuted"

    entry = agent.conversation_log[0]
    record_charge = next(event for event in account.events if event["type"] == "record_charged")
    assert record_charge["to"] == "market"
    assert record_charge["amount"] == CFG.ledger_read_fee
    assert record_charge["detail"] == {"entries": 1}
    assert account.spent == pytest.approx(CFG.ledger_read_fee + CFG.round_fee)
    assert account.lab_revenue() == pytest.approx(CFG.round_fee)
    assert sum(event["amount"] for event in account.events if event["to"] == "market") == pytest.approx(
        CFG.ledger_read_fee
    )
    assert account.lab_revenue() + sum(
        event["amount"] for event in account.events if event["to"] == "market"
    ) == pytest.approx(account.spent)
    assert entry["record_bought"] is True
    assert "<buy_record/>" in entry["buy_reply"]
    assert len(agent.conversation_log) == 1
    assert len(llm.calls) == 2
    notice = llm.calls[1]["messages"][-1]["content"]
    assert "other: withdrew after 2 round(s)" in notice
    assert "RAW_DATA_SENTINEL" in notice
    assert f"Record bought for {CFG.ledger_read_fee:g} credits" in notice


def test_buy_then_walk_away_only_pays_the_record_fee():
    reply = conf("0.2") + "<withdraw>not worth it</withdraw>"
    agent, account, _ = make_agent(
        [reply], ledger=[_ledger_entry()], buy_record_first=True
    )

    assert agent.run() is None

    assert agent.outcome == "walked_away"
    assert account.spent == pytest.approx(CFG.ledger_read_fee)
    assert [event["amount"] for event in account.events if event["type"] == "round_charged"] == [0]


def test_followup_buy_tag_is_ignored_and_does_not_charge_again():
    second_buy_and_verdict = (
        conf("0.9") + "<buy_record/>"
        "<verdict>refuted</verdict><evidence>Offline evidence.</evidence>"
    )
    agent, account, llm = make_agent(
        [second_buy_and_verdict], ledger=[_ledger_entry()], buy_record_first=True
    )

    assert agent.run() == "refuted"

    assert len(llm.calls) == 2
    assert sum(event["type"] == "record_charged" for event in account.events) == 1
    assert agent.conversation_log[0]["record_bought"] is True


def test_already_bought_record_request_is_explained_and_reasked():
    cfg = dataclasses.replace(CFG, max_rounds=2)
    second_buy = conf("0.7") + "<buy_record/>"
    agent, account, llm = make_agent(
        [run_exp(), second_buy, VERDICT],
        cfg=cfg,
        ledger=[_ledger_entry()],
        buy_record_first=True,
    )

    assert agent.run() == "refuted"

    assert sum(event["type"] == "record_charged" for event in account.events) == 1
    assert agent.conversation_log[1]["record_bought"] is False
    assert "<buy_record/>" in agent.conversation_log[1]["buy_reply"]
    assert "already bought" in llm.calls[3]["messages"][-1]["content"]


def test_buy_record_with_no_entries_is_not_charged():
    agent, account, llm = make_agent([VERDICT], ledger=[], buy_record_first=True)

    assert agent.run() == "refuted"

    assert not agent.conversation_log[0]["record_bought"]
    assert not any(event["type"] == "record_charged" for event in account.events)
    assert account.spent == pytest.approx(CFG.round_fee)
    assert "nothing to buy" in llm.calls[1]["messages"][-1]["content"]


def test_buy_record_is_not_charged_when_unaffordable():
    cfg = dataclasses.replace(CFG, budget=CFG.ledger_read_fee - 1)
    agent, account, llm = make_agent(
        [VERDICT], cfg=cfg, ledger=[_ledger_entry()], buy_record_first=True
    )

    assert agent.run() == "refuted"

    assert not agent.conversation_log[0]["record_bought"]
    assert not any(event["type"] == "record_charged" for event in account.events)
    assert account.spent == pytest.approx(cfg.round_fee)
    assert "cannot afford" in llm.calls[1]["messages"][-1]["content"]


@pytest.mark.parametrize("bought,fee,seen", [
    (True, CFG.ledger_read_fee, ["earlier-run"]),
    (False, 0, []),
])
def test_attempt_record_tracks_public_record_purchase(
    monkeypatch, tmp_path, bought, fee, seen
):
    import scienceagent.trajectory_logger
    from poc import attempt

    class FakeAgent:
        def __init__(self, **_kwargs):
            self._system = "system"
            self.outcome = "walked_away"
            self.verdict = None
            self.evidence = None
            self.withdraw_reason = "not worth it"
            self.executor = SimpleNamespace(experiments=0)
            self.record_bought = bought
            self.record_fee = fee
            self.conversation_log = [{
                "round": 1,
                "action": "walk_away",
                "assessment": "not worth it",
                "p_success": 0.2,
                "record_bought": bought,
                "round_fee": 0,
                "experiments_cost": 0,
                "spent_so_far": fee,
            }]

        def run(self):
            return None

    class FakeLogger:
        def __init__(self, **_kwargs):
            pass

    monkeypatch.setattr(attempt, "MarketAgent", FakeAgent)
    monkeypatch.setattr(scienceagent.trajectory_logger, "TrajectoryLogger", FakeLogger)
    monkeypatch.setattr(C, "TRAJECTORIES_DIR", tmp_path / "trajectories")
    monkeypatch.setattr(C, "TRANSCRIPTS_DIR", tmp_path / "transcripts")
    world_spec = {
        "executor": StubExecutor(),
        "mission": "mission",
        "system_prompt": "prompt.md",
        "instructions": "instructions.md",
        "law_stub": "",
        "experiment_format": "",
    }

    submitted = attempt.run_attempt(
        "m", HYP.id, 0, [_ledger_entry()], cfg=CFG, world_spec=world_spec
    )

    assert submitted.extra["ledger_seen"] == seen
    assert submitted.extra["record_bought"] is bought
    assert submitted.extra["record_fee"] == fee


def test_bench_fake_stops_after_claim_is_solved(monkeypatch, tmp_path):
    calls = []

    def fake_attempt(model, hypothesis_id, seed, *_args, **_kwargs):
        calls.append((model, hypothesis_id, seed))
        hyp = CFG.hypothesis(hypothesis_id)
        return SubmittedAttempt(
            source="live",
            protocol=C.PROTOCOL,
            venue=C.VENUE,
            world=hyp.world,
            solver=model,
            seed=seed,
            stated_p_success=0.9,
            rounds=1,
            experiments=0,
            lab_cost=CFG.round_fee,
            extra={
                "hypothesis_id": hypothesis_id,
                "outcome": "verdict",
                "agent_verdict": hyp.answer,
            },
        )

    monkeypatch.setattr(bench, "run_attempt", fake_attempt)
    store = tmp_path / "attempts.jsonl"
    bench.main([
        "--fake", "--hypotheses", HYP.id,
        "--models", "model-a", "model-b",
        "--seeds", "0", "1",
        "--store", str(store),
    ])

    assert calls == [("model-a", HYP.id, 0)]
    assert len(bench.AttemptStore(store).load()) == 1


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


def test_resolve():
    from dm.types import SubmittedAttempt
    from poc.bench import resolve

    def sub(outcome, verdict):
        return SubmittedAttempt(source="live", protocol=C.PROTOCOL, venue=C.VENUE, world=HYP.world,
                                solver="m", seed=0, stated_p_success=0.5, rounds=1, experiments=0,
                                lab_cost=10, extra={"hypothesis_id": HYP.id, "outcome": outcome,
                                                    "agent_verdict": verdict})
    assert resolve(HYP, sub("verdict", HYP.answer)).passed
    assert resolve(HYP, sub("verdict", HYP.answer)).extra["prize_paid"] == HYP.prize
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
