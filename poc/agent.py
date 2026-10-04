"""The vendor DiscoveryAgent turned into a bounty hunter: it settles a posted hypothesis
(supported / refuted) instead of submitting a law, pays per round and per experiment, states its
assessment and p_success every round, may walk away or withdraw, and sees the public record of
failed runs.

The vendor loop (``DiscoveryAgent.run``) cannot be extended from outside, so ``run`` is
re-implemented here from vendor commit 450818fa. Critic, random-experiment and no-MSE modes
are not part of this protocol and are left out. Vendor code is not edited.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from typing import Callable, Optional

from scienceagent import llm_client
from scienceagent.agent import (
    DiscoveryAgent, _MSE_FIT_PROMPT_BLOCK, _compact_json, _extract_tag, _join_sys,
    _load_system_prompt,
)

from poc import protocol
from poc.config import Config, Hypothesis
from poc.pricing import Account, OverBudget, experiment_price

Complete = Callable[..., str]


class MeteredExecutor:
    """Prices a batch of experiments, refuses it if over budget, charges it only if it runs."""

    def __init__(self, inner, cfg: Config, hyp: Hypothesis, account: Account):
        self.inner = inner
        self.cfg = cfg
        self.hyp = hyp
        self.account = account
        self.round_num = 0
        self.round_cost = 0.0
        self.experiments = 0

    def __getattr__(self, name):  # anything else is the real executor's
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    def run(self, exp_input):
        if not isinstance(exp_input, list):
            raise ValueError("<run_experiment> must contain a JSON list of experiments")
        priced = [experiment_price(inp, self.cfg, self.hyp) for inp in exp_input]
        total = round(sum(p for p, _ in priced), 6)
        # This round's fee is charged when the round ends, so reserve it now; otherwise
        # a batch that just fits the budget pushes total spend over it.
        if not self.account.can_afford(round(total + self.cfg.round_fee, 6)):
            self.account.refuse(total, self.round_num, len(exp_input))
            raise OverBudget(total, self.account.remaining)
        results = self.inner.run(exp_input)
        self.account.charge("experiments_charged", total, self.round_num,
                            {"count": len(exp_input), "items": [b for _, b in priced]})
        self.round_cost = round(self.round_cost + total, 6)
        self.experiments += len(exp_input)
        return results


class MarketAgent(DiscoveryAgent):
    def __init__(self, *, cfg: Config, hyp: Hypothesis, account: Account,
                 ledger_entries: list[dict], complete: Optional[Complete] = None,
                 on_round: Callable[[dict], None] | None = None, **kwargs):
        # Set before super().__init__, which builds the system prompt.
        self.cfg = cfg
        self.hyp = hyp
        self.account = account
        self.ledger_entries = ledger_entries
        self.on_round = on_round
        self._complete = complete or llm_client.complete
        executor = MeteredExecutor(kwargs.pop("executor"), cfg, hyp, account)
        super().__init__(executor=executor, max_rounds=cfg.max_rounds, min_rounds=1,
                         critic=None, random_experiments=False, no_mse=False, **kwargs)
        self.outcome: str | None = None  # verdict | walked_away | withdrawn | out_of_rounds
        self.verdict: str | None = None
        self.evidence: str | None = None
        self.withdraw_reason: str | None = None

    # ------------------------------------------------------------------ prompt

    def _build_system_prompt(self) -> str:
        # No silent fallback to a generic prompt: a missing prompt file is an error.
        base = _load_system_prompt(self._system_prompt_path, self._instructions_path)
        return "\n\n".join([base.rstrip(), self._run_policy_note(),
                            protocol.market_block(self.cfg, self.hyp),
                            protocol.ledger_block(self.ledger_entries)])

    def _run_policy_note(self) -> str:
        note = (
            "## RUN-SPECIFIC CONSTRAINTS (these override any conflicting numbers above)\n"
            f"- You have at most {self.max_rounds} round(s). On round {self.max_rounds} you must "
            "give your <verdict> or withdraw.\n"
            "- If any law you test integrates the trajectory with a for-loop over some "
            "timestep `dt`, the SMALLEST value of `dt` you are allowed to use is 0.01.\n"
        )
        if self.trajectory_logger is not None:
            note += "\n" + _MSE_FIT_PROMPT_BLOCK
        return note

    # ------------------------------------------------------------------ loop

    def _ask(self, messages: list[dict]) -> str:
        return self._complete(model=self.model, messages=messages, system=self._system,
                              max_tokens=self.max_tokens)

    def _read_confidence(self, reply: str, messages: list[dict], entry: dict) -> None:
        """Assessment and p; one re-prompt in the same round if assessment or p is missing."""
        assessment, p = protocol.parse_text(reply, "assessment"), protocol.parse_p(reply)
        if assessment is None or p is None:
            ask = {"role": "user", "content": protocol.CONFIDENCE_REPROMPT}
            follow = self._ask(messages + [ask])
            entry["confidence_reprompt"] = follow
            assessment = assessment or protocol.parse_text(follow, "assessment")
            p = p if p is not None else protocol.parse_p(follow)
            messages.extend([ask, {"role": "assistant", "content": follow}])
        entry.update(assessment=assessment, p_success=p)

    def _end_round(self, round_num: int, entry: dict) -> None:
        fee = 0.0 if entry["action"] == "walk_away" else self.cfg.round_fee
        self.account.charge("round_charged", fee, round_num)
        entry.update(experiments_cost=self.executor.round_cost, round_fee=fee,
                     spent_so_far=self.account.spent)
        self.conversation_log.append(entry)
        if self.on_round is not None:
            try:
                self.on_round(deepcopy(entry))
            except Exception as exc:
                print(f"on_round callback failed: {type(exc).__name__}", file=sys.stderr)

    def _status(self, round_num: int) -> str:
        return protocol.status_line(round_num, self.cfg, self.cfg.round_fee,
                                    self.executor.round_cost, self.account.spent,
                                    self.account.remaining)

    def run(self) -> Optional[str]:
        """Returns the verdict, or None if the agent walked away, withdrew or ran out of rounds."""
        self.conversation_log = []
        self.outcome = self.verdict = self.evidence = self.withdraw_reason = None
        messages: list[dict] = []
        if self.mission:
            messages.append({"role": "user", "content": self.mission})

        for round_num in range(1, self.max_rounds + 1):
            self.executor.round_num = round_num
            self.executor.round_cost = 0.0
            entry = {
                "round": round_num, "system_message": None, "llm_reply": None, "action": None,
                "experiment_input": None, "experiment_output": None, "experiment_error": None,
                "mse_fit_input": None, "mse_fit_output": None, "assessment": None,
                "p_success": None, "verdict": None, "evidence": None,
                "withdraw_reason": None,
            }
            if self.max_rounds >= 2 and round_num == self.max_rounds - 1:
                warn = (
                    f"Note: this is round {round_num} of {self.max_rounds}. You may still run "
                    "an experiment or an MSE fit in this round. In round "
                    f"{self.max_rounds} you must give your <verdict> or <withdraw>."
                )
                messages.append({"role": "user", "content": warn})
                entry["system_message"] = _join_sys(entry["system_message"], warn)
            if round_num == self.max_rounds:
                force = ("This is your final round. Give your <verdict> with <evidence> "
                         "(plus <assessment> and <p_success>), or "
                         "<withdraw>reason</withdraw>. Do not run more experiments.")
                messages.append({"role": "user", "content": force})
                entry["system_message"] = _join_sys(entry["system_message"], force)

            reply = self._ask(messages)
            entry["llm_reply"] = reply
            messages.append({"role": "assistant", "content": reply})
            self._read_confidence(reply, messages, entry)

            verdict = protocol.parse_verdict(reply)
            if verdict is not None:
                evidence = protocol.parse_text(reply, "evidence")
                if evidence is None:
                    follow = self._ask(messages + [
                        {"role": "user", "content": protocol.EVIDENCE_REPROMPT}])
                    entry["evidence_reprompt"] = follow
                    evidence = protocol.parse_text(follow, "evidence")
                self.verdict, self.evidence, self.outcome = verdict, evidence, "verdict"
                entry.update(action="verdict", verdict=verdict, evidence=evidence)
                self._end_round(round_num, entry)
                return verdict

            reason = protocol.parse_withdraw(reply)
            if reason is not None:
                self.withdraw_reason = reason or None
                first = round_num == 1
                self.outcome = "walked_away" if first else "withdrawn"
                entry.update(action="walk_away" if first else "withdraw",
                             withdraw_reason=self.withdraw_reason)
                self._end_round(round_num, entry)
                return None

            experiment_block = _extract_tag(reply, "run_experiment")
            mse_fit_block = _extract_tag(reply, "run_mse_fit")
            if experiment_block is None and mse_fit_block is None:
                empty_reply = not (reply or "").strip()
                no_tag = (
                    (
                        "ERROR: your reply was empty (it may have been cut off at the token "
                        "limit). "
                    )
                    if empty_reply else ""
                ) + (
                    "ERROR: No <run_experiment>, <run_mse_fit>, <verdict> or <withdraw> tag found "
                    "in your response. Respond with one of these XML tags (no code fences), plus "
                    "<assessment> and <p_success>.\n\n"
                    "Run an experiment:\n" + self._experiment_format + "\n\n"
                    "Test a candidate law against your data:\n<run_mse_fit>\n" + self._law_stub
                    + "</run_mse_fit>\n\n"
                    "Settle the hypothesis: <verdict>supported|refuted|inconclusive</verdict> "
                    "<evidence>...</evidence>\n\n"
                    "Or give up: <withdraw>reason</withdraw>"
                )
                entry["action"] = "no_tag"
                entry["system_message"] = _join_sys(entry["system_message"], no_tag)
                self._end_round(round_num, entry)
                messages.append({"role": "user", "content": no_tag + "\n\n" + self._status(round_num)})
                continue

            outputs = []
            if experiment_block is not None and round_num == self.max_rounds:
                outputs.append("<experiment_output>\nNot run: no experiments in the final round."
                               "\n</experiment_output>")
                entry.update(action="experiment", experiment_error="final round: not run")
            elif experiment_block is not None:
                try:
                    exp_input = json.loads(experiment_block)
                    results = self.executor.run(exp_input)
                    outputs.append("<experiment_output>\n" + _compact_json(results)
                                   + "\n</experiment_output>")
                    entry.update(action="experiment", experiment_input=exp_input,
                                 experiment_output=results)
                    self._log_trajectories(round_num, "agent", exp_input, results)
                except Exception as e:
                    outputs.append(f"<experiment_output>\nError running experiment: {e}\n"
                                   "</experiment_output>")
                    entry.update(action="experiment", experiment_error=str(e))
            if mse_fit_block is not None:
                outputs.append(self._run_mse_fit(round_num, mse_fit_block, entry))
                if entry["action"] is None:
                    entry["action"] = "mse_fit"

            self._end_round(round_num, entry)
            outputs.append(self._status(round_num))
            messages.append({"role": "user", "content": "\n\n".join(outputs)})

        self.outcome = "out_of_rounds"
        return None
