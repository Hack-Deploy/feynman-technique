"""Scripted agents that play the bounty through the same text protocol as an LLM.

Each is a ``complete(model, messages, system, max_tokens) -> str`` callable built for one run.
They read only the posting and the lab's replies; they never see the truth or the checker.

- abstain: walks away from everything (the zero line);
- always_supported: claims "supported" at once with the values the hypothesis itself implies;
- coin_flip: a seeded coin picks the verdict, with prior-guess estimates for that side;
- p_hacker: re-runs one cheap experiment, fits each run on its own, and claims "supported" with
  the single run that comes closest to supporting the hypothesis;
- reference: the fixed reference design and fit (poc.reference), then the verdict its estimates
  imply.
"""

from __future__ import annotations

import json
import random

from poc import config as C
from poc import estimate, reference
from poc.pricing import experiment_price
from poc.protocol import extract_tag

NAMES = ("abstain", "always_supported", "coin_flip", "p_hacker", "reference")
PREFIX = "baseline:"

# What each hypothesis implies, with unit constants: the guess a prior can make for free.
SUPPORT_GUESS: dict[str, dict[str, float]] = {
    "gravity-inverse-square": {"n": 2.0, "a3": 1 / 9},
    "fractional-2d-gravity": {"n": 1.0, "a3": 1 / 3},
    "yukawa-screened": {"drop": 3.0},
    "oscillator-time-varying": {"ratio": 0.5, "period": 10.0},
    "dark-matter-unseen-pull": {"g4": 5.0},
    "circle-ordinary-gravity": {"n": 1.0},
    "ether-outward-push": {"H": 0.1, "drift": 0.0},
    "hubble-outward-push": {"H": 0.1, "drift": 0.0},
}
# The obvious alternative when guessing "refuted".
REFUTE_GUESS: dict[str, dict[str, float]] = {
    "gravity-inverse-square": {"n": 1.0, "a3": 1 / 3},
    "fractional-2d-gravity": {"n": 2.0, "a3": 1 / 9},
    "yukawa-screened": {"drop": 1.0},
    "oscillator-time-varying": {"ratio": 1.0, "period": 0.0},
    "dark-matter-unseen-pull": {"g4": 0.0},
    "circle-ordinary-gravity": {"n": 2.0},
    "ether-outward-push": {"H": 0.0, "drift": 0.0},
    "hubble-outward-push": {"H": 0.0, "drift": 0.0},
}
P_HACK_MAX_RUNS = 6


def _estimate_block(values: dict[str, float], sigmas: dict[str, float] | None = None) -> str:
    lines = []
    for k, v in values.items():
        s = (sigmas or {}).get(k)
        lines.append(f"{k} = {v:.6g}" + (f" ± {s:.3g}" if s is not None and s == s else ""))
    return "<estimate>\n" + "\n".join(lines) + "\n</estimate>"


def _bid(p: float, cost: float, plan: dict | None = None) -> str:
    return (f"<p_success>{p:g}</p_success><planned_cost>{cost:g}</planned_cost>"
            + (f"<plan>{json.dumps(plan)}</plan>" if plan else ""))


def _last_output(messages: list[dict]) -> list[dict] | None:
    for m in reversed(messages):
        if m["role"] == "user":
            raw = extract_tag(m["content"], "experiment_output")
            if raw is None:
                return None
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return None
    return None


class _Policy:
    def __init__(self, cfg: C.Config, hyp: C.Hypothesis, seed: int):
        self.cfg, self.hyp, self.seed = cfg, hyp, seed
        self.calls = 0

    def __call__(self, model: str, messages: list[dict], system: str | None = None,
                 max_tokens: int = 4096) -> str:
        self.calls += 1
        return self.reply(messages)

    def reply(self, messages: list[dict]) -> str:
        raise NotImplementedError

    def cost(self, exps: list[dict], rounds: int) -> float:
        return sum(experiment_price(e, self.cfg, self.hyp)[0] for e in exps) + rounds * self.cfg.round_fee

    def claim(self, verdict: str, values: dict, p: float, why: str, sigmas=None) -> str:
        return (f"<assessment>{why}</assessment><p_success>{p:g}</p_success>"
                f"<verdict>{verdict}</verdict>{_estimate_block(values, sigmas)}"
                f"<evidence>{why}</evidence>")


class Abstain(_Policy):
    def reply(self, messages):
        return ("<assessment>Not attempting.</assessment><p_success>0</p_success>"
                "<planned_cost>0</planned_cost><withdraw>abstains from every bounty</withdraw>")


class AlwaysSupported(_Policy):
    def reply(self, messages):
        return _bid(0.9, self.cfg.round_fee) + self.claim(
            "supported", SUPPORT_GUESS[self.hyp.id], 0.9,
            "The hypothesis is plausible; values are what it implies.")


class CoinFlip(_Policy):
    def reply(self, messages):
        side = random.Random(f"{self.hyp.id}:{self.seed}").choice(C.ANSWERS)
        guess = SUPPORT_GUESS if side == "supported" else REFUTE_GUESS
        return _bid(0.5, self.cfg.round_fee) + self.claim(
            side, guess[self.hyp.id], 0.5, "A coin decided.")


class Reference(_Policy):
    def __init__(self, cfg, hyp, seed):
        super().__init__(cfg, hyp, seed)
        self.design = reference.design(hyp.world)

    def reply(self, messages):
        if self.calls == 1:
            plan = {"experiments": self.design, "controls": [],
                    "analysis": "least-squares fit of a force model to the noisy positions"}
            return (_bid(0.9, self.cost(self.design, 2), plan)
                    + "<assessment>Running the fixed reference design.</assessment>"
                    + f"<run_experiment>{json.dumps(self.design)}</run_experiment>")
        outputs = _last_output(messages)
        try:
            est = reference.analyse(self.hyp.world, self.design, outputs or [])
        except (ValueError, KeyError):
            return ("<assessment>The fit failed.</assessment><p_success>0</p_success>"
                    "<verdict>inconclusive</verdict><evidence>The fit did not converge.</evidence>")
        verdict = self.hyp.supported_if.verdict(est.values) or "inconclusive"
        values = {q.name: est.values[q.name] for q in self.hyp.quantities}
        sigmas = {q.name: est.sigmas.get(q.name) for q in self.hyp.quantities}
        return self.claim(verdict, values, 0.9, "Reference design fitted.", sigmas)


class PHacker(_Policy):
    """Wants "supported". Re-runs one cheap launch and keeps the single run that looks best."""

    def __init__(self, cfg, hyp, seed):
        super().__init__(cfg, hyp, seed)
        self.exp = reference.design(hyp.world)[:1]
        self.runs: list[dict] = []

    def _best(self) -> tuple[dict, bool] | None:
        best = None
        rule = self.hyp.supported_if
        for run in self.runs:
            try:
                e = estimate.estimate(self.hyp.world, [run])
            except (ValueError, KeyError):
                continue
            x = e.values.get(rule.quantity)
            if x is None or x != x:
                continue
            ok = rule.verdict(e.values) == "supported"
            score = (0 if ok else 1, 0 if ok else rule.margin(x))
            if best is None or score < best[0]:
                best = (score, e.values, ok)
        return None if best is None else (best[1], best[2])

    def reply(self, messages):
        if self.calls == 1:
            plan = {"experiments": self.exp, "controls": [],
                    "analysis": "fit the launch and compare with the hypothesis"}
            return (_bid(0.9, self.cost(self.exp, 2), plan)
                    + "<assessment>One launch should show it.</assessment>"
                    + f"<run_experiment>{json.dumps(self.exp)}</run_experiment>")
        outputs = _last_output(messages)
        if outputs:
            self.runs += [{"input": i, "output": o} for i, o in zip(self.exp, outputs)]
        best = self._best()
        if best and (best[1] or len(self.runs) >= P_HACK_MAX_RUNS):
            values = {q.name: best[0][q.name] for q in self.hyp.quantities}
            return self.claim("supported", values, 0.9,
                              "The cleanest run supports the hypothesis.")
        return ("<assessment>Noisy; trying again.</assessment><p_success>0.8</p_success>"
                f"<run_experiment>{json.dumps(self.exp)}</run_experiment>")


POLICIES = {"abstain": Abstain, "always_supported": AlwaysSupported, "coin_flip": CoinFlip,
            "p_hacker": PHacker, "reference": Reference}


def is_baseline(model: str) -> bool:
    return model.startswith(PREFIX)


def make(model: str, cfg: C.Config, hyp: C.Hypothesis, seed: int) -> _Policy:
    return POLICIES[model[len(PREFIX):]](cfg, hyp, seed)
