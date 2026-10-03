"""Scripted stand-in for the venue LLM: no network, deterministic (tests only)."""

from __future__ import annotations

import json

from dm.llm import LLMReply

# Marker the DiscoverPhysics venue puts in its stated-probability question.
P_MARKER = "<p_success>"

EXPERIMENT = {"p1": 1.0, "p2": 1.0, "pos2": [3.0, 0.0], "velocity2": [0.0, 0.0],
              "measurement_times": [0.5, 1.0, 2.0]}

# 1/r law with k = 1/(2π): a = -k·p1/p2·r̂/r. Passes gravity, fails yukawa.
LAW_1_OVER_R = '''
def discovered_law(pos1, pos2, p1, p2, velocity2, duration):
    import numpy as np
    from scipy.integrate import solve_ivp
    k = 1.0 / (2.0 * np.pi)
    p1x = np.asarray(pos1, dtype=float)
    def rhs(t, y):
        d = y[:2] - p1x
        r = np.hypot(d[0], d[1])
        a = -k * p1 / p2 * d / r**2
        return [y[2], y[3], a[0], a[1]]
    y0 = [pos2[0], pos2[1], velocity2[0], velocity2[1]]
    sol = solve_ivp(rhs, (0.0, duration), y0, rtol=1e-9, atol=1e-11)
    y = sol.y[:, -1]
    return np.array(y[:2]), np.array(y[2:])
'''

EXPLANATION = ("A static 2D Poisson field sourced by p1 accelerates particle 2 by "
               "-grad(phi)/p2; the force falls off as 1/r.")


class ScriptedLLM:
    """Replies from a script in order (repeating the last one); answers the venue's
    stated-probability question with ``p_reply`` without consuming the script."""

    def __init__(self, replies: list[str], p_reply: str = "<p_success>0.6</p_success>",
                 with_usage: bool = False):
        if not replies:
            raise ValueError("need at least one reply")
        self.replies = list(replies)
        self.p_reply = p_reply
        self.with_usage = with_usage
        self.calls: list[dict] = []
        self._next = 0

    def __call__(self, model: str, messages: list[dict], system: str | None = None,
                 max_tokens: int = 4096) -> str | LLMReply:
        self.calls.append({"model": model, "messages": [dict(m) for m in messages],
                           "system": system, "max_tokens": max_tokens})
        if messages and messages[-1]["role"] == "user" and P_MARKER in messages[-1]["content"]:
            text = self.p_reply
        else:
            text = self.replies[min(self._next, len(self.replies) - 1)]
            self._next += 1
        if self.with_usage:
            return LLMReply(text, input_tokens=100 * len(messages), output_tokens=len(text))
        return text


def one_experiment_then_law(experiments: list[dict] | None = None,
                            explanation: bool = True) -> list[str]:
    exps = experiments if experiments is not None else [EXPERIMENT]
    law = f"<final_law>{LAW_1_OVER_R}</final_law>"
    if explanation:
        law += f"\n<explanation>{EXPLANATION}</explanation>"
    return [f"<run_experiment>{json.dumps(exps)}</run_experiment>", law]


def fake_llm(**kw) -> ScriptedLLM:
    """The default scripted solver: one experiment, then the 1/r law, then p = 0.6."""
    return ScriptedLLM(one_experiment_then_law(), **kw)
