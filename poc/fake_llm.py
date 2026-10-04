"""Scripted stand-in for ``llm_client.complete``: no network, deterministic."""

from __future__ import annotations


class ScriptedLLM:
    def __init__(self, replies: list[str]):
        if not replies:
            raise ValueError("need at least one reply")
        self.replies = list(replies)
        self.calls: list[dict] = []

    def __call__(self, model: str, messages: list[dict], system: str | None = None,
                 max_tokens: int = 4096) -> str:
        self.calls.append({"model": model, "messages": [dict(m) for m in messages],
                           "system": system})
        i = min(len(self.calls) - 1, len(self.replies) - 1)  # repeat the last reply when out
        return self.replies[i]

    @classmethod
    def default(cls) -> "ScriptedLLM":
        """Works on any world: a bid with an MSE-fit round, then a verdict with a made-up
        estimate (missing quantities are re-prompted for and stay missing)."""
        return cls([
            "<assessment>No data yet.</assessment><p_success>0.6</p_success>"
            "<planned_cost>10</planned_cost>"
            "<run_mse_fit>def discovered_law(*args, **params):\n    return None\n</run_mse_fit>",
            "<assessment>Nothing contradicts it.</assessment><p_success>0.5</p_success>"
            "<verdict>supported</verdict><estimate>n = 2 ± 0.1</estimate>"
            "<evidence>Scripted reply, no experiments.</evidence>",
        ])
