"""Scripted stand-in for ``llm_client.complete``: no network, deterministic."""

from __future__ import annotations

import json


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
        """Works on any world: one free MSE-fit round, then a verdict."""
        return cls([
            "<assessment>No data yet.</assessment><p_success>0.6</p_success>"
            "<run_mse_fit>def discovered_law(*args, **params):\n    return None\n</run_mse_fit>",
            "<assessment>Nothing contradicts it.</assessment><p_success>0.5</p_success>"
            "<verdict>supported</verdict>"
            "<evidence>Scripted reply, no experiments.</evidence>",
        ])


_TWO_PARTICLE_WORLDS = {
    "gravity", "yukawa", "coulomb_easy", "oscillator", "fractional", "extra_dimensions",
}


def scripted_experiment(hyp, cfg) -> dict:
    if hyp.world in _TWO_PARTICLE_WORLDS:
        return {
            "p1": 1,
            "p2": 1,
            "pos2": [3, 0],
            "velocity2": [0, 0],
            "measurement_times": [0.5, 1, 2],
        }

    from scienceagent.worlds import get_world
    from poc import config as C

    world_spec = get_world(
        hyp.world,
        engine=C.ENGINE,
        noise_std=cfg.noise_std,
        noise_seed=0,
    )
    formatted = world_spec["experiment_format"]
    start = formatted.find("[", formatted.find("<run_experiment>"))
    if start < 0:
        raise ValueError(f"no scripted experiment format for {hyp.world}")
    experiments, _ = json.JSONDecoder().raw_decode(formatted[start:])
    if not experiments:
        raise ValueError(f"no scripted experiment example for {hyp.world}")
    experiment = experiments[0]
    experiment["measurement_times"] = [1.0, 2.0]
    return experiment


def scripted_transport(replies, usage_fn=None):
    replies = list(replies)
    if not replies:
        raise ValueError("need at least one reply")
    call_count = 0

    def transport(model, system, messages, max_tokens):
        nonlocal call_count
        text = replies[min(call_count, len(replies) - 1)]
        call_count += 1
        usage = (
            usage_fn(system, messages, text)
            if usage_fn is not None
            else {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            }
        )
        return text, usage

    return transport
