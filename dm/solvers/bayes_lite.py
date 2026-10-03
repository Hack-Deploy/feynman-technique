"""Bayesian-lite greedy experimental design over the fixed ForceBench menu."""

from __future__ import annotations

from dm.solvers._inference import OfflineMenuSolver


class BayesLite(OfflineMenuSolver):
    name = "bayes_lite"

    def next_action(self, *, unused: tuple[int, ...], price: float) -> int | None:
        scores = self._voi_and_continue(unused)
        if scores is None:
            return None
        if price <= 0:
            return min(action for action, score in scores.items() if score == max(scores.values()))
        return min(
            scores,
            key=lambda action: (-(scores[action] / price), action),
        )
