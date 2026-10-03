"""Seeded uniform-choice solver over the fixed ForceBench menu."""

from __future__ import annotations

import numpy as np

from dm.solvers._inference import OfflineMenuSolver


class RandomMenu(OfflineMenuSolver):
    name = "random_menu"

    def start(self, *, seed, menu, noise_std, budget, seed_obs) -> None:
        super().start(
            seed=seed,
            menu=menu,
            noise_std=noise_std,
            budget=budget,
            seed_obs=seed_obs,
        )
        self.rng = np.random.default_rng(seed)

    def next_action(self, *, unused: tuple[int, ...], price: float) -> int | None:
        scores = self._voi_and_continue(unused)
        if scores is None:
            return None
        return int(self.rng.choice(sorted(unused)))
