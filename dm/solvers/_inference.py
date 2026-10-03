"""Shared deterministic model fitting and experimental-design utilities."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Iterable

import numpy as np
from scipy.optimize import least_squares
from scipy.special import k1

from dm.solvers import Observation, Submission

MODEL_FAMILIES = ("power", "log", "yukawa", "timemod", "crossover")
ROLES = ("ratio", "p1", "product")
SOFTENING = 0.05
DT = 0.01
STOP_THRESHOLD_SIGMAS = 12.0


@dataclass(frozen=True)
class ModelFit:
    family: str
    role: str
    params: tuple[float, ...]
    rss: float
    bic: float
    weight: float = 0.0


def _launch_dict(launch: dict | object) -> dict:
    return launch.experiment() if hasattr(launch, "experiment") else launch


def _role_factor(role: str, launches: list[dict]) -> np.ndarray:
    p1 = np.asarray([float(item["p1"]) for item in launches])
    p2 = np.asarray([float(item["p2"]) for item in launches])
    if role == "ratio":
        return p1 / p2
    if role == "p1":
        return p1
    if role == "product":
        return p1 * p2
    raise ValueError(f"Unknown charge role: {role!r}")


def _force_magnitude(
    family: str, params: np.ndarray, r_eff: np.ndarray, time: float
) -> np.ndarray:
    """Return signed radial force magnitudes with leading shape (B, L)."""
    if family == "power":
        k = params[:, 0, None]
        exponent = params[:, 1, None]
        return k / np.power(r_eff, exponent)
    if family == "log":
        return params[:, 0, None] / r_eff
    if family == "yukawa":
        k = params[:, 0, None]
        length = params[:, 1, None]
        return k * k1(r_eff / length) / length
    if family == "timemod":
        k, omega, phase = (params[:, i, None] for i in range(3))
        return k * np.cos(omega * time + phase) / r_eff
    if family == "crossover":
        k1_values = params[:, 0, None]
        k2_values = params[:, 1, None]
        return k1_values / r_eff + k2_values / np.square(r_eff)
    raise ValueError(f"Unknown force family: {family!r}")


def simulate(
    family: str,
    role: str,
    params: np.ndarray | Iterable[float],
    launches: list[dict] | list[object],
) -> np.ndarray:
    """Batched KDK leapfrog predictions: (B, launches, times, 2)."""
    parameters = np.atleast_2d(np.asarray(params, dtype=float))
    experiments = [_launch_dict(item) for item in launches]
    if not experiments:
        return np.empty((len(parameters), 0, 0, 2), dtype=float)
    times = tuple(float(t) for t in experiments[0]["measurement_times"])
    if any(tuple(float(t) for t in item["measurement_times"]) != times for item in experiments):
        raise ValueError("All batched launches must use the same measurement times")
    indices = tuple(int(round(t / DT)) for t in times)
    if any(abs(index * DT - t) > 1e-9 for index, t in zip(indices, times)):
        raise ValueError("Measurement times must be multiples of the 0.01 integration step")

    n_batch, n_launches = len(parameters), len(experiments)
    x = np.broadcast_to(
        np.asarray([float(item["pos2"][0]) for item in experiments])[None, :],
        (n_batch, n_launches),
    ).copy()
    y = np.broadcast_to(
        np.asarray([float(item["pos2"][1]) for item in experiments])[None, :],
        (n_batch, n_launches),
    ).copy()
    vx = np.broadcast_to(
        np.asarray([float(item["velocity2"][0]) for item in experiments])[None, :],
        (n_batch, n_launches),
    ).copy()
    vy = np.broadcast_to(
        np.asarray([float(item["velocity2"][1]) for item in experiments])[None, :],
        (n_batch, n_launches),
    ).copy()
    role_values = _role_factor(role, experiments)[None, :]
    max_steps = max(indices, default=0)
    recorded = np.empty((n_batch, n_launches, len(times), 2), dtype=float)
    time_slots: dict[int, list[int]] = {}
    for slot, step in enumerate(indices):
        time_slots.setdefault(step, []).append(slot)

    def acceleration(at_time: float) -> tuple[np.ndarray, np.ndarray]:
        radius = np.hypot(x, y)
        softened = np.hypot(radius, SOFTENING)
        force = _force_magnitude(family, parameters, softened, at_time)
        direction_radius = np.maximum(radius, 1e-12)
        ax = -role_values * force * x / direction_radius
        ay = -role_values * force * y / direction_radius
        return ax, ay

    if 0 in time_slots:
        for slot in time_slots[0]:
            recorded[:, :, slot, 0] = x
            recorded[:, :, slot, 1] = y
    with np.errstate(all="ignore"):
        ax, ay = acceleration(0.0)
        for step in range(max_steps):
            vx += (0.5 * DT) * ax
            vy += (0.5 * DT) * ay
            x += DT * vx
            y += DT * vy
            ax, ay = acceleration((step + 1) * DT)
            vx += (0.5 * DT) * ax
            vy += (0.5 * DT) * ay
            for slot in time_slots.get(step + 1, ()):
                recorded[:, :, slot, 0] = x
                recorded[:, :, slot, 1] = y
    return np.nan_to_num(recorded, nan=1e3, posinf=1e3, neginf=-1e3)


def _branch_bounds(family: str, sign: int | None) -> tuple[np.ndarray, np.ndarray]:
    if family == "power":
        return (
            np.array([0.01 if sign == 1 else -5.0, 0.5]),
            np.array([5.0 if sign == 1 else -0.01, 2.5]),
        )
    if family == "log":
        return (
            np.array([0.01 if sign == 1 else -5.0]),
            np.array([5.0 if sign == 1 else -0.01]),
        )
    if family == "yukawa":
        return (
            np.array([0.01 if sign == 1 else -5.0, 0.5]),
            np.array([5.0 if sign == 1 else -0.01, 40.0]),
        )
    if family == "timemod":
        return np.array([0.01, 0.1, 0.0]), np.array([5.0, 6.0, 2 * math.pi])
    if family == "crossover":
        return np.array([-5.0, -5.0]), np.array([5.0, 5.0])
    raise ValueError(f"Unknown force family: {family!r}")


def _grid(family: str, sign: int | None) -> np.ndarray:
    positive_k = np.geomspace(0.01, 5.0, 12)
    signed_k = positive_k if sign == 1 else -positive_k
    if family == "power":
        return np.asarray(list(product(signed_k, np.linspace(0.5, 2.5, 9))))
    if family == "log":
        return signed_k[:, None]
    if family == "yukawa":
        lengths = np.geomspace(0.5, 40.0, 10)
        return np.asarray(list(product(signed_k, lengths)))
    if family == "timemod":
        return np.asarray(
            list(
                product(
                    positive_k,
                    np.linspace(0.1, 6.0, 12),
                    np.linspace(0.0, 2 * math.pi, 8, endpoint=False),
                )
            )
        )
    if family == "crossover":
        values = np.linspace(-5.0, 5.0, 9)
        return np.asarray(list(product(values, values)))
    raise ValueError(f"Unknown force family: {family!r}")


def _fit_branch(
    family: str,
    role: str,
    sign: int | None,
    launches: list[dict],
    observed: np.ndarray,
    warm: tuple[float, ...] | None,
    n_starts: int,
) -> tuple[np.ndarray, float]:
    candidates = _grid(family, sign)
    predictions = simulate(family, role, candidates, launches)
    targets = observed[None, ...]
    rss_grid = np.sum(np.square(predictions - targets), axis=(1, 2, 3))
    low, high = _branch_bounds(family, sign)
    starts: list[np.ndarray] = []
    for index in np.argsort(rss_grid, kind="stable"):
        candidate = candidates[index]
        if not any(np.allclose(candidate, prior, atol=1e-10) for prior in starts):
            starts.append(candidate.copy())
        if len(starts) == n_starts:
            break
    if warm is not None:
        warm_values = np.asarray(warm, dtype=float)
        if warm_values.shape == low.shape and np.all(warm_values >= low) and np.all(warm_values <= high):
            starts.append(warm_values)

    def residual(params: np.ndarray) -> np.ndarray:
        return (simulate(family, role, params, launches)[0] - observed).ravel()

    def jacobian(params: np.ndarray) -> np.ndarray:
        base = np.asarray(params, dtype=float)
        points = [base]
        deltas: list[float] = []
        for i in range(len(base)):
            step = max(abs(base[i]) * 1e-4, 1e-5)
            if base[i] + step <= high[i]:
                changed = base.copy()
                changed[i] += step
                deltas.append(step)
            else:
                changed = base.copy()
                changed[i] = max(low[i], base[i] - step)
                deltas.append(changed[i] - base[i])
            points.append(changed)
        fitted = simulate(family, role, np.asarray(points), launches)
        base_residual = (fitted[0] - observed).ravel()
        columns = [
            ((fitted[index + 1] - observed).ravel() - base_residual) / deltas[index]
            for index in range(len(base))
        ]
        return np.column_stack(columns)

    best_params = starts[0]
    best_rss = float("inf")
    for start in starts:
        result = least_squares(
            residual,
            start,
            jac=jacobian,
            bounds=(low, high),
            method="trf",
            max_nfev=50,
            x_scale="jac",
        )
        rss = float(np.dot(result.fun, result.fun))
        if rss < best_rss:
            best_params = result.x.copy()
            best_rss = rss
    return best_params, best_rss


def _sign_branches(family: str) -> tuple[int | None, ...]:
    return (1, -1) if family in ("power", "log", "yukawa") else (None,)


def _fit_model(
    family: str,
    role: str,
    launches: list[dict],
    observed: np.ndarray,
    warm: tuple[float, ...] | None,
    n_starts: int = 3,
) -> tuple[np.ndarray, float]:
    fits = [
        _fit_branch(family, role, sign, launches, observed, warm, n_starts)
        for sign in _sign_branches(family)
    ]
    return min(fits, key=lambda result: (result[1], tuple(result[0])))


def _model_order(fit: ModelFit) -> tuple[int, int]:
    return MODEL_FAMILIES.index(fit.family), ROLES.index(fit.role)


class ModelInference:
    """Fit the fixed family/role library and score menu actions by weighted VoI."""

    def __init__(self, menu: tuple[object, ...], noise_std: float, n_starts: int = 3):
        if n_starts < 1:
            raise ValueError("n_starts must be positive")
        self.menu = menu
        self.noise_std = float(noise_std)
        self.n_starts = n_starts
        self.observations: list[Observation] = []
        self.fits: list[ModelFit] = []
        self._warm: dict[tuple[str, str], tuple[float, ...]] = {}

    def add_observation(self, observation: Observation) -> None:
        self.observations.append(observation)
        self.refit()

    def refit(self) -> None:
        launches = [observation.input for observation in self.observations]
        observed = np.asarray(
            [observation.output["pos2"] for observation in self.observations], dtype=float
        )
        same_charge = all(
            float(item["p1"]) == float(item["p2"]) == 1.0 for item in launches
        )
        roles = ("ratio",) if same_charge else ROLES
        fitted: list[ModelFit] = []
        residual_count = observed.size
        for family in MODEL_FAMILIES:
            for role in roles:
                params, rss = _fit_model(
                    family,
                    role,
                    launches,
                    observed,
                    self._warm.get((family, role)),
                    self.n_starts,
                )
                self._warm[(family, role)] = tuple(float(value) for value in params)
                bic = rss / (self.noise_std**2) + len(params) * math.log(residual_count)
                fitted.append(
                    ModelFit(
                        family,
                        role,
                        tuple(float(value) for value in params),
                        float(rss),
                        float(bic),
                    )
                )
                if same_charge:
                    fitted.extend(
                        ModelFit(
                            family,
                            other_role,
                            tuple(float(value) for value in params),
                            float(rss),
                            float(rss / (self.noise_std**2) + len(params) * math.log(residual_count)),
                        )
                        for other_role in ("p1", "product")
                    )
        minimum = min(fit.bic for fit in fitted)
        raw_weights = np.asarray([math.exp(-(fit.bic - minimum) / 2.0) for fit in fitted])
        total = float(raw_weights.sum())
        self.fits = [
            ModelFit(
                fit.family,
                fit.role,
                fit.params,
                fit.rss,
                fit.bic,
                float(weight / total),
            )
            for fit, weight in zip(fitted, raw_weights)
        ]
        self.fits.sort(key=_model_order)

    @property
    def top(self) -> ModelFit:
        return min(self.fits, key=lambda fit: (-fit.weight, *_model_order(fit)))

    def voi_actions(self, actions: Iterable[int]) -> dict[int, float]:
        selected = tuple(sorted(set(actions)))
        if not selected:
            return {}
        launches = [
            _launch_dict(next(item for item in self.menu if item.action == action))
            for action in selected
        ]
        means = []
        weights = np.asarray([fit.weight for fit in self.fits])
        for fit in self.fits:
            means.append(
                simulate(fit.family, fit.role, fit.params, launches)[0]
            )
        predictions = np.asarray(means)
        weighted_mean = np.tensordot(weights, predictions, axes=(0, 0))
        differences = predictions - weighted_mean[None, ...]
        scores = np.sum(
            weights[:, None, None, None] * np.square(differences),
            axis=(0, 2, 3),
        )
        return {action: float(score) for action, score in zip(selected, scores)}

    def round_log(self, actions: Iterable[int]) -> dict:
        return {
            "models": [
                {
                    "family": fit.family,
                    "role": fit.role,
                    "params": [float(value) for value in fit.params],
                    "rss": fit.rss,
                    "bic": fit.bic,
                    "weight": fit.weight,
                }
                for fit in self.fits
            ],
            "top_model": {
                "family": self.top.family,
                "role": self.top.role,
                "params": list(self.top.params),
                "weight": self.top.weight,
            },
            "voi": {str(action): score for action, score in self.voi_actions(actions).items()},
        }

    def submit(self) -> Submission:
        top = self.top
        law_source = generate_law(top.family, top.role, top.params)
        force_at_one = _force_magnitude(
            top.family, np.asarray([top.params]), np.asarray([math.hypot(1.0, SOFTENING)]), 0.0
        )[0, 0]
        sign = "attractive" if force_at_one >= 0 else "repulsive"
        params = _parameter_dict(top.family, top.params)
        explanation = (
            f"The fitted {top.family} family with {top.role} coupling is {sign}; "
            f"parameters are {params}."
        )
        summary = {
            "top_model": {
                "family": top.family,
                "role": top.role,
                "params": list(top.params),
                "weight": top.weight,
                "bic": top.bic,
            },
            "models": [
                {
                    "family": fit.family,
                    "role": fit.role,
                    "params": list(fit.params),
                    "weight": fit.weight,
                    "bic": fit.bic,
                }
                for fit in self.fits
            ],
        }
        return Submission(law_source, top.weight, explanation, summary)


class OfflineMenuSolver:
    """Common model, stopping, and submission behavior for offline solvers."""

    name = "offline"

    def __init__(self, n_starts: int = 3):
        if n_starts < 1:
            raise ValueError("n_starts must be positive")
        self.n_starts = n_starts

    def start(
        self,
        *,
        seed: int,
        menu: tuple[object, ...],
        noise_std: float,
        budget: int,
        seed_obs: Observation,
    ) -> None:
        self.seed = seed
        self.menu = menu
        self.noise_std = noise_std
        self.budget = budget
        self.inference = ModelInference(menu, noise_std, n_starts=self.n_starts)
        self.used = {seed_obs.action}
        self.remaining: tuple[int, ...] = tuple(
            item.action for item in menu if item.action != seed_obs.action
        )
        self.inference.add_observation(seed_obs)

    def observe(self, obs: Observation) -> None:
        self.used.add(obs.action)
        self.inference.add_observation(obs)
        self.remaining = tuple(
            sorted(item.action for item in self.menu if item.action not in self.used)
        )

    def _voi_and_continue(self, unused: tuple[int, ...]) -> dict[int, float] | None:
        self.remaining = tuple(sorted(unused))
        scores = self.inference.voi_actions(self.remaining)
        if not scores or max(scores.values()) < STOP_THRESHOLD_SIGMAS * self.noise_std**2:
            return None
        return scores

    def submit(self) -> Submission:
        return self.inference.submit()

    def round_log(self) -> dict:
        return self.inference.round_log(self.remaining)


def _parameter_dict(family: str, params: tuple[float, ...]) -> dict[str, float]:
    names = {
        "power": ("k", "p"),
        "log": ("k",),
        "yukawa": ("k", "lambda"),
        "timemod": ("k", "omega", "phi"),
        "crossover": ("k1", "k2"),
    }[family]
    return {name: float(value) for name, value in zip(names, params)}


def generate_law(family: str, role: str, params: Iterable[float]) -> str:
    """Create the self-contained law source for an optimized library model."""
    fitted = tuple(float(value) for value in params)
    names = {
        "power": ("k", "exponent"),
        "log": ("k",),
        "yukawa": ("k", "length"),
        "timemod": ("k", "omega", "phase"),
        "crossover": ("k1", "k2"),
    }[family]
    assignments = "\n".join(
        f"    {name} = {value!r}" for name, value in zip(names, fitted)
    )
    role_code = {
        "ratio": "    coupling = float(p1) / float(p2)",
        "p1": "    coupling = float(p1)",
        "product": "    coupling = float(p1) * float(p2)",
    }[role]
    if family == "power":
        force_code = "        force = k / (softened ** exponent)"
    elif family == "log":
        force_code = "        force = k / softened"
    elif family == "yukawa":
        force_code = "        force = k * k1(softened / length) / length"
    elif family == "timemod":
        force_code = "        force = k * math.cos(omega * time + phase) / softened"
    else:
        force_code = "        force = k1 / softened + k2 / (softened * softened)"
    yukawa_import = "    from scipy.special import k1\n" if family == "yukawa" else ""
    return (
        "def discovered_law(pos1, pos2, p1, p2, velocity2, duration):\n"
        "    import math\n"
        f"{yukawa_import}"
        f"{assignments}\n"
        f"{role_code}\n"
        "    n = max(1, int(round(float(duration) / 0.01)))\n"
        "    h = float(duration) / n\n"
        "    x, y = float(pos2[0]), float(pos2[1])\n"
        "    vx, vy = float(velocity2[0]), float(velocity2[1])\n"
        "    def acceleration(x, y, time):\n"
        "        radius = math.hypot(x, y)\n"
        "        softened = math.hypot(radius, 0.05)\n"
        f"{force_code}\n"
        "        direction_radius = max(radius, 1e-12)\n"
        "        return -coupling * force * x / direction_radius, -coupling * force * y / direction_radius\n"
        "    ax, ay = acceleration(x, y, 0.0)\n"
        "    for step in range(n):\n"
        "        vx += 0.5 * h * ax\n"
        "        vy += 0.5 * h * ay\n"
        "        x += h * vx\n"
        "        y += h * vy\n"
        "        ax, ay = acceleration(x, y, (step + 1) * h)\n"
        "        vx += 0.5 * h * ax\n"
        "        vy += 0.5 * h * ay\n"
        "    return [x, y], [vx, vy]\n"
    )
