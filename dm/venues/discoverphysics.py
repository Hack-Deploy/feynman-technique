"""DiscoverPhysics venue: the vendor's free-form agent loop, metered.

``run_attempt`` runs the vendor ``DiscoveryAgent`` unchanged (same system prompt,
mission, 16 rounds, mid-round MSE fitting) against a ``MeteredExecutor`` that charges
the solver's wallet for every experiment, and hands back a ``SubmittedAttempt`` for
``dm.settle``. Why it is wrapped rather than edited:

- The agent calls ``scienceagent.llm_client.complete`` through the module attribute,
  so a per-attempt ``UsageMeter`` is swapped in for the duration of the attempt
  (under a lock: one attempt at a time per process). This is how usage is attributed
  to an attempt and how the scripted fake LLM replaces the real one in tests.
- The executor is wrapped, not subclassed: the agent catches any exception from
  ``executor.run`` and shows its message to the solver, so ``InsufficientCredits``
  reaches the solver as an experiment error and it can still submit.
- ``market_aware`` appends the market terms to the mission (the first user message);
  the system prompt stays identical to the vendor benchmark's.
- After the loop the solver is asked once, in the same conversation, for the
  probability that its law passes (``stated_p_success``), before any verdict exists.

The oracle never sees this executor: hidden cases are scored on a fresh one in the
oracle's own process (see the readiness test on metered evaluators).
"""

from __future__ import annotations

import json
import re
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterator

from dm.llm import LLM, SpendCap, UsageMeter, live_llm, live_settings
from dm.store import ATTEMPTS_DIR
from dm.types import InsufficientCredits, SubmittedAttempt
from dm.wallet import Wallet

WORLDS = (
    "gravity", "yukawa", "coulomb_easy", "oscillator", "fractional", "extra_dimensions",
    "circle", "three_species", "dark_matter", "ether", "hubble",
)
NOISE_STD = 0.075  # absolute σ, as the vendor runner (PLAN C6)
MAX_ROUNDS = 16  # configs/bench.yml
MAX_TOKENS = 8192  # run_discovery.py default for non-reasoning models
PROTOCOL = "discoverphysics_native"

STATED_P_PROMPT = (
    "Before your law is scored: what is the probability, between 0 and 1, that it passes, "
    "that is, that its normalised mean squared error on hidden test cases you have not seen "
    "is below 0.1? Reply with only <p_success>0.xx</p_success>."
)
_P_RE = re.compile(r"<p_success>\s*([0-9]*\.?[0-9]+)\s*</p_success>")

_LLM_LOCK = threading.Lock()


class MeteredExecutor:
    """Charges ``len(experiments) × price`` before the simulator runs.

    A batch that is not a JSON list of objects is refused uncharged. A batch the
    wallet cannot pay for raises ``InsufficientCredits`` and never runs. If the
    simulator rejects a paid batch (e.g. a missing key), the charge is refunded with
    its own event and the error is re-raised, so no data is ever free and no failed
    batch is paid for. Anything else (noise controls, world state) is the inner
    executor's.
    """

    def __init__(self, inner, wallet: Wallet, price: float, world: str,
                 round_fn: Callable[[], int] | None = None):
        self._inner = inner
        self._wallet = wallet
        self._price = price
        self._world = world
        self._batch = 0
        self.round_fn = round_fn

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def run(self, experiments):
        self._batch += 1
        round_num = self.round_fn() if self.round_fn else self._batch
        if not isinstance(experiments, list) or not all(isinstance(e, dict)
                                                        for e in experiments):
            raise ValueError("experiments must be a JSON list of objects; nothing was charged")
        n = len(experiments)
        self._wallet.charge(n, self._price, world=self._world, round_num=round_num)
        try:
            return self._inner.run(experiments)
        except Exception as e:
            self._wallet.refund(n, self._price, world=self._world, round_num=round_num,
                                reason=f"{type(e).__name__}: {e}")
            raise


def market_note(price: float, balance: float, prize: float | None) -> str:
    prize_part = f"; the prize is {prize:g} credits" if prize is not None else ""
    return (f"\n\nMARKET TERMS: each experiment costs {price:g} credits, charged before it "
            f"runs. Your balance is {balance:g} credits{prize_part}. If you cannot pay, the "
            "experiment returns an error and you must submit your <final_law>.")


def parse_p(text: str | None) -> float | None:
    m = _P_RE.search(text or "")
    if m is None:
        return None
    p = float(m.group(1))
    return p if 0.0 <= p <= 1.0 else None


@contextmanager
def _llm_installed(meter: UsageMeter) -> Iterator[None]:
    from scienceagent import llm_client

    with _LLM_LOCK:
        saved = llm_client.complete
        llm_client.complete = meter
        try:
            yield
        finally:
            llm_client.complete = saved


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def _repo_relative(path: Path) -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path.resolve())


def run_attempt(
    solver_model: str,
    world: str,
    seed: int,
    wallet: Wallet,
    price: float,
    market_aware: bool = True,
    llm: LLM | None = None,
    *,
    prize: float | None = None,
    max_rounds: int = MAX_ROUNDS,
    noise_std: float = NOISE_STD,
    ask_p: bool = True,
    cap: SpendCap | None = None,
    transcript_dir: Path | str | None = None,
) -> SubmittedAttempt:
    """One solver, one world, one seed through the vendor loop; no verdict yet.

    ``llm=None`` means the paid client for ``solver_model``, which needs
    ``ENABLE_LIVE=1`` and ``DM_MAX_USD``; every call is then capped by ``cap``
    (default: a fresh cap at ``DM_MAX_USD``; share one cap across a whole command).
    """
    if world not in WORLDS:
        raise ValueError(f"Unknown DiscoverPhysics world: {world!r}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int (noise_seed=None is not reproducible)")
    if llm is None:
        llm = live_llm(solver_model)
        cap = cap if cap is not None else SpendCap(live_settings())

    from scienceagent.agent import DiscoveryAgent
    from scienceagent.evaluator import _extract_training_trajectories
    from scienceagent.trajectory_logger import TrajectoryLogger
    from scienceagent.worlds import get_world

    w = get_world(world, engine="nbody", noise_std=noise_std, noise_seed=seed)
    initial_credits = wallet.balance + wallet.lab_revenue
    start_balance = wallet.balance
    revenue_before = wallet.lab_revenue
    events_before = len(wallet.events)
    mission = w["mission"] + (market_note(price, start_balance, prize) if market_aware else "")
    meter = UsageMeter(llm, cap=cap)
    executor = MeteredExecutor(w["executor"], wallet, price, world)
    run_id = f"{_safe(solver_model)}-{world}-s{seed}"

    with tempfile.TemporaryDirectory(prefix="dm-dp-") as tmp:
        logger = TrajectoryLogger(world=world, executor=executor,
                                  csv_path=Path(tmp) / f"{world}.csv", run_id=run_id)
        agent = DiscoveryAgent(
            model=solver_model, executor=executor, mission=mission, max_tokens=MAX_TOKENS,
            verbose=False, system_prompt_path=w["system_prompt"],
            instructions_path=w["instructions"], law_stub=w["law_stub"],
            experiment_format=w["experiment_format"], max_rounds=max_rounds,
            trajectory_logger=logger)
        executor.round_fn = lambda: len(agent.conversation_log) + 1
        with _llm_installed(meter):
            law = agent.run()
            stated_p, p_exchange = None, None
            if law is not None and ask_p:
                question = {"role": "user", "content": STATED_P_PROMPT}
                messages = meter.last_messages + [
                    {"role": "assistant", "content": meter.last_reply or ""}, question]
                reply = meter(model=solver_model, messages=messages,
                              system=meter.last_system, max_tokens=MAX_TOKENS)
                stated_p = parse_p(reply)
                p_exchange = {"question": STATED_P_PROMPT, "reply": reply}

    training = _extract_training_trajectories(agent.conversation_log)
    events = wallet.events[events_before:]
    net_count = (sum(e["count"] for e in events if e["type"] == "experiment_charged")
                 - sum(e["count"] for e in events if e["type"] == "experiment_refunded"))
    lab_cost = wallet.lab_revenue - revenue_before
    if abs(wallet.balance + wallet.lab_revenue - initial_credits) > 1e-9:
        raise AssertionError("credit conservation violated during DiscoverPhysics attempt")
    if net_count != len(training) or abs(lab_cost - len(training) * price) > 1e-9:
        raise AssertionError(
            f"charged for {net_count} experiments ({lab_cost} credits) but "
            f"{len(training)} produced data")

    refused = sum(1 for e in events if e["type"] == "insufficient_credits")
    stopped = ("submitted" if law is not None else "max_rounds")
    terms = {"price": price, "market_aware": market_aware, "prize": prize,
             "start_balance": start_balance, "noise_std": noise_std,
             "max_rounds": max_rounds}
    attempt = SubmittedAttempt(
        source="live", protocol=PROTOCOL, venue="discoverphysics", world=world,
        solver=solver_model, seed=seed, stated_p_success=stated_p,
        rounds=len(agent.conversation_log), experiments=len(training), lab_cost=lab_cost,
        llm_usage=meter.usage(), submitted_law=law,
        explanation=agent.discovered_explanation, training=training,
        extra={"terms": terms, "stopped_reason": stopped,
               "insufficient_credit_refusals": refused,
               "stated_p_asked": law is not None and ask_p})

    target = (Path(transcript_dir) if transcript_dir is not None
              else ATTEMPTS_DIR / "transcripts" / "discoverphysics")
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{_safe(solver_model)}_{world}_s{seed}.json"
    transcript = {
        "solver": solver_model, "world": world, "seed": seed, "terms": terms,
        "mission": mission, "system": agent._system, "rounds": agent.conversation_log,
        "final_law": law, "explanation": agent.discovered_explanation,
        "stated_p": p_exchange, "llm_usage": meter.usage(), "wallet_events": events,
    }
    path.write_text(json.dumps(transcript, sort_keys=True, indent=1, allow_nan=False) + "\n")
    return replace(attempt, transcript_path=_repo_relative(path))
