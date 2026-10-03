"""LLM calls for venues: per-attempt usage metering and the live spend guard.

A venue's LLM is any callable ``(model, messages, system, max_tokens) -> str | LLMReply``,
the signature of the vendor's ``scienceagent.llm_client.complete``. The vendor client
returns text only, so for plain-string replies usage is estimated (about 4 characters
per token) and marked ``estimated``. ``anthropic_llm`` calls the Anthropic SDK directly
and reports the real token counts.

Paid calls need ``ENABLE_LIVE=1`` and ``DM_MAX_USD`` (CLAUDE.md rule 1): ``live_llm``
refuses otherwise, and ``SpendCap`` refuses any call whose worst case (estimated input
plus ``max_tokens`` of output) would take the total over ``DM_MAX_USD``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Callable, Protocol

# USD per million tokens (input, output), Anthropic first-party rates, checked 2026-10-03.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
}
CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class LLMReply:
    """A reply with the provider's own token counts."""

    text: str
    input_tokens: int
    output_tokens: int


class LLM(Protocol):
    def __call__(self, model: str, messages: list[dict], system: str | None = None,
                 max_tokens: int = 4096) -> str | LLMReply: ...


class LiveDisabled(RuntimeError):
    """A paid call was requested without ENABLE_LIVE=1 and DM_MAX_USD."""


class BudgetExceeded(RuntimeError):
    """The next call could take total spend over the cap."""


def price_of(model: str) -> tuple[float, float] | None:
    return PRICES_PER_MTOK.get(model)


def usd_for(model: str, input_tokens: int, output_tokens: int) -> float | None:
    p = price_of(model)
    if p is None:
        return None
    return (input_tokens * p[0] + output_tokens * p[1]) / 1_000_000


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def prompt_tokens(messages: list[dict], system: str | None) -> int:
    return estimate_tokens((system or "") + "".join(str(m["content"]) for m in messages))


def live_settings() -> float:
    """Return DM_MAX_USD, or raise LiveDisabled unless both live variables are set."""
    if os.environ.get("ENABLE_LIVE") != "1":
        raise LiveDisabled("paid LLM calls need ENABLE_LIVE=1 and DM_MAX_USD")
    raw = os.environ.get("DM_MAX_USD")
    try:
        cap = float(raw) if raw is not None else math.nan
    except ValueError:
        cap = math.nan
    if not math.isfinite(cap) or cap < 0:
        raise LiveDisabled(f"DM_MAX_USD must be a non-negative number, got {raw!r}")
    return cap


@dataclass
class SpendCap:
    """Shared across every attempt of one command; refuses calls that could overspend."""

    max_usd: float
    spent_usd: float = 0.0

    def check(self, model: str, messages: list[dict], system: str | None,
              max_tokens: int) -> None:
        worst = usd_for(model, prompt_tokens(messages, system), max_tokens)
        if worst is None:
            raise BudgetExceeded(f"no price known for {model!r}; cannot cap its spend")
        if self.spent_usd + worst > self.max_usd:
            raise BudgetExceeded(
                f"next call could cost up to ${worst:.4f}; spent ${self.spent_usd:.4f} "
                f"of DM_MAX_USD ${self.max_usd:.2f}")

    def add(self, usd: float | None) -> None:
        self.spent_usd += usd or 0.0


@dataclass
class UsageMeter:
    """Wraps an LLM for one attempt: counts calls and tokens, keeps the last exchange."""

    inner: LLM
    cap: SpendCap | None = None
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated: bool = False
    usd: float | None = 0.0
    last_messages: list[dict] = field(default_factory=list)
    last_system: str | None = None
    last_reply: str | None = None

    def __call__(self, model: str, messages: list[dict], system: str | None = None,
                 max_tokens: int = 4096) -> str:
        if self.cap is not None:
            self.cap.check(model, messages, system, max_tokens)
        reply = self.inner(model=model, messages=messages, system=system,
                           max_tokens=max_tokens)
        if isinstance(reply, LLMReply):
            text, tin, tout = reply.text, reply.input_tokens, reply.output_tokens
        else:
            text = str(reply)
            tin, tout = prompt_tokens(messages, system), estimate_tokens(text)
            self.estimated = True
        cost = usd_for(model, tin, tout)
        self.calls += 1
        self.input_tokens += tin
        self.output_tokens += tout
        self.usd = None if self.usd is None or cost is None else self.usd + cost
        if self.cap is not None:
            self.cap.add(cost)
        self.last_messages = [dict(m) for m in messages]
        self.last_system = system
        self.last_reply = text
        return text

    def usage(self) -> dict:
        return {"calls": self.calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "usd": self.usd,
                "estimated": self.estimated}


def anthropic_llm(model: str, messages: list[dict], system: str | None = None,
                  max_tokens: int = 4096) -> LLMReply:
    """One Messages API call with the real token counts.

    No server-side fallback model: a fallback would answer as a different model and
    the attempt would be attributed to the wrong solver. A refusal returns an empty
    text, which the vendor loop treats as a reply without tags.
    """
    import anthropic  # read the key from the environment only

    client = anthropic.Anthropic()
    kwargs: dict = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if system:
        kwargs["system"] = system
    response = client.messages.create(**kwargs)
    text = "".join(b.text for b in response.content if b.type == "text")
    return LLMReply(text=text, input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens)


def vendor_llm() -> Callable[..., str]:
    """The vendor's multi-provider client (text only, so usage is estimated).

    Bound now, because venues swap ``llm_client.complete`` for a meter during an attempt.
    """
    from scienceagent import llm_client

    return llm_client.complete


def live_llm(model: str) -> Callable[..., str | LLMReply]:
    """The paid client for ``model``; raises LiveDisabled unless live mode is on."""
    live_settings()
    return anthropic_llm if model.startswith("claude") else vendor_llm()
