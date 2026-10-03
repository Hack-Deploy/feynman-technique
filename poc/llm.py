"""Anthropic transport with per-call reservations against the live spend ledger."""

from __future__ import annotations

import os
from typing import Callable

from poc.spend import ModelPrice, SpendLedger, usd_for_usage

Transport = Callable[[str, str | None, list[dict], int], tuple[str, dict]]


def anthropic_transport(
    model: str, system: str | None, messages: list[dict], max_tokens: int
) -> tuple[str, dict]:
    import anthropic

    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        timeout=300,
    )
    params = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if system is not None:
        params["system"] = system
    response = client.messages.create(**params)
    text = "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )
    usage = response.usage
    return text, {
        key: int(getattr(usage, key, 0) or 0)
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        )
    }


def redact(message: str) -> str:
    secret = os.environ.get("ANTHROPIC_API_KEY")
    if secret:
        message = message.replace(secret, "[redacted]")
    return message[:1000]


def _message_bytes(messages: list[dict]) -> int:
    return sum(len(str(message.get("content", "")).encode("utf-8")) for message in messages)


class MeteredLLM:
    def __init__(
        self,
        model: str,
        price: ModelPrice,
        ledger: SpendLedger,
        run_key: str,
        transport: Transport | None = None,
    ):
        self.model = model
        self.price = price
        self.ledger = ledger
        self.run_key = run_key
        self.transport = transport or anthropic_transport
        self.calls = 0
        self.usage = {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "usd": 0.0,
        }
        self.usd = 0.0
        self._previous_messages: list[dict] | None = None
        self._previous_system: str | None = None
        self._previous_usage: dict | None = None

    def __call__(
        self,
        model: str,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 3072,
    ) -> str:
        current_messages = [dict(message) for message in messages]
        if (
            self._previous_messages is not None
            and self._previous_system == system
            and current_messages[:len(self._previous_messages)] == self._previous_messages
        ):
            new_messages = current_messages[len(self._previous_messages):]
            input_upper = (
                self._previous_usage["input_tokens"]
                + self._previous_usage["output_tokens"]
                + _message_bytes(new_messages)
                + 8 * len(new_messages)
            )
        else:
            input_upper = (
                len((system or "").encode("utf-8"))
                + _message_bytes(current_messages)
                + 8 * len(current_messages)
            )
        upper_usd = usd_for_usage(
            self.price,
            {"input_tokens": input_upper, "output_tokens": max_tokens},
        )
        reservation_id = self.ledger.reserve(self.run_key, self.model, upper_usd)
        try:
            text, usage = self.transport(model, system, current_messages, max_tokens)
        except Exception as exc:
            try:
                import anthropic
            except ImportError:
                raise
            if isinstance(exc, anthropic.APIStatusError):
                self.ledger.void(reservation_id, "api_error")
            raise

        normalized_usage = {
            key: int(usage.get(key, 0) or 0)
            for key in (
                "input_tokens",
                "output_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            )
        }
        call_usd = usd_for_usage(self.price, normalized_usage)
        self.ledger.settle(
            reservation_id,
            self.run_key,
            self.model,
            normalized_usage,
            call_usd,
        )
        self.calls += 1
        self.usd = round(self.usd + call_usd, 6)
        self.usage["calls"] = self.calls
        self.usage["usd"] = self.usd
        for key, amount in normalized_usage.items():
            self.usage[key] += amount
        self._previous_messages = current_messages
        self._previous_system = system
        self._previous_usage = normalized_usage
        return text
