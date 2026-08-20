"""Provider-neutral LLM layer.

Every free provider worth using (Groq, OpenRouter, Cerebras, Together, Ollama,
Gemini's compat endpoint) speaks the OpenAI chat-completions shape, so there is one
HTTP client and the provider is an env var. A deterministic mock provider drives the
same code path offline, which is what makes the scheduler, governor and checkpoint
logic testable without spending a single token.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import httpx
from pydantic import BaseModel, ValidationError

from .core import (
    Governor,
    Throttled,
    Tier,
    TIERS,
    Usage,
    estimate_tokens,
)


class LLMError(RuntimeError):
    pass


class RateLimited(LLMError):
    """A 429. `daily` separates the two kinds, which need opposite responses: a
    per-minute throttle is waited out, a per-day exhaustion is failed over."""

    def __init__(self, retry_after: float, detail: str = "", daily: bool = False):
        super().__init__(f"429 ({'daily quota' if daily else 'throttle'}): {detail}")
        self.retry_after = retry_after
        self.daily = daily


@dataclass
class Call:
    """One completed model call, with everything needed for a trace span."""
    text: str
    usage: Usage
    model: str
    tier: Tier
    ms: int
    waited_s: float
    attempts: int = 1
    repaired: bool = False


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #


class Provider(Protocol):
    name: str

    async def chat(
        self, *, model: str, system: str, user: str, max_out: int,
        json_mode: bool, role: str,
    ) -> tuple[str, Usage]:
        ...


class OpenAICompatProvider:
    """Chat-completions over HTTP. Works against any OpenAI-compatible base URL.

    Defaults to Groq's free tier; `QUORUM_BASE_URL=http://localhost:11434/v1` with no
    key points the whole system at a local Ollama instead.
    """

    name = "openai-compat"

    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self.base_url = (
            base_url
            or os.getenv("QUORUM_BASE_URL")
            or "https://api.groq.com/openai/v1"
        ).rstrip("/")
        self.api_key = (
            api_key or os.getenv("QUORUM_API_KEY") or os.getenv("GROQ_API_KEY") or ""
        )
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=10.0))

    async def close(self) -> None:
        await self._client.aclose()

    async def list_models(self) -> list[str]:
        r = await self._client.get(f"{self.base_url}/models", headers=self._headers())
        r.raise_for_status()
        return sorted(m["id"] for m in r.json().get("data", []))

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def chat(self, *, model, system, user, max_out, json_mode, role):
        body: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_out,
            "temperature": 0.2,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        r = await self._client.post(
            f"{self.base_url}/chat/completions", headers=self._headers(), json=body
        )
        if r.status_code == 429:
            detail = r.text[:300]
            low = detail.lower()
            daily = "per day" in low or "tpd" in low or "rpd" in low
            raise RateLimited(
                float(r.headers.get("retry-after", "10")), detail, daily
            )
        if r.status_code >= 400:
            raise LLMError(f"{r.status_code} {r.text[:400]}")

        data = r.json()
        text = data["choices"][0]["message"]["content"] or ""
        u = data.get("usage") or {}
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        return text, Usage(
            input_tokens=int(u.get("prompt_tokens", estimate_tokens(system + user))),
            output_tokens=int(u.get("completion_tokens", estimate_tokens(text))),
            cached_tokens=int(cached or 0),
        )


class MockProvider:
    """Deterministic offline provider.

    Fixtures are registered per agent role and keyed off a hash of the prompt, so a
    given question always produces the same plan, the same findings and the same
    verdicts. That is what lets the eval harness measure orchestration behaviour --
    parallel speedup, checkpoint resume, throttle waits, verifier effect -- as a
    reproducible property of the system rather than a property of a sampling seed.
    """

    name = "mock"

    def __init__(self, latency: tuple[float, float] = (0.15, 0.6), fail_rate: float = 0.0):
        self.fixtures: dict[str, Callable[[str, random.Random], Any]] = {}
        self.latency = latency
        self.fail_rate = fail_rate
        self.calls = 0

    def register(self, role: str, fn: Callable[[str, random.Random], Any]) -> None:
        self.fixtures[role] = fn

    async def chat(self, *, model, system, user, max_out, json_mode, role):
        self.calls += 1
        seed = int(hashlib.sha256(f"{role}|{user}".encode()).hexdigest()[:16], 16)
        rng = random.Random(seed)
        await asyncio.sleep(rng.uniform(*self.latency))
        if rng.random() < self.fail_rate:
            raise LLMError(f"mock transient failure in {role}")

        fn = self.fixtures.get(role) or self.fixtures.get(role.split(":", 1)[0])
        if fn is None:
            raise LLMError(f"no mock fixture registered for role {role!r}")
        payload = fn(user, rng)
        text = json.dumps(payload) if json_mode else str(payload)
        return text, Usage(
            input_tokens=estimate_tokens(system + user),
            output_tokens=estimate_tokens(text),
        )


# --------------------------------------------------------------------------- #
# The LLM facade: governance + structured output with a repair loop
# --------------------------------------------------------------------------- #

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> str:
    """Free models fence their JSON, prepend 'Here is the JSON:', or emit a leading
    reasoning paragraph roughly a third of the time. Salvage the object rather than
    burning a whole repair round-trip on formatting."""
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text


SCHEMA_INSTRUCTION = (
    "Reply with a single JSON object and nothing else. No prose, no markdown fence.\n"
    "It must validate against this JSON Schema:\n{schema}"
)


@dataclass
class LLMStats:
    calls: int = 0
    json_first_pass: int = 0
    json_repaired: int = 0
    json_failed: int = 0
    errors: int = 0

    @property
    def first_pass_rate(self) -> float:
        n = self.json_first_pass + self.json_repaired + self.json_failed
        return self.json_first_pass / n if n else 0.0

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "json_first_pass": self.json_first_pass,
            "json_repaired": self.json_repaired,
            "json_failed": self.json_failed,
            "json_first_pass_rate": round(self.first_pass_rate, 3),
            "errors": self.errors,
        }


class LLM:
    """Governed, schema-validating wrapper around a Provider.

    Responsibilities, in order: admission through the Governor, the HTTP call with
    429 backoff, JSON salvage, pydantic validation, and one bounded repair turn that
    feeds the validation error back to the model. Everything is metered.
    """

    def __init__(self, provider: Provider, governor: Governor, *, max_repairs: int = 1):
        self.provider = provider
        self.gov = governor
        self.max_repairs = max_repairs
        self.stats = LLMStats()

    async def complete(
        self, *, role: str, system: str, user: str, tier: Tier,
        max_out: int | None = None, json_mode: bool = False,
    ) -> Call:
        spec = TIERS[tier]
        max_out = max_out or spec.max_out
        est_in = estimate_tokens(system) + estimate_tokens(user)

        for attempt in range(4):
            try:
                grant = await self.gov.admit(tier, est_in, max_out)
            except Throttled:
                if attempt == 3:
                    self.stats.errors += 1
                    raise
                await asyncio.sleep(1.0 + attempt)
                continue
            t0 = time.monotonic()
            try:
                text, usage = await self.provider.chat(
                    model=grant.model, system=system, user=user,
                    max_out=max_out, json_mode=json_mode, role=role,
                )
            except RateLimited as e:
                await self.gov.on_429(grant.tier, e.retry_after, e.daily)
                if e.daily:
                    # Do not burn an attempt waiting; re-admit immediately so the
                    # governor routes to a model that still has quota.
                    continue
                if attempt == 3:
                    self.stats.errors += 1
                    raise
                continue
            except Exception:
                self.stats.errors += 1
                raise
            await self.gov.settle(grant, usage)
            self.stats.calls += 1
            return Call(
                text=text, usage=usage, model=grant.model, tier=grant.tier,
                ms=int((time.monotonic() - t0) * 1000), waited_s=grant.waited_s,
            )
        raise LLMError("exhausted rate-limit retries")

    async def structured(
        self, *, role: str, system: str, user: str, tier: Tier,
        out_model: type[BaseModel], max_out: int | None = None,
    ) -> tuple[BaseModel, Call]:
        schema = json.dumps(out_model.model_json_schema(), separators=(",", ":"))
        sys_full = f"{system}\n\n{SCHEMA_INSTRUCTION.format(schema=schema)}"

        call = await self.complete(
            role=role, system=sys_full, user=user, tier=tier,
            max_out=max_out, json_mode=True,
        )
        total = call.usage
        err: str | None = None

        for repair in range(self.max_repairs + 1):
            try:
                obj = out_model.model_validate_json(extract_json(call.text))
            except (ValidationError, ValueError) as e:
                err = str(e)[:600]
                if repair == self.max_repairs:
                    break
                fix = await self.complete(
                    role=role, tier=tier, system=sys_full,
                    user=(
                        f"{user}\n\n---\nYour previous reply did not validate.\n"
                        f"Previous reply:\n{call.text[:2000]}\n\n"
                        f"Validation error:\n{err}\n\nReturn corrected JSON only."
                    ),
                    max_out=max_out, json_mode=True,
                )
                total = total + fix.usage
                call = fix
                continue

            call.usage = total
            call.attempts = repair + 1
            call.repaired = repair > 0
            if repair:
                self.stats.json_repaired += 1
            else:
                self.stats.json_first_pass += 1
            return obj, call

        self.stats.json_failed += 1
        raise LLMError(f"{role}: schema validation failed after repair -- {err}")


def build_provider(mock: bool = False, **kw) -> Provider:
    if mock or os.getenv("QUORUM_MOCK") == "1":
        return MockProvider(**kw)
    return OpenAICompatProvider()
