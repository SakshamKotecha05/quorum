"""Core primitives: model tiers, token-bucket rate limiting, the run governor,
DAG node types, the SQLite checkpoint store, and BM25 corpus retrieval.

No network calls happen in this module.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sqlite3
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# --------------------------------------------------------------------------- #
# Model tiers
#
# Model IDs churn (Groq deprecated the Llama line during this project's life), so
# tiers are resolved from the environment at import time rather than hardcoded.
# `python -m quorum models` lists what the configured provider actually serves.
# --------------------------------------------------------------------------- #


class Tier(str, Enum):
    DEEP = "deep"   # planning + synthesis: needs reasoning
    MID = "mid"     # verification: needs judgement, runs at high fan-out
    FAST = "fast"   # extraction/reformatting: needs throughput


@dataclass(frozen=True)
class ModelSpec:
    """A model plus the rate ceiling the provider enforces on it.

    rpm/tpm default to the Groq free-tier floor (30 req/min, 6k tok/min). These are
    *per model*; the org-wide ceiling is enforced separately by the global limiter.
    """
    name: str
    rpm: int = 30
    tpm: int = 6_000
    max_out: int = 4_096


def _spec(env: str, default: str, tpm: int = 6_000) -> ModelSpec:
    return ModelSpec(
        name=os.getenv(f"QUORUM_MODEL_{env}", default),
        rpm=int(os.getenv(f"QUORUM_RPM_{env}", "30")),
        tpm=int(os.getenv(f"QUORUM_TPM_{env}", str(tpm))),
    )


TIERS: dict[Tier, ModelSpec] = {
    Tier.DEEP: _spec("DEEP", "openai/gpt-oss-120b", 8_000),
    Tier.MID: _spec("MID", "openai/gpt-oss-20b", 8_000),
    Tier.FAST: _spec("FAST", "openai/gpt-oss-20b", 8_000),
}

# What a tier degrades to when its own rate ceiling is saturated but a cheaper
# model still has headroom. Degrading trades answer quality for forward progress.
DOWNGRADE: dict[Tier, Tier | None] = {
    Tier.DEEP: Tier.MID,
    Tier.MID: Tier.FAST,
    Tier.FAST: None,
}

# Shadow cost accounting. Free tiers cost $0, so these default to zero and the
# governor meters *tokens*, which are the real scarce resource here. Drop a
# prices.json of {"model-id": [usd_per_1m_in, usd_per_1m_out]} beside the db to
# see what an equivalent paid run would have cost.
def _load_prices() -> dict[str, tuple[float, float]]:
    p = Path(os.getenv("QUORUM_PRICES", "prices.json"))
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    return {k: (float(v[0]), float(v[1])) for k, v in raw.items()}


PRICES = _load_prices()


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0  # provider-reported prompt cache hits, if any

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cached_tokens + other.cached_tokens,
        )

    def cost(self, model: str) -> float:
        c_in, c_out = PRICES.get(model, (0.0, 0.0))
        billable_in = max(0, self.input_tokens - self.cached_tokens)
        return (billable_in * c_in + self.output_tokens * c_out) / 1_000_000


def estimate_tokens(text: str) -> int:
    """Pre-call token estimate for admission control.

    ponytail: chars/4 heuristic, not a real tokenizer. Admission runs before every
    call and a tokenizer round-trip would cost more than the error. The limiter
    settles on provider-reported usage afterwards, so drift self-corrects within a
    minute. Swap in tiktoken only if observed overshoot exceeds ~10%.
    """
    return max(1, len(text) // 4)


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #


class ContextTooLarge(Exception):
    """A single call is bigger than the per-minute token ceiling, so no amount of
    waiting will admit it. The caller must trim context, not retry."""


class BudgetExceeded(Exception):
    """The run hit its own token or request cap."""


class Exhausted(BudgetExceeded):
    """A model's daily quota is gone. Unlike a per-minute throttle this cannot be
    waited out inside a run, so it must trigger failover, never a retry."""


class Throttled(Exception):
    """Waited past the ceiling for admission. The node fails and the scheduler
    retries it later rather than the run hanging on a saturated bucket."""


class TokenBucket:
    def __init__(self, capacity: float, per_second: float):
        self.capacity = float(capacity)
        self.per_second = float(per_second)
        self.tokens = float(capacity)
        self._ts = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self._ts) * self.per_second)
        self._ts = now

    def wait_for(self, n: float) -> float:
        """Seconds until `n` is available. 0.0 means available now. Does not consume."""
        self._refill()
        if n <= self.tokens:
            return 0.0
        return (n - self.tokens) / self.per_second

    def take(self, n: float) -> None:
        self.tokens -= n


class RateLimiter:
    """Paired request/token buckets with 429 backoff.

    Both buckets must clear before a call is admitted, and they are taken together
    under one lock so concurrent agents cannot interleave into an overdraft. A 429
    from the provider parks every waiter until `retry-after` elapses -- retrying
    individually is what turns one 429 into a cascade.
    """

    def __init__(self, rpm: int, tpm: int, name: str = "global"):
        self.name = name
        self.req = TokenBucket(rpm, rpm / 60.0)
        self.tok = TokenBucket(tpm, tpm / 60.0)
        self.waits = 0
        self.wait_seconds = 0.0
        self.throttle_429 = 0
        self.exhausted_until = 0.0
        self._penalty_until = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self, est_tokens: int, max_wait_s: float = 120.0) -> float:
        if est_tokens > self.tok.capacity:
            raise ContextTooLarge(
                f"{self.name}: call needs {est_tokens} tok but ceiling is "
                f"{int(self.tok.capacity)} tok/min -- trim context"
            )
        waited = 0.0
        while True:
            if waited >= max_wait_s:
                raise Throttled(
                    f"{self.name}: still throttled after {waited:.0f}s"
                )
            async with self._lock:
                pen = self._penalty_until - time.monotonic()
                if pen <= 0:
                    w = max(self.req.wait_for(1), self.tok.wait_for(est_tokens))
                    if w <= 0:
                        self.req.take(1)
                        self.tok.take(est_tokens)
                        if waited:
                            self.waits += 1
                            self.wait_seconds += waited
                        return waited
                else:
                    w = pen
            sleep = min(w, 5.0)
            await asyncio.sleep(sleep)
            waited += sleep

    async def settle(self, estimated: int, actual: int) -> None:
        """Reconcile the estimate against provider-reported usage."""
        async with self._lock:
            self.tok.take(max(0, actual - estimated))

    async def penalize(self, retry_after: float, daily: bool = False) -> None:
        async with self._lock:
            self.throttle_429 += 1
            if daily:
                # A daily quota does not refill on any timescale this run cares
                # about. Park the bucket permanently so admission routes around it.
                self.exhausted_until = math.inf
            else:
                self._penalty_until = max(
                    self._penalty_until, time.monotonic() + retry_after
                )

    def headroom(self, est_tokens: int) -> float:
        """Seconds a call of this size would have to wait. Used to pick a tier with
        free capacity instead of queueing behind a saturated one."""
        if self.exhausted_until == math.inf or est_tokens > self.tok.capacity:
            return math.inf
        return max(self.req.wait_for(1), self.tok.wait_for(est_tokens))


@dataclass
class Grant:
    tier: Tier
    model: str
    est_tokens: int
    waited_s: float = 0.0
    downgraded_from: Tier | None = None


class Governor:
    """Admission control for a single run.

    Enforces three things at once:
      1. per-run hard caps on tokens and requests, so a runaway plan cannot eat a
         14,400 req/day quota in one go;
      2. per-model rate ceilings, with tier downgrade when the requested tier is
         saturated and a cheaper one is free;
      3. the org-wide ceiling, which every model shares.
    """

    def __init__(
        self,
        max_tokens: int = 200_000,
        max_requests: int = 200,
        org_rpm: int = 30,
        org_tpm: int = 6_000,
        model_rpm: int | None = None,
        model_tpm: int | None = None,
    ):
        self.max_tokens = max_tokens
        self.max_requests = max_requests
        self.tokens_used = 0
        self.requests_used = 0
        self.usd_shadow = 0.0
        self.downgrades = 0
        self.org = RateLimiter(org_rpm, org_tpm, "org")
        # Reserving `max_tokens` as the output estimate is the intuitive choice and
        # it deadlocks the run: a 4k cap against a 6k/min ceiling admits one call a
        # minute regardless of how big the replies actually are. Track what each tier
        # really emits and reserve from that, with headroom.
        self._out_ema: dict[Tier, float] = {t: 400.0 for t in TIERS}
        # Tier aliases share quota. If aliases declare different ceilings, use
        # the strictest one rather than multiplying the provider's allowance.
        self.per_model = {
            name: RateLimiter(
                model_rpm or min(s.rpm for s in TIERS.values() if s.name == name),
                model_tpm or min(s.tpm for s in TIERS.values() if s.name == name),
                name,
            )
            for name in dict.fromkeys(s.name for s in TIERS.values())
        }
        self._lock = asyncio.Lock()

    async def admit(self, tier: Tier, est_in: int, max_out: int) -> Grant:
        est_out = int(min(max_out, max(64.0, self._out_ema[tier] * 1.5)))
        est = est_in + est_out
        async with self._lock:
            if self.requests_used + 1 > self.max_requests:
                raise BudgetExceeded(f"run request cap {self.max_requests} reached")
            if self.tokens_used + est > self.max_tokens:
                raise BudgetExceeded(
                    f"run token cap {self.max_tokens} would be exceeded "
                    f"({self.tokens_used} used, {est} requested)"
                )
            self.requests_used += 1

        # Pick the requested tier, or degrade to one that is neither saturated nor
        # out of daily quota. A tier whose day is spent is skipped unconditionally;
        # a merely busy tier is skipped only if a cheaper one is idle.
        chosen, wanted = tier, tier
        while chosen is not None:
            if self.per_model[TIERS[chosen].name].headroom(est) < 5.0:
                break
            nxt = DOWNGRADE[chosen]
            if nxt is None:
                break
            chosen = nxt
        if chosen is None or self.per_model[TIERS[chosen].name].exhausted_until == math.inf:
            alive = [t for t in TIERS if self.per_model[TIERS[t].name].exhausted_until != math.inf]
            if not alive:
                raise Exhausted(
                    "every configured model is out of daily quota; try again "
                    "tomorrow or point QUORUM_MODEL_* at different models"
                )
            chosen = alive[0] if chosen is None or chosen not in alive else chosen
        if chosen is not wanted:
            self.downgrades += 1

        waited = await self.per_model[TIERS[chosen].name].acquire(est)
        waited += await self.org.acquire(est)
        return Grant(
            tier=chosen,
            model=TIERS[chosen].name,
            est_tokens=est,
            waited_s=waited,
            downgraded_from=wanted if chosen is not wanted else None,
        )

    async def settle(self, grant: Grant, usage: Usage) -> None:
        await self.per_model[grant.model].settle(grant.est_tokens, usage.total)
        await self.org.settle(grant.est_tokens, usage.total)
        async with self._lock:
            self.tokens_used += usage.total
            self.usd_shadow += usage.cost(grant.model)
            prev = self._out_ema[grant.tier]
            self._out_ema[grant.tier] = 0.7 * prev + 0.3 * usage.output_tokens

    async def on_429(self, tier: Tier, retry_after: float, daily: bool = False) -> None:
        await self.per_model[TIERS[tier].name].penalize(retry_after, daily)
        # A per-model daily exhaustion says nothing about the org-wide bucket, so
        # never park every other model because one ran out of its own quota.
        if not daily:
            await self.org.penalize(retry_after)

    def concurrency(self, avg_call_tokens: int) -> int:
        """Adaptive concurrency: how many calls of the observed average size can be
        in flight without immediately queueing behind the org token ceiling.

        A fixed semaphore is the usual choice and it is wrong here -- 8 parallel
        researchers at 2k tokens each is 16k tokens against a 6k/min ceiling, so
        seven of them sit in backoff and the run is slower than sequential.
        """
        if avg_call_tokens <= 0:
            return 4
        return max(1, min(8, int(self.org.tok.capacity // avg_call_tokens)))

    @property
    def _limiters(self) -> list[RateLimiter]:
        return [self.org, *self.per_model.values()]

    def snapshot(self) -> dict:
        # Aggregate across every bucket. Reporting only the org limiter hides the
        # case where a per-model ceiling is the binding constraint, which is exactly
        # the case that makes a run mysteriously slower than its model time.
        waits = sum(l.waits for l in self._limiters)
        wait_s = sum(l.wait_seconds for l in self._limiters)
        return {
            "tokens_used": self.tokens_used,
            "requests_used": self.requests_used,
            "token_cap": self.max_tokens,
            "request_cap": self.max_requests,
            "usd_shadow": round(self.usd_shadow, 6),
            "tier_downgrades": self.downgrades,
            "throttle_waits": waits,
            "throttle_wait_s": round(wait_s, 2),
            "http_429": sum(l.throttle_429 for l in self._limiters),
            "out_tok_ema": {t.value: int(v) for t, v in self._out_ema.items()},
            "exhausted_models": [
                name for name, l in self.per_model.items()
                if l.exhausted_until == math.inf
            ],
        }


# --------------------------------------------------------------------------- #
# DAG nodes
# --------------------------------------------------------------------------- #


class Status(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Node:
    id: str
    role: str
    payload: dict
    depends_on: list[str] = field(default_factory=list)
    status: Status = Status.PENDING
    output: dict | None = None
    error: str | None = None
    attempts: int = 0

    def to_row(self) -> tuple:
        return (
            self.id,
            self.role,
            json.dumps(self.payload),
            json.dumps(self.depends_on),
            str(self.status.value),
            json.dumps(self.output) if self.output is not None else None,
            self.error,
            self.attempts,
        )


class CycleError(Exception):
    """The planner emitted a dependency cycle or a dangling reference."""


def validate_dag(nodes: dict[str, Node]) -> list[str]:
    """Kahn's algorithm. Returns a topological order; raises on cycles or unknown
    dependencies. Runs before execution so a malformed plan fails in microseconds
    instead of after we have already spent half the token budget on it."""
    indeg = {nid: 0 for nid in nodes}
    children: dict[str, list[str]] = {nid: [] for nid in nodes}
    for nid, node in nodes.items():
        for dep in node.depends_on:
            if dep not in nodes:
                raise CycleError(f"node {nid!r} depends on unknown node {dep!r}")
            indeg[nid] += 1
            children[dep].append(nid)

    queue = sorted(n for n, d in indeg.items() if d == 0)
    order: list[str] = []
    while queue:
        nid = queue.pop(0)
        order.append(nid)
        for child in children[nid]:
            indeg[child] -= 1
            if indeg[child] == 0:
                queue.append(child)
    if len(order) != len(nodes):
        raise CycleError(f"dependency cycle among {sorted(set(nodes) - set(order))}")
    return order


# --------------------------------------------------------------------------- #
# Checkpoint store
# --------------------------------------------------------------------------- #

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, question TEXT, status TEXT,
  governor TEXT, created_at REAL, updated_at REAL);
CREATE TABLE IF NOT EXISTS nodes (
  run_id TEXT, node_id TEXT, role TEXT, payload TEXT, depends_on TEXT,
  status TEXT, output TEXT, error TEXT, attempts INTEGER,
  PRIMARY KEY (run_id, node_id));
CREATE TABLE IF NOT EXISTS spans (
  span_id TEXT PRIMARY KEY, run_id TEXT, node_id TEXT, role TEXT, model TEXT,
  tier TEXT, in_tok INTEGER, out_tok INTEGER, cached_tok INTEGER,
  waited_s REAL, ms INTEGER, ok INTEGER, err TEXT, ts REAL);
CREATE INDEX IF NOT EXISTS spans_run ON spans(run_id);
"""


class Store:
    """SQLite checkpoint + trace store.

    ponytail: one connection behind a threading.Lock. Writes are sub-millisecond and
    a run makes tens of them, not thousands. Revisit only if a profile shows lock
    contention -- then a single writer task with a queue, not Postgres.
    """

    def __init__(self, path: str | Path = "quorum.db"):
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self.db.commit()
        self._lock = threading.Lock()

    def _write(self, sql: str, args: tuple) -> None:
        with self._lock:
            self.db.execute(sql, args)
            self.db.commit()

    def save_run(self, run_id, question, status, governor: dict) -> None:
        now = time.time()
        self._write(
            "INSERT INTO runs VALUES (?,?,?,?,?,?) ON CONFLICT(run_id) DO UPDATE SET "
            "status=excluded.status, governor=excluded.governor, updated_at=excluded.updated_at",
            (run_id, question, status, json.dumps(governor), now, now),
        )

    def save_node(self, run_id: str, node: Node) -> None:
        self._write(
            "INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(run_id, node_id) DO UPDATE SET status=excluded.status, "
            "output=excluded.output, error=excluded.error, attempts=excluded.attempts",
            (run_id, *node.to_row()),
        )

    def load_nodes(self, run_id: str) -> dict[str, Node]:
        cur = self.db.execute(
            "SELECT node_id, role, payload, depends_on, status, output, error, attempts "
            "FROM nodes WHERE run_id=?",
            (run_id,),
        )
        out: dict[str, Node] = {}
        for nid, role, payload, deps, status, output, error, attempts in cur:
            out[nid] = Node(
                id=nid, role=role, payload=json.loads(payload),
                depends_on=json.loads(deps), status=Status(status),
                output=json.loads(output) if output else None,
                error=error, attempts=attempts,
            )
        return out

    def load_run(self, run_id: str) -> dict | None:
        row = self.db.execute(
            "SELECT run_id, question, status, governor FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if not row:
            return None
        return {
            "run_id": row[0], "question": row[1], "status": row[2],
            "governor": json.loads(row[3] or "{}"),
        }

    def list_runs(self, limit: int = 20) -> list[dict]:
        cur = self.db.execute(
            "SELECT run_id, question, status, updated_at FROM runs "
            "ORDER BY updated_at DESC LIMIT ?", (limit,)
        )
        return [dict(zip(("run_id", "question", "status", "updated_at"), r)) for r in cur]

    def add_span(self, run_id, node_id, role, model, tier, usage: Usage,
                 waited_s, ms, ok, err=None) -> None:
        self._write(
            "INSERT INTO spans VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (uuid.uuid4().hex, run_id, node_id, role, model, str(tier.value),
             usage.input_tokens, usage.output_tokens, usage.cached_tokens,
             waited_s, ms, int(ok), err, time.time()),
        )

    def trace(self, run_id: str) -> list[dict]:
        keys = ("node_id", "role", "model", "tier", "in_tok", "out_tok", "cached_tok",
                "waited_s", "ms", "ok", "err")
        cur = self.db.execute(
            f"SELECT {','.join(keys)} FROM spans WHERE run_id=? ORDER BY ts", (run_id,)
        )
        return [dict(zip(keys, r)) for r in cur]


# --------------------------------------------------------------------------- #
# Corpus retrieval (BM25)
# --------------------------------------------------------------------------- #

_WORD = re.compile(r"[a-z0-9]+")
_K1, _B = 1.5, 0.75


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _chunk(text: str, size: int) -> list[str]:
    out: list[str] = []
    buf = ""
    for para in (p.strip() for p in text.split("\n\n")):
        if not para:
            continue
        if buf and len(buf) + len(para) > size:
            out.append(buf)
            buf = para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf:
        out.append(buf)
    return out


@dataclass
class Chunk:
    source_id: str
    text: str
    score: float = 0.0


class Corpus:
    """BM25 over a directory of .md/.txt files, chunked on paragraph boundaries.

    ponytail: stdlib BM25, no FAISS/Chroma/embeddings. A few thousand chunks scan in
    under a millisecond, there is no index to build or keep in sync, and lexical
    match is the right prior for citation-grounded research where the verifier needs
    the literal quote to exist in the source. Upgrade path: swap `search` for a
    hybrid dense retriever when recall@5 on the golden set measurably drops.
    """

    def __init__(self, path: str | Path | None = None, chunk_chars: int = 1200):
        self.chunks: list[Chunk] = []
        self._tf: list[Counter] = []
        self._df: Counter = Counter()
        self._lens: list[int] = []
        self._avgdl = 1.0
        self.chunk_chars = chunk_chars
        if path and Path(path).exists():
            self.load(path)

    def add(self, source_id: str, text: str) -> None:
        toks = _tokens(text)
        self.chunks.append(Chunk(source_id, text))
        self._tf.append(Counter(toks))
        self._lens.append(len(toks))
        self._df.update(set(toks))
        self._avgdl = sum(self._lens) / max(1, len(self._lens))

    def load(self, path: str | Path) -> None:
        for f in sorted(Path(path).rglob("*")):
            if f.suffix.lower() not in {".md", ".txt"}:
                continue
            for i, ch in enumerate(_chunk(f.read_text(errors="ignore"), self.chunk_chars)):
                self.add(f"{f.stem}#{i}", ch)

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        if not self.chunks:
            return []
        n = len(self.chunks)
        terms = set(_tokens(query))
        scored: list[Chunk] = []
        for i, tf in enumerate(self._tf):
            score = 0.0
            for term in terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                df = self._df[term]
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                denom = f + _K1 * (1 - _B + _B * self._lens[i] / self._avgdl)
                score += idf * f * (_K1 + 1) / denom
            if score > 0:
                c = self.chunks[i]
                scored.append(Chunk(c.source_id, c.text, round(score, 4)))
        scored.sort(key=lambda c: -c.score)
        return scored[:k]

    def get(self, source_id: str) -> Chunk | None:
        return next((c for c in self.chunks if c.source_id == source_id), None)
