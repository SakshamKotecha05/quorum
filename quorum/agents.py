"""The four agent roles, their output schemas, and the deterministic mock fixtures.

Roles and why each sits on the tier it does:

  planner      DEEP  one call, decides the shape of everything downstream
  researcher   MID   high fan-out, the bulk of the token spend
  verifier     MID   judgement, but batched so 30 RPM is survivable
  synthesizer  DEEP  one call, writes the artefact the user reads
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

from pydantic import BaseModel, Field

from .core import Corpus, Store, Tier
from .llm import LLM, MockProvider

# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #


class Subtask(BaseModel):
    id: str = Field(description="short slug, e.g. 's1'")
    question: str
    depends_on: list[str] = Field(
        description="ids of subtasks whose findings this one needs; [] if independent"
    )


class Plan(BaseModel):
    subtasks: list[Subtask]


class Claim(BaseModel):
    id: str
    text: str = Field(description="one factual assertion, self-contained")
    source_id: str = Field(description="exact source id of the excerpt used")
    quote: str = Field(description="verbatim span from that source supporting the claim")


class Findings(BaseModel):
    summary: str
    claims: list[Claim]


class ClaimVerdict(BaseModel):
    claim_id: str
    supported: bool
    reason: str


class VerdictBatch(BaseModel):
    verdicts: list[ClaimVerdict]


class Report(BaseModel):
    markdown: str
    open_questions: list[str]


# --------------------------------------------------------------------------- #
# Context
# --------------------------------------------------------------------------- #


@dataclass
class AgentContext:
    llm: LLM
    corpus: Corpus
    store: Store
    run_id: str
    emit: Callable[[str, dict], None] = lambda ev, data: None


# --------------------------------------------------------------------------- #
# Deterministic pre-filter: the cheapest verifier is not a model
# --------------------------------------------------------------------------- #

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s.lower()).strip()


def quote_is_grounded(claim: Claim, corpus: Corpus) -> bool:
    """Does the cited source exist, and does the quote literally occur in it?

    This runs before any verifier call. A fabricated citation is caught by string
    containment for free, so the model quorum only ever spends tokens on claims that
    are at least anchored to real text. Catching hallucinated citations with an LLM
    is paying for a `in` operator.
    """
    chunk = corpus.get(claim.source_id)
    if chunk is None:
        return False
    return _norm(claim.quote) in _norm(chunk.text)


def prefilter(claims: list[Claim], corpus: Corpus) -> tuple[list[Claim], list[Claim]]:
    kept, dropped = [], []
    for c in claims:
        (kept if quote_is_grounded(c, corpus) else dropped).append(c)
    return kept, dropped


# --------------------------------------------------------------------------- #
# Agents
# --------------------------------------------------------------------------- #

PLANNER_SYS = """You are the supervisor of a research system. You decompose a \
research question into independent sub-questions that specialist agents will answer \
in parallel against a document corpus.

Rules:
- Emit between 3 and {max_subtasks} subtasks.
- Prefer independent subtasks; parallelism is the point. Use depends_on only when a \
subtask genuinely cannot be written without another's findings.
- depends_on must reference ids you define in this same plan. No cycles.
- Each question must be answerable from documents, and narrow enough that one agent \
can answer it in a few hundred words."""


async def plan(ctx: AgentContext, question: str, max_subtasks: int = 5) -> Plan:
    obj, call = await ctx.llm.structured(
        role="planner",
        system=PLANNER_SYS.format(max_subtasks=max_subtasks),
        user=f"Research question:\n{question}",
        tier=Tier.DEEP,
        out_model=Plan,
    )
    ctx.store.add_span(ctx.run_id, "plan", "planner", call.model, call.tier,
                       call.usage, call.waited_s, call.ms, True)
    ctx.emit("plan", {"subtasks": len(obj.subtasks)})
    return obj


RESEARCHER_SYS = """You are a research agent. Answer the sub-question using ONLY the \
numbered excerpts provided.

Every claim you make must carry the exact source_id of the excerpt it came from and a \
verbatim quote copied character-for-character from that excerpt. Do not paraphrase \
inside `quote`. If the excerpts do not answer the sub-question, return zero claims and \
say so in the summary. Returning nothing is a correct answer; inventing a citation is \
not."""


async def research(
    ctx: AgentContext, node_id: str, subtask: Subtask,
    upstream: list[str], k: int = 5,
) -> Findings:
    hits = ctx.corpus.search(subtask.question, k=k)
    excerpts = "\n\n".join(
        f"[source_id: {h.source_id}]\n{h.text}" for h in hits
    ) or "(no excerpts matched)"

    # ponytail: upstream context is passed as summaries, never full findings. Under a
    # 6k tok/min ceiling, forwarding raw upstream claims is what makes a 5-node plan
    # cost more than the ceiling allows in a single minute.
    prior = ""
    if upstream:
        prior = "\n\nFindings from prerequisite subtasks:\n" + "\n".join(
            f"- {s}" for s in upstream
        )

    obj, call = await ctx.llm.structured(
        role="researcher",
        system=RESEARCHER_SYS,
        user=f"Sub-question:\n{subtask.question}{prior}\n\nExcerpts:\n{excerpts}",
        tier=Tier.MID,
        out_model=Findings,
    )
    for i, c in enumerate(obj.claims):
        c.id = f"{node_id}.c{i}"
    ctx.store.add_span(ctx.run_id, node_id, "researcher", call.model, call.tier,
                       call.usage, call.waited_s, call.ms, True)
    ctx.emit("findings", {"node": node_id, "claims": len(obj.claims)})
    return obj


LENSES: dict[str, str] = {
    "support": (
        "Judge only whether the QUOTE literally supports the CLAIM. A claim that is "
        "true in the world but not stated in the quote is NOT supported."
    ),
    "overreach": (
        "Judge whether the CLAIM adds quantification, causality, generality or "
        "certainty that the QUOTE does not state. Any such addition is NOT supported."
    ),
    "attribution": (
        "Judge whether the QUOTE is the right evidence for this CLAIM -- whether it is "
        "on-topic and specific rather than incidentally overlapping wording."
    ),
}

VERIFIER_SYS = """You are a verification agent. For each numbered item you are given a \
CLAIM and the QUOTE it cites.

{lens}

Return one verdict per claim_id, in the same order. Default to supported=false when \
you are unsure. Being wrong in the strict direction costs a true claim; being wrong in \
the permissive direction ships a hallucination."""


async def verify_batch(
    ctx: AgentContext, node_id: str, lens: str, claims: list[Claim]
) -> dict[str, bool]:
    items = "\n\n".join(
        f"{i+1}. claim_id: {c.id}\nCLAIM: {c.text}\nQUOTE: {c.quote}"
        for i, c in enumerate(claims)
    )
    obj, call = await ctx.llm.structured(
        # The lens must be part of the role, not just the system prompt: it is what
        # gives mock lenses distinct error seeds. Live calls use different criteria
        # but may share a model, so their errors are not necessarily independent.
        role=f"verifier:{lens}",
        system=VERIFIER_SYS.format(lens=LENSES[lens]),
        user=items,
        tier=Tier.MID,
        out_model=VerdictBatch,
    )
    ctx.store.add_span(ctx.run_id, node_id, f"verifier:{lens}", call.model, call.tier,
                       call.usage, call.waited_s, call.ms, True)
    return {v.claim_id: v.supported for v in obj.verdicts}


def unescape_newlines(md: str) -> str:
    """Some models double-escape newlines inside the JSON string, so the markdown
    arrives with literal backslash-n instead of line breaks and renders as one blob.
    Only rewrite when there are no real newlines at all, so a correctly formatted
    reply that happens to discuss "\\n" is left alone."""
    if "\\n" in md and "\n" not in md:
        return md.replace("\\n", "\n")
    return md


def batch(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


SYNTH_SYS = """You are the synthesis agent. Write the final research brief in markdown \
from the verified claims you are given.

Every factual sentence must cite the source_id it came from, inline, as [source_id]. \
Do not introduce any fact that is not present in the verified claims -- you have no \
other knowledge for this task. Where the claims are thin or conflicting, say so in \
open_questions instead of papering over it."""


async def synthesize(
    ctx: AgentContext, question: str, claims: list[Claim], rejected: int
) -> Report:
    body = "\n".join(f"- [{c.source_id}] {c.text}" for c in claims) or "(none survived)"
    obj, call = await ctx.llm.structured(
        role="synthesizer",
        system=SYNTH_SYS,
        user=(
            f"Research question:\n{question}\n\nVerified claims:\n{body}\n\n"
            f"({rejected} claims were rejected in verification and must not appear.)"
        ),
        tier=Tier.DEEP,
        out_model=Report,
        max_out=3000,
    )
    ctx.store.add_span(ctx.run_id, "synth", "synthesizer", call.model, call.tier,
                       call.usage, call.waited_s, call.ms, True)
    obj.markdown = unescape_newlines(obj.markdown)
    return obj


# --------------------------------------------------------------------------- #
# Mock fixtures -- fault injection harness
# --------------------------------------------------------------------------- #


def register_mocks(
    mock: MockProvider,
    corpus: Corpus,
    *,
    p_fabricated: float = 0.20,
    p_overreach: float = 0.20,
    judge_error: float = 0.20,
    claims_per_agent: int = 4,
) -> dict[str, bool]:
    """Wire deterministic fixtures that inject two *different* known defects, and
    return the ground-truth map keyed by (claim text, quote).

    Two failure modes, because they are caught by different machinery:

      fabricated  the quote appears in no source. Caught by string containment in
                  `prefilter`, at zero token cost. Cheap to catch, so an eval that
                  only injects this makes the model verifier look worthless.
      overreach   the quote is real and correctly cited, but the claim asserts more
                  than the quote supports. String matching cannot see this at all.
                  This is the case the verifier quorum exists for, and it is the
                  common failure in real citation-grounded systems.

    `judge_error` corrupts each lens independently, so the harness can measure what a
    3-lens majority buys over one noisy judge. Fault injection with a known defect
    rate is how a verifier gets evaluated without first hand-labelling a corpus.
    """
    truth: dict[str, bool] = {}

    def _key(text: str, quote: str) -> str:
        return f"{_norm(text)}||{_norm(quote)}"

    def _planner(user: str, rng: random.Random) -> dict:
        q = user.split("\n", 1)[-1].strip().rstrip("?")
        stems = [
            "What is the current state of", "What are the main constraints on",
            "What evidence exists about", "What are the tradeoffs in",
            "What do sources disagree about regarding",
        ]
        n = rng.randint(3, 5)
        subs = [
            {"id": f"s{i+1}", "question": f"{stems[i % len(stems)]} {q}?",
             "depends_on": []}
            for i in range(n)
        ]
        if n >= 3:  # one real dependency edge, so the DAG is not a flat fan-out
            subs[-1]["depends_on"] = [subs[0]["id"]]
        return {"subtasks": subs}

    def _researcher(user: str, rng: random.Random) -> dict:
        ids = re.findall(r"\[source_id: ([^\]]+)\]", user)
        claims = []
        for i in range(claims_per_agent):
            roll = rng.random()
            if not ids or roll < p_fabricated:
                text = f"Assertion {i} resting on a citation that does not exist."
                quote = f"this exact sentence appears in no source document {i}."
                sid = ids[0] if ids else "ghost#0"
                ok = False
            else:
                sid = rng.choice(ids)
                chunk = corpus.get(sid)
                words = (chunk.text if chunk else "").split()
                start = rng.randint(0, max(0, len(words) - 14))
                quote = " ".join(words[start : start + 14])
                if roll < p_fabricated + p_overreach:
                    text = (f"Finding {i} from {sid}, which holds universally and "
                            f"causes the outcome in every case.")
                    ok = False
                else:
                    text = f"Finding {i} reported in {sid}."
                    ok = True
            truth[_key(text, quote)] = ok
            claims.append({"id": f"c{i}", "text": text, "source_id": sid,
                           "quote": quote})
        return {"summary": "mock findings", "claims": claims}

    def _verifier(user: str, rng: random.Random) -> dict:
        verdicts = []
        for block in user.split("\n\n"):
            cid = re.search(r"claim_id: (\S+)", block)
            claim = re.search(r"CLAIM: (.*)", block)
            quote = re.search(r"QUOTE: (.*)", block, re.S)
            if not (cid and claim and quote):
                continue
            actual = truth.get(_key(claim.group(1), quote.group(1).strip()), True)
            called = (not actual) if rng.random() < judge_error else actual
            verdicts.append({"claim_id": cid.group(1), "supported": called,
                             "reason": "mock verdict"})
        return {"verdicts": verdicts}

    def _synth(user: str, rng: random.Random) -> dict:
        cites = re.findall(r"- \[([^\]]+)\]", user)
        lines = [f"Verified point sourced from [{c}]." for c in cites[:20]]
        return {
            "markdown": "## Findings\n\n" + "\n\n".join(lines or ["No verified claims."]),
            "open_questions": ["mock open question"],
        }

    mock.register("planner", _planner)
    mock.register("researcher", _researcher)
    mock.register("verifier", _verifier)
    mock.register("synthesizer", _synth)
    return truth
