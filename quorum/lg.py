"""The same four-stage pipeline expressed in LangGraph, for comparison.

Same agents, same prompts, same LLM facade, same governor. The only variable is the
orchestration layer, so a diff between this and `orchestrator.py` isolates what the
framework provides and what it costs.

Read `docs/langgraph-comparison.md` for the conclusions. Run both with:

    python -m quorum run "..." --mock --unthrottled
    python -m quorum.lg  "..."
"""
from __future__ import annotations

import asyncio
import math
import operator
import sys
import time
import uuid
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .agents import (
    AgentContext, Claim, Findings, Plan, Subtask,
    batch, plan as plan_agent, prefilter, research, synthesize, verify_batch,
    register_mocks,
)
from .core import Corpus, Governor, Store, validate_dag, Node
from .llm import LLM, MockProvider, OpenAICompatProvider
from .orchestrator import Quorum


class GraphState(TypedDict, total=False):
    question: str
    levels: list[list[dict]]      # subtasks grouped into dependency levels
    level: int
    findings: Annotated[list[dict], operator.add]
    kept: list[dict]
    dropped: int
    groups: list[list[dict]]
    verdicts: Annotated[list[dict], operator.add]
    verified: list[dict]
    report: dict


LENSES = ("support", "overreach", "attribution")


def dependency_levels(subtasks: list[Subtask]) -> list[list[dict]]:
    """Group a DAG into topological levels.

    `run_dag` does not need this: it schedules any node whose dependencies are done,
    so an independent subtask starts the instant it can. LangGraph's `Send` fan-out
    is one superstep and every branch in a superstep must finish before the next
    begins, so a DAG has to be flattened into levels first and the widest-node-wins
    barrier is reintroduced at every level.
    """
    by_id = {s.id: s for s in subtasks}
    depth: dict[str, int] = {}

    def d(sid: str, seen: frozenset[str] = frozenset()) -> int:
        if sid in depth:
            return depth[sid]
        deps = [x for x in by_id[sid].depends_on if x in by_id and x not in seen]
        depth[sid] = 1 + max((d(x, seen | {sid}) for x in deps), default=-1)
        return depth[sid]

    for s in subtasks:
        d(s.id)
    out: list[list[dict]] = [[] for _ in range(max(depth.values(), default=-1) + 1)]
    for s in subtasks:
        out[depth[s.id]].append(s.model_dump())
    return out


def build_graph(ctx: AgentContext, quorum_lenses=LENSES, batch_size: int = 8):
    async def plan_node(state: GraphState) -> dict:
        p = await plan_agent(ctx, state["question"])
        # The planner can emit a cycle. LangGraph will happily fan out a cyclic
        # dependency set because Send does not inspect it, so validation stays ours.
        validate_dag({s.id: Node(s.id, "r", {}, [d for d in s.depends_on])
                      for s in p.subtasks})
        return {"levels": dependency_levels(p.subtasks), "level": 0}

    async def research_node(state: dict) -> dict:
        st = Subtask.model_validate(state["subtask"])
        f = await research(ctx, f"research:{st.id}", st, state.get("upstream", []))
        return {"findings": [f.model_dump() | {"_id": st.id}]}

    def advance(state: GraphState) -> dict:
        return {"level": state["level"] + 1}

    def dispatch_research(state: GraphState):
        lvl = state["level"] - 1
        levels = state["levels"]
        if lvl >= len(levels):
            return "collect"
        summaries = [f["summary"] for f in state.get("findings", [])]
        return [
            Send("research", {"subtask": st, "upstream": summaries})
            for st in levels[lvl]
        ]

    def collect(state: GraphState) -> dict:
        claims: list[Claim] = []
        for f in state.get("findings", []):
            fin = Findings.model_validate({k: v for k, v in f.items() if k != "_id"})
            for i, c in enumerate(fin.claims):
                c.id = f"{f['_id']}.c{i}"
            claims.extend(fin.claims)
        kept, dropped = prefilter(claims, ctx.corpus)
        return {
            "kept": [c.model_dump() for c in kept],
            "dropped": len(dropped),
            "groups": [[c.model_dump() for c in g] for g in batch(kept, batch_size)],
        }

    async def verify_node(state: dict) -> dict:
        grp = [Claim.model_validate(c) for c in state["group"]]
        v = await verify_batch(ctx, f"verify:{state['lens']}:{state['i']}",
                               state["lens"], grp)
        return {"verdicts": [{"lens": state["lens"], "verdicts": v}]}

    def dispatch_verify(state: GraphState):
        if not state["groups"]:
            return "tally"
        return [
            Send("verify", {"group": g, "lens": lens, "i": i})
            for lens in quorum_lenses
            for i, g in enumerate(state["groups"])
        ]

    def tally(state: GraphState) -> dict:
        votes: dict[str, int] = {}
        for v in state.get("verdicts", []):
            for cid, ok in v["verdicts"].items():
                votes[cid] = votes.get(cid, 0) + int(bool(ok))
        need = math.ceil(len(quorum_lenses) / 2)
        return {"verified": [c for c in state["kept"]
                             if votes.get(c["id"], 0) >= need]}

    async def synth_node(state: GraphState) -> dict:
        claims = [Claim.model_validate(c) for c in state["verified"]]
        rejected = state["dropped"] + len(state["kept"]) - len(claims)
        r = await synthesize(ctx, state["question"], claims, rejected)
        return {"report": r.model_dump()}

    g = StateGraph(GraphState)
    g.add_node("plan", plan_node)
    g.add_node("advance", advance)
    g.add_node("research", research_node)
    g.add_node("collect", collect)
    g.add_node("dispatch_verify", lambda s: {})
    g.add_node("verify", verify_node)
    g.add_node("tally", tally)
    g.add_node("synth", synth_node)

    g.add_edge(START, "plan")
    g.add_edge("plan", "advance")
    # The loop exists only to walk dependency levels one superstep at a time.
    g.add_conditional_edges("advance", dispatch_research, ["research", "collect"])
    g.add_edge("research", "advance")
    g.add_edge("collect", "dispatch_verify")
    g.add_conditional_edges("dispatch_verify", dispatch_verify, ["verify", "tally"])
    g.add_edge("verify", "tally")
    g.add_edge("tally", "synth")
    g.add_edge("synth", END)
    return g


async def run(question: str, *, mock: bool = True, thread_id: str | None = None,
              db: str = "lg.db") -> dict:
    corpus = Corpus("corpus")
    store = Store("lg-spans.db")
    gov = Governor(org_rpm=100_000, org_tpm=10_000_000,
                   model_rpm=100_000, model_tpm=10_000_000)
    if mock:
        provider = MockProvider(latency=(0.15, 0.6))
        register_mocks(provider, corpus)
    else:
        provider = OpenAICompatProvider()

    thread_id = thread_id or uuid.uuid4().hex[:12]
    ctx = AgentContext(LLM(provider, gov), corpus, store, thread_id)
    graph = build_graph(ctx)

    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        app = graph.compile(checkpointer=saver)
        t0 = time.monotonic()
        final = await app.ainvoke(
            {"question": question, "findings": [], "verdicts": []},
            config={"configurable": {"thread_id": thread_id},
                    "recursion_limit": 100},
        )
        wall = time.monotonic() - t0

    spans = store.trace(thread_id)
    return {
        "thread_id": thread_id,
        "report": final.get("report"),
        "metrics": {
            "wall_s": round(wall, 2),
            "llm_calls": len(spans),
            "sum_call_s": round(sum(s["ms"] for s in spans) / 1000, 2),
            "effective_parallelism": round(
                sum(s["ms"] for s in spans) / 1000 / wall, 2) if wall else 0,
            "tokens_used": gov.tokens_used,
            "claims_kept": len(final.get("kept", [])),
            "claims_verified": len(final.get("verified", [])),
            "rejected_by_prefilter": final.get("dropped", 0),
        },
    }


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "What makes orchestration hard?"
    out = asyncio.run(run(q))
    print(f"thread {out['thread_id']}")
    for k, v in out["metrics"].items():
        print(f"  {k:<24} {v}")
