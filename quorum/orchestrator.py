"""The scheduler and the run loop.

`run_dag` is a generic async DAG executor: ready-set scheduling, bounded concurrency,
per-node timeout, bounded retries, and skip-propagation from failed dependencies. It
does not know what an agent is.

`Quorum` composes four stages on top of it -- plan, research, verify, synthesize --
checkpointing every node transition so a killed run resumes from where it stopped.
"""
from __future__ import annotations

import asyncio
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from .agents import (
    AgentContext, Claim, Findings, LENSES, Plan, Subtask,
    batch, plan as plan_agent, prefilter, research, synthesize, verify_batch,
)
from .core import (
    BudgetExceeded, Corpus, Governor, Node, Status, Store, Tier, validate_dag,
)
from .llm import LLM, MockProvider, Provider

Executor = Callable[[Node], Awaitable[dict]]


# --------------------------------------------------------------------------- #
# Generic DAG scheduler
# --------------------------------------------------------------------------- #


class Deadlock(RuntimeError):
    pass


async def run_dag(
    nodes: dict[str, Node],
    execute: Executor,
    *,
    concurrency: int = 4,
    timeout_s: float = 120.0,
    max_attempts: int = 2,
    on_change: Callable[[Node], None] | None = None,
) -> dict[str, Node]:
    """Execute a node DAG. Nodes already marked DONE are left alone, which is the
    whole of the resume mechanism -- resuming is just running the same DAG again
    against a store that already has answers in it."""
    validate_dag(nodes)

    def notify(n: Node) -> None:
        if on_change:
            on_change(n)

    running: dict[asyncio.Task, str] = {}

    while True:
        # A failed or skipped dependency poisons everything downstream. Propagate to
        # a fixed point before scheduling, so we never start work whose input is gone.
        changed = True
        while changed:
            changed = False
            for n in nodes.values():
                if n.status is not Status.PENDING:
                    continue
                if any(nodes[d].status in (Status.FAILED, Status.SKIPPED)
                       for d in n.depends_on):
                    n.status = Status.SKIPPED
                    n.error = "upstream dependency failed"
                    notify(n)
                    changed = True

        ready = [
            n for n in nodes.values()
            if n.status is Status.PENDING
            and all(nodes[d].status is Status.DONE for d in n.depends_on)
        ]
        while ready and len(running) < concurrency:
            n = ready.pop(0)
            n.status = Status.RUNNING
            n.attempts += 1
            notify(n)
            running[asyncio.ensure_future(
                asyncio.wait_for(execute(n), timeout_s)
            )] = n.id

        if not running:
            if any(n.status is Status.PENDING for n in nodes.values()):
                raise Deadlock("pending nodes with no runnable path")
            return nodes

        done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            node = nodes[running.pop(task)]
            try:
                node.output = task.result()
                node.status = Status.DONE
                node.error = None
            except BudgetExceeded as e:
                # Not transient. Retrying burns the last of the quota for nothing.
                node.status = Status.FAILED
                node.error = f"budget: {e}"
            except Exception as e:  # noqa: BLE001 -- scheduler must survive any agent
                if node.attempts < max_attempts:
                    node.status = Status.PENDING
                    node.error = f"attempt {node.attempts} failed: {e}"[:500]
                else:
                    node.status = Status.FAILED
                    node.error = str(e)[:500]
            notify(node)


# --------------------------------------------------------------------------- #
# The orchestrator
# --------------------------------------------------------------------------- #


@dataclass
class RunResult:
    run_id: str
    question: str
    status: str
    report: dict | None
    metrics: dict
    events: list[dict] = field(default_factory=list)
    claims: list[dict] = field(default_factory=list)


class Quorum:
    def __init__(
        self,
        corpus: Corpus,
        store: Store,
        provider: Provider,
        governor: Governor,
        *,
        lenses: tuple[str, ...] = ("support", "overreach", "attribution"),
        verify_batch_size: int = 8,
        max_subtasks: int = 5,
        node_timeout_s: float = 180.0,
    ):
        self.corpus = corpus
        self.store = store
        self.gov = governor
        self.llm = LLM(provider, governor)
        self.lenses = lenses
        self.verify_batch_size = verify_batch_size
        self.max_subtasks = max_subtasks
        self.node_timeout_s = node_timeout_s

    # -- helpers ----------------------------------------------------------- #

    def _ctx(self, run_id: str, emit) -> AgentContext:
        return AgentContext(self.llm, self.corpus, self.store, run_id, emit)

    def _concurrency(self) -> int:
        """Size the semaphore from the observed average call, not a guess.

        A fixed concurrency of 8 against a 6k tok/min ceiling with ~2k-token calls
        means seven workers sit in backoff and the run finishes slower than
        sequential. The governor knows the ceiling; ask it.
        """
        spans = [s for s in self.store.trace(self._run_id) if s["ok"]]
        avg = int(sum(s["in_tok"] + s["out_tok"] for s in spans) / len(spans)) if spans else 0
        return self.gov.concurrency(avg or 1500)

    async def _stage(self, nodes, execute, *, concurrency, on_change):
        return await run_dag(
            nodes, execute, concurrency=concurrency,
            timeout_s=self.node_timeout_s, on_change=on_change,
        )

    # -- the run ----------------------------------------------------------- #

    async def run(
        self, question: str, *, run_id: str | None = None, resume: bool = False,
        emit: Callable[[str, dict], None] | None = None,
    ) -> RunResult:
        events: list[dict] = []

        def _emit(ev: str, data: dict) -> None:
            rec = {"t": round(time.time(), 3), "event": ev, **data}
            events.append(rec)
            if emit:
                emit(ev, rec)

        run_id = run_id or uuid.uuid4().hex[:12]
        self._run_id = run_id
        nodes: dict[str, Node] = self.store.load_nodes(run_id) if resume else {}
        if resume and nodes:
            saved = self.store.load_run(run_id)
            question = (saved or {}).get("question", question)
            # Anything still marked RUNNING was in flight when the process died --
            # nothing is running it now. Left alone it is neither PENDING (so the
            # scheduler never picks it up) nor DONE (so nothing notices), and the run
            # reports success with that node's work silently missing. Requeue it.
            # RUNNING: in flight when the process died, nothing is running it now.
            # FAILED/SKIPPED: a resume is an explicit "try again", so retry them and
            # let the poisoned-dependency rule re-derive which skips still apply.
            orphans = [n for n in nodes.values()
                       if n.status in (Status.RUNNING, Status.FAILED, Status.SKIPPED)]
            for n in orphans:
                n.status = Status.PENDING
                n.attempts = 0
                n.error = f"requeued on resume (was {n.error or 'in flight'})"
                self.store.save_node(run_id, n)
            _emit("resume", {
                "run_id": run_id,
                "done": sum(1 for n in nodes.values() if n.status is Status.DONE),
                "requeued": len(orphans),
                "total": len(nodes),
            })
        self.store.save_run(run_id, question, "running", self.gov.snapshot())
        ctx = self._ctx(run_id, _emit)
        t0 = time.monotonic()

        def checkpoint(n: Node) -> None:
            """Every state transition hits disk before the next one starts. This is
            the only reason `--resume` can be trusted after a kill -9."""
            self.store.save_node(run_id, n)
            _emit("node", {"id": n.id, "role": n.role, "status": n.status.value,
                           "error": n.error})

        # -- stage 1: plan ------------------------------------------------- #
        if "plan" not in nodes:
            nodes["plan"] = Node(id="plan", role="planner", payload={"question": question})

        async def _do_plan(n: Node) -> dict:
            p = await plan_agent(ctx, question, self.max_subtasks)
            return p.model_dump()

        await self._stage({"plan": nodes["plan"]}, _do_plan, concurrency=1,
                          on_change=checkpoint)
        if nodes["plan"].status is not Status.DONE:
            return self._finish(run_id, question, "failed", None, t0, events, nodes)

        the_plan = Plan.model_validate(nodes["plan"].output)

        # -- stage 2: research (the planner's DAG, run as a DAG) ------------ #
        research_nodes: dict[str, Node] = {}
        for st in the_plan.subtasks:
            nid = f"research:{st.id}"
            research_nodes[nid] = nodes.get(nid) or Node(
                id=nid, role="researcher", payload=st.model_dump(),
                depends_on=[f"research:{d}" for d in st.depends_on],
            )
        nodes.update(research_nodes)

        async def _do_research(n: Node) -> dict:
            st = Subtask.model_validate(n.payload)
            upstream = [
                Findings.model_validate(nodes[d].output).summary
                for d in n.depends_on
                if nodes[d].output
            ]
            f = await research(ctx, n.id, st, upstream)
            return f.model_dump()

        conc = self._concurrency()
        _emit("concurrency", {"stage": "research", "workers": conc,
                              "nodes": len(research_nodes)})
        await self._stage(research_nodes, _do_research, concurrency=conc,
                          on_change=checkpoint)

        claims: list[Claim] = []
        for n in research_nodes.values():
            if n.status is Status.DONE and n.output:
                claims.extend(Findings.model_validate(n.output).claims)
        _emit("claims_proposed", {"n": len(claims)})

        # -- stage 3: verification ------------------------------------------ #
        kept, dropped = prefilter(claims, self.corpus)
        _emit("prefilter", {"kept": len(kept), "rejected_free": len(dropped)})

        by_id = {c.id: c for c in kept}
        groups = batch(kept, self.verify_batch_size)
        verify_nodes: dict[str, Node] = {}
        for lens in self.lenses:
            for i, grp in enumerate(groups):
                nid = f"verify:{lens}:{i}"
                verify_nodes[nid] = nodes.get(nid) or Node(
                    id=nid, role=f"verifier:{lens}",
                    payload={"lens": lens, "claim_ids": [c.id for c in grp]},
                )
        nodes.update(verify_nodes)

        async def _do_verify(n: Node) -> dict:
            grp = [by_id[cid] for cid in n.payload["claim_ids"] if cid in by_id]
            verdicts = await verify_batch(ctx, n.id, n.payload["lens"], grp)
            return {"lens": n.payload["lens"], "verdicts": verdicts}

        if verify_nodes:
            conc = self._concurrency()
            _emit("concurrency", {"stage": "verify", "workers": conc,
                                  "nodes": len(verify_nodes), "batches": len(groups),
                                  "lenses": len(self.lenses)})
            await self._stage(verify_nodes, _do_verify, concurrency=conc,
                              on_change=checkpoint)

        votes: dict[str, list[bool]] = {c.id: [] for c in kept}
        for n in verify_nodes.values():
            if n.status is Status.DONE and n.output:
                for cid, ok in n.output["verdicts"].items():
                    if cid in votes:
                        votes[cid].append(bool(ok))

        need = math.ceil(len(self.lenses) / 2)
        verified = [c for c in kept if sum(votes[c.id]) >= need]
        killed = len(kept) - len(verified)
        _emit("quorum", {"verified": len(verified), "rejected_by_quorum": killed,
                         "threshold": f"{need}/{len(self.lenses)}"})

        # -- stage 4: synthesis --------------------------------------------- #
        nodes["synth"] = nodes.get("synth") or Node(
            id="synth", role="synthesizer", payload={"n_claims": len(verified)}
        )

        async def _do_synth(n: Node) -> dict:
            r = await synthesize(ctx, question, verified, len(dropped) + killed)
            return r.model_dump()

        await self._stage({"synth": nodes["synth"]}, _do_synth, concurrency=1,
                          on_change=checkpoint)

        surviving = {c.id for c in verified}
        killed_ids = {c.id for c in kept} - surviving
        outcomes = []
        for c in claims:
            if c.id in surviving:
                stage = "verified"
            elif c.id in killed_ids:
                stage = "rejected_by_quorum"
            else:
                stage = "rejected_by_prefilter"
            outcomes.append({**c.model_dump(), "outcome": stage,
                             "votes": votes.get(c.id, [])})

        report = nodes["synth"].output
        status = "done" if nodes["synth"].status is Status.DONE else "failed"
        return self._finish(run_id, question, status, report, t0, events, nodes,
                            claims=outcomes,
                            extra={
                                "claims_proposed": len(claims),
                                "rejected_by_prefilter": len(dropped),
                                "rejected_by_quorum": killed,
                                "claims_verified": len(verified),
                                "verifier_calls": len(verify_nodes),
                                "verifier_calls_unbatched":
                                    len(kept) * len(self.lenses),
                            })

    def _finish(self, run_id, question, status, report, t0, events, nodes,
                claims=None, extra=None):
        wall = time.monotonic() - t0
        spans = self.store.trace(run_id)
        call_ms = sum(s["ms"] for s in spans)
        metrics = {
            "wall_s": round(wall, 2),
            "llm_calls": len(spans),
            "sum_call_s": round(call_ms / 1000, 2),
            # sum of per-call latency over wall clock: how much parallelism the
            # scheduler actually extracted, after rate-limit queueing is paid for.
            "effective_parallelism": round((call_ms / 1000) / wall, 2) if wall else 0,
            "throttle_share": round(
                sum(s["waited_s"] for s in spans) / wall, 2) if wall else 0,
            "nodes": {s.value: sum(1 for n in nodes.values() if n.status is s)
                      for s in Status},
            **self.gov.snapshot(),
            **self.llm.stats.as_dict(),
            **(extra or {}),
        }
        self.store.save_run(run_id, question, status, metrics)
        return RunResult(run_id, question, status, report, metrics, events,
                         claims or [])

    # -- baseline for the eval harness ------------------------------------- #

    async def baseline(self, question: str) -> dict:
        """Single agent, one shot, no plan, no verification. The thing the whole
        orchestrator has to beat to justify its own existence."""
        run_id = f"base-{uuid.uuid4().hex[:8]}"
        self._run_id = run_id
        self.store.save_run(run_id, question, "running", {})
        ctx = self._ctx(run_id, lambda e, d: None)
        t0 = time.monotonic()
        f = await research(
            ctx, "baseline", Subtask(id="b", question=question, depends_on=[]),
            upstream=[], k=8,
        )
        wall = time.monotonic() - t0
        kept, dropped = prefilter(f.claims, self.corpus)
        self.store.save_run(run_id, question, "done", {})
        return {
            "run_id": run_id,
            "claims": [c.model_dump() for c in f.claims],
            "grounded": len(kept),
            "ungrounded": len(dropped),
            "wall_s": round(wall, 2),
            "llm_calls": len(self.store.trace(run_id)),
        }
