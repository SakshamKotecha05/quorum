"""HTTP serving layer: start a run, stream its events, inspect its trace.

ponytail: run registry is an in-process dict. Runs are minutes long and this is one
uvicorn worker; a broker would be infrastructure with nothing to broker yet. The
checkpoint store is already the durable half -- a restarted server resumes any run by
id, which is the property that actually matters.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .agents import register_mocks
from .core import Corpus, Governor, Store
from .llm import MockProvider, OpenAICompatProvider
from .orchestrator import Quorum

app = FastAPI(title="Quorum", description=__doc__)

CORPUS = Corpus(os.getenv("QUORUM_CORPUS", "corpus"))
STORE = Store(os.getenv("QUORUM_DB", "quorum.db"))
_QUEUES: dict[str, asyncio.Queue] = {}
_TASKS: dict[str, asyncio.Task] = {}


class RunRequest(BaseModel):
    question: str
    mock: bool = False
    lenses: int = 3
    org_rpm: int = 30
    org_tpm: int = 6000
    model_rpm: int | None = None
    model_tpm: int | None = None


def _quorum(req: RunRequest) -> Quorum:
    gov = Governor(org_rpm=req.org_rpm, org_tpm=req.org_tpm,
                   model_rpm=req.model_rpm, model_tpm=req.model_tpm)
    if req.mock:
        provider = MockProvider(latency=(0.05, 0.2))
        register_mocks(provider, CORPUS)
    else:
        provider = OpenAICompatProvider()
    return Quorum(CORPUS, STORE, provider, gov,
                  lenses=("support", "overreach", "attribution")[: req.lenses])


@app.post("/runs")
async def start_run(req: RunRequest) -> dict[str, Any]:
    q = _quorum(req)
    queue: asyncio.Queue = asyncio.Queue()
    run_id = os.urandom(6).hex()
    _QUEUES[run_id] = queue

    async def go() -> None:
        try:
            res = await q.run(req.question, run_id=run_id,
                              emit=lambda ev, rec: queue.put_nowait(rec))
            queue.put_nowait({"event": "done", "metrics": res.metrics,
                              "report": res.report})
        except Exception as e:  # noqa: BLE001 -- surface it on the stream
            queue.put_nowait({"event": "error", "detail": str(e)})
        finally:
            queue.put_nowait(None)

    _TASKS[run_id] = asyncio.create_task(go())
    return {"run_id": run_id, "events": f"/runs/{run_id}/events"}


@app.get("/runs/{run_id}/events")
async def stream_events(run_id: str) -> StreamingResponse:
    queue = _QUEUES.get(run_id)
    if queue is None:
        raise HTTPException(404, "unknown or already-drained run")

    async def gen():
        while True:
            item = await queue.get()
            if item is None:
                yield "event: end\ndata: {}\n\n"
                return
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    run = STORE.load_run(run_id)
    if not run:
        raise HTTPException(404, "no such run")
    nodes = STORE.load_nodes(run_id)
    run["nodes"] = [
        {"id": n.id, "role": n.role, "status": n.status.value, "error": n.error}
        for n in nodes.values()
    ]
    return run


@app.get("/runs/{run_id}/trace")
async def get_trace(run_id: str) -> dict:
    spans = STORE.trace(run_id)
    if not spans:
        raise HTTPException(404, "no spans")
    return {
        "spans": spans,
        "totals": {
            "calls": len(spans),
            "tokens": sum(s["in_tok"] + s["out_tok"] for s in spans),
            "model_s": round(sum(s["ms"] for s in spans) / 1000, 2),
            "throttled_s": round(sum(s["waited_s"] for s in spans), 2),
        },
    }


@app.get("/runs")
async def list_runs() -> list[dict]:
    return STORE.list_runs()
