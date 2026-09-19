"""HTTP serving layer: start a run, stream its events, inspect its trace.

ponytail: run registry is an in-process dict. Runs are minutes long and this is one
uvicorn worker; a broker would be infrastructure with nothing to broker yet. The
checkpoint store is the durable half. Resume is currently exposed through the CLI,
not through an HTTP route.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agents import register_mocks
from .core import Corpus, Governor, Store, _chunk
from .llm import MockProvider, OpenAICompatProvider
from .orchestrator import Quorum

app = FastAPI(title="Quorum", description=__doc__)

CORPUS = Corpus(os.getenv("QUORUM_CORPUS", "corpus"))
STORE = Store(os.getenv("QUORUM_DB", ":memory:" if os.getenv("VERCEL") else "quorum.db"))
_QUEUES: dict[str, asyncio.Queue] = {}
_TASKS: dict[str, asyncio.Task] = {}


class RunRequest(BaseModel):
    question: str
    mock: bool = False
    documents: Literal["bundled", "uploaded"] = "bundled"
    lenses: int = Field(default=3, ge=1, le=3, strict=True)
    org_rpm: int = 30
    org_tpm: int = 6000
    model_rpm: int | None = None
    model_tpm: int | None = None


def _quorum(req: RunRequest) -> Quorum:
    corpus = CORPUS
    if req.documents == "uploaded":
        corpus = library_corpus(read_library())
        if not corpus.chunks:
            raise HTTPException(400, "Add documents before starting research.")
    gov = Governor(org_rpm=req.org_rpm, org_tpm=req.org_tpm,
                   model_rpm=req.model_rpm, model_tpm=req.model_tpm)
    if req.mock:
        provider = MockProvider(latency=(0.05, 0.2))
        register_mocks(provider, corpus)
    else:
        provider = OpenAICompatProvider()
    return Quorum(corpus, STORE, provider, gov,
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
                              "report": res.report, "claims": res.claims, "status": res.status})
        except Exception as e:  # noqa: BLE001 -- surface it on the stream
            queue.put_nowait({"event": "error", "detail": str(e)})
        finally:
            queue.put_nowait(None)
            if isinstance(q.llm.provider, OpenAICompatProvider):
                await q.llm.provider.close()

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


# Local demonstration UI. Assets are explicitly scoped, never the project root.
ROOT = Path(__file__).resolve().parents[1]
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
_demo_lock = asyncio.Lock()


@app.get("/")
async def dashboard():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/demo/architecture")
async def architecture():
    return FileResponse(ROOT / "docs/loom/architecture.png")


@app.get("/demo/measurements")
async def measurements():
    return json.loads((ROOT / "docs/measurements.json").read_text())


@app.post("/demo/evaluate")
async def evaluate_demo():
    from evals.run_eval import main
    async with _demo_lock:
        with tempfile.TemporaryDirectory(prefix="quorum-evaluation-") as tmp:
            await main(Path(tmp))
            return json.loads((Path(tmp) / "verification.json").read_text())


@app.post("/demo/recover")
async def recover_demo():
    from evals.check_resume import main
    async with _demo_lock:
        return await asyncio.to_thread(main)


@app.post("/demo/retrieval")
async def retrieval_demo():
    from evals.retrieval_eval import evaluate
    golden = json.loads((ROOT / "evals/retrieval_golden.json").read_text())
    return evaluate(Corpus(ROOT / "corpus"), golden)


@app.get("/demo/provider")
async def provider_status():
    configured = bool(os.getenv("QUORUM_API_KEY") or os.getenv("GROQ_API_KEY"))
    groq = os.getenv("QUORUM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/") == "https://api.groq.com/openai/v1"
    return {"live_available": configured, "label": "Groq live" if groq else "Live provider"}


# ponytail: one local library, atomically saved; each run holds its own snapshot.
LIBRARY = ROOT / ".eval" / "documents.json"


def read_library() -> dict[str, str]:
    return json.loads(LIBRARY.read_text()) if LIBRARY.exists() else {}


def library_corpus(documents: dict[str, str]) -> Corpus:
    corpus = Corpus()
    for name, text in sorted(documents.items()):
        for i, chunk in enumerate(_chunk(text, corpus.chunk_chars)):
            corpus.add(f"{Path(name).stem}#{i}", chunk)
    return corpus


def library_summary(documents: dict[str, str]) -> dict:
    return {"documents": [{"name": name, "bytes": len(text.encode()),
                            "passages": len(_chunk(text, 1200)), "text": text}
                           for name, text in sorted(documents.items())],
            "passages": len(library_corpus(documents).chunks)}


@app.get("/documents")
async def list_documents():
    return library_summary(read_library())


@app.post("/documents")
async def upload_documents(request: Request):
    body = bytearray()
    async for part in request.stream():
        body.extend(part)
        if len(body) > 2_000_000:
            raise HTTPException(413, "Upload at most 1 MB of text at a time.")
    try:
        incoming = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(400, "Invalid document upload.")
    documents = read_library() | validate_documents(incoming)
    validate_documents(documents)
    LIBRARY.parent.mkdir(parents=True, exist_ok=True)
    temporary = LIBRARY.with_suffix(".tmp")
    temporary.write_text(json.dumps(documents))
    temporary.replace(LIBRARY)
    return library_summary(documents)


@app.delete("/documents")
async def clear_documents():
    LIBRARY.unlink(missing_ok=True)
    return library_summary({})


def validate_documents(incoming: Any) -> dict[str, str]:
    if not isinstance(incoming, dict) or not 1 <= len(incoming) <= 20:
        raise HTTPException(400, "Choose between 1 and 20 documents.")
    for name, text in incoming.items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _.-]{0,95}\.(?:md|txt)", name, re.I):
            raise HTTPException(400, "Use .md or .txt filenames with letters, numbers, spaces, dots, hyphens or underscores.")
        if not isinstance(text, str) or not text.strip() or any(ord(c) < 32 and c not in "\n\r\t" for c in text):
            raise HTTPException(400, f"{name}: choose a non-empty UTF-8 text file.")
        try:
            size = len(text.encode())
        except UnicodeEncodeError:
            raise HTTPException(400, f"{name}: choose valid UTF-8 text.")
        if size > 200_000:
            raise HTTPException(413, f"{name}: maximum file size is 200 KB.")
    documents = incoming
    if len(documents) > 20 or sum(len(t.encode()) for t in documents.values()) > 1_000_000:
        raise HTTPException(413, "The library can hold 20 documents and 1 MB of text.")
    stems = [Path(n).stem.casefold() for n in documents]
    if len(stems) != len(set(stems)):
        raise HTTPException(400, "Use distinct document names, including before the file extension.")
    return documents
