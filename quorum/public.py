"""Stateless public prototype: isolated requests, no provider credentials or jobs."""
import asyncio
import json
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles

from . import api
from .agents import register_mocks
from .core import Governor, Store
from .llm import MockProvider
from .orchestrator import Quorum

app = FastAPI(title="Quorum interactive prototype", docs_url=None, redoc_url=None)
app.mount('/static', StaticFiles(directory=Path(__file__).parent / 'static'), name='static')
for path, endpoint, methods in [
    ('/', api.dashboard, ['GET']),
    ('/demo/architecture', api.architecture, ['GET']),
    ('/demo/measurements', api.measurements, ['GET']),
    ('/demo/evaluate', api.evaluate_demo, ['POST']),
    ('/demo/recover', api.recover_demo, ['POST']),
    ('/demo/retrieval', api.retrieval_demo, ['POST']),
]:
    app.add_api_route(path, endpoint, methods=methods)


@app.middleware('http')
async def bounded_requests(request: Request, call_next):
    from fastapi.responses import JSONResponse
    if request.method == 'POST':
        body = bytearray()
        async for part in request.stream():
            body.extend(part)
            if len(body) > 2_000_000:
                return JSONResponse({'detail': 'Upload at most 1 MB of text.'}, status_code=413)
        request._body = bytes(body)
    response = await call_next(request)
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response


@app.get('/demo/provider')
async def provider():
    return {'live_available': False, 'label': 'Live mode is local only', 'public_demo': True}


@app.get('/demo/samples')
async def samples():
    return {p.name: p.read_text() for p in sorted((api.ROOT / 'corpus').glob('*.md'))}


@app.get('/documents')
async def documents():
    return api.library_summary({})


@app.post('/documents')
async def validate(request: Request):
    try:
        data = json.loads(await request.body())
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(400, 'Invalid document upload.')
    return api.library_summary(api.validate_documents(data))


@app.post('/runs')
async def run(request: Request):
    try:
        data = json.loads(await request.body())
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(400, 'Invalid research request.')
    if not isinstance(data, dict) or data.get('mock') is not True:
        raise HTTPException(400, 'The public prototype supports offline demo runs only.')
    question = data.get('question')
    if not isinstance(question, str) or not question.strip() or len(question) > 1000:
        raise HTTPException(400, 'Enter a question between 1 and 1000 characters.')
    corpus = api.library_corpus(api.validate_documents(data.get('uploaded_documents')))
    provider = MockProvider(latency=(0.05, 0.2))
    register_mocks(provider, corpus)
    # No background tasks or shared document/run state across serverless requests.
    with tempfile.TemporaryDirectory(prefix='quorum-public-') as tmp:
        store = Store(Path(tmp) / 'run.db')
        try:
            q = Quorum(corpus, store, provider, Governor(
                org_rpm=100000, org_tpm=10000000,
                model_rpm=100000, model_tpm=10000000))
            result = await asyncio.wait_for(q.run(question), timeout=45)
            return {'result': {'event': 'done', 'status': result.status,
                               'metrics': result.metrics, 'claims': result.claims,
                               'report': result.report},
                    'trace': {'spans': store.trace(result.run_id)}}
        finally:
            store.db.close()
