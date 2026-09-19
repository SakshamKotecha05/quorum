"""The deployed app executes real offline workflows without shared visitor data."""
import asyncio
import json
from pathlib import Path

import httpx

from quorum.public import app


def test_public_demo_is_isolated_and_offline(monkeypatch):
    from quorum.llm import OpenAICompatProvider
    def forbidden(*args, **kwargs):
        raise AssertionError('public requests must never use live credentials')
    monkeypatch.setattr(OpenAICompatProvider, '__init__', forbidden)
    monkeypatch.setenv('GROQ_API_KEY', 'not-a-real-secret')

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            assert (await client.get('/demo/provider')).json()['live_available'] is False
            docs = (await client.get('/demo/samples')).json()
            assert len(docs) == 4
            assert (await client.post('/documents', json=docs)).json()['passages'] == 8
            assert (await client.get('/documents')).json()['documents'] == []
            assert (await client.post('/runs', json={'mock': False})).status_code == 400
            result = await client.post('/runs', json={'mock': True,
                'question': 'What makes multi-agent orchestration hard to run in production?',
                'uploaded_documents': docs})
            assert result.status_code == 200, result.text
            assert result.json()['result']['status'] == 'done'
            assert result.json()['result']['metrics']['claims_verified'] == 7
            assert len(result.json()['trace']['spans']) == 8
            assert (await client.get('/runs')).status_code == 405
            assert (await client.get('/documents')).json()['documents'] == []
            assert (await client.post('/documents', content=json.dumps({'bad.md': '\ud800'}))).status_code == 400
            assert (await client.post('/runs', content=b'x' * 2000001)).status_code == 413
            for path in ['/demo/evaluate', '/demo/retrieval', '/demo/recover']:
                assert (await client.post(path)).status_code == 200
            for path in ['/.env', '/.eval/documents.json', '/quorum.db']:
                assert (await client.get(path)).status_code == 404
    asyncio.run(go())
