"""Dashboard actions use the real offline pipeline and evaluation checks."""
import asyncio
import json
from pathlib import Path

import httpx


def test_dashboard_runs_and_checks(tmp_path, monkeypatch):
    from quorum import api
    from quorum.core import Store
    monkeypatch.setattr(api, 'STORE', Store(tmp_path / 'dashboard.db'))
    monkeypatch.setattr(api, 'LIBRARY', tmp_path / 'documents.json')

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app),
                                    base_url='http://test') as client:
            page = await client.get('/')
            assert page.status_code == 200
            assert 'Quorum' in page.text
            image = await client.get('/demo/architecture')
            assert image.headers['content-type'] == 'image/png'
            assert (await client.get('/documents')).json()['documents'] == []
            assert (await client.post('/runs', json={'question': 'test', 'mock': True, 'documents': 'uploaded'})).status_code == 400
            documents = {p.name: p.read_text() for p in Path('corpus').glob('*.md')}
            uploaded = await client.post('/documents', json=documents)
            assert uploaded.status_code == 200
            assert uploaded.json()['passages'] == 8
            assert len((await client.get('/documents')).json()['documents']) == 4
            response = await client.post('/runs', json={
                'question': 'What makes multi-agent orchestration hard to run in production?',
                'mock': True, 'documents': 'uploaded', 'org_rpm': 100000, 'org_tpm': 10000000,
                'model_rpm': 100000, 'model_tpm': 10000000,
            })
            stream = await client.get(response.json()['events'])
            events = [json.loads(line[6:]) for line in stream.text.splitlines()
                      if line.startswith('data: ') and line != 'data: {}']
            final = next(e for e in events if e['event'] == 'done')
            assert final['status'] == 'done'
            assert len(final['claims']) == 12
            assert final['metrics']['claims_verified'] == 7
            evaluation = await client.post('/demo/evaluate')
            assert evaluation.status_code == 200
            assert [r['proposed'] for r in evaluation.json()['rows']] == [84, 84, 84]
            assert [r['false_reject'] for r in evaluation.json()['rows']] == [0, 13, 2]
            recovery = await client.post('/demo/recover')
            assert recovery.status_code == 200
            assert recovery.json()['replayed'] == 0
            assert recovery.json()['reused'] > 0
            retrieval = await client.post('/demo/retrieval')
            assert retrieval.status_code == 200
            assert retrieval.json()['queries'] == 14
            assert retrieval.json()['recall@5'] < 1
            assert (await client.get('/demo/measurements')).status_code == 200
    asyncio.run(go())


def test_provider_status_never_exposes_key(monkeypatch):
    from quorum import api
    monkeypatch.setenv('GROQ_API_KEY', 'test-secret-not-real')
    monkeypatch.delenv('QUORUM_API_KEY', raising=False)

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app),
                                    base_url='http://test') as client:
            response = await client.get('/demo/provider')
            assert response.status_code == 200
            assert response.json()['live_available'] is True
            assert 'test-secret-not-real' not in response.text
            monkeypatch.delenv('GROQ_API_KEY')
            assert (await client.get('/demo/provider')).json()['live_available'] is False
    asyncio.run(go())


def test_document_validation_and_snapshot(tmp_path, monkeypatch):
    from quorum import api
    monkeypatch.setattr(api, 'LIBRARY', tmp_path / 'documents.json')

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url='http://test') as client:
            for payload in [{'../bad.md': 'text'}, {'bad.pdf': 'text'}, {'empty.md': ' '}, {'binary.md': 'a\x00b'}, {'text.md': 1}, []]:
                assert (await client.post('/documents', json=payload)).status_code == 400
            assert (await client.post('/documents', json={'big.md': 'a' * 200001})).status_code == 413
            assert (await client.post('/documents', content=b'x' * 2000001)).status_code == 413
            assert (await client.post('/documents', json={'only.md': 'Unique violet evidence.'})).status_code == 200
            q = api._quorum(api.RunRequest(question='violet', mock=True, documents='uploaded'))
            assert q.corpus.get('only#0').text == 'Unique violet evidence.'
            assert len(q.corpus.chunks) == 1
            assert (await client.post('/documents', json={'only.txt': 'collision'})).status_code == 400
            assert (await client.post('/documents', json={'only.md': 'Replaced evidence.'})).status_code == 200
            assert api.read_library() == {'only.md': 'Replaced evidence.'}
            assert (await client.delete('/documents')).json()['documents'] == []
            assert q.corpus.get('only#0').text == 'Unique violet evidence.'
            assert not api.LIBRARY.exists()
    asyncio.run(go())
