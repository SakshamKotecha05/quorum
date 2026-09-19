"""Regressions exercised through the real pipeline, graph, and HTTP boundary."""
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import pytest

from quorum.agents import AgentContext, register_mocks
from quorum.core import Corpus, Governor, ModelSpec, Store, TIERS, Tier
from quorum.llm import LLM, MockProvider, RateLimited
from quorum.orchestrator import Quorum
from quorum.lg import build_graph


def setup_run(tmp_path, lenses=("support", "overreach", "attribution")):
    corpus = Corpus(Path(__file__).resolve().parents[1] / "corpus")
    store = Store(tmp_path / "test.db")
    provider = MockProvider(latency=(0, 0))
    register_mocks(provider, corpus)
    gov = Governor(org_rpm=100000, org_tpm=10000000,
                   model_rpm=100000, model_tpm=10000000)
    return Quorum(corpus, store, provider, gov, lenses=lenses)


def test_exhausted_model_is_not_retried_through_an_alias(tmp_path, monkeypatch):
    for tier in TIERS:
        monkeypatch.setitem(TIERS, tier, ModelSpec("shared-model"))
    q = setup_run(tmp_path)
    models = []

    async def exhausted(**kwargs):
        models.append(kwargs["model"])
        raise RateLimited(600, "daily quota", daily=True)

    q.llm.provider.chat = exhausted
    result = asyncio.run(q.run("What limits parallelism?"))
    assert result.status == "failed"
    assert models == ["shared-model"]
    assert q.gov.snapshot()["exhausted_models"] == ["shared-model"]


@pytest.mark.parametrize("engine", ["custom", "langgraph"])
def test_two_lenses_require_both_votes(tmp_path, engine):
    q = setup_run(tmp_path, ("support", "overreach"))
    for lens, supported in [("support", True), ("overreach", False)]:
        q.llm.provider.register(f"verifier:{lens}", lambda user, rng, ok=supported: {
            "verdicts": [{"claim_id": cid, "supported": ok, "reason": "split vote"}
                         for cid in re.findall(r"claim_id: (\S+)", user)]
        })
    if engine == "custom":
        result = asyncio.run(q.run("What limits parallelism?"))
        assert result.metrics["claims_proposed"] > result.metrics["rejected_by_prefilter"]
        assert result.metrics["claims_verified"] == 0
    else:
        ctx = AgentContext(q.llm, q.corpus, q.store, "graph")
        result = asyncio.run(build_graph(ctx, q.lenses).compile().ainvoke({
            "question": "What limits parallelism?", "findings": [], "verdicts": []
        }))
        assert result["kept"]
        assert result["verified"] == []


@pytest.mark.parametrize("interleaved", [False, True])
def test_orchestrators_send_identical_prompts(tmp_path, interleaved):
    async def go():
        captured = []
        for engine in ("custom", "langgraph"):
            q = setup_run(tmp_path)
            if interleaved:
                q.llm.provider.register("planner", lambda user, rng: {"subtasks": [
                    {"id": "b", "question": "state failures", "depends_on": ["a"]},
                    {"id": "a", "question": "token ceilings", "depends_on": []},
                    {"id": "d", "question": "verification", "depends_on": ["c"]},
                    {"id": "c", "question": "supervisor topology", "depends_on": []},
                ]})
            prompts = []
            chat = q.llm.provider.chat

            async def record(**kwargs):
                prompts.append((kwargs["role"], kwargs["system"], kwargs["user"]))
                return await chat(**kwargs)

            q.llm.provider.chat = record
            question = "What makes multi-agent orchestration hard to run in production?"
            if engine == "custom":
                await q.run(question)
            else:
                ctx = AgentContext(q.llm, q.corpus, q.store, "graph")
                await build_graph(ctx).compile().ainvoke({
                    "question": question, "findings": [], "verdicts": []
                })
            captured.append(sorted(prompts))
        assert captured[0] == captured[1]
    asyncio.run(go())


def test_api_rejects_invalid_lens_counts(tmp_path, monkeypatch):
    monkeypatch.setenv("QUORUM_DB", str(tmp_path / "api.db"))
    from quorum.api import app

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                    base_url="http://test") as client:
            for lenses in (0, -1, 4):
                response = await client.post("/runs", json={
                    "question": "test", "mock": True, "lenses": lenses
                })
                assert response.status_code == 422
    asyncio.run(go())


def test_eval_compares_the_same_claim_workload(tmp_path, capsys):
    from evals import run_eval
    asyncio.run(run_eval.main(tmp_path))
    output = capsys.readouterr().out
    rows = [line for line in output.splitlines()
            if line.startswith(("no verification", "prefilter +"))]
    assert len(rows) == 3
    assert all(re.search(r"\s84\s+33\s", row) for row in rows)


def test_score_requires_known_ground_truth():
    from evals.run_eval import score
    with pytest.raises(KeyError):
        score([{"text": "unknown", "quote": "unknown"}], {})


def test_aliases_share_consumption_and_strictest_limits(monkeypatch):
    monkeypatch.setitem(TIERS, Tier.MID, ModelSpec("shared", rpm=20, tpm=9000))
    monkeypatch.setitem(TIERS, Tier.FAST, ModelSpec("shared", rpm=30, tpm=6000))
    gov = Governor(org_rpm=1000, org_tpm=100000)

    async def go():
        await gov.admit(Tier.MID, 100, 100)
        await gov.admit(Tier.FAST, 100, 100)
    asyncio.run(go())
    shared = gov.per_model["shared"]
    assert shared.req.capacity == 20
    assert shared.tok.capacity == 6000
    assert shared.req.tokens < 19
    assert shared.tok.tokens < 5700
    shared.waits = 1
    shared.wait_seconds = 2
    assert gov.snapshot()["throttle_waits"] == 1
    assert gov.snapshot()["throttle_wait_s"] == 2


def test_retrieval_recall_counts_all_relevant_chunks():
    from evals.retrieval_eval import evaluate
    corpus = Corpus()
    corpus.add("a", "checkpoint recovery")
    corpus.add("b", "unrelated text")
    result = evaluate(corpus, [{"query": "checkpoint", "relevant": ["a", "b"]}])
    assert result["recall@1"] == 0.5
    assert result["recall@5"] == 0.5
    with pytest.raises(ValueError):
        evaluate(corpus, [{"query": "checkpoint", "relevant": ["missing"]}])
