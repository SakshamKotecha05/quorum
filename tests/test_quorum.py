"""Checks for the parts that would fail silently: the scheduler, the governor,
retrieval, the free grounding filter, and checkpoint resume.

Plain asyncio.run, no pytest-asyncio -- one fewer dependency to keep current.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from quorum.agents import Claim, prefilter, register_mocks, unescape_newlines
from quorum.core import (
    BudgetExceeded, ContextTooLarge, Corpus, CycleError, Exhausted, Governor,
    Node, RateLimiter, Status, Store, Tier, Usage, validate_dag,
)
from quorum.llm import MockProvider, extract_json
from quorum.orchestrator import Deadlock, Quorum, run_dag

CORPUS = Path(__file__).resolve().parents[1] / "corpus"


# --------------------------------------------------------------------------- #
# DAG scheduler
# --------------------------------------------------------------------------- #


def test_validate_dag_rejects_cycles_and_dangling_refs():
    with pytest.raises(CycleError):
        validate_dag({
            "a": Node("a", "r", {}, ["b"]),
            "b": Node("b", "r", {}, ["a"]),
        })
    with pytest.raises(CycleError):
        validate_dag({"a": Node("a", "r", {}, ["ghost"])})


def test_scheduler_runs_independent_nodes_concurrently():
    """Four 100ms nodes with no dependencies must finish in ~100ms, not ~400ms."""
    nodes = {f"n{i}": Node(f"n{i}", "r", {}) for i in range(4)}

    async def execute(n):
        await asyncio.sleep(0.1)
        return {"id": n.id}

    t0 = time.monotonic()
    asyncio.run(run_dag(nodes, execute, concurrency=4))
    elapsed = time.monotonic() - t0

    assert elapsed < 0.25, f"ran sequentially: {elapsed:.2f}s"
    assert all(n.status is Status.DONE for n in nodes.values())


def test_scheduler_respects_dependencies_and_concurrency_cap():
    order: list[str] = []
    nodes = {
        "a": Node("a", "r", {}),
        "b": Node("b", "r", {}, ["a"]),
        "c": Node("c", "r", {}, ["a"]),
        "d": Node("d", "r", {}, ["b", "c"]),
    }

    async def execute(n):
        await asyncio.sleep(0.02)
        order.append(n.id)
        return {}

    asyncio.run(run_dag(nodes, execute, concurrency=2))
    assert order[0] == "a" and order[-1] == "d"
    assert set(order[1:3]) == {"b", "c"}


def test_failure_propagates_as_skip_not_crash():
    nodes = {
        "a": Node("a", "r", {}),
        "b": Node("b", "r", {}, ["a"]),
        "c": Node("c", "r", {}),
    }

    async def execute(n):
        if n.id == "a":
            raise RuntimeError("boom")
        return {}

    asyncio.run(run_dag(nodes, execute, max_attempts=1))
    assert nodes["a"].status is Status.FAILED
    assert nodes["b"].status is Status.SKIPPED  # downstream, poisoned
    assert nodes["c"].status is Status.DONE     # unrelated branch survives


def test_transient_failure_is_retried():
    calls = {"n": 0}

    async def execute(n):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return {"ok": True}

    nodes = {"a": Node("a", "r", {})}
    asyncio.run(run_dag(nodes, execute, max_attempts=2))
    assert nodes["a"].status is Status.DONE
    assert nodes["a"].attempts == 2


def test_node_timeout_is_enforced():
    async def execute(n):
        await asyncio.sleep(5)

    nodes = {"a": Node("a", "r", {})}
    asyncio.run(run_dag(nodes, execute, timeout_s=0.05, max_attempts=1))
    assert nodes["a"].status is Status.FAILED


def test_already_done_nodes_are_not_re_executed():
    """The whole of resume: a DONE node is skipped when the DAG runs again."""
    ran: list[str] = []

    async def execute(n):
        ran.append(n.id)
        return {}

    nodes = {
        "a": Node("a", "r", {}, status=Status.DONE, output={}),
        "b": Node("b", "r", {}, ["a"]),
    }
    asyncio.run(run_dag(nodes, execute))
    assert ran == ["b"]


# --------------------------------------------------------------------------- #
# Governor and rate limiting
# --------------------------------------------------------------------------- #


def test_token_bucket_blocks_then_admits():
    async def go():
        rl = RateLimiter(rpm=60, tpm=120, name="t")
        await rl.acquire(100)          # drains most of the token bucket
        t0 = time.monotonic()
        await rl.acquire(100)          # must wait for refill (2 tok/s)
        return time.monotonic() - t0

    waited = asyncio.run(go())
    assert waited > 0.5, f"limiter did not throttle (waited {waited:.2f}s)"


def test_call_larger_than_ceiling_fails_fast():
    """No amount of waiting admits it, so the caller must trim context, not retry."""
    async def go():
        rl = RateLimiter(rpm=60, tpm=1000, name="t")
        await rl.acquire(5000)

    with pytest.raises(ContextTooLarge):
        asyncio.run(go())


def test_concurrent_admissions_cannot_overdraw():
    """Ten workers admitting at once must not collectively exceed the bucket."""
    async def go():
        rl = RateLimiter(rpm=1000, tpm=1000, name="t")
        rl.tok.per_second = 0.0  # freeze refill so the cap is absolute
        results = await asyncio.gather(
            *(rl.acquire(200, max_wait_s=0.2) for _ in range(10)),
            return_exceptions=True,
        )
        return sum(1 for r in results if not isinstance(r, Exception))

    admitted = asyncio.run(go())
    assert admitted == 5, f"bucket of 1000 admitted {admitted} x 200-token calls"


def test_run_caps_are_enforced():
    async def go():
        gov = Governor(max_tokens=10_000, max_requests=2, org_rpm=1000, org_tpm=100_000,
                       model_rpm=1000, model_tpm=100_000)
        await gov.admit(Tier.FAST, 100, 100)
        await gov.admit(Tier.FAST, 100, 100)
        await gov.admit(Tier.FAST, 100, 100)  # third exceeds the request cap

    with pytest.raises(BudgetExceeded):
        asyncio.run(go())


def test_daily_exhaustion_fails_over_instead_of_waiting():
    """Regression: a per-day quota 429 was treated as a transient throttle, so the
    governor parked for the 615s retry-after and the node timed out instead of
    routing to a model that still had quota."""
    async def go():
        gov = Governor(org_rpm=1000, org_tpm=100_000, model_rpm=1000, model_tpm=100_000)
        await gov.on_429(Tier.DEEP, retry_after=615, daily=True)
        g = await gov.admit(Tier.DEEP, 100, 200)
        return g.tier, gov.org.headroom(300)

    tier, org_headroom = asyncio.run(go())
    assert tier is not Tier.DEEP, "kept routing to the exhausted model"
    assert org_headroom == 0.0, "one model's daily quota parked the org bucket"


def test_all_models_exhausted_fails_fast():
    async def go():
        gov = Governor(org_rpm=1000, org_tpm=100_000, model_rpm=1000, model_tpm=100_000)
        for t in (Tier.DEEP, Tier.MID, Tier.FAST):
            await gov.on_429(t, retry_after=615, daily=True)
        await gov.admit(Tier.DEEP, 100, 200)

    with pytest.raises(Exhausted):
        asyncio.run(go())


def test_output_estimate_learns_from_observed_usage():
    """Regression: reserving max_out deadlocked the run at 1 call/min."""
    async def go():
        gov = Governor(org_rpm=1000, org_tpm=100_000, model_rpm=1000, model_tpm=100_000)
        g = await gov.admit(Tier.MID, 500, max_out=4096)
        first = g.est_tokens
        for _ in range(10):
            g2 = await gov.admit(Tier.MID, 500, max_out=4096)
            await gov.settle(g2, Usage(input_tokens=500, output_tokens=90))
        return first, (await gov.admit(Tier.MID, 500, max_out=4096)).est_tokens

    first, later = asyncio.run(go())
    assert first < 500 + 4096, "reserved the cap instead of an estimate"
    assert later < first, f"estimate did not adapt down: {first} -> {later}"


# --------------------------------------------------------------------------- #
# Retrieval and the free grounding filter
# --------------------------------------------------------------------------- #


def test_bm25_ranks_the_relevant_document_first():
    c = Corpus(CORPUS)
    assert len(c.chunks) > 4
    top = c.search("token bucket admission control rate ceiling", k=1)[0]
    assert top.source_id.startswith("limits")


def test_prefilter_kills_fabricated_citations_without_a_model():
    c = Corpus(CORPUS)
    real = c.chunks[0]
    quote = " ".join(real.text.split()[3:12])
    good = Claim(id="g", text="t", source_id=real.source_id, quote=quote)
    bad_quote = Claim(id="b1", text="t", source_id=real.source_id,
                      quote="a sentence that appears in no document anywhere")
    bad_source = Claim(id="b2", text="t", source_id="ghost#9", quote=quote)

    kept, dropped = prefilter([good, bad_quote, bad_source], c)
    assert [k.id for k in kept] == ["g"]
    assert {d.id for d in dropped} == {"b1", "b2"}


def test_escaped_newlines_in_report_are_repaired():
    """Live models double-escape newlines inside the JSON string, which renders the
    whole report as one blob."""
    assert unescape_newlines("## H\\n\\n- a\\n- b") == "## H\n\n- a\n- b"
    # a properly formatted reply that merely mentions the escape is untouched
    already = "## H\n\nuse \\n for a line break"
    assert unescape_newlines(already) == already


def test_json_salvage_handles_fenced_and_prefixed_replies():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json('thinking... {"b":2} done') == '{"b":2}'


# --------------------------------------------------------------------------- #
# End to end, including checkpoint resume
# --------------------------------------------------------------------------- #


def _quorum(tmp_path, **kw):
    corpus = Corpus(CORPUS)
    store = Store(tmp_path / "t.db")
    provider = MockProvider(latency=(0.01, 0.03), **kw)
    register_mocks(provider, corpus, p_fabricated=0.5, p_overreach=0.0, judge_error=0.0)
    gov = Governor(org_rpm=100_000, org_tpm=10_000_000,
                   model_rpm=100_000, model_tpm=10_000_000)
    return Quorum(corpus, store, provider, gov), store, provider


def test_end_to_end_produces_only_grounded_claims(tmp_path):
    q, store, _ = _quorum(tmp_path)
    res = asyncio.run(q.run("what constrains parallelism?"))
    assert res.status == "done"
    m = res.metrics
    assert m["claims_proposed"] > 0
    # 50% injected fabrication rate, and a noiseless judge, so nothing ungrounded
    # may survive to the report.
    assert m["rejected_by_prefilter"] > 0
    assert m["claims_verified"] == m["claims_proposed"] - m["rejected_by_prefilter"] \
        - m["rejected_by_quorum"]
    assert m["verifier_calls"] < m["verifier_calls_unbatched"]


def test_resume_requeues_orphaned_running_nodes(tmp_path):
    """Regression: a node still RUNNING when the process died is neither PENDING nor
    DONE, so the scheduler skips it and the run reports success with work missing."""
    q, store, _ = _quorum(tmp_path)
    run_id = "orphaned"
    store.save_run(run_id, "q", "running", {})
    store.save_node(run_id, Node("plan", "planner", {}, status=Status.RUNNING))

    loaded = store.load_nodes(run_id)
    assert loaded["plan"].status is Status.RUNNING

    res = asyncio.run(q.run("what constrains parallelism?", run_id=run_id, resume=True))
    after = store.load_nodes(run_id)
    assert res.status == "done"
    assert not any(n.status is Status.RUNNING for n in after.values())
    assert after["plan"].status is Status.DONE


def test_resume_retries_failed_nodes(tmp_path):
    """A resume is an explicit retry. Leaving FAILED nodes alone made resume a no-op
    on exactly the runs that most needed resuming."""
    q, store, _ = _quorum(tmp_path)
    run_id = "hadfailure"
    store.save_run(run_id, "q", "failed", {})
    store.save_node(run_id, Node("plan", "planner", {}, status=Status.FAILED,
                                 error="boom", attempts=2))

    res = asyncio.run(q.run("what constrains parallelism?", run_id=run_id, resume=True))
    assert res.status == "done"
    assert store.load_nodes(run_id)["plan"].status is Status.DONE


def test_resume_reuses_completed_nodes(tmp_path):
    """Kill a run mid-flight, resume it, and confirm the finished work is not redone."""
    q, store, provider = _quorum(tmp_path)

    async def crash_after_research():
        run_id = "crashme"
        original = q.llm.structured
        state = {"n": 0}

        async def flaky(*a, **kw):
            if kw.get("role") == "verifier":
                state["n"] += 1
                if state["n"] > 1:
                    raise RuntimeError("simulated process death")
            return await original(*a, **kw)

        q.llm.structured = flaky
        try:
            await q.run("what constrains parallelism?", run_id=run_id)
        except Exception:
            pass
        q.llm.structured = original
        return run_id

    run_id = asyncio.run(crash_after_research())
    done_before = sum(
        1 for n in store.load_nodes(run_id).values() if n.status is Status.DONE
    )
    assert done_before > 0

    calls_before = provider.calls
    res = asyncio.run(q.run("what constrains parallelism?", run_id=run_id, resume=True))
    replayed = provider.calls - calls_before

    assert res.status == "done"
    assert replayed < done_before, (
        f"resume replayed {replayed} calls but {done_before} nodes were already done"
    )
