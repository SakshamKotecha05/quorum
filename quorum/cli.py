"""Command line entry point."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid

from .agents import register_mocks
from .core import Corpus, Governor, Store, TIERS
from .llm import MockProvider, OpenAICompatProvider
from .orchestrator import Quorum


def build(args) -> tuple[Quorum, Store]:
    corpus = Corpus(args.corpus)
    store = Store(args.db)
    rpm, tpm = (100_000, 10_000_000) if args.unthrottled else (args.rpm, args.tpm)
    gov = Governor(
        max_tokens=args.max_tokens,
        max_requests=args.max_requests,
        org_rpm=rpm,
        org_tpm=tpm,
        model_rpm=rpm,
        model_tpm=tpm,
    )
    if args.mock:
        provider = MockProvider(latency=(args.mock_latency, args.mock_latency * 3),
                                fail_rate=args.mock_fail)
        register_mocks(provider, corpus, p_fabricated=args.fabricated,
                       p_overreach=args.overreach, judge_error=args.judge_error)
    else:
        provider = OpenAICompatProvider()
    q = Quorum(corpus, store, provider, gov,
               lenses=tuple(("support", "overreach", "attribution")[: args.lenses]),
               verify_batch_size=args.batch)
    return q, store


def _print_metrics(m: dict) -> None:
    width = max(len(k) for k in m)
    for k, v in m.items():
        print(f"  {k:<{width}}  {v}")


async def cmd_run(args) -> int:
    q, store = build(args)

    def emit(ev, rec):
        if args.quiet:
            return
        detail = {k: v for k, v in rec.items() if k not in ("t", "event")}
        print(f"  [{ev}] {json.dumps(detail)}", file=sys.stderr)

    run_id = args.resume or uuid.uuid4().hex[:12]
    print(f"run {run_id}  (resume with: quorum run --resume {run_id})",
          file=sys.stderr)
    res = await q.run(args.question, run_id=run_id, resume=bool(args.resume),
                      emit=emit)
    print(f"\nrun {res.run_id}  status={res.status}\n")
    if res.report:
        print(res.report["markdown"])
        if res.report.get("open_questions"):
            print("\n### Open questions")
            for oq in res.report["open_questions"]:
                print(f"- {oq}")
    print("\nmetrics:")
    _print_metrics(res.metrics)
    return 0 if res.status == "done" else 1


async def cmd_models(args) -> int:
    import httpx

    p = OpenAICompatProvider()
    try:
        for m in await p.list_models():
            print(m)
    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        hint = {
            401: "no valid key -- set GROQ_API_KEY (free, no card) or QUORUM_API_KEY",
            403: "key rejected for this endpoint",
            404: f"{p.base_url} does not serve /models",
        }.get(code, "")
        print(f"provider returned {code} for {p.base_url}. {hint}", file=sys.stderr)
        print("run offline instead:  python -m quorum run \"...\" --mock",
              file=sys.stderr)
        return 1
    except httpx.RequestError as e:
        print(f"cannot reach {p.base_url}: {e}", file=sys.stderr)
        return 1
    finally:
        await p.close()
    return 0


def cmd_trace(args) -> int:
    store = Store(args.db)
    rows = store.trace(args.run_id)
    if not rows:
        print("no spans for that run id")
        return 1
    hdr = ("node_id", "role", "tier", "in_tok", "out_tok", "waited_s", "ms")
    w = {h: max(len(h), *(len(str(r[h])) for r in rows)) for h in hdr}
    print("  ".join(h.ljust(w[h]) for h in hdr))
    for r in rows:
        print("  ".join(str(r[h]).ljust(w[h]) for h in hdr))
    print(f"\n{len(rows)} calls, "
          f"{sum(r['in_tok'] + r['out_tok'] for r in rows)} tokens, "
          f"{sum(r['ms'] for r in rows) / 1000:.1f}s of model time, "
          f"{sum(r['waited_s'] for r in rows):.1f}s throttled")
    return 0


def cmd_runs(args) -> int:
    for r in Store(args.db).list_runs():
        print(f"{r['run_id']}  {r['status']:<8}  {r['question'][:60]}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser("quorum", description=__doc__)
    p.add_argument("--db", default="quorum.db")
    p.add_argument("--corpus", default="corpus")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a research question")
    r.add_argument("question")
    r.add_argument("--resume", metavar="RUN_ID", help="resume a previous run")
    r.add_argument("--mock", action="store_true", help="offline deterministic provider")
    r.add_argument("--quiet", action="store_true")
    r.add_argument("--lenses", type=int, default=3, choices=(1, 2, 3))
    r.add_argument("--batch", type=int, default=8, help="claims per verifier call")
    r.add_argument("--max-tokens", type=int, default=200_000)
    r.add_argument("--max-requests", type=int, default=200)
    r.add_argument("--rpm", type=int, default=int(os.getenv("QUORUM_ORG_RPM", "30")))
    r.add_argument("--tpm", type=int, default=int(os.getenv("QUORUM_ORG_TPM", "6000")))
    r.add_argument("--unthrottled", action="store_true",
                   help="ignore rate ceilings; isolates orchestration cost from "
                        "throttle cost when comparing runs")
    r.add_argument("--fabricated", type=float, default=0.20,
                   help="mock: fraction of claims citing a nonexistent quote")
    r.add_argument("--overreach", type=float, default=0.20,
                   help="mock: fraction of claims that overstate a real quote")
    r.add_argument("--judge-error", type=float, default=0.20,
                   help="mock: per-lens verifier error rate")
    r.add_argument("--mock-latency", type=float, default=0.15)
    r.add_argument("--mock-fail", type=float, default=0.0,
                   help="mock: transient failure rate, exercises scheduler retries")
    r.set_defaults(fn=lambda a: asyncio.run(cmd_run(a)))

    m = sub.add_parser("models", help="list models the provider actually serves")
    m.set_defaults(fn=lambda a: asyncio.run(cmd_models(a)))

    t = sub.add_parser("trace", help="per-call trace for a run")
    t.add_argument("run_id")
    t.set_defaults(fn=cmd_trace)

    l = sub.add_parser("runs", help="recent runs")
    l.set_defaults(fn=cmd_runs)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
