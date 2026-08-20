"""Evaluation harness.

Question under test: does the orchestration actually buy anything over one agent
making one call, and does a 3-lens verifier quorum beat a single noisy judge?

Method: fault injection. The mock researcher emits claims with a known defect rate
in two modes -- fabricated citations (quote exists nowhere) and overreach (quote is
real and correctly cited, claim overstates it). Because ground truth is known per
claim, precision and recall of each configuration are exact, with no hand-labelled
corpus needed.

Every configuration sees identical claims: fixtures are seeded from claim content, so
the only variable across rows is the verification strategy.

Run: python evals/run_eval.py
"""
from __future__ import annotations

import asyncio
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quorum.agents import _norm, register_mocks
from quorum.core import Corpus, Governor, Store
from quorum.llm import MockProvider
from quorum.orchestrator import Quorum

ROOT = Path(__file__).resolve().parents[1]

QUESTIONS = [
    "What makes multi-agent orchestration hard to run in production?",
    "How should an agent system handle state across failures?",
    "What limits parallelism when inference is rate limited?",
    "How should an agent system be evaluated?",
    "When is a supervisor topology preferable to a swarm?",
]

JUDGE_ERROR = 0.20
P_FABRICATED = 0.20
P_OVERREACH = 0.20


def _key(text: str, quote: str) -> str:
    return f"{_norm(text)}||{_norm(quote)}"


def _make(lenses: int, tmp: Path):
    corpus = Corpus(ROOT / "corpus")
    store = Store(tmp / f"eval{lenses}.db")
    provider = MockProvider(latency=(0.01, 0.03))
    truth = register_mocks(
        provider, corpus, p_fabricated=P_FABRICATED, p_overreach=P_OVERREACH,
        judge_error=JUDGE_ERROR,
    )
    # Ceilings lifted so this measures verification quality, not throttle behaviour.
    gov = Governor(org_rpm=100_000, org_tpm=10_000_000,
                   model_rpm=100_000, model_tpm=10_000_000)
    q = Quorum(corpus, store, provider, gov,
               lenses=("support", "overreach", "attribution")[:lenses])
    return q, truth


def score(claims: list[dict], truth: dict[str, bool], shipped_key="outcome") -> dict:
    """Confusion matrix over claims that reached the final report."""
    tp = fp = fn = tn = 0
    for c in claims:
        grounded = truth.get(_key(c["text"], c["quote"]), True)
        shipped = c[shipped_key] == "verified" if shipped_key in c else True
        if shipped and grounded:
            tp += 1
        elif shipped and not grounded:
            fp += 1          # a hallucination reached the report
        elif not shipped and grounded:
            fn += 1          # a true claim was wrongly discarded
        else:
            tn += 1          # correctly caught
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    bad = fp + tn
    return {
        "proposed": len(claims),
        "defective": bad,
        "leaked": fp,
        "leak_rate": fp / bad if bad else 0.0,
        "false_reject": fn,
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
    }


async def main() -> None:
    tmp = ROOT / ".eval"
    tmp.mkdir(exist_ok=True)
    rows: list[tuple[str, dict, dict]] = []

    # --- baseline: one agent, one call, no plan, no verification ------------ #
    q, truth = _make(1, tmp)
    b_claims, b_calls, b_wall = [], 0, 0.0
    for question in QUESTIONS:
        r = await q.baseline(question)
        b_claims += r["claims"]
        b_calls += r["llm_calls"]
        b_wall += r["wall_s"]
    rows.append(("single agent, no verification",
                 score(b_claims, truth),
                 {"llm_calls": b_calls, "wall_s": round(b_wall, 2)}))

    # --- orchestrated, 1 lens and 3 lenses ---------------------------------- #
    for lenses in (1, 3):
        q, truth = _make(lenses, tmp)
        claims, calls, wall, par = [], 0, 0.0, []
        for question in QUESTIONS:
            res = await q.run(question)
            claims += res.claims
            calls += res.metrics["llm_calls"]
            wall += res.metrics["wall_s"]
            par.append(res.metrics["effective_parallelism"])
        rows.append((
            f"quorum, {lenses} lens{'es' if lenses > 1 else ''} "
            f"(majority {lenses // 2 + 1}/{lenses})",
            score(claims, truth),
            {"llm_calls": calls, "wall_s": round(wall, 2),
             "parallelism": round(statistics.mean(par), 2)},
        ))

    # --- report -------------------------------------------------------------- #
    print(f"\n{len(QUESTIONS)} questions | injected defects: "
          f"{P_FABRICATED:.0%} fabricated citation, {P_OVERREACH:.0%} overreach | "
          f"per-lens judge error {JUDGE_ERROR:.0%}\n")
    hdr = ["configuration", "claims", "bad", "leaked", "leak%", "wrongly cut",
           "prec", "recall", "F1", "calls"]
    w = [34, 7, 5, 7, 7, 12, 6, 7, 6, 6]
    print("  ".join(h.ljust(x) for h, x in zip(hdr, w)))
    print("  ".join("-" * x for x in w))
    for name, s, extra in rows:
        cells = [
            name, str(s["proposed"]), str(s["defective"]), str(s["leaked"]),
            f"{s['leak_rate']:.0%}", str(s["false_reject"]),
            f"{s['precision']:.2f}", f"{s['recall']:.2f}", f"{s['f1']:.2f}",
            str(extra["llm_calls"]),
        ]
        print("  ".join(c.ljust(x) for c, x in zip(cells, w)))

    base, one, three = (r[1] for r in rows)
    print(f"\nhallucination leak rate: {base['leak_rate']:.0%} unverified "
          f"-> {one['leak_rate']:.0%} with 1 lens "
          f"-> {three['leak_rate']:.0%} with a 3-lens majority")
    print(f"cost of the 3-lens quorum: {rows[2][2]['llm_calls']} calls vs "
          f"{rows[0][2]['llm_calls']} for the baseline "
          f"({rows[2][2]['llm_calls'] / max(1, rows[0][2]['llm_calls']):.1f}x)")
    print(f"true claims wrongly discarded by the quorum: {three['false_reject']} "
          f"of {three['proposed'] - three['defective']}")


if __name__ == "__main__":
    asyncio.run(main())
