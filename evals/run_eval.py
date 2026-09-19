"""Controlled verification ablation on a single frozen claim workload.

Generate each question once with the full pipeline, retain claims and per-lens
votes, then score no filtering, prefilter + support, and prefilter + majority.
The support-only row reuses the exact support votes from the three-lens row.
Counts are verification calls only; generation and synthesis are excluded.
This measures synthetic fault rejection, not live-model or final-prose accuracy.

Run: python evals/run_eval.py
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import platform
from importlib.metadata import version
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
    """Confusion matrix over claims accepted for synthesis, before prose generation."""
    tp = fp = fn = tn = 0
    for c in claims:
        grounded = truth[_key(c["text"], c["quote"])]
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


async def main(output_dir: Path | None = None) -> None:
    tmp = output_dir or ROOT / ".eval"
    tmp.mkdir(exist_ok=True)
    rows: list[tuple[str, dict, dict]] = []

    q, truth = _make(3, tmp)
    workloads = []
    verifier_calls = 0
    for question in QUESTIONS:
        result = await q.run(question)
        if result.status != "done" or any(
            c["outcome"] != "rejected_by_prefilter" and len(c["votes"]) != 3
            for c in result.claims
        ):
            raise RuntimeError("incomplete evaluation run")
        workloads.append({"question": question, "claims": result.claims})
        verifier_calls += result.metrics["verifier_calls"]

    frozen = [c for workload in workloads for c in workload["claims"]]
    unverified = [{**c, "outcome": "verified"} for c in frozen]
    support = [{**c, "outcome": "verified" if c["votes"] and c["votes"][0]
                else "rejected"} for c in frozen]
    for name, claims, calls in (
        ("no verification", unverified, 0),
        ("prefilter + 1 lens", support, verifier_calls // 3),
        ("prefilter + 3 lenses (2/3)", frozen, verifier_calls),
    ):
        rows.append((name, score(claims, truth), {"llm_calls": calls}))

    identity = [{k: c[k] for k in ("id", "text", "quote", "source_id")}
                for c in frozen]
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    (tmp / "verification.json").write_text(json.dumps({
        "python": platform.python_version(), "pydantic": version("pydantic"),
        "workload_sha256": digest, "workloads": workloads, "truth": truth,
        "rows": [{"configuration": name, **scores, "verifier_calls": extra["llm_calls"]}
                 for name, scores, extra in rows],
    }, indent=2) + "\n")
    q.store.db.close()

    # --- report -------------------------------------------------------------- #
    print(f"\n{len(QUESTIONS)} questions | injected defects: "
          f"{P_FABRICATED:.0%} fabricated citation, {P_OVERREACH:.0%} overreach | "
          f"per-lens judge error {JUDGE_ERROR:.0%}\n")
    hdr = ["configuration", "claims", "bad", "leaked", "leak%", "wrongly cut",
           "prec", "recall", "F1", "verify calls"]
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
    print("Call counts cover verification only; all rows share the same generated claims.")
    print(f"Frozen workload: {digest}; evidence: .eval/verification.json")
    print(f"true claims wrongly discarded by the quorum: {three['false_reject']} "
          f"of {three['proposed'] - three['defective']}")


if __name__ == "__main__":
    asyncio.run(main())
