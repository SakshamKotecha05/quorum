"""Reproducible mock comparison; add --throttled for the slower 30 RPM / 6K TPM run."""
import argparse
import asyncio
import hashlib
import json
import platform
import sys
import tempfile
import time
from importlib.metadata import distributions
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quorum.agents import AgentContext, register_mocks
from quorum.core import Corpus, Governor, Store
from quorum.lg import build_graph
from quorum.llm import LLM, MockProvider
from quorum.orchestrator import Quorum

ROOT = Path(__file__).resolve().parents[1]
QUESTION = "What makes multi-agent orchestration hard to run in production?"


async def main(throttled=False):
    rows = []
    modes = ["custom", "langgraph"] + (["custom-throttled"] if throttled else [])
    with tempfile.TemporaryDirectory(prefix="quorum-benchmark-") as tmp:
        for mode in modes:
            corpus = Corpus(ROOT / "corpus")
            store = Store(Path(tmp) / f"{mode}.db")
            provider = MockProvider(latency=(0.15, 0.45))
            register_mocks(provider, corpus)
            rpm, tpm = (30, 6000) if mode.endswith("throttled") else (100000, 10000000)
            gov = Governor(org_rpm=rpm, org_tpm=tpm, model_rpm=rpm, model_tpm=tpm)
            t0 = time.monotonic()
            if mode == "langgraph":
                ctx = AgentContext(LLM(provider, gov), corpus, store, mode)
                result = await build_graph(ctx).compile().ainvoke({
                    "question": QUESTION, "findings": [], "verdicts": []
                })
                counts = {"claims_proposed": len(result["kept"]) + result["dropped"],
                          "rejected_by_prefilter": result["dropped"],
                          "claims_verified": len(result["verified"])}
            else:
                result = await Quorum(corpus, store, provider, gov).run(QUESTION, run_id=mode)
                if result.status != "done":
                    raise RuntimeError(f"benchmark failed: {result.metrics}")
                counts = {key: result.metrics[key] for key in (
                    "claims_proposed", "rejected_by_prefilter", "claims_verified")}
            spans = store.trace(mode)
            row = {"engine": mode, "wall_s": round(time.monotonic() - t0, 2),
                   "llm_calls": len(spans), "model_s": round(sum(s["ms"] for s in spans) / 1000, 2),
                   **gov.snapshot(), **counts}
            rows.append(row)
            store.db.close()
            print(json.dumps(row))
    artifact = {
        "question": QUESTION, "python": platform.python_version(),
        "platform": platform.platform(), "mock_latency": [0.15, 0.45],
        "packages": {d.metadata["Name"]: d.version for d in distributions()},
        "corpus_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((ROOT / "corpus").glob("*.md"))},
        "rows": rows,
    }
    destination = ROOT / ".eval"
    destination.mkdir(exist_ok=True)
    (destination / "benchmark.json").write_text(json.dumps(artifact, indent=2) + "\n")
    print("Evidence and environment: .eval/benchmark.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--throttled", action="store_true")
    asyncio.run(main(parser.parse_args().throttled))
