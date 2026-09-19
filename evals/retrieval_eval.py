"""Recall@K on a small authored, in-corpus relevance set, not a held-out benchmark."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quorum.core import Corpus

ROOT = Path(__file__).resolve().parents[1]


def evaluate(corpus, golden):
    rows = []
    for case in golden:
        relevant = set(case["relevant"])
        if not relevant or any(corpus.get(sid) is None for sid in relevant):
            raise ValueError(f"empty or stale relevance labels: {case['query']}")
        ranked = [c.source_id for c in corpus.search(case["query"], k=5)]
        rows.append({**case, "retrieved": ranked, **{
            f"recall@{k}": len(relevant.intersection(ranked[:k])) / len(relevant)
            for k in (1, 3, 5)
        }})
    if not rows:
        raise ValueError("golden set is empty")
    return {"queries": len(rows), "chunks": len(corpus.chunks), "rows": rows,
            **{f"recall@{k}": sum(r[f"recall@{k}"] for r in rows) / len(rows)
               for k in (1, 3, 5)}}


def main():
    golden = json.loads((ROOT / "evals/retrieval_golden.json").read_text())
    result = evaluate(Corpus(ROOT / "corpus"), golden)
    destination = ROOT / ".eval"
    destination.mkdir(exist_ok=True)
    (destination / "retrieval.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"BM25: {result['queries']} authored queries over {result['chunks']} chunks")
    for k in (1, 3, 5):
        print(f"Recall@{k}: {result[f'recall@{k}']:.3f}")
    print("Macro recall = mean of retrieved relevant chunks / labelled relevant chunks per query.")
    print("Small in-corpus diagnostic, not evidence of general retrieval quality.")
    for row in result["rows"]:
        if row["recall@5"] < 1:
            print(f"Miss at 5: {row['query']} -> {row['retrieved']}")
    print("Per-query evidence: .eval/retrieval.json")


if __name__ == "__main__":
    main()
