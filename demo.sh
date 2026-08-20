#!/usr/bin/env bash
# Three-minute walkthrough. Runs fully offline -- no API key, no network, no cost.
#   ./demo.sh
set -euo pipefail
PY=${PY:-.venv/bin/python}
Q="What makes multi-agent orchestration hard to run in production?"
RID=$($PY -c "import uuid;print(uuid.uuid4().hex[:12])")
rm -f demo.db
hr() { printf '\n\033[1m%s\033[0m\n%s\n' "$1" "$(printf '%.0s-' {1..70})"; }

hr "1. A run. Planner emits a DAG, researchers fan out, verifiers vote, synthesizer writes."
$PY -m quorum --db demo.db run "$Q" --mock --unthrottled --resume "$RID" 2>&1 | tail -32

hr "2. Every model call is a trace span: tier, tokens, latency, time lost to throttling."
$PY -m quorum --db demo.db trace "$RID"

hr "3. Same run under real free-tier ceilings (30 req/min, 6k tok/min). Identical work."
$PY -m quorum --db demo.db run "$Q" --mock --quiet 2>&1 \
  | grep -E "wall_s|sum_call_s|effective_parallelism|throttle_wait_s|tier_downgrades"
echo "   ^ same calls, same tokens, an order of magnitude more wall clock."
echo "     The token ceiling sets concurrency, not the worker count."

hr "4. Crash recovery. Kill a run mid-flight, resume it, count the replayed work."
RID2=$($PY -c "import uuid;print(uuid.uuid4().hex[:12])")
$PY -m quorum --db demo.db run "$Q" --mock --mock-latency 0.8 --resume "$RID2" >/dev/null 2>&1 &
PID=$!; $PY -c "import time;time.sleep(9)"; kill -9 $PID 2>/dev/null || true; wait $PID 2>/dev/null || true
$PY -c "
from quorum.core import Store; from collections import Counter
print('   at kill -9:', dict(Counter(n.status.value for n in Store('demo.db').load_nodes('$RID2').values())))"
$PY -m quorum --db demo.db run "$Q" --mock --mock-latency 0.8 --resume "$RID2" 2>&1 \
  | grep -E "\[resume\]|requests_used|^  nodes"
echo "   ^ completed nodes reused; nodes caught mid-flight requeued rather than silently dropped."

hr "5. Does the orchestration earn its cost? Fault-injection eval, known defect rates."
$PY evals/run_eval.py 2>&1 | tail -12

hr "6. The same pipeline in LangGraph, for comparison. See docs/langgraph-comparison.md."
$PY -m quorum.lg "$Q" 2>&1 | tail -7

hr "Done. Full writeup: README.md"
