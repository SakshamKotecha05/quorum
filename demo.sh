#!/usr/bin/env bash
# Offline recording demo. Add --full to include the ~50-second throttled benchmark.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
Q="What makes multi-agent orchestration hard to run in production?"
mkdir -p .eval
RID=$($PY -c 'import uuid; print(uuid.uuid4().hex[:12])')
DB=".eval/demo-$RID.db"
hr() { printf '\n\033[1m%s\033[0m\n' "$1"; }

hr "1. Offline research: plan, retrieve, filter, verify, synthesize."
"$PY" -m quorum --db "$DB" run "$Q" --mock --unthrottled --resume "$RID" --quiet

hr "2. Trace: successful model calls, tokens, latency, throttle wait."
"$PY" -m quorum --db "$DB" trace "$RID"

hr "3. Real process-kill recovery."
"$PY" evals/check_resume.py

hr "4. Same 84 claims, three acceptance rules."
"$PY" evals/run_eval.py

hr "5. Retrieval recall, including the miss."
"$PY" evals/retrieval_eval.py

if [[ "${1:-}" == "--full" ]]; then
    hr "6. Matched engines and the slower rate-limited run."
    "$PY" evals/benchmark.py --throttled
fi
hr "Recording script: docs/loom-prep.md"
