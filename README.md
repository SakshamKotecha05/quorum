# Quorum

A multi-agent research orchestrator built for inference you do not pay for.

A supervisor agent decomposes a research question into a DAG of sub-questions.
An async scheduler fans specialist researcher agents across that DAG.
Every claim they produce is checked twice, first by string containment and then by a quorum of independent LLM judges, before a synthesizer is allowed to write it into the final brief.

The interesting constraint is that the whole thing runs on free-tier inference, which turns out to have three nested ceilings rather than one.
That ceiling is what shaped the design.
Most of what is worth reading below exists because of it.

Zero dependencies on any paid API. Runs fully offline against a deterministic mock provider.

---

## Why this is not just "agents calling agents"

| Problem | What most tutorials do | What Quorum does |
|---|---|---|
| Parallelism | fixed `concurrency=8` | concurrency derived from the token ceiling divided by observed call size |
| Rate limits | retry on 429 | token-bucket admission *before* dispatch; a 429 parks every waiter, not just the caller |
| Hallucinated citations | ask an LLM to check | string containment first, at zero token cost; the model only sees claims that are already anchored to real text |
| Verifier reliability | one judge | three judges with genuinely different criteria, majority vote |
| Crash mid-run | start over | per-node checkpoint; resume reuses completed work and requeues orphans |
| Daily quota exhaustion | retry the 429 | classified separately from a throttle and failed over to a model with quota left |
| Cost | ignored | every call is a trace span with tokens, latency and throttle wait |

---

## Architecture

```
question
   |
   v
[planner]  DEEP tier, 1 call
   |  emits a DAG of sub-questions with real dependency edges
   v
[researcher] x N   MID tier, scheduled by run_dag()
   |  BM25 retrieval over the corpus; every claim must carry a verbatim quote
   v
[prefilter]  0 model calls
   |  drops any claim whose quote does not literally occur in its cited source
   v
[verifier] x 3 lenses x ceil(claims/8) batches   MID tier
   |  support / overreach / attribution, majority vote
   v
[synthesizer]  DEEP tier, 1 call
   |  may only use claims that survived
   v
report + metrics + trace
```

`quorum/core.py` — tiers, token buckets, the governor, DAG validation, checkpoint store, BM25.
`quorum/llm.py` — provider-neutral chat client, JSON salvage, schema validation with a bounded repair turn.
`quorum/agents.py` — the four roles, their schemas, and the fault-injection fixtures.
`quorum/orchestrator.py` — `run_dag`, the generic scheduler, and the four-stage run.
`quorum/api.py` — FastAPI, SSE event stream, trace endpoints.

---

## Measured behaviour

All figures below are from real runs of this code, reproducible with the commands in each section.
Live figures come from Groq's free tier; comparative figures use the deterministic mock, so orchestration behaviour is measured without a sampling seed as a confounder.

### One live run, end to end

`openai/gpt-oss-120b` / `openai/gpt-oss-20b` / `qwen/qwen3.6-27b`, one question, free tier:

| | |
|---|---|
| wall clock | 237.5 s |
| model time | 32.6 s |
| aggregate throttle wait | 371.8 s |
| calls / tokens | 15 / 28,510 |
| claims proposed | 18 |
| dropped by containment prefilter | 1 (0 tokens spent) |
| rejected by 3-lens quorum | 6 |
| shipped to the report | 11 |
| verifier calls | 9, against 51 unbatched |
| JSON valid on first attempt | 15 / 15 |

Three things this surfaced that the mock could not:

**The deep tier was never available.** `gpt-oss-120b` was already out of daily quota at admission, so the planner ran one tier down and the run recorded 6 downgrades and 3 daily-quota 429s. It completed anyway. Failover is the only reason there is a report at all.

**Schema compliance was not the problem I expected.** I built the JSON salvage and repair loop assuming free models would fence their output and prepend prose. Across 15 calls the first-pass validity rate was 100% and the repair path never fired. `response_format: {"type": "json_object"}` plus the schema in the system prompt was sufficient. The salvage code stays because it costs nothing when unused, but it solved a problem that did not appear.

**One researcher correctly returned nothing.** Sub-question `s3` found no supporting excerpts and returned zero claims rather than inventing any, which is what the researcher prompt asks for and the failure mode most worth designing against.

### Free tiers have three ceilings, and two of them need opposite handling

Discovered the hard way on the first live run, which spent 360 seconds making zero successful calls:

```
Rate limit reached for model `openai/gpt-oss-120b` ... on tokens per day (TPD):
Limit 200000, Used 200000, Requested 1422. Please try again in 10m14.304s
```

| ceiling | scope | correct response |
|---|---|---|
| requests/minute | org-wide | queue, admit when the bucket refills |
| tokens/minute | per model | queue, and size concurrency from it |
| **tokens/day** | **per model** | **fail over to another model; waiting is useless** |

The first version modelled only the per-minute ceilings, so a daily exhaustion looked like an ordinary throttle.
The governor duly parked for the 615-second `retry-after`, twice, until the planner node hit its timeout and the run failed having produced nothing.

A 429 is not one condition.
`RateLimited` now carries a `daily` flag parsed from the error body: a per-minute throttle parks every waiter, a per-day exhaustion marks that model dead for the run and re-admits immediately so the tier ladder routes around it.
A per-model daily exhaustion also must not park the org-wide bucket, which is a separate resource.

The practical consequence is a configuration one: map each tier to a *different* model, so each tier draws on its own daily budget rather than three tiers racing for one.

### The rate ceiling dominates everything

The same 11-call run, once with ceilings lifted and once under Groq free-tier limits:

| | unthrottled | 30 RPM / 6K TPM |
|---|---|---|
| wall clock | 1.91 s | 69.60 s |
| model time | 3.81 s | 3.81 s |
| effective parallelism | 1.99x | 0.05x |
| aggregate throttle wait | 0 s | 155.42 s |
| tier downgrades | 0 | 2 |

Identical work, 36x the wall clock.
Nothing about the agents changed; the entire difference is queueing against a 6,000 token/minute bucket for a run that consumes 9,257 tokens.

This is why the token ceiling, not the worker count, sets concurrency.

```
python -m quorum run "..." --mock --unthrottled
python -m quorum run "..." --mock
```

### Verification: two stages, because they catch different things

Fault-injection harness over 5 questions and 84 claims.
20% of claims are given a fabricated citation, another 20% quote a real source but overstate it, and each judge is corrupted independently 20% of the time.

| configuration | precision | recall | F1 | leaked | true claims wrongly cut | calls |
|---|---|---|---|---|---|---|
| single agent, no verification | 0.40 | 1.00 | 0.57 | 100% | 0 | 5 |
| quorum, 1 lens | 0.93 | 0.75 | 0.83 | 9% | 13 | 41 |
| quorum, 3 lenses (majority 2/3) | 0.94 | 0.96 | **0.95** | 9% | 2 | 61 |

Two results worth stating plainly:

**The quorum's value is recall, not precision.**
A single strict judge already stops most hallucinations, but it throws away 13 of 51 true claims doing it.
The majority vote recovers 11 of those 13 for a 50% increase in verifier calls.
That is the opposite of what the "ensemble to catch more errors" framing predicts.

**Correlated judges are worthless.**
The first version of this harness seeded all three lenses identically, so they returned the same verdict every time and majority-of-3 measured exactly the same as a single judge.
The lens has to be part of the agent's identity, not just its prompt.

```
python evals/run_eval.py
```

### Free work beats cheap work

Fabricated citations are caught by `str.__contains__` against the cited chunk.
Of 12 claims in a representative run, 3 were dropped this way before a single verification token was spent.
Only claims already anchored to real text reach the model quorum, which then only has to judge the harder question of whether the quote actually supports the claim.

### Batching is what makes 30 RPM survivable

Verification is embarrassingly parallel across claims, which is exactly the shape that a request-per-minute ceiling punishes.
Batching 8 claims per call turns 27 verifier requests into 6.
Under a 30 RPM ceiling that is the difference between verification fitting in one minute and not.

### Crash resume

```
python -m quorum run "..." --mock --resume $RID   # kill -9 mid-flight
python -m quorum run "..." --mock --resume $RID   # resume
```

Killed at 4 of 8 nodes complete, the resumed run finished all 11 nodes with 7 model calls instead of 11.

The subtle part is not the checkpoint, it is the orphans.
A node still marked `RUNNING` when the process dies is neither `PENDING` (so the scheduler never schedules it) nor `DONE` (so nothing notices it is missing).
The first version of resume reported `status=done` having silently dropped four subtasks.
Recovery has to requeue in-flight nodes, not just skip completed ones.

---

## Running it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# fully offline, no key, no account
.venv/bin/python -m quorum run "What makes multi-agent orchestration hard?" --mock

# against a free Groq key
export GROQ_API_KEY=...
.venv/bin/python -m quorum models          # what the provider actually serves today
.venv/bin/python -m quorum run "..."

# or fully local, no key at all
export QUORUM_BASE_URL=http://localhost:11434/v1
export QUORUM_MODEL_DEEP=qwen3:8b QUORUM_MODEL_MID=qwen3:8b QUORUM_MODEL_FAST=qwen3:8b

.venv/bin/python -m quorum runs            # recent runs
.venv/bin/python -m quorum trace <run_id>  # per-call tokens, latency, throttle wait
.venv/bin/python -m pytest tests/ -q
.venv/bin/uvicorn quorum.api:app           # POST /runs, GET /runs/{id}/events (SSE)
```

Any OpenAI-compatible endpoint works.
Model IDs are read from the environment rather than hardcoded, because they churn: Groq retired the Llama line during this project's development, and a hardcoded tier map would have been a code change instead of a config change.

---

## Built twice

The same pipeline also exists as a LangGraph implementation in `quorum/lg.py`, over
the identical agents and governor, so the orchestration layer can be compared directly.
Throughput and cost come out a wash; the measurable difference is crash recovery, where
LangGraph's per-superstep checkpointing discarded two completed branches of a failed
fan-out (13 calls to finish against a cold-run cost of 11) versus per-node
checkpointing finishing in 11 with nothing wasted.

Full writeup, including what the framework does better: [docs/langgraph-comparison.md](docs/langgraph-comparison.md).

```
python -m quorum.lg "What makes multi-agent orchestration hard?"
```

## Deliberate omissions

Marked in the source with `ponytail:` comments, with the upgrade trigger for each.

- **BM25, not embeddings.** Lexical match is the right prior when the verifier needs the literal quote to exist. Revisit when recall@5 on the golden set drops.
- **One SQLite connection behind a lock.** Sub-millisecond writes, tens per run. Revisit on measured contention, and then with a single writer task, not a different database.
- **chars/4 token estimation.** A tokenizer round-trip before every admission costs more than the estimate error, and the limiter reconciles against reported usage within the minute.
- **In-process run registry in the API.** The checkpoint store is the durable half; a restarted server resumes any run by id.

## Known limits

- The scheduler cannot expand the DAG mid-run, so a researcher cannot spawn follow-up researchers. Four fixed stages, dynamic within each.
- Retries are immediate rather than backed off; the LLM layer owns 429 backoff, and the scheduler only retries genuinely transient failures.
- Web search is not wired in. The corpus is the only source, which is what makes citations verifiable by containment.
