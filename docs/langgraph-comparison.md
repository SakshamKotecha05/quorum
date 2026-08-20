# The same pipeline, twice: hand-built vs LangGraph

`quorum/orchestrator.py` and `quorum/lg.py` run the identical four-stage pipeline over
the identical agents, prompts, LLM facade and governor.
The only variable is the orchestration layer.

Both were run against the deterministic mock provider so the comparison is not
confounded by sampling.

## Results

| | hand-built `run_dag` | LangGraph |
|---|---|---|
| orchestration LOC | 403 | 223 |
| model calls, cold run | 11 | 11 |
| effective parallelism | 1.99x | 1.98x |
| tokens | 9,257 | 9,357 |
| crash mid-fan-out: total calls to finish | **11** | 13 |
| work discarded by the crash | **0 calls** | 2 calls |

Throughput and cost are a wash, which is the expected result: both are running the
same agents over the same DAG, and the bottleneck is the model, not the scheduler.

## What LangGraph gave me for free

Real value, and I would not want to rebuild any of it:

- **Checkpointing.** `AsyncSqliteSaver` plus a `thread_id` and resume works. I wrote
  a store, a node table, a status enum and a requeue rule to get the same property.
- **`Send` fan-out with reducers.** `Annotated[list, operator.add]` handles fan-in
  merging declaratively. My scheduler collects results by walking node outputs.
- **Streaming, retry policies, and `interrupt`** for human-in-the-loop, none of which
  I implemented.
- **The graph is inspectable.** `app.get_graph().draw_mermaid()` renders the topology.
  Mine is legible only by reading the run loop.

## What it did not cover

These are the reasons the hand-built version still exists.

**Rate governance sits below the graph.** LangGraph schedules nodes; it has no concept
of a token-per-minute ceiling, so it cannot size a fan-out against one. The entire
governor -- token buckets, admission control, tier downgrade, daily-quota failover --
had to be written either way, and in the LangGraph port it lives *inside* the node
functions where the framework cannot see it. A superstep still dispatches every branch
at once and lets them queue in my limiter. `Quorum._concurrency()` derives its worker
count from the observed token ceiling; there is no equivalent hook.

**Supersteps are a barrier; a DAG is not.** `Send` fans out one superstep, and every
branch must finish before the next begins. The planner emits a dependency DAG, so the
LangGraph port has to flatten it into topological levels and loop a superstep per
level (`dependency_levels` plus the `advance` node). That reintroduces a
slowest-branch-wins barrier at every level, which `run_dag` does not have: it starts
any node the moment its own dependencies are done, regardless of what else is running.
On this corpus the plan is shallow enough that it does not matter. On a deep plan it
would.

**Checkpoint granularity is per superstep, not per node.** This is the one measurable
difference. Crashing partway through a six-branch verification fan-out, the two
verifier calls that had already succeeded were discarded and redone, costing 13 calls
total against a cold-run cost of 11. Per-node checkpointing finished in 11 with
nothing wasted. On a free tier metered by tokens per day, redoing completed branches
is the expensive kind of waste.

**Plan validation is still mine.** `Send` does not inspect the dependency set it fans
out, so a planner that emits a cycle is caught by `validate_dag`, not by the
framework.

## When I would use each

Use LangGraph when the topology is the hard part -- human-in-the-loop approvals,
long-lived conversational threads, streaming to a UI, branching that changes shape at
runtime. The checkpointer and `interrupt` alone justify it, and hand-rolling them is
how you get subtle bugs like the orphaned-node one I shipped and had to fix.

Own the scheduler when the *resource* is the hard part. Everything expensive I learned
on this project -- that free tiers have three nested ceilings needing three different
responses, that concurrency must derive from the token ceiling rather than a worker
count, that a daily quota exhaustion is a failover signal and not a retry signal --
lives at a layer the graph runtime does not model. I would have had to write all of it
regardless, and per-node checkpointing came along with it.

The honest summary: LangGraph would have saved me roughly a day of scheduler and
checkpoint work, and none of the work that turned out to be interesting.
