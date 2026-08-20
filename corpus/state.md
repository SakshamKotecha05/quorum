# State, Checkpointing and Durable Execution

An agent loop written as a plain Python loop loses all progress when the process dies. Durable execution frameworks address this by persisting the workflow state after every step, so conversation history survives activity failures and retries without re-running prior turns.

Durable execution is most valuable for long-running agent workflows that span hours or days, where the cost of re-running completed work is high. For a workflow that completes in seconds, the persistence overhead usually exceeds the benefit.

The minimum viable form of durability is a checkpoint written after every state transition of every node, keyed by a run identifier. Resuming is then equivalent to replaying the same workflow against a store that already contains answers for the completed nodes.

Checkpointing granularity is a tradeoff. Checkpointing per node makes resume precise but multiplies writes. Checkpointing per stage is cheaper but discards partial progress inside a stage, which matters most when a stage contains a wide parallel fan-out.

State management overhead is a recognised bottleneck in multi-agent systems, alongside context window limitations, tool execution latency, and the coordination cost of the agents themselves.
