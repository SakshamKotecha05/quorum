# Multi-Agent Orchestration Patterns

Five topologies dominate production multi-agent systems: fan-out, pipeline, debate, supervisor, and swarm. The supervisor pattern has become the default choice, with several major vendors converging on orchestrator-plus-isolated-subagents as their reference architecture.

In the supervisor pattern a single orchestrator owns the full conversation context and spawns ephemeral subagents that return compressed summaries. There is no peer-to-peer communication between subagents. This makes the execution path a tree, which is what makes failures traceable.

The swarm pattern allows peer-to-peer handoffs between agents. Swarm offers more flexibility but obscures the execution path, which makes it difficult to trace errors when a chain of agents fails. Debugging a failed swarm run generally requires reconstructing the handoff order from logs.

Two failure signatures recur in supervisor deployments. Hub fragility means one bad routing decision at the orchestrator cascades into every downstream specialist. Translation loss means the orchestrator paraphrases subagent output on the way back, and detail is silently lost at the center of the tree.

A practical rule that follows from these failure modes: the orchestrator must have a dedicated system prompt, never a reuse of the subagent prompt, and subagents must receive role-scoped context rather than the full conversation.
