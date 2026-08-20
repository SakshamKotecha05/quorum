# Evaluating Agent Systems

The biggest pitfall in agent evaluation is overfitting to benchmarks. A system tuned until it scores well on a fixed task set frequently fails on tasks drawn from the same distribution but phrased differently.

Adversarial evaluation is the standard mitigation. Benchmark tasks are systematically modified, by rephrasing, by reordering, or by injecting distractors, and the evaluator checks whether performance holds under the modification.

Fault injection is a complementary technique for evaluating a verification stage specifically. A known fraction of deliberately defective outputs is injected into the pipeline, and the verifier is scored on precision and recall against that known ground truth. This avoids the bootstrap problem of needing a hand-labelled corpus before any verifier can be measured.

Ensembling noisy judges is effective when individual judge error is independent. Three independent judges voting by majority reduce error substantially relative to a single judge, provided the judges are given genuinely different evaluation criteria rather than the same criterion three times.

Evaluation of agentic systems increasingly covers observability as a first-class dimension. Monitoring, evaluation, and debugging infrastructure for autonomous systems is materially more complex than for traditional software, because failures are probabilistic and often silent.
