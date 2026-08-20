# Rate Limits and Cost Control

Free and low-cost inference tiers impose hard ceilings on requests per minute, tokens per minute, and requests per day. These ceilings are typically enforced at the organisation level, so issuing multiple API keys does not multiply available quota.

A token-per-minute ceiling constrains parallelism directly. If the ceiling is six thousand tokens per minute and each agent call consumes two thousand tokens, then at most three calls can be admitted per minute regardless of how many workers are available. Launching more workers than the ceiling supports produces queueing and throttling, not speed.

The correct response to a token ceiling is admission control rather than retry. A token bucket that both requests and tokens must clear before a call is dispatched keeps concurrent workers from collectively overdrawing the quota between the moment they check the balance and the moment they spend it.

Retrying an individual rate-limited request in isolation converts a single throttle event into a cascade, because every other in-flight worker retries at the same moment. Parking all waiters for the duration of the retry-after interval is the behaviour that recovers.

Three levers reduce consumption against a fixed ceiling: trimming context so each call carries less input, batching independent judgements into a single call so the request ceiling is not the binding constraint, and caching identical sub-queries so repeated work is not paid for twice.
