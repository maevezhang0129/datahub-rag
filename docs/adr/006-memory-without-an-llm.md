# ADR 006: Conversation memory maintained without an LLM call

## Status
Accepted.

## Context
The conventional design asks the model to summarise the conversation each turn
and feeds that summary back on the next one. It works, and it costs a second
model call on every turn.

## Decision
Maintain memory locally: a rolling window of recent turns, a term-frequency
table of recurring topics, and a tracked subject for resolving follow-ups.

What follow-up resolution actually needs is narrow. "Tell me more about that"
fails to retrieve anything on its own not because the system lacks a nuanced
summary, but because it lacks a **subject** to substitute in. A tracked string
solves that. The rest — recent turns, recurring terms — is context for the
prompt, and a window plus a counter produces that adequately.

The rule that matters: a follow-up preserves the subject, a fresh question
replaces it. Without it, "what about Ghana?" resets the topic to "Ghana" and
loses what was being asked about it.

## Consequences
Halves the model calls per turn and removes their latency from the critical
path.

More importantly it makes the conversational layer **deterministic**, and
therefore testable. Memory behaviour is covered by ordinary unit tests with no
mocking and no API key, which is also what lets the stub backend exercise the
full chain end to end.

The limitation is real: this cannot summarise or infer. It tracks what was
discussed, not what was concluded. A long, topic-shifting conversation will
hold a cruder picture than an LLM-written summary would. For a
retrieval-grounded assistant, where each turn is answered from the corpus
rather than from conversation history, that trade is worth taking — for an
assistant that reasons over the dialogue itself, it would not be.
