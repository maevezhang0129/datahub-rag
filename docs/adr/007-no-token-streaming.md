# ADR 007: The web UI does not stream tokens

## Status
Accepted.

## Context
Streaming is the default interaction pattern for chat interfaces, and its
absence reads as a missing feature. The web UI added in v0.3 does not stream:
the reader waits for the whole answer, then sees it at once.

That was not a shortcut. It is forced by where verification happens.

## Decision
Return the finished, audited answer in one response.

The citation audit in `chat/answer.py` runs **over the complete text**. It finds
every `[n]` marker, checks each against the sources that were actually supplied,
and deletes the ones pointing at sources that do not exist — a `[7]` against
five sources is a fabricated reference, and leaving it in presents the reader
with a citation they cannot follow. It then measures what fraction of
substantive sentences carry a citation at all.

Both steps need the whole answer. Marker validity is knowable per-token, but
`normalise_citations` has to see the sentence boundary to decide which sentence
a trailing marker belongs to, and groundedness is a ratio over all sentences.

So streaming would mean displaying text that has not been verified yet, and
then retracting parts of it. The reader would watch a citation appear and then
vanish. For a system whose entire claim is that its citations are checked
rather than promised, showing unchecked output first is the wrong trade — the
streamed frame is a lie the audit corrects a second later.

## Consequences
Answers appear after a wait rather than progressively. On this corpus that wait
is 150–260 ms with the stub backend, where it does not matter; with a hosted
model it would be seconds, where it would.

In exchange, everything the UI displays has passed the audit, and the inspector
panel can show the audit result *alongside* the answer instead of after it.

The honest alternative, if streaming became necessary: stream the prose with
markers withheld, then attach citations on completion. That keeps the
verification guarantee and gives up only the appearance of citations arriving
with their sentence. It is more machinery than this project needs, and it is
worth stating that the reason for not streaming is a verification property
rather than an unimplemented feature.
