# ADR 005: A calibrated relevance gate, not a prompt instruction

## Status
Accepted.

## Context
The answer prompt says: *"If the sources do not answer the question, say so
plainly and stop."* Measuring it showed that instruction does essentially
nothing, because of where the failure actually originates.

Dense retrieval always returns its top-k. Ask "how do I configure a Kubernetes
ingress controller" against a corpus of disaster-risk literature and the ANN
still returns eight chunks — its nearest neighbours, just distant ones. Those
chunks reach the answer stage looking exactly like good context: real
documents, real titles, real prose. The model then does what it was asked to
do, grounds an answer in them, and cites them correctly.

The first chat evaluation measured a **0% refusal rate on out-of-domain
questions against an expected 100%**. Every single one produced a confidently
cited answer.

## Decision
Gate on retrieval score before generating, in `ChatSession.ask`.

The signal is the **top-1 dense cosine similarity**, not the hybrid score. RRF
produces reciprocal-rank sums with no absolute meaning — around 0.05 whether
the match is perfect or hopeless — so a threshold on them would be
uninterpretable. Cosine similarity is comparable across queries.

The threshold is calibrated against the evaluation set rather than guessed:

| group | n | min | median | max |
|---|---|---|---|---|
| answerable | 10 | **0.743** | 0.830 | 0.924 |
| unanswerable | 6 | 0.521 | 0.536 | **0.629** |

The two groups separate cleanly with no overlap. `MIN_RELEVANCE = 0.68` sits in
the gap. After the change, out-of-domain refusal is 100% and in-domain refusal
stays 0%.

## Consequences
The threshold is specific to **this corpus and this embedding model**. Cosine
scales differ between models, and a broader corpus raises the floor for
genuinely unanswerable questions. It is exposed as `DATAHUB_MIN_RELEVANCE` and
`make eval-chat` recalibrates it; a test asserts the underlying signal still
separates, so a model change fails loudly rather than silently degrading.

Calibrating on the same 16 questions used to report the result is a real
limitation — with a clean gap this wide it is defensible, but the honest
version of this needs a held-out set, and a larger one.

The gate runs before generation, so an out-of-scope question costs no answer
tokens. Cheaper and safer in the same move.

The general lesson: **a behaviour you need is not a behaviour you can prompt
for.** The prompt asked for refusal and the model complied with the letter of
it — the sources genuinely did seem to answer the question. The fix belonged in
retrieval, where the actual information was.
