"""Evaluate the conversational layer.

    python -m eval.run_chat_eval

Measures four things the retrieval eval cannot:

  citation integrity   are the markers in an answer real and validated
  context diversity    how many distinct documents ground each answer
  guard accuracy       precision and recall on regulated-advice routing
  refusal behaviour    does the system decline when the corpus cannot answer

IMPORTANT, and stated up front because it changes how the numbers should be
read: under the default stub backend the answer text is extractive by
construction, so groundedness is trivially 1.0. That figure measures the audit
machinery, not a model's honesty. Run with DATAHUB_CHAT_BACKEND=openai to make
groundedness a claim about a real model.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
from typing import Dict, List

import yaml

from datahub_rag import store
from datahub_rag.chat import guard
from datahub_rag.chat.llm import get_llm
from datahub_rag.chat.session import ChatSession

EVAL_DIR = pathlib.Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"


def load() -> dict:
    return yaml.safe_load((EVAL_DIR / "chat_queries.yaml").read_text())


def eval_answers(questions: List[str], llm, expect_refusal: bool) -> dict:
    groundedness, diversity, refusals, fabricated, chunk_counts = [], [], 0, 0, []

    for question in questions:
        turn = ChatSession(llm=llm, persist=False).ask(question)
        groundedness.append(turn.audit.groundedness)
        diversity.append(len({s.document_id for s in turn.sources}))
        chunk_counts.append(len(turn.sources))
        refusals += int(turn.refused)
        fabricated += int(bool(turn.audit.invalid_markers))

    total = len(questions)
    return {
        "questions": total,
        "groundedness": round(statistics.fmean(groundedness), 4) if groundedness else 0.0,
        "answers_with_fabricated_citations": fabricated,
        "mean_distinct_documents": round(statistics.fmean(diversity), 2) if diversity else 0.0,
        "mean_chunks": round(statistics.fmean(chunk_counts), 2) if chunk_counts else 0.0,
        "refusal_rate": round(refusals / total, 4) if total else 0.0,
        "expected_refusal_rate": 1.0 if expect_refusal else 0.0,
    }


def eval_guard(guarded: List[dict], benign: List[str]) -> dict:
    true_positive = sum(1 for g in guarded if guard.check(g["question"]).triggered)
    domain_correct = sum(
        1 for g in guarded
        if g["domain"] in guard.check(g["question"]).domains
    )
    false_positive = sum(1 for q in benign if guard.check(q).triggered)

    recall = true_positive / len(guarded) if guarded else 0.0
    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) else 0.0
    )
    return {
        "regulated_questions": len(guarded),
        "benign_questions": len(benign),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "domain_accuracy": round(domain_correct / len(guarded), 4) if guarded else 0.0,
        "false_positives": false_positive,
    }


def render(results: dict, backend: str) -> str:
    answers = results["in_domain"]
    ood = results["out_of_domain"]
    g = results["guard"]

    out = [f"## Chat layer evaluation\n", f"Backend: `{backend}`\n"]

    out.append("### Citation integrity and context\n")
    out.append("| metric | in-domain | out-of-domain |")
    out.append("|---|---|---|")
    out.append(f"| questions | {answers['questions']} | {ood['questions']} |")
    out.append(f"| groundedness | {answers['groundedness']:.3f} | {ood['groundedness']:.3f} |")
    out.append(f"| answers with fabricated citations | {answers['answers_with_fabricated_citations']} | {ood['answers_with_fabricated_citations']} |")
    out.append(f"| mean distinct documents per answer | {answers['mean_distinct_documents']:.2f} | {ood['mean_distinct_documents']:.2f} |")
    out.append(f"| refusal rate | {answers['refusal_rate']:.3f} | {ood['refusal_rate']:.3f} |")
    out.append(f"| **expected** refusal rate | {answers['expected_refusal_rate']:.3f} | {ood['expected_refusal_rate']:.3f} |")

    out.append("\n### Regulated-advice guard\n")
    out.append("| metric | value |")
    out.append("|---|---|")
    out.append(f"| recall (regulated questions caught) | {g['recall']:.3f} |")
    out.append(f"| precision | {g['precision']:.3f} |")
    out.append(f"| correct domain assigned | {g['domain_accuracy']:.3f} |")
    out.append(f"| false positives on benign questions | {g['false_positives']} / {g['benign_questions']} |")

    gap = ood["refusal_rate"] - ood["expected_refusal_rate"]
    if gap < 0:
        out.append(
            f"\n> **Open weakness.** Out-of-domain questions are refused "
            f"{ood['refusal_rate']:.0%} of the time against an expected 100%. "
            f"Dense retrieval always returns its top-k regardless of how weak "
            f"the match is, so the answer stage is handed plausible-looking "
            f"context for a question the corpus cannot answer. Fixing this "
            f"needs a relevance floor on retrieval, not a prompt change."
        )
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default=None)
    args = ap.parse_args()

    llm = get_llm(args.backend)
    data = load()
    census = store.corpus_stats()
    print(f"corpus: {census['documents']} documents, {census['chunks']} chunks")
    print(f"backend: {llm.name}\n")

    results = {
        "backend": llm.name,
        "in_domain": eval_answers(data["in_domain"], llm, expect_refusal=False),
        "out_of_domain": eval_answers(data["out_of_domain"], llm, expect_refusal=True),
        "guard": eval_guard(data["guarded"], data["benign"]),
    }

    report = render(results, llm.name)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "chat.json").write_text(json.dumps(results, indent=2))
    (RESULTS_DIR / "chat.md").write_text(report)
    print(report)
    print(f"\nwritten to {RESULTS_DIR}/chat.md")


if __name__ == "__main__":
    main()
