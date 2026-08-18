"""Interactive chat CLI.

    python -m datahub_rag.chat.cli
    python -m datahub_rag.chat.cli --ask "what drives flash droughts"
"""

from __future__ import annotations

import argparse
import json
import sys

from . import guard  # noqa: F401  (import kept so guard config loads eagerly)
from .llm import get_llm
from .session import ChatSession, Turn

BANNER = """\
datahub-rag chat  ·  backend={backend}  ·  retrieval={mode}
  /sources   show the sources behind the last answer
  /audit     show the citation audit for the last answer
  /memory    show the running conversation memory
  /quit      exit
"""


def render(turn: Turn, verbose: bool = False) -> str:
    out = [turn.answer]
    if turn.sources:
        out.append("")
        cited = turn.audit.cited_sources if turn.audit else set()
        for index, source in enumerate(turn.sources, start=1):
            if index in cited or verbose:
                mark = "*" if index in cited else " "
                out.append(f" {mark}[{index}] {source.title[:70]}")
                if source.url:
                    out.append(f"      {source.url}")
    if turn.audit and turn.audit.invalid_markers:
        out.append(f"\n  ! fabricated citations removed: {turn.audit.invalid_markers}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Chat with the DRR knowledge base.")
    ap.add_argument("--ask", help="ask one question and exit")
    ap.add_argument("--session", type=int, help="resume an existing session id")
    ap.add_argument("--mode", default="hybrid", choices=["vector", "lexical", "hybrid"])
    ap.add_argument("--backend", default=None, help="stub (default) or openai")
    ap.add_argument("--json", action="store_true", help="emit the full turn as JSON")
    ap.add_argument("--no-persist", action="store_true")
    args = ap.parse_args()

    llm = get_llm(args.backend)
    session = ChatSession(session_id=args.session, llm=llm, mode=args.mode,
                          persist=not args.no_persist)

    if args.ask:
        turn = session.ask(args.ask)
        print(json.dumps(turn.as_dict(), indent=2) if args.json else render(turn))
        return

    print(BANNER.format(backend=llm.name, mode=args.mode))
    if session.session_id:
        print(f"session {session.session_id}\n")

    last: Turn | None = None
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question in ("/quit", "/exit"):
            break
        if question == "/sources":
            print(render(last, verbose=True) if last else "no turn yet")
            continue
        if question == "/audit":
            print(json.dumps(last.audit.as_dict(), indent=2) if last and last.audit
                  else "no turn yet")
            continue
        if question == "/memory":
            print(json.dumps(session.memory.as_dict(), indent=2))
            continue

        try:
            last = session.ask(question)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            continue
        print("\n" + render(last) + "\n")


if __name__ == "__main__":
    main()
