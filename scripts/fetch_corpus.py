#!/usr/bin/env python3
"""Build the frozen corpus from public, key-free APIs.

Two sources, deliberately different in shape:

  wikipedia  long-form articles (~7k tokens each) from disaster-risk categories.
             These are what make chunking and overlap actually matter.
  openalex   scholarly abstracts on disaster risk reduction. Short, numerous,
             and precisely dated, which gives the date filters something real.

Output is JSONL under eval/corpus/, committed to the repository so that the
demo and the evaluation are reproducible offline and identical for everyone.

Usage:  python scripts/fetch_corpus.py --wikipedia 120 --openalex 350
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any, Dict, Iterable, List

import requests

CORPUS_DIR = pathlib.Path(__file__).resolve().parents[1] / "eval" / "corpus"
UA = "datahub-rag-portfolio/0.1 (https://github.com/; educational portfolio project)"
MAX_BODY_CHARS = 40_000

WIKIPEDIA_CATEGORIES = [
    "Category:Disaster preparedness",
    "Category:Emergency management",
    "Category:Natural disasters",
    "Category:Climate change adaptation",
    "Category:Humanitarian aid",
    "Category:Floods",
    "Category:Earthquakes",
    "Category:Droughts",
    "Category:Tropical cyclones",
    "Category:Wildfires",
]

OPENALEX_QUERIES = [
    "disaster risk reduction",
    "disaster resilience community",
    "early warning system hazard",
    "flood risk management",
    "climate adaptation vulnerability",
    "seismic risk assessment",
]


def _get(url: str, params: dict, tries: int = 4) -> dict:
    """GET with linear backoff. Public APIs rate-limit; a portfolio script that
    dies on the first 429 is not a good advertisement."""
    for attempt in range(1, tries + 1):
        resp = requests.get(url, params=params, headers={"User-Agent": UA}, timeout=60)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (429, 500, 502, 503) and attempt < tries:
            time.sleep(attempt * 2)
            continue
        resp.raise_for_status()
    raise RuntimeError(f"giving up on {url}")


# --------------------------------------------------------------------- wiki
WIKI_API = "https://en.wikipedia.org/w/api.php"


def wikipedia_titles(limit: int) -> List[str]:
    seen: list[str] = []
    per_category = max(1, limit // len(WIKIPEDIA_CATEGORIES) + 1)
    for category in WIKIPEDIA_CATEGORIES:
        data = _get(WIKI_API, {
            "action": "query", "format": "json", "list": "categorymembers",
            "cmtitle": category, "cmlimit": per_category, "cmtype": "page",
        })
        for member in data.get("query", {}).get("categorymembers", []):
            if member["title"] not in seen:
                seen.append(member["title"])
    return seen[:limit]


def fetch_wikipedia(limit: int) -> Iterable[Dict[str, Any]]:
    titles = wikipedia_titles(limit)
    print(f"  wikipedia: {len(titles)} titles", file=sys.stderr)

    # One title per request: MediaWiki only returns a full (non-intro) extract
    # for a single page per query, regardless of exlimit. Batching here silently
    # yields one article per batch.
    kept = 0
    for index, title in enumerate(titles, 1):
        data = _get(WIKI_API, {
            "action": "query", "format": "json", "prop": "extracts|revisions",
            "explaintext": 1, "rvprop": "timestamp",
            "titles": title, "redirects": 1,
        })
        for page in data.get("query", {}).get("pages", {}).values():
            body = (page.get("extract") or "").strip()
            if len(body) < 500:  # stubs carry no retrievable content
                continue
            revisions = page.get("revisions") or [{}]
            kept += 1
            yield {
                "source": "wikipedia",
                "source_id": str(page["pageid"]),
                "url": f"https://en.wikipedia.org/?curid={page['pageid']}",
                "title": page["title"],
                "body": body[:MAX_BODY_CHARS],
                "published_at": (revisions[0].get("timestamp") or "")[:10] or None,
                "meta": {"license": "CC BY-SA 4.0", "retrieved_from": "en.wikipedia.org"},
            }
        if index % 25 == 0:
            print(f"    {index}/{len(titles)} fetched, {kept} kept", file=sys.stderr)
        time.sleep(0.15)


# ----------------------------------------------------------------- openalex
OPENALEX_API = "https://api.openalex.org/works"


def _reconstruct_abstract(inverted: Dict[str, List[int]]) -> str:
    """OpenAlex ships abstracts as {word: [positions]}; invert it back to prose."""
    if not inverted:
        return ""
    positions: list[tuple[int, str]] = []
    for word, spots in inverted.items():
        positions.extend((spot, word) for spot in spots)
    positions.sort()
    return " ".join(word for _, word in positions)


def fetch_openalex(limit: int, mailto: str) -> Iterable[Dict[str, Any]]:
    per_query = max(1, limit // len(OPENALEX_QUERIES) + 1)
    emitted = 0
    for query in OPENALEX_QUERIES:
        cursor = "*"
        got = 0
        while got < per_query and emitted < limit:
            data = _get(OPENALEX_API, {
                "search": query,
                "per-page": min(200, per_query - got),
                "cursor": cursor,
                "mailto": mailto,
                "filter": "has_abstract:true,language:en",
            })
            results = data.get("results", [])
            if not results:
                break
            for work in results:
                abstract = _reconstruct_abstract(work.get("abstract_inverted_index") or {})
                if len(abstract) < 400:
                    continue
                yield {
                    "source": "openalex",
                    "source_id": work["id"].rsplit("/", 1)[-1],
                    "url": work.get("doi") or work["id"],
                    "title": work.get("title") or "",
                    "body": abstract[:MAX_BODY_CHARS],
                    "published_at": work.get("publication_date"),
                    "meta": {
                        "query": query,
                        "cited_by_count": work.get("cited_by_count", 0),
                        "license": "CC0 (OpenAlex metadata)",
                    },
                }
                got += 1
                emitted += 1
                if emitted >= limit:
                    break
            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break
            time.sleep(0.2)
        print(f"  openalex[{query}]: {got}", file=sys.stderr)


# ---------------------------------------------------------------------- io
def write_jsonl(path: pathlib.Path, docs: Iterable[Dict[str, Any]]) -> int:
    seen: set[str] = set()
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for doc in docs:
            key = f"{doc['source']}:{doc['source_id']}"
            if key in seen:
                continue
            seen.add(key)
            handle.write(json.dumps(doc, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wikipedia", type=int, default=120)
    ap.add_argument("--openalex", type=int, default=350)
    ap.add_argument("--mailto", default="portfolio@example.com",
                    help="OpenAlex polite-pool contact")
    args = ap.parse_args()

    if args.wikipedia:
        n = write_jsonl(CORPUS_DIR / "wikipedia.jsonl", fetch_wikipedia(args.wikipedia))
        print(f"wrote {n} wikipedia documents")
    if args.openalex:
        n = write_jsonl(CORPUS_DIR / "openalex.jsonl", fetch_openalex(args.openalex, args.mailto))
        print(f"wrote {n} openalex documents")


if __name__ == "__main__":
    main()
