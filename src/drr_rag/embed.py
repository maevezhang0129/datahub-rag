"""Embedding backends and the embedding pipeline stage.

The default backend is a local sentence-transformers model, so the project runs
end to end with no API key. A hosted OpenAI/Azure model is available by setting
DRR_EMBED_MODEL=text-embedding-3-small plus the corresponding credentials.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Protocol, Sequence

from pgvector.psycopg import register_vector

from . import config, store

# Embeddings are L2-normalised at write time, which makes cosine distance and
# inner product equivalent and keeps scores comparable across backends.


class Embedder(Protocol):
    dim: int

    def embed_passages(self, texts: Sequence[str]) -> List[List[float]]: ...
    def embed_query(self, text: str) -> List[float]: ...


class LocalEmbedder:
    """sentence-transformers backend. Runs on CPU; no network, no API key."""

    def __init__(self, spec: dict) -> None:
        from sentence_transformers import SentenceTransformer

        self.dim = spec["dim"]
        self._query_prefix = spec.get("query_prefix", "")
        self._passage_prefix = spec.get("passage_prefix", "")
        self._model = SentenceTransformer(
            spec["hf_name"],
            cache_folder=os.getenv("DRR_MODEL_CACHE", "/models"),
        )

    def embed_passages(self, texts: Sequence[str]) -> List[List[float]]:
        prefixed = [self._passage_prefix + t for t in texts]
        vecs = self._model.encode(prefixed, normalize_embeddings=True,
                                  batch_size=32, show_progress_bar=False)
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> List[float]:
        vec = self._model.encode([self._query_prefix + text],
                                 normalize_embeddings=True,
                                 show_progress_bar=False)[0]
        return vec.tolist()


class OpenAIEmbedder:
    """OpenAI or Azure OpenAI backend, selected by which env vars are present."""

    def __init__(self, spec: dict) -> None:
        self.dim = spec["dim"]
        self._deployment = spec["deployment"]

        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if azure_endpoint:
            from openai import AzureOpenAI

            self._client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=_require("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            )
        else:
            from openai import OpenAI

            self._client = OpenAI(api_key=_require("OPENAI_API_KEY"))

    def _embed(self, texts: Sequence[str]) -> List[List[float]]:
        resp = self._client.embeddings.create(model=self._deployment, input=list(texts))
        return [_normalise(d.embedding) for d in resp.data]

    def embed_passages(self, texts: Sequence[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Either provide it, or use the default local "
            f"model by unsetting DRR_EMBED_MODEL."
        )
    return value


def _normalise(vec: Sequence[float]) -> List[float]:
    total = sum(x * x for x in vec) ** 0.5
    return [x / total for x in vec] if total else list(vec)


_CACHE: dict[str, Embedder] = {}


def get_embedder(model_key: str | None = None) -> Embedder:
    """Return a cached embedder. Loading a local model takes seconds, so the
    process-level cache matters for the API and the eval harness."""
    spec = config.get_model(model_key)
    key = spec["key"]
    if key not in _CACHE:
        _CACHE[key] = (
            LocalEmbedder(spec) if spec["backend"] == "local" else OpenAIEmbedder(spec)
        )
    return _CACHE[key]


# --------------------------------------------------------------------------
# Pipeline stage
# --------------------------------------------------------------------------

def embed_corpus(
    model_key: str | None = None,
    batch_size: int = 128,
    limit: int = 0,
) -> dict:
    """Embed every chunk that has no vector for this model.

    Batches are claimed with FOR UPDATE ... SKIP LOCKED so that several workers
    can run concurrently against one database without handing the same chunk to
    two of them and without blocking each other.
    """
    spec = config.get_model(model_key)
    table = store.ensure_embedding_table(spec["key"])
    embedder = get_embedder(spec["key"])

    done = 0
    while True:
        take = batch_size if not limit else min(batch_size, limit - done)
        if take <= 0:
            break

        with store.connect() as conn:
            register_vector(conn)
            rows = conn.execute(
                f"""
                SELECT c.id, c.text
                FROM chunks c
                WHERE NOT EXISTS (
                    SELECT 1 FROM {table} e WHERE e.chunk_id = c.id
                )
                ORDER BY c.id
                LIMIT %s
                FOR UPDATE OF c SKIP LOCKED
                """,
                (take,),
            ).fetchall()

            if not rows:
                break

            vectors = embedder.embed_passages([r["text"] for r in rows])
            with conn.cursor() as cur:
                cur.executemany(
                    f"INSERT INTO {table} (chunk_id, embedding) VALUES (%s, %s) "
                    f"ON CONFLICT (chunk_id) DO NOTHING",
                    list(zip([r["id"] for r in rows], vectors)),
                )
            done += len(rows)
            print(f"  embedded {done} chunks", flush=True)

    return {"model": spec["key"], "table": table, "embedded": done}


def main() -> None:
    ap = argparse.ArgumentParser(description="Embed chunks into pgvector.")
    ap.add_argument("--model", default=None, help=f"default: {config.DEFAULT_MODEL}")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="0 = all pending")
    args = ap.parse_args()

    result = embed_corpus(args.model, args.batch_size, args.limit)
    print(f"embedded {result['embedded']} chunks with {result['model']} -> {result['table']}")


if __name__ == "__main__":
    main()
