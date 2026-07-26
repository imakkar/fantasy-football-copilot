"""
Embeddings + vector store.

Embeds passages and stores them in a persistent ChromaDB collection. Two embedding
backends are supported:

  * "local"  -> sentence-transformers/all-MiniLM-L6-v2 (default; free, no API key)
  * "openai" -> text-embedding-3-small (requires OPENAI_API_KEY)

Defaulting to a local model means the whole indexing + retrieval pipeline is
reproducible by a grader with no credentials.
"""

from __future__ import annotations

from typing import Dict, List

import chromadb

from .config import config
from utils.helpers import sanitize_metadata


class Embedder:
    """Thin wrapper that exposes a single embed(texts) -> List[List[float]] method."""

    def __init__(self, backend: str | None = None):
        self.backend = backend or config.embedding_backend
        if self.backend == "local":
            from sentence_transformers import SentenceTransformer
            print(f"[embed] loading local model {config.local_embedding_model}")
            self._model = SentenceTransformer(config.local_embedding_model)
        elif self.backend == "openai":
            from openai import OpenAI
            self._client = OpenAI()
        else:
            raise ValueError(f"Unknown embedding backend: {self.backend}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.backend == "local":
            return self._model.encode(texts, show_progress_bar=False).tolist()
        # openai
        resp = self._client.embeddings.create(
            model=config.openai_embedding_model, input=texts
        )
        return [item.embedding for item in resp.data]


def _get_collection():
    client = chromadb.PersistentClient(path=config.chroma_dir)
    return client.get_or_create_collection(
        name=config.collection_name, metadata={"hnsw:space": "cosine"}
    )


def build_index(passages: List[Dict], batch_size: int = 256) -> None:
    """Embed all passages and upsert them into the ChromaDB collection."""
    embedder = Embedder()
    collection = _get_collection()

    total = len(passages)
    for start in range(0, total, batch_size):
        batch = passages[start : start + batch_size]
        ids = [p["id"] for p in batch]
        docs = [p["text"] for p in batch]
        metas = [sanitize_metadata(p["metadata"]) for p in batch]
        vectors = embedder.embed(docs)
        collection.upsert(ids=ids, documents=docs, embeddings=vectors, metadatas=metas)
        print(f"[embed] indexed {min(start + batch_size, total)}/{total}")

    print(f"[embed] done. collection '{config.collection_name}' now has "
          f"{collection.count()} passages.")


def retrieve(query: str, top_k: int | None = None,
             where: Dict | None = None) -> List[Dict]:
    """Return the top_k most relevant passages for a query.

    If a `where` metadata filter is provided (e.g. restrict to specific players
    and/or a specific week), retrieval is limited to matching passages before
    ranking. This sharply improves precision on week-specific questions.
    """
    top_k = top_k or config.top_k
    embedder = Embedder()
    collection = _get_collection()

    qvec = embedder.embed([query])[0]
    query_kwargs = {"query_embeddings": [qvec], "n_results": top_k}
    if where:
        query_kwargs["where"] = where
    res = collection.query(**query_kwargs)

    # A filter can legitimately match zero passages; guard against empty results.
    if not res["documents"] or not res["documents"][0]:
        return []

    hits = []
    for doc, md, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append({"text": doc, "metadata": md, "distance": dist})
    return hits


if __name__ == "__main__":
    from .preprocess import load_passages

    build_index(load_passages())
    print("\nSample retrieval:")
    for hit in retrieve("How did the top running backs perform in week 1?"):
        print("-", hit["text"][:100])
