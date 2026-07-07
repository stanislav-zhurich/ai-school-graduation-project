"""ChromaDB query helpers used by the agent's ``search_wine_descriptions`` tool."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import chromadb

import config


@lru_cache(maxsize=1)
def get_collection():
    """Open (and cache) the persistent ``wines`` collection with its embedding function."""
    embedding_fn = config.get_embedding_function()
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_collection(
        name=config.COLLECTION_NAME, embedding_function=embedding_fn
    )


def search_wine_descriptions(
    query: str,
    n_results: int = 5,
    where: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Semantic search over wine descriptions.

    Returns a list of hits, each with ``description``, ``metadata`` (country, variety,
    points, price, title, row_id, ...) and ``distance``.
    """
    collection = get_collection()
    result = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where or None,
    )

    hits: list[dict[str, Any]] = []
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    for doc, meta, dist in zip(documents, metadatas, distances):
        hits.append({"description": doc, "metadata": meta, "distance": dist})
    return hits
