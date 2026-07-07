"""Azure OpenAI embedding function (EPAM proxy), used at index and query time.

A Chroma-compatible embedding function that explicitly uses ``AzureOpenAI`` from the
openai SDK. The same instance is used by ``indexer.py`` (build time) and ``rag.py``
(query time) so index and query embeddings always come from the same deployment.
"""
from __future__ import annotations

from chromadb import Documents, EmbeddingFunction, Embeddings

import config

# Azure/OpenAI embeddings accept an array of inputs per request; sub-batch large
# calls to stay within request size/token limits.
_BATCH_SIZE = 256


class AzureOpenAIEmbeddingFunction(EmbeddingFunction[Documents]):
    """Chroma embedding function backed by an Azure OpenAI embedding deployment.

    Subclasses Chroma's ``EmbeddingFunction`` so both document embedding
    (``__call__``) and query embedding (``embed_query``, which defaults to
    ``__call__``) route through the same Azure deployment. The Azure client is
    created by ``config`` so all client construction lives in one place.
    """

    def __init__(
        self,
        model: str | None = None,
        max_retries: int = 5,
        batch_size: int = _BATCH_SIZE,
    ) -> None:
        self.model = model or config.EMBEDDING_MODEL
        self.batch_size = batch_size
        self._client = config.get_embedding_client(max_retries=max_retries)

    def __call__(self, input: Documents) -> Embeddings:
        texts = [str(t) for t in input]
        embeddings: Embeddings = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            response = self._client.embeddings.create(model=self.model, input=chunk)
            embeddings.extend(item.embedding for item in response.data)
        return embeddings

    @staticmethod
    def name() -> str:
        """Identifier used by Chroma for the embedding-function config."""
        return "azure-openai"
