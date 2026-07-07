"""Shared configuration: environment loading and the Azure OpenAI client factory.

All secrets are read from the environment (`.env` is loaded for local dev). Every
other module imports its settings from here so there is a single source of truth.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (one level above src/) for local development.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT: Path = _PROJECT_ROOT
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_CSV: Path = DATA_DIR / "winemag.csv"
CHROMA_DIR: Path = PROJECT_ROOT / "chroma_db"

# --- Vector store ----------------------------------------------------------
COLLECTION_NAME: str = "wines"
EMBEDDING_MODEL: str =  "text-embedding-3-small-1"
METADATA_FIELDS: list[str] = [
    "country",
    "province",
    "variety",
    "winery",
    "points",
    "price",
    "title",
]

# --- Azure OpenAI (EPAM proxy) --------------------------------------------
AZURE_OPENAI_ENDPOINT: str =  "https://ai-proxy.lab.epam.com"
AZURE_OPENAI_API_VERSION: str = "AZURE_OPENAI_API_VERSION", "2024-10-21"
LLM_MODEL: str = "gpt-4o"
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")


def _make_azure_client(max_retries: int | None = None):
    """Build an ``AzureOpenAI`` client pointed at the EPAM proxy.

    Shared by the chat and embedding client factories so every Azure OpenAI client is
    created in one place. Imported lazily so modules that don't need it (indexer,
    downloader) avoid importing the openai SDK.
    """
    from openai import AzureOpenAI

    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it as an environment variable "
            "(see .env.example)."
        )
    kwargs = dict(
        api_key=OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
    )
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    return AzureOpenAI(**kwargs)


def get_llm_client():
    """Azure OpenAI client for chat completions (used by ``agent.py``)."""
    return _make_azure_client()


def get_embedding_client(max_retries: int = 5):
    """Azure OpenAI client for embeddings (used by ``embeddings.py``)."""
    return _make_azure_client(max_retries=max_retries)


def get_embedding_function():
    """Chroma embedding function backed by the Azure OpenAI embedding deployment.

    Used by both ``indexer.py`` (build time) and ``rag.py`` (query time) so index and
    query embeddings always come from the same model. Imported lazily so modules that
    don't touch the vector store avoid importing the openai SDK.
    """
    from embeddings import AzureOpenAIEmbeddingFunction

    return AzureOpenAIEmbeddingFunction()
