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
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

# Metadata fields stored alongside each embedded description so RAG hits can be
# cross-referenced with the Pandas DataFrame.
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
AZURE_OPENAI_ENDPOINT: str = os.getenv(
    "AZURE_OPENAI_ENDPOINT", "https://ai-proxy.lab.epam.com"
)
AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o")
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")


def get_llm_client():
    """Build an ``AzureOpenAI`` client pointed at the EPAM proxy.

    Imported lazily so modules that don't need the LLM (indexer, downloader) avoid
    importing the openai SDK.
    """
    from openai import AzureOpenAI

    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it as an environment variable "
            "(see .env.example)."
        )
    return AzureOpenAI(
        api_key=OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
    )
