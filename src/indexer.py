"""``build-index`` entry point: embed wine descriptions into a persistent ChromaDB collection."""
from __future__ import annotations

import argparse
import logging
import math
from typing import Any

import chromadb
import pandas as pd

import config
from download import download_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("indexer")

BATCH_SIZE = 1000


def _build_metadata(row: pd.Series, row_id: int) -> dict[str, Any]:
    """Metadata for one embedded row. Chroma rejects None, so NaN/missing becomes ''."""
    meta: dict[str, Any] = {"row_id": int(row_id)}
    for field in config.METADATA_FIELDS:
        value = row.get(field)
        is_nan = isinstance(value, float) and math.isnan(value)
        meta[field] = "" if value is None or is_nan else value
    return meta


def build_index(sample: int, collection_name: str, refresh_data: bool = False) -> None:
    # Ensure the dataset is present — download it on first run so the whole
    # setup is a single command. Re-indexing reuses the cached CSV.
    download_dataset(force=refresh_data)

    logger.info("Reading %s ...", config.DATA_CSV)
    df = pd.read_csv(config.DATA_CSV)
    logger.info("Loaded %d rows", len(df))

    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=42)
        logger.info("Sampled down to %d rows", len(df))

    # Drop rows without a description — nothing to embed.
    df = df[df["description"].notna()]

    embedding_fn = config.get_embedding_function()
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    # Start clean so re-running build-index doesn't duplicate documents.
    try:
        client.delete_collection(collection_name)
        logger.info("Deleted existing collection %r", collection_name)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=embedding_fn
    )

    total = len(df)
    logger.info("Embedding %d descriptions into collection %r ...", total, collection_name)

    for start in range(0, total, BATCH_SIZE):
        chunk = df.iloc[start : start + BATCH_SIZE]
        ids = [str(idx) for idx in chunk.index]
        documents = chunk["description"].astype(str).tolist()
        metadatas = [_build_metadata(row, idx) for idx, row in chunk.iterrows()]
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        logger.info("Indexed %d / %d", min(start + BATCH_SIZE, total), total)

    logger.info("Done. Collection %r now has %d documents.", collection_name, collection.count())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the dataset (if needed) and build the ChromaDB wine index."
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=15000,
        help="Number of rows to sample/embed (default: 15000; 0 = all rows).",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=config.COLLECTION_NAME,
        help=f"Chroma collection name (default: {config.COLLECTION_NAME}).",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Force a fresh download of the dataset even if data/winemag.csv exists.",
    )
    args = parser.parse_args()
    build_index(
        sample=args.sample,
        collection_name=args.collection,
        refresh_data=args.refresh_data,
    )


if __name__ == "__main__":
    main()
