"""``download-data`` entry point: pull the Wine Reviews dataset from Kaggle to disk."""
from __future__ import annotations

import logging

import kagglehub
from kagglehub import KaggleDatasetAdapter

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("download")

DATASET = "zynicide/wine-reviews"
SOURCE_FILE = "winemag-data-130k-v2.csv"


def download_dataset(force: bool = False) -> None:
    """Download the dataset to ``data/winemag.csv`` (skipped if it already exists).

    Reusable from ``build-index`` so first-time setup is a single command.
    Pass ``force=True`` to re-download even when the file is present.
    """
    if config.DATA_CSV.exists() and not force:
        logger.info("%s already exists — skipping download.", config.DATA_CSV)
        return

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Loading %s (%s) from Kaggle...", DATASET, SOURCE_FILE)
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        DATASET,
        SOURCE_FILE,
    )
    df.to_csv(config.DATA_CSV, index=False)
    logger.info("Saved %d rows to %s", len(df), config.DATA_CSV)


def main() -> None:
    """``download-data`` entry point — force a fresh download of the dataset."""
    download_dataset(force=True)


if __name__ == "__main__":
    main()
