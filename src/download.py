"""``download-data`` entry point: pull the Wine Reviews dataset from Kaggle to disk."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import kagglehub

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("download")

DATASET = "zynicide/wine-reviews"
SOURCE_FILE = "winemag-data-130k-v2.csv"


def download_dataset(force: bool = False) -> None:
    """Download the dataset to ``data/winemag.csv`` (skipped if it already exists).

    Reusable from ``build-index`` so first-time setup is a single command.
    Pass ``force=True`` to re-download even when the file is present.

    We fetch the raw dataset files with ``dataset_download`` and copy the CSV
    ourselves, rather than using kagglehub's PANDAS adapter (which mis-parses this
    particular file). The CSV is standard UTF-8, so downstream reads are unambiguous.
    """
    if config.DATA_CSV.exists() and not force:
        logger.info("%s already exists — skipping download.", config.DATA_CSV)
        return

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s from Kaggle...", DATASET)
    dataset_dir = Path(kagglehub.dataset_download(DATASET, force_download=force))

    source = dataset_dir / SOURCE_FILE
    if not source.exists():
        raise FileNotFoundError(f"{SOURCE_FILE} not found in downloaded dataset {dataset_dir}")

    shutil.copyfile(source, config.DATA_CSV)
    logger.info("Saved %s", config.DATA_CSV)


def main() -> None:
    """``download-data`` entry point — force a fresh download of the dataset."""
    download_dataset(force=True)


if __name__ == "__main__":
    main()
