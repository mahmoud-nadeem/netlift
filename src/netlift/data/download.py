"""Download the NetLift raw datasets and verify them against recorded checksums.

Fetches the Criteo uplift and Hillstrom datasets into ``data/raw/``. A file that
already exists is only skipped if its SHA-256 matches the value recorded in
``docs/data-sources.md``; otherwise it is downloaded again. After each dataset
the script logs its path, size, row count, checksum and download date.

Run from the repository root:

    python src/netlift/data/download.py
"""

import gzip
import logging
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import TextIO, TypedDict
from urllib.request import urlopen

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[3]
RAW_DIR = BASE_DIR / "data" / "raw"


class Dataset(TypedDict):
    """Static description of one downloadable dataset."""

    url: str
    filename: str
    sha256: str
    compressed: bool


# Expected checksums are recorded in docs/data-sources.md.
DATASETS: dict[str, Dataset] = {
    "criteo": {
        "url": "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz",
        "filename": "criteo-research-uplift-v2.1.csv.gz",
        "sha256": "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc",
        "compressed": True,
    },
    "hillstrom": {
        "url": "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv",
        "filename": "hillstrom.csv",
        "sha256": "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece",
        "compressed": False,
    },
}


def download_file(url: str, destination: Path) -> None:
    """Stream a URL to a local file, removing the partial file on failure."""
    try:
        with urlopen(url) as response, open(destination, "wb") as file:
            while chunk := response.read(1024 * 1024):
                file.write(chunk)
    except Exception as exc:
        if destination.exists():
            destination.unlink()
        raise RuntimeError(f"Download failed for {url}: {exc}") from exc


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file, read in 1 MiB chunks."""
    digest = sha256()
    with open(path, "rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def open_text(path: Path, compressed: bool) -> TextIO:
    """Open a CSV file for text reading, handling gzip transparently."""
    if compressed:
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def row_count(path: Path, compressed: bool) -> int:
    """Count the data rows in a CSV file, excluding the header line."""
    with open_text(path, compressed) as file:
        return sum(1 for _ in file) - 1


def ensure_file(name: str, dataset: Dataset) -> tuple[Path, str]:
    """Return the file path and SHA-256, downloading unless a verified copy exists."""
    destination = RAW_DIR / dataset["filename"]
    expected = dataset["sha256"]

    if destination.exists():
        checksum = sha256_file(destination)
        if checksum == expected:
            logger.info("%s: file exists and checksum matches, skipping download", name)
            return destination, checksum
        logger.warning("%s: file exists but checksum differs, re-downloading", name)
    else:
        logger.info("%s: downloading...", name)

    download_file(dataset["url"], destination)
    checksum = sha256_file(destination)
    if checksum != expected:
        raise RuntimeError(f"{name}: expected sha256 {expected}, got {checksum}")
    logger.info("%s: download complete and verified", name)
    return destination, checksum


def main() -> None:
    """Download and verify every dataset, then log its size, rows and checksum."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for name, dataset in DATASETS.items():
        destination, checksum = ensure_file(name, dataset)
        rows = row_count(destination, dataset["compressed"])
        size = destination.stat().st_size

        logger.info("%s:", name)
        logger.info("  file: %s", destination)
        logger.info("  size: %d bytes", size)
        logger.info("  rows: %s", f"{rows:,}")
        logger.info("  sha256: %s", checksum)
        logger.info("  download date: %s", datetime.now(tz=UTC).date())


if __name__ == "__main__":
    main()
