"""Download the NetLift raw datasets and verify their recorded checksums.

Fetches the Criteo uplift and Hillstrom datasets into ``data/raw/``.
Existing files are skipped only when their SHA-256 matches the expected
checksum. New downloads are verified before being moved to their final path.

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
    """Stream a URL to a file and remove it if downloading fails."""
    try:
        with urlopen(url, timeout=60) as response, destination.open("wb") as file:
            while chunk := response.read(1024 * 1024):
                file.write(chunk)
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed for {url}: {exc}") from exc
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file, read in 1 MiB chunks."""
    digest = sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def open_text(path: Path, compressed: bool) -> TextIO:
    """Open a CSV file for text reading, handling gzip transparently."""
    if compressed:
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def row_count(path: Path, compressed: bool) -> int:
    """Count the data rows in a CSV file, excluding the header line."""
    with open_text(path, compressed) as file:
        return sum(1 for _ in file) - 1


def ensure_file(name: str, dataset: Dataset) -> tuple[Path, str, bool]:
    """Return a verified file, downloading it if necessary."""
    destination = RAW_DIR / dataset["filename"]
    partial = destination.with_name(destination.name + ".part")
    expected = dataset["sha256"]

    if destination.exists():
        checksum = sha256_file(destination)
        if checksum == expected:
            logger.info("%s: file exists and checksum matches, skipping download", name)
            return destination, checksum, True

        logger.warning("%s: file exists but checksum differs, re-downloading", name)
        destination.unlink(missing_ok=True)
    else:
        logger.info("%s: downloading...", name)

    try:
        download_file(dataset["url"], partial)
        checksum = sha256_file(partial)

        if checksum != expected:
            raise RuntimeError(f"{name}: expected sha256 {expected}, got {checksum}")

        partial.replace(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    logger.info("%s: download complete and verified", name)
    return destination, checksum, False


def main() -> None:
    """Download and verify datasets, then log size, rows and checksum."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for name, dataset in DATASETS.items():
        destination, checksum, skipped = ensure_file(name, dataset)
        rows = row_count(destination, dataset["compressed"])
        size = destination.stat().st_size

        logger.info("%s:", name)
        logger.info("  file: %s", destination)
        logger.info("  size: %d bytes", size)
        logger.info("  rows: %s", f"{rows:,}")
        logger.info("  sha256: %s", checksum)

        date_label = "verified on" if skipped else "download date"
        logger.info("  %s: %s", date_label, datetime.now(tz=UTC).date())


if __name__ == "__main__":
    main()
