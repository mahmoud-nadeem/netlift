from pathlib import Path
from urllib.request import urlopen
from hashlib import sha256
import gzip
from datetime import date

BASE_DIR = Path(__file__).resolve().parents[3]
RAW_DIR = BASE_DIR / "data" / "raw"

DATASETS = {
    "criteo": {
        "url": "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz",
        "filename": "criteo-research-uplift-v2.1.csv.gz",
        "compressed": True,
    },
    "hillstrom": {
        "url": "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv",
        "filename": "hillstrom.csv",
        "compressed": False,
    },
}


def download_file(url, destination):
    try:
        with urlopen(url) as response, open(destination, "wb") as file:
            while chunk := response.read(1024 * 1024):
                file.write(chunk)
    except Exception as exc:
        if destination.exists():
            destination.unlink()
        raise RuntimeError(f"Download failed for {url}: {exc}") from exc


def sha256_file(path):
    digest = sha256()
    with open(path, "rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def row_count(path, compressed):
    opener = gzip.open if compressed else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as file:
        return sum(1 for _ in file) - 1


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for name, dataset in DATASETS.items():
        destination = RAW_DIR / dataset["filename"]

        if destination.exists():
            print(f"{name}: file already exists, skipping download.")
        else:
            print(f"{name}: downloading...")
            download_file(dataset["url"], destination)
            print(f"{name}: download complete.")

        rows = row_count(destination, dataset["compressed"])
        checksum = sha256_file(destination)
        size = destination.stat().st_size

        print(f"{name}:")
        print(f"  file: {destination}")
        print(f"  size: {size} bytes")
        print(f"  rows: {rows:,}")
        print(f"  sha256: {checksum}")
        print(f"  download date: {date.today()}")


if __name__ == "__main__":
    main()