"""One-off: convert the Criteo CSV.gz to a typed Parquet file (risk R08).

The data is written to a temporary file first. The file is renamed to its final
name only after the row count matches ADR 0003, so a stopped run or an
incomplete source never leaves a short criteo.parquet behind.
"""

import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

SRC = Path("data/raw/criteo-research-uplift-v2.1.csv.gz")
DST = Path("data/interim/criteo.parquet")
TMP = DST.with_name(DST.name + ".tmp")
EXPECTED_ROWS = 13_979_592  # ADR 0003
FEATURES = [f"f{i}" for i in range(12)]
DTYPES = {
    **{c: "float32" for c in FEATURES},
    "treatment": "int8",
    "conversion": "int8",
    "visit": "int8",
    "exposure": "int8",
}


def main() -> None:
    """Read the CSV.gz in chunks and write one typed Parquet file."""
    DST.parent.mkdir(parents=True, exist_ok=True)
    TMP.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        for chunk in pd.read_csv(SRC, dtype=DTYPES, chunksize=1_000_000):
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(TMP, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(chunk)
            logger.info("%s rows", f"{rows:,}")
        if writer is None:
            raise SystemExit(f"{SRC} contained no rows")
        writer.close()
        writer = None
        on_disk = pq.ParquetFile(TMP).metadata.num_rows
        if rows != EXPECTED_ROWS or on_disk != EXPECTED_ROWS:
            raise SystemExit(
                f"row count mismatch: read {rows:,}, file has {on_disk:,}, "
                f"expected {EXPECTED_ROWS:,}. {DST} was not changed."
            )
        TMP.replace(DST)
    except BaseException:  # also Ctrl+C
        if writer is not None:
            writer.close()
        TMP.unlink(missing_ok=True)
        raise
    logger.info("done: %s rows -> %s", f"{rows:,}", DST)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
