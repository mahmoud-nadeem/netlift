"""One-off: convert the Criteo CSV.gz to a typed Parquet file (risk R08)."""
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SRC = Path("data/raw/criteo-research-uplift-v2.1.csv.gz")
DST = Path("data/interim/criteo.parquet")
FEATURES = [f"f{i}" for i in range(12)]
DTYPES = {
    **{c: "float32" for c in FEATURES},
    "treatment": "int8",
    "conversion": "int8",
    "visit": "int8",
    "exposure": "int8",
}


def main() -> None:
    DST.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    rows = 0
    for chunk in pd.read_csv(SRC, dtype=DTYPES, chunksize=1_000_000):
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(DST, table.schema, compression="zstd")
        writer.write_table(table)
        rows += len(chunk)
        print(f"{rows:,} rows", flush=True)
    writer.close()
    print(f"done: {rows:,} rows -> {DST}")


if __name__ == "__main__":
    main()
