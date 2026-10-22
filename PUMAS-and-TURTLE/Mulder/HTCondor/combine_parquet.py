#!/usr/bin/env python3
"""Concatenate per-job muon-state parquet files without loading them all at once."""

import argparse
import os

import pyarrow.parquet as pq


parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("files", nargs="+")
args = parser.parse_args()

output_tmp = f"{args.output}.tmp"
first_schema = None

try:
    for filename in args.files:
        input_path = os.path.join(args.input_dir, filename)
        parquet_file = pq.ParquetFile(input_path)
        if first_schema is None:
            first_schema = parquet_file.schema_arrow
            break

    if first_schema is None:
        raise ValueError("no parquet files were supplied")

    with pq.ParquetWriter(output_tmp, first_schema, compression="zstd") as writer:
        for filename in args.files:
            input_path = os.path.join(args.input_dir, filename)
            parquet_file = pq.ParquetFile(input_path)
            if parquet_file.schema_arrow != first_schema:
                raise ValueError(f"schema mismatch in {input_path}")

            for batch in parquet_file.iter_batches():
                writer.write_batch(batch)

    os.replace(output_tmp, args.output)
except Exception:
    if os.path.exists(output_tmp):
        os.unlink(output_tmp)
    raise