#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Count chi2-selected tracks in a parquet file."
    )

    parser.add_argument(
        "input_file",
        nargs="?",
        default="SelectedTracks_FS_chi22_ROSSO_test.parquet",
        help="Input parquet file"
    )

    parser.add_argument(
        "--output",
        default="track_count_chi2.txt",
        help="Output text file"
    )

    args = parser.parse_args()

    input_file = Path(args.input_file)
    output_file = Path(args.output)

    # Check that the file exists
    if not input_file.exists():
        print(f"ERROR: File not found: {input_file}")
        return

    print(f"Reading: {input_file}")

    try:
        df = pd.read_parquet(input_file)
    except Exception as e:
        print(f"ERROR while reading parquet file:")
        print(e)
        return

    # Number of selected tracks = number of rows
    n_tracks = len(df)

    print(f"Number of chi2-selected tracks: {n_tracks}")

    # Write result to output file
    with open(output_file, "w") as f:
        f.write(f"Input file: {input_file}\n")
        f.write(f"Chi2-selected tracks: {n_tracks}\n")

    print(f"Result written to: {output_file}")


if __name__ == "__main__":
    main()