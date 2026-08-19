#!/usr/bin/env python3
"""
Filter MURAVES run data by flag (VES or FS).

For the chosen flag, reads all existing MURAVES_AnalyzedData_run<N>.jsonl
files in the relevant run-number range(s), applies a chi-square quality
cut, and writes out only the selected keys (Theta_3p, Theta_4p, Phi_3p,
Phi_4p) to a single output JSON file.

Usage:
    python filter_runs.py FS
    python filter_runs.py VES
    python filter_runs.py VES --threshold 3 --data-dir /path/to/NERO/v1 --output out.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd

# Run-number ranges per flag. Each flag maps to a list of (start, end) tuples,
# both inclusive.
RUN_RANGES = {
    "VES": [(2500, 5600), (7894, 11990)],
    "FS": [(5636, 7893)],
}

KEYS_TO_KEEP = ["Theta_3p", "Theta_4p", "Phi_3p", "Phi_4p"]

DEFAULT_DATA_DIR = (
    "/Users/dorageeraerts/Documents/PhD/software/muraves/RECO_data/NERO/v1"
)
FILENAME_TEMPLATE = "MURAVES_AnalyzedData_run{run}.jsonl"


def iter_run_numbers(flag: str):
    """Yield every run number in the range(s) associated with `flag`."""
    for start, end in RUN_RANGES[flag]:
        yield from range(start, end + 1)


def load_and_filter(
    data_dir: Path, flag: str, threshold: float, verbose: bool = True
):
    """Read all available jsonl files for `flag`, apply the chi2 cut,
    and return a single concatenated, filtered DataFrame."""
    frames = []
    n_found = 0
    n_missing = 0

    for run in iter_run_numbers(flag):
        file_path = data_dir / FILENAME_TEMPLATE.format(run=run)
        if not file_path.is_file():
            n_missing += 1
            continue

        n_found += 1
        try:
            df = pd.read_json(file_path, lines=True)
        except ValueError as exc:
            print(f"  [warn] could not parse {file_path.name}: {exc}")
            continue

        if df.empty:
            continue

        required_cols = {
            "BestTrack_4p_ChiSquare_xy",
            "BestTrack_4p_ChiSquare_xz",
        } | set(KEYS_TO_KEEP)
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            print(f"  [warn] {file_path.name} missing columns {missing_cols}, skipping")
            continue

        df_selected = df[
            (df["BestTrack_4p_ChiSquare_xy"] >= 0)
            & (df["BestTrack_4p_ChiSquare_xy"] < threshold)
            & (df["BestTrack_4p_ChiSquare_xz"] >= 0)
            & (df["BestTrack_4p_ChiSquare_xz"] < threshold)
        ]

        if not df_selected.empty:
            frames.append(df_selected[KEYS_TO_KEEP])

    if verbose:
        print(f"Runs found on disk: {n_found}")
        print(f"Runs missing (skipped): {n_missing}")

    if not frames:
        return pd.DataFrame(columns=KEYS_TO_KEEP)

    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description="Filter MURAVES run data by flag (VES or FS)."
    )
    parser.add_argument(
        "flag",
        choices=["FS", "VES"],
        help="Which dataset to process: FS (free sky) or VES (vesuvius).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2,
        help="Chi-square cut threshold (default: 2).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the MURAVES_AnalyzedData_run<N>.jsonl files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: <flag>_filtered.json in the data dir).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = (
        Path(args.output)
        if args.output
        else data_dir / f"{args.flag}_filtered.json"
    )

    # If the given/derived output path is an existing directory, write the
    # default filename inside it instead of failing.
    if output_path.is_dir():
        output_path = output_path / f"{args.flag}_filtered.json"

    # Make sure the parent directory exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Flag: {args.flag}")
    print(f"Data directory: {data_dir}")
    print(f"Chi-square threshold: {args.threshold}")

    result_df = load_and_filter(data_dir, args.flag, args.threshold)

    print(f"Total selected entries: {len(result_df)}")

    # Save as a JSON array of records, keeping only the requested keys.
    records = result_df.to_dict(orient="records")
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()