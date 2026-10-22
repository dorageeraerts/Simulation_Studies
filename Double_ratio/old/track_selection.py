#!/usr/bin/env python3
"""
Filter MURAVES run data by flag (VES or FS).

You can explicitly select runs using:
    3
    2-5
    2,5,8
    2-5,10,20-22
If no runs are specified, all runs in the predefined RUN_RANGES
for the selected detector and flag are processed.

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

# Run-number ranges per detector and flag.
# Each range is given as an inclusive (start, end) tuple.
RUN_RANGES = {
    "NERO": {
        "VES": [(2500, 5600), (7894, 11990)],
        "FS": [(5636, 7893)],
    },
    "ROSSO": {
        "VES": [(4000, 10060), (13458, 14792)],
        "FS": [(10061, 13457)],
    },
    "BLU": {
        "VES": [(2800, 8818)],
        "FS": [(8819, 9925)],
    },
}

KEYS_TO_KEEP = ["Theta_3p", "Theta_4p", "Phi_3p", "Phi_4p"]

DEFAULT_DATA_DIR = (
    "/Users/dorageeraerts/Documents/PhD/software/muraves/RECO_data/NERO/v1"
)
FILENAME_TEMPLATE = "MURAVES_AnalyzedData_run{run}.jsonl"

def parse_runs(run_string):
    """
    Parse a run specification such as:

        3
        2-5
        2,5,8
        2-5,10,20-22

    and return a sorted list of individual run numbers.
    """
    runs = set()

    for part in run_string.split(","):
        part = part.strip()

        if not part:
            continue

        if "-" in part:
            pieces = part.split("-")

            if len(pieces) != 2:
                raise ValueError(
                    f"Invalid run range '{part}'. Use e.g. 2-5.")

            start = int(pieces[0])
            end = int(pieces[1])

            if start > end:
                raise ValueError(
                    f"Invalid run range '{part}': start must be <= end.")
            runs.update(range(start, end + 1))

        else:
            runs.add(int(part))

    if not runs:
        raise ValueError("No valid run numbers were provided.")

    return sorted(runs)


def iter_run_numbers(detector: str, flag: str):
    """Yield every run number in the range(s) associated with `flag`."""
    for start, end in RUN_RANGES[detector][flag]:
        yield from range(start, end + 1)


def load_and_filter(
    data_dir: Path, detector: str, flag: str, threshold: float, verbose: bool = True, runs=None
):
    """Read all available jsonl files for `flag`, apply the chi2 cut,
    and return a single concatenated, filtered DataFrame."""
    frames = []
    n_found = 0
    n_missing = 0

    if runs is None:
        runs_to_process = iter_run_numbers(detector, flag)
    else:
        runs_to_process = runs

    for run in runs_to_process:
        file_path = data_dir / FILENAME_TEMPLATE.format(run=run)
        if not file_path.is_file():
            print(f"Run {run} not available, skipping")
            n_missing += 1
            continue

        n_found += 1
        print(f"Processing run {run}: {file_path.name}")
        try:
            df = pd.read_json(file_path, lines=True)
        except ValueError as exc:
            print(f"  [warn] could not parse {file_path.name}: {exc}")
            continue

        if df.empty:
            print(f"  [warn] {file_path.name} is empty, skipping")
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

        print(
            f"  Selected {len(df_selected)} / {len(df)} entries"
        )

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
            "detector",
            choices=["ROSSO", "BLU", "NERO"],
            help="Which detector data to process: BLU, NERO or ROSSO.",
        )
    parser.add_argument(
        "runs",
        nargs="?",
        default=None,
        help=(
            "Runs to process, e.g. 3, 2-5, or "
            "2-5,10,20-22. If omitted, process all "
            "runs in the predefined range."
        ),
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
        help="Output JSON file path (default: <detector>_<flag>_filtered.json in the data dir).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    runs = None
    if args.runs is not None:
        try:
            runs = parse_runs(args.runs)

        except ValueError as exc:
            parser.error(str(exc))


    '''output_path = (
        Path(args.output)
        if args.output
        else data_dir / f"{args.detector}_{args.flag}_filtered.json"
    )

    # If the given/derived output path is an existing directory, write the
    # default filename inside it instead of failing.
    if output_path.is_dir():
        output_path = output_path / f"{args.detector}_{args.flag}_filtered.json"'''

    if args.output:
        output_path = Path(args.output)
    else:
        if args.runs is not None:
            output_filename = (
                f"{args.detector}_{args.flag}_runs{args.runs}_filtered.json"
            )
        else:
            output_filename = (
                f"{args.detector}_{args.flag}_filtered.json"
            )

        output_path = data_dir / output_filename

    # If the given output path is an existing directory,
    # write the automatically generated filename inside it.
    if output_path.is_dir():
        if args.runs is not None:
            output_filename = (
                f"{args.detector}_{args.flag}_runs{args.runs}_filtered_chi2-{args.threshold}.json"
            )
        else:
            output_filename = (
                f"{args.detector}_{args.flag}_filtered_chi2-{args.threshold}.json"
            )

        output_path = output_path / output_filename

    # Make sure the parent directory exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Flag: {args.flag}")
    print(f"Detector: {args.detector}")
    print(f"Data directory: {data_dir}")
    print(f"Chi-square threshold: {args.threshold}")
    if runs is not None:
        print(f"Requested runs: {runs}")
    else:
        print("Requested runs: all predefined runs")

    result_df = load_and_filter(data_dir, args.detector, args.flag, args.threshold, runs=runs)

    print(f"Total selected entries: {len(result_df)}")

    # Save as a JSON array of records, keeping only the requested keys.
    records = result_df.to_dict(orient="records")
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()