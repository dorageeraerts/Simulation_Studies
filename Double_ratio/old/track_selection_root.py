#!/usr/bin/env python3
"""
Filter MURAVES run data by flag (VES or FS) and track selection based on chi-square criteria.

Usage:
    python track_selection_root.py FS NERO
    python track_selection_root.py VES ROSSO --threshold 3
    python track_selection_root.py VES BLU 2-5,10 --output out.parquet
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import uproot

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
CHI2_XY = "BestTrack_4p_ChiSquare_xy"
CHI2_XZ = "BestTrack_4p_ChiSquare_xz"

TREE_NAME = "AnalyzedData"

DEFAULT_DATA_DIR = (
    "/Users/dorageeraerts/Documents/PhD/software/muraves/RECO_data/NERO/v1"
)
FILENAME_TEMPLATE = "MURAVES_AnalyzedData_run{run}.root"


def parse_runs(run_string):
    runs = set()
    for part in run_string.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            pieces = part.split("-")
            if len(pieces) != 2:
                raise ValueError(f"Invalid run range '{part}'. Use e.g. 2-5.")
            start, end = int(pieces[0]), int(pieces[1])
            if start > end:
                raise ValueError(f"Invalid run range '{part}': start must be <= end.")
            runs.update(range(start, end + 1))
        else:
            runs.add(int(part))
    if not runs:
        raise ValueError("No valid run numbers were provided.")
    return sorted(runs)


def iter_run_numbers(detector: str, flag: str):
    for start, end in RUN_RANGES[detector][flag]:
        yield from range(start, end + 1)


def load_and_filter(
    data_dir: Path,
    detector: str,
    flag: str,
    threshold: float,
    tree_name: str,
    verbose: bool = True,
    runs=None,
):
    """Read only the needed branches for each available run.root file,
    apply the chi2 cut with plain numpy boolean masking, and return a
    single concatenated DataFrame with just KEYS_TO_KEEP."""
    branches = KEYS_TO_KEEP + [CHI2_XY, CHI2_XZ]

    frames = []
    n_found = 0
    n_missing = 0

    runs_to_process = iter_run_numbers(detector, flag) if runs is None else runs

    for run in runs_to_process:
        file_path = data_dir / FILENAME_TEMPLATE.format(run=run)
        if not file_path.is_file():
            print(f"Run {run} not available, skipping")
            n_missing += 1
            continue

        n_found += 1
        print(f"Processing run {run}: {file_path.name}")

        try:
            with uproot.open(file_path) as f:
                if tree_name not in f:
                    print(f"  [warn] tree '{tree_name}' not found in {file_path.name}, skipping")
                    continue
                tree = f[tree_name]

                missing = set(branches) - set(tree.keys())
                if missing:
                    print(f"  [warn] {file_path.name} missing branches {missing}, skipping")
                    continue

                arrays = tree.arrays(branches, library="np")
        except Exception as exc:  # corrupt file, wrong format, etc.
            print(f"  [warn] could not read {file_path.name}: {exc}")
            continue

        n_total = len(arrays[CHI2_XY])
        if n_total == 0:
            print(f"  [warn] {file_path.name} is empty, skipping")
            continue

        chi2_xy = arrays[CHI2_XY]
        chi2_xz = arrays[CHI2_XZ]
        mask = (chi2_xy >= 0) & (chi2_xy < threshold) & (chi2_xz >= 0) & (chi2_xz < threshold)

        n_selected = int(mask.sum())
        print(f"  Selected {n_selected} / {n_total} entries")

        if n_selected:
            frames.append(pd.DataFrame({k: arrays[k][mask] for k in KEYS_TO_KEEP}))

    if verbose:
        print(f"Runs found on disk: {n_found}")
        print(f"Runs missing (skipped): {n_missing}")

    if not frames:
        return pd.DataFrame(columns=KEYS_TO_KEEP)

    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description="Filter MURAVES run data by flag (VES or FS), reading ROOT files."
    )
    parser.add_argument("flag", choices=["FS", "VES"])
    parser.add_argument("detector", choices=["ROSSO", "BLU", "NERO"])
    parser.add_argument("runs", nargs="?", default=None,
                         help="e.g. 3, 2-5, or 2-5,10,20-22. Omit for all predefined runs.")
    parser.add_argument("--threshold", type=float, default=2,
                         help="Chi-square cut threshold (default: 2).")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--tree-name", type=str, default=TREE_NAME,
                         help=f"Name of the TTree inside each ROOT file (default: {TREE_NAME}).")
    parser.add_argument("--output-dir", type=str, default=None,
                         help="Directory to write the output file into (default: --data-dir). "
                              "The filename is always auto-generated as "
                              "<detector>_<flag>[_runs<runs>]_filtered_chi2-<threshold>.parquet")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir

    runs = None
    if args.runs is not None:
        try:
            runs = parse_runs(args.runs)
        except ValueError as exc:
            parser.error(str(exc))

    runs_suffix = f"_runs{args.runs}" if args.runs is not None else ""
    output_path = output_dir / f"{args.detector}_{args.flag}{runs_suffix}_filtered_chi2-{args.threshold}.parquet"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Flag: {args.flag}")
    print(f"Detector: {args.detector}")
    print(f"Data directory: {data_dir}")
    print(f"Tree name: {args.tree_name}")
    print(f"Chi-square threshold: {args.threshold}")
    print(f"Requested runs: {runs if runs is not None else 'all predefined runs'}")

    result_df = load_and_filter(
        data_dir, args.detector, args.flag, args.threshold, args.tree_name, runs=runs
    )

    print(f"Total selected entries: {len(result_df)}")

    result_df.to_parquet(output_path, index=False)

    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()