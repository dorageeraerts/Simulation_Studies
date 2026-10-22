#!/usr/bin/env python3

"""
Count selected MURAVES tracks per Working Point and total run duration.

For every reconstructed ROOT file:

    MURAVES_AnalyzedData_run<N>.root

the corresponding slow-control file is:

    SLOWCONTROL_run<N>

The Working Point is taken from field [5] of the slow-control file
(zero-based Python indexing).

A ROOT entry is counted as a selected track when BOTH:

    BestTrack_4p_ChiSquare_xz > 0
    BestTrack_4p_ChiSquare_xy > 0

The run duration is calculated from the ROOT "timestamp" branch:

    duration = (max(timestamp) - min(timestamp)) / 1000

because timestamp is in milliseconds.

The output is separated by:

    detector  : ROSSO / BLU / NERO
    data type : VES / FS
    Working Point

For each Working Point the output contains:

    SelectedTracks
    TotalEntries
    Runs
    RunDurationSeconds
    RunDurationMinutes
    RunDurationHours

Two output files are produced:

    1. counts_per_working_point.csv

       Selected track counts and total run duration per
       detector / VES-FS / Working Point.

    2. run_processing_summary.txt

       Number of ROOT files found/missing, slow-control files missing,
       successfully processed runs, and the actual run numbers in each
       category.

Consecutive run numbers are compressed into ranges.

Usage:

    python count_tracks_per_wp.py

    python count_tracks_per_wp.py --detectors ROSSO

    python count_tracks_per_wp.py --detectors ROSSO BLU

    python count_tracks_per_wp.py \
        --output counts_per_working_point.csv \
        --summary-output run_processing_summary.txt
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import uproot


# ======================================================================
# Configuration
# ======================================================================

DETECTORS = ["ROSSO", "BLU", "NERO"]

TREE_NAME = "AnalyzedData"

CHI2_XY = "BestTrack_4p_ChiSquare_xy"
CHI2_XZ = "BestTrack_4p_ChiSquare_xz"

TIMESTAMP_BRANCH = "timestamp"

ROOT_FILENAME_TEMPLATE = "MURAVES_AnalyzedData_run{run}.root"
SLOWCONTROL_FILENAME_TEMPLATE = "SLOWCONTROL_run{run}"


# Same run ranges as in your existing analysis script.
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


# ======================================================================
# Run-number formatting
# ======================================================================

def compress_runs(runs):
    """
    Convert a list/set of run numbers into a compact string.

    Example:
        [1, 2, 3, 4, 5, 8, 10, 11, 12]

    becomes:
        "1-5, 8, 10-12"
    """

    if not runs:
        return "None"

    runs = sorted(set(runs))

    ranges = []

    start = runs[0]
    previous = runs[0]

    for run in runs[1:]:

        if run == previous + 1:
            previous = run
            continue

        if start == previous:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{previous}")

        start = run
        previous = run

    # Add final range.
    if start == previous:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{previous}")

    return ", ".join(ranges)


# ======================================================================
# Run handling
# ======================================================================

def iter_run_numbers(detector, data_type):
    """Yield all predefined run numbers for a detector and VES/FS."""

    for start, end in RUN_RANGES[detector][data_type]:
        yield from range(start, end + 1)


# ======================================================================
# Working Point
# ======================================================================

def get_working_point(slowcontrol_file):
    """
    Read the Working Point from a SLOWCONTROL file.

    The Working Point is field [5], using zero-based Python indexing.

    Example:

        11267  1620208547823  0101100  19.2  97.4  25 ...

    gives:

        Working Point = 25

    The first field is deliberately ignored.
    """

    try:

        with open(slowcontrol_file, "r") as f:
            line = f.readline().strip()

        if not line:
            print(
                f"  [warn] empty slow-control file: "
                f"{slowcontrol_file}"
            )
            return None

        fields = line.split()

        if len(fields) <= 5:
            print(
                f"  [warn] slow-control file has only "
                f"{len(fields)} fields: {slowcontrol_file}"
            )
            return None

        working_point = fields[5]

        # Convert numeric values to int where appropriate.
        try:

            wp_float = float(working_point)

            if wp_float.is_integer():
                return int(wp_float)

            return wp_float

        except ValueError:

            # If the field isn't numeric, retain it as a string.
            return working_point

    except Exception as exc:

        print(
            f"  [warn] could not read slow-control file "
            f"{slowcontrol_file}: {exc}"
        )

        return None


# ======================================================================
# ROOT processing
# ======================================================================

def process_root_file(root_file):
    """
    Read one ROOT file.

    Returns:

        (selected_count, total_entries, duration_seconds)

    or None if the ROOT file cannot be processed.

    Selection:

        BestTrack_4p_ChiSquare_xz > 0
        AND
        BestTrack_4p_ChiSquare_xy > 0

    Duration:

        (max(timestamp) - min(timestamp)) / 1000

    because timestamp is in milliseconds.
    """

    try:

        with uproot.open(root_file) as f:

            if TREE_NAME not in f:

                print(
                    f"  [warn] tree '{TREE_NAME}' not found in "
                    f"{root_file}"
                )

                return None

            tree = f[TREE_NAME]

            branches = set(tree.keys())

            required_branches = {
                CHI2_XY,
                CHI2_XZ,
                TIMESTAMP_BRANCH,
            }

            missing = required_branches - branches

            if missing:

                print(
                    f"  [warn] missing branches {missing} in "
                    f"{root_file}"
                )

                return None

            arrays = tree.arrays(
                [
                    CHI2_XY,
                    CHI2_XZ,
                    TIMESTAMP_BRANCH,
                ],
                library="np",
            )

    except Exception as exc:

        print(
            f"  [warn] could not read {root_file}: {exc}"
        )

        return None

    # --------------------------------------------------------------
    # Number of entries
    # --------------------------------------------------------------

    chi2_xy = np.asarray(arrays[CHI2_XY])
    chi2_xz = np.asarray(arrays[CHI2_XZ])

    n_total = len(chi2_xy)

    if n_total == 0:

        print(
            f"  [warn] empty tree in {root_file}"
        )

        return None

    # --------------------------------------------------------------
    # Run duration
    # --------------------------------------------------------------

    timestamps = np.asarray(
        arrays[TIMESTAMP_BRANCH],
        dtype=float,
    )

    valid_timestamps = np.isfinite(timestamps)

    if not np.any(valid_timestamps):

        print(
            f"  [warn] no valid timestamps in {root_file}"
        )

        return None

    tmin = np.min(
        timestamps[valid_timestamps]
    )

    tmax = np.max(
        timestamps[valid_timestamps]
    )

    # Timestamp is in milliseconds.
    duration_seconds = (tmax - tmin) / 1000.0

    if duration_seconds <= 0:

        print(
            f"  [warn] invalid run duration: "
            f"{duration_seconds:.3f} s in {root_file}"
        )

        return None

    # --------------------------------------------------------------
    # Track selection
    # --------------------------------------------------------------

    selected_mask = (
        np.isfinite(chi2_xy)
        & np.isfinite(chi2_xz)
        & (chi2_xy > 0)
        & (chi2_xz > 0)
    )

    n_selected = int(
        np.count_nonzero(selected_mask)
    )

    return (
        n_selected,
        n_total,
        duration_seconds,
    )


# ======================================================================
# Detector processing
# ======================================================================

def process_detector(
    detector,
    reconstructed_base,
    slowcontrol_base,
    results,
    summary_sections,
):
    """
    Process all VES and FS runs for one detector.

    Track counts and run durations are accumulated separately
    for each Working Point.
    """

    print()
    print("=" * 70)
    print(f"Detector: {detector}")
    print("=" * 70)

    reconstructed_dir = (
        reconstructed_base
        / detector
        / "v1"
    )

    slowcontrol_dir = (
        slowcontrol_base
        / detector
    )

    print(
        f"ROOT directory:         {reconstructed_dir}"
    )

    print(
        f"Slow-control directory: {slowcontrol_dir}"
    )

    for data_type in ["VES", "FS"]:

        print()
        print(f"--- {data_type} ---")

        # ----------------------------------------------------------
        # Counters
        # ----------------------------------------------------------

        n_root_found = 0
        n_root_missing = 0
        n_slowcontrol_missing = 0
        n_successful = 0

        # ----------------------------------------------------------
        # Actual run-number lists
        # ----------------------------------------------------------

        missing_root_runs = []
        missing_slowcontrol_runs = []
        unsuccessful_runs = []
        successful_runs = []

        # ----------------------------------------------------------
        # Working Point totals
        # ----------------------------------------------------------

        wp_totals = {}

        # ----------------------------------------------------------
        # Process runs
        # ----------------------------------------------------------

        for run in iter_run_numbers(
            detector,
            data_type,
        ):

            root_file = (
                reconstructed_dir
                / ROOT_FILENAME_TEMPLATE.format(
                    run=run
                )
            )

            slowcontrol_file = (
                slowcontrol_dir
                / SLOWCONTROL_FILENAME_TEMPLATE.format(
                    run=run
                )
            )

            # ======================================================
            # Check ROOT file
            # ======================================================

            if not root_file.is_file():

                n_root_missing += 1
                missing_root_runs.append(run)

                continue

            n_root_found += 1

            # ======================================================
            # Check slow-control file
            # ======================================================

            if not slowcontrol_file.is_file():

                n_slowcontrol_missing += 1
                missing_slowcontrol_runs.append(run)

                print(
                    f"Run {run}: ROOT exists but slow-control "
                    f"file is missing"
                )

                continue

            # ======================================================
            # Get Working Point
            # ======================================================

            working_point = get_working_point(
                slowcontrol_file
            )

            if working_point is None:

                unsuccessful_runs.append(run)

                print(
                    f"Run {run}: could not determine "
                    f"Working Point"
                )

                continue

            # ======================================================
            # Process ROOT file
            # ======================================================

            result = process_root_file(
                root_file
            )

            if result is None:

                unsuccessful_runs.append(run)

                continue

            (
                n_selected,
                n_total,
                duration_seconds,
            ) = result

            # ======================================================
            # Successful run
            # ======================================================

            n_successful += 1
            successful_runs.append(run)

            # ------------------------------------------------------
            # Create WP entry if necessary
            # ------------------------------------------------------

            if working_point not in wp_totals:

                wp_totals[working_point] = {
                    "selected": 0,
                    "entries": 0,
                    "runs": 0,
                    "duration_seconds": 0.0,
                }

            # ------------------------------------------------------
            # Add track counts
            # ------------------------------------------------------

            wp_totals[working_point]["selected"] += (
                n_selected
            )

            wp_totals[working_point]["entries"] += (
                n_total
            )

            wp_totals[working_point]["runs"] += 1

            # ------------------------------------------------------
            # Add run duration
            # ------------------------------------------------------

            wp_totals[working_point]["duration_seconds"] += (
                duration_seconds
            )

            print(
                f"Run {run:5d} | "
                f"{data_type:3s} | "
                f"WP {str(working_point):>5s} | "
                f"{n_selected:8d} / "
                f"{n_total:8d} selected | "
                f"Duration "
                f"{duration_seconds / 60:8.2f} min"
            )

        # ==========================================================
        # Print Working Point summary
        # ==========================================================

        print()
        print(
            f"{detector} {data_type} Working Point summary:"
        )

        for wp in sorted(
            wp_totals,
            key=lambda x: float(x),
        ):

            data = wp_totals[wp]

            duration_seconds = data[
                "duration_seconds"
            ]

            duration_minutes = (
                duration_seconds / 60.0
            )

            duration_hours = (
                duration_seconds / 3600.0
            )

            print(
                f"  WP {wp}: "
                f"{data['selected']} selected tracks "
                f"from {data['runs']} runs | "
                f"duration = "
                f"{duration_hours:.2f} h"
            )

            # ------------------------------------------------------
            # Store result
            # ------------------------------------------------------

            results.append(
                {
                    "Detector": detector,
                    "DataType": data_type,
                    "WorkingPoint": wp,
                    "SelectedTracks": data["selected"],
                    "TotalEntries": data["entries"],
                    "Runs": data["runs"],
                    "RunDurationSeconds": duration_seconds,
                    "RunDurationMinutes": duration_minutes,
                    "RunDurationHours": duration_hours,
                }
            )

        # ==========================================================
        # Console run summary
        # ==========================================================

        print()

        print(
            f"ROOT files found:       {n_root_found}"
        )

        print(
            f"ROOT files missing:     {n_root_missing}"
        )

        print(
            f"Slow-control missing:   {n_slowcontrol_missing}"
        )

        print(
            f"Runs successfully used: {n_successful}"
        )

        # ==========================================================
        # Store summary for output file
        # ==========================================================

        summary_sections.append(
            {
                "detector": detector,
                "data_type": data_type,
                "root_found": n_root_found,
                "root_missing": n_root_missing,
                "slowcontrol_missing": (
                    n_slowcontrol_missing
                ),
                "successful": n_successful,
                "missing_root_runs": (
                    missing_root_runs
                ),
                "missing_slowcontrol_runs": (
                    missing_slowcontrol_runs
                ),
                "unsuccessful_runs": (
                    unsuccessful_runs
                ),
                "successful_runs": (
                    successful_runs
                ),
            }
        )


# ======================================================================
# Write counts CSV
# ======================================================================

def write_counts_output(results, output_file):
    """Write Working Point track counts and durations to CSV."""

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "Detector",
        "DataType",
        "WorkingPoint",
        "SelectedTracks",
        "TotalEntries",
        "Runs",
        "RunDurationSeconds",
        "RunDurationMinutes",
        "RunDurationHours",
    ]

    with open(
        output_file,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print(
        f"Track-count and duration output written to:\n"
        f"  {output_file}"
    )


# ======================================================================
# Write run-processing summary
# ======================================================================

def write_summary_output(
    summary_sections,
    output_file,
):
    """
    Write the run-processing information to a separate text file.
    """

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_file,
        "w",
    ) as f:

        f.write(
            "MURAVES run processing summary\n"
        )

        f.write(
            "=" * 80 + "\n\n"
        )

        for section in summary_sections:

            detector = section["detector"]
            data_type = section["data_type"]

            f.write(
                "=" * 80 + "\n"
            )

            f.write(
                f"{detector} - {data_type}\n"
            )

            f.write(
                "=" * 80 + "\n"
            )

            f.write(
                f"ROOT files found:       "
                f"{section['root_found']}\n"
            )

            f.write(
                f"ROOT files missing:     "
                f"{section['root_missing']}\n"
            )

            f.write(
                f"Slow-control missing:   "
                f"{section['slowcontrol_missing']}\n"
            )

            f.write(
                f"Runs successfully used: "
                f"{section['successful']}\n"
            )

            f.write("\n")

            # ------------------------------------------------------
            # Missing ROOT
            # ------------------------------------------------------

            f.write(
                "Missing ROOT runs:\n"
            )

            f.write(
                f"  {compress_runs(section['missing_root_runs'])}\n"
            )

            f.write("\n")

            # ------------------------------------------------------
            # Missing slow control
            # ------------------------------------------------------

            f.write(
                "Missing slow-control runs:\n"
            )

            f.write(
                "  "
                f"{compress_runs(section['missing_slowcontrol_runs'])}\n"
            )

            f.write("\n")

            # ------------------------------------------------------
            # Unsuccessful processing
            # ------------------------------------------------------

            f.write(
                "Unsuccessfully processed runs:\n"
            )

            f.write(
                f"  {compress_runs(section['unsuccessful_runs'])}\n"
            )

            f.write("\n")

            # ------------------------------------------------------
            # Successfully processed runs
            # ------------------------------------------------------

            f.write(
                "Successfully processed runs:\n"
            )

            f.write(
                f"  {compress_runs(section['successful_runs'])}\n"
            )

            f.write("\n")

    print(
        f"Run-processing summary written to:\n"
        f"  {output_file}"
    )


# ======================================================================
# Main
# ======================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Count MURAVES chi2-selected tracks and "
            "total run duration per detector, VES/FS "
            "and Working Point."
        )
    )

    parser.add_argument(
        "--reconstructed-dir",
        type=Path,
        default=Path(
            "/pnfs/iihe/muraves/muraves_DATA/RECONSTRUCTED"
        ),
        help=(
            "Base directory containing reconstructed data."
        ),
    )

    parser.add_argument(
        "--slowcontrol-dir",
        type=Path,
        default=Path(
            "/pnfs/iihe/muraves/muraves_DATA/DECOMPRESSED"
        ),
        help=(
            "Base directory containing slow-control files."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "counts_per_working_point.csv"
        ),
        help=(
            "Output CSV for Working Point track counts "
            "and run durations."
        ),
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "run_processing_summary.txt"
        ),
        help=(
            "Output text file for run processing information."
        ),
    )

    parser.add_argument(
        "--detectors",
        nargs="+",
        choices=DETECTORS,
        default=DETECTORS,
        help=(
            "Detectors to process "
            "(default: ROSSO BLU NERO)."
        ),
    )

    args = parser.parse_args()

    results = []
    summary_sections = []

    # ==============================================================
    # Process requested detectors
    # ==============================================================

    for detector in args.detectors:

        process_detector(
            detector=detector,
            reconstructed_base=args.reconstructed_dir,
            slowcontrol_base=args.slowcontrol_dir,
            results=results,
            summary_sections=summary_sections,
        )

    # ==============================================================
    # Write outputs
    # ==============================================================

    write_counts_output(
        results,
        args.output,
    )

    write_summary_output(
        summary_sections,
        args.summary_output,
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()