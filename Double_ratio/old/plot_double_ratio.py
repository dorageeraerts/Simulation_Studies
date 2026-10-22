#!/usr/bin/env python3
"""
plot_selected_tracks.py

Post-processing of selected-track Parquet files.

What this script does
---------------------
1. Reads selected tracks from VES and FS Parquet files.
2. Finds the corresponding ROOT files for the selected runs.
3. Reads ONLY the timestamp branch from each ROOT file to determine run duration.
4. Calculates selected-track rates in angular bins:
       rate = number of selected tracks / total live time
5. Calculates measured transmission from rates:
       T_measured = rate_VES / rate_FS
6. Loads either PUMAS or Mulder simulation.
7. Rebinns the simulated transmission into the same angular bins.
8. Calculates:
       double_ratio = T_measured / T_simulation
9. Produces plots for:
       - FS selected-track rate
       - VES selected-track rate
       - measured transmission
       - simulated transmission
       - double ratio
       - selected-track counts vs working point
       - selected-track rates vs working point

Important
---------
The Parquet files do not contain timestamps, so the ROOT files are accessed
again only to reconstruct the duration of each run.

The ROOT files are NOT reread for the track selection itself.

Example
-------
python plot_selected_tracks.py \
    --ves-parquet VES_selected_tracks.parquet \
    --fs-parquet FS_selected_tracks.parquet \
    --root-dir /pnfs/iihe/muraves/muraves_DATA/DECOMPRESSED \
    --sim pumas \
    --sim-file pumas_output.txt \
    --output-dir plots

or:

python plot_selected_tracks.py \
    --ves-parquet VES_selected_tracks.parquet \
    --fs-parquet FS_selected_tracks.parquet \
    --root-dir /pnfs/iihe/muraves/muraves_DATA/DECOMPRESSED \
    --sim mulder \
    --sim-file mulder_output.txt \
    --output-dir plots
"""

import argparse
import os
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

# Angular binning.
#
# Change these if your original analysis used different bins.
#
# Phi:
#   degrees
#
# Theta:
#   degrees
#
PHI_BINS = np.arange(-180.0, 181.0, 10.0)
THETA_BINS = np.arange(0.0, 91.0, 5.0)

# ROOT timestamp branch
TIMESTAMP_BRANCH = "timestamp"

# Simulation column names
SIM_COLUMNS = [
    "azimuth",
    "elevation",
    "flux_rock",
    "flux_rock_err",
    "flux_open",
    "transmission",
    "transmission_err",
    "n_events",
]


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot selected-track rates, transmission and double ratios."
    )

    parser.add_argument(
        "--ves-parquet",
        required=True,
        help="VES selected-track Parquet file.",
    )

    parser.add_argument(
        "--fs-parquet",
        required=True,
        help="FS selected-track Parquet file.",
    )

    parser.add_argument(
        "--root-dir",
        required=True,
        help=(
            "Root directory containing the decompressed detector directories. "
            "Example: /pnfs/iihe/muraves/muraves_DATA/DECOMPRESSED"
        ),
    )

    parser.add_argument(
        "--sim",
        required=True,
        choices=["pumas", "mulder"],
        help="Simulation to use for the double ratio.",
    )

    parser.add_argument(
        "--sim-file",
        required=True,
        help="PUMAS or Mulder simulation text output.",
    )

    parser.add_argument(
        "--output-dir",
        default="plots_selected_tracks",
        help="Directory where plots and tables are saved.",
    )

    parser.add_argument(
        "--root-pattern",
        default="*.root",
        help=(
            "ROOT filename pattern used when searching detector directories. "
            "Default: *.root"
        ),
    )

    return parser.parse_args()


# ============================================================
# LOAD PARQUETS
# ============================================================

def load_selected_tracks(filename, expected_type):
    print(f"\nLoading {filename}")

    df = pd.read_parquet(filename)

    required_columns = [
        "run_number",
        "detector",
        "data_type",
        "working_point",
        "Phi_3p",
        "Theta_3p",
        "Phi_4p",
        "Theta_4p",
    ]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"{filename} is missing required columns: {missing}"
        )

    if "data_type" in df.columns:
        unique_types = df["data_type"].dropna().unique()

        if len(unique_types) > 0:
            print(f"  data_type values: {unique_types}")

    print(f"  Number of selected tracks: {len(df):,}")
    print(f"  Runs: {df['run_number'].nunique():,}")
    print(f"  Detectors: {df['detector'].nunique():,}")

    return df


# ============================================================
# ROOT FILE FINDING
# ============================================================

def find_root_file(root_dir, run_number):
    """
    Find the AnalyzedData ROOT file for a given run.

    Expected filename:
        MURAVES_AnalyzedData_run<N>.root
    """
    root_dir = Path(root_dir)

    exact_file = root_dir / f"MURAVES_AnalyzedData_run{run_number}.root"

    if exact_file.exists():
        return exact_file

    print(
        f"WARNING: AnalyzedData ROOT file not found for run {run_number}: "
        f"{exact_file}"
    )
    return None


# ============================================================
# RUN DURATION
# ============================================================

def get_run_duration(root_file):
    """
    Read only the timestamp branch from a ROOT file and calculate:

        duration_seconds = (tmax - tmin) / 1000

    This matches the calculation used in the original pipeline.
    """

    try:
        with uproot.open(root_file) as f:

            # Find a tree containing the timestamp branch.
            tree = None

            for key, obj in f.items():
                try:
                    if hasattr(obj, "keys"):
                        keys = obj.keys()

                        # uproot can return names with ;1 etc.
                        clean_keys = [
                            str(k).split(";")[0]
                            for k in keys
                        ]

                        if TIMESTAMP_BRANCH in clean_keys:
                            tree = obj
                            break

                except Exception:
                    continue

            if tree is None:
                raise RuntimeError(
                    f"Could not find a tree containing "
                    f"'{TIMESTAMP_BRANCH}' in {root_file}"
                )

            timestamps = tree[TIMESTAMP_BRANCH].array(
                library="np"
            )

            timestamps = np.asarray(timestamps)

            if len(timestamps) == 0:
                raise RuntimeError(
                    f"No timestamps found in {root_file}"
                )

            timestamps = timestamps[np.isfinite(timestamps)]

            if len(timestamps) == 0:
                raise RuntimeError(
                    f"No finite timestamps found in {root_file}"
                )

            tmin = np.min(timestamps)
            tmax = np.max(timestamps)

            duration_seconds = (tmax - tmin) / 1000.0

            if duration_seconds <= 0:
                raise RuntimeError(
                    f"Non-positive run duration: "
                    f"{duration_seconds} s"
                )

            return float(duration_seconds)

    except Exception as exc:
        print(f"ERROR reading duration from {root_file}")
        print(f"      {exc}")
        return None


# ============================================================
# BUILD RUN-DURATION TABLE
# ============================================================

def build_run_duration_table(df, root_dir, root_pattern):
    """
    Determine the duration for every unique detector/run combination
    appearing in the selected-track Parquet.
    """

    combinations = (
        df[["run_number", "detector", "data_type"]]
        .drop_duplicates()
        .sort_values(["detector", "run_number"])
    )

    rows = []

    print("\n" + "=" * 70)
    print("CALCULATING RUN DURATIONS")
    print("=" * 70)

    for _, row in combinations.iterrows():

        run_number = row["run_number"]
        detector = row["detector"]
        data_type = row["data_type"]

        root_file = find_root_file(
            root_dir=root_dir,
            run_number=run_number
        )

        if root_file is None:
            print(
                f"WARNING: ROOT file not found for "
                f"detector={detector}, run={run_number}"
            )
            continue

        duration = get_run_duration(root_file)

        if duration is None:
            continue

        rows.append(
            {
                "run_number": run_number,
                "detector": detector,
                "data_type": data_type,
                "duration_seconds": duration,
                "root_file": root_file,
            }
        )

        print(
            f"  {detector} | run {run_number} | "
            f"{data_type} | {duration:.3f} s"
        )

    duration_df = pd.DataFrame(rows)

    if duration_df.empty:
        raise RuntimeError(
            "No run durations could be calculated. "
            "Check --root-dir and the ROOT filename convention."
        )

    return duration_df


# ============================================================
# ATTACH DURATION TO TRACKS
# ============================================================

def attach_durations(df, duration_df):
    """
    Attach duration_seconds to every selected track based on
    run_number + detector + data_type.
    """

    merge_columns = [
        "run_number",
        "detector",
        "data_type",
    ]

    result = df.merge(
        duration_df[
            merge_columns + ["duration_seconds"]
        ],
        on=merge_columns,
        how="left",
        validate="many_to_one",
    )

    missing = result["duration_seconds"].isna()

    if missing.any():
        missing_runs = (
            result.loc[
                missing,
                ["run_number", "detector", "data_type"]
            ]
            .drop_duplicates()
        )

        print("\nWARNING: Some tracks have no run duration:")
        print(missing_runs.to_string(index=False))

        result = result.loc[~missing].copy()

    return result


# ============================================================
# ANGULAR HISTOGRAM
# ============================================================

def angular_histogram(
    df,
    phi_column,
    theta_column,
    phi_bins,
    theta_bins,
):
    """
    Return a 2D histogram:

        rows    = theta bins
        columns = phi bins
    """

    valid = (
        np.isfinite(df[phi_column])
        & np.isfinite(df[theta_column])
    )

    phi = df.loc[valid, phi_column].to_numpy()
    theta = df.loc[valid, theta_column].to_numpy()

    hist, _, _ = np.histogram2d(
        theta,
        phi,
        bins=[theta_bins, phi_bins],
    )

    return hist


# ============================================================
# TOTAL EXPOSURE
# ============================================================

def total_exposure(df):
    """
    Sum unique run durations.

    We must NOT sum duration_seconds once per track.
    """

    run_info = (
        df[
            [
                "run_number",
                "detector",
                "data_type",
                "duration_seconds",
            ]
        ]
        .drop_duplicates()
    )

    return run_info["duration_seconds"].sum()


# ============================================================
# RATE MAP
# ============================================================

def make_rate_map(
    df,
    phi_column,
    theta_column,
    phi_bins,
    theta_bins,
):
    """
    Calculate:

        rate(theta, phi) =
            counts(theta, phi) / total exposure

    Units: tracks / second
    """

    counts = angular_histogram(
        df,
        phi_column,
        theta_column,
        phi_bins,
        theta_bins,
    )

    exposure = total_exposure(df)

    if exposure <= 0:
        raise RuntimeError("Total exposure is <= 0.")

    rate = counts / exposure

    return counts, rate, exposure


# ============================================================
# SIMULATION
# ============================================================

def load_simulation(filename):
    """
    Load PUMAS/Mulder output.

    Expected columns:

        azimuth
        elevation
        flux_rock
        flux_rock_err
        flux_open
        transmission
        transmission_err
        n_events
    """

    print(f"\nLoading simulation: {filename}")

    try:
        sim = pd.read_csv(
            filename,
            sep=r"\s+",
            comment="#",
            names=SIM_COLUMNS,
            header=None,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not read simulation file {filename}: {exc}"
        )

    # Remove accidental header rows.
    sim["azimuth"] = pd.to_numeric(
        sim["azimuth"],
        errors="coerce",
    )

    sim["elevation"] = pd.to_numeric(
        sim["elevation"],
        errors="coerce",
    )

    sim["transmission"] = pd.to_numeric(
        sim["transmission"],
        errors="coerce",
    )

    sim = sim.dropna(
        subset=[
            "azimuth",
            "elevation",
            "transmission",
        ]
    ).copy()

    print(f"  Simulation points: {len(sim):,}")

    return sim


# ============================================================
# REBIN SIMULATION
# ============================================================

def rebin_simulation(
    sim,
    phi_bins,
    theta_bins,
):
    """
    Rebin simulation transmission into the analysis angular bins.

    Uses the mean simulation transmission of points falling into
    each angular bin.
    """

    result = np.full(
        (
            len(theta_bins) - 1,
            len(phi_bins) - 1,
        ),
        np.nan,
        dtype=float,
    )

    for i in range(len(theta_bins) - 1):

        theta_low = theta_bins[i]
        theta_high = theta_bins[i + 1]

        theta_mask = (
            (sim["elevation"] >= theta_low)
            & (sim["elevation"] < theta_high)
        )

        for j in range(len(phi_bins) - 1):

            phi_low = phi_bins[j]
            phi_high = phi_bins[j + 1]

            phi_mask = (
                (sim["azimuth"] >= phi_low)
                & (sim["azimuth"] < phi_high)
            )

            values = sim.loc[
                theta_mask & phi_mask,
                "transmission",
            ].to_numpy()

            if len(values) > 0:
                result[i, j] = np.nanmean(values)

    return result


# ============================================================
# PLOTTING HELPERS
# ============================================================

def plot_2d(
    array,
    phi_bins,
    theta_bins,
    title,
    cbar_label,
    output_file,
    vmin=None,
    vmax=None,
    cmap="viridis",
):
    plt.figure(figsize=(10, 7))

    masked = np.ma.masked_invalid(array)

    mesh = plt.pcolormesh(
        phi_bins,
        theta_bins,
        masked,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    plt.colorbar(mesh, label=cbar_label)

    plt.xlabel(r"$\phi$ [deg]")
    plt.ylabel(r"$\theta$ [deg]")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Saved: {output_file}")


def plot_wp(
    x,
    y,
    xlabel,
    ylabel,
    title,
    output_file,
):
    plt.figure(figsize=(9, 6))

    plt.plot(
        x,
        y,
        marker="o",
        linestyle="-",
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Saved: {output_file}")


# ============================================================
# WORKING POINT PLOTS
# ============================================================

def calculate_wp_statistics(df):
    """
    Calculate selected-track count and rate for every WP.

    Rate is calculated from the total exposure of the unique
    runs belonging to that WP.
    """

    rows = []

    for wp, wp_df in df.groupby("working_point"):

        counts = len(wp_df)

        exposure = total_exposure(wp_df)

        rate = (
            counts / exposure
            if exposure > 0
            else np.nan
        )

        rows.append(
            {
                "working_point": wp,
                "track_count": counts,
                "duration_seconds": exposure,
                "track_rate_hz": rate,
            }
        )

    result = pd.DataFrame(rows)

    return result.sort_values("working_point")


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --------------------------------------------------------
    # Load selected tracks
    # --------------------------------------------------------

    ves = load_selected_tracks(
        args.ves_parquet,
        "VES",
    )

    fs = load_selected_tracks(
        args.fs_parquet,
        "FS",
    )

    # --------------------------------------------------------
    # Calculate run durations
    # --------------------------------------------------------

    ves_durations = build_run_duration_table(
        ves,
        args.root_dir,
        args.root_pattern,
    )

    fs_durations = build_run_duration_table(
        fs,
        args.root_dir,
        args.root_pattern,
    )

    # Save durations so you have them afterwards too.
    all_durations = pd.concat(
        [ves_durations, fs_durations],
        ignore_index=True,
    ).drop_duplicates(
        subset=[
            "run_number",
            "detector",
            "data_type",
        ]
    )

    duration_output = os.path.join(
        args.output_dir,
        "run_durations_reconstructed.csv",
    )

    all_durations.to_csv(
        duration_output,
        index=False,
    )

    print(f"\nSaved reconstructed durations: {duration_output}")

    # --------------------------------------------------------
    # Attach duration to selected tracks
    # --------------------------------------------------------

    ves = attach_durations(
        ves,
        ves_durations,
    )

    fs = attach_durations(
        fs,
        fs_durations,
    )

    # --------------------------------------------------------
    # Make angular rate maps
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CALCULATING TRACK RATES")
    print("=" * 70)

    ves_counts, ves_rate, ves_exposure = make_rate_map(
        ves,
        "Phi_3p",
        "Theta_3p",
        PHI_BINS,
        THETA_BINS,
    )

    fs_counts, fs_rate, fs_exposure = make_rate_map(
        fs,
        "Phi_3p",
        "Theta_3p",
        PHI_BINS,
        THETA_BINS,
    )

    print(
        f"VES exposure: {ves_exposure:.3f} s "
        f"({ves_exposure / 3600.0:.3f} h)"
    )

    print(
        f"FS exposure:  {fs_exposure:.3f} s "
        f"({fs_exposure / 3600.0:.3f} h)"
    )

    print(
        f"VES selected tracks: {len(ves):,}"
    )

    print(
        f"FS selected tracks:  {len(fs):,}"
    )

    # --------------------------------------------------------
    # Measured transmission
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # This is a RATE ratio:
    #
    #       T = R_VES / R_FS
    #
    # not:
    #
    #       T = N_VES / N_FS
    #
    # --------------------------------------------------------

    measured_transmission = np.full_like(
        ves_rate,
        np.nan,
        dtype=float,
    )

    valid_fs = fs_rate > 0

    measured_transmission[valid_fs] = (
        ves_rate[valid_fs]
        / fs_rate[valid_fs]
    )

    # --------------------------------------------------------
    # Simulation
    # --------------------------------------------------------

    sim = load_simulation(
        args.sim_file
    )

    sim_transmission = rebin_simulation(
        sim,
        PHI_BINS,
        THETA_BINS,
    )

    # --------------------------------------------------------
    # Double ratio
    # --------------------------------------------------------
    #
    #       DR = measured transmission /
    #            simulated transmission
    #
    # --------------------------------------------------------

    double_ratio = np.full_like(
        measured_transmission,
        np.nan,
        dtype=float,
    )

    valid = (
        np.isfinite(measured_transmission)
        & np.isfinite(sim_transmission)
        & (sim_transmission != 0)
    )

    double_ratio[valid] = (
        measured_transmission[valid]
        / sim_transmission[valid]
    )

    # --------------------------------------------------------
    # Save numerical maps
    # --------------------------------------------------------

    theta_centers = (
        THETA_BINS[:-1]
        + THETA_BINS[1:]
    ) / 2.0

    phi_centers = (
        PHI_BINS[:-1]
        + PHI_BINS[1:]
    ) / 2.0

    def save_map(array, filename, value_name):

        rows = []

        for i, theta in enumerate(theta_centers):
            for j, phi in enumerate(phi_centers):

                rows.append(
                    {
                        "theta_center": theta,
                        "phi_center": phi,
                        value_name: array[i, j],
                    }
                )

        pd.DataFrame(rows).to_csv(
            filename,
            index=False,
        )

    save_map(
        ves_rate,
        os.path.join(
            args.output_dir,
            "VES_track_rate_map.csv",
        ),
        "ves_rate_hz",
    )

    save_map(
        fs_rate,
        os.path.join(
            args.output_dir,
            "FS_track_rate_map.csv",
        ),
        "fs_rate_hz",
    )

    save_map(
        measured_transmission,
        os.path.join(
            args.output_dir,
            "measured_transmission_map.csv",
        ),
        "measured_transmission",
    )

    save_map(
        sim_transmission,
        os.path.join(
            args.output_dir,
            f"{args.sim}_transmission_map.csv",
        ),
        "simulated_transmission",
    )

    save_map(
        double_ratio,
        os.path.join(
            args.output_dir,
            f"double_ratio_{args.sim}.csv",
        ),
        "double_ratio",
    )

    # --------------------------------------------------------
    # Plots: track rates
    # --------------------------------------------------------

    plot_2d(
        fs_rate,
        PHI_BINS,
        THETA_BINS,
        "FS selected-track rate",
        "Track rate [Hz]",
        os.path.join(
            args.output_dir,
            "FS_track_rate.png",
        ),
    )

    plot_2d(
        ves_rate,
        PHI_BINS,
        THETA_BINS,
        "VES selected-track rate",
        "Track rate [Hz]",
        os.path.join(
            args.output_dir,
            "VES_track_rate.png",
        ),
    )

    # --------------------------------------------------------
    # Plot: measured transmission
    # --------------------------------------------------------

    plot_2d(
        measured_transmission,
        PHI_BINS,
        THETA_BINS,
        "Measured transmission from track rates",
        "Transmission",
        os.path.join(
            args.output_dir,
            "measured_transmission.png",
        ),
        vmin=0,
        vmax=1,
    )

    # --------------------------------------------------------
    # Plot: simulation
    # --------------------------------------------------------

    plot_2d(
        sim_transmission,
        PHI_BINS,
        THETA_BINS,
        f"{args.sim.upper()} simulated transmission",
        "Transmission",
        os.path.join(
            args.output_dir,
            f"{args.sim}_transmission.png",
        ),
        vmin=0,
        vmax=1,
    )

    # --------------------------------------------------------
    # Plot: double ratio
    # --------------------------------------------------------

    plot_2d(
        double_ratio,
        PHI_BINS,
        THETA_BINS,
        f"Double ratio: measured / {args.sim.upper()}",
        "Double ratio",
        os.path.join(
            args.output_dir,
            f"double_ratio_{args.sim}.png",
        ),
        vmin=0,
        vmax=2,
    )

    # --------------------------------------------------------
    # Working point statistics
    # --------------------------------------------------------

    ves_wp = calculate_wp_statistics(ves)
    fs_wp = calculate_wp_statistics(fs)

    ves_wp.to_csv(
        os.path.join(
            args.output_dir,
            "VES_working_point_statistics.csv",
        ),
        index=False,
    )

    fs_wp.to_csv(
        os.path.join(
            args.output_dir,
            "FS_working_point_statistics.csv",
        ),
        index=False,
    )

    # --------------------------------------------------------
    # WP count plots
    # --------------------------------------------------------

    plot_wp(
        ves_wp["working_point"],
        ves_wp["track_count"],
        "Working point",
        "Selected tracks",
        "VES selected tracks vs working point",
        os.path.join(
            args.output_dir,
            "VES_tracks_vs_WP.png",
        ),
    )

    plot_wp(
        fs_wp["working_point"],
        fs_wp["track_count"],
        "Working point",
        "Selected tracks",
        "FS selected tracks vs working point",
        os.path.join(
            args.output_dir,
            "FS_tracks_vs_WP.png",
        ),
    )

    # --------------------------------------------------------
    # WP rate plots
    # --------------------------------------------------------

    plot_wp(
        ves_wp["working_point"],
        ves_wp["track_rate_hz"],
        "Working point",
        "Track rate [Hz]",
        "VES selected-track rate vs working point",
        os.path.join(
            args.output_dir,
            "VES_rate_vs_WP.png",
        ),
    )

    plot_wp(
        fs_wp["working_point"],
        fs_wp["track_rate_hz"],
        "Working point",
        "Track rate [Hz]",
        "FS selected-track rate vs working point",
        os.path.join(
            args.output_dir,
            "FS_rate_vs_WP.png",
        ),
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print(f"Output directory: {args.output_dir}")
    print(f"Simulation:       {args.sim.upper()}")

    print("\nExposures:")
    print(
        f"  VES: {ves_exposure:.3f} s "
        f"({ves_exposure / 3600.0:.3f} h)"
    )
    print(
        f"  FS:  {fs_exposure:.3f} s "
        f"({fs_exposure / 3600.0:.3f} h)"
    )

    print("\nTotal selected tracks:")
    print(f"  VES: {len(ves):,}")
    print(f"  FS:  {len(fs):,}")

    print("\nMean overall rates:")
    print(
        f"  VES: {len(ves) / ves_exposure:.6g} Hz"
    )
    print(
        f"  FS:  {len(fs) / fs_exposure:.6g} Hz"
    )

    print("\nThe measured transmission was calculated as:")
    print("  T_measured = rate_VES / rate_FS")

    print("\nThe double ratio was calculated as:")
    print(
        f"  DR = T_measured / T_{args.sim}"
    )


if __name__ == "__main__":
    main()