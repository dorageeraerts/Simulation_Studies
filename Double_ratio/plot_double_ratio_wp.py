#!/usr/bin/env python3

"""
Plot MURAVES selected-track data and compare with PUMAS/Mulder simulation.

INPUTS
------
Two Parquet files containing the selected tracks:

    VES parquet
    FS  parquet

The Parquet files must contain at least:

    run_number
    detector
    data_type
    working_point
    Phi_3p
    Theta_3p

ROOT files are read ONLY to reconstruct the duration of each run.

The ROOT files are expected to be:

    MURAVES_AnalyzedData_runXXXX.root

and to contain:

    AnalyzedData/timestamp

The miniRunTree files are deliberately ignored.

For each run:

    duration = (max(timestamp) - min(timestamp)) / 1000

because timestamp is assumed to be in milliseconds.

DATA PRODUCTS
-------------
1. FS selected-track rate
2. VES selected-track rate
3. Measured transmission

       T_measured = R_VES / R_FS

4. Simulated transmission
5. Double ratio

       D = T_measured / T_sim

WORKING POINT
-------------
By default all Working Points are combined.

To select one WP:

    --working-point 25

The WP filtering is applied to BOTH VES and FS.

BINNING
-------
The same azimuth/elevation binning is used for:

    VES rate
    FS rate
    measured transmission
    simulated transmission
    double ratio

ANGLES
------
The selected-track angular variables are:

    Phi_3p
    Theta_3p

The default binning is:

    azimuth   = 0 -> 360 deg, 5 deg bins
    elevation = 0 -> 90 deg,  2 deg bins

EXAMPLE
-------
python plot_double_ratio.py \
    --ves-parquet /path/SelectedTracks_VES.parquet \
    --fs-parquet /path/SelectedTracks_FS.parquet \
    --root-dir /pnfs/iihe/muraves/muraves_DATA/RECONSTRUCTED/ROSSO/v1 \
    --sim mulder \
    --sim-file /path/to/mulder.txt \
    --output-dir plots_mulder

For one Working Point:

python plot_double_ratio.py \
    --ves-parquet /path/SelectedTracks_VES.parquet \
    --fs-parquet /path/SelectedTracks_FS.parquet \
    --root-dir /pnfs/iihe/muraves/muraves_DATA/RECONSTRUCTED/ROSSO/v1 \
    --sim mulder \
    --sim-file /path/to/mulder.txt \
    --working-point 25 \
    --output-dir plots_mulder_wp25
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import uproot


# ============================================================
# Constants
# ============================================================

TREE_NAME = "AnalyzedData"

TIMESTAMP_BRANCH = "timestamp"

ANALYZEDDATA_FILENAME_TEMPLATE = (
    "MURAVES_AnalyzedData_run{run}.root"
)

REQUIRED_PARQUET_COLUMNS = [
    "run_number",
    "detector",
    "data_type",
    "working_point",
    "Phi_3p",
    "Theta_3p",
]

FLUX_COLUMNS = [
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
# Argument parsing
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Plot MURAVES selected-track rates, measured "
            "transmission, simulated transmission and double ratio."
        )
    )

    # --------------------------------------------------------
    # Parquets
    # --------------------------------------------------------

    parser.add_argument(
        "--ves-parquet",
        required=True,
        type=str,
        help="VES selected-track Parquet file.",
    )

    parser.add_argument(
        "--fs-parquet",
        required=True,
        type=str,
        help="FS selected-track Parquet file.",
    )

    # --------------------------------------------------------
    # ROOT
    # --------------------------------------------------------

    parser.add_argument(
        "--root-dir",
        required=True,
        type=str,
        help=(
            "Directory containing "
            "MURAVES_AnalyzedData_runXXXX.root files."
        ),
    )

    # --------------------------------------------------------
    # Simulation
    # --------------------------------------------------------

    parser.add_argument(
        "--sim",
        required=True,
        choices=["pumas", "mulder"],
        help="Simulation type.",
    )

    parser.add_argument(
        "--sim-file",
        required=True,
        type=str,
        help="PUMAS/Mulder flux-output text file.",
    )

    parser.add_argument(
        "--azimuth-offset",
        type=float,
        default=0.0,
        help="Degrees added to simulated azimuth.",
    )

    # --------------------------------------------------------
    # Working Point
    # --------------------------------------------------------

    parser.add_argument(
        "--working-point",
        type=float,
        default=None,
        help=(
            "Only use this Working Point. "
            "If omitted, all Working Points are used."
        ),
    )

    # --------------------------------------------------------
    # Binning
    # --------------------------------------------------------

    parser.add_argument(
        "--az-min",
        type=float,
        default=150.0,
        help="Minimum azimuth [deg].",
    )

    parser.add_argument(
        "--az-max",
        type=float,
        default=210.0,
        help="Maximum azimuth [deg].",
    )

    parser.add_argument(
        "--az-bin-size",
        type=float,
        default=1.0,
        help="Azimuth bin width [deg].",
    )

    parser.add_argument(
        "--el-min",
        type=float,
        default=0.0,
        help="Minimum elevation [deg].",
    )

    parser.add_argument(
        "--el-max",
        type=float,
        default=40.0,
        help="Maximum elevation [deg].",
    )

    parser.add_argument(
        "--el-bin-size",
        type=float,
        default=1.0,
        help="Elevation bin width [deg].",
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Output directory.",
    )

    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Annotate double-ratio bins with numerical values.",
    )

    return parser.parse_args()


# ============================================================
# Binning
# ============================================================

def make_edges(min_val, max_val, bin_size):

    n_bins_float = (
        (max_val - min_val) / bin_size
    )

    n_bins = int(round(n_bins_float))

    if not np.isclose(
        n_bins_float,
        n_bins,
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError(
            f"Range [{min_val}, {max_val}] is not an integer "
            f"multiple of bin size {bin_size}."
        )

    if n_bins <= 0:
        raise ValueError(
            "Number of bins must be positive."
        )

    return np.linspace(
        min_val,
        max_val,
        n_bins + 1,
    )


# ============================================================
# Parquet loading
# ============================================================

def load_selected_tracks(path, expected_data_type):

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Parquet file not found:\n{path}"
        )

    print(f"\nLoading {expected_data_type} tracks:")
    print(f"  {path}")

    df = pd.read_parquet(path)

    missing = set(REQUIRED_PARQUET_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"{path} is missing required columns:\n"
            f"{sorted(missing)}"
        )

    if df.empty:
        print("  WARNING: Parquet is empty.")

    # --------------------------------------------------------
    # Enforce expected data type
    # --------------------------------------------------------

    unique_types = (
        df["data_type"]
        .dropna()
        .astype(str)
        .unique()
    )

    if len(unique_types) > 0:
        print(
            f"  Data types found: "
            f"{', '.join(unique_types)}"
        )

    wrong_type = (
        df["data_type"].astype(str)
        != expected_data_type
    )

    if wrong_type.any():

        print(
            f"  WARNING: "
            f"{wrong_type.sum()} rows do not have "
            f"data_type='{expected_data_type}'. "
            f"They will be removed."
        )

        df = df.loc[~wrong_type].copy()

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for column in [
        "run_number",
        "working_point",
        "Phi_3p",
        "Theta_3p",
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    before = len(df)

    df = df.dropna(
        subset=[
            "run_number",
            "working_point",
            "Phi_3p",
            "Theta_3p",
        ]
    ).copy()

    removed = before - len(df)

    if removed:
        print(
            f"  Removed {removed} rows with "
            f"invalid required values."
        )

    df["run_number"] = (
        df["run_number"]
        .astype(int)
    )

    print(
        f"  Selected tracks: {len(df)}"
    )

    print(
        f"  Runs: {df['run_number'].nunique()}"
    )

    print(
        f"  Working Points: "
        f"{sorted(df['working_point'].unique())}"
    )

    return df


# ============================================================
# Working Point filtering
# ============================================================

def filter_working_point(df, working_point, label):

    if working_point is None:

        print(
            f"{label}: using ALL Working Points"
        )

        return df.copy()

    # Exact comparison is appropriate because WP values are
    # read from the same slow-control-derived Parquet field.

    mask = np.isclose(
        df["working_point"].to_numpy(dtype=float),
        working_point,
        rtol=0,
        atol=1e-9,
    )

    filtered = df.loc[mask].copy()

    print(
        f"{label}: Working Point = {working_point:g}"
    )

    print(
        f"  Tracks after WP selection: "
        f"{len(filtered)}"
    )

    print(
        f"  Runs after WP selection: "
        f"{filtered['run_number'].nunique()}"
    )

    return filtered


# ============================================================
# ROOT file lookup
# ============================================================

def find_analyzeddata_file(root_dir, run):

    root_dir = Path(root_dir)

    exact_file = (
        root_dir
        / ANALYZEDDATA_FILENAME_TEMPLATE.format(
            run=run
        )
    )

    if exact_file.is_file():
        return exact_file

    # --------------------------------------------------------
    # Fallback search
    # --------------------------------------------------------

    candidates = sorted(
        root_dir.glob(
            f"MURAVES_AnalyzedData_run{run}.root"
        )
    )

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) == 0:

        print(
            f"  WARNING: no AnalyzedData ROOT file "
            f"found for run {run}"
        )

        return None

    print(
        f"  WARNING: multiple AnalyzedData ROOT files "
        f"found for run {run}:"
    )

    for candidate in candidates:
        print(
            f"      {candidate}"
        )

    # In the unlikely case of multiple matching
    # AnalyzedData files, use the exact standard name
    # if possible.

    for candidate in candidates:

        if candidate.name == (
            ANALYZEDDATA_FILENAME_TEMPLATE.format(
                run=run
            )
        ):
            return candidate

    return candidates[0]


# ============================================================
# Run duration
# ============================================================

def get_run_duration(root_file):

    """
    Read timestamp only and calculate:

        duration = (max(timestamp) - min(timestamp)) / 1000

    Timestamp is assumed to be milliseconds.
    """

    try:

        with uproot.open(root_file) as f:

            if TREE_NAME not in f:

                print(
                    f"  WARNING: tree '{TREE_NAME}' "
                    f"not found in {root_file.name}"
                )

                return None

            tree = f[TREE_NAME]

            if TIMESTAMP_BRANCH not in tree.keys():

                print(
                    f"  WARNING: branch "
                    f"'{TIMESTAMP_BRANCH}' not found "
                    f"in {root_file.name}"
                )

                return None

            timestamps = tree[
                TIMESTAMP_BRANCH
            ].array(
                library="np"
            )

    except Exception as exc:

        print(
            f"  WARNING: could not read "
            f"{root_file}: {exc}"
        )

        return None

    timestamps = np.asarray(
        timestamps,
        dtype=float,
    )

    valid = np.isfinite(timestamps)

    if not np.any(valid):

        print(
            f"  WARNING: no valid timestamps "
            f"in {root_file.name}"
        )

        return None

    tmin = np.min(
        timestamps[valid]
    )

    tmax = np.max(
        timestamps[valid]
    )

    duration_seconds = (
        tmax - tmin
    ) / 1000.0

    if duration_seconds <= 0:

        print(
            f"  WARNING: invalid duration "
            f"for {root_file.name}: "
            f"{duration_seconds:.3f} s"
        )

        return None

    return {
        "run_number": int(
            re.search(
                r"run(\d+)",
                root_file.name
            ).group(1)
        ),
        "timestamp_min": tmin,
        "timestamp_max": tmax,
        "duration_seconds": duration_seconds,
    }


# ============================================================
# Build run-duration table
# ============================================================

def build_run_duration_table(
    run_numbers,
    root_dir,
):

    root_dir = Path(root_dir)

    if not root_dir.is_dir():

        raise FileNotFoundError(
            f"ROOT directory not found:\n{root_dir}"
        )

    run_numbers = sorted(
        set(int(r) for r in run_numbers)
    )

    print(
        "\n========================================"
    )

    print(
        "Reconstructing run durations"
    )

    print(
        "========================================"
    )

    print(
        f"ROOT directory: {root_dir}"
    )

    print(
        f"Runs required: {len(run_numbers)}"
    )

    rows = []

    for run in run_numbers:

        root_file = find_analyzeddata_file(
            root_dir,
            run,
        )

        if root_file is None:
            continue

        duration_info = get_run_duration(
            root_file
        )

        if duration_info is None:
            continue

        rows.append(
            duration_info
        )

        print(
            f"  run {run}: "
            f"{duration_info['duration_seconds']:.3f} s"
        )

    duration_df = pd.DataFrame(
        rows,
        columns=[
            "run_number",
            "timestamp_min",
            "timestamp_max",
            "duration_seconds",
        ],
    )

    if duration_df.empty:

        raise RuntimeError(
            "No run durations could be reconstructed."
        )

    print(
        f"\nSuccessfully reconstructed "
        f"{len(duration_df)} / "
        f"{len(run_numbers)} run durations."
    )

    print(
        f"Total exposure: "
        f"{duration_df['duration_seconds'].sum():.3f} s "
        f"("
        f"{duration_df['duration_seconds'].sum() / 3600:.3f} h"
        f")"
    )

    return duration_df


# ============================================================
# Attach duration to selected tracks
# ============================================================

def attach_run_durations(
    df,
    duration_df,
    label,
):

    merged = df.merge(
        duration_df[
            [
                "run_number",
                "duration_seconds",
            ]
        ],
        on="run_number",
        how="left",
        validate="many_to_one",
    )

    missing_mask = (
        merged["duration_seconds"]
        .isna()
    )

    if missing_mask.any():

        missing_runs = sorted(
            merged.loc[
                missing_mask,
                "run_number"
            ].unique()
        )

        print(
            f"\nWARNING: {label} has "
            f"{missing_mask.sum()} tracks from "
            f"runs without a reconstructed duration:"
        )

        print(
            f"  {missing_runs}"
        )

        merged = merged.loc[
            ~missing_mask
        ].copy()

    return merged


# ============================================================
# Angular histogram
# ============================================================

def make_track_histogram(
    df,
    az_edges,
    el_edges,
):

    if df.empty:

        return np.zeros(
            (
                len(az_edges) - 1,
                len(el_edges) - 1,
            ),
            dtype=float,
        )

    phi = df["Phi_3p"].to_numpy(
        dtype=float
    )

    theta = df["Theta_3p"].to_numpy(
        dtype=float
    )

    valid = (
        np.isfinite(phi)
        & np.isfinite(theta)
    )

    counts, _, _ = np.histogram2d(
        phi[valid],
        theta[valid],
        bins=[
            az_edges,
            el_edges,
        ],
    )

    return counts


# ============================================================
# Total exposure
# ============================================================

def get_total_exposure(df):

    """
    The exposure is the sum of the duration of unique runs,
    NOT the sum over event rows.

    This is important because every selected track from a
    given run carries the same run duration.
    """

    if df.empty:
        return 0.0

    exposure = (
        df[
            [
                "run_number",
                "duration_seconds",
            ]
        ]
        .drop_duplicates(
            subset=["run_number"]
        )[
            "duration_seconds"
        ]
        .sum()
    )

    return float(exposure)


# ============================================================
# Rate map
# ============================================================

def calculate_track_rate(
    df,
    az_edges,
    el_edges,
):

    counts = make_track_histogram(
        df,
        az_edges,
        el_edges,
    )

    exposure = get_total_exposure(
        df
    )

    if exposure <= 0:

        rate = np.zeros_like(
            counts,
            dtype=float,
        )

    else:

        rate = (
            counts
            / exposure
        )

    return (
        counts,
        rate,
        exposure,
    )


# ============================================================
# Simulation
# ============================================================

def parse_density(path):

    basename = os.path.basename(
        path
    )

    match = re.search(
        r"rho(\d+)",
        basename,
    )

    if not match:
        return None

    return (
        int(match.group(1))
        / 100.0
    )


def load_simulation(
    path,
    azimuth_offset=0.0,
):

    path = Path(path)

    if not path.is_file():

        raise FileNotFoundError(
            f"Simulation file not found:\n{path}"
        )

    data = np.loadtxt(path)

    if data.ndim == 1:
        data = data.reshape(
            1,
            -1,
        )

    if data.shape[1] != len(
        FLUX_COLUMNS
    ):

        raise ValueError(
            f"Simulation file has "
            f"{data.shape[1]} columns, "
            f"expected {len(FLUX_COLUMNS)}."
        )

    sim = {
        name: data[:, i]
        for i, name in enumerate(
            FLUX_COLUMNS
        )
    }

    sim["azimuth"] = (
        sim["azimuth"]
        + azimuth_offset
    )

    sim["label"] = path.name
    sim["path"] = str(path)
    sim["density"] = parse_density(
        path
    )

    return sim


# ============================================================
# Rebin simulated transmission
# ============================================================

def rebin_simulation(
    sim,
    az_edges,
    el_edges,
):

    az = np.asarray(
        sim["azimuth"],
        dtype=float,
    )

    el = np.asarray(
        sim["elevation"],
        dtype=float,
    )

    transmission = np.asarray(
        sim["transmission"],
        dtype=float,
    )

    valid = (
        np.isfinite(az)
        & np.isfinite(el)
        & np.isfinite(transmission)
    )

    az = az[valid]
    el = el[valid]
    transmission = transmission[valid]

    az_index = (
        np.searchsorted(
            az_edges,
            az,
            side="right",
        )
        - 1
    )

    el_index = (
        np.searchsorted(
            el_edges,
            el,
            side="right",
        )
        - 1
    )

    valid = (
        (az_index >= 0)
        & (
            az_index
            < len(az_edges) - 1
        )
        & (el_index >= 0)
        & (
            el_index
            < len(el_edges) - 1
        )
    )

    az_index = az_index[valid]
    el_index = el_index[valid]
    transmission = transmission[valid]

    n_az = len(az_edges) - 1
    n_el = len(el_edges) - 1

    sums = np.zeros(
        (n_az, n_el),
        dtype=float,
    )

    counts = np.zeros(
        (n_az, n_el),
        dtype=int,
    )

    np.add.at(
        sums,
        (az_index, el_index),
        transmission,
    )

    np.add.at(
        counts,
        (az_index, el_index),
        1,
    )

    transmission_binned = np.zeros(
        (n_az, n_el),
        dtype=float,
    )

    populated = counts > 0

    transmission_binned[
        populated
    ] = (
        sums[populated]
        / counts[populated]
    )

    print(
        "\nSimulation:"
    )

    print(
        f"  Points used: {len(transmission)}"
    )

    print(
        f"  Populated bins: "
        f"{np.count_nonzero(populated)} / "
        f"{n_az * n_el}"
    )

    return transmission_binned


# ============================================================
# Plot
# ============================================================

def make_plot(
    values,
    az_edges,
    el_edges,
    title,
    colorbar_label,
    output_file,
    annotate=False,
    fmt="{:.2f}",
    vmin=None,
    vmax=None,
):

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    mesh = ax.pcolormesh(
        az_edges,
        el_edges,
        values.T,
        shading="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )

    cbar = fig.colorbar(
        mesh,
        ax=ax,
    )

    cbar.set_label(
        colorbar_label
    )

    ax.set_xlabel(
        "Azimuth [deg]"
    )

    ax.set_ylabel(
        "Elevation [deg]"
    )

    ax.set_title(
        title
    )

    ax.set_xlim(
        az_edges[0],
        az_edges[-1],
    )

    ax.set_ylim(
        el_edges[0],
        el_edges[-1],
    )

    if len(az_edges) <= 73:
        ax.set_xticks(
            az_edges
        )

    if len(el_edges) <= 46:
        ax.set_yticks(
            el_edges
        )

    ax.grid(
        alpha=0.2,
        linewidth=0.5,
    )

    if annotate:

        az_centers = (
            0.5
            * (
                az_edges[:-1]
                + az_edges[1:]
            )
        )

        el_centers = (
            0.5
            * (
                el_edges[:-1]
                + el_edges[1:]
            )
        )

        finite = values[
            np.isfinite(values)
        ]

        if finite.size:

            finite_positive = finite[
                finite != 0
            ]

            if finite_positive.size:
                annotation_scale = (
                    np.nanmax(
                        np.abs(
                            finite_positive
                        )
                    )
                )
            else:
                annotation_scale = 1.0

        else:
            annotation_scale = 1.0

        for i, ac in enumerate(
            az_centers
        ):

            for j, ec in enumerate(
                el_centers
            ):

                value = values[i, j]

                if not np.isfinite(value):
                    continue

                if value == 0:
                    continue

                if abs(value) < 0.6 * annotation_scale:
                    text_color = "white"
                else:
                    text_color = "black"

                ax.text(
                    ac,
                    ec,
                    fmt.format(value),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=text_color,
                )

    fig.tight_layout()

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_file}"
    )


# ============================================================
# Save numerical map
# ============================================================

def save_map_csv(
    values,
    az_edges,
    el_edges,
    output_file,
):

    az_centers = (
        0.5
        * (
            az_edges[:-1]
            + az_edges[1:]
        )
    )

    el_centers = (
        0.5
        * (
            el_edges[:-1]
            + el_edges[1:]
        )
    )

    rows = []

    for i, az in enumerate(
        az_centers
    ):

        for j, el in enumerate(
            el_centers
        ):

            rows.append(
                {
                    "azimuth_center": az,
                    "elevation_center": el,
                    "value": values[i, j],
                }
            )

    pd.DataFrame(rows).to_csv(
        output_file,
        index=False,
    )

    print(
        f"Saved: {output_file}"
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Binning
    # --------------------------------------------------------

    az_edges = make_edges(
        args.az_min,
        args.az_max,
        args.az_bin_size,
    )

    el_edges = make_edges(
        args.el_min,
        args.el_max,
        args.el_bin_size,
    )

    print(
        "\n========================================"
    )

    print(
        "MURAVES selected-track plotting"
    )

    print(
        "========================================"
    )

    print(
        f"Azimuth bins: "
        f"{len(az_edges) - 1} "
        f"x {args.az_bin_size:g} deg"
    )

    print(
        f"Elevation bins: "
        f"{len(el_edges) - 1} "
        f"x {args.el_bin_size:g} deg"
    )

    if args.working_point is None:

        wp_label = "all_WP"

        print(
            "Working Point: ALL"
        )

    else:

        wp_label = (
            f"WP{args.working_point:g}"
        )

        print(
            f"Working Point: "
            f"{args.working_point:g}"
        )

    # --------------------------------------------------------
    # Load Parquets
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
    # Working Point selection
    # --------------------------------------------------------

    ves = filter_working_point(
        ves,
        args.working_point,
        "VES",
    )

    fs = filter_working_point(
        fs,
        args.working_point,
        "FS",
    )

    if ves.empty:
        print(
            "\nWARNING: no VES tracks remain "
            "after filtering."
        )

    if fs.empty:
        print(
            "\nWARNING: no FS tracks remain "
            "after filtering."
        )

    # --------------------------------------------------------
    # Determine required runs
    # --------------------------------------------------------

    all_runs = sorted(
        set(
            ves["run_number"].unique()
        )
        |
        set(
            fs["run_number"].unique()
        )
    )

    print(
        f"\nTotal unique runs required: "
        f"{len(all_runs)}"
    )

    # --------------------------------------------------------
    # Reconstruct run durations
    # --------------------------------------------------------

    duration_df = build_run_duration_table(
        all_runs,
        args.root_dir,
    )

    # Save this because it is useful for checking the
    # reconstructed exposure, but it is NOT an event-level
    # table and it is NOT needed as a second input dataset.

    duration_csv = (
        output_dir
        / f"run_durations_{wp_label}.csv"
    )

    duration_df.to_csv(
        duration_csv,
        index=False,
    )

    print(
        f"Saved run durations: "
        f"{duration_csv}"
    )

    # --------------------------------------------------------
    # Attach durations
    # --------------------------------------------------------

    ves = attach_run_durations(
        ves,
        duration_df,
        "VES",
    )

    fs = attach_run_durations(
        fs,
        duration_df,
        "FS",
    )

    # --------------------------------------------------------
    # Calculate rates
    # --------------------------------------------------------

    (
        counts_ves,
        rate_ves,
        exposure_ves,
    ) = calculate_track_rate(
        ves,
        az_edges,
        el_edges,
    )

    (
        counts_fs,
        rate_fs,
        exposure_fs,
    ) = calculate_track_rate(
        fs,
        az_edges,
        el_edges,
    )

    print(
        "\n========================================"
    )

    print(
        "Exposure / rate summary"
    )

    print(
        "========================================"
    )

    print(
        f"VES selected tracks: "
        f"{len(ves)}"
    )

    print(
        f"VES exposure: "
        f"{exposure_ves:.3f} s "
        f"({exposure_ves / 3600:.3f} h)"
    )

    print(
        f"VES total rate: "
        f"{len(ves) / exposure_ves:.6g} tracks/s"
        if exposure_ves > 0
        else "VES total rate: undefined"
    )

    print()

    print(
        f"FS selected tracks: "
        f"{len(fs)}"
    )

    print(
        f"FS exposure: "
        f"{exposure_fs:.3f} s "
        f"({exposure_fs / 3600:.3f} h)"
    )

    print(
        f"FS total rate: "
        f"{len(fs) / exposure_fs:.6g} tracks/s"
        if exposure_fs > 0
        else "FS total rate: undefined"
    )

    # --------------------------------------------------------
    # Measured transmission
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # R_VES and R_FS are independently normalized by their
    # corresponding total exposures.
    #
    # Therefore:
    #
    #     T_measured = R_VES / R_FS
    #
    # --------------------------------------------------------

    measured_transmission = np.divide(
        rate_ves,
        rate_fs,
        out=np.zeros_like(
            rate_ves,
            dtype=float,
        ),
        where=rate_fs > 0,
    )

    # --------------------------------------------------------
    # Simulation
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        f"Loading {args.sim.upper()} simulation"
    )

    print(
        "========================================"
    )

    print(
        f"Simulation file: "
        f"{args.sim_file}"
    )

    simulation = load_simulation(
        args.sim_file,
        azimuth_offset=args.azimuth_offset,
    )

    if simulation["density"] is not None:

        print(
            f"Density: "
            f"{simulation['density']:.2f} g/cm^3"
        )

    transmission_sim = rebin_simulation(
        simulation,
        az_edges,
        el_edges,
    )

    # --------------------------------------------------------
    # Double ratio
    # --------------------------------------------------------

    double_ratio = np.divide(
        measured_transmission,
        transmission_sim,
        out=np.zeros_like(
            measured_transmission,
            dtype=float,
        ),
        where=transmission_sim > 0,
    )

    # --------------------------------------------------------
    # Save numerical maps
    # --------------------------------------------------------

    save_map_csv(
        counts_fs,
        az_edges,
        el_edges,
        output_dir
        / f"FS_selected_counts_{wp_label}.csv",
    )

    save_map_csv(
        counts_ves,
        az_edges,
        el_edges,
        output_dir
        / f"VES_selected_counts_{wp_label}.csv",
    )

    save_map_csv(
        rate_fs,
        az_edges,
        el_edges,
        output_dir
        / f"FS_track_rate_{wp_label}.csv",
    )

    save_map_csv(
        rate_ves,
        az_edges,
        el_edges,
        output_dir
        / f"VES_track_rate_{wp_label}.csv",
    )

    save_map_csv(
        measured_transmission,
        az_edges,
        el_edges,
        output_dir
        / f"MeasuredTransmission_{wp_label}.csv",
    )

    save_map_csv(
        transmission_sim,
        az_edges,
        el_edges,
        output_dir
        / f"{args.sim}_transmission_{wp_label}.csv",
    )

    save_map_csv(
        double_ratio,
        az_edges,
        el_edges,
        output_dir
        / f"DoubleRatio_{args.sim}_{wp_label}.csv",
    )

    # --------------------------------------------------------
    # Titles
    # --------------------------------------------------------

    bin_label = (
        f"{args.az_bin_size:g}x"
        f"{args.el_bin_size:g} deg"
    )

    if args.working_point is None:
        wp_title = "all Working Points"
    else:
        wp_title = (
            f"Working Point = "
            f"{args.working_point:g}"
        )

    density_title = ""

    if simulation["density"] is not None:

        density_title = (
            f", rho={simulation['density']:.2f} g/cm³"
        )

    # --------------------------------------------------------
    # FS rate
    # --------------------------------------------------------

    make_plot(
        rate_fs,
        az_edges,
        el_edges,
        (
            f"FS selected-track rate — "
            f"{bin_label} — "
            f"{wp_title}"
        ),
        "Track rate [s⁻¹]",
        output_dir
        / f"FS_track_rate_{wp_label}.png",
    )

    # --------------------------------------------------------
    # VES rate
    # --------------------------------------------------------

    make_plot(
        rate_ves,
        az_edges,
        el_edges,
        (
            f"VES selected-track rate — "
            f"{bin_label} — "
            f"{wp_title}"
        ),
        "Track rate [s⁻¹]",
        output_dir
        / f"VES_track_rate_{wp_label}.png",
    )

    # --------------------------------------------------------
    # Measured transmission
    # --------------------------------------------------------

    make_plot(
        measured_transmission,
        az_edges,
        el_edges,
        (
            f"Measured transmission — "
            f"VES / FS — "
            f"{bin_label} — "
            f"{wp_title}"
        ),
        "VES rate / FS rate",
        output_dir
        / f"MeasuredTransmission_{wp_label}.png",
    )

    # --------------------------------------------------------
    # Simulation
    # --------------------------------------------------------

    make_plot(
        transmission_sim,
        az_edges,
        el_edges,
        (
            f"{args.sim.upper()} simulated transmission"
            f"{density_title}"
        ),
        "Transmission",
        output_dir
        / f"{args.sim}_Transmission_{wp_label}.png",
    )

    # --------------------------------------------------------
    # Double ratio
    # --------------------------------------------------------

    make_plot(
        double_ratio,
        az_edges,
        el_edges,
        (
            f"Double ratio — "
            f"data / {args.sim.upper()} — "
            f"{bin_label} — "
            f"{wp_title}"
            f"{density_title}"
        ),
        "Double ratio",
        output_dir
        / f"DoubleRatio_{args.sim}_{wp_label}.png",
        annotate=args.annotate,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "Finished"
    )

    print(
        "========================================"
    )

    print(
        f"Working Point: "
        f"{'ALL' if args.working_point is None else args.working_point:g}"
    )

    print(
        f"VES tracks: {len(ves)}"
    )

    print(
        f"FS tracks: {len(fs)}"
    )

    print(
        f"VES exposure: "
        f"{exposure_ves:.3f} s"
    )

    print(
        f"FS exposure: "
        f"{exposure_fs:.3f} s"
    )

    print(
        f"Output directory: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()