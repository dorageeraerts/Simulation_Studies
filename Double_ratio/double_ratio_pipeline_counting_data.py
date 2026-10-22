#!/usr/bin/env python3

"""
End-to-end MURAVES pipeline:

    ROOT data
        -> chi2 selection
        -> angular binning
        -> Working Point lookup
        -> run duration extraction
        -> selected-track Parquet files
        -> simulated PUMAS/Mulder transmission
        -> measured transmission
        -> double ratio
        -> plots

The selected-track Parquet files contain:

    run_number
    detector
    data_type
    working_point

    run_start_timestamp
    run_end_timestamp
    run_duration_seconds

    Phi_3p
    Theta_3p
    Phi_4p
    Theta_4p

The individual event timestamp is NOT saved.

Run duration is calculated from the ROOT timestamp branch as:

    duration_seconds = (tmax - tmin) / 1000

where timestamp is assumed to be in milliseconds.

The ROOT files used for timing and track selection are explicitly:

    MURAVES_AnalyzedData_run<N>.root

and NOT:

    MURAVES_miniRunTree_run<N>.root
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import uproot
import pandas as pd


# ============================================================
# Run ranges
# ============================================================

RUN_RANGES = {

    "NERO": {

        "VES": [
            (2500, 5600),
            (7894, 11990),
        ],

        "FS": [
            (5636, 7893),
        ],

    },

    "ROSSO": {

        "VES": [
            (4000, 10060),
            (13458, 14792),
        ],

        "FS": [
            (10061, 13457),
        ],

    },

    "BLU": {

        "VES": [
            (2800, 8818),
        ],

        "FS": [
            (8819, 9925),
        ],

    },

}


CHI2_XY = "BestTrack_4p_ChiSquare_xy"
CHI2_XZ = "BestTrack_4p_ChiSquare_xz"

TREE_NAME = "AnalyzedData"

DEFAULT_DATA_DIR_TEMPLATE = (
    "/pnfs/iihe/muraves/muraves_DATA/RECONSTRUCTED/{detector}/v1"
)

DEFAULT_SLOWCONTROL_DIR = (
    "/pnfs/iihe/muraves/muraves_DATA/DECOMPRESSED"
)

FILENAME_TEMPLATE = (
    "MURAVES_AnalyzedData_run{run}.root"
)

SLOWCONTROL_TEMPLATE = (
    "SLOWCONTROL_run{run}"
)


# ============================================================
# PUMAS / Mulder flux-output file columns
# ============================================================

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
# Run-number parsing
# ============================================================

def parse_runs(run_string):

    runs = set()

    for part in run_string.split(","):

        part = part.strip()

        if not part:
            continue

        if "-" in part:

            pieces = part.split("-")

            if len(pieces) != 2:
                raise ValueError(
                    f"Invalid run range '{part}'. "
                    "Use e.g. 2-5."
                )

            start, end = (
                int(pieces[0]),
                int(pieces[1]),
            )

            if start > end:
                raise ValueError(
                    f"Invalid run range '{part}': "
                    "start must be <= end."
                )

            runs.update(
                range(start, end + 1)
            )

        else:

            runs.add(int(part))

    if not runs:
        raise ValueError(
            "No valid run numbers were provided."
        )

    return sorted(runs)


def iter_run_numbers(detector: str, flag: str):

    for start, end in RUN_RANGES[detector][flag]:

        yield from range(
            start,
            end + 1,
        )


def make_edges(
    min_val,
    max_val,
    bin_size,
):

    """
    Build bin edges of the given size,
    covering [min_val, max_val].
    """

    n_bins = max(
        1,
        round(
            (max_val - min_val)
            / bin_size
        ),
    )

    return np.linspace(
        min_val,
        max_val,
        n_bins + 1,
    )


# ============================================================
# Slow-control / Working Point
# ============================================================

def get_working_point(
    slowcontrol_dir,
    detector,
    run,
):

    """
    Read the Working Point for a run from:

        <slowcontrol_dir>/<detector>/SLOWCONTROL_run<run>

    The Working Point is field [5] of the first valid line.

    Example:

        11267 1620208547823 0101100 19.2 97.4 25 ...

    -> Working Point = 25
    """

    slowcontrol_file = (
        Path(slowcontrol_dir)
        / detector
        / SLOWCONTROL_TEMPLATE.format(
            run=run
        )
    )

    if not slowcontrol_file.is_file():

        print(
            f"  [warn] slow-control file not found: "
            f"{slowcontrol_file}"
        )

        return None

    try:

        with open(
            slowcontrol_file,
            "r",
            errors="replace",
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                fields = line.split()

                if len(fields) <= 5:

                    print(
                        f"  [warn] malformed slow-control line "
                        f"in {slowcontrol_file}"
                    )

                    return None

                try:

                    working_point = float(
                        fields[5]
                    )

                except ValueError:

                    print(
                        f"  [warn] invalid Working Point "
                        f"in {slowcontrol_file}: "
                        f"{fields[5]}"
                    )

                    return None

                return working_point

    except Exception as exc:

        print(
            f"  [warn] could not read slow-control file "
            f"{slowcontrol_file}: {exc}"
        )

        return None

    print(
        f"  [warn] no valid lines in slow-control file: "
        f"{slowcontrol_file}"
    )

    return None


# ============================================================
# ROOT file lookup
# ============================================================

def find_root_file(
    data_dir,
    run_number,
):
    """
    Find the exact AnalyzedData ROOT file.

    Expected filename:

        MURAVES_AnalyzedData_run<N>.root

    This deliberately does NOT search for arbitrary ROOT files,
    because both AnalyzedData and miniRunTree files can exist.
    """

    data_dir = Path(data_dir)

    root_file = (
        data_dir
        / FILENAME_TEMPLATE.format(
            run=run_number
        )
    )

    if root_file.is_file():
        return root_file

    print(
        f"  [warn] AnalyzedData ROOT file not found "
        f"for run {run_number}:"
    )

    print(
        f"         {root_file}"
    )

    return None


# ============================================================
# ROOT reading + selection + binning
# ============================================================

def process_run(
    file_path,
    tree_name,
    phi_key,
    theta_key,
    threshold,
    az_edges,
    el_edges,
    run_number,
    detector,
    data_type,
    working_point,
):

    """
    Read one run's needed branches, apply the chi2 cut,
    calculate the run duration, and return:

        counts
        duration_seconds
        selected_df
        n_selected

    selected_df contains the exact tracks saved to Parquet.

    The Parquet rows contain run-level timing metadata,
    but NOT the individual event timestamp.
    """

    branches = [

        phi_key,
        theta_key,

        CHI2_XY,
        CHI2_XZ,

        "timestamp",

        "Phi_3p",
        "Theta_3p",

        "Phi_4p",
        "Theta_4p",

    ]

    # --------------------------------------------------------
    # Read ROOT
    # --------------------------------------------------------

    try:

        with uproot.open(file_path) as f:

            if tree_name not in f:

                print(
                    f"  [warn] tree '{tree_name}' "
                    f"not found, skipping"
                )

                return None

            tree = f[tree_name]

            missing = (
                set(branches)
                - set(tree.keys())
            )

            if missing:

                print(
                    f"  [warn] missing branches "
                    f"{missing}, skipping"
                )

                return None

            arrays = tree.arrays(
                branches,
                library="np",
            )

    except Exception as exc:

        print(
            f"  [warn] could not read "
            f"{file_path.name}: {exc}"
        )

        return None

    n_total = len(
        arrays[CHI2_XY]
    )

    if n_total == 0:

        print(
            "  [warn] empty tree, skipping"
        )

        return None

    # --------------------------------------------------------
    # Run duration
    # --------------------------------------------------------

    timestamps = np.asarray(
        arrays["timestamp"],
        dtype=float,
    )

    valid_timestamps = np.isfinite(
        timestamps
    )

    if not np.any(valid_timestamps):

        print(
            "  [warn] no valid timestamps, skipping"
        )

        return None

    tmin = np.min(
        timestamps[
            valid_timestamps
        ]
    )

    tmax = np.max(
        timestamps[
            valid_timestamps
        ]
    )

    # Timestamp is in milliseconds
    duration_seconds = (
        tmax - tmin
    ) / 1000.0

    if duration_seconds <= 0:

        print(
            f"  [warn] invalid run duration: "
            f"{duration_seconds:.3f} s, skipping"
        )

        return None

    print(
        f"  Run start timestamp: {tmin:.0f}"
    )

    print(
        f"  Run end timestamp:   {tmax:.0f}"
    )

    print(
        f"  Duration: "
        f"{duration_seconds:.3f} s "
        f"({duration_seconds / 60:.2f} min)"
    )

    # --------------------------------------------------------
    # Chi2 selection
    # --------------------------------------------------------

    chi2_xy = np.asarray(
        arrays[CHI2_XY],
        dtype=float,
    )

    chi2_xz = np.asarray(
        arrays[CHI2_XZ],
        dtype=float,
    )

    quality_mask = (

        np.isfinite(chi2_xy)

        & np.isfinite(chi2_xz)

        & (chi2_xy >= 0)

        & (chi2_xy < threshold)

        & (chi2_xz >= 0)

        & (chi2_xz < threshold)

    )

    # --------------------------------------------------------
    # Angular histogram
    # --------------------------------------------------------

    phi = arrays[phi_key][
        quality_mask
    ]

    theta = arrays[theta_key][
        quality_mask
    ]

    finite_angle_mask = (

        np.isfinite(phi)

        & np.isfinite(theta)

    )

    phi = phi[
        finite_angle_mask
    ]

    theta = theta[
        finite_angle_mask
    ]

    print(
        f"  Selected "
        f"{quality_mask.sum()} / "
        f"{n_total} entries "
        f"({len(phi)} finite after "
        f"angle check)"
    )

    counts, _, _ = np.histogram2d(
        phi,
        theta,
        bins=[
            az_edges,
            el_edges,
        ],
    )

    # --------------------------------------------------------
    # Build exact DataFrame saved to Parquet
    # --------------------------------------------------------

    parquet_mask = (
        quality_mask.copy()
    )

    parquet_mask &= np.isfinite(
        arrays["Phi_3p"]
    )

    parquet_mask &= np.isfinite(
        arrays["Theta_3p"]
    )

    parquet_mask &= np.isfinite(
        arrays["Phi_4p"]
    )

    parquet_mask &= np.isfinite(
        arrays["Theta_4p"]
    )

    n_parquet = np.count_nonzero(
        parquet_mask
    )

    selected_df = pd.DataFrame({

        "run_number": np.full(
            n_parquet,
            run_number,
            dtype=np.int64,
        ),

        "detector": np.full(
            n_parquet,
            detector,
            dtype=object,
        ),

        "data_type": np.full(
            n_parquet,
            data_type,
            dtype=object,
        ),

        "working_point": np.full(
            n_parquet,
            working_point,
            dtype=float,
        ),

        # Run-level timing metadata
        "run_start_timestamp": np.full(
            n_parquet,
            tmin,
            dtype=float,
        ),

        "run_end_timestamp": np.full(
            n_parquet,
            tmax,
            dtype=float,
        ),

        "run_duration_seconds": np.full(
            n_parquet,
            duration_seconds,
            dtype=float,
        ),

        "Phi_3p": arrays["Phi_3p"][
            parquet_mask
        ],

        "Theta_3p": arrays["Theta_3p"][
            parquet_mask
        ],

        "Phi_4p": arrays["Phi_4p"][
            parquet_mask
        ],

        "Theta_4p": arrays["Theta_4p"][
            parquet_mask
        ],

    })

    # This is deliberately the number
    # of rows saved to Parquet.
    n_selected = len(
        selected_df
    )

    print(
        f"  Tracks saved to Parquet: "
        f"{n_selected}"
    )

    return (
        counts,
        duration_seconds,
        selected_df,
        n_selected,
    )


# ============================================================
# Count tracks per Working Point
# ============================================================

def count_tracks_per_working_point(
    selected_df,
):

    """
    Count the rows of the selected-track DataFrame by:

        detector / data_type / working_point

    The count is exactly the number of rows saved
    to the corresponding Parquet file.
    """

    if selected_df.empty:

        return pd.DataFrame(
            columns=[
                "Detector",
                "DataType",
                "WorkingPoint",
                "Chi2SelectedTracks",
            ]
        )

    counts_df = (

        selected_df

        .groupby(
            [
                "detector",
                "data_type",
                "working_point",
            ],
            dropna=False,
        )

        .size()

        .reset_index(
            name="Chi2SelectedTracks"
        )

    )

    counts_df = counts_df.rename(
        columns={
            "detector": "Detector",
            "data_type": "DataType",
            "working_point": "WorkingPoint",
        }
    )

    counts_df = (
        counts_df
        .sort_values(
            [
                "Detector",
                "DataType",
                "WorkingPoint",
            ]
        )
        .reset_index(drop=True)
    )

    return counts_df


# ============================================================
# Pipeline over runs
# ============================================================

def run_pipeline(
    data_dir,
    slowcontrol_dir,
    detector,
    flag,
    threshold,
    tree_name,
    key_pair,
    az_edges,
    el_edges,
    runs=None,
):

    phi_key = (
        f"Phi_{key_pair}"
    )

    theta_key = (
        f"Theta_{key_pair}"
    )

    total_counts = np.zeros(
        (
            len(az_edges) - 1,
            len(el_edges) - 1,
        )
    )

    total_duration_seconds = 0.0

    selected_tracks = []

    n_found = 0
    n_missing = 0
    n_failed = 0
    n_slowcontrol_missing = 0

    runs_to_process = (
        iter_run_numbers(
            detector,
            flag,
        )
        if runs is None
        else runs
    )

    for run in runs_to_process:

        # ----------------------------------------------------
        # ROOT file
        # ----------------------------------------------------

        file_path = find_root_file(
            data_dir,
            run,
        )

        if file_path is None:

            print(
                f"Run {run} not available, skipping"
            )

            n_missing += 1

            continue

        n_found += 1

        print(
            f"\nProcessing run {run}: "
            f"{file_path.name}"
        )

        # ----------------------------------------------------
        # Working Point
        # ----------------------------------------------------

        working_point = get_working_point(
            slowcontrol_dir,
            detector,
            run,
        )

        if working_point is None:

            print(
                f"  [warn] Working Point unavailable "
                f"for run {run}, skipping"
            )

            n_slowcontrol_missing += 1

            continue

        print(
            f"  Working Point: "
            f"{working_point:g}"
        )

        # ----------------------------------------------------
        # Process ROOT
        # ----------------------------------------------------

        result = process_run(
            file_path,
            tree_name,
            phi_key,
            theta_key,
            threshold,
            az_edges,
            el_edges,
            run_number=run,
            detector=detector,
            data_type=flag,
            working_point=working_point,
        )

        if result is None:

            n_failed += 1

            continue

        (
            counts,
            duration_seconds,
            selected_df,
            n_selected,
        ) = result

        total_counts += counts

        total_duration_seconds += (
            duration_seconds
        )

        if len(selected_df) > 0:

            selected_tracks.append(
                selected_df
            )

    # --------------------------------------------------------
    # Combine all selected tracks
    # --------------------------------------------------------

    if selected_tracks:

        selected_tracks_df = pd.concat(
            selected_tracks,
            ignore_index=True,
        )

    else:

        selected_tracks_df = pd.DataFrame(
            columns=[
                "run_number",
                "detector",
                "data_type",
                "working_point",
                "run_start_timestamp",
                "run_end_timestamp",
                "run_duration_seconds",
                "Phi_3p",
                "Theta_3p",
                "Phi_4p",
                "Theta_4p",
            ]
        )

    # --------------------------------------------------------
    # Track rate
    # --------------------------------------------------------

    if total_duration_seconds > 0:

        track_rate = (
            total_counts
            / total_duration_seconds
        )

    else:

        track_rate = np.zeros_like(
            total_counts
        )

    # --------------------------------------------------------
    # WP counting
    # --------------------------------------------------------

    wp_counts = (
        count_tracks_per_working_point(
            selected_tracks_df
        )
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        f"\nRuns found on disk: "
        f"{n_found}"
    )

    print(
        f"Runs missing (skipped): "
        f"{n_missing}"
    )

    print(
        f"Runs with missing slow-control: "
        f"{n_slowcontrol_missing}"
    )

    print(
        f"Runs unsuccessfully processed: "
        f"{n_failed}"
    )

    print(
        f"Total entries in map: "
        f"{int(total_counts.sum())}"
    )

    print(
        f"Total duration: "
        f"{total_duration_seconds:.3f} s "
        f"({total_duration_seconds / 60:.2f} min)"
    )

    print(
        f"Total selected tracks: "
        f"{len(selected_tracks_df)}"
    )

    if total_duration_seconds > 0:

        print(
            f"Average track rate: "
            f"{total_counts.sum() / total_duration_seconds:.4f} "
            f"tracks/s"
        )

    else:

        print(
            "Average track rate: undefined"
        )

    print(
        f"Selected tracks saved for Parquet: "
        f"{len(selected_tracks_df)}"
    )

    # --------------------------------------------------------
    # Print WP table
    # --------------------------------------------------------

    if not wp_counts.empty:

        print(
            "\nChi2-selected tracks per "
            "Working Point:"
        )

        print(
            wp_counts.to_string(
                index=False
            )
        )

    return (
        total_counts,
        total_duration_seconds,
        track_rate,
        selected_tracks_df,
        wp_counts,
    )


# ============================================================
# PUMAS / Mulder reference loading
# ============================================================

def parse_density(path):

    m = re.search(
        r"rho(\d+)",
        os.path.basename(path),
    )

    if not m:
        return None

    return int(m.group(1)) / 100.0


def load_flux_output(
    path,
    label=None,
    azimuth_offset=0.0,
):

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
            f"{path}: expected "
            f"{len(FLUX_COLUMNS)} columns "
            f"({', '.join(FLUX_COLUMNS)}), "
            f"found {data.shape[1]}"
        )

    out = {
        name: data[:, i]
        for i, name in enumerate(
            FLUX_COLUMNS
        )
    }

    out["azimuth"] = (
        out["azimuth"]
        + azimuth_offset
    )

    out["label"] = (
        label
        if label is not None
        else os.path.basename(path)
    )

    out["path"] = path

    out["density"] = parse_density(
        path
    )

    return out


def rebin_transmission(
    ref,
    az_edges,
    el_edges,
):

    az = np.asarray(
        ref["azimuth"],
        dtype=float,
    )

    el = np.asarray(
        ref["elevation"],
        dtype=float,
    )

    transmission = np.asarray(
        ref["transmission"],
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

    n_az = (
        len(az_edges) - 1
    )

    n_el = (
        len(el_edges) - 1
    )

    t_sum = np.zeros(
        (
            n_az,
            n_el,
        )
    )

    t_count = np.zeros(
        (
            n_az,
            n_el,
        ),
        dtype=int,
    )

    np.add.at(
        t_sum,
        (
            az_index,
            el_index,
        ),
        transmission,
    )

    np.add.at(
        t_count,
        (
            az_index,
            el_index,
        ),
        1,
    )

    t_binned = np.zeros(
        (
            n_az,
            n_el,
        )
    )

    populated = (
        t_count > 0
    )

    t_binned[populated] = (
        t_sum[populated]
        / t_count[populated]
    )

    print(
        f"\nReference: "
        f"{ref['label']}"
    )

    print(
        f"  Reference points used: "
        f"{len(transmission)}"
    )

    print(
        f"  Populated output bins: "
        f"{np.count_nonzero(populated)} / "
        f"{n_az * n_el}"
    )

    n_empty = np.count_nonzero(
        ~populated
    )

    if n_empty:

        print(
            f"  [WARNING] {n_empty} bins have "
            f"no simulated transmission points"
        )

    return t_binned


# ============================================================
# Plotting
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
    cmap="viridis",
):

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    mesh = ax.pcolormesh(
        az_edges,
        el_edges,
        values.T,
        shading="auto",
        cmap=cmap,
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

    if len(az_edges) <= 41:

        ax.set_xticks(
            az_edges
        )

    if len(el_edges) <= 41:

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

        vmax = (
            finite.max()
            if finite.size
            else 1.0
        )

        for i, ac in enumerate(
            az_centers
        ):

            for j, ec in enumerate(
                el_centers
            ):

                v = values[i, j]

                if v == 0:
                    continue

                color = (
                    "white"
                    if v < 0.6 * vmax
                    else "black"
                )

                ax.text(
                    ac,
                    ec,
                    fmt.format(v),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=color,
                )

    plt.tight_layout()

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
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "MURAVES pipeline: "
            "read ROOT -> select -> bin -> "
            "calculate rates -> compare with "
            "PUMAS/Mulder -> double ratio."
        )
    )

    parser.add_argument(
        "detector",
        choices=[
            "ROSSO",
            "BLU",
            "NERO",
        ],
    )

    # --------------------------------------------------------
    # Data selection
    # --------------------------------------------------------

    parser.add_argument(
        "--threshold-ves",
        type=float,
        default=2,
        help=(
            "Chi-square cut threshold for VES "
            "(default: 2)."
        ),
    )

    parser.add_argument(
        "--threshold-fs",
        type=float,
        default=2,
        help=(
            "Chi-square cut threshold for FS "
            "(default: 2)."
        ),
    )

    parser.add_argument(
        "--runs-ves",
        type=str,
        default=None,
        help=(
            "VES runs to process, e.g. 2500-2600. "
            "Omit for all predefined VES runs."
        ),
    )

    parser.add_argument(
        "--runs-fs",
        type=str,
        default=None,
        help=(
            "FS runs to process, e.g. 5636-5700. "
            "Omit for all predefined FS runs."
        ),
    )

    parser.add_argument(
        "--data-dir-ves",
        type=str,
        default=None,
        help=(
            "Directory containing VES ROOT files."
        ),
    )

    parser.add_argument(
        "--data-dir-fs",
        type=str,
        default=None,
        help=(
            "Directory containing FS ROOT files."
        ),
    )

    # --------------------------------------------------------
    # Slow control
    # --------------------------------------------------------

    parser.add_argument(
        "--slowcontrol-dir",
        type=str,
        default=DEFAULT_SLOWCONTROL_DIR,
        help=(
            "Base directory containing detector "
            "slow-control directories."
        ),
    )

    parser.add_argument(
        "--tree-name",
        type=str,
        default=TREE_NAME,
    )

    parser.add_argument(
        "--key-pair",
        type=str,
        default="4p",
        choices=[
            "3p",
            "4p",
        ],
        help=(
            "Use Phi_3p/Theta_3p or "
            "Phi_4p/Theta_4p for histogramming."
        ),
    )

    # --------------------------------------------------------
    # Binning
    # --------------------------------------------------------

    parser.add_argument(
        "--az-min",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--az-max",
        type=float,
        default=360.0,
    )

    parser.add_argument(
        "--az-bin-size",
        type=float,
        default=5.0,
        help=(
            "Azimuth bin width in degrees."
        ),
    )

    parser.add_argument(
        "--el-min",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--el-max",
        type=float,
        default=90.0,
    )

    parser.add_argument(
        "--el-bin-size",
        type=float,
        default=2.0,
        help=(
            "Elevation bin width in degrees."
        ),
    )

    # --------------------------------------------------------
    # PUMAS / Mulder reference
    # --------------------------------------------------------

    parser.add_argument(
        "--ref-dir",
        type=str,
        required=True,
        help=(
            "Directory containing the "
            "PUMAS/Mulder flux-output file."
        ),
    )

    parser.add_argument(
        "--ref-file",
        type=str,
        required=True,
        help=(
            "Filename of the PUMAS/Mulder "
            "flux-output file."
        ),
    )

    parser.add_argument(
        "--azimuth-offset",
        type=float,
        default=0.0,
        help=(
            "Degrees added to reference azimuth."
        ),
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory to write plots and Parquet "
            "(default: --data-dir-ves)."
        ),
    )

    parser.add_argument(
        "--flag",
        type=str,
        default=None,
        help=(
            "Tag used for output subfolder "
            "and filenames."
        ),
    )

    parser.add_argument(
        "--annotate",
        action="store_true",
        help=(
            "Write bin values as text on "
            "the double-ratio plot."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Directories
    # --------------------------------------------------------

    default_dir = (
        DEFAULT_DATA_DIR_TEMPLATE.format(
            detector=args.detector
        )
    )

    data_dir_ves = (
        Path(args.data_dir_ves)
        if args.data_dir_ves
        else Path(default_dir)
    )

    data_dir_fs = (
        Path(args.data_dir_fs)
        if args.data_dir_fs
        else Path(default_dir)
    )

    slowcontrol_dir = Path(
        args.slowcontrol_dir
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else data_dir_ves
    )

    flag = (
        args.flag
        if args.flag
        else args.detector
    )

    # --------------------------------------------------------
    # Runs
    # --------------------------------------------------------

    runs_ves = (
        parse_runs(args.runs_ves)
        if args.runs_ves is not None
        else None
    )

    runs_fs = (
        parse_runs(args.runs_fs)
        if args.runs_fs is not None
        else None
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
        f"Detector: {args.detector} | "
        f"Key pair: {args.key_pair} | "
        f"Flag: {flag}"
    )

    print(
        f"Az bins: {len(az_edges) - 1} "
        f"of {args.az_bin_size} deg | "
        f"El bins: {len(el_edges) - 1} "
        f"of {args.el_bin_size} deg"
    )

    print(
        f"Slow-control directory: "
        f"{slowcontrol_dir}\n"
    )

    # ========================================================
    # VES
    # ========================================================

    print(
        f"=== VES (chi2 < "
        f"{args.threshold_ves}) ==="
    )

    print(
        "Requested VES runs: "
        f"{runs_ves if runs_ves is not None else 'all predefined runs'}"
    )

    print(
        f"VES data directory: "
        f"{data_dir_ves}"
    )

    (
        N_VES,
        duration_VES,
        rate_VES,
        tracks_VES,
        wp_counts_VES,
    ) = run_pipeline(
        data_dir_ves,
        slowcontrol_dir,
        args.detector,
        "VES",
        args.threshold_ves,
        args.tree_name,
        args.key_pair,
        az_edges,
        el_edges,
        runs=runs_ves,
    )

    # ========================================================
    # FS
    # ========================================================

    print(
        f"\n=== FS (chi2 < "
        f"{args.threshold_fs}) ==="
    )

    print(
        "Requested FS runs: "
        f"{runs_fs if runs_fs is not None else 'all predefined runs'}"
    )

    print(
        f"FS data directory: "
        f"{data_dir_fs}"
    )

    (
        N_FS,
        duration_FS,
        rate_FS,
        tracks_FS,
        wp_counts_FS,
    ) = run_pipeline(
        data_dir_fs,
        slowcontrol_dir,
        args.detector,
        "FS",
        args.threshold_fs,
        args.tree_name,
        args.key_pair,
        az_edges,
        el_edges,
        runs=runs_fs,
    )

    # ========================================================
    # VES / FS measured transmission
    # ========================================================

    VES_FS = np.divide(
        rate_VES,
        rate_FS,
        out=np.zeros_like(rate_VES),
        where=rate_FS > 0,
    )

    print(
        "\nMeasured transmission calculated as:"
    )

    print(
        "    T_measured = rate_VES / rate_FS"
    )

    print(
        f"    VES total duration = "
        f"{duration_VES:.3f} s"
    )

    print(
        f"    FS total duration  = "
        f"{duration_FS:.3f} s"
    )

    # ========================================================
    # Simulation
    # ========================================================

    ref_path = (
        Path(args.ref_dir)
        / args.ref_file
    )

    print(
        "\nLoading simulated reference:"
    )

    print(
        f"  {ref_path}"
    )

    ref = load_flux_output(
        ref_path,
        azimuth_offset=args.azimuth_offset,
    )

    if ref["density"] is not None:

        print(
            f"  Parsed density tag: "
            f"rho = {ref['density']:.2f} g/cm^3"
        )

    transmission_sim = (
        rebin_transmission(
            ref,
            az_edges,
            el_edges,
        )
    )

    # ========================================================
    # Double ratio
    # ========================================================

    double_ratio = np.divide(
        VES_FS,
        transmission_sim,
        out=np.zeros_like(VES_FS),
        where=transmission_sim > 0,
    )

    # ========================================================
    # Output directory
    # ========================================================

    output_subdir = (
        output_dir
        / flag
    )

    output_subdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Save selected tracks
    # ========================================================

    ves_chi2_tag = (
        f"chi2{args.threshold_ves:g}"
        .replace(".", "p")
    )

    fs_chi2_tag = (
        f"chi2{args.threshold_fs:g}"
        .replace(".", "p")
    )

    ves_parquet = (
        output_subdir
        / (
            f"SelectedTracks_VES_"
            f"{ves_chi2_tag}_{flag}.parquet"
        )
    )

    fs_parquet = (
        output_subdir
        / (
            f"SelectedTracks_FS_"
            f"{fs_chi2_tag}_{flag}.parquet"
        )
    )

    tracks_VES.to_parquet(
        ves_parquet,
        index=False,
    )

    tracks_FS.to_parquet(
        fs_parquet,
        index=False,
    )

    print(
        f"\nSaved VES selected tracks: "
        f"{ves_parquet}"
    )

    print(
        f"Saved FS selected tracks: "
        f"{fs_parquet}"
    )

    # ========================================================
    # Save combined WP track-count table
    # ========================================================

    wp_counts = pd.concat(
        [
            wp_counts_VES,
            wp_counts_FS,
        ],
        ignore_index=True,
    )

    if not wp_counts.empty:

        wp_counts = (
            wp_counts
            .sort_values(
                [
                    "Detector",
                    "DataType",
                    "WorkingPoint",
                ]
            )
            .reset_index(drop=True)
        )

    wp_counts_csv = (
        output_subdir
        / f"TrackChi2Counts_perWP_{flag}.csv"
    )

    wp_counts_txt = (
        output_subdir
        / f"TrackChi2Counts_perWP_{flag}.txt"
    )

    wp_counts.to_csv(
        wp_counts_csv,
        index=False,
    )

    with open(
        wp_counts_txt,
        "w",
    ) as f:

        f.write(
            f"Detector: {args.detector}\n"
        )

        f.write(
            "Chi2-selected tracks per "
            "Working Point\n"
        )

        f.write(
            "========================================\n\n"
        )

        if wp_counts.empty:

            f.write(
                "No selected tracks.\n"
            )

        else:

            f.write(
                wp_counts.to_string(
                    index=False
                )
            )

            f.write("\n")

    print(
        f"\nSaved WP track counts: "
        f"{wp_counts_csv}"
    )

    print(
        f"Saved WP track-count summary: "
        f"{wp_counts_txt}"
    )

    print(
        "\n=== Final chi2-selected track counts "
        "per WP ==="
    )

    if wp_counts.empty:

        print(
            "No selected tracks."
        )

    else:

        print(
            wp_counts.to_string(
                index=False
            )
        )

    # ========================================================
    # Save angular maps as CSV
    # ========================================================

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

    def save_map_csv(
        values,
        filename,
    ):

        rows = []

        for i, az in enumerate(
            az_centers
        ):

            for j, el in enumerate(
                el_centers
            ):

                rows.append(
                    {
                        "azimuth": az,
                        "elevation": el,
                        "value": values[i, j],
                    }
                )

        pd.DataFrame(rows).to_csv(
            output_subdir / filename,
            index=False,
        )

        print(
            f"Saved: "
            f"{output_subdir / filename}"
        )

    save_map_csv(
        rate_FS,
        f"FS_track_rate_{flag}.csv",
    )

    save_map_csv(
        rate_VES,
        f"VES_track_rate_{flag}.csv",
    )

    save_map_csv(
        VES_FS,
        f"MeasuredTransmission_{flag}.csv",
    )

    save_map_csv(
        transmission_sim,
        f"SimulatedTransmission_{flag}.csv",
    )

    save_map_csv(
        double_ratio,
        f"DoubleRatio_{flag}.csv",
    )

    # ========================================================
    # Plots
    # ========================================================

    rho_tag = (
        f", rho={ref['density']:.2f} g/cm^3"
        if ref["density"] is not None
        else ""
    )

    bs_az = args.az_bin_size
    bs_el = args.el_bin_size

    make_plot(
        rate_FS,
        az_edges,
        el_edges,
        (
            f"Freesky data — "
            f"{bs_az:g}x{bs_el:g} deg bins — "
            f"{args.key_pair}"
        ),
        "Track rate [s$^{-1}$]",
        output_subdir
        / f"FS_track_rate_{flag}.png",
    )

    make_plot(
        rate_VES,
        az_edges,
        el_edges,
        (
            f"Vesuvius data — "
            f"{bs_az:g}x{bs_el:g} deg bins — "
            f"{args.key_pair}"
        ),
        "Track rate [s$^{-1}$]",
        output_subdir
        / f"VES_track_rate_{flag}.png",
    )

    make_plot(
        VES_FS,
        az_edges,
        el_edges,
        (
            f"VES / FS data ratio — "
            f"{bs_az:g}x{bs_el:g} deg bins — "
            f"{args.key_pair}"
        ),
        r"$R_{VES}/R_{FS}$",
        output_subdir
        / f"MeasuredTransmission_{flag}.png",
    )

    make_plot(
        transmission_sim,
        az_edges,
        el_edges,
        (
            f"Simulated transmission "
            f"({ref['label']})"
            f"{rho_tag}"
        ),
        "Transmission",
        output_subdir
        / f"SimulatedTransmission_{flag}.png",
    )

    make_plot(
        double_ratio,
        az_edges,
        el_edges,
        (
            f"Double ratio — "
            f"{bs_az:g}x{bs_el:g} deg bins"
            f"{rho_tag}"
        ),
        r"$D=(R_{VES}/R_{FS})/T_{sim}$",
        output_subdir
        / f"DoubleRatio_{flag}.png",
        annotate=args.annotate,
    )

    # ========================================================
    # Final summary
    # ========================================================

    print("\n========================================")
    print("FINAL SUMMARY")
    print("========================================")

    print(
        f"VES selected tracks: "
        f"{len(tracks_VES)}"
    )

    print(
        f"FS selected tracks:  "
        f"{len(tracks_FS)}"
    )

    print(
        f"VES exposure: "
        f"{duration_VES:.3f} s "
        f"({duration_VES / 3600:.3f} h)"
    )

    print(
        f"FS exposure:  "
        f"{duration_FS:.3f} s "
        f"({duration_FS / 3600:.3f} h)"
    )

    if duration_VES > 0:

        print(
            f"VES average rate: "
            f"{len(tracks_VES) / duration_VES:.6f} tracks/s"
        )

    if duration_FS > 0:

        print(
            f"FS average rate: "
            f"{len(tracks_FS) / duration_FS:.6f} tracks/s"
        )

    print(
        f"\nOutput directory:"
        f"\n  {output_subdir}"
    )

    print(
        "\nParquet columns:"
    )

    print(
        "  run_number"
    )

    print(
        "  detector"
    )

    print(
        "  data_type"
    )

    print(
        "  working_point"
    )

    print(
        "  run_start_timestamp"
    )

    print(
        "  run_end_timestamp"
    )

    print(
        "  run_duration_seconds"
    )

    print(
        "  Phi_3p"
    )

    print(
        "  Theta_3p"
    )

    print(
        "  Phi_4p"
    )

    print(
        "  Theta_4p"
    )

    print(
        "\nDone."
    )


if __name__ == "__main__":
    main()