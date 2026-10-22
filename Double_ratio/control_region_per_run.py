#!/usr/bin/env python3
"""
control_region_per_run.py

Plot the number of tracks and track rate in the control region for every run.

For each run:
    - identify VES / FS
    - read the Working Point from slow-control
    - select the angular control region
    - count tracks with no chi2 cut
    - count tracks for integer chi2 cuts 1 ... 10
    - calculate the corresponding track rate using the run duration

Plots:
    One figure for each chi2 selection:
        no chi2 cut
        chi2 < 1
        chi2 < 2
        ...
        chi2 < 10

Each figure contains:
    left : number of tracks in control region
    right: track rate [Hz]

Plot encoding:
    colour = Working Point
    marker = VES / FS
    x      = run number

The individual runs are deliberately shown as scatter points because
there are O(10,000) runs.

Selection convention:
    CHI2_XY < chi2_cut AND CHI2_XZ < chi2_cut

Equivalently:
    max(CHI2_XY, CHI2_XZ) < chi2_cut

"No chi2 cut" means that no upper chi2 threshold is applied, although
the chi2 values must still be finite and non-negative, following the
selection convention of the original script.

Example:

    python control_region_per_run.py --detector ROSSO

    python control_region_per_run.py --detectors NERO ROSSO BLU

    python control_region_per_run.py \
        --detector ROSSO \
        --ves-ranges 4000-4010 5000-5010 \
        --fs-ranges 10061-10070
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import uproot


# ============================================================
# USER CONFIGURATION
# ============================================================

# Angular control region
#MIN_AZ, MAX_AZ = 172.0, 188.0
#MIN_EL, MAX_EL = 25.0, 32.0
MIN_AZ, MAX_AZ = 170.0, 190.0
MIN_EL, MAX_EL = 12.0, 18.0

# Integer chi2 cuts to plot
CHI2_CUTS = list(range(1, 11))

# Paths
DATA_DIR_TEMPLATE = (
    "/pnfs/iihe/muraves/muraves_DATA/RECONSTRUCTED/{detector}/v1"
)
SLOWCONTROL_DIR = "/pnfs/iihe/muraves/muraves_DATA/DECOMPRESSED"

OUTPUT_DIR = Path("./muraves_control_region_per_run")

# ROOT configuration
TREE_NAME = "AnalyzedData"

CHI2_XY = "BestTrack_4p_ChiSquare_xy"
CHI2_XZ = "BestTrack_4p_ChiSquare_xz"

PHI_BRANCH = "Phi_3p"
THETA_BRANCH = "Theta_3p"

TIMESTAMP_BRANCH = "timestamp"

FILENAME_TEMPLATE = "MURAVES_AnalyzedData_run{run}.root"
SLOWCONTROL_TEMPLATE = "SLOWCONTROL_run{run}"


# ============================================================
# DEFAULT RUN RANGES
# ============================================================

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


# ============================================================
# HELPERS
# ============================================================

def parse_run_range(value):
    """Parse START-END, START:END, or a single run number."""

    value = value.strip()

    if "-" in value:
        parts = value.split("-", 1)
    elif ":" in value:
        parts = value.split(":", 1)
    else:
        try:
            run = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid run number: {value}"
            ) from exc

        return run, run

    try:
        start, stop = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid run range: {value}. "
            "Use START-END, e.g. 4000-4010."
        ) from exc

    if stop < start:
        raise argparse.ArgumentTypeError(
            f"Invalid run range: {value}: end must be >= start."
        )

    return start, stop


def parse_range_list(values):
    """Convert command-line ranges into a list of (start, stop) tuples."""

    if values is None:
        return None

    return [parse_run_range(value) for value in values]


def iter_run_numbers(run_ranges):
    """Yield run numbers from a list of (start, stop) ranges."""

    for start, stop in run_ranges:
        yield from range(start, stop + 1)


def get_working_point(slowcontrol_dir, detector, run):
    """
    Read the Working Point from:

        <slowcontrol_dir>/<detector>/SLOWCONTROL_run<run>

    This is field [5] of the first valid line.
    """

    slowcontrol_file = (
        Path(slowcontrol_dir)
        / detector
        / SLOWCONTROL_TEMPLATE.format(run=run)
    )

    if not slowcontrol_file.is_file():
        print(f"  [warn] slow-control file not found: {slowcontrol_file}")
        return None

    try:
        with open(slowcontrol_file, "r", errors="replace") as f:

            for line in f:
                line = line.strip()

                if not line:
                    continue

                fields = line.split()

                if len(fields) <= 5:
                    print(
                        f"  [warn] malformed slow-control line in "
                        f"{slowcontrol_file}"
                    )
                    return None

                try:
                    working_point = float(fields[5])
                except ValueError:
                    print(
                        f"  [warn] invalid Working Point in "
                        f"{slowcontrol_file}: {fields[5]}"
                    )
                    return None

                if not np.isfinite(working_point):
                    print(
                        f"  [warn] non-finite Working Point in "
                        f"{slowcontrol_file}: {fields[5]}"
                    )
                    return None

                if not working_point.is_integer():
                    print(
                        f"  [warn] non-integer Working Point in "
                        f"{slowcontrol_file}: {fields[5]}"
                    )
                    return None

                return int(working_point)

    except Exception as exc:
        print(
            f"  [warn] could not read {slowcontrol_file}: {exc}"
        )
        return None

    print(f"  [warn] no valid lines in {slowcontrol_file}")
    return None


def get_run_duration_seconds(timestamps):
    """
    Calculate run duration from timestamp.

    Timestamp is assumed to be in milliseconds.
    """

    timestamps = np.asarray(timestamps, dtype=float)

    valid = np.isfinite(timestamps)

    if not np.any(valid):
        return None

    tmin = np.min(timestamps[valid])
    tmax = np.max(timestamps[valid])

    duration = (tmax - tmin) / 1000.0

    if duration <= 0:
        return None

    return float(duration)


def read_run(file_path):
    """Read only branches required for this analysis."""

    branches = [
        PHI_BRANCH,
        THETA_BRANCH,
        CHI2_XY,
        CHI2_XZ,
        TIMESTAMP_BRANCH,
    ]

    try:
        with uproot.open(file_path) as f:

            if TREE_NAME not in f:
                print(f"  [warn] tree '{TREE_NAME}' not found")
                return None

            tree = f[TREE_NAME]

            missing = set(branches) - set(tree.keys())

            if missing:
                print(
                    f"  [warn] missing branches: {sorted(missing)}"
                )
                return None

            arrays = tree.arrays(branches, library="np")

    except Exception as exc:
        print(
            f"  [warn] could not read {file_path}: {exc}"
        )
        return None

    return arrays


def process_run(
    file_path,
    min_az,
    max_az,
    min_el,
    max_el,
):
    """
    Process one run.

    Returns:
        duration_seconds
        effective_chi2

    effective_chi2 is:
        max(CHI2_XY, CHI2_XZ)

    Only finite, non-negative chi2 values are accepted.
    """

    arrays = read_run(file_path)

    if arrays is None:
        return None

    timestamps = arrays[TIMESTAMP_BRANCH]

    duration_seconds = get_run_duration_seconds(timestamps)

    if duration_seconds is None:
        print("  [warn] invalid run duration")
        return None

    phi = np.asarray(arrays[PHI_BRANCH], dtype=float)
    theta = np.asarray(arrays[THETA_BRANCH], dtype=float)

    chi2_xy = np.asarray(arrays[CHI2_XY], dtype=float)
    chi2_xz = np.asarray(arrays[CHI2_XZ], dtype=float)

    # --------------------------------------------------------
    # Basic validity
    # --------------------------------------------------------

    valid = (
        np.isfinite(phi)
        & np.isfinite(theta)
        & np.isfinite(chi2_xy)
        & np.isfinite(chi2_xz)
        & (chi2_xy >= 0.0)
        & (chi2_xz >= 0.0)
    )

    # --------------------------------------------------------
    # Angular control region
    # --------------------------------------------------------

    control_region = (
        (phi >= min_az)
        & (phi <= max_az)
        & (theta >= min_el)
        & (theta <= max_el)
    )

    valid &= control_region

    if not np.any(valid):
        return duration_seconds, np.empty(0, dtype=float)

    # --------------------------------------------------------
    # Effective chi2
    # --------------------------------------------------------

    effective_chi2 = np.maximum(
        chi2_xy[valid],
        chi2_xz[valid],
    )

    if np.any(effective_chi2 < 0.0):
        raise RuntimeError(
            "Internal error: negative effective chi2 remained."
        )

    return duration_seconds, effective_chi2


def calculate_counts(effective_chi2):
    """
    Calculate:
        no-cut count
        counts for chi2 < 1 ... 10

    Strict inequality is used.
    """

    n_total = len(effective_chi2)

    counts = {
        None: n_total,
    }

    if n_total == 0:
        for cut in CHI2_CUTS:
            counts[cut] = 0

        return counts

    sorted_chi2 = np.sort(effective_chi2)

    for cut in CHI2_CUTS:
        # side='left' implements strict:
        # effective_chi2 < cut
        counts[cut] = int(
            np.searchsorted(
                sorted_chi2,
                cut,
                side="left",
            )
        )

    return counts


# ============================================================
# PLOTTING
# ============================================================

WP_COLORS = {
    6:  "#0072B2",
    10:  "#E69F00",
    15:  "#009E73",
    20:  "#D55E00",
    25:  "#CC79A7",
    30:  "#56B4E9",
    35:  "#F0E442",
}


def make_working_point_colours(working_points):
    """
    Return a high-contrast colour for each Working Point.
    """

    colours = {}

    for wp in sorted(working_points):

        if wp in WP_COLORS:
            colours[wp] = WP_COLORS[wp]

        else:
            # Fallback for unexpected WPs
            colours[wp] = "grey"

    return colours


def plot_one_cut(
    df,
    detector,
    chi2_cut,
    output_dir,
):
    """
    Make one figure for one chi2 selection.

    Left:
        number of tracks in control region

    Right:
        track rate [Hz]

    Plot encoding:
        colour = Working Point
        marker = VES / FS
    """

    # --------------------------------------------------------
    # Select the correct columns
    # --------------------------------------------------------

    if chi2_cut is None:

        title_cut = "No $\\chi^2$ cut"
        filename_cut = "no_chi2_cut"

        n_tracks_column = "n_tracks_no_cut"
        rate_column = "track_rate_hz_no_cut"

    else:

        title_cut = rf"$\chi^2 < {chi2_cut}$"
        filename_cut = f"chi2_lt_{chi2_cut}"

        n_tracks_column = f"n_tracks_chi2_{chi2_cut}"
        rate_column = f"track_rate_hz_chi2_{chi2_cut}"

    # --------------------------------------------------------
    # Check that columns exist
    # --------------------------------------------------------

    if n_tracks_column not in df.columns:
        raise KeyError(
            f"Column '{n_tracks_column}' not found in dataframe.\n"
            f"Available columns are:\n"
            f"{list(df.columns)}"
        )

    if rate_column not in df.columns:
        raise KeyError(
            f"Column '{rate_column}' not found in dataframe.\n"
            f"Available columns are:\n"
            f"{list(df.columns)}"
        )

    # --------------------------------------------------------
    # Working-point colours
    # --------------------------------------------------------

    colours = make_working_point_colours(
        df["working_point"].dropna().unique()
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1]},
    )

    ax_tracks = axes[0]
    ax_rate = axes[1]

    # --------------------------------------------------------
    # Plot VES and FS
    # --------------------------------------------------------

    for data_type, marker in [
        ("VES", "o"),
        ("FS", "s"),
    ]:

        subset = df[df["data_type"] == data_type]

        for wp in sorted(subset["working_point"].unique()):

            wp_subset = subset[
                subset["working_point"] == wp
            ]

            colour = colours[wp]

            # Number of tracks
            ax_tracks.scatter(
                wp_subset["run"],
                wp_subset[n_tracks_column],
                s=9,
                alpha=0.45,
                color=colour,
                marker=marker,
                linewidths=0,
            )

            # Track rate
            ax_rate.scatter(
                wp_subset["run"],
                wp_subset[rate_column],
                s=9,
                alpha=0.45,
                color=colour,
                marker=marker,
                linewidths=0,
            )

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ax_tracks.set_ylabel(
        "Tracks in control region"
    )

    ax_rate.set_ylabel(
        "Track rate [Hz]"
    )

    ax_rate.set_xlabel(
        "Run number"
    )

    ax_tracks.set_title(
        f"{detector} — Control-region tracks — {title_cut}"
    )

    ax_tracks.grid(
        True,
        alpha=0.2,
    )

    ax_rate.grid(
        True,
        alpha=0.2,
    )

    # --------------------------------------------------------
    # Legends
    # --------------------------------------------------------

    # Working-point legend
    wp_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=7,
            markerfacecolor=colours[wp],
            markeredgecolor=colours[wp],
            label=f"WP {wp}",
        )
        for wp in sorted(colours)
    ]

    # VES / FS legend
    type_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=7,
            color="black",
            label="VES",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markersize=7,
            color="black",
            label="FS",
        ),
    ]

    # Working-point legend
    if wp_handles:

        legend_wp = ax_tracks.legend(
            handles=wp_handles,
            title="Working Point",
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=8,
        )

        ax_tracks.add_artist(legend_wp)

    # VES / FS legend
    ax_tracks.legend(
        handles=type_handles,
        title="Data type",
        loc="lower left",
    )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    fig.suptitle(
        f"MURAVES {detector} — Control region "
        f"AZ [{MIN_AZ}, {MAX_AZ}], "
        f"EL [{MIN_EL}, {MAX_EL}]",
        fontsize=14,
    )

    fig.tight_layout()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = (
        output_dir / f"{filename_cut}.png"
    )

    fig.savefig(
        output_file,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"  Wrote plot: {output_file}"
    )




def plot_overview(df, detector, output_dir):
    """
    Make a compact overview figure showing the median rate per WP/data type
    as a function of chi2 cut.

    This is useful as a summary in addition to the per-run plots.
    """

    # Build long-form summary
    rows = []

    for cut in [None] + CHI2_CUTS:

        if cut is None:
            label = "No cut"
        else:
            label = f"< {cut}"

        for (data_type, wp), group in df.groupby(
            ["data_type", "working_point"]
        ):

            column = (
                "track_rate_hz_no_cut"
                if cut is None
                else f"track_rate_hz_chi2_{cut}"
            )

            if column not in group:
                continue

            values = group[column].dropna()

            if len(values) == 0:
                continue

            rows.append(
                {
                    "chi2_cut": (
                        11 if cut is None else cut
                    ),
                    "label": label,
                    "data_type": data_type,
                    "working_point": wp,
                    "median_rate_hz": values.median(),
                }
            )

    summary = pd.DataFrame(rows)

    if summary.empty:
        return

    colours = make_working_point_colours(
        summary["working_point"].unique()
    )

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    for data_type, marker in [
        ("VES", "o"),
        ("FS", "s"),
    ]:

        subset = summary[
            summary["data_type"] == data_type
        ]

        for wp in sorted(subset["working_point"].unique()):

            wp_subset = subset[
                subset["working_point"] == wp
            ].sort_values("chi2_cut")

            ax.plot(
                wp_subset["chi2_cut"],
                wp_subset["median_rate_hz"],
                marker=marker,
                linestyle="-",
                color=colours[wp],
                label=f"WP {wp}, {data_type}",
            )

    ax.set_xlabel(
        r"$\chi^2$ cut "
        "(11 = no cut)"
    )

    ax.set_ylabel(
        "Median control-region track rate [Hz]"
    )

    ax.set_title(
        f"{detector} — Median control-region rate vs $\\chi^2$ cut"
    )

    ax.grid(
        True,
        alpha=0.2,
    )

    ax.legend(
        fontsize=8,
        ncol=2,
    )

    fig.tight_layout()

    output_file = output_dir / "chi2_cut_rate_overview.png"

    fig.savefig(
        output_file,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"  Wrote overview: {output_file}")


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Plot MURAVES control-region tracks and rates per run."
        )
    )

    parser.add_argument(
        "--detector",
        choices=sorted(RUN_RANGES.keys()),
        help="Detector to process.",
    )

    parser.add_argument(
        "--detectors",
        nargs="+",
        choices=sorted(RUN_RANGES.keys()),
        help="Multiple detectors to process.",
    )

    parser.add_argument(
        "--ves-ranges",
        nargs="+",
        metavar="START-END",
        help=(
            "Override default VES run ranges. "
            "Examples: 4000-4010 5000-5010"
        ),
    )

    parser.add_argument(
        "--fs-ranges",
        nargs="+",
        metavar="START-END",
        help=(
            "Override default FS run ranges. "
            "Example: 10061-10070"
        ),
    )

    parser.add_argument(
        "--az-range",
        nargs=2,
        type=float,
        metavar=("MIN_AZ", "MAX_AZ"),
        default=(MIN_AZ, MAX_AZ),
        help=(
            f"Azimuth control region. "
            f"Default: {MIN_AZ} {MAX_AZ}"
        ),
    )

    parser.add_argument(
        "--el-range",
        nargs=2,
        type=float,
        metavar=("MIN_EL", "MAX_EL"),
        default=(MIN_EL, MAX_EL),
        help=(
            f"Elevation control region. "
            f"Default: {MIN_EL} {MAX_EL}"
        ),
    )

    args = parser.parse_args()

    if args.detector and args.detectors:
        parser.error(
            "Use either --detector or --detectors, not both."
        )

    if args.az_range[1] <= args.az_range[0]:
        parser.error(
            "--az-range requires MAX_AZ > MIN_AZ"
        )

    if args.el_range[1] <= args.el_range[0]:
        parser.error(
            "--el-range requires MAX_EL > MIN_EL"
        )

    if args.detector:
        detectors = [args.detector]
    elif args.detectors:
        detectors = args.detectors
    else:
        detectors = ["ROSSO"]

    ves_ranges = parse_range_list(args.ves_ranges)
    fs_ranges = parse_range_list(args.fs_ranges)

    min_az, max_az = args.az_range
    min_el, max_el = args.el_range

    return (
        args,
        detectors,
        ves_ranges,
        fs_ranges,
        min_az,
        max_az,
        min_el,
        max_el,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    (
        args,
        detectors,
        ves_override,
        fs_override,
        min_az,
        max_az,
        min_el,
        max_el,
    ) = parse_arguments()

    print()
    print("=" * 70)
    print("MURAVES CONTROL-REGION PER-RUN PLOTS")
    print("=" * 70)

    print(f"Detectors        : {', '.join(detectors)}")
    print(
        f"Azimuth region   : [{min_az}, {max_az}]"
    )
    print(
        f"Elevation region : [{min_el}, {max_el}]"
    )

    print()
    print("Plots:")
    print("    no chi2 cut")
    print("    chi2 < 1")
    print("    chi2 < 2")
    print("    ...")
    print("    chi2 < 10")

    print()
    print("Plot encoding:")
    print("    colour = Working Point")
    print("    marker = VES / FS")
    print("    x      = run number")

    print("=" * 70)

    for detector in detectors:

        print()
        print("#" * 70)
        print(f"DETECTOR: {detector}")
        print("#" * 70)

        data_dir = Path(
            DATA_DIR_TEMPLATE.format(
                detector=detector
            )
        )

        detector_output = (
            OUTPUT_DIR / detector / "vesuvius_region"
        )

        detector_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        run_ranges = {
            "VES": (
                ves_override
                if ves_override is not None
                else RUN_RANGES[detector]["VES"]
            ),
            "FS": (
                fs_override
                if fs_override is not None
                else RUN_RANGES[detector]["FS"]
            ),
        }

        records = []

        # ====================================================
        # PROCESS RUNS
        # ====================================================

        for data_type in ["VES", "FS"]:

            print()
            print(
                f"========== "
                f"{detector} / {data_type} "
                f"=========="
            )

            for run in iter_run_numbers(
                run_ranges[data_type]
            ):

                file_path = (
                    data_dir
                    / FILENAME_TEMPLATE.format(
                        run=run
                    )
                )

                if not file_path.is_file():
                    print(
                        f"Run {run}: file not found, skipping"
                    )
                    continue

                # --------------------------------------------
                # Working point
                # --------------------------------------------

                working_point = get_working_point(
                    SLOWCONTROL_DIR,
                    detector,
                    run,
                )

                if working_point is None:
                    print(
                        f"Run {run}: no WP, skipping"
                    )
                    continue

                # --------------------------------------------
                # ROOT
                # --------------------------------------------

                result = process_run(
                    file_path,
                    min_az,
                    max_az,
                    min_el,
                    max_el,
                )

                if result is None:
                    continue

                duration_seconds, effective_chi2 = result

                # --------------------------------------------
                # Counts
                # --------------------------------------------

                counts = calculate_counts(
                    effective_chi2
                )

                record = {
                    "detector": detector,
                    "data_type": data_type,
                    "run": run,
                    "working_point": working_point,
                    "duration_seconds": duration_seconds,

                    "n_tracks_no_cut": counts[None],

                    "track_rate_hz_no_cut": (
                        counts[None]
                        / duration_seconds
                    ),
                }

                for cut in CHI2_CUTS:

                    record[
                        f"n_tracks_chi2_{cut}"
                    ] = counts[cut]

                    record[
                        f"track_rate_hz_chi2_{cut}"
                    ] = (
                        counts[cut]
                        / duration_seconds
                    )

                records.append(record)

                print(
                    f"Run {run:6d} | "
                    f"{data_type:3s} | "
                    f"WP {working_point:2d} | "
                    f"duration "
                    f"{duration_seconds:8.1f} s | "
                    f"tracks "
                    f"{counts[None]:6d}"
                )

        # ====================================================
        # DATAFRAME
        # ====================================================

        if not records:
            print(
                f"No valid runs found for {detector}."
            )
            continue

        df = pd.DataFrame(records)

        df = df.sort_values(
            ["run"]
        ).reset_index(drop=True)

        # ----------------------------------------------------
        # Save the complete per-run dataset
        # ----------------------------------------------------

        parquet_file = (
            detector_output
            / "control_region_per_run.parquet"
        )

        df.to_parquet(
            parquet_file,
            index=False,
            compression="zstd",
        )

        print()
        print(
            f"Wrote per-run data: {parquet_file}"
        )

        # Also save CSV because it is convenient for inspection
        csv_file = (
            detector_output
            / "control_region_per_run.csv"
        )

        df.to_csv(
            csv_file,
            index=False,
        )

        print(
            f"Wrote CSV: {csv_file}"
        )

        # ====================================================
        # PLOTS
        # ====================================================

        print()
        print("Making plots...")

        # No chi2 cut
        plot_one_cut(
            df=df,
            detector=detector,
            chi2_cut=None,
            output_dir=detector_output,
        )

        # chi2 < 1 ... 10
        for cut in CHI2_CUTS:

            plot_one_cut(
                df=df,
                detector=detector,
                chi2_cut=cut,
                output_dir=detector_output,
            )

        # ====================================================
        # SUMMARY PLOT
        # ====================================================

        plot_overview(
            df=df,
            detector=detector,
            output_dir=detector_output,
        )

        print()
        print(
            f"Processed {len(df)} valid runs "
            f"for {detector}."
        )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
