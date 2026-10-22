#!/usr/bin/env python3

"""
angular_track_rate_maps.py

Create 1° x 1° angular track-rate maps for MURAVES.

For 3p:
    Selection:
        BestTrack_3p_ChiSquare_xy > 0
        AND
        BestTrack_3p_ChiSquare_xz > 0

    Direction:
        Phi_3p
        Theta_3p

For 4p:
    Selection:
        BestTrack_4p_ChiSquare_xy > 0
        AND
        BestTrack_4p_ChiSquare_xz > 0

    Direction:
        Phi_4p
        Theta_4p

For each reconstruction (3p and 4p), three maps are produced:

    1. Free Sky (FS) track rate
    2. Vesuvius (VES) track rate
    3. VES / FS track-rate ratio

Track rate:

    rate = number of selected tracks / total run duration

where timestamp is assumed to be in milliseconds.

Thus:

    rate units = tracks / second / (1° x 1° bin)

The VES/FS ratio is:

    (VES tracks / VES duration)
    ----------------------------
    (FS tracks / FS duration)

Bins where the FS rate is zero are set to NaN.

Example:

    python angular_track_rate_maps.py --detector ROSSO

or:

    python angular_track_rate_maps.py --detectors NERO ROSSO BLU
"""


from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
import uproot
from matplotlib.colors import LogNorm


# ============================================================
# USER CONFIGURATION
# ============================================================

DATA_DIR_TEMPLATE = (
    "/pnfs/iihe/muraves/muraves_DATA/RECONSTRUCTED/{detector}/v1"
)

OUTPUT_DIR = Path("./muraves_angular_rate_maps")

TREE_NAME = "AnalyzedData"

FILENAME_TEMPLATE = "MURAVES_AnalyzedData_run{run}.root"


# ============================================================
# BRANCHES
# ============================================================

# 3p chi2
CHI2_3P_XY = "BestTrack_3p_ChiSquare_xy"
CHI2_3P_XZ = "BestTrack_3p_ChiSquare_xz"

# 3p direction
PHI_3P = "Phi_3p"
THETA_3P = "Theta_3p"


# 4p chi2
CHI2_4P_XY = "BestTrack_4p_ChiSquare_xy"
CHI2_4P_XZ = "BestTrack_4p_ChiSquare_xz"

# 4p direction
PHI_4P = "Phi_4p"
THETA_4P = "Theta_4p"


# Timestamp
TIMESTAMP_BRANCH = "timestamp"


# ============================================================
# ANGULAR BINNING
# ============================================================

D_PHI = 1.0
D_THETA = 1.0


# Angular range
PHI_MIN = 0.0
PHI_MAX = 360.0

THETA_MIN = 0.0
THETA_MAX = 90.0


# ============================================================
# DEFAULT RUN RANGES
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


# ============================================================
# HELPERS
# ============================================================

def make_edges(min_val, max_val, bin_size):
    """Build equally spaced bin edges."""

    if bin_size <= 0:
        raise ValueError("Bin size must be > 0.")

    if max_val <= min_val:
        raise ValueError("max_val must be > min_val.")

    n_bins = int(
        np.ceil(
            (max_val - min_val) / bin_size
        )
    )

    return np.linspace(
        min_val,
        min_val + n_bins * bin_size,
        n_bins + 1,
    )


def parse_run_range(value):
    """
    Parse:

        START-END
        START:END
        single run number
    """

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
        start = int(parts[0])
        stop = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid run range: {value}. "
            f"Use START-END, e.g. 4000-4010."
        ) from exc

    if stop < start:
        raise argparse.ArgumentTypeError(
            f"Invalid run range: {value}: "
            f"end must be >= start."
        )

    return start, stop


def parse_range_list(values):
    """Convert command-line ranges to (start, stop) tuples."""

    if values is None:
        return None

    return [
        parse_run_range(value)
        for value in values
    ]


def iter_run_numbers(run_ranges):
    """Yield all run numbers in the supplied ranges."""

    for start, stop in run_ranges:
        yield from range(start, stop + 1)


# ============================================================
# RUN DURATION
# ============================================================

def get_run_duration_seconds(timestamps):
    """
    Calculate run duration.

    timestamp is assumed to be in milliseconds.
    """

    timestamps = np.asarray(
        timestamps,
        dtype=float,
    )

    valid = np.isfinite(timestamps)

    if not np.any(valid):
        return None

    tmin = np.min(
        timestamps[valid]
    )

    tmax = np.max(
        timestamps[valid]
    )

    duration = (
        tmax - tmin
    ) / 1000.0

    if duration <= 0:
        return None

    return float(duration)


# ============================================================
# ROOT READING
# ============================================================

def read_run(file_path):
    """
    Read all branches needed for the analysis.
    """

    branches = [

        # 3p
        PHI_3P,
        THETA_3P,
        CHI2_3P_XY,
        CHI2_3P_XZ,

        # 4p
        PHI_4P,
        THETA_4P,
        CHI2_4P_XY,
        CHI2_4P_XZ,

        # Time
        TIMESTAMP_BRANCH,
    ]

    try:

        with uproot.open(file_path) as f:

            if TREE_NAME not in f:

                print(
                    f"  [warn] tree "
                    f"'{TREE_NAME}' not found"
                )

                return None

            tree = f[TREE_NAME]

            missing = (
                set(branches)
                - set(tree.keys())
            )

            if missing:

                print(
                    f"  [warn] missing branches: "
                    f"{sorted(missing)}"
                )

                return None

            arrays = tree.arrays(
                branches,
                library="np",
            )

    except Exception as exc:

        print(
            f"  [warn] could not read "
            f"{file_path}: {exc}"
        )

        return None

    return arrays


# ============================================================
# CHI2 SELECTION
# ============================================================

def get_3p_valid_events(arrays):
    """
    3p event selection.

    Both 3p chi2 values must be > 0.
    """

    chi2_xy = np.asarray(
        arrays[CHI2_3P_XY],
        dtype=float,
    )

    chi2_xz = np.asarray(
        arrays[CHI2_3P_XZ],
        dtype=float,
    )

    valid = (

        np.isfinite(chi2_xy)

        & np.isfinite(chi2_xz)

        & (chi2_xy > 0.0)

        & (chi2_xz > 0.0)
    )

    return valid


def get_4p_valid_events(arrays):
    """
    4p event selection.

    Both 4p chi2 values must be > 0.
    """

    chi2_xy = np.asarray(
        arrays[CHI2_4P_XY],
        dtype=float,
    )

    chi2_xz = np.asarray(
        arrays[CHI2_4P_XZ],
        dtype=float,
    )

    valid = (

        np.isfinite(chi2_xy)

        & np.isfinite(chi2_xz)

        & (chi2_xy > 0.0)

        & (chi2_xz > 0.0)
    )

    return valid


# ============================================================
# ANGULAR HISTOGRAM
# ============================================================

def make_angular_histogram(
    phi,
    theta,
    valid,
    phi_edges,
    theta_edges,
):
    """
    Create a 2D histogram of:

        Phi x Theta

    for the selected events.
    """

    phi = np.asarray(
        phi,
        dtype=float,
    )

    theta = np.asarray(
        theta,
        dtype=float,
    )

    valid = (
        valid
        & np.isfinite(phi)
        & np.isfinite(theta)
    )

    phi_selected = phi[valid]
    theta_selected = theta[valid]

    hist, _, _ = np.histogram2d(
        phi_selected,
        theta_selected,
        bins=[
            phi_edges,
            theta_edges,
        ],
    )

    # np.histogram2d gives:
    #
    #     H[phi_bin, theta_bin]
    #
    # transpose for plotting as:
    #
    #     x = phi
    #     y = theta

    return hist.T.astype(np.int64)


# ============================================================
# PROCESS FS / VES
# ============================================================

def process_data_type(
    detector,
    data_type,
    run_ranges,
    phi_edges,
    theta_edges,
):
    """
    Process all runs for FS or VES.

    Returns:

        counts_3p
        counts_4p
        total_duration
    """

    data_dir = Path(
        DATA_DIR_TEMPLATE.format(
            detector=detector
        )
    )

    n_phi_bins = len(phi_edges) - 1
    n_theta_bins = len(theta_edges) - 1

    counts_3p = np.zeros(
        (
            n_theta_bins,
            n_phi_bins,
        ),
        dtype=np.int64,
    )

    counts_4p = np.zeros_like(
        counts_3p
    )

    total_duration = 0.0

    total_3p_events = 0
    total_4p_events = 0

    # --------------------------------------------------------
    # Loop over runs
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(f"{detector} / {data_type}")
    print("=" * 70)

    for run in iter_run_numbers(
        run_ranges
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

        print()
        print(
            f"Run {run}: "
            f"{file_path.name}"
        )

        # ----------------------------------------------------
        # Read ROOT
        # ----------------------------------------------------

        arrays = read_run(
            file_path
        )

        if arrays is None:
            continue

        # ----------------------------------------------------
        # Run duration
        # ----------------------------------------------------

        duration = get_run_duration_seconds(
            arrays[TIMESTAMP_BRANCH]
        )

        if duration is None:

            print(
                "  [warn] invalid run duration, "
                "skipping run"
            )

            continue

        print(
            f"  Duration: "
            f"{duration:.3f} s "
            f"({duration / 3600.0:.3f} h)"
        )

        # ----------------------------------------------------
        # 3p selection
        # ----------------------------------------------------

        valid_3p = get_3p_valid_events(
            arrays
        )

        n_3p = int(
            np.count_nonzero(valid_3p)
        )

        print(
            f"  3p events with "
            f"both chi2 > 0: {n_3p}"
        )

        # ----------------------------------------------------
        # 4p selection
        # ----------------------------------------------------

        valid_4p = get_4p_valid_events(
            arrays
        )

        n_4p = int(
            np.count_nonzero(valid_4p)
        )

        print(
            f"  4p events with "
            f"both chi2 > 0: {n_4p}"
        )

        # ----------------------------------------------------
        # 3p angular histogram
        # ----------------------------------------------------

        h3p = make_angular_histogram(
            arrays[PHI_3P],
            arrays[THETA_3P],
            valid_3p,
            phi_edges,
            theta_edges,
        )

        # ----------------------------------------------------
        # 4p angular histogram
        # ----------------------------------------------------

        h4p = make_angular_histogram(
            arrays[PHI_4P],
            arrays[THETA_4P],
            valid_4p,
            phi_edges,
            theta_edges,
        )

        # ----------------------------------------------------
        # Accumulate
        # ----------------------------------------------------

        counts_3p += h3p
        counts_4p += h4p

        total_duration += duration

        total_3p_events += n_3p
        total_4p_events += n_4p

        print(
            f"  3p tracks in angular range: "
            f"{int(h3p.sum())}"
        )

        print(
            f"  4p tracks in angular range: "
            f"{int(h4p.sum())}"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print(
        f"{detector} / {data_type} summary"
    )

    print(
        f"  Total duration: "
        f"{total_duration:.3f} s "
        f"({total_duration / 3600.0:.3f} h)"
    )

    print(
        f"  Total selected 3p events: "
        f"{total_3p_events}"
    )

    print(
        f"  Total selected 4p events: "
        f"{total_4p_events}"
    )

    print(
        f"  Total 3p tracks in range: "
        f"{int(counts_3p.sum())}"
    )

    print(
        f"  Total 4p tracks in range: "
        f"{int(counts_4p.sum())}"
    )

    return (
        counts_3p,
        counts_4p,
        total_duration,
    )


# ============================================================
# COUNTS -> RATE
# ============================================================

def counts_to_rate(
    counts,
    duration_seconds,
):
    """
    Convert total accumulated counts to:

        tracks / second / angular bin
    """

    if duration_seconds <= 0:

        return np.full(
            counts.shape,
            np.nan,
            dtype=float,
        )

    return (
        counts.astype(float)
        / duration_seconds
    )


# ============================================================
# VES / FS RATIO
# ============================================================

def calculate_ratio(
    ves_rate,
    fs_rate,
):
    """
    Calculate:

        VES rate / FS rate

    FS bins with zero rate are NaN.
    """

    ratio = np.full(
        fs_rate.shape,
        np.nan,
        dtype=float,
    )

    valid = (
        np.isfinite(ves_rate)
        & np.isfinite(fs_rate)
        & (fs_rate > 0.0)
    )

    ratio[valid] = (
        ves_rate[valid]
        / fs_rate[valid]
    )

    return ratio


# ============================================================
# PLOT RATE MAP
# ============================================================

def plot_rate_map(
    rate,
    phi_edges,
    theta_edges,
    detector,
    data_type,
    track_type,
    output_dir,
):
    """
    Plot track rate in tracks/s/bin.
    """

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    mesh = ax.pcolormesh(
        phi_edges,
        theta_edges,
        rate,
        shading="auto",
    )

    cbar = fig.colorbar(
        mesh,
        ax=ax,
    )

    cbar.set_label(
        "Track rate [tracks/s/bin]"
    )

    ax.set_xlabel(
        r"$\phi$ [degrees]"
    )

    ax.set_ylabel(
        r"$\theta$ [degrees]"
    )

    ax.set_title(
        f"{detector} — {data_type} — "
        f"{track_type}"
    )

    ax.set_xlim(
        phi_edges[0],
        phi_edges[-1],
    )

    ax.set_ylim(
        theta_edges[0],
        theta_edges[-1],
    )

    fig.tight_layout()

    output_file = (
        output_dir
        / f"{detector}_{data_type}_{track_type}_rate.png"
    )

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Wrote: {output_file}"
    )


# ============================================================
# PLOT RATIO
# ============================================================

def plot_ratio_map(
    ratio,
    phi_edges,
    theta_edges,
    detector,
    track_type,
    output_dir,
):
    """
    Plot:

        VES rate / FS rate
    """

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    mesh = ax.pcolormesh(
        phi_edges,
        theta_edges,
        ratio,
        shading="auto",
        norm=LogNorm()
    )

    cbar = fig.colorbar(
        mesh,
        ax=ax,
    )

    cbar.set_label(
        "VES / FS track-rate ratio"
    )

    ax.set_xlabel(
        r"$\phi$ [degrees]"
    )

    ax.set_ylabel(
        r"$\theta$ [degrees]"
    )

    ax.set_title(
        f"{detector} — VES / FS — "
        f"{track_type}"
    )

    ax.set_xlim(
        phi_edges[0],
        phi_edges[-1],
    )

    ax.set_ylim(
        theta_edges[0],
        theta_edges[-1],
    )

    fig.tight_layout()

    output_file = (
        output_dir
        / f"{detector}_VES_over_FS_{track_type}_rate.png"
    )

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Wrote: {output_file}"
    )


# ============================================================
# SAVE DATA
# ============================================================

def save_map(
    rate,
    phi_edges,
    theta_edges,
    output_file,
):
    """
    Save numerical rate map and bin edges.
    """

    np.savez_compressed(
        output_file,
        rate=rate,
        phi_edges=phi_edges,
        theta_edges=theta_edges,
    )

    print(
        f"Wrote: {output_file}"
    )


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Create MURAVES 1° x 1° "
            "angular track-rate maps."
        )
    )

    parser.add_argument(
        "--detector",
        choices=sorted(RUN_RANGES.keys()),
        help="Single detector to process.",
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
            "Example: 4000-4010 5000-5010"
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
        "--phi-range",
        nargs=2,
        type=float,
        metavar=("MIN_PHI", "MAX_PHI"),
        default=(
            PHI_MIN,
            PHI_MAX,
        ),
        help=(
            f"Phi range. "
            f"Default: {PHI_MIN} {PHI_MAX}"
        ),
    )

    parser.add_argument(
        "--theta-range",
        nargs=2,
        type=float,
        metavar=("MIN_THETA", "MAX_THETA"),
        default=(
            THETA_MIN,
            THETA_MAX,
        ),
        help=(
            f"Theta range. "
            f"Default: {THETA_MIN} {THETA_MAX}"
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Detector selection
    # --------------------------------------------------------

    if args.detector and args.detectors:

        parser.error(
            "Use either --detector or "
            "--detectors, not both."
        )

    if args.detector:

        detectors = [
            args.detector
        ]

    elif args.detectors:

        detectors = args.detectors

    else:

        detectors = [
            "ROSSO"
        ]

    # --------------------------------------------------------
    # Run ranges
    # --------------------------------------------------------

    ves_ranges = parse_range_list(
        args.ves_ranges
    )

    fs_ranges = parse_range_list(
        args.fs_ranges
    )

    # --------------------------------------------------------
    # Angular ranges
    # --------------------------------------------------------

    min_phi, max_phi = args.phi_range

    min_theta, max_theta = args.theta_range

    if max_phi <= min_phi:

        parser.error(
            "--phi-range requires "
            "MAX_PHI > MIN_PHI"
        )

    if max_theta <= min_theta:

        parser.error(
            "--theta-range requires "
            "MAX_THETA > MIN_THETA"
        )

    return (
        detectors,
        ves_ranges,
        fs_ranges,
        min_phi,
        max_phi,
        min_theta,
        max_theta,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    (
        detectors,
        ves_override,
        fs_override,
        min_phi,
        max_phi,
        min_theta,
        max_theta,
    ) = parse_arguments()

    # --------------------------------------------------------
    # Angular bins
    # --------------------------------------------------------

    phi_edges = make_edges(
        min_phi,
        max_phi,
        D_PHI,
    )

    theta_edges = make_edges(
        min_theta,
        max_theta,
        D_THETA,
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MURAVES ANGULAR TRACK-RATE MAPS")
    print("=" * 70)

    print(
        f"Detectors      : "
        f"{', '.join(detectors)}"
    )

    print(
        f"Phi range      : "
        f"[{min_phi}, {max_phi}]"
    )

    print(
        f"Theta range    : "
        f"[{min_theta}, {max_theta}]"
    )

    print(
        f"Phi bin        : "
        f"{D_PHI} degree"
    )

    print(
        f"Theta bin      : "
        f"{D_THETA} degree"
    )

    print()
    print("3p selection:")
    print(
        f"    {CHI2_3P_XY} > 0"
    )
    print(
        f"    {CHI2_3P_XZ} > 0"
    )
    print(
        f"    direction = "
        f"{PHI_3P}, {THETA_3P}"
    )

    print()
    print("4p selection:")
    print(
        f"    {CHI2_4P_XY} > 0"
    )
    print(
        f"    {CHI2_4P_XZ} > 0"
    )
    print(
        f"    direction = "
        f"{PHI_4P}, {THETA_4P}"
    )

    print()
    print(
        "Rate = selected tracks / total run duration"
    )

    print(
        "Ratio = VES rate / FS rate"
    )

    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # LOOP OVER DETECTORS
    # ========================================================

    for detector in detectors:

        print()
        print("#" * 70)
        print(
            f"DETECTOR: {detector}"
        )
        print("#" * 70)

        detector_output = (
            OUTPUT_DIR / detector
        )

        detector_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Run ranges
        # ----------------------------------------------------

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

        print()
        print("Run ranges:")

        for data_type in [
            "VES",
            "FS",
        ]:

            formatted = ", ".join(
                f"{start}-{stop}"
                for start, stop
                in run_ranges[data_type]
            )

            print(
                f"  {data_type}: "
                f"{formatted}"
            )

        # ====================================================
        # PROCESS FS
        # ====================================================

        (
            fs_counts_3p,
            fs_counts_4p,
            fs_duration,
        ) = process_data_type(
            detector,
            "FS",
            run_ranges["FS"],
            phi_edges,
            theta_edges,
        )

        # ====================================================
        # PROCESS VES
        # ====================================================

        (
            ves_counts_3p,
            ves_counts_4p,
            ves_duration,
        ) = process_data_type(
            detector,
            "VES",
            run_ranges["VES"],
            phi_edges,
            theta_edges,
        )

        # ====================================================
        # CONVERT TO RATES
        # ====================================================

        fs_rate_3p = counts_to_rate(
            fs_counts_3p,
            fs_duration,
        )

        ves_rate_3p = counts_to_rate(
            ves_counts_3p,
            ves_duration,
        )

        fs_rate_4p = counts_to_rate(
            fs_counts_4p,
            fs_duration,
        )

        ves_rate_4p = counts_to_rate(
            ves_counts_4p,
            ves_duration,
        )

        # ====================================================
        # VES / FS RATIOS
        # ====================================================

        ratio_3p = calculate_ratio(
            ves_rate_3p,
            fs_rate_3p,
        )

        ratio_4p = calculate_ratio(
            ves_rate_4p,
            fs_rate_4p,
        )

        # ====================================================
        # 3p PLOTS
        # ====================================================

        plot_rate_map(
            fs_rate_3p,
            phi_edges,
            theta_edges,
            detector,
            "FS",
            "3p",
            detector_output,
        )

        plot_rate_map(
            ves_rate_3p,
            phi_edges,
            theta_edges,
            detector,
            "VES",
            "3p",
            detector_output,
        )

        plot_ratio_map(
            ratio_3p,
            phi_edges,
            theta_edges,
            detector,
            "3p",
            detector_output,
        )

        # ====================================================
        # 4p PLOTS
        # ====================================================

        plot_rate_map(
            fs_rate_4p,
            phi_edges,
            theta_edges,
            detector,
            "FS",
            "4p",
            detector_output,
        )

        plot_rate_map(
            ves_rate_4p,
            phi_edges,
            theta_edges,
            detector,
            "VES",
            "4p",
            detector_output,
        )

        plot_ratio_map(
            ratio_4p,
            phi_edges,
            theta_edges,
            detector,
            "4p",
            detector_output,
        )

        # ====================================================
        # SAVE NUMERICAL MAPS
        # ====================================================

        save_map(
            fs_rate_3p,
            phi_edges,
            theta_edges,
            detector_output
            / "FS_3p_rate.npz",
        )

        save_map(
            ves_rate_3p,
            phi_edges,
            theta_edges,
            detector_output
            / "VES_3p_rate.npz",
        )

        save_map(
            fs_rate_4p,
            phi_edges,
            theta_edges,
            detector_output
            / "FS_4p_rate.npz",
        )

        save_map(
            ves_rate_4p,
            phi_edges,
            theta_edges,
            detector_output
            / "VES_4p_rate.npz",
        )

        save_map(
            ratio_3p,
            phi_edges,
            theta_edges,
            detector_output
            / "VES_over_FS_3p_rate.npz",
        )

        save_map(
            ratio_4p,
            phi_edges,
            theta_edges,
            detector_output
            / "VES_over_FS_4p_rate.npz",
        )

        # ====================================================
        # PRINT FINAL EXPOSURE
        # ====================================================

        print()
        print(
            f"{detector} final exposure:"
        )

        print(
            f"  FS  : "
            f"{fs_duration:.3f} s "
            f"({fs_duration / 3600.0:.3f} h)"
        )

        print(
            f"  VES : "
            f"{ves_duration:.3f} s "
            f"({ves_duration / 3600.0:.3f} h)"
        )

    # ========================================================
    # DONE
    # ========================================================

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()