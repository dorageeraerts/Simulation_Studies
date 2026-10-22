#!/usr/bin/env python3

"""
Fit the VES/FS chi2-equals-one relation and calculate transmission for
both 4p and 3p track chi2 selections.

Normal mode:
    1. Fit the (VES chi2, FS chi2) pairs for which the existing scan has
       VES rate / FS rate ~= 1.
    2. Evaluate the fitted FS cut for VES cuts from 1.0 to 12.0 in steps
       of 0.1.
    3. Re-read the ROOT data and count control-region tracks with either
       the 4p or 3p chi2 pair below the corresponding cuts.
    4. Write the rates and transmission values to parquet.
    5. Make separate 3p and 4p transmission plots.

Parquet-only mode:
    With --plot-from-parquet, skip all fitting and ROOT reading.
    Read existing transmission_3p4p_along_fit.parquet files and remake
    the separate 3p and 4p plots.

The parquet contains separate 4p and 3p counts/rates for VES and FS,
so the plotting result can be reproduced independently of the ROOT data.
"""

from pathlib import Path

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot


# ============================================================
# Configuration
# ============================================================

DEFAULT_INPUT_DIR = Path("./muraves_chi2_scan")
DEFAULT_OUTPUT_DIR = Path("./muraves_chi2_plots")
DEFAULT_DETECTOR = "ROSSO"

CHI2_MIN = 0.0
CHI2_MAX = 12.0
CHI2_STEP = 0.1
VES_PLOT_MIN = 1.0

FIT_TOLERANCE = 0.01
FIT_DEGREE = 2

MIN_AZ, MAX_AZ = 170.0, 190.0
MIN_EL, MAX_EL = 12.0, 18.0

DATA_DIR_TEMPLATE = (
    "/pnfs/iihe/muraves/muraves_DATA/RECONSTRUCTED/{detector}/v1"
)
SLOWCONTROL_DIR = "/pnfs/iihe/muraves/muraves_DATA/DECOMPRESSED"

TREE_NAME = "AnalyzedData"

TIMESTAMP_BRANCH = "timestamp"
PHI_BRANCH = "Phi_3p"
THETA_BRANCH = "Theta_3p"

FILENAME_TEMPLATE = "MURAVES_AnalyzedData_run{run}.root"
SLOWCONTROL_TEMPLATE = "SLOWCONTROL_run{run}"

CHI2_BRANCHES = {
    "4p": (
        "BestTrack_4p_ChiSquare_xy",
        "BestTrack_4p_ChiSquare_xz",
    ),
    "3p": (
        "BestTrack_3p_ChiSquare_xy",
        "BestTrack_3p_ChiSquare_xz",
    ),
}

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
# Fit helpers
# ============================================================

def get_parquet_files(input_dir, working_point):
    ves_file = input_dir / f"scan_VES_WP{working_point}.parquet"
    fs_file = input_dir / f"scan_FS_WP{working_point}.parquet"

    if not ves_file.exists():
        raise FileNotFoundError(f"VES file not found: {ves_file}")

    if not fs_file.exists():
        raise FileNotFoundError(f"FS file not found: {fs_file}")

    return ves_file, fs_file


def find_common_working_points(input_dir):
    working_points = []

    for ves_file in input_dir.glob("scan_VES_WP*.parquet"):
        wp_text = ves_file.stem.removeprefix("scan_VES_WP")

        try:
            working_point = int(wp_text)
        except ValueError:
            continue

        if (input_dir / f"scan_FS_WP{working_point}.parquet").exists():
            working_points.append(working_point)

    return sorted(set(working_points))


def load_scan(parquet_file):
    columns = ["chi2_cut", "track_rate_hz"]

    scan_df = pd.read_parquet(parquet_file, columns=columns)

    if scan_df.empty:
        raise ValueError(f"Parquet file is empty: {parquet_file}")

    return scan_df


def calculate_ratio_table(ves_df, fs_df):
    ves_df = ves_df.sort_values("chi2_cut")
    fs_df = fs_df.sort_values("chi2_cut")

    ves_cuts = ves_df["chi2_cut"].to_numpy(dtype=float)
    fs_cuts = fs_df["chi2_cut"].to_numpy(dtype=float)

    ves_rates = ves_df["track_rate_hz"].to_numpy(dtype=float)
    fs_rates = fs_df["track_rate_hz"].to_numpy(dtype=float)

    ratio_matrix = np.full(
        (len(fs_cuts), len(ves_cuts)),
        np.nan,
        dtype=float,
    )

    for row, fs_rate in enumerate(fs_rates):
        if np.isfinite(fs_rate) and fs_rate > 0:
            ratio_matrix[row, :] = ves_rates / fs_rate

    return pd.DataFrame(
        ratio_matrix,
        index=fs_cuts,
        columns=ves_cuts,
    )


def fit_ratio_one_contour(ratio_df, working_point):
    ves_cuts = ratio_df.columns.to_numpy(dtype=float)
    fs_cuts = ratio_df.index.to_numpy(dtype=float)
    ratio_matrix = ratio_df.to_numpy(dtype=float)

    mask = (
        np.isfinite(ratio_matrix)
        & np.isclose(
            ratio_matrix,
            1.0,
            atol=FIT_TOLERANCE,
            rtol=0,
        )
    )

    fs_indices, ves_indices = np.where(mask)

    ves_points = ves_cuts[ves_indices]
    fs_points = fs_cuts[fs_indices]

    displayed = ves_points >= VES_PLOT_MIN

    ves_points = ves_points[displayed]
    fs_points = fs_points[displayed]

    if len(ves_points) < FIT_DEGREE + 1:
        raise ValueError(
            f"WP {working_point}: only {len(ves_points)} ratio-one points; "
            f"need at least {FIT_DEGREE + 1} for degree {FIT_DEGREE}."
        )

    fit_parameters = np.polyfit(
        ves_points,
        fs_points,
        deg=FIT_DEGREE,
    )

    print(
        f"WP {working_point}: fit parameters "
        f"(degree {FIT_DEGREE}, highest power first) = "
        f"{fit_parameters}"
    )

    return fit_parameters


# ============================================================
# ROOT counting
# ============================================================

def make_chi2_cuts():
    count = int(round((CHI2_MAX - CHI2_MIN) / CHI2_STEP))

    return CHI2_MIN + np.arange(
        count + 1,
        dtype=float,
    ) * CHI2_STEP


def parse_run_range(value):
    separator = "-" if "-" in value else ":" if ":" in value else None

    if separator is None:
        run = int(value)
        return run, run

    start, stop = value.split(separator, 1)
    start, stop = int(start), int(stop)

    if stop < start:
        raise ValueError(f"Invalid run range: {value}")

    return start, stop


def iter_run_numbers(run_ranges):
    for start, stop in run_ranges:
        yield from range(start, stop + 1)


def get_working_point(detector, run):
    slowcontrol_file = (
        Path(SLOWCONTROL_DIR)
        / detector
        / SLOWCONTROL_TEMPLATE.format(run=run)
    )

    if not slowcontrol_file.is_file():
        return None

    try:
        with open(
            slowcontrol_file,
            "r",
            errors="replace",
        ) as stream:
            for line in stream:
                fields = line.split()

                if len(fields) <= 5:
                    continue

                working_point = float(fields[5])

                if (
                    np.isfinite(working_point)
                    and working_point.is_integer()
                ):
                    return int(working_point)

    except (OSError, ValueError):
        return None

    return None


def get_run_duration_seconds(timestamps):
    timestamps = np.asarray(
        timestamps,
        dtype=float,
    )

    valid = np.isfinite(timestamps)

    if not np.any(valid):
        return None

    duration = (
        timestamps[valid].max()
        - timestamps[valid].min()
    ) / 1000.0

    return float(duration) if duration > 0 else None


def read_run(file_path):
    branches = [
        PHI_BRANCH,
        THETA_BRANCH,
        TIMESTAMP_BRANCH,
        *CHI2_BRANCHES["4p"],
        *CHI2_BRANCHES["3p"],
    ]

    try:
        with uproot.open(file_path) as root_file:
            if TREE_NAME not in root_file:
                print(
                    f"  [warn] tree '{TREE_NAME}' not found"
                )
                return None

            tree = root_file[TREE_NAME]

            missing = set(branches) - set(tree.keys())

            if missing:
                print(
                    f"  [warn] missing branches: "
                    f"{sorted(missing)}"
                )
                return None

            return tree.arrays(
                branches,
                library="np",
            )

    except Exception as exc:
        print(
            f"  [warn] could not read {file_path}: {exc}"
        )
        return None


def process_run(file_path):
    arrays = read_run(file_path)

    if arrays is None:
        return None

    duration = get_run_duration_seconds(
        arrays[TIMESTAMP_BRANCH]
    )

    if duration is None:
        return None

    phi = np.asarray(
        arrays[PHI_BRANCH],
        dtype=float,
    )

    theta = np.asarray(
        arrays[THETA_BRANCH],
        dtype=float,
    )

    valid = (
        np.isfinite(phi)
        & np.isfinite(theta)
        & (phi >= MIN_AZ)
        & (phi <= MAX_AZ)
        & (theta >= MIN_EL)
        & (theta <= MAX_EL)
    )

    result = {
        "duration_seconds": duration,
    }

    for track_type, (
        xy_branch,
        xz_branch,
    ) in CHI2_BRANCHES.items():

        chi2_xy = np.asarray(
            arrays[xy_branch],
            dtype=float,
        )

        chi2_xz = np.asarray(
            arrays[xz_branch],
            dtype=float,
        )

        valid_type = (
            valid
            & np.isfinite(chi2_xy)
            & np.isfinite(chi2_xz)
            & (chi2_xy >= 0.0)
            & (chi2_xz >= 0.0)
        )

        result[track_type] = np.maximum(
            chi2_xy[valid_type],
            chi2_xz[valid_type],
        )

    return result


def count_runs_for_working_point(
    detector,
    data_type,
    working_point,
    cuts,
):
    run_ranges = RUN_RANGES[detector][data_type]

    data_dir = Path(
        DATA_DIR_TEMPLATE.format(
            detector=detector
        )
    )

    durations = []

    chi2_values = {
        "4p": [],
        "3p": [],
    }

    for run in iter_run_numbers(run_ranges):

        if get_working_point(
            detector,
            run,
        ) != working_point:
            continue

        file_path = (
            data_dir
            / FILENAME_TEMPLATE.format(run=run)
        )

        if not file_path.is_file():
            continue

        result = process_run(file_path)

        if result is None:
            continue

        durations.append(
            result["duration_seconds"]
        )

        for track_type in chi2_values:
            if len(result[track_type]) > 0:
                chi2_values[track_type].append(
                    result[track_type]
                )

    duration = float(np.sum(durations))

    cuts = np.asarray(
        cuts,
        dtype=float,
    )

    output = {
        "duration_seconds": duration,
        "chi2_cut": cuts,
    }

    for track_type in ("4p", "3p"):

        if chi2_values[track_type]:
            all_chi2 = np.sort(
                np.concatenate(
                    chi2_values[track_type]
                )
            )

            counts = np.searchsorted(
                all_chi2,
                cuts,
                side="left",
            ).astype(np.int64)

        else:
            counts = np.zeros(
                len(cuts),
                dtype=np.int64,
            )

        output[
            f"n_tracks_{track_type}"
        ] = counts

        output[
            f"track_rate_hz_{track_type}"
        ] = (
            counts.astype(float) / duration
            if duration > 0
            else np.full(
                len(cuts),
                np.nan,
            )
        )

    return pd.DataFrame(output)


# ============================================================
# Output
# ============================================================

def build_transmission_table(
    ves_rates,
    fs_rates,
    fit_parameters,
):
    ves_cuts = make_chi2_cuts()

    ves_cuts = ves_cuts[
        ves_cuts >= VES_PLOT_MIN
    ]

    fs_fit_cuts = np.polyval(
        fit_parameters,
        ves_cuts,
    )

    output = {
        "chi2_ves_cut": ves_cuts,
        "chi2_fs_cut_fit": fs_fit_cuts,
        "ves_duration_seconds":
            ves_rates["duration_seconds"].iloc[0],
        "fs_duration_seconds":
            fs_rates["duration_seconds"].iloc[0],
    }

    for track_type in ("4p", "3p"):

        ves_counts = ves_rates[
            f"n_tracks_{track_type}"
        ].to_numpy(dtype=float)

        fs_counts = fs_rates[
            f"n_tracks_{track_type}"
        ].to_numpy(dtype=float)

        ves_rates_at_cut = ves_rates[
            f"track_rate_hz_{track_type}"
        ].to_numpy(dtype=float)

        fs_rates_at_cut = fs_rates[
            f"track_rate_hz_{track_type}"
        ].to_numpy(dtype=float)

        output[
            f"ves_track_rate_hz_{track_type}"
        ] = ves_rates_at_cut

        output[
            f"fs_track_rate_hz_{track_type}"
        ] = fs_rates_at_cut

        transmission = np.divide(
            ves_rates_at_cut,
            fs_rates_at_cut,
            out=np.full(
                len(ves_cuts),
                np.nan,
            ),
            where=(
                np.isfinite(fs_rates_at_cut)
                & (fs_rates_at_cut > 0)
            ),
        )

        transmission_error = np.full(
            len(ves_cuts),
            np.nan,
        )

        valid_counts = (
            np.isfinite(transmission)
            & (ves_counts > 0)
            & (fs_counts > 0)
        )

        transmission_error[valid_counts] = (
            transmission[valid_counts]
            * np.sqrt(
                1.0 / ves_counts[valid_counts]
                + 1.0 / fs_counts[valid_counts]
            )
        )

        output[
            f"transmission_{track_type}"
        ] = transmission

        output[
            f"transmission_error_{track_type}"
        ] = transmission_error

    return pd.DataFrame(output)


# ============================================================
# Plotting
# ============================================================

def plot_transmission_separate(
    transmission_df,
    detector,
    working_point,
    output_dir,
):
    """
    Make one independent plot for 3p and one for 4p.

    Each plot gets its own y-axis scaling, preventing the typically
    larger 3p transmission values from washing out the 4p curve.
    """

    for track_type in ("3p", "4p"):

        y_column = f"transmission_{track_type}"

        fig, ax = plt.subplots(
            figsize=(8, 6)
        )

        error_column = (
            f"transmission_error_{track_type}"
        )

        if error_column in transmission_df:
            ax.errorbar(
                transmission_df["chi2_ves_cut"],
                transmission_df[y_column],
                yerr=transmission_df[error_column],
                fmt="o-",
                markersize=3,
                linewidth=1.5,
                capsize=2,
                label=f"{track_type} cut",
            )
        else:
            ax.plot(
                transmission_df["chi2_ves_cut"],
                transmission_df[y_column],
                marker="o",
                markersize=3,
                linewidth=1.5,
                label=f"{track_type} cut",
            )

        ax.set_xlabel(
            r"VES $\chi^2$ cut"
        )

        ax.set_ylabel(
            "Transmission (VES rate / FS rate)"
        )

        ax.set_title(
            f"{detector} — WP {working_point} — {track_type}"
        )

        #if track_type == "3p":
         #   ax.set_xlim(left=0)

        ax.set_xlim(
            VES_PLOT_MIN,
            CHI2_MAX,
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.legend()

        fig.tight_layout()

        output_file = (
            output_dir
            / f"transmission_{track_type}_along_fit.png"
        )

        fig.savefig(
            output_file,
            dpi=150,
        )

        plt.close(fig)

        print(f"Saved: {output_file}")


def plot_from_parquet(
    parquet_file,
    output_dir,
):
    """
    Read an existing transmission parquet and remake the 3p and 4p
    plots without fitting, ROOT reading, or recalculating anything.
    """

    print(f"Reading: {parquet_file}")

    transmission_df = pd.read_parquet(
        parquet_file
    )

    required_columns = {
        "detector",
        "working_point",
        "chi2_ves_cut",
        "transmission_3p",
        "transmission_4p",
    }

    missing = (
        required_columns
        - set(transmission_df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing columns in {parquet_file}: "
            f"{sorted(missing)}"
        )

    if transmission_df.empty:
        raise ValueError(
            f"Parquet file is empty: {parquet_file}"
        )

    detector_values = (
        transmission_df["detector"]
        .dropna()
        .unique()
    )

    working_point_values = (
        transmission_df["working_point"]
        .dropna()
        .unique()
    )

    if len(detector_values) != 1:
        raise ValueError(
            f"Expected exactly one detector in "
            f"{parquet_file}, found {detector_values}"
        )

    if len(working_point_values) != 1:
        raise ValueError(
            f"Expected exactly one working point in "
            f"{parquet_file}, found {working_point_values}"
        )

    detector = str(detector_values[0])
    working_point = int(working_point_values[0])

    transmission_df = transmission_df.sort_values(
        "chi2_ves_cut"
    )

    plot_transmission_separate(
        transmission_df,
        detector,
        working_point,
        output_dir,
    )


def find_transmission_parquets(
    output_dir,
    working_points=None,
):
    """
    Find existing transmission parquets below output_dir.
    """

    files = sorted(
        output_dir.glob(
            "WP*/transmission_3p4p_along_fit.parquet"
        )
    )

    if working_points is None:
        return files

    working_points = set(
        working_points
    )

    selected = []

    for parquet_file in files:
        try:
            wp = int(
                parquet_file.parent.name.removeprefix(
                    "WP"
                )
            )
        except ValueError:
            continue

        if wp in working_points:
            selected.append(parquet_file)

    return selected


# ============================================================
# Arguments
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate transmission along the fitted chi2 relation "
            "and make separate 3p/4p plots."
        )
    )

    parser.add_argument(
        "--detector",
        default=DEFAULT_DETECTOR,
        choices=sorted(RUN_RANGES),
    )

    parser.add_argument(
        "--working-points",
        type=int,
        nargs="+",
        default=None,
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--plot-from-parquet",
        action="store_true",
        help=(
            "Skip fitting and ROOT counting. Read existing "
            "transmission_3p4p_along_fit.parquet files and "
            "only remake the separate 3p/4p plots."
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_arguments()

    detector_input_dir = (
        args.input_dir
        / args.detector
    )

    output_dir = (
        args.output_dir
        / args.detector
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Parquet-only plotting mode
    # --------------------------------------------------------

    if args.plot_from_parquet:

        parquet_files = find_transmission_parquets(
            output_dir,
            args.working_points,
        )

        if not parquet_files:
            raise RuntimeError(
                "No transmission_3p4p_along_fit.parquet files "
                f"found below {output_dir}"
            )

        print(
            "\nParquet-only plotting mode:"
        )

        for parquet_file in parquet_files:

            wp_output_dir = (
                parquet_file.parent
            )

            plot_from_parquet(
                parquet_file,
                wp_output_dir,
            )

        return

    # --------------------------------------------------------
    # Normal calculation mode
    # --------------------------------------------------------

    if args.working_points is None:

        working_points = (
            find_common_working_points(
                detector_input_dir
            )
        )

    else:
        working_points = sorted(
            set(args.working_points)
        )

    if not working_points:
        raise RuntimeError(
            "No common scan working points found in "
            f"{detector_input_dir}"
        )

    for working_point in working_points:

        print(
            f"\nWorking point: {working_point}"
        )

        scan_ves_file, scan_fs_file = (
            get_parquet_files(
                detector_input_dir,
                working_point,
            )
        )

        ratio_df = calculate_ratio_table(
            load_scan(scan_ves_file),
            load_scan(scan_fs_file),
        )

        fit_parameters = fit_ratio_one_contour(
            ratio_df,
            working_point,
        )

        ves_cuts = make_chi2_cuts()

        ves_cuts = ves_cuts[
            ves_cuts >= VES_PLOT_MIN
        ]

        fs_fit_cuts = np.polyval(
            fit_parameters,
            ves_cuts,
        )

        ves_rates = (
            count_runs_for_working_point(
                args.detector,
                "VES",
                working_point,
                ves_cuts,
            )
        )

        fs_rates = (
            count_runs_for_working_point(
                args.detector,
                "FS",
                working_point,
                fs_fit_cuts,
            )
        )

        transmission_df = (
            build_transmission_table(
                ves_rates,
                fs_rates,
                fit_parameters,
            )
        )

        transmission_df.insert(
            0,
            "working_point",
            working_point,
        )

        transmission_df.insert(
            0,
            "detector",
            args.detector,
        )

        wp_output_dir = (
            output_dir
            / f"WP{working_point}"
        )

        wp_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        parquet_file = (
            wp_output_dir
            / "transmission_3p4p_along_fit.parquet"
        )

        transmission_df.to_parquet(
            parquet_file,
            index=False,
            compression="zstd",
        )

        print(
            f"Saved: {parquet_file}"
        )

        # Make separate 3p and 4p plots.
        plot_transmission_separate(
            transmission_df,
            args.detector,
            working_point,
            wp_output_dir,
        )


if __name__ == "__main__":
    main()
