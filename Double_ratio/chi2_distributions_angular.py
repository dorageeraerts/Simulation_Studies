#!/usr/bin/env python3

"""
Plot MURAVES chi2 distributions vs angular variables
(Theta_3p, Phi_3p, Theta_4p, Phi_4p), per Working Point (WP),
comparing VES and FS.

For each track type (3p / 4p) there are two chi2 values
(xy and xz projections). Each is correlated against the
corresponding track's Theta and Phi.

    3p_xy, 3p_xz  <->  Theta_3p, Phi_3p
    4p_xy, 4p_xz  <->  Theta_4p, Phi_4p

For each detector / WP / chi2-branch / angle-variable:

    1. Chi2-vs-angle heatmap
       2D histogram, chi2 (y) vs angle (x), each angle
       column normalized to a density (so you can see how
       the chi2 *shape* changes across angle, independent
       of how much data is in each angle bin).
       VES and FS side by side, shared color scale.

    2. Chi2-vs-angle profile
       Median + 25-75 / 10-90 percentile bands of chi2 as
       a function of angle bin. VES and FS overlaid.

    3. Theta-Phi acceptance map
       2D map of (Theta, Phi) colored by median chi2.
       This is usually the most physically informative
       plot: it shows whether chi2 quality depends on
       *where* in the detector's angular acceptance the
       track points, e.g. edge / acceptance effects.
       VES and FS side by side, shared color scale.

Output:
    ./chi2_distributions/angular/<detector>/

NOTE: The angular branch names below are a best guess based
on the naming convention of the chi2 branches. Please verify
against your ROOT file (e.g. `uproot.open(f)[TREE_NAME].keys()`)
and adjust ANGLE_BRANCHES if needed.
"""

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
import uproot


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR_TEMPLATE = (
    "/pnfs/iihe/muraves/muraves_DATA/"
    "RECONSTRUCTED/{detector}/v1"
)

SLOWCONTROL_DIR = (
    "/pnfs/iihe/muraves/muraves_DATA/DECOMPRESSED"
)

OUTPUT_DIR = Path("./chi2_distributions/angular")

TREE_NAME = "AnalyzedData"

FILENAME_TEMPLATE = "MURAVES_AnalyzedData_run{run}.root"
SLOWCONTROL_TEMPLATE = "SLOWCONTROL_run{run}"


CHI2_MIN = 0.0
CHI2_MAX = 20.0
CHI2_BIN_WIDTH = 0.2

N_ANGLE_BINS = 30

# Default angular ranges in degrees. Adjust to match your
# detector's convention (e.g. Theta could be 0-90 zenith,
# Phi could be -180-180 or 0-360).
THETA_MIN_DEFAULT = 0.0
THETA_MAX_DEFAULT = 90.0
PHI_MIN_DEFAULT = 0.0
PHI_MAX_DEFAULT = 360.0


CHI2_BRANCHES = {
    "3p_xy": "BestTrack_3p_ChiSquare_xy",
    "3p_xz": "BestTrack_3p_ChiSquare_xz",
    "4p_xy": "BestTrack_4p_ChiSquare_xy",
    "4p_xz": "BestTrack_4p_ChiSquare_xz",
}

# Which angle pair goes with which chi2 branch.
CHI2_TO_TRACK = {
    "3p_xy": "3p",
    "3p_xz": "3p",
    "4p_xy": "4p",
    "4p_xz": "4p",
}

# NOTE: verify these branch names against your file.
ANGLE_BRANCHES = {
    "3p": {
        "Theta": "Theta_3p",
        "Phi": "Phi_3p",
    },
    "4p": {
        "Theta": "Theta_4p",
        "Phi": "Phi_4p",
    },
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


VES_COLOR = "tab:blue"
FS_COLOR = "tab:orange"


# ============================================================
# ARGUMENT / RUN HELPERS
# ============================================================

def parse_run_range(value):
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
            f"Invalid run range: {value}"
        ) from exc

    if stop < start:
        raise argparse.ArgumentTypeError(
            f"Invalid run range: {value}: end must be >= start."
        )

    return start, stop


def parse_range_list(values):
    return (
        None if values is None
        else [parse_run_range(v) for v in values]
    )


def iter_run_numbers(run_ranges):
    for start, stop in run_ranges:
        yield from range(start, stop + 1)


# ============================================================
# SLOW CONTROL (Working Point only - no timestamp needed here)
# ============================================================

def get_working_point(slowcontrol_dir, detector, run):
    """
    Return the integer Working Point from the first valid
    slow-control line, or None.
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
                    print(f"  [warn] malformed slow-control line in {slowcontrol_file}")
                    return None

                try:
                    working_point = float(fields[5])
                except ValueError:
                    print(f"  [warn] invalid Working Point in {slowcontrol_file}: {fields[5]}")
                    return None

                if not np.isfinite(working_point) or not working_point.is_integer():
                    print(f"  [warn] invalid Working Point value in {slowcontrol_file}")
                    return None

                return int(working_point)

    except Exception as exc:
        print(f"  [warn] could not read {slowcontrol_file}: {exc}")
        return None

    print(f"  [warn] no valid lines in {slowcontrol_file}")
    return None


# ============================================================
# ROOT DATA
# ============================================================

def all_branches_needed():
    branches = set(CHI2_BRANCHES.values())
    for track in ANGLE_BRANCHES.values():
        branches.update(track.values())
    return list(branches)


def read_run(file_path):
    branches = all_branches_needed()

    try:
        with uproot.open(file_path) as f:
            if TREE_NAME not in f:
                print(f"  [warn] tree '{TREE_NAME}' not found")
                return None

            tree = f[TREE_NAME]
            missing = set(branches) - set(tree.keys())
            if missing:
                print(f"  [warn] missing branches: {sorted(missing)}")
                return None

            return tree.arrays(branches, library="np")

    except Exception as exc:
        print(f"  [warn] could not read {file_path}: {exc}")
        return None


def get_valid_pairs(chi2_values, angle_values, angle_min, angle_max):
    """
    Return (chi2, angle) arrays restricted to entries where
    both chi2 and angle are finite and in-range.
    """

    chi2_values = np.asarray(chi2_values, dtype=float)
    angle_values = np.asarray(angle_values, dtype=float)

    valid = (
        np.isfinite(chi2_values)
        & np.isfinite(angle_values)
        & (chi2_values >= CHI2_MIN)
        & (chi2_values <= CHI2_MAX)
        & (angle_values >= angle_min)
        & (angle_values <= angle_max)
    )

    return chi2_values[valid], angle_values[valid]


def get_valid_triplet(
    chi2_values, theta_values, phi_values,
    theta_min, theta_max, phi_min, phi_max,
):
    """
    Return matched (chi2, theta, phi) arrays restricted to
    entries where chi2, theta, AND phi are all finite and
    in-range, using a single joint mask so the three arrays
    stay event-aligned. This is what the Theta-Phi acceptance
    map needs (it plots one point per event, not per-angle
    marginal distributions).
    """

    chi2_values = np.asarray(chi2_values, dtype=float)
    theta_values = np.asarray(theta_values, dtype=float)
    phi_values = np.asarray(phi_values, dtype=float)

    valid = (
        np.isfinite(chi2_values)
        & np.isfinite(theta_values)
        & np.isfinite(phi_values)
        & (chi2_values >= CHI2_MIN)
        & (chi2_values <= CHI2_MAX)
        & (theta_values >= theta_min)
        & (theta_values <= theta_max)
        & (phi_values >= phi_min)
        & (phi_values <= phi_max)
    )

    return chi2_values[valid], theta_values[valid], phi_values[valid]


def make_chi2_bins():
    n_bins = int(np.ceil((CHI2_MAX - CHI2_MIN) / CHI2_BIN_WIDTH))
    return np.linspace(CHI2_MIN, CHI2_MAX, n_bins + 1)


def make_angle_bins(angle_min, angle_max, n_bins):
    return np.linspace(angle_min, angle_max, n_bins + 1)


def get_valid_angle(angle_values, angle_min, angle_max):
    """
    Return angle values restricted to finite, in-range entries.
    Used for the plain angle-distribution histograms (no chi2
    pairing needed, so no chi2-validity cut is applied here).
    """

    angle_values = np.asarray(angle_values, dtype=float)

    valid = (
        np.isfinite(angle_values)
        & (angle_values >= angle_min)
        & (angle_values <= angle_max)
    )

    return angle_values[valid]


# ============================================================
# 0. ANGLE DISTRIBUTION HISTOGRAMS (Theta / Phi, 3p / 4p)
# ============================================================

def plot_angle_histogram(
    values_ves, values_fs,
    angle_min, angle_max, n_bins,
    angle_label, track_label,
    detector, working_point,
    output_file,
):
    """
    Plain 1D histogram of one angular variable (Theta or Phi)
    for one track type (3p or 4p), VES vs FS overlaid.
    """

    bins = make_angle_bins(angle_min, angle_max, n_bins)

    fig, ax = plt.subplots(figsize=(9, 6))

    if values_ves is not None and len(values_ves) > 0:
        ax.hist(
            values_ves, bins=bins, histtype="step", linewidth=1.8,
            density=True, color=VES_COLOR,
            label=f"VES (N={len(values_ves):,})",
        )

    if values_fs is not None and len(values_fs) > 0:
        ax.hist(
            values_fs, bins=bins, histtype="step", linewidth=1.8,
            density=True, color=FS_COLOR,
            label=f"FS (N={len(values_fs):,})",
        )

    ax.set_xlabel(f"{track_label} {angle_label} [deg]")
    ax.set_ylabel("Density")
    ax.set_xlim(angle_min, angle_max)
    ax.set_title(
        f"{track_label} {angle_label} distribution - "
        f"{detector}, WP{working_point}"
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 1. CHI2-VS-ANGLE HEATMAP (column-normalized density)
# ============================================================

def column_normalized_hist2d(chi2, angle, angle_bins, chi2_bins):
    """
    2D histogram of chi2 (rows) vs angle (columns), where
    each angle column is independently normalized to a
    density over chi2 (integral = 1 per column with data).
    """

    counts, _, _ = np.histogram2d(
        angle, chi2, bins=[angle_bins, chi2_bins]
    )

    chi2_widths = np.diff(chi2_bins)
    col_sums = counts.sum(axis=1, keepdims=True)

    with np.errstate(invalid="ignore", divide="ignore"):
        density = counts / col_sums / chi2_widths[np.newaxis, :]

    density[~np.isfinite(density)] = np.nan

    return density, counts.sum(axis=1)


def plot_chi2_vs_angle_heatmap(
    chi2_ves, angle_ves,
    chi2_fs, angle_fs,
    angle_min, angle_max,
    angle_label,
    detector, working_point, branch_name, track_label,
    output_file,
):
    chi2_bins = make_chi2_bins()
    angle_bins = make_angle_bins(angle_min, angle_max, N_ANGLE_BINS)

    density_ves, n_ves = column_normalized_hist2d(
        chi2_ves, angle_ves, angle_bins, chi2_bins
    )
    density_fs, n_fs = column_normalized_hist2d(
        chi2_fs, angle_fs, angle_bins, chi2_bins
    )

    finite_vals = []
    for d in (density_ves, density_fs):
        v = d[np.isfinite(d)]
        if len(v) > 0:
            finite_vals.append(v)

    vmax = np.max(np.concatenate(finite_vals)) if finite_vals else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 6), sharey=True,
        gridspec_kw={"width_ratios": [1, 1]},
    )

    mesh = None
    for ax, density, n, label, entries in (
        (axes[0], density_ves, n_ves, "VES", int(n_ves.sum())),
        (axes[1], density_fs, n_fs, "FS", int(n_fs.sum())),
    ):
        mesh = ax.pcolormesh(
            angle_bins, chi2_bins, density.T,
            shading="auto", cmap="viridis", vmin=0, vmax=vmax,
        )
        ax.set_title(f"{label} (N={entries:,})")
        ax.set_xlabel(f"{angle_label} [deg]")
        ax.set_ylim(CHI2_MIN, CHI2_MAX)
        ax.set_xlim(angle_min, angle_max)
        ax.grid(True, alpha=0.2, linestyle="--")

    axes[0].set_ylabel(r"$\chi^2$")

    cbar = fig.colorbar(mesh, ax=axes, fraction=0.046, pad=0.02)
    cbar.set_label("Density (normalized per angle bin)")

    fig.suptitle(
        f"{branch_name} vs {track_label} {angle_label} - "
        f"{detector}, WP{working_point}"
    )

    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 2. CHI2-VS-ANGLE PROFILE (median + quantile bands)
# ============================================================

def binned_quantiles(chi2, angle, angle_bins):
    bin_idx = np.digitize(angle, angle_bins) - 1
    n_bins = len(angle_bins) - 1

    q10 = np.full(n_bins, np.nan)
    q25 = np.full(n_bins, np.nan)
    q50 = np.full(n_bins, np.nan)
    q75 = np.full(n_bins, np.nan)
    q90 = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = bin_idx == i
        values = chi2[mask]
        counts[i] = len(values)

        if len(values) == 0:
            continue

        q10[i] = np.percentile(values, 10)
        q25[i] = np.percentile(values, 25)
        q50[i] = np.percentile(values, 50)
        q75[i] = np.percentile(values, 75)
        q90[i] = np.percentile(values, 90)

    return q10, q25, q50, q75, q90, counts


def plot_chi2_vs_angle_profile(
    chi2_ves, angle_ves,
    chi2_fs, angle_fs,
    angle_min, angle_max,
    angle_label,
    detector, working_point, branch_name, track_label,
    output_file,
):
    angle_bins = make_angle_bins(angle_min, angle_max, N_ANGLE_BINS)
    centers = 0.5 * (angle_bins[:-1] + angle_bins[1:])

    fig, ax = plt.subplots(figsize=(11, 6))

    for chi2, angle, color, marker, label in (
        (chi2_ves, angle_ves, VES_COLOR, "o", "VES"),
        (chi2_fs, angle_fs, FS_COLOR, "s", "FS"),
    ):
        if len(chi2) == 0:
            continue

        q10, q25, q50, q75, q90, counts = binned_quantiles(
            chi2, angle, angle_bins
        )

        ax.fill_between(
            centers, q10, q90, color=color, alpha=0.12,
            label=f"{label} 10-90 pct",
        )
        ax.fill_between(
            centers, q25, q75, color=color, alpha=0.25,
            label=f"{label} 25-75 pct",
        )
        ax.plot(
            centers, q50, color=color, marker=marker,
            linewidth=1.8, markersize=4,
            label=f"{label} median (N={len(chi2):,})",
        )

    ax.set_xlabel(f"{angle_label} [deg]")
    ax.set_ylabel(r"$\chi^2$")
    ax.set_xlim(angle_min, angle_max)
    ax.set_ylim(CHI2_MIN, CHI2_MAX)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(ncol=2, loc="best", fontsize=8)

    ax.set_title(
        f"{branch_name} vs {track_label} {angle_label} - "
        f"{detector}, WP{working_point}"
    )

    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 3. THETA-PHI ACCEPTANCE MAP COLORED BY MEDIAN CHI2
# ============================================================

def theta_phi_median_chi2_map(
    chi2, theta, phi, theta_bins, phi_bins
):
    """
    Median chi2 per (Theta, Phi) bin, computed via
    digitize + per-bin percentile (robust to outliers,
    unlike a mean).
    """

    theta_idx = np.digitize(theta, theta_bins) - 1
    phi_idx = np.digitize(phi, phi_bins) - 1

    n_theta = len(theta_bins) - 1
    n_phi = len(phi_bins) - 1

    median_map = np.full((n_theta, n_phi), np.nan)
    counts = np.zeros((n_theta, n_phi), dtype=int)

    for i in range(n_theta):
        for j in range(n_phi):
            mask = (theta_idx == i) & (phi_idx == j)
            values = chi2[mask]
            counts[i, j] = len(values)
            if len(values) > 0:
                median_map[i, j] = np.median(values)

    return median_map, counts


def plot_theta_phi_chi2_map(
    chi2_ves, theta_ves, phi_ves,
    chi2_fs, theta_fs, phi_fs,
    theta_min, theta_max, phi_min, phi_max,
    detector, working_point, branch_name, track_label,
    output_file,
    n_theta_bins=25, n_phi_bins=36,
):
    theta_bins = make_angle_bins(theta_min, theta_max, n_theta_bins)
    phi_bins = make_angle_bins(phi_min, phi_max, n_phi_bins)

    map_ves, n_ves = theta_phi_median_chi2_map(
        chi2_ves, theta_ves, phi_ves, theta_bins, phi_bins
    )
    map_fs, n_fs = theta_phi_median_chi2_map(
        chi2_fs, theta_fs, phi_fs, theta_bins, phi_bins
    )

    finite_vals = []
    for m in (map_ves, map_fs):
        v = m[np.isfinite(m)]
        if len(v) > 0:
            finite_vals.append(v)

    if finite_vals:
        all_vals = np.concatenate(finite_vals)
        vmin, vmax = np.percentile(all_vals, [2, 98])
    else:
        vmin, vmax = CHI2_MIN, CHI2_MAX

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 7), subplot_kw={"projection": "polar"}
    )

    mesh = None
    for ax, med_map, n_map, label, total in (
        (axes[0], map_ves, n_ves, "VES", int(n_ves.sum())),
        (axes[1], map_fs, n_fs, "FS", int(n_fs.sum())),
    ):
        phi_rad = np.deg2rad(phi_bins)
        mesh = ax.pcolormesh(
            phi_rad, theta_bins, med_map,
            shading="auto", cmap="viridis", vmin=vmin, vmax=vmax,
        )
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title(f"{label} (N={total:,})", pad=20)

    cbar = fig.colorbar(mesh, ax=axes, fraction=0.046, pad=0.08)
    cbar.set_label(r"Median $\chi^2$")

    fig.suptitle(
        f"{branch_name}: {track_label} Theta-Phi acceptance map - "
        f"{detector}, WP{working_point}"
    )

    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Plot MURAVES chi2 distributions vs angular "
            "variables (Theta/Phi, 3p/4p), comparing VES and FS."
        )
    )

    parser.add_argument(
        "--detector", choices=sorted(RUN_RANGES.keys()),
        help="Detector to process, e.g. ROSSO",
    )
    parser.add_argument(
        "--detectors", nargs="+", choices=sorted(RUN_RANGES.keys()),
        help="Multiple detectors to process",
    )
    parser.add_argument(
        "--ves-ranges", nargs="+", metavar="START-END",
        help="Override default VES run ranges",
    )
    parser.add_argument(
        "--fs-ranges", nargs="+", metavar="START-END",
        help="Override default FS run ranges",
    )
    parser.add_argument(
        "--chi2-max", type=float, default=CHI2_MAX,
        help=f"Maximum chi2 shown, default={CHI2_MAX}",
    )
    parser.add_argument(
        "--theta-range", nargs=2, type=float,
        default=[THETA_MIN_DEFAULT, THETA_MAX_DEFAULT],
        metavar=("MIN", "MAX"),
        help="Theta range in degrees",
    )
    parser.add_argument(
        "--phi-range", nargs=2, type=float,
        default=[PHI_MIN_DEFAULT, PHI_MAX_DEFAULT],
        metavar=("MIN", "MAX"),
        help="Phi range in degrees",
    )
    parser.add_argument(
        "--n-angle-bins", type=int, default=N_ANGLE_BINS,
        help=f"Number of angle bins for heatmap/profile, default={N_ANGLE_BINS}",
    )

    args = parser.parse_args()

    if args.detector and args.detectors:
        parser.error("Use either --detector or --detectors, not both.")

    detectors = (
        [args.detector] if args.detector
        else args.detectors if args.detectors
        else ["ROSSO"]
    )

    if args.chi2_max <= CHI2_MIN:
        parser.error("--chi2-max must be > 0")

    return (
        args, detectors,
        parse_range_list(args.ves_ranges),
        parse_range_list(args.fs_ranges),
    )


# ============================================================
# MAIN
# ============================================================

def main():
    global CHI2_MAX
    global N_ANGLE_BINS

    args, detectors, ves_override, fs_override = parse_arguments()

    CHI2_MAX = args.chi2_max
    N_ANGLE_BINS = args.n_angle_bins
    theta_min, theta_max = args.theta_range
    phi_min, phi_max = args.phi_range

    print("=" * 70)
    print("MURAVES CHI2 vs ANGLE DISTRIBUTIONS")
    print("=" * 70)
    print(f"Detectors  : {', '.join(detectors)}")
    print(f"Chi2 range : {CHI2_MIN} -> {CHI2_MAX}")
    print(f"Theta range: {theta_min} -> {theta_max} deg")
    print(f"Phi range  : {phi_min} -> {phi_max} deg")
    print("=" * 70)

    for detector in detectors:
        print()
        print("#" * 70)
        print(f"DETECTOR: {detector}")
        print("#" * 70)

        data_dir = Path(DATA_DIR_TEMPLATE.format(detector=detector))
        detector_output = OUTPUT_DIR / detector
        detector_output.mkdir(parents=True, exist_ok=True)

        run_ranges = {
            "VES": ves_override if ves_override is not None
                   else RUN_RANGES[detector]["VES"],
            "FS": fs_override if fs_override is not None
                  else RUN_RANGES[detector]["FS"],
        }

        # data[data_type][WP][chi2_branch] = {"chi2": [...], "theta": [...], "phi": [...]}
        data = {"VES": {}, "FS": {}}

        # angle_data[data_type][WP][track]["Theta"/"Phi"] = [arrays...]
        # Kept separate from `data` (and keyed by track, not by
        # chi2 branch) so each event's angle is only counted once,
        # even though 3p_xy/3p_xz (and 4p_xy/4p_xz) share the same
        # Theta/Phi values.
        angle_data = {"VES": {}, "FS": {}}

        for data_type in ["VES", "FS"]:
            print()
            print(f"========== {detector} / {data_type} ==========")

            for run in iter_run_numbers(run_ranges[data_type]):
                file_path = data_dir / FILENAME_TEMPLATE.format(run=run)

                if not file_path.is_file():
                    continue

                working_point = get_working_point(SLOWCONTROL_DIR, detector, run)
                if working_point is None:
                    print(f"Run {run}: no Working Point, skipping")
                    continue

                arrays = read_run(file_path)
                if arrays is None:
                    continue

                print(f"Run {run}: WP{working_point}, {file_path.name}")

                if working_point not in data[data_type]:
                    data[data_type][working_point] = {
                        key: {
                            "chi2": [], "theta": [],
                            "phi_chi2": [], "phi": [],
                            "joint_chi2": [], "joint_theta": [], "joint_phi": [],
                        }
                        for key in CHI2_BRANCHES
                    }

                for short_name, branch_name in CHI2_BRANCHES.items():
                    track = CHI2_TO_TRACK[short_name]
                    theta_branch = ANGLE_BRANCHES[track]["Theta"]
                    phi_branch = ANGLE_BRANCHES[track]["Phi"]

                    # Independent filtering for the marginal
                    # Theta-only and Phi-only plots (heatmap,
                    # profile) — keeps full available stats for
                    # each, since a cut on Phi shouldn't reduce
                    # entries shown in the Theta-vs-chi2 plots
                    # and vice versa.
                    chi2_valid, theta_valid = get_valid_pairs(
                        arrays[branch_name], arrays[theta_branch],
                        theta_min, theta_max,
                    )
                    chi2_valid2, phi_valid = get_valid_pairs(
                        arrays[branch_name], arrays[phi_branch],
                        phi_min, phi_max,
                    )

                    # Joint filtering for the Theta-Phi acceptance
                    # map — chi2, theta, phi must all come from the
                    # same event, using a single mask so the three
                    # arrays stay aligned.
                    chi2_joint, theta_joint, phi_joint = get_valid_triplet(
                        arrays[branch_name], arrays[theta_branch], arrays[phi_branch],
                        theta_min, theta_max, phi_min, phi_max,
                    )

                    bucket = data[data_type][working_point][short_name]
                    if len(chi2_valid) > 0:
                        bucket["chi2"].append(chi2_valid)
                        bucket["theta"].append(theta_valid)
                    if len(phi_valid) > 0:
                        bucket["phi_chi2"].append(chi2_valid2)
                        bucket["phi"].append(phi_valid)
                    if len(chi2_joint) > 0:
                        bucket["joint_chi2"].append(chi2_joint)
                        bucket["joint_theta"].append(theta_joint)
                        bucket["joint_phi"].append(phi_joint)

                # ------------------------------------------------
                # Plain angle distributions (once per track per
                # run, independent of chi2 and of which chi2
                # branch is being processed).
                # ------------------------------------------------

                if working_point not in angle_data[data_type]:
                    angle_data[data_type][working_point] = {
                        track: {"Theta": [], "Phi": []}
                        for track in ANGLE_BRANCHES
                    }

                for track, angle_branches in ANGLE_BRANCHES.items():
                    theta_only = get_valid_angle(
                        arrays[angle_branches["Theta"]], theta_min, theta_max
                    )
                    phi_only = get_valid_angle(
                        arrays[angle_branches["Phi"]], phi_min, phi_max
                    )

                    bucket_angle = angle_data[data_type][working_point][track]
                    if len(theta_only) > 0:
                        bucket_angle["Theta"].append(theta_only)
                    if len(phi_only) > 0:
                        bucket_angle["Phi"].append(phi_only)

        # ====================================================
        # PLOTS
        # ====================================================

        print()
        print(f"Creating angular comparison plots for {detector}...")

        working_points = sorted(
            set(data["VES"]) | set(data["FS"])
            | set(angle_data["VES"]) | set(angle_data["FS"])
        )

        for working_point in working_points:

            # ---- Plain angle distribution histograms ----
            # One per (track, angle) combination: 3p-Theta,
            # 3p-Phi, 4p-Theta, 4p-Phi.

            empty_angle_bucket = {"Theta": [], "Phi": []}

            for track, angle_ranges in (
                ("3p", (theta_min, theta_max, "Theta")),
                ("3p", (phi_min, phi_max, "Phi")),
                ("4p", (theta_min, theta_max, "Theta")),
                ("4p", (phi_min, phi_max, "Phi")),
            ):
                angle_min_val, angle_max_val, angle_name = angle_ranges

                ves_angle_bucket = angle_data["VES"].get(
                    working_point, {}
                ).get(track, empty_angle_bucket)
                fs_angle_bucket = angle_data["FS"].get(
                    working_point, {}
                ).get(track, empty_angle_bucket)

                values_ves = (
                    np.concatenate(ves_angle_bucket[angle_name])
                    if ves_angle_bucket[angle_name] else np.array([])
                )
                values_fs = (
                    np.concatenate(fs_angle_bucket[angle_name])
                    if fs_angle_bucket[angle_name] else np.array([])
                )

                if len(values_ves) == 0 and len(values_fs) == 0:
                    continue

                angle_hist_file = detector_output / (
                    f"angle_hist_{track}_{angle_name}_"
                    f"WP{working_point}_VES_FS.png"
                )

                plot_angle_histogram(
                    values_ves, values_fs,
                    angle_min_val, angle_max_val, N_ANGLE_BINS,
                    angle_name, track,
                    detector, working_point,
                    angle_hist_file,
                )

                print(
                    f"  WP{working_point} {track} {angle_name}: "
                    f"VES={len(values_ves):,}, FS={len(values_fs):,} "
                    f"-> {angle_hist_file}"
                )

            for short_name, branch_name in CHI2_BRANCHES.items():
                track = CHI2_TO_TRACK[short_name]

                empty_bucket = {
                    "chi2": [], "theta": [],
                    "phi_chi2": [], "phi": [],
                    "joint_chi2": [], "joint_theta": [], "joint_phi": [],
                }

                ves_bucket = data["VES"].get(working_point, {}).get(
                    short_name, empty_bucket
                )
                fs_bucket = data["FS"].get(working_point, {}).get(
                    short_name, empty_bucket
                )

                chi2_theta_ves = (
                    np.concatenate(ves_bucket["chi2"]) if ves_bucket["chi2"] else np.array([])
                )
                theta_ves = (
                    np.concatenate(ves_bucket["theta"]) if ves_bucket["theta"] else np.array([])
                )
                chi2_theta_fs = (
                    np.concatenate(fs_bucket["chi2"]) if fs_bucket["chi2"] else np.array([])
                )
                theta_fs = (
                    np.concatenate(fs_bucket["theta"]) if fs_bucket["theta"] else np.array([])
                )

                chi2_phi_ves = (
                    np.concatenate(ves_bucket.get("phi_chi2", []))
                    if ves_bucket.get("phi_chi2") else np.array([])
                )
                phi_ves = (
                    np.concatenate(ves_bucket["phi"]) if ves_bucket["phi"] else np.array([])
                )
                chi2_phi_fs = (
                    np.concatenate(fs_bucket.get("phi_chi2", []))
                    if fs_bucket.get("phi_chi2") else np.array([])
                )
                phi_fs = (
                    np.concatenate(fs_bucket["phi"]) if fs_bucket["phi"] else np.array([])
                )

                if (
                    len(chi2_theta_ves) == 0 and len(chi2_theta_fs) == 0
                    and len(chi2_phi_ves) == 0 and len(chi2_phi_fs) == 0
                ):
                    print(f"  WP{working_point} {short_name}: no entries, skipping")
                    continue

                # ---- Theta heatmap + profile ----
                theta_heatmap_file = detector_output / (
                    f"chi2_vs_theta_heatmap_{short_name}_WP{working_point}_VES_FS.png"
                )
                plot_chi2_vs_angle_heatmap(
                    chi2_theta_ves, theta_ves, chi2_theta_fs, theta_fs,
                    theta_min, theta_max, "Theta",
                    detector, working_point, branch_name, track,
                    theta_heatmap_file,
                )

                theta_profile_file = detector_output / (
                    f"chi2_vs_theta_profile_{short_name}_WP{working_point}_VES_FS.png"
                )
                plot_chi2_vs_angle_profile(
                    chi2_theta_ves, theta_ves, chi2_theta_fs, theta_fs,
                    theta_min, theta_max, "Theta",
                    detector, working_point, branch_name, track,
                    theta_profile_file,
                )

                # ---- Phi heatmap + profile ----
                phi_heatmap_file = detector_output / (
                    f"chi2_vs_phi_heatmap_{short_name}_WP{working_point}_VES_FS.png"
                )
                plot_chi2_vs_angle_heatmap(
                    chi2_phi_ves, phi_ves, chi2_phi_fs, phi_fs,
                    phi_min, phi_max, "Phi",
                    detector, working_point, branch_name, track,
                    phi_heatmap_file,
                )

                phi_profile_file = detector_output / (
                    f"chi2_vs_phi_profile_{short_name}_WP{working_point}_VES_FS.png"
                )
                plot_chi2_vs_angle_profile(
                    chi2_phi_ves, phi_ves, chi2_phi_fs, phi_fs,
                    phi_min, phi_max, "Phi",
                    detector, working_point, branch_name, track,
                    phi_profile_file,
                )

                # ---- Theta-Phi acceptance map ----
                # Uses the joint (chi2, theta, phi) triplets collected
                # per run with a single mask (see get_valid_triplet),
                # so the three arrays are guaranteed event-aligned —
                # no truncation, no independent-filter mismatch.

                acceptance_file = detector_output / (
                    f"chi2_theta_phi_map_{short_name}_WP{working_point}_VES_FS.png"
                )

                chi2_map_ves = (
                    np.concatenate(ves_bucket["joint_chi2"])
                    if ves_bucket["joint_chi2"] else np.array([])
                )
                theta_map_ves = (
                    np.concatenate(ves_bucket["joint_theta"])
                    if ves_bucket["joint_theta"] else np.array([])
                )
                phi_map_ves = (
                    np.concatenate(ves_bucket["joint_phi"])
                    if ves_bucket["joint_phi"] else np.array([])
                )

                chi2_map_fs = (
                    np.concatenate(fs_bucket["joint_chi2"])
                    if fs_bucket["joint_chi2"] else np.array([])
                )
                theta_map_fs = (
                    np.concatenate(fs_bucket["joint_theta"])
                    if fs_bucket["joint_theta"] else np.array([])
                )
                phi_map_fs = (
                    np.concatenate(fs_bucket["joint_phi"])
                    if fs_bucket["joint_phi"] else np.array([])
                )

                plot_theta_phi_chi2_map(
                    chi2_map_ves, theta_map_ves, phi_map_ves,
                    chi2_map_fs, theta_map_fs, phi_map_fs,
                    theta_min, theta_max, phi_min, phi_max,
                    detector, working_point, branch_name, track,
                    acceptance_file,
                )

                print(f"  WP{working_point} {short_name}:")
                print(f"      VES entries (theta-marginal): {len(chi2_theta_ves):,}")
                print(f"      FS  entries (theta-marginal): {len(chi2_theta_fs):,}")
                print(f"      VES entries (joint theta+phi): {len(chi2_map_ves):,}")
                print(f"      FS  entries (joint theta+phi): {len(chi2_map_fs):,}")
                print(f"      theta heatmap -> {theta_heatmap_file}")
                print(f"      theta profile -> {theta_profile_file}")
                print(f"      phi heatmap   -> {phi_heatmap_file}")
                print(f"      phi profile   -> {phi_profile_file}")
                print(f"      theta-phi map -> {acceptance_file}")

        print()
        print(f"Output directory: {detector_output}")

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()