#!/usr/bin/env python3

"""
chi2_control_region.py

Read the MURAVES reconstructed track ROOT files (MURAVES_AnalyzedData_Run*.root)
once and produce Parquet files containing the control-region track-rate (sum) versus chi2 cut.

Only finite, non-negative chi2 values are accepted. Track directions are from the 3p-tracks:
PHI_BRANCH = "Phi_3p" and THETA_BRANCH = "Theta_3p". Chi2-cut is performed on the 4p-tracks:
CHI2_XY = "BestTrack_4p_ChiSquare_xy" and CHI2_XZ = "BestTrack_4p_ChiSquare_xz".

Selection convention:
    CHI2_XY < chi2_cut AND CHI2_XZ < chi2_cut
Equivalently:
    max(CHI2_XY, CHI2_XZ) < chi2_cut

The output is grouped by:
    detector, data_type (VES / FS), working_point, chi2_cut

Command-line examples:

    # Default control region: AZ 175-185, EL 25-30
    python chi2_control_region.py --detector ROSSO

    # Custom control region
    python chi2_control_region.py \
        --detector ROSSO \
        --az-range 170 190 \
        --el-range 20 35

    # Select only a few runs
    python chi2_control_region.py \
        --detector ROSSO NERO BLU \
        --ves-ranges 4000-4010 5000-5010 \
        --fs-ranges 10061-10070
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import uproot


# ============================================================
# USER CONFIGURATION
# ============================================================

# Default angular control region
MIN_AZ, MAX_AZ = 172.0, 188.0
MIN_EL, MAX_EL = 25.0, 32.0

# Bin sizes
D_AZ, D_EL = 1.0, 1.0

# Chi2 cuts
CHI2_MIN, CHI2_MAX, CHI2_STEP = 0.0, 12.0, 0.1

# Paths
DATA_DIR_TEMPLATE = "/pnfs/iihe/muraves/muraves_DATA/RECONSTRUCTED/{detector}/v1"
SLOWCONTROL_DIR = "/pnfs/iihe/muraves/muraves_DATA/DECOMPRESSED"
OUTPUT_DIR = Path("./muraves_chi2_scan")

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
    "NERO": {   "VES": [(2500, 5600), (7894, 11990)], 
                "FS": [(5636, 7893)]},
    "ROSSO": {  "VES": [(4000, 10060), (13458, 14792)], 
                "FS": [(10061, 13457)]},
    "BLU": {    "VES": [(2800, 8818)], 
                "FS": [(8819, 9925)]},
}


# ============================================================
# HELPERS
# ============================================================

def make_edges(min_val, max_val, bin_size):
    """Build approximately bin_size-wide bin edges covering [min_val, max_val]."""
    if bin_size <= 0: raise ValueError("Bin size must be > 0.")
    if max_val <= min_val: raise ValueError("max_val must be > min_val.")
    n_bins = int(np.ceil((max_val - min_val) / bin_size))
    return np.linspace(min_val, max_val, n_bins + 1)


def make_chi2_cuts():
    """Generate stable floating-point chi2 cuts: 0.0, 0.1, ..., 12.0."""
    n = int(round((CHI2_MAX - CHI2_MIN) / CHI2_STEP))
    return CHI2_MIN + np.arange(n + 1, dtype=float) * CHI2_STEP


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
            raise argparse.ArgumentTypeError(f"Invalid run number: {value}") from exc
        return run, run

    try:
        start, stop = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid run range: {value}. Use START-END, e.g. 4000-4010."
        ) from exc

    if stop < start:
        raise argparse.ArgumentTypeError(
            f"Invalid run range: {value}: end must be >= start."
        )

    return start, stop


def parse_range_list(values):
    """Convert command-line ranges into a list of (start, stop) tuples."""
    if values is None: return None
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
    slowcontrol_file = Path(slowcontrol_dir) / detector / SLOWCONTROL_TEMPLATE.format(run=run)

    if not slowcontrol_file.is_file():
        print(f"  [warn] slow-control file not found: {slowcontrol_file}")
        return None

    try:
        with open(slowcontrol_file, "r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line: continue

                fields = line.split()
                if len(fields) <= 5:
                    print(f"  [warn] malformed slow-control line in {slowcontrol_file}")
                    return None

                try:
                    working_point = float(fields[5])
                except ValueError:
                    print(f"  [warn] invalid Working Point in {slowcontrol_file}: {fields[5]}")
                    return None

                if not np.isfinite(working_point):
                    print(f"  [warn] non-finite Working Point in {slowcontrol_file}: {fields[5]}")
                    return None

                if not working_point.is_integer():
                    print(f"  [warn] non-integer Working Point in {slowcontrol_file}: {fields[5]}")
                    return None

                return int(working_point)

    except Exception as exc:
        print(f"  [warn] could not read {slowcontrol_file}: {exc}")
        return None

    print(f"  [warn] no valid lines in {slowcontrol_file}")
    return None


def format_wp(wp):
    """Human-readable integer Working Point."""
    return str(int(wp))


def get_run_duration_seconds(timestamps):
    """Calculate run duration from timestamp, which is in milliseconds."""
    timestamps = np.asarray(timestamps, dtype=float)
    valid = np.isfinite(timestamps)

    if not np.any(valid): return None

    tmin, tmax = np.min(timestamps[valid]), np.max(timestamps[valid])
    duration = (tmax - tmin) / 1000.0

    if duration <= 0: return None
    return float(duration)


def read_run(file_path):
    """Read only the branches required for this analysis."""
    branches = [PHI_BRANCH, THETA_BRANCH, CHI2_XY, CHI2_XZ, TIMESTAMP_BRANCH]

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

            arrays = tree.arrays(branches, library="np")

    except Exception as exc:
        print(f"  [warn] could not read {file_path}: {exc}")
        return None

    return arrays


def process_run(file_path, min_az, max_az, min_el, max_el):
    """
    Read one ROOT run.

    Returns:
        duration_seconds
        effective_chi2

    effective_chi2 = max(CHI2_XY, CHI2_XZ)

    Only finite, non-negative chi2 values are accepted.
    """
    arrays = read_run(file_path)
    if arrays is None: return None

    timestamps = arrays[TIMESTAMP_BRANCH]
    duration_seconds = get_run_duration_seconds(timestamps)

    if duration_seconds is None:
        print("  [warn] invalid run duration")
        return None

    phi = np.asarray(arrays[PHI_BRANCH], dtype=float)
    theta = np.asarray(arrays[THETA_BRANCH], dtype=float)
    chi2_xy = np.asarray(arrays[CHI2_XY], dtype=float)
    chi2_xz = np.asarray(arrays[CHI2_XZ], dtype=float)

    # Finite + non-negative chi2 selection
    valid = (
        np.isfinite(phi)
        & np.isfinite(theta)
        & np.isfinite(chi2_xy)
        & np.isfinite(chi2_xz)
        & (chi2_xy >= 0.0)
        & (chi2_xz >= 0.0)
    )

    # Angular control region
    control_region = (
        (phi >= min_az)
        & (phi <= max_az)
        & (theta >= min_el)
        & (theta <= max_el)
    )

    valid &= control_region

    if not np.any(valid):
        return duration_seconds, np.empty(0, dtype=float)

    # Effective chi2 = maximum of XY and XZ.
    effective_chi2 = np.maximum(chi2_xy[valid], chi2_xz[valid])

    # Defensive check: negative effective chi2 must never remain.
    if np.any(effective_chi2 < 0.0):
        raise RuntimeError("Internal error: negative effective chi2 remained after selection.")

    return duration_seconds, effective_chi2


def calculate_scan_counts(effective_chi2, cuts):
    """
    Calculate the number of tracks satisfying:
        effective_chi2 < cut
    """
    if len(effective_chi2) == 0:
        return np.zeros(len(cuts), dtype=np.int64)

    sorted_chi2 = np.sort(effective_chi2)

    # side='left' implements strict chi2 < cut.
    counts = np.searchsorted(sorted_chi2, cuts, side="left")
    return counts.astype(np.int64)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Export MURAVES control-region chi2 cuts to Parquet."
    )

    parser.add_argument(
        "--detector",
        choices=sorted(RUN_RANGES.keys()),
        help="Detector to process. For example: --detector ROSSO",
    )

    parser.add_argument(
        "--detectors",
        nargs="+",
        choices=sorted(RUN_RANGES.keys()),
        help="Multiple detectors to process. For example: --detectors NERO ROSSO BLU",
    )

    parser.add_argument(
        "--ves-ranges",
        nargs="+",
        metavar="START-END",
        help="Override the default VES run ranges. Examples: --ves-ranges 4000-4010 5000-5010",
    )

    parser.add_argument(
        "--fs-ranges",
        nargs="+",
        metavar="START-END",
        help="Override the default FS run ranges. Example: --fs-ranges 10061-10070",
    )

    parser.add_argument(
        "--az-range",
        nargs=2,
        type=float,
        metavar=("MIN_AZ", "MAX_AZ"),
        default=(MIN_AZ, MAX_AZ),
        help=f"Azimuth control region. Default: {MIN_AZ} {MAX_AZ}",
    )

    parser.add_argument(
        "--el-range",
        nargs=2,
        type=float,
        metavar=("MIN_EL", "MAX_EL"),
        default=(MIN_EL, MAX_EL),
        help=f"Elevation control region. Default: {MIN_EL} {MAX_EL}",
    )

    args = parser.parse_args()

    # --detector and --detectors are mutually exclusive.
    if args.detector and args.detectors:
        parser.error("Use either --detector or --detectors, not both.")

    if args.az_range[1] <= args.az_range[0]:
        parser.error("--az-range requires MAX_AZ > MIN_AZ")

    if args.el_range[1] <= args.el_range[0]:
        parser.error("--el-range requires MAX_EL > MIN_EL")

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

    return args, detectors, ves_ranges, fs_ranges, min_az, max_az, min_el, max_el


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

    az_edges = make_edges(min_az, max_az, D_AZ)
    el_edges = make_edges(min_el, max_el, D_EL)
    cuts = make_chi2_cuts()

    print()
    print("=" * 70)
    print("MURAVES CONTROL-REGION CHI2 CUT EXPORT")
    print("=" * 70)
    print(f"Detectors        : {', '.join(detectors)}")
    print(f"Azimuth region   : [{min_az}, {max_az}]")
    print(f"Elevation region : [{min_el}, {max_el}]")
    print(f"Azimuth bin      : {D_AZ}")
    print(f"Elevation bin    : {D_EL}")
    print(f"Chi2 cuts        : {cuts[0]:.1f} -> {cuts[-1]:.1f} step {CHI2_STEP}")
    print()
    print("Selection:")
    print("    finite chi2")
    print("    chi2_xy >= 0")
    print("    chi2_xz >= 0")
    print("    CHI2_XY < cut")
    print("    AND")
    print("    CHI2_XZ < cut")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for detector in detectors:
        if detector not in RUN_RANGES:
            raise ValueError(f"Unknown detector: {detector}")

        print()
        print("#" * 70)
        print(f"DETECTOR: {detector}")
        print("#" * 70)

        data_dir = Path(DATA_DIR_TEMPLATE.format(detector=detector))
        detector_output = OUTPUT_DIR / detector / "vesuvius_region"
        detector_output.mkdir(parents=True, exist_ok=True)

        run_ranges = {
            "VES": ves_override if ves_override is not None else RUN_RANGES[detector]["VES"],
            "FS": fs_override if fs_override is not None else RUN_RANGES[detector]["FS"],
        }

        print()
        print("Run ranges:")

        for data_type in ["VES", "FS"]:
            formatted_ranges = ", ".join(
                f"{start}-{stop}" for start, stop in run_ranges[data_type]
            )
            print(f"  {data_type}: {formatted_ranges}")

        # Store information separately for VES and FS.
        # Each WP gets all effective chi2 values and total duration.
        data = {"VES": {}, "FS": {}}

        for data_type in ["VES", "FS"]:
            print()
            print(f"========== {detector} / {data_type} ==========")

            for run in iter_run_numbers(run_ranges[data_type]):
                file_path = data_dir / FILENAME_TEMPLATE.format(run=run)

                if not file_path.is_file():
                    print(f"Run {run}: file not found, skipping")
                    continue

                print(f"Run {run}: {file_path.name}")

                # Working Point
                working_point = get_working_point(SLOWCONTROL_DIR, detector, run)

                if working_point is None:
                    print("  [warn] no WP, skipping run")
                    continue

                print(f"  Working Point: {format_wp(working_point)}")

                # ROOT
                result = process_run(file_path, min_az, max_az, min_el, max_el)

                if result is None:
                    continue

                duration_seconds, effective_chi2 = result

                print(f"  Duration: {duration_seconds:.3f} s")
                print(f"  Tracks in control region: {len(effective_chi2)}")

                # Accumulate by WP
                if working_point not in data[data_type]:
                    data[data_type][working_point] = {"duration": 0.0, "chi2": [], "runs": []}

                wp_data = data[data_type][working_point]
                wp_data["duration"] += duration_seconds

                if len(effective_chi2) > 0:
                    wp_data["chi2"].append(effective_chi2)

                wp_data["runs"].append(run)

        # ====================================================
        # WRITE PARQUET FILES
        # ====================================================

        for data_type in ["VES", "FS"]:
            for working_point, wp_data in sorted(data[data_type].items()):
                duration = wp_data["duration"]

                if wp_data["chi2"]:
                    effective_chi2 = np.concatenate(wp_data["chi2"])
                else:
                    effective_chi2 = np.empty(0, dtype=float)

                # Defensive check before writing.
                if np.any(effective_chi2 < 0.0):
                    raise RuntimeError(
                        f"Negative effective chi2 found for "
                        f"{detector} {data_type} WP{working_point}."
                    )

                counts = calculate_scan_counts(effective_chi2, cuts)

                rates = (
                    counts.astype(float) / duration
                    if duration > 0
                    else np.full(len(cuts), np.nan)
                )

                n_az_bins = len(az_edges) - 1
                n_el_bins = len(el_edges) - 1
                n_bins = n_az_bins * n_el_bins

                # One row per chi2 cut.
                output_df = pd.DataFrame({
                    "detector": detector,
                    "data_type": data_type,
                    "working_point": working_point,
                    "chi2_cut": cuts,
                    "n_tracks": counts,
                    "duration_seconds": duration,
                    "track_rate_hz": rates,

                    # Control-region metadata
                    "min_az": min_az,
                    "max_az": max_az,
                    "min_el": min_el,
                    "max_el": max_el,
                    "d_az": D_AZ,
                    "d_el": D_EL,
                    "n_az_bins": n_az_bins,
                    "n_el_bins": n_el_bins,
                    "n_control_bins": n_bins,
                })

                output_file = (
                    detector_output
                    / f"scan_{data_type}_WP{working_point}.parquet"
                )

                output_df.to_parquet(
                    output_file,
                    index=False,
                    compression="zstd",
                )

                print()
                print(f"Wrote: {output_file}")
                print(f"  WP = {format_wp(working_point)}")
                print(f"  duration = {duration / 3600.0:.3f} h")
                print(f"  control-region tracks = {len(effective_chi2)}")

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()