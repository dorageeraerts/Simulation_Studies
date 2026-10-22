#!/usr/bin/env python3

"""
Plot MURAVES chi2 distributions per Working Point (WP), separately for VES and FS.

Branches:
    BestTrack_3p_ChiSquare_xy
    BestTrack_3p_ChiSquare_xz
    BestTrack_4p_ChiSquare_xy
    BestTrack_4p_ChiSquare_xz

Output:
    ./chi2_distributions/<detector>/

Example:
    chi2_hist_3p_xy_WP20_VES.png
    chi2_hist_3p_xz_WP20_VES.png
    chi2_hist_4p_xy_WP20_VES.png
    chi2_hist_4p_xz_WP20_VES.png
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import uproot


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR_TEMPLATE = "/pnfs/iihe/muraves/muraves_DATA/RECONSTRUCTED/{detector}/v1"
SLOWCONTROL_DIR = "/pnfs/iihe/muraves/muraves_DATA/DECOMPRESSED"
OUTPUT_DIR = Path("./chi2_distributions")

TREE_NAME = "AnalyzedData"
FILENAME_TEMPLATE = "MURAVES_AnalyzedData_run{run}.root"
SLOWCONTROL_TEMPLATE = "SLOWCONTROL_run{run}"

CHI2_MIN = 0.0
CHI2_MAX = 20.0
BIN_WIDTH = 0.1

CHI2_BRANCHES = {
    "3p_xy": "BestTrack_3p_ChiSquare_xy",
    "3p_xz": "BestTrack_3p_ChiSquare_xz",
    "4p_xy": "BestTrack_4p_ChiSquare_xy",
    "4p_xz": "BestTrack_4p_ChiSquare_xz",
}

RUN_RANGES = {
    "NERO": {"VES": [(2500, 5600), (7894, 11990)], "FS": [(5636, 7893)]},
    "ROSSO": {"VES": [(4000, 10060), (13458, 14792)], "FS": [(10061, 13457)]},
    "BLU": {"VES": [(2800, 8818)], "FS": [(8819, 9925)]},
}


# ============================================================
# HELPERS
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
            raise argparse.ArgumentTypeError(f"Invalid run number: {value}") from exc
        return run, run

    try:
        start, stop = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid run range: {value}") from exc

    if stop < start:
        raise argparse.ArgumentTypeError(f"Invalid run range: {value}: end must be >= start.")

    return start, stop


def parse_range_list(values):
    return None if values is None else [parse_run_range(value) for value in values]


def iter_run_numbers(run_ranges):
    for start, stop in run_ranges:
        yield from range(start, stop + 1)


def get_working_point(slowcontrol_dir, detector, run):
    slowcontrol_file = Path(slowcontrol_dir) / detector / SLOWCONTROL_TEMPLATE.format(run=run)

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

                if not np.isfinite(working_point):
                    print(f"  [warn] non-finite Working Point in {slowcontrol_file}")
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


def read_run(file_path):
    branches = list(CHI2_BRANCHES.values())

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


def get_valid_chi2(values):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values) & (values >= CHI2_MIN) & (values <= CHI2_MAX)
    return values[valid]


def make_bins():
    n_bins = int(np.ceil((CHI2_MAX - CHI2_MIN) / BIN_WIDTH))
    return np.linspace(CHI2_MIN, CHI2_MAX, n_bins + 1)


def plot_distribution(values, detector, working_point, data_type, branch_name, output_file):
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.hist(values, bins=make_bins(), histtype="step", linewidth=1.5, density=True)
    ax.set_xlabel(r"$\chi^2$")
    ax.set_ylabel("Density")
    ax.set_title(f"{branch_name} — {detector}, WP{working_point}, {data_type}")
    ax.grid(True, alpha=0.3, linestyle="--")

    ax.text(0.98, 0.95, f"Entries: {len(values):,}", transform=ax.transAxes, ha="right", va="top")

    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot MURAVES chi2 distributions per WP, separately for VES and FS.")

    parser.add_argument("--detector", choices=sorted(RUN_RANGES.keys()), help="Detector to process, e.g. ROSSO")
    parser.add_argument("--detectors", nargs="+", choices=sorted(RUN_RANGES.keys()), help="Multiple detectors to process")
    parser.add_argument("--ves-ranges", nargs="+", metavar="START-END", help="Override default VES run ranges")
    parser.add_argument("--fs-ranges", nargs="+", metavar="START-END", help="Override default FS run ranges")
    parser.add_argument("--chi2-max", type=float, default=CHI2_MAX, help=f"Maximum chi2 shown, default={CHI2_MAX}")
    parser.add_argument("--bin-width", type=float, default=BIN_WIDTH, help=f"Histogram bin width, default={BIN_WIDTH}")

    args = parser.parse_args()

    if args.detector and args.detectors:
        parser.error("Use either --detector or --detectors, not both.")

    detectors = [args.detector] if args.detector else args.detectors if args.detectors else ["ROSSO"]

    if args.chi2_max <= CHI2_MIN:
        parser.error("--chi2-max must be > 0")

    if args.bin_width <= 0:
        parser.error("--bin-width must be > 0")

    return args, detectors, parse_range_list(args.ves_ranges), parse_range_list(args.fs_ranges)


# ============================================================
# MAIN
# ============================================================

def main():
    global CHI2_MAX, BIN_WIDTH

    args, detectors, ves_override, fs_override = parse_arguments()

    CHI2_MAX = args.chi2_max
    BIN_WIDTH = args.bin_width

    print("=" * 70)
    print("MURAVES CHI2 DISTRIBUTIONS")
    print("=" * 70)
    print(f"Detectors : {', '.join(detectors)}")
    print(f"Chi2 range: {CHI2_MIN} -> {CHI2_MAX}")
    print(f"Bin width : {BIN_WIDTH}")
    print("VES and FS are kept separate.")
    print("No angular selection is applied.")
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
            "VES": ves_override if ves_override is not None else RUN_RANGES[detector]["VES"],
            "FS": fs_override if fs_override is not None else RUN_RANGES[detector]["FS"],
        }

        # data[data_type][WP][short_name] = list of numpy arrays
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

                working_point = get_working_point(SLOWCONTROL_DIR, detector, run)

                if working_point is None:
                    print("  [warn] no WP, skipping run")
                    continue

                print(f"  Working Point: {working_point}")

                arrays = read_run(file_path)

                if arrays is None:
                    continue

                if working_point not in data[data_type]:
                    data[data_type][working_point] = {key: [] for key in CHI2_BRANCHES}

                for short_name, branch_name in CHI2_BRANCHES.items():
                    values = get_valid_chi2(arrays[branch_name])

                    if len(values) > 0:
                        data[data_type][working_point][short_name].append(values)

                    print(f"    {short_name}: {len(values):,} entries")

        # --------------------------------------------------------
        # CREATE HISTOGRAMS
        # --------------------------------------------------------

        print()
        print(f"Creating plots for {detector}...")

        for data_type in ["VES", "FS"]:
            for working_point in sorted(data[data_type]):
                for short_name, branch_name in CHI2_BRANCHES.items():

                    arrays = data[data_type][working_point][short_name]

                    if not arrays:
                        print(f"  {data_type} WP{working_point} {short_name}: no entries, skipping")
                        continue

                    values = np.concatenate(arrays)

                    output_file = detector_output / f"chi2_hist_{short_name}_WP{working_point}_{data_type}.png"

                    plot_distribution(
                        values,
                        detector,
                        working_point,
                        data_type,
                        branch_name,
                        output_file,
                    )

                    print(f"  {data_type} WP{working_point} {short_name}: {len(values):,} entries -> {output_file}")

        print()
        print(f"Output directory: {detector_output}")

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()

