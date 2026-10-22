#!/usr/bin/env python3
"""
End-to-end MURAVES pipeline: read ROOT data -> select tracks -> rebin ->
load simulated PUMAS/Mulder transmission -> compute double ratio -> plot.

    N_VES, N_FS      : raw track counts per angular bin, read + selected
                        directly from ROOT files (chi2 cut applied).
    VES_FS           : N_VES / N_FS                (measured transmission)
    transmission_sim : PUMAS/Mulder "transmission" column, averaged into
                        the same angular bins as the data (simulated transmission)
    D                 : VES_FS / transmission_sim   (double ratio)

For each run's ROOT file:
  1. uproot reads the branches needed (angles + chi2 columns).
  2. The chi2 quality cut is applied with a numpy boolean mask.
  3. The surviving (phi, theta) pairs are histogrammed into an
     (n_az, n_el) grid and accumulated into a running total.

Usage:
    python run_pipeline.py NERO --ref-dir /path/to/pumas_outputs --ref-file pumas_output_rho265.txt

    python run_pipeline.py ROSSO --runs-ves 4000-4100 --runs-fs 10061-10200 \
        --threshold-ves 3 --threshold-fs 1.5 --key-pair 3p \
        --az-min 170 --az-max 190 --az-bin-size 1 \
        --el-min 18 --el-max 25 --el-bin-size 1 \
        --data-dir-ves /path/to/ves_data --data-dir-fs /path/to/fs_data \
        --ref-dir /path/to/pumas_outputs --ref-file pumas_output_rho265.txt \
        --flag ROSSO_WP20 --annotate

  - ROOT file per run: MURAVES_AnalyzedData_run<N>.root
  - TTree name: "AnalyzedData" (see TREE_NAME / --tree-name)
  - Branch names: Theta_3p, Theta_4p, Phi_3p, Phi_4p,
    BestTrack_4p_ChiSquare_xy, BestTrack_4p_ChiSquare_xz.
  - Theta_4p / Phi_4p / chi2 branches use -1 as a sentinel for
    "no 4-point track" -- the >= 0 half of the cut removes these.
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe backend; still writes png/pdf fine
import matplotlib.pyplot as plt
import uproot

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

CHI2_XY = "BestTrack_4p_ChiSquare_xy"
CHI2_XZ = "BestTrack_4p_ChiSquare_xz"

TREE_NAME = "AnalyzedData"

# {detector} is filled in from the "detector" positional argument.
DEFAULT_DATA_DIR_TEMPLATE = "/pnfs/iihe/muraves/muraves_DATA/RECONSTRUCTED/{detector}/v1"
FILENAME_TEMPLATE = "MURAVES_AnalyzedData_run{run}.root"

# PUMAS / Mulder flux-output file columns.
FLUX_COLUMNS = [
    "azimuth", "elevation", "flux_rock", "flux_rock_err",
    "flux_open", "transmission", "transmission_err", "n_events",
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
                raise ValueError(f"Invalid run range '{part}'. Use e.g. 2-5.")
            start, end = int(pieces[0]), int(pieces[1])
            if start > end:
                raise ValueError(f"Invalid run range '{part}': start must be <= end.")
            runs.update(range(start, end + 1))
        else:
            runs.add(int(part))
    if not runs:
        raise ValueError("No valid run numbers were provided.")
    return sorted(runs)


def iter_run_numbers(detector: str, flag: str):
    for start, end in RUN_RANGES[detector][flag]:
        yield from range(start, end + 1)


def make_edges(min_val, max_val, bin_size):
    """Build bin edges of the given size, covering [min_val, max_val].
    If the range isn't an exact multiple of bin_size, the last bin is
    stretched slightly to still land exactly on max_val."""
    n_bins = max(1, round((max_val - min_val) / bin_size))
    return np.linspace(min_val, max_val, n_bins + 1)


# ============================================================
# ROOT reading + selection + binning (data: VES / FS)
# ============================================================

def process_run(file_path, tree_name, phi_key, theta_key, threshold, az_edges, el_edges):
    """Read one run's needed branches, apply the chi2 cut, and return
    a 2D histogram of the surviving (phi, theta) pairs. Returns None
    if the run couldn't be processed."""
    branches = [phi_key, theta_key, CHI2_XY, CHI2_XZ]

    try:
        with uproot.open(file_path) as f:
            if tree_name not in f:
                print(f"  [warn] tree '{tree_name}' not found, skipping")
                return None
            tree = f[tree_name]
            missing = set(branches) - set(tree.keys())
            if missing:
                print(f"  [warn] missing branches {missing}, skipping")
                return None
            arrays = tree.arrays(branches, library="np")
    except Exception as exc:
        print(f"  [warn] could not read {file_path.name}: {exc}")
        return None

    n_total = len(arrays[CHI2_XY])
    if n_total == 0:
        print("  [warn] empty tree, skipping")
        return None

    chi2_xy, chi2_xz = arrays[CHI2_XY], arrays[CHI2_XZ]
    quality_mask = (chi2_xy >= 0) & (chi2_xy < threshold) & (chi2_xz >= 0) & (chi2_xz < threshold)

    phi, theta = arrays[phi_key][quality_mask], arrays[theta_key][quality_mask]
    finite_mask = np.isfinite(phi) & np.isfinite(theta)
    phi, theta = phi[finite_mask], theta[finite_mask]

    print(f"  Selected {quality_mask.sum()} / {n_total} entries "
          f"({len(phi)} finite after angle check)")

    counts, _, _ = np.histogram2d(phi, theta, bins=[az_edges, el_edges])
    return counts


def run_pipeline(data_dir, detector, flag, threshold, tree_name, key_pair, az_edges, el_edges, runs=None):
    phi_key, theta_key = f"Phi_{key_pair}", f"Theta_{key_pair}"

    total_counts = np.zeros((len(az_edges) - 1, len(el_edges) - 1))
    n_found = n_missing = 0

    runs_to_process = iter_run_numbers(detector, flag) if runs is None else runs

    for run in runs_to_process:
        file_path = data_dir / FILENAME_TEMPLATE.format(run=run)
        if not file_path.is_file():
            print(f"Run {run} not available, skipping")
            n_missing += 1
            continue

        n_found += 1
        print(f"Processing run {run}: {file_path.name}")
        counts = process_run(file_path, tree_name, phi_key, theta_key, threshold, az_edges, el_edges)
        if counts is not None:
            total_counts += counts

    print(f"\nRuns found on disk: {n_found}")
    print(f"Runs missing (skipped): {n_missing}")
    print(f"Total entries in map: {int(total_counts.sum())}")

    return total_counts


# ============================================================
# PUMAS / Mulder reference loading + rebinning (simulation)
# ============================================================

def parse_density(path):
    """
    Pull a rock density (g/cm^3) out of a filename like '...rho265...'
    -> 2.65. Returns None if no such tag is present.
    """
    m = re.search(r"rho(\d+)", os.path.basename(path))
    if not m:
        return None
    return int(m.group(1)) / 100.0


def load_flux_output(path, label=None, azimuth_offset=0.0):
    """
    Load an 8-column PUMAS or Mulder flux-output file.

    Returns a dict with one 1D np.ndarray per FLUX_COLUMNS entry, plus
    'label', 'path', 'density' (float or None).
    """
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != len(FLUX_COLUMNS):
        raise ValueError(
            f"{path}: expected {len(FLUX_COLUMNS)} columns "
            f"({', '.join(FLUX_COLUMNS)}), found {data.shape[1]}"
        )

    out = {name: data[:, i] for i, name in enumerate(FLUX_COLUMNS)}
    out["azimuth"] = out["azimuth"] + azimuth_offset
    out["label"] = label if label is not None else os.path.basename(path)
    out["path"] = path
    out["density"] = parse_density(path)
    return out


def rebin_transmission(ref, az_edges, el_edges):
    """
    Average the (typically finer-grained) PUMAS/Mulder 'transmission'
    column into the data's angular binning. Bins with no reference
    points falling inside them are left at 0 (and reported).
    """
    az = np.asarray(ref["azimuth"], dtype=float)
    el = np.asarray(ref["elevation"], dtype=float)
    transmission = np.asarray(ref["transmission"], dtype=float)

    valid = np.isfinite(az) & np.isfinite(el) & np.isfinite(transmission)
    az, el, transmission = az[valid], el[valid], transmission[valid]

    az_index = np.searchsorted(az_edges, az, side="right") - 1
    el_index = np.searchsorted(el_edges, el, side="right") - 1

    valid = (
        (az_index >= 0) & (az_index < len(az_edges) - 1) &
        (el_index >= 0) & (el_index < len(el_edges) - 1)
    )
    az_index = az_index[valid]
    el_index = el_index[valid]
    transmission = transmission[valid]

    n_az, n_el = len(az_edges) - 1, len(el_edges) - 1
    t_sum = np.zeros((n_az, n_el))
    t_count = np.zeros((n_az, n_el), dtype=int)

    np.add.at(t_sum, (az_index, el_index), transmission)
    np.add.at(t_count, (az_index, el_index), 1)

    t_binned = np.zeros((n_az, n_el))
    populated = t_count > 0
    t_binned[populated] = t_sum[populated] / t_count[populated]

    print(f"\nReference: {ref['label']}")
    print(f"  Reference points used:   {len(transmission)}")
    print(f"  Populated output bins:   {np.count_nonzero(populated)} / {n_az * n_el}")
    n_empty = np.count_nonzero(~populated)
    if n_empty:
        print(f"  [WARNING] {n_empty} bins have no simulated transmission points")

    return t_binned


# ============================================================
# Plotting
# ============================================================

def make_plot(values, az_edges, el_edges, title, colorbar_label, output_file,
              annotate=False, fmt="{:.2f}", cmap="viridis"):
    fig, ax = plt.subplots(figsize=(12, 7))

    mesh = ax.pcolormesh(az_edges, el_edges, values.T, shading="auto", cmap=cmap)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(colorbar_label)

    ax.set_xlabel("Azimuth [deg]")
    ax.set_ylabel("Elevation [deg]")
    ax.set_title(title)
    ax.set_xlim(az_edges[0], az_edges[-1])
    ax.set_ylim(el_edges[0], el_edges[-1])

    if len(az_edges) <= 41:
        ax.set_xticks(az_edges)
    if len(el_edges) <= 41:
        ax.set_yticks(el_edges)
    ax.grid(alpha=0.2, linewidth=0.5)

    if annotate:
        az_centers = 0.5 * (az_edges[:-1] + az_edges[1:])
        el_centers = 0.5 * (el_edges[:-1] + el_edges[1:])
        finite = values[np.isfinite(values)]
        vmax = finite.max() if finite.size else 1.0
        for i, ac in enumerate(az_centers):
            for j, ec in enumerate(el_centers):
                v = values[i, j]
                if v == 0:
                    continue
                color = "white" if v < 0.6 * vmax else "black"
                ax.text(ac, ec, fmt.format(v), ha="center", va="center",
                        fontsize=6, color=color)

    plt.tight_layout()
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_file}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="MURAVES pipeline: read ROOT -> select -> bin -> double ratio vs PUMAS/Mulder sim."
    )
    parser.add_argument("detector", choices=["ROSSO", "BLU", "NERO"])

    # -- data selection --
    parser.add_argument("--threshold-ves", type=float, default=2,
                         help="Chi-square cut threshold for VES data (default: 2).")
    parser.add_argument("--threshold-fs", type=float, default=2,
                         help="Chi-square cut threshold for FS data (default: 2).")
    parser.add_argument("--runs-ves", type=str, default=None,
                         help="VES runs to process, e.g. 2500-2600. Omit for all predefined VES runs.")
    parser.add_argument("--runs-fs", type=str, default=None,
                         help="FS runs to process, e.g. 5636-5700. Omit for all predefined FS runs.")
    parser.add_argument("--data-dir-ves", type=str, default=None,
                         help="Directory containing VES run ROOT files "
                              "(default: " + DEFAULT_DATA_DIR_TEMPLATE + " for the chosen detector).")
    parser.add_argument("--data-dir-fs", type=str, default=None,
                         help="Directory containing FS run ROOT files, can differ from --data-dir-ves "
                              "(default: " + DEFAULT_DATA_DIR_TEMPLATE + " for the chosen detector).")
    parser.add_argument("--tree-name", type=str, default=TREE_NAME)
    parser.add_argument("--key-pair", type=str, default="4p", choices=["3p", "4p"],
                         help="Use Phi_3p/Theta_3p or Phi_4p/Theta_4p (default: 4p).")

    # -- binning --
    parser.add_argument("--az-min", type=float, default=0.0)
    parser.add_argument("--az-max", type=float, default=360.0)
    parser.add_argument("--az-bin-size", type=float, default=5.0,
                         help="Azimuth bin width in degrees (default: 5.0).")
    parser.add_argument("--el-min", type=float, default=0.0)
    parser.add_argument("--el-max", type=float, default=90.0)
    parser.add_argument("--el-bin-size", type=float, default=2.0,
                         help="Elevation bin width in degrees (default: 2.0).")

    # -- PUMAS / Mulder simulated reference --
    parser.add_argument("--ref-dir", type=str, required=True,
                         help="Directory containing the PUMAS/Mulder flux-output file.")
    parser.add_argument("--ref-file", type=str, required=True,
                         help="Filename (within --ref-dir) of the 8-column PUMAS/Mulder flux-output file.")
    parser.add_argument("--azimuth-offset", type=float, default=0.0,
                         help="Degrees added to the reference file's azimuth to align conventions.")

    # -- output --
    parser.add_argument("--output-dir", type=str, default=None,
                         help="Directory to write plots into (default: --data-dir-ves). "
                              "A subfolder named after --flag is created inside it.")
    parser.add_argument("--flag", type=str, default=None,
                         help="Tag used for the output subfolder and filenames (default: detector name).")
    parser.add_argument("--annotate", action="store_true",
                         help="Write bin values as text on the double-ratio plot.")
    args = parser.parse_args()

    default_dir = DEFAULT_DATA_DIR_TEMPLATE.format(detector=args.detector)
    data_dir_ves = Path(args.data_dir_ves) if args.data_dir_ves else Path(default_dir)
    data_dir_fs = Path(args.data_dir_fs) if args.data_dir_fs else Path(default_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir_ves
    flag = args.flag if args.flag else args.detector

    runs_ves = parse_runs(args.runs_ves) if args.runs_ves is not None else None
    runs_fs = parse_runs(args.runs_fs) if args.runs_fs is not None else None

    az_edges = make_edges(args.az_min, args.az_max, args.az_bin_size)
    el_edges = make_edges(args.el_min, args.el_max, args.el_bin_size)

    print(f"Detector: {args.detector} | Key pair: {args.key_pair} | Flag: {flag}")
    print(f"Az bins: {len(az_edges) - 1} of {args.az_bin_size} deg | "
          f"El bins: {len(el_edges) - 1} of {args.el_bin_size} deg\n")

    # ---- Data: raw track counts per bin, read + selected from ROOT ----
    print(f"=== VES (chi2 < {args.threshold_ves}) ===")
    print(f"Requested VES runs: {runs_ves if runs_ves is not None else 'all predefined runs'}")
    print(f"VES data directory: {data_dir_ves}")
    N_VES = run_pipeline(
        data_dir_ves, args.detector, "VES", args.threshold_ves, args.tree_name,
        args.key_pair, az_edges, el_edges, runs=runs_ves,
    )

    print(f"\n=== FS (chi2 < {args.threshold_fs}) ===")
    print(f"Requested FS runs: {runs_fs if runs_fs is not None else 'all predefined runs'}")
    print(f"FS data directory: {data_dir_fs}")
    N_FS = run_pipeline(
        data_dir_fs, args.detector, "FS", args.threshold_fs, args.tree_name,
        args.key_pair, az_edges, el_edges, runs=runs_fs,
    )

    VES_FS = np.divide(N_VES, N_FS, out=np.zeros_like(N_VES), where=N_FS > 0)

    # ---- Simulation: transmission read from the PUMAS/Mulder ref file ----
    ref_path = Path(args.ref_dir) / args.ref_file
    print(f"\nLoading simulated reference:")
    print(f"  {ref_path}")
    ref = load_flux_output(ref_path, azimuth_offset=args.azimuth_offset)
    if ref["density"] is not None:
        print(f"  Parsed density tag: rho = {ref['density']:.2f} g/cm^3")

    transmission_sim = rebin_transmission(ref, az_edges, el_edges)

    # ---- Double ratio ----
    double_ratio = np.divide(VES_FS, transmission_sim, out=np.zeros_like(VES_FS),
                              where=transmission_sim > 0)

    # ---- Plots ----
    output_subdir = output_dir / flag
    output_subdir.mkdir(parents=True, exist_ok=True)

    rho_tag = f", #rho={ref['density']:.2f} g/cm^3" if ref["density"] is not None else ""
    bs_az, bs_el = args.az_bin_size, args.el_bin_size

    make_plot(N_FS, az_edges, el_edges,
              f"Freesky data — {bs_az:g}x{bs_el:g} deg bins — {args.key_pair}",
              "Track count", output_subdir / f"FS_tracks_{flag}.png")

    make_plot(N_VES, az_edges, el_edges,
              f"Vesuvius data — {bs_az:g}x{bs_el:g} deg bins — {args.key_pair}",
              "Track count", output_subdir / f"VES_tracks_{flag}.png")

    make_plot(VES_FS, az_edges, el_edges,
              f"VES / FS data ratio — {bs_az:g}x{bs_el:g} deg bins — {args.key_pair}",
              r"$N_{VES}/N_{FS}$", output_subdir / f"MeasuredTransmission_{flag}.png")

    make_plot(transmission_sim, az_edges, el_edges,
              f"Simulated transmission ({ref['label']}){rho_tag}",
              "Transmission", output_subdir / f"SimulatedTransmission_{flag}.png")

    make_plot(double_ratio, az_edges, el_edges,
              f"Double ratio — {bs_az:g}x{bs_el:g} deg bins{rho_tag}",
              r"$D=(N_{VES}/N_{FS})/T_{sim}$",
              output_subdir / f"DoubleRatio_{flag}.png",
              annotate=args.annotate)


if __name__ == "__main__":
    main()