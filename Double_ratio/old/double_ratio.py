#!/usr/bin/env python3
"""
Data double ratio:

    VES_FS = N_VES / N_FS              (measured track-count ratio: transmission)
    D = VES_FS / transmission_sim      (double ratio)

- N_FS and N_VES are binned directly from JSON track lists (value in
  each angular bin = raw number of tracks that fall in it).
- transmission_sim comes straight from a single PUMAS/Mulder flux-output
  file's "transmission" column, averaged into the same angular bins.

Usage
-----
    python double_ratio.py \
        --FS FS.json --VES VES.json --REF pumas_output_rho265.txt \
        --az 170 190 --el 18 25 --bin-size 1.0 \
        --coordinate 4p --flag NERO_WP20 --annotate
"""

import argparse
import json
import os
import re

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# PUMAS / Mulder flux-output file columns
# ------------------------------------------------------------------
FLUX_COLUMNS = [
    "azimuth", "elevation", "flux_rock", "flux_rock_err",
    "flux_open", "transmission", "transmission_err", "n_events",
]


# ============================================================
# Loaders
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

    Parameters
    ----------
    path : str
    label : str, optional
        Human-readable name (defaults to the basename of `path`).
    azimuth_offset : float, optional
        Degrees added to the azimuth column on load, to align PUMAS's
        raw azimuth convention with Mulder's (or the data's), if needed.

    Returns
    -------
    dict with one 1D np.ndarray per FLUX_COLUMNS entry, plus
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


def load_and_bin_json(filename, az_edges, el_edges, phi_key, theta_key):
    """
    Load a JSON list of track dicts (each with phi_key/theta_key angles)
    and histogram them into an (n_az, n_el) array of raw track counts.
    """
    with open(filename, "r") as f:
        data = json.load(f)

    phi = np.array([entry[phi_key] for entry in data], dtype=float)
    theta = np.array([entry[theta_key] for entry in data], dtype=float)

    valid = np.isfinite(phi) & np.isfinite(theta)
    phi, theta = phi[valid], theta[valid]

    counts, _, _ = np.histogram2d(phi, theta, bins=[az_edges, el_edges])

    print(f"\n{filename}")
    print(f"  Total entries:            {len(data)}")
    print(f"  Valid angular entries:    {len(phi)}")
    print(f"  Entries inside [az,el]:   {int(counts.sum())}")

    return counts


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
        description="Double ratio: (VES/FS track-count ratio) / (simulated transmission)."
    )
    parser.add_argument("--FS", required=True, help="Freesky JSON track file")
    parser.add_argument("--VES", required=True, help="Vesuvius JSON track file")
    parser.add_argument("--REF", required=True, help="PUMAS/Mulder flux-output file (8 columns)")
    parser.add_argument("--az", nargs=2, type=float, default=[170.0, 190.0], metavar=("MIN", "MAX"))
    parser.add_argument("--el", nargs=2, type=float, default=[18.0, 25.0], metavar=("MIN", "MAX"))
    parser.add_argument("--bin-size", type=float, default=1.0, help="Angular bin width [deg]")
    parser.add_argument("--coordinate", choices=["3p", "4p"], default="4p",
                         help="Which Phi_*/Theta_* angle pair to read from the JSON files")
    parser.add_argument("--azimuth-offset", type=float, default=0.0,
                         help="Degrees added to the reference file's azimuth to align conventions")
    parser.add_argument("--flag", required=True, help="Output subfolder / filename tag")
    parser.add_argument("--annotate", action="store_true",
                         help="Write bin values as text on the double-ratio plot")
    args = parser.parse_args()

    az_min, az_max = args.az
    el_min, el_max = args.el
    if az_max <= az_min or el_max <= el_min or args.bin_size <= 0:
        raise ValueError("Invalid azimuth/elevation range or bin size.")

    az_edges = np.arange(az_min, az_max + args.bin_size, args.bin_size)
    el_edges = np.arange(el_min, el_max + args.bin_size, args.bin_size)

    print("\nAngular binning:")
    print(f"  Azimuth:   {az_min} -> {az_max} deg, {len(az_edges) - 1} bins")
    print(f"  Elevation: {el_min} -> {el_max} deg, {len(el_edges) - 1} bins")
    print(f"  Bin size:  {args.bin_size} deg")

    phi_key, theta_key = f"Phi_{args.coordinate}", f"Theta_{args.coordinate}"

    # ---- Data: raw track counts per bin ----
    N_FS = load_and_bin_json(args.FS, az_edges, el_edges, phi_key, theta_key)
    N_VES = load_and_bin_json(args.VES, az_edges, el_edges, phi_key, theta_key)

    VES_FS = np.divide(N_VES, N_FS, out=np.zeros_like(N_VES), where=N_FS > 0)

    # ---- Simulation: transmission read directly from the single ref file ----
    print("\nLoading simulated reference:")
    print(f"  {args.REF}")
    ref = load_flux_output(args.REF, azimuth_offset=args.azimuth_offset)
    if ref["density"] is not None:
        print(f"  Parsed density tag: rho = {ref['density']:.2f} g/cm^3")

    transmission = rebin_transmission(ref, az_edges, el_edges)

    # ---- Double ratio ----
    double_ratio = np.divide(VES_FS, transmission, out=np.zeros_like(VES_FS),
                              where=transmission > 0)

    output_dir = os.path.join("images", args.flag)
    os.makedirs(output_dir, exist_ok=True)

    rho_tag = f", #rho={ref['density']:.2f} g/cm^3" if ref["density"] is not None else ""
    bs = args.bin_size

    make_plot(N_FS, az_edges, el_edges,
              f"Freesky data — {bs:g} deg bins — {args.coordinate}",
              "Track count", os.path.join(output_dir, f"FS_tracks_{args.flag}.png"))

    make_plot(N_VES, az_edges, el_edges,
              f"Vesuvius data — {bs:g} deg bins — {args.coordinate}",
              "Track count", os.path.join(output_dir, f"VES_tracks_{args.flag}.png"))

    make_plot(VES_FS, az_edges, el_edges,
              f"VES / FS data ratio — {bs:g} deg bins — {args.coordinate}",
              r"$N_{VES}/N_{FS}$", os.path.join(output_dir, f"MeasuredTransmission_{args.flag}.png"))

    make_plot(transmission, az_edges, el_edges,
              f"Simulated transmission ({ref['label']}){rho_tag}",
              "Transmission", os.path.join(output_dir, f"SimulatedTransmission_{args.flag}.png"))

    make_plot(double_ratio, az_edges, el_edges,
              f"Double ratio — {bs:g} deg bins{rho_tag}",
              r"$D=(N_{VES}/N_{FS})/T_{sim}$",
              os.path.join(output_dir, f"DoubleRatio_{args.flag}.png"),
              annotate=args.annotate)


if __name__ == "__main__":
    main()