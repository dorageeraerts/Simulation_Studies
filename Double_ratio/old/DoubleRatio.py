#!/usr/bin/env python3

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Load JSON and histogram tracks
# ============================================================

def load_and_bin_json(
    filename,
    az_edges,
    el_edges,
    phi_key="Phi_3p",
    theta_key="Theta_3p",
):
    """
    Load event/track data from a JSON file.

    Each JSON entry contributes one count to the
    corresponding angular bin.
    """

    with open(filename, "r") as f:
        data = json.load(f)

    phi = np.array(
        [entry[phi_key] for entry in data],
        dtype=float,
    )

    theta = np.array(
        [entry[theta_key] for entry in data],
        dtype=float,
    )

    # Remove non-finite entries
    valid = np.isfinite(phi) & np.isfinite(theta)

    phi = phi[valid]
    theta = theta[valid]

    # Histogram
    counts, _, _ = np.histogram2d(
        phi,
        theta,
        bins=[az_edges, el_edges],
    )

    print(f"\n{filename}")
    print(f"  Total JSON entries:       {len(data)}")
    print(f"  Valid angular entries:    {len(phi)}")
    print(f"  Entries in plotted range: {int(counts.sum())}")

    return counts


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Bin FS and VES JSON track data and plot VES/FS."
    )
    parser.add_argument("--fs",
        required=True,
        help="FS JSON file",
    )
    parser.add_argument("--ves",
        required=True,
        help="VES JSON file",
    )
    parser.add_argument("--az",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=[170.0, 190.0],
        help="Azimuth range in degrees (default: 170 190)",
    )
    parser.add_argument("--el",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=[18.0, 25.0],
        help="Elevation range in degrees (default: 18 25)",
    )
    parser.add_argument("--bin-size",
        type=float,
        default=1.0,
        help="Angular bin size in degrees (default: 1)",
    )
    parser.add_argument("--flag",
        required=True,
        help="Name used for the output directory",
    )
    parser.add_argument("--coordinate",
        choices=["3p", "4p"],
        default="3p",
        help="Use 3p or 4p angular coordinates (default: 3p)",
    )

    args = parser.parse_args()


    az_min, az_max = args.az
    el_min, el_max = args.el

    if az_max <= az_min:
        raise ValueError("Azimuth max must be greater than azimuth min.")

    if el_max <= el_min:
        raise ValueError("Elevation max must be greater than elevation min.")

    if args.bin_size <= 0:
        raise ValueError("Bin size must be greater than zero.")

    az_edges = np.arange(
        az_min,
        az_max + args.bin_size,
        args.bin_size,
    )

    el_edges = np.arange(
        el_min,
        el_max + args.bin_size,
        args.bin_size,
    )

    print("\nAngular binning:")
    print(f"  Azimuth:   {az_min} -> {az_max} deg")
    print(f"  Elevation: {el_min} -> {el_max} deg")
    print(f"  Bin size:  {args.bin_size} deg")
    print(f"  Number of azimuth bins:   {len(az_edges) - 1}")
    print(f"  Number of elevation bins: {len(el_edges) - 1}")

    if args.coordinate == "3p":
        phi_key = "Phi_3p"
        theta_key = "Theta_3p"
    else:
        phi_key = "Phi_4p"
        theta_key = "Theta_4p"

    N_FS = load_and_bin_json(
        args.fs,
        az_edges,
        el_edges,
        phi_key=phi_key,
        theta_key=theta_key,
    )

    N_VES = load_and_bin_json(
        args.ves,
        az_edges,
        el_edges,
        phi_key=phi_key,
        theta_key=theta_key,
    )

    VES_FS = np.divide( # transmission
        N_VES,
        N_FS,
        out=np.zeros_like(N_VES, dtype=float),
        where=N_FS > 0,
    )


    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = os.path.join(
        "images",
        args.flag,
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    output_file = os.path.join(
        output_dir,
        f"DoubleRatio_{args.flag}.png",
    )


    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 7),)

    mesh = ax.pcolormesh(az_edges, el_edges, VES_FS.T, shading="auto",)

    cbar = fig.colorbar(mesh, ax=ax,)

    cbar.set_label(r"$N_{\mathrm{VES}} / N_{\mathrm{FS}}$")

    ax.set_xlabel("Azimuth [deg]")

    ax.set_ylabel("Elevation [deg]")

    ax.set_title(f"VES / FS data ratio — " f"{args.bin_size:g}° bins — {args.coordinate}")

    ax.set_xlim(az_min, az_max,)

    ax.set_ylim(el_min, el_max,)

    # Grid aligned with bins
    ax.set_xticks(az_edges)
    ax.set_yticks(el_edges)
    ax.grid(
        alpha=0.2,
        linewidth=0.5,
    )

    plt.tight_layout()

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()