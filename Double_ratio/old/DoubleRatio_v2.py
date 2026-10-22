```python
import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt


# PUMAS / Mulder output columns
FLUX_COLUMNS = [
    "azimuth", "elevation", "flux_rock", "flux_rock_err",
    "flux_open", "transmission", "transmission_err", "n_events",
]


# ============================================================
# Load PUMAS / Mulder file
# ============================================================

def load_flux_output(path, label=None, azimuth_offset=0.0):
    data = np.loadtxt(path)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] != len(FLUX_COLUMNS):
        raise ValueError(
            f"{path}: expected {len(FLUX_COLUMNS)} columns "
            f"({', '.join(FLUX_COLUMNS)}), found {data.shape[1]}"
        )

    out = {name: data[:, i] for i, name in enumerate(FLUX_COLUMNS)}
    out["azimuth"] += azimuth_offset
    out["label"] = label if label is not None else os.path.basename(path)
    out["path"] = path

    return out


# ============================================================
# Load JSON and bin tracks
# ============================================================

def load_and_bin_json(filename, az_edges, el_edges, phi_key, theta_key):
    with open(filename, "r") as f:
        data = json.load(f)

    phi = np.array([entry[phi_key] for entry in data], dtype=float)
    theta = np.array([entry[theta_key] for entry in data], dtype=float)

    valid = np.isfinite(phi) & np.isfinite(theta)
    phi = phi[valid]
    theta = theta[valid]

    counts, _, _ = np.histogram2d(phi, theta, bins=[az_edges, el_edges])

    print(f"\n{filename}")
    print(f"  Total entries:          {len(data)}")
    print(f"  Valid angular entries:  {len(phi)}")
    print(f"  Entries in range:       {int(counts.sum())}")

    return counts


# ============================================================
# Rebin PUMAS / Mulder transmission
# ============================================================

def rebin_transmission(ref, az_edges, el_edges):
    az = np.asarray(ref["azimuth"], dtype=float)
    el = np.asarray(ref["elevation"], dtype=float)
    transmission = np.asarray(ref["transmission"], dtype=float)

    valid = np.isfinite(az) & np.isfinite(el) & np.isfinite(transmission)

    az = az[valid]
    el = el[valid]
    transmission = transmission[valid]

    az_index = np.searchsorted(az_edges, az, side="right") - 1
    el_index = np.searchsorted(el_edges, el, side="right") - 1

    valid = (
        (az_index >= 0)
        & (az_index < len(az_edges) - 1)
        & (el_index >= 0)
        & (el_index < len(el_edges) - 1)
    )

    az_index = az_index[valid]
    el_index = el_index[valid]
    transmission = transmission[valid]

    n_az = len(az_edges) - 1
    n_el = len(el_edges) - 1

    transmission_sum = np.zeros((n_az, n_el), dtype=float)
    transmission_count = np.zeros((n_az, n_el), dtype=int)

    np.add.at(transmission_sum, (az_index, el_index), transmission)
    np.add.at(transmission_count, (az_index, el_index), 1)

    transmission_binned = np.zeros((n_az, n_el), dtype=float)

    populated = transmission_count > 0

    transmission_binned[populated] = (
        transmission_sum[populated] / transmission_count[populated]
    )

    print(f"\nReference: {ref['label']}")
    print(f"  Reference points used: {len(transmission)}")
    print(
        f"  Populated output bins: "
        f"{np.count_nonzero(populated)} / {n_az * n_el}"
    )

    return transmission_binned


# ============================================================
# Plot helper
# ============================================================

def make_plot(values, az_edges, el_edges, az_min, az_max, el_min, el_max,
              title, colorbar_label, output_file):

    fig, ax = plt.subplots(figsize=(12, 7))

    mesh = ax.pcolormesh(
        az_edges,
        el_edges,
        values.T,
        shading="auto",
    )

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(colorbar_label)

    ax.set_xlabel("Azimuth [deg]")
    ax.set_ylabel("Elevation [deg]")
    ax.set_title(title)

    ax.set_xlim(az_min, az_max)
    ax.set_ylim(el_min, el_max)

    ax.set_xticks(az_edges)
    ax.set_yticks(el_edges)

    ax.grid(alpha=0.2, linewidth=0.5)

    plt.tight_layout()
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_file}")


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Calculate VES/FS divided by PUMAS/Mulder transmission."
    )

    parser.add_argument("--FS", required=True, help="FS JSON file")
    parser.add_argument("--VES", required=True, help="VES JSON file")
    parser.add_argument("--ref", required=True, help="PUMAS/Mulder reference file")

    parser.add_argument(
        "--az", nargs=2, type=float, default=[170.0, 190.0],
        metavar=("MIN", "MAX"), help="Azimuth range"
    )

    parser.add_argument(
        "--el", nargs=2, type=float, default=[18.0, 25.0],
        metavar=("MIN", "MAX"), help="Elevation range"
    )

    parser.add_argument(
        "--bin-size", type=float, default=1.0,
        help="Angular bin size in degrees"
    )

    parser.add_argument(
        "--flag", required=True,
        help="Output directory/name"
    )

    parser.add_argument(
        "--coordinate", choices=["3p", "4p"], default="3p",
        help="Angular coordinate system"
    )

    args = parser.parse_args()

    # ========================================================
    # Angular settings
    # ========================================================

    az_min, az_max = args.az
    el_min, el_max = args.el

    if az_max <= az_min:
        raise ValueError("Azimuth max must be greater than azimuth min.")

    if el_max <= el_min:
        raise ValueError("Elevation max must be greater than elevation min.")

    if args.bin_size <= 0:
        raise ValueError("Bin size must be greater than zero.")

    az_edges = np.arange(az_min, az_max + args.bin_size, args.bin_size)
    el_edges = np.arange(el_min, el_max + args.bin_size, args.bin_size)

    print("\nAngular binning:")
    print(f"  Azimuth:   {az_min} -> {az_max} deg")
    print(f"  Elevation: {el_min} -> {el_max} deg")
    print(f"  Bin size:  {args.bin_size} deg")
    print(f"  Number of azimuth bins:   {len(az_edges) - 1}")
    print(f"  Number of elevation bins: {len(el_edges) - 1}")

    # ========================================================
    # Coordinate choice
    # ========================================================

    if args.coordinate == "3p":
        phi_key = "Phi_3p"
        theta_key = "Theta_3p"
    else:
        phi_key = "Phi_4p"
        theta_key = "Theta_4p"

    # ========================================================
    # Load and bin measured tracks
    # ========================================================

    N_FS = load_and_bin_json(
        args.FS, az_edges, el_edges, phi_key, theta_key
    )

    N_VES = load_and_bin_json(
        args.VES, az_edges, el_edges, phi_key, theta_key
    )

    # transmission = N_VES / N_FS
    VES_FS = np.divide(
        N_VES,
        N_FS,
        out=np.zeros_like(N_VES, dtype=float),
        where=N_FS > 0,
    )

    # ========================================================
    # Load PUMAS / Mulder and rebin transmission
    # ========================================================

    print("\nLoading PUMAS/Mulder reference:")
    print(f"  {args.ref}")

    ref = load_flux_output(args.ref, label="PUMAS/Mulder")

    transmission = rebin_transmission(
        ref, az_edges, el_edges
    )

    # ========================================================
    double_ratio = np.divide(
        VES_FS,
        transmission,
        out=np.zeros_like(VES_FS, dtype=float),
        where=transmission > 0,
    )

    output_dir = os.path.join("images", args.flag)
    os.makedirs(output_dir, exist_ok=True)

    # ========================================================
    # Plot 1: Measured transmission (VES / FS)
    # ========================================================

    make_plot(
        VES_FS,
        az_edges,
        el_edges,
        az_min,
        az_max,
        el_min,
        el_max,
        f"VES / FS data ratio — {args.bin_size:g}° bins — {args.coordinate}",
        r"$N_{\mathrm{VES}} / N_{\mathrm{FS}}$",
        os.path.join(output_dir, f"VES_FS_{args.flag}.png"),
    )

    # ========================================================
    # Plot 2: Simulated transmission (PUMAS/Mulder)
    # ========================================================

    make_plot(
        transmission,
        az_edges,
        el_edges,
        az_min,
        az_max,
        el_min,
        el_max,
        f"Simulated transmission — {args.bin_size:g}° bins",
        "Transmission",
        os.path.join(output_dir, f"Transmission_{args.flag}.png"),
    )

    # ========================================================
    # Plot 3: Double ratio
    # ========================================================

    make_plot(
        double_ratio,
        az_edges,
        el_edges,
        az_min,
        az_max,
        el_min,
        el_max,
        f"VES / FS / transmission — {args.bin_size:g}° bins — {args.coordinate}",
        r"$(N_{\mathrm{VES}}/N_{\mathrm{FS}}) / T$",
        os.path.join(output_dir, f"DoubleRatio_{args.flag}.png"),
    )


if __name__ == "__main__":
    main()
