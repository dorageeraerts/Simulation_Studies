#!/usr/bin/env python3
"""Analyze incoming and outgoing muon states saved by Mulder."""

import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


AZIMUTH_OFFSET = 136.0


def shifted_azimuth(azimuth):
    return (np.asarray(azimuth, dtype=float) + AZIMUTH_OFFSET) % 360.0


def signed_azimuth_difference(incoming, outgoing):
    """Return outgoing - incoming in the interval [-180, 180) degrees."""
    return (outgoing - incoming + 180.0) % 360.0 - 180.0


def angular_separation(incoming_azimuth, incoming_elevation,
                       outgoing_azimuth, outgoing_elevation):
    """Return the 3D angle between two directions in degrees."""
    incoming_azimuth = np.radians(incoming_azimuth)
    incoming_elevation = np.radians(incoming_elevation)
    outgoing_azimuth = np.radians(outgoing_azimuth)
    outgoing_elevation = np.radians(outgoing_elevation)

    cosine = (
        np.sin(incoming_elevation) * np.sin(outgoing_elevation)
        + np.cos(incoming_elevation)
        * np.cos(outgoing_elevation)
        * np.cos(outgoing_azimuth - incoming_azimuth)
    )
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def add_derived_columns(states):
    states = states.copy()
    states["incoming_azimuth_shifted"] = shifted_azimuth(
        states["incoming_azimuth"]
    )
    states["outgoing_azimuth_shifted"] = shifted_azimuth(
        states["outgoing_azimuth"]
    )
    states["azimuth_difference"] = signed_azimuth_difference(
        states["incoming_azimuth_shifted"],
        states["outgoing_azimuth_shifted"],
    )
    states["absolute_azimuth_difference"] = states["azimuth_difference"].abs()
    states["theta_difference"] = (
        states["outgoing_elevation"] - states["incoming_elevation"]
    )
    states["absolute_theta_difference"] = states["theta_difference"].abs()
    states["scattering_angle"] = angular_separation(
        states["incoming_azimuth_shifted"],
        states["incoming_elevation"],
        states["outgoing_azimuth_shifted"],
        states["outgoing_elevation"],
    )
    states["survived"] = states["transport_weight"] > 0.0
    return states


def save_energy_plot(states, output_path):
    positive_energy = states[states["incoming_energy"] > 0.0].copy()
    energy_edges = np.logspace(
        np.log10(positive_energy["incoming_energy"].min()),
        np.log10(positive_energy["incoming_energy"].max()),
        25,
    )
    positive_energy["energy_bin"] = pd.cut(
        positive_energy["incoming_energy"],
        bins=energy_edges,
        include_lowest=True,
    )
    grouped = positive_energy.groupby("energy_bin", observed=True).agg(
        energy=("incoming_energy", "median"),
        median_scattering=("scattering_angle", "median"),
        lower_scattering=("scattering_angle", lambda values: values.quantile(0.16)),
        upper_scattering=("scattering_angle", lambda values: values.quantile(0.84)),
        median_azimuth_difference=("absolute_azimuth_difference", "median"),
        lower_azimuth_difference=("absolute_azimuth_difference", lambda values: values.quantile(0.16)),
        upper_azimuth_difference=("absolute_azimuth_difference", lambda values: values.quantile(0.84)),
        count=("scattering_angle", "size"),
    ).dropna()

    figure, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    axes[0].fill_between(
        grouped["energy"],
        grouped["lower_scattering"],
        grouped["upper_scattering"],
        alpha=0.2,
        label="16th-84th percentile",
    )
    axes[0].plot(grouped["energy"], grouped["median_scattering"], marker="o")
    axes[0].set_ylabel("3D scattering angle [deg]")
    axes[0].set_title("Muon deflection as a function of incoming energy")
    axes[0].legend()

    axes[1].fill_between(
        grouped["energy"],
        grouped["lower_azimuth_difference"],
        grouped["upper_azimuth_difference"],
        alpha=0.2,
        label="16th-84th percentile",
    )
    axes[1].plot(
        grouped["energy"],
        grouped["median_azimuth_difference"],
        marker="o",
        color="tab:orange",
    )
    axes[1].set_xlabel("Incoming energy [GeV]")
    axes[1].set_ylabel("Absolute azimuth difference [deg]")
    axes[1].legend()

    for axis in axes:
        axis.set_xscale("log")
        axis.grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(os.path.join(output_path, "scattering_vs_energy.png"), dpi=180)
    plt.close(figure)


def save_incoming_outgoing_scatter(states, output_path):
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    comparisons = [
        (
            "incoming_azimuth_shifted",
            "outgoing_azimuth_shifted",
            "Azimuth + 136° [deg]",
            False,
        ),
        (
            "incoming_elevation",
            "outgoing_elevation",
            "Elevation [deg]",
            False,
        ),
        (
            "incoming_energy",
            "outgoing_energy",
            "Energy [GeV]",
            True,
        ),
    ]

    for axis, (incoming, outgoing, label, logarithmic) in zip(axes, comparisons):
        values = states[[incoming, outgoing]].replace([np.inf, -np.inf], np.nan).dropna()
        if logarithmic:
            values = values[(values[incoming] > 0.0) & (values[outgoing] > 0.0)]

        axis.scatter(
            values[incoming],
            values[outgoing],
            s=7,
            alpha=0.35,
            rasterized=True,
        )
        limits = [values[incoming].min(), values[outgoing].min(),
                  values[incoming].max(), values[outgoing].max()]
        lower = min(limits[:2])
        upper = max(limits[2:])
        axis.plot([lower, upper], [lower, upper], color="tab:red", linestyle="--")
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        if logarithmic:
            axis.set_xscale("log")
            axis.set_yscale("log")
        axis.set_xlabel(f"Incoming {label}")
        axis.set_ylabel(f"Outgoing {label}")
        axis.set_title(label.split(" [")[0])
        axis.grid(alpha=0.25)

    figure.suptitle("Incoming versus outgoing muon states")
    figure.tight_layout()
    figure.savefig(os.path.join(output_path, "incoming_vs_outgoing.png"), dpi=180)
    plt.close(figure)


def save_direction_grid(
    states,
    column,
    title,
    colorbar_label,
    filename,
    output_path,
    log_scale=False,
):
    grouped = (
        states.assign(
            outgoing_azimuth_bin=np.floor(states["outgoing_azimuth_shifted"]),
            outgoing_elevation_bin=np.floor(states["outgoing_elevation"]),
        )
        .groupby(["outgoing_elevation_bin", "outgoing_azimuth_bin"])[column]
        .mean()
        .unstack()
        .sort_index()
    )
    if grouped.empty:
        return

    norm = None
    if log_scale:
        positive_values = grouped.values[grouped.values > 0.0]
        norm = LogNorm(vmin=positive_values.min(), vmax=positive_values.max())

    figure, axis = plt.subplots(figsize=(10, 6))
    image = axis.imshow(
        grouped.values,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[
            grouped.columns.min() - 0.5,
            grouped.columns.max() + 0.5,
            grouped.index.min() - 0.5,
            grouped.index.max() + 0.5,
        ],
        cmap="viridis",
        norm=norm,
    )
    figure.colorbar(image, ax=axis, label=colorbar_label)
    axis.set_xlabel("Outgoing azimuth + 136° [deg]")
    axis.set_ylabel("Outgoing elevation [deg]")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(os.path.join(output_path, filename), dpi=180)
    plt.close(figure)


def save_sideways_plot(sideways, threshold, output_path):
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    if sideways.empty:
        message = "No rows in outgoing azimuth 150°-170°\nand elevation 5°-20°"
        for axis in axes:
            axis.text(
                0.5,
                0.5,
                message,
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_axis_off()
        figure.suptitle("No muons in requested outgoing direction window")
        figure.tight_layout()
        figure.savefig(os.path.join(output_path, "sideways_outgoing_window.png"), dpi=180)
        plt.close(figure)
        return

    difference_limit = max(
        sideways["absolute_azimuth_difference"].max() * 1.1,
        1.0,
    )
    axes[0].hist(
        sideways["azimuth_difference"],
        bins=np.linspace(-difference_limit, difference_limit, 51),
        color="tab:blue",
        alpha=0.85,
    )
    if threshold <= difference_limit:
        axes[0].axvline(threshold, color="tab:red", linestyle="--")
        axes[0].axvline(-threshold, color="tab:red", linestyle="--")
    axes[0].set_xlim(-difference_limit, difference_limit)
    axes[0].set_xlabel("Signed outgoing - incoming azimuth [deg]")
    axes[0].set_ylabel("Muon count")
    axes[0].set_title("Azimuth deflection in outgoing direction window")
    axes[0].grid(alpha=0.25)

    hexbin = axes[1].hexbin(
        sideways["outgoing_azimuth_shifted"],
        sideways["outgoing_elevation"],
        C=sideways["absolute_azimuth_difference"],
        gridsize=45,
        mincnt=1,
        reduce_C_function=np.mean,
        cmap="magma",
        linewidths=0,
    )
    figure.colorbar(hexbin, ax=axes[1], label="Mean absolute azimuth difference [deg]")
    axes[1].set_xlabel("Outgoing azimuth + 136° [deg]")
    axes[1].set_ylabel("Outgoing elevation [deg]")
    axes[1].set_title("Outgoing window: 150°-170°, 5°-20°")
    axes[1].grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(os.path.join(output_path, "sideways_outgoing_window.png"), dpi=180)
    plt.close(figure)


def write_summary(states, sideways, threshold, input_path, output_path):
    large_deflection = sideways["absolute_azimuth_difference"] >= threshold
    survived = states["survived"]
    large_deflection_fraction = (
        large_deflection.mean() * 100.0 if len(sideways) else 0.0
    )
    summary_path = os.path.join(output_path, "analysis_summary.txt")
    with open(summary_path, "w") as summary:
        summary.write(f"Input: {input_path}\n")
        summary.write(f"Azimuth offset: +{AZIMUTH_OFFSET:.1f} deg\n")
        summary.write(f"All transported rows: {len(states)}\n")
        summary.write(f"Positive-weight rows: {survived.sum()}\n")
        summary.write("\nOutgoing window after offset: azimuth 150-170 deg, elevation 5-20 deg\n")
        summary.write(f"Rows in window: {len(sideways)}\n")
        summary.write(f"Rows with positive weight: {sideways['survived'].sum()}\n")
        summary.write(
            f"Rows with absolute azimuth difference >= {threshold:.1f} deg: "
            f"{large_deflection.sum()} ({large_deflection_fraction:.2f}%)\n"
        )
        if len(sideways):
            summary.write(
                f"Median absolute azimuth difference: "
                f"{sideways['absolute_azimuth_difference'].median():.3f} deg\n"
            )
            summary.write(
                f"Median 3D scattering angle: "
                f"{sideways['scattering_angle'].median():.3f} deg\n"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", help="Muon-state Parquet file")
    parser.add_argument(
        "--output-path",
        default="muon_state_analysis",
        help="Directory for plots and summary (default: muon_state_analysis)",
    )
    parser.add_argument(
        "--sideways-threshold",
        type=float,
        default=20.0,
        help="Absolute azimuth difference defining a large sideways deflection",
    )
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    states = add_derived_columns(pd.read_parquet(args.input_path))

    sideways = states[
        states["outgoing_azimuth_shifted"].between(150.0, 170.0)
        & states["outgoing_elevation"].between(5.0, 20.0)
    ].copy()

    save_energy_plot(states, args.output_path)
    save_incoming_outgoing_scatter(states, args.output_path)
    save_direction_grid(
        states,
        "scattering_angle",
        "Mean 3D scattering angle by outgoing direction",
        "Mean scattering angle [deg]",
        "scattering_angle_grid.png",
        args.output_path,
    )
    save_direction_grid(
        states,
        "absolute_azimuth_difference",
        "Mean absolute azimuth difference by outgoing direction",
        "Mean absolute azimuth difference [deg]",
        "azimuth_difference_grid.png",
        args.output_path,
    )
    save_direction_grid(
        states,
        "absolute_theta_difference",
        "Mean absolute theta difference by outgoing direction",
        "Mean absolute theta difference [deg]",
        "theta_difference_grid.png",
        args.output_path,
    )
    save_direction_grid(
        states,
        "incoming_energy",
        "Mean incoming energy by outgoing direction",
        "Mean incoming energy [GeV]",
        "incoming_energy_grid.png",
        args.output_path,
        log_scale=True,
    )
    save_sideways_plot(sideways, args.sideways_threshold, args.output_path)
    sideways.to_csv(os.path.join(args.output_path, "sideways_muons.csv"), index=False)
    write_summary(
        states,
        sideways,
        args.sideways_threshold,
        args.input_path,
        args.output_path,
    )

    print(f"Analyzed {len(states)} transported rows.")
    print(f"Outgoing-window rows: {len(sideways)}")
    print(f"Plots and summary written to: {args.output_path}")


if __name__ == "__main__":
    main()
