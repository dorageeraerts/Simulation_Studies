#!/usr/bin/env python3

from pathlib import Path
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

OUTPUT_DIR = Path(
    "./chi2_distributions/time_evolution_avg"
)

TREE_NAME = "AnalyzedData"

FILENAME_TEMPLATE = (
    "MURAVES_AnalyzedData_run{run}.root"
)

SLOWCONTROL_TEMPLATE = "SLOWCONTROL_run{run}"

# Used for the standard distribution plots.
# Values above this limit are excluded from those plots.
CHI2_MIN = 0.0
CHI2_MAX = 20.0
BIN_WIDTH = 0.1

CHI2_BRANCHES = {
    "3p_xy": "BestTrack_3p_ChiSquare_xy",
    "3p_xz": "BestTrack_3p_ChiSquare_xz",
    "4p_xy": "BestTrack_4p_ChiSquare_xy",
    "4p_xz": "BestTrack_4p_ChiSquare_xz",
}

PLOT_WPS = {15, 20, 25}

# Smaller marker size for all per-run scatter points
RUN_MARKER_SIZE = 2

# Set to True to re-enable all the previous plots
# (histograms, heatmaps, mean/max/quantiles per run, ...)
ENABLE_OLD_PLOTS = False

# Plane order = order of the board pairs in the slow-control file.
# Slow-control indices 6..21 = boards 0..15:
#   index 6,7   -> boards 0,1   -> Z1
#   index 8,9   -> boards 2,3   -> Y1
#   index 10,11 -> boards 4,5   -> Z2
#   ... and so on
# The DAC10 of a plane is the average of its two boards.
PLANES = ["Z1", "Y1", "Z2", "Y2", "Z3", "Y3", "Z4", "Y4"]
DAC_FIRST_INDEX = 6


def planes_for_branch(short_name):
    """3p tracks do not use plane 4."""
    if short_name.startswith("3p"):
        return [p for p in PLANES if not p.endswith("4")]

    return list(PLANES)


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
# COLORS AND MARKERS
# ============================================================

# Working point -> color
WP_COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]

# VES / FS -> marker
TYPE_MARKERS = {
    "VES": "o",
    "FS": "^",
}

# Colors used only for the old distribution plots
VES_COLOR = "tab:blue"
FS_COLOR = "tab:orange"


def make_wp_color_map(records):
    """Assign one color to each working point."""
    wps = sorted(
        set(record["wp"] for record in records)
    )

    return {
        wp: WP_COLORS[i % len(WP_COLORS)]
        for i, wp in enumerate(wps)
    }


def make_wp_color_map_from_run_data(run_data):
    """Assign one color to each WP present in run_data."""
    wps = set()

    for data_type in ("VES", "FS"):
        for wp in run_data.get(data_type, {}):
            wps.add(wp)

    wps = sorted(wps)

    return {
        wp: WP_COLORS[i % len(WP_COLORS)]
        for i, wp in enumerate(wps)
    }


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
            f"Invalid run range: {value}: "
            "end must be >= start."
        )

    return start, stop


def parse_range_list(values):
    if values is None:
        return None

    return [
        parse_run_range(value)
        for value in values
    ]


def iter_run_numbers(run_ranges):
    for start, stop in run_ranges:
        yield from range(start, stop + 1)


# ============================================================
# TIME / SLOW CONTROL
# ============================================================

def parse_slowcontrol_timestamp(fields):
    if len(fields) <= 1:
        return None

    raw = fields[1].strip()

    candidates = [
        raw,
        raw.replace("T", " "),
        raw.replace("/", "-"),
    ]

    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
        "%d-%m-%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
    ]

    for candidate in candidates:
        for fmt in formats:
            try:
                return datetime.strptime(
                    candidate,
                    fmt,
                )
            except ValueError:
                pass

    try:
        value = float(raw)

        if np.isfinite(value):
            if abs(value) >= 1e11:
                value /= 1000.0

            return datetime.fromtimestamp(value)

    except (
        ValueError,
        OverflowError,
        OSError,
    ):
        pass

    return None


def get_slowcontrol_info(
    slowcontrol_dir,
    detector,
    run,
):
    """
    Returns (timestamp, working_point, dac_per_plane) or None.

    dac_per_plane is a dict {plane: DAC10 averaged over the
    two boards of that plane}, or None if the DAC10 values
    could not be read.
    """
    slowcontrol_file = (
        Path(slowcontrol_dir)
        / detector
        / SLOWCONTROL_TEMPLATE.format(run=run)
    )

    if not slowcontrol_file.is_file():
        print(
            "  [warn] slow-control file not found: "
            f"{slowcontrol_file}"
        )
        return None

    try:
        with open(
            slowcontrol_file,
            "r",
            errors="replace",
        ) as f:

            for line in f:
                line = line.strip()

                if not line:
                    continue

                fields = line.split()

                if len(fields) <= 5:
                    print(
                        "  [warn] malformed slow-control "
                        f"line in {slowcontrol_file}"
                    )
                    return None

                timestamp = parse_slowcontrol_timestamp(
                    fields
                )

                if timestamp is None:
                    print(
                        "  [warn] could not parse timestamp "
                        "from slow-control column/index 1: "
                        f"{fields[1]}"
                    )
                    return None

                try:
                    working_point = float(fields[5])
                except ValueError:
                    print(
                        "  [warn] invalid Working Point "
                        f"in {slowcontrol_file}: "
                        f"{fields[5]}"
                    )
                    return None

                if not np.isfinite(working_point):
                    print(
                        "  [warn] non-finite Working Point "
                        f"in {slowcontrol_file}"
                    )
                    return None

                if not working_point.is_integer():
                    print(
                        "  [warn] non-integer Working Point "
                        f"in {slowcontrol_file}: "
                        f"{working_point}"
                    )
                    return None

                # ----------------------------------------
                # DAC10 per plane (average of the 2 boards)
                # ----------------------------------------

                dac_per_plane = None

                try:
                    dac_values = [
                        float(x)
                        for x in fields[
                            DAC_FIRST_INDEX:
                            DAC_FIRST_INDEX + 2 * len(PLANES)
                        ]
                    ]

                    if (
                        len(dac_values) == 2 * len(PLANES)
                        and np.all(np.isfinite(dac_values))
                    ):
                        dac_per_plane = {
                            plane: round(
                                0.5 * (
                                    dac_values[2 * i]
                                    + dac_values[2 * i + 1]
                                ),
                                1,
                            )
                            for i, plane in enumerate(PLANES)
                        }
                    else:
                        print(
                            "  [warn] incomplete DAC10 values "
                            f"in {slowcontrol_file}"
                        )

                except ValueError:
                    print(
                        "  [warn] invalid DAC10 values "
                        f"in {slowcontrol_file}"
                    )

                return (
                    timestamp,
                    int(working_point),
                    dac_per_plane,
                )

    except Exception as exc:
        print(
            f"  [warn] could not read "
            f"{slowcontrol_file}: {exc}"
        )
        return None

    print(
        f"  [warn] no valid lines in "
        f"{slowcontrol_file}"
    )

    return None


def month_key(timestamp):
    return timestamp.strftime("%Y-%m")


def week_key(timestamp):
    iso_year, iso_week, _ = timestamp.isocalendar()

    return (
        f"{iso_year}-W{iso_week:02d}"
    )


def format_month_labels(months):
    return [
        datetime.strptime(
            month,
            "%Y-%m",
        ).strftime("%m-%Y")
        for month in months
    ]


def format_week_labels(weeks):
    return list(weeks)


# ============================================================
# ROOT DATA
# ============================================================

def read_run(file_path):
    branches = list(
        CHI2_BRANCHES.values()
    )

    try:
        with uproot.open(file_path) as f:

            if TREE_NAME not in f:
                print(
                    f"  [warn] tree '{TREE_NAME}' "
                    "not found"
                )
                return None

            tree = f[TREE_NAME]

            tree_keys = set(tree.keys())

            missing = (
                set(branches)
                - tree_keys
            )

            if missing:
                print(
                    "  [warn] missing branches: "
                    f"{sorted(missing)}"
                )
                return None

            return tree.arrays(
                branches,
                library="np",
            )

    except Exception as exc:
        print(
            f"  [warn] could not read "
            f"{file_path}: {exc}"
        )
        return None


def get_valid_chi2(
    values,
    max_value=None,
):
    """
    If max_value is supplied, apply the usual histogram
    range. If max_value is None, keep all finite chi2
    values >= CHI2_MIN.

    The per-run statistics use max_value=None so that
    unusually large chi2 values are not lost from the
    maximum calculation.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    valid = (
        np.isfinite(values)
        & (values >= CHI2_MIN)
    )

    if max_value is not None:
        valid &= (
            values <= max_value
        )

    return values[valid]


def make_bins():
    n_bins = int(
        np.ceil(
            (CHI2_MAX - CHI2_MIN)
            / BIN_WIDTH
        )
    )

    return np.linspace(
        CHI2_MIN,
        CHI2_MAX,
        n_bins + 1,
    )


# ============================================================
# 1D ALL-TIME HISTOGRAM
# ============================================================

def plot_distribution_comparison(
    values_ves,
    values_fs,
    detector,
    working_point,
    branch_name,
    output_file,
):
    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    bins = make_bins()

    if (
        values_ves is not None
        and len(values_ves) > 0
    ):
        ax.hist(
            values_ves,
            bins=bins,
            histtype="step",
            linewidth=1.8,
            density=True,
            color=VES_COLOR,
            label=(
                f"VES (N={len(values_ves):,})"
            ),
        )

    if (
        values_fs is not None
        and len(values_fs) > 0
    ):
        ax.hist(
            values_fs,
            bins=bins,
            histtype="step",
            linewidth=1.8,
            density=True,
            color=FS_COLOR,
            label=(
                f"FS (N={len(values_fs):,})"
            ),
        )

    ax.set_xlabel(
        r"$\chi^2$"
    )

    ax.set_ylabel(
        "Density"
    )

    ax.set_title(
        f"{branch_name} - "
        f"{detector}, WP{working_point}"
    )

    ax.set_xlim(
        CHI2_MIN,
        CHI2_MAX,
    )

    ax.grid(
        True,
        alpha=0.3,
        linestyle="--",
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# MONTHLY HISTOGRAMS
# ============================================================

def plot_monthly_histograms(
    values_by_month_ves,
    values_by_month_fs,
    detector,
    working_point,
    branch_name,
    output_file,
):
    months = sorted(
        set(values_by_month_ves)
        | set(values_by_month_fs)
    )

    if not months:
        return

    n_months = len(months)

    ncols = min(
        3,
        n_months,
    )

    nrows = int(
        np.ceil(
            n_months / ncols
        )
    )

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(
            5.0 * ncols,
            3.8 * nrows,
        ),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    axes = axes.ravel()

    bins = make_bins()

    ymax = 0.0

    for month in months:
        for values_by_month in (
            values_by_month_ves,
            values_by_month_fs,
        ):
            if month not in values_by_month:
                continue

            values = values_by_month[month]

            if len(values) == 0:
                continue

            hist, _ = np.histogram(
                values,
                bins=bins,
                density=True,
            )

            finite_hist = hist[
                np.isfinite(hist)
            ]

            if len(finite_hist) > 0:
                ymax = max(
                    ymax,
                    np.max(finite_hist),
                )

    if ymax <= 0:
        ymax = 1.0

    ymax *= 1.10

    for i, month in enumerate(months):
        ax = axes[i]

        values_ves = (
            values_by_month_ves.get(
                month,
                np.array([]),
            )
        )

        values_fs = (
            values_by_month_fs.get(
                month,
                np.array([]),
            )
        )

        if len(values_ves) > 0:
            ax.hist(
                values_ves,
                bins=bins,
                histtype="step",
                linewidth=1.5,
                density=True,
                color=VES_COLOR,
                label=(
                    f"VES "
                    f"(N={len(values_ves):,})"
                ),
            )

        if len(values_fs) > 0:
            ax.hist(
                values_fs,
                bins=bins,
                histtype="step",
                linewidth=1.5,
                density=True,
                color=FS_COLOR,
                label=(
                    f"FS "
                    f"(N={len(values_fs):,})"
                ),
            )

        ax.set_title(
            datetime.strptime(
                month,
                "%Y-%m",
            ).strftime("%m-%Y")
        )

        ax.set_xlim(
            CHI2_MIN,
            CHI2_MAX,
        )

        ax.set_ylim(
            0,
            ymax,
        )

        ax.grid(
            True,
            alpha=0.25,
            linestyle="--",
        )

        ax.legend(
            fontsize=8,
        )

    for i in range(
        n_months,
        len(axes),
    ):
        fig.delaxes(
            axes[i]
        )

    fig.supxlabel(
        r"$\chi^2$"
    )

    fig.supylabel(
        "Density"
    )

    fig.suptitle(
        f"{branch_name} - "
        f"{detector}, WP{working_point}",
        y=0.995,
    )

    fig.tight_layout(
        rect=[
            0.03,
            0.03,
            1.0,
            0.95,
        ]
    )

    fig.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# HEATMAP MATRIX
# ============================================================

def build_heatmap_matrix(
    values_by_period,
    periods,
):
    bins = make_bins()

    matrix = []

    for period in periods:

        if period in values_by_period:
            values = (
                values_by_period[period]
            )

            if len(values) > 0:
                hist, _ = np.histogram(
                    values,
                    bins=bins,
                    density=True,
                )
            else:
                hist = np.full(
                    len(bins) - 1,
                    np.nan,
                )

        else:
            hist = np.full(
                len(bins) - 1,
                np.nan,
            )

        matrix.append(hist)

    if matrix:
        matrix = np.asarray(matrix)
    else:
        matrix = np.empty(
            (
                0,
                len(bins) - 1,
            )
        )

    return bins, matrix


# ============================================================
# WEEKLY VS MONTHLY HEATMAPS
# ============================================================

def plot_weekly_monthly_heatmap_comparison(
    values_by_week_ves,
    values_by_week_fs,
    values_by_month_ves,
    values_by_month_fs,
    detector,
    working_point,
    branch_name,
    output_file,
):
    weeks = sorted(
        set(values_by_week_ves)
        | set(values_by_week_fs)
    )

    months = sorted(
        set(values_by_month_ves)
        | set(values_by_month_fs)
    )

    if not weeks and not months:
        return

    bins_week, weekly_ves = (
        build_heatmap_matrix(
            values_by_week_ves,
            weeks,
        )
    )

    _, weekly_fs = (
        build_heatmap_matrix(
            values_by_week_fs,
            weeks,
        )
    )

    bins_month, monthly_ves = (
        build_heatmap_matrix(
            values_by_month_ves,
            months,
        )
    )

    _, monthly_fs = (
        build_heatmap_matrix(
            values_by_month_fs,
            months,
        )
    )

    if not np.allclose(
        bins_week,
        bins_month,
    ):
        raise RuntimeError(
            "Weekly and monthly heatmaps "
            "do not use the same chi2 bins."
        )

    bins = bins_week

    finite_arrays = []

    for matrix in (
        weekly_ves,
        weekly_fs,
        monthly_ves,
        monthly_fs,
    ):
        finite_values = matrix[
            np.isfinite(matrix)
        ]

        if len(finite_values) > 0:
            finite_arrays.append(
                finite_values
            )

    if finite_arrays:
        all_values = np.concatenate(
            finite_arrays
        )

        vmax = np.max(all_values)
    else:
        vmax = 1.0

    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    fig = plt.figure(
        figsize=(
            16,
            max(
                8,
                0.45
                * max(
                    len(weeks),
                    len(months),
                    1,
                )
                + 5,
            ),
        )
    )

    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        width_ratios=[
            1,
            1,
            0.045,
        ],
        height_ratios=[
            1,
            1,
        ],
        wspace=0.15,
        hspace=0.25,
    )

    ax_week_ves = fig.add_subplot(
        gs[0, 0]
    )

    ax_month_ves = fig.add_subplot(
        gs[0, 1]
    )

    ax_week_fs = fig.add_subplot(
        gs[1, 0],
        sharey=ax_week_ves,
    )

    ax_month_fs = fig.add_subplot(
        gs[1, 1],
        sharey=ax_week_ves,
    )

    cbar_ax = fig.add_subplot(
        gs[:, 2]
    )

    def plot_heatmap(
        ax,
        matrix,
        periods,
        labels,
        title,
        show_xlabel,
    ):
        x = np.arange(
            len(periods) + 1
        )

        mesh = ax.pcolormesh(
            x,
            bins,
            matrix.T,
            shading="auto",
            cmap="viridis",
            vmin=0,
            vmax=vmax,
        )

        ax.set_title(title)

        ax.set_ylabel(
            r"$\chi^2$"
        )

        ax.set_ylim(
            CHI2_MIN,
            CHI2_MAX,
        )

        ax.set_xlim(
            0,
            len(periods),
        )

        ax.set_xticks(
            np.arange(
                len(periods)
            ) + 0.5
        )

        if show_xlabel:
            ax.set_xticklabels(
                labels,
                rotation=45,
                ha="right",
            )

            ax.set_xlabel(
                "Time period"
            )
        else:
            ax.set_xticklabels([])

        return mesh

    mesh_week_ves = plot_heatmap(
        ax_week_ves,
        weekly_ves,
        weeks,
        format_week_labels(weeks),
        "Weekly - VES",
        False,
    )

    plot_heatmap(
        ax_week_fs,
        weekly_fs,
        weeks,
        format_week_labels(weeks),
        "Weekly - FS",
        True,
    )

    plot_heatmap(
        ax_month_ves,
        monthly_ves,
        months,
        format_month_labels(months),
        "Monthly - VES",
        False,
    )

    plot_heatmap(
        ax_month_fs,
        monthly_fs,
        months,
        format_month_labels(months),
        "Monthly - FS",
        True,
    )

    cbar = fig.colorbar(
        mesh_week_ves,
        cax=cbar_ax,
    )

    cbar.set_label(
        "Density"
    )

    fig.suptitle(
        f"{branch_name} - "
        f"{detector}, WP{working_point}",
        y=0.98,
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.93,
        bottom=0.13,
        top=0.90,
    )

    fig.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# WEEKLY QUANTILES
# ============================================================

def calculate_quantiles(
    values_by_period,
    periods,
):
    q10 = []
    q25 = []
    q50 = []
    q75 = []
    q90 = []

    for period in periods:

        if (
            period not in values_by_period
            or len(values_by_period[period]) == 0
        ):
            q10.append(np.nan)
            q25.append(np.nan)
            q50.append(np.nan)
            q75.append(np.nan)
            q90.append(np.nan)
            continue

        values = (
            values_by_period[period]
        )

        q10.append(
            np.percentile(values, 10)
        )

        q25.append(
            np.percentile(values, 25)
        )

        q50.append(
            np.percentile(values, 50)
        )

        q75.append(
            np.percentile(values, 75)
        )

        q90.append(
            np.percentile(values, 90)
        )

    return (
        np.asarray(q10),
        np.asarray(q25),
        np.asarray(q50),
        np.asarray(q75),
        np.asarray(q90),
    )


def plot_weekly_quantiles_comparison(
    values_by_week_ves,
    values_by_week_fs,
    detector,
    working_point,
    branch_name,
    output_file,
):
    weeks = sorted(
        set(values_by_week_ves)
        | set(values_by_week_fs)
    )

    if not weeks:
        return

    x = np.arange(
        len(weeks)
    )

    (
        q10_ves,
        q25_ves,
        q50_ves,
        q75_ves,
        q90_ves,
    ) = calculate_quantiles(
        values_by_week_ves,
        weeks,
    )

    (
        q10_fs,
        q25_fs,
        q50_fs,
        q75_fs,
        q90_fs,
    ) = calculate_quantiles(
        values_by_week_fs,
        weeks,
    )

    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    ax.fill_between(
        x,
        q10_ves,
        q90_ves,
        color=VES_COLOR,
        alpha=0.12,
        label="VES 10-90 percentile",
    )

    ax.fill_between(
        x,
        q25_ves,
        q75_ves,
        color=VES_COLOR,
        alpha=0.25,
        label="VES 25-75 percentile",
    )

    ax.plot(
        x,
        q50_ves,
        color=VES_COLOR,
        marker="o",
        linewidth=1.8,
        markersize=3,
        label="VES median",
    )

    ax.fill_between(
        x,
        q10_fs,
        q90_fs,
        color=FS_COLOR,
        alpha=0.12,
        label="FS 10-90 percentile",
    )

    ax.fill_between(
        x,
        q25_fs,
        q75_fs,
        color=FS_COLOR,
        alpha=0.25,
        label="FS 25-75 percentile",
    )

    ax.plot(
        x,
        q50_fs,
        color=FS_COLOR,
        marker="s",
        linewidth=1.8,
        markersize=3,
        label="FS median",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        format_week_labels(weeks),
        rotation=45,
        ha="right",
    )

    ax.set_xlabel(
        "ISO week"
    )

    ax.set_ylabel(
        r"$\chi^2$"
    )

    ax.set_title(
        f"{branch_name} - "
        f"{detector}, WP{working_point}"
    )

    ax.set_ylim(
        CHI2_MIN,
        CHI2_MAX,
    )

    ax.grid(
        True,
        alpha=0.3,
        linestyle="--",
    )

    ax.legend(
        ncol=2,
        loc="best",
    )

    fig.tight_layout()

    fig.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# PER-RUN STATISTICS
# ============================================================

def calculate_run_statistics(
    run_values
):
    runs = sorted(run_values)

    mean_values = []
    median_values = []
    max_values = []
    q25_values = []
    q75_values = []
    q05_values = []
    q95_values = []

    for run in runs:

        values = run_values[run]

        if (
            values is None
            or len(values) == 0
        ):
            mean_values.append(np.nan)
            median_values.append(np.nan)
            max_values.append(np.nan)
            q25_values.append(np.nan)
            q75_values.append(np.nan)
            q05_values.append(np.nan)
            q95_values.append(np.nan)
            continue

        mean_values.append(
            np.mean(values)
        )

        median_values.append(
            np.percentile(values, 50)
        )

        max_values.append(
            np.max(values)
        )

        q25_values.append(
            np.percentile(values, 25)
        )

        q75_values.append(
            np.percentile(values, 75)
        )

        q05_values.append(
            np.percentile(values, 5)
        )

        q95_values.append(
            np.percentile(values, 95)
        )

    return {
        "runs": np.asarray(runs),
        "mean": np.asarray(mean_values),
        "median": np.asarray(median_values),
        "max": np.asarray(max_values),
        "q25": np.asarray(q25_values),
        "q75": np.asarray(q75_values),
        "q05": np.asarray(q05_values),
        "q95": np.asarray(q95_values),
    }


# ============================================================
# SECOND X AXIS: MONTH OR ISO WEEK
# ============================================================

def add_period_axis(
    ax,
    run_info,
    period_type,
):
    if not run_info:
        return None

    runs = sorted(run_info)

    if len(runs) == 0:
        return None

    period_to_runs = {}

    for run in runs:

        timestamp = (
            run_info[run]["timestamp"]
        )

        if period_type == "month":
            period = month_key(timestamp)
        else:
            period = week_key(timestamp)

        period_to_runs.setdefault(
            period,
            [],
        ).append(run)

    period_positions = []
    period_labels = []

    for period, period_runs in (
        period_to_runs.items()
    ):
        position = np.mean(period_runs)

        period_positions.append(
            position
        )

        if period_type == "month":
            label = datetime.strptime(
                period,
                "%Y-%m",
            ).strftime("%m-%Y")
        else:
            label = period

        period_labels.append(label)

    # Avoid an unreadable top axis when there are many weeks.
    max_labels = 20

    if len(period_positions) > max_labels:
        indices = np.linspace(
            0,
            len(period_positions) - 1,
            max_labels,
        ).round().astype(int)

        indices = np.unique(indices)

        period_positions = [
            period_positions[i]
            for i in indices
        ]

        period_labels = [
            period_labels[i]
            for i in indices
        ]

    ax_top = ax.secondary_xaxis(
        "top"
    )

    ax_top.set_xticks(
        period_positions
    )

    ax_top.set_xticklabels(
        period_labels,
        rotation=45,
        ha="left",
        fontsize=8,
    )

    if period_type == "month":
        ax_top.set_xlabel(
            "Month"
        )
    else:
        ax_top.set_xlabel(
            "ISO week"
        )

    return ax_top


# ============================================================
# LEGEND FOR WP COLORS + VES/FS MARKERS
# ============================================================

def add_wp_type_legend(
    ax,
    wp_colors,
):
    handles = []

    # WP colors
    for wp in sorted(wp_colors):

        handles.append(
            Line2D(
                [0],
                [0],
                color=wp_colors[wp],
                marker="o",
                linestyle="-",
                markersize=6,
                linewidth=1.5,
                label=f"WP{wp}",
            )
        )

    # VES / FS markers
    for data_type in (
        "VES",
        "FS",
    ):
        handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                marker=TYPE_MARKERS[data_type],
                linestyle="None",
                markersize=7,
                label=data_type,
            )
        )

    ax.legend(
        handles=handles,
        ncol=2,
        fontsize=8,
        loc="best",
    )


# ============================================================
# PER-RUN MEAN / MEDIAN / MAX
# ============================================================

def plot_run_statistic(
    run_data,
    run_info,
    detector,
    branch_name,
    statistic,
    output_file,
    period_type="month",
):
    if not run_data:
        return

    fig, ax = plt.subplots(
        figsize=(15, 7)
    )

    all_runs = []

    for data_type in (
        "VES",
        "FS",
    ):
        for wp in run_data.get(
            data_type,
            {},
        ):
            if wp not in PLOT_WPS:
                continue
            all_runs.extend(
                run_data[data_type][wp].keys()
            )

    if not all_runs:
        plt.close(fig)
        return

    xmin = min(all_runs)
    xmax = max(all_runs)

    wp_colors = make_wp_color_map_from_run_data(
        run_data
    )

    for data_type in (
        "VES",
        "FS",
    ):
        for wp in sorted(
            run_data.get(
                data_type,
                {},
            )
        ):
            if wp not in PLOT_WPS:
                continue

            stats = calculate_run_statistics(
                run_data[data_type][wp]
            )

            runs = stats["runs"]

            if statistic not in stats:
                continue

            y = stats[statistic]

            if len(runs) == 0:
                continue

            valid = np.isfinite(y)

            if not np.any(valid):
                continue

            color = wp_colors[wp]
            marker = TYPE_MARKERS[data_type]

            ax.plot(
                runs[valid],
                y[valid],
                linestyle="None",
                marker=marker,
                markersize=RUN_MARKER_SIZE,
                linewidth=1.4,
                color=color,
                alpha=0.9,
            )

    xrange = max(
        1,
        xmax - xmin,
    )

    ax.set_xlim(
        xmin - 0.02 * xrange,
        xmax + 0.02 * xrange,
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # The MAX plot is NOT limited to CHI2_MAX.
    # --------------------------------------------------------
    if statistic == "max":

        finite_y = []

        for data_type in (
            "VES",
            "FS",
        ):
            for wp in run_data.get(
                data_type,
                {},
            ):
                if wp not in PLOT_WPS:
                    continue

                stats = calculate_run_statistics(
                    run_data[data_type][wp]
                )

                values = stats["max"]

                values = values[
                    np.isfinite(values)
                ]

                if len(values) > 0:
                    finite_y.append(values)

        if finite_y:
            maximum = np.max(
                np.concatenate(finite_y)
            )

            if maximum > 0:
                ymax = maximum * 1.08
            else:
                ymax = 1.0

            ax.set_ylim(
                CHI2_MIN,
                ymax,
            )

    else:
        ax.set_ylim(
            bottom=CHI2_MIN
        )

    ax.set_xlabel(
        "Run number"
    )

    ax.set_ylabel(
        r"$\chi^2$"
    )

    statistic_labels = {
        "mean": "Mean",
        "median": "Median",
        "max": "Maximum",
    }

    ax.set_title(
        f"{branch_name} - "
        f"{detector} - "
        f"{statistic_labels[statistic]} per run"
    )

    ax.grid(
        True,
        alpha=0.3,
        linestyle="--",
    )

    add_period_axis(
        ax,
        run_info,
        period_type,
    )

    add_wp_type_legend(
        ax,
        wp_colors,
    )

    fig.tight_layout()

    fig.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# NEW: MEDIAN PER RUN, ONE PLOT PER PLANE, COLORED BY (WP, DAC10)
# ============================================================

def plot_median_per_run_dac(
    run_data,
    run_info,
    detector,
    branch_name,
    plane,
    output_file,
    period_type="month",
):
    """
    Median chi2 per run, for one plane.

    Color  = (working point, DAC10 of this plane)
    Marker = VES / FS

    The median itself is a track quantity, so it is identical
    in the plots of all planes; only the coloring changes.
    """
    points = {}   # (wp, dac, data_type) -> ([runs], [medians])

    for data_type in ("VES", "FS"):
        for wp, runs_dict in run_data.get(data_type, {}).items():

            if wp not in PLOT_WPS:
                continue

            for run, values in runs_dict.items():

                if values is None or len(values) == 0:
                    continue

                dac_dict = run_info.get(run, {}).get("dac")

                if dac_dict is None:
                    continue

                key = (wp, dac_dict[plane], data_type)

                points.setdefault(key, ([], []))
                points[key][0].append(run)
                points[key][1].append(np.median(values))

    if not points:
        return

    combos = sorted({(wp, dac) for wp, dac, _ in points})

    if len(combos) <= 20:
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i) for i in range(len(combos))]
    else:
        cmap = plt.get_cmap("turbo")
        colors = [
            cmap(x)
            for x in np.linspace(0, 1, len(combos))
        ]

    color_map = dict(zip(combos, colors))

    fig, ax = plt.subplots(figsize=(15, 7))

    for (wp, dac, data_type), (runs, meds) in points.items():

        order = np.argsort(runs)

        ax.plot(
            np.asarray(runs)[order],
            np.asarray(meds)[order],
            linestyle="None",
            marker=TYPE_MARKERS[data_type],
            markersize=RUN_MARKER_SIZE + 1,
            color=color_map[(wp, dac)],
            alpha=0.9,
        )

    all_runs = [
        r
        for runs, _ in points.values()
        for r in runs
    ]

    xmin = min(all_runs)
    xmax = max(all_runs)
    xrange = max(1, xmax - xmin)

    ax.set_xlim(
        xmin - 0.02 * xrange,
        xmax + 0.02 * xrange,
    )

    ax.set_ylim(bottom=CHI2_MIN)

    ax.set_xlabel("Run number")
    ax.set_ylabel(r"$\chi^2$")

    ax.set_title(
        f"{branch_name} - {detector} - "
        f"Median per run - {plane} DAC10 setting"
    )

    ax.grid(True, alpha=0.3, linestyle="--")

    add_period_axis(
        ax,
        run_info,
        period_type,
    )

    handles = [
        Line2D(
            [0],
            [0],
            color=color_map[(wp, dac)],
            marker="o",
            linestyle="None",
            markersize=6,
            label=f"WP{wp}, DAC10={dac:g}",
        )
        for wp, dac in combos
    ]

    for data_type in ("VES", "FS"):
        handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                marker=TYPE_MARKERS[data_type],
                linestyle="None",
                markersize=7,
                label=data_type,
            )
        )

    ax.legend(
        handles=handles,
        ncol=2,
        fontsize=8,
        loc="best",
    )

    fig.tight_layout()

    fig.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# PER-RUN MEDIAN + PERCENTILE BANDS
# ============================================================

def plot_run_quantiles(
    run_data,
    run_info,
    detector,
    branch_name,
    output_file,
    low_key,
    high_key,
    band_label,
    period_type="month",
):
    if not run_data:
        return

    fig, ax = plt.subplots(
        figsize=(15, 7)
    )

    all_runs = []

    for data_type in (
        "VES",
        "FS",
    ):
        for wp in run_data.get(
            data_type,
            {},
        ):
            all_runs.extend(
                run_data[data_type][wp].keys()
            )

    if not all_runs:
        plt.close(fig)
        return

    xmin = min(all_runs)
    xmax = max(all_runs)

    wp_colors = make_wp_color_map_from_run_data(
        run_data
    )

    for data_type in (
        "VES",
        "FS",
    ):
        for wp in sorted(
            run_data.get(
                data_type,
                {},
            )
        ):
            stats = calculate_run_statistics(
                run_data[data_type][wp]
            )

            runs = stats["runs"]

            if len(runs) == 0:
                continue

            median = stats["median"]
            low = stats[low_key]
            high = stats[high_key]

            valid = (
                np.isfinite(median)
                & np.isfinite(low)
                & np.isfinite(high)
            )

            if not np.any(valid):
                continue

            color = wp_colors[wp]
            marker = TYPE_MARKERS[data_type]

            # Percentile band uses the WP color.
            ax.fill_between(
                runs[valid],
                low[valid],
                high[valid],
                color=color,
                alpha=0.15,
            )

            # Median uses the WP color and VES/FS marker.
            ax.plot(
                runs[valid],
                median[valid],
                color=color,
                linewidth=1.4,
                marker=marker,
                markersize=5,
            )

    xrange = max(
        1,
        xmax - xmin,
    )

    ax.set_xlim(
        xmin - 0.02 * xrange,
        xmax + 0.02 * xrange,
    )

    ax.set_ylim(
        CHI2_MIN,
        CHI2_MAX,
    )

    ax.set_xlabel(
        "Run number"
    )

    ax.set_ylabel(
        r"$\chi^2$"
    )

    ax.set_title(
        f"{branch_name} - "
        f"{detector} - "
        f"Median with {band_label} per run"
    )

    ax.grid(
        True,
        alpha=0.3,
        linestyle="--",
    )

    add_period_axis(
        ax,
        run_info,
        period_type,
    )

    add_wp_type_legend(
        ax,
        wp_colors,
    )

    fig.tight_layout()

    fig.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# PER-RUN QUANTILES: 5-25-50-75-95%
# ============================================================

def plot_run_quantiles_full(
    run_data,
    run_info,
    detector,
    branch_name,
    output_file,
    period_type="month",
):
    if not run_data:
        return

    wp_colors = make_wp_color_map_from_run_data(run_data)

    working_points = sorted(
        {
            wp
            for data_type in ("VES", "FS")
            for wp in run_data.get(data_type, {})
        }
    )

    for wp in working_points:
        fig, ax = plt.subplots(figsize=(15, 7))

        wp_runs = []

        for data_type in ("VES", "FS"):
            if wp not in run_data.get(data_type, {}):
                continue

            wp_runs.extend(
                run_data[data_type][wp].keys()
            )

        if not wp_runs:
            plt.close(fig)
            continue

        xmin = min(wp_runs)
        xmax = max(wp_runs)

        color = wp_colors[wp]

        for data_type in ("VES", "FS"):
            if wp not in run_data.get(data_type, {}):
                continue

            stats = calculate_run_statistics(
                run_data[data_type][wp]
            )

            runs = stats["runs"]

            if len(runs) == 0:
                continue

            q05 = stats["q05"]
            q25 = stats["q25"]
            median = stats["median"]
            q75 = stats["q75"]
            q95 = stats["q95"]

            valid = (
                np.isfinite(q05)
                & np.isfinite(q25)
                & np.isfinite(median)
                & np.isfinite(q75)
                & np.isfinite(q95)
            )

            if not np.any(valid):
                continue

            marker = TYPE_MARKERS[data_type]

            # 5-95 percentile band
            ax.fill_between(
                runs[valid],
                q05[valid],
                q95[valid],
                color=color,
                alpha=0.12,
            )

            # 25-75 percentile band
            ax.fill_between(
                runs[valid],
                q25[valid],
                q75[valid],
                color=color,
                alpha=0.25,
            )

            # Median
            ax.plot(
                runs[valid],
                median[valid],
                color=color,
                linewidth=1.5,
                marker=marker,
                markersize=5,
                label=data_type,
            )

        xrange = max(
            1,
            xmax - xmin,
        )

        ax.set_xlim(
            xmin - 0.02 * xrange,
            xmax + 0.02 * xrange,
        )

        # Let matplotlib choose the upper y-axis limit
        ax.set_ylim(
            bottom=CHI2_MIN
        )

        ax.set_xlabel(
            "Run number"
        )

        ax.set_ylabel(
            r"$\chi^2$"
        )

        ax.set_title(
            f"{branch_name} - "
            f"{detector} - "
            f"WP{wp} - "
            "Median + 5-25-75-95 percentiles per run"
        )

        ax.grid(
            True,
            alpha=0.3,
            linestyle="--",
        )

        add_period_axis(
            ax,
            run_info,
            period_type,
        )

        ax.legend(
            title=f"Working point {wp}",
        )

        fig.tight_layout()

        output_path = Path(output_file)

        output_path = output_path.with_name(
            output_path.stem
            + f"_WP{wp}"
            + output_path.suffix
        )

        fig.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Plot MURAVES chi2 distributions "
            "and per-run statistics."
        )
    )

    parser.add_argument(
        "--detector",
        choices=sorted(
            RUN_RANGES.keys()
        ),
        help="Detector to process, e.g. ROSSO",
    )

    parser.add_argument(
        "--detectors",
        nargs="+",
        choices=sorted(
            RUN_RANGES.keys()
        ),
        help="Multiple detectors to process",
    )

    parser.add_argument(
        "--ves-ranges",
        nargs="+",
        metavar="START-END",
        help="Override default VES run ranges",
    )

    parser.add_argument(
        "--fs-ranges",
        nargs="+",
        metavar="START-END",
        help="Override default FS run ranges",
    )

    parser.add_argument(
        "--chi2-max",
        type=float,
        default=CHI2_MAX,
        help=(
            f"Maximum chi2 used for the "
            f"distribution plots, "
            f"default={CHI2_MAX}"
        ),
    )

    parser.add_argument(
        "--bin-width",
        type=float,
        default=BIN_WIDTH,
        help=(
            f"Histogram bin width, "
            f"default={BIN_WIDTH}"
        ),
    )

    parser.add_argument(
        "--period-axis",
        choices=[
            "month",
            "week",
        ],
        default="month",
        help=(
            "Period shown on the second x-axis "
            "(default: month)"
        ),
    )

    parser.add_argument(
        "--old-plots",
        action="store_true",
        help=(
            "Also produce the previous plots "
            "(same as ENABLE_OLD_PLOTS = True)"
        ),
    )

    args = parser.parse_args()

    if args.detector and args.detectors:
        parser.error(
            "Use either --detector or "
            "--detectors, not both."
        )

    if args.detector:
        detectors = [args.detector]

    elif args.detectors:
        detectors = args.detectors

    else:
        detectors = ["ROSSO"]

    if args.chi2_max <= CHI2_MIN:
        parser.error(
            "--chi2-max must be > 0"
        )

    if args.bin_width <= 0:
        parser.error(
            "--bin-width must be > 0"
        )

    return (
        args,
        detectors,
        parse_range_list(
            args.ves_ranges
        ),
        parse_range_list(
            args.fs_ranges
        ),
    )


# ============================================================
# MAIN
# ============================================================

def main():
    global CHI2_MAX
    global BIN_WIDTH
    global ENABLE_OLD_PLOTS

    (
        args,
        detectors,
        ves_override,
        fs_override,
    ) = parse_arguments()

    CHI2_MAX = args.chi2_max
    BIN_WIDTH = args.bin_width

    if args.old_plots:
        ENABLE_OLD_PLOTS = True

    print("=" * 70)
    print("MURAVES CHI2 DISTRIBUTIONS")
    print("=" * 70)

    print(
        "Detectors : "
        f"{', '.join(detectors)}"
    )

    print(
        "Chi2 range for distribution plots: "
        f"{CHI2_MIN} -> {CHI2_MAX}"
    )

    print(
        "Bin width : "
        f"{BIN_WIDTH}"
    )

    print(
        "Per-run statistics use all finite "
        "chi2 values >= 0."
    )

    print(
        "Color      = (WP, DAC10 of the plane)"
    )

    print(
        "VES        = circle marker"
    )

    print(
        "FS         = triangle marker"
    )

    print(
        "Period axis: "
        f"{args.period_axis}"
    )

    print(
        "Old plots  : "
        f"{'ON' if ENABLE_OLD_PLOTS else 'OFF'}"
    )

    print("=" * 70)

    # ========================================================
    # DETECTORS
    # ========================================================

    for detector in detectors:

        print()
        print("#" * 70)
        print(
            f"DETECTOR: {detector}"
        )
        print("#" * 70)

        data_dir = Path(
            DATA_DIR_TEMPLATE.format(
                detector=detector
            )
        )

        detector_output = (
            OUTPUT_DIR / detector
        )

        detector_output.mkdir(
            parents=True,
            exist_ok=True,
        )

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

        # ====================================================
        # Original period-based data
        # ====================================================

        data = {
            "VES": {},
            "FS": {},
        }

        # ====================================================
        # Per-run data
        #
        # run_data[data_type][WP][run][branch] = chi2 array
        # ====================================================

        run_data = {
            "VES": {},
            "FS": {},
        }

        # ====================================================
        # Run information
        # ====================================================

        run_info = {}

        # ====================================================
        # READ DATA
        # ====================================================

        for data_type in (
            "VES",
            "FS",
        ):

            print()
            print(
                "========== "
                f"{detector} / {data_type} "
                "=========="
            )

            for run in iter_run_numbers(
                run_ranges[data_type]
            ):

                file_path = (
                    data_dir
                    / FILENAME_TEMPLATE.format(
                        run=run
                    )
                )

                if not file_path.is_file():
                    print(
                        f"Run {run}: file not found, "
                        "skipping"
                    )
                    continue

                print(
                    f"Run {run}: "
                    f"{file_path.name}"
                )

                # --------------------------------------------
                # Slow control
                # --------------------------------------------

                slowcontrol_info = (
                    get_slowcontrol_info(
                        SLOWCONTROL_DIR,
                        detector,
                        run,
                    )
                )

                if slowcontrol_info is None:
                    print(
                        "  [warn] no timestamp/WP, "
                        "skipping run"
                    )
                    continue

                (
                    timestamp,
                    working_point,
                    dac_per_plane,
                ) = slowcontrol_info

                month = month_key(
                    timestamp
                )

                week = week_key(
                    timestamp
                )

                print(
                    "  Timestamp: "
                    f"{timestamp.strftime('%d-%m-%Y %H:%M:%S')}"
                )

                print(
                    f"  Month: {month}"
                )

                print(
                    f"  ISO week: {week}"
                )

                print(
                    "  Working Point: "
                    f"{working_point}"
                )

                if dac_per_plane is not None:
                    print(
                        "  DAC10: "
                        + ", ".join(
                            f"{plane}={dac_per_plane[plane]:g}"
                            for plane in PLANES
                        )
                    )

                # --------------------------------------------
                # Save run metadata
                # --------------------------------------------

                run_info[run] = {
                    "timestamp": timestamp,
                    "month": month,
                    "week": week,
                    "working_point": working_point,
                    "data_type": data_type,
                    "dac": dac_per_plane,
                }

                # --------------------------------------------
                # Read ROOT file
                # --------------------------------------------

                arrays = read_run(
                    file_path
                )

                if arrays is None:
                    continue

                # --------------------------------------------
                # Create WP containers
                # --------------------------------------------

                if (
                    working_point
                    not in data[data_type]
                ):
                    data[data_type][
                        working_point
                    ] = {
                        key: {
                            "monthly": {},
                            "weekly": {},
                        }
                        for key
                        in CHI2_BRANCHES
                    }

                if (
                    working_point
                    not in run_data[data_type]
                ):
                    run_data[data_type][
                        working_point
                    ] = {}

                # --------------------------------------------
                # Process branches
                # --------------------------------------------

                for (
                    short_name,
                    branch_name,
                ) in CHI2_BRANCHES.items():

                    raw_values = arrays[
                        branch_name
                    ]

                    # ----------------------------------------
                    # Values for the standard plots:
                    # restricted to 0 <= chi2 <= CHI2_MAX
                    # ----------------------------------------

                    values_hist = get_valid_chi2(
                        raw_values,
                        max_value=CHI2_MAX,
                    )

                    # ----------------------------------------
                    # Values for per-run statistics:
                    # keep all finite values >= 0.
                    # ----------------------------------------

                    values_run = get_valid_chi2(
                        raw_values,
                        max_value=None,
                    )

                    # ----------------------------------------
                    # PERIOD DATA
                    # ----------------------------------------

                    period_data = (
                        data[data_type]
                        [working_point]
                        [short_name]
                    )

                    monthly = (
                        period_data["monthly"]
                    )

                    weekly = (
                        period_data["weekly"]
                    )

                    if len(values_hist) > 0:

                        monthly.setdefault(
                            month,
                            [],
                        ).append(
                            values_hist
                        )

                        weekly.setdefault(
                            week,
                            [],
                        ).append(
                            values_hist
                        )

                    # ----------------------------------------
                    # PER-RUN DATA
                    # ----------------------------------------

                    if len(values_run) > 0:

                        if run not in run_data[
                            data_type
                        ][working_point]:

                            run_data[
                                data_type
                            ][working_point][run] = {}

                        run_data[
                            data_type
                        ][working_point][run][
                            short_name
                        ] = values_run

                    print(
                        f"    {short_name}: "
                        f"{len(values_run):,} entries "
                        f"for per-run statistics"
                    )

        # ====================================================
        # EXISTING WP/PERIOD PLOTS (old, disabled by default)
        # ====================================================

        working_points = sorted(
            set(data["VES"])
            | set(data["FS"])
        )

        if ENABLE_OLD_PLOTS:

            print()
            print(
                f"Creating comparison plots "
                f"for {detector}..."
            )

            for working_point in working_points:

                for (
                    short_name,
                    branch_name,
                ) in CHI2_BRANCHES.items():

                    monthly_arrays_ves = (
                        data["VES"]
                        .get(working_point, {})
                        .get(short_name, {})
                        .get("monthly", {})
                    )

                    monthly_arrays_fs = (
                        data["FS"]
                        .get(working_point, {})
                        .get(short_name, {})
                        .get("monthly", {})
                    )

                    weekly_arrays_ves = (
                        data["VES"]
                        .get(working_point, {})
                        .get(short_name, {})
                        .get("weekly", {})
                    )

                    weekly_arrays_fs = (
                        data["FS"]
                        .get(working_point, {})
                        .get(short_name, {})
                        .get("weekly", {})
                    )

                    if (
                        not monthly_arrays_ves
                        and not monthly_arrays_fs
                        and not weekly_arrays_ves
                        and not weekly_arrays_fs
                    ):
                        print(
                            f"  WP{working_point} "
                            f"{short_name}: "
                            "no entries, skipping"
                        )
                        continue

                    # ----------------------------------------
                    # Combine runs within month
                    # ----------------------------------------

                    values_by_month_ves = {
                        month: np.concatenate(arrays)
                        for month, arrays
                        in monthly_arrays_ves.items()
                        if arrays
                    }

                    values_by_month_fs = {
                        month: np.concatenate(arrays)
                        for month, arrays
                        in monthly_arrays_fs.items()
                        if arrays
                    }

                    # ----------------------------------------
                    # Combine runs within week
                    # ----------------------------------------

                    values_by_week_ves = {
                        week: np.concatenate(arrays)
                        for week, arrays
                        in weekly_arrays_ves.items()
                        if arrays
                    }

                    values_by_week_fs = {
                        week: np.concatenate(arrays)
                        for week, arrays
                        in weekly_arrays_fs.items()
                        if arrays
                    }

                    # ----------------------------------------
                    # All-time arrays
                    # ----------------------------------------

                    values_ves = (
                        np.concatenate(
                            list(values_by_month_ves.values())
                        )
                        if values_by_month_ves
                        else np.array([])
                    )

                    values_fs = (
                        np.concatenate(
                            list(values_by_month_fs.values())
                        )
                        if values_by_month_fs
                        else np.array([])
                    )

                    # ----------------------------------------
                    # 1. ALL-TIME HISTOGRAM
                    # ----------------------------------------

                    histogram_file = (
                        detector_output
                        / (
                            f"chi2_hist_"
                            f"{short_name}_"
                            f"WP{working_point}_VES_FS.png"
                        )
                    )

                    plot_distribution_comparison(
                        values_ves,
                        values_fs,
                        detector,
                        working_point,
                        branch_name,
                        histogram_file,
                    )

                    # ----------------------------------------
                    # 2. MONTHLY HISTOGRAMS
                    # ----------------------------------------

                    monthly_histogram_file = (
                        detector_output
                        / (
                            f"chi2_monthly_hists_"
                            f"{short_name}_"
                            f"WP{working_point}_VES_FS.png"
                        )
                    )

                    plot_monthly_histograms(
                        values_by_month_ves,
                        values_by_month_fs,
                        detector,
                        working_point,
                        branch_name,
                        monthly_histogram_file,
                    )

                    # ----------------------------------------
                    # 3. HEATMAP
                    # ----------------------------------------

                    heatmap_file = (
                        detector_output
                        / (
                            f"chi2_heatmap_weekly_monthly_"
                            f"{short_name}_"
                            f"WP{working_point}_VES_FS.png"
                        )
                    )

                    plot_weekly_monthly_heatmap_comparison(
                        values_by_week_ves,
                        values_by_week_fs,
                        values_by_month_ves,
                        values_by_month_fs,
                        detector,
                        working_point,
                        branch_name,
                        heatmap_file,
                    )

                    # ----------------------------------------
                    # 4. WEEKLY QUANTILES
                    # ----------------------------------------

                    quantile_file = (
                        detector_output
                        / (
                            f"chi2_quantiles_weekly_"
                            f"{short_name}_"
                            f"WP{working_point}_VES_FS.png"
                        )
                    )

                    plot_weekly_quantiles_comparison(
                        values_by_week_ves,
                        values_by_week_fs,
                        detector,
                        working_point,
                        branch_name,
                        quantile_file,
                    )

        # ====================================================
        # PER-RUN PLOTS
        # ====================================================

        print()
        print(
            f"Creating per-run plots "
            f"for {detector}..."
        )

        for (
            short_name,
            branch_name,
        ) in CHI2_BRANCHES.items():

            # --------------------------------------------
            # Extract one branch from the run data.
            #
            # Result:
            #
            # branch_run_data["VES"][WP][run] = values
            # branch_run_data["FS"][WP][run] = values
            # --------------------------------------------

            branch_run_data = {
                "VES": {},
                "FS": {},
            }

            for data_type in (
                "VES",
                "FS",
            ):
                for wp in run_data.get(
                    data_type,
                    {},
                ):

                    for (
                        run,
                        branch_dict,
                    ) in run_data[
                        data_type
                    ][wp].items():

                        if (
                            short_name
                            not in branch_dict
                        ):
                            continue

                        if (
                            wp
                            not in branch_run_data[
                                data_type
                            ]
                        ):
                            branch_run_data[
                                data_type
                            ][wp] = {}

                        branch_run_data[
                            data_type
                        ][wp][run] = (
                            branch_dict[
                                short_name
                            ]
                        )

            if (
                not branch_run_data["VES"]
                and not branch_run_data["FS"]
            ):
                continue

            # ============================================
            # NEW: MEDIAN PER RUN, ONE PLOT PER PLANE,
            #      COLORED BY (WP, DAC10)
            #
            # 3p tracks: Z1 Y1 Z2 Y2 Z3 Y3   (6 plots)
            # 4p tracks: Z1 Y1 ... Z4 Y4     (8 plots)
            # ============================================

            for plane in planes_for_branch(short_name):

                dac_file = (
                    detector_output
                    / (
                        f"chi2_median_per_run_"
                        f"{short_name}_{plane}_DAC10_VES_FS.png"
                    )
                )

                plot_median_per_run_dac(
                    branch_run_data,
                    run_info,
                    detector,
                    branch_name,
                    plane,
                    dac_file,
                    args.period_axis,
                )

                print(
                    f"  {short_name} {plane} -> {dac_file}"
                )

            # ============================================
            # OLD per-run plots (kept, disabled by default)
            # ============================================

            if ENABLE_OLD_PLOTS:

                # 1. MEAN
                mean_file = (
                    detector_output
                    / (
                        f"chi2_mean_per_run_"
                        f"{short_name}_VES_FS.png"
                    )
                )

                plot_run_statistic(
                    branch_run_data,
                    run_info,
                    detector,
                    branch_name,
                    "mean",
                    mean_file,
                    args.period_axis,
                )

                # 2. MEDIAN (per WP only)
                median_file = (
                    detector_output
                    / (
                        f"chi2_median_per_run_"
                        f"{short_name}_VES_FS.png"
                    )
                )

                plot_run_statistic(
                    branch_run_data,
                    run_info,
                    detector,
                    branch_name,
                    "median",
                    median_file,
                    args.period_axis,
                )

                # 3. MAXIMUM (y-axis automatically expanded)
                max_file = (
                    detector_output
                    / (
                        f"chi2_max_per_run_"
                        f"{short_name}_VES_FS.png"
                    )
                )

                plot_run_statistic(
                    branch_run_data,
                    run_info,
                    detector,
                    branch_name,
                    "max",
                    max_file,
                    args.period_axis,
                )

                # 4. MEDIAN + 25-75 PERCENTILE
                percentile_25_75_file = (
                    detector_output
                    / (
                        f"chi2_median_q25_q75_per_run_"
                        f"{short_name}_VES_FS.png"
                    )
                )

                plot_run_quantiles(
                    branch_run_data,
                    run_info,
                    detector,
                    branch_name,
                    percentile_25_75_file,
                    "q25",
                    "q75",
                    "25-75 percentile",
                    args.period_axis,
                )

                # 5. MEDIAN + 5-95 PERCENTILE
                percentile_05_95_file = (
                    detector_output
                    / (
                        f"chi2_median_q05_q95_per_run_"
                        f"{short_name}_VES_FS.png"
                    )
                )

                plot_run_quantiles(
                    branch_run_data,
                    run_info,
                    detector,
                    branch_name,
                    percentile_05_95_file,
                    "q05",
                    "q95",
                    "5-95 percentile",
                    args.period_axis,
                )

                # 6. FULL QUANTILES (one plot per WP)
                quantiles_full_file = (
                    detector_output
                    / (
                        f"chi2_quantiles_per_run_"
                        f"{short_name}_VES_FS.png"
                    )
                )

                plot_run_quantiles_full(
                    branch_run_data,
                    run_info,
                    detector,
                    branch_name,
                    quantiles_full_file,
                    args.period_axis,
                )

                print(
                    f"  {short_name}:"
                )

                print(
                    f"      mean     -> {mean_file}"
                )

                print(
                    f"      median   -> {median_file}"
                )

                print(
                    f"      max      -> {max_file}"
                )

                print(
                    f"      q25-q75  -> "
                    f"{percentile_25_75_file}"
                )

                print(
                    f"      q05-q95  -> "
                    f"{percentile_05_95_file}"
                )

                print(
                    f"      quantiles-> "
                    f"{quantiles_full_file}"
                )

        print()
        print(
            f"Output directory: "
            f"{detector_output}"
        )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()