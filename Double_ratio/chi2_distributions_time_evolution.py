#!/usr/bin/env python3

"""
Plot MURAVES chi2 distributions per Working Point (WP),
comparing VES and FS.

Branches:
    BestTrack_3p_ChiSquare_xy
    BestTrack_3p_ChiSquare_xz
    BestTrack_4p_ChiSquare_xy
    BestTrack_4p_ChiSquare_xz

For each detector / WP / branch:

    1. All-time histogram:
       VES and FS overlaid on the same axes.

    2. Monthly 1D histograms:
       One subplot per month, with VES and FS overlaid.

    3. Weekly vs monthly heatmaps:
       Weekly and monthly distributions are compared
       side-by-side.
       VES and FS have separate axes.
       All four panels share the same color scale.

    4. Weekly quantiles:
       VES and FS overlaid on the same axes.

Time definition:
    The timestamp is read from the slow-control file.
    Months are calendar months: YYYY-MM.
    Weeks are ISO weeks: Monday-Sunday, YYYY-Www.

Output:
    ./chi2_distributions/<detector>/
"""


from pathlib import Path
import argparse
from datetime import datetime

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

OUTPUT_DIR = Path("./chi2_distributions/time_evolution")

TREE_NAME = "AnalyzedData"

FILENAME_TEMPLATE = (
    "MURAVES_AnalyzedData_run{run}.root"
)

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


# ------------------------------------------------------------
# Consistent colors
# ------------------------------------------------------------

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
            f"Invalid run range: {value}: "
            "end must be >= start."
        )

    return start, stop


def parse_range_list(values):
    return (
        None
        if values is None
        else [
            parse_run_range(value)
            for value in values
        ]
    )


def iter_run_numbers(run_ranges):

    for start, stop in run_ranges:

        yield from range(
            start,
            stop + 1,
        )


# ============================================================
# TIME / SLOW CONTROL
# ============================================================

def parse_slowcontrol_timestamp(fields):
    """
    Parse the timestamp stored in slow-control
    column/index 1.
    """

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

    # Also support numeric Unix timestamp.
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
    Return:

        timestamp, working_point

    from the first valid slow-control line.
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

                timestamp = (
                    parse_slowcontrol_timestamp(
                        fields
                    )
                )

                if timestamp is None:

                    print(
                        "  [warn] could not parse timestamp "
                        "from slow-control column/index 1: "
                        f"{fields[1]}"
                    )

                    return None

                try:

                    working_point = float(
                        fields[5]
                    )

                except ValueError:

                    print(
                        "  [warn] invalid Working Point "
                        f"in {slowcontrol_file}: "
                        f"{fields[5]}"
                    )

                    return None

                if not np.isfinite(
                    working_point
                ):

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

                return (
                    timestamp,
                    int(working_point),
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
    """
    Calendar month identifier.
    """

    return timestamp.strftime("%Y-%m")


def week_key(timestamp):
    """
    ISO calendar week.

    ISO weeks run Monday through Sunday.

    Example:
        2024-01-08 -> 2024-W02
    """

    iso_year, iso_week, _ = (
        timestamp.isocalendar()
    )

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
    """
    Keep ISO week labels as YYYY-Www.
    """

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

            missing = (
                set(branches)
                - set(tree.keys())
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


def get_valid_chi2(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    valid = (
        np.isfinite(values)
        & (values >= CHI2_MIN)
        & (values <= CHI2_MAX)
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
    """
    Overlay VES and FS normalized chi2 histograms
    using all available data.
    """

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
# MONTHLY 1D HISTOGRAMS
# ============================================================

def plot_monthly_histograms(
    values_by_month_ves,
    values_by_month_fs,
    detector,
    working_point,
    branch_name,
    output_file,
):
    """
    Plot one normalized 1D histogram per month.

    Each subplot contains:
        VES = blue
        FS  = orange

    The chi2 bins are identical between all panels.
    The y-axis is shared between all panels so that
    density amplitudes can be compared visually.
    """

    months = sorted(
        set(values_by_month_ves)
        | set(values_by_month_fs)
    )

    if not months:
        return

    n_months = len(months)

    # Approximately 3 panels per row.
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

    # --------------------------------------------------------
    # Determine common y-axis maximum.
    #
    # This is important: every month has the same vertical
    # scale, so density changes between months remain visible.
    # --------------------------------------------------------

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

    # Add some headroom.
    ymax *= 1.10

    # --------------------------------------------------------
    # Plot every month.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Remove unused axes.
    # --------------------------------------------------------

    for i in range(
        n_months,
        len(axes),
    ):

        fig.delaxes(
            axes[i]
        )

    # --------------------------------------------------------
    # Common labels.
    # --------------------------------------------------------

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
    """
    Convert distributions into a density matrix.

    Rows:
        time periods

    Columns:
        chi2 bins
    """

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

        matrix = np.asarray(
            matrix
        )

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
    """
    Compare weekly and monthly chi2 heatmaps.

    Layout:

        Weekly VES    Monthly VES    colorbar
        Weekly FS     Monthly FS     colorbar

    The four heatmaps share:
        - chi2 binning
        - y-axis range
        - density color scale
        - colorbar
    """

    # --------------------------------------------------------
    # Time periods
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Build matrices.
    # --------------------------------------------------------

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

    # Sanity check.
    if not np.allclose(
        bins_week,
        bins_month,
    ):
        raise RuntimeError(
            "Weekly and monthly heatmaps "
            "do not use the same chi2 bins."
        )

    bins = bins_week

    # --------------------------------------------------------
    # Shared color scale.
    # --------------------------------------------------------

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

        vmax = np.max(
            all_values
        )

    else:

        vmax = 1.0

    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    # --------------------------------------------------------
    # Figure.
    #
    #       Weekly          Monthly       Colorbar
    #
    #       VES             VES
    #
    #       FS              FS
    # --------------------------------------------------------

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

    # Dedicated colorbar axis.
    cbar_ax = fig.add_subplot(
        gs[:, 2]
    )

    # --------------------------------------------------------
    # Heatmap helper.
    # --------------------------------------------------------

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

        # pcolormesh expects:
        #   x       -> number of periods + 1
        #   y       -> number of chi2 bins + 1
        #   matrix  -> periods x chi2 bins
        mesh = ax.pcolormesh(
            x,
            bins,
            matrix.T,
            shading="auto",
            cmap="viridis",
            vmin=0,
            vmax=vmax,
        )

        ax.set_title(
            title
        )

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

    # --------------------------------------------------------
    # Weekly panels.
    # --------------------------------------------------------

    mesh_week_ves = plot_heatmap(
        ax_week_ves,
        weekly_ves,
        weeks,
        format_week_labels(weeks),
        "Weekly — VES",
        show_xlabel=False,
    )

    plot_heatmap(
        ax_week_fs,
        weekly_fs,
        weeks,
        format_week_labels(weeks),
        "Weekly — FS",
        show_xlabel=True,
    )

    # --------------------------------------------------------
    # Monthly panels.
    # --------------------------------------------------------

    plot_heatmap(
        ax_month_ves,
        monthly_ves,
        months,
        format_month_labels(months),
        "Monthly — VES",
        show_xlabel=False,
    )

    plot_heatmap(
        ax_month_fs,
        monthly_fs,
        months,
        format_month_labels(months),
        "Monthly — FS",
        show_xlabel=True,
    )

    # --------------------------------------------------------
    # Shared colorbar.
    #
    # Dedicated axis = physically outside all plots.
    # --------------------------------------------------------

    cbar = fig.colorbar(
        mesh_week_ves,
        cax=cbar_ax,
    )

    cbar.set_label(
        "Density"
    )

    # --------------------------------------------------------
    # Overall title.
    # --------------------------------------------------------

    fig.suptitle(
        f"{branch_name} - "
        f"{detector}, WP{working_point}\n",
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
# QUANTILES
# ============================================================

def calculate_quantiles(
    values_by_period,
    periods,
):
    """
    Calculate:

        q10
        q25
        q50
        q75
        q90

    for each time period.
    """

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


# ============================================================
# WEEKLY QUANTILES
# ============================================================

def plot_weekly_quantiles_comparison(
    values_by_week_ves,
    values_by_week_fs,
    detector,
    working_point,
    branch_name,
    output_file,
):
    """
    Plot weekly VES and FS quantile evolution.

    10-90 percentile:
        light transparent band

    25-75 percentile:
        darker transparent band

    Median:
        solid line
    """

    weeks = sorted(
        set(values_by_week_ves)
        | set(values_by_week_fs)
    )

    if not weeks:
        return

    x = np.arange(
        len(weeks)
    )

    q_ves = calculate_quantiles(
        values_by_week_ves,
        weeks,
    )

    q_fs = calculate_quantiles(
        values_by_week_fs,
        weeks,
    )

    (
        q10_ves,
        q25_ves,
        q50_ves,
        q75_ves,
        q90_ves,
    ) = q_ves

    (
        q10_fs,
        q25_fs,
        q50_fs,
        q75_fs,
        q90_fs,
    ) = q_fs

    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    # --------------------------------------------------------
    # VES
    # --------------------------------------------------------

    ax.fill_between(
        x,
        q10_ves,
        q90_ves,
        color=VES_COLOR,
        alpha=0.12,
        label="VES 10–90 percentile",
    )

    ax.fill_between(
        x,
        q25_ves,
        q75_ves,
        color=VES_COLOR,
        alpha=0.25,
        label="VES 25–75 percentile",
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

    # --------------------------------------------------------
    # FS
    # --------------------------------------------------------

    ax.fill_between(
        x,
        q10_fs,
        q90_fs,
        color=FS_COLOR,
        alpha=0.12,
        label="FS 10–90 percentile",
    )

    ax.fill_between(
        x,
        q25_fs,
        q75_fs,
        color=FS_COLOR,
        alpha=0.25,
        label="FS 25–75 percentile",
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

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

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
        f"{detector}, WP{working_point}\n",
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
# ARGUMENTS
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Plot MURAVES chi2 distributions "
            "per WP, comparing VES and FS."
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
            f"Maximum chi2 shown, "
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

    args = parser.parse_args()

    if args.detector and args.detectors:

        parser.error(
            "Use either --detector or "
            "--detectors, not both."
        )

    detectors = (
        [args.detector]
        if args.detector
        else args.detectors
        if args.detectors
        else ["ROSSO"]
    )

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

    (
        args,
        detectors,
        ves_override,
        fs_override,
    ) = parse_arguments()

    CHI2_MAX = args.chi2_max
    BIN_WIDTH = args.bin_width

    print("=" * 70)
    print("MURAVES CHI2 DISTRIBUTIONS")
    print("=" * 70)

    print(
        "Detectors : "
        f"{', '.join(detectors)}"
    )

    print(
        "Chi2 range: "
        f"{CHI2_MIN} -> {CHI2_MAX}"
    )

    print(
        "Bin width : "
        f"{BIN_WIDTH}"
    )

    print(
        "Time source: slow-control timestamp"
    )

    print(
        "Week      : ISO week (Monday-Sunday)"
    )

    print(
        "VES and FS are compared on common plots."
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

        # ----------------------------------------------------
        # Data structure:
        #
        # data[data_type][WP][branch]["monthly"][month]
        # data[data_type][WP][branch]["weekly"][week]
        #
        # Each time-period entry contains a list of numpy
        # arrays, one array per run.
        # ----------------------------------------------------

        data = {
            "VES": {},
            "FS": {},
        }

        # ====================================================
        # READ DATA
        # ====================================================

        for data_type in [
            "VES",
            "FS",
        ]:

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

                # ------------------------------------------------
                # Slow-control timestamp and WP.
                # ------------------------------------------------

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

                timestamp, working_point = (
                    slowcontrol_info
                )

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
                    f"  Working Point: "
                    f"{working_point}"
                )

                # ------------------------------------------------
                # Read ROOT file.
                # ------------------------------------------------

                arrays = read_run(
                    file_path
                )

                if arrays is None:
                    continue

                # ------------------------------------------------
                # Create WP/branch containers.
                # ------------------------------------------------

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

                # ------------------------------------------------
                # Process branches.
                # ------------------------------------------------

                for (
                    short_name,
                    branch_name,
                ) in CHI2_BRANCHES.items():

                    values = get_valid_chi2(
                        arrays[branch_name]
                    )

                    if len(values) > 0:

                        period_data = (
                            data[data_type]
                            [working_point]
                            [short_name]
                        )

                        monthly = (
                            period_data[
                                "monthly"
                            ]
                        )

                        weekly = (
                            period_data[
                                "weekly"
                            ]
                        )

                        monthly.setdefault(
                            month,
                            []
                        ).append(
                            values
                        )

                        weekly.setdefault(
                            week,
                            []
                        ).append(
                            values
                        )

                    print(
                        f"    {short_name}: "
                        f"{len(values):,} entries"
                    )

        # ====================================================
        # CREATE COMPARISON PLOTS
        # ====================================================

        print()
        print(
            f"Creating comparison plots "
            f"for {detector}..."
        )

        # Union of WPs from VES and FS.
        working_points = sorted(
            set(data["VES"])
            | set(data["FS"])
        )

        for working_point in working_points:

            for (
                short_name,
                branch_name,
            ) in CHI2_BRANCHES.items():

                # ------------------------------------------------
                # Retrieve period dictionaries.
                # ------------------------------------------------

                monthly_arrays_ves = (
                    data["VES"]
                    .get(
                        working_point,
                        {},
                    )
                    .get(
                        short_name,
                        {},
                    )
                    .get(
                        "monthly",
                        {},
                    )
                )

                monthly_arrays_fs = (
                    data["FS"]
                    .get(
                        working_point,
                        {},
                    )
                    .get(
                        short_name,
                        {},
                    )
                    .get(
                        "monthly",
                        {},
                    )
                )

                weekly_arrays_ves = (
                    data["VES"]
                    .get(
                        working_point,
                        {},
                    )
                    .get(
                        short_name,
                        {},
                    )
                    .get(
                        "weekly",
                        {},
                    )
                )

                weekly_arrays_fs = (
                    data["FS"]
                    .get(
                        working_point,
                        {},
                    )
                    .get(
                        short_name,
                        {},
                    )
                    .get(
                        "weekly",
                        {},
                    )
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

                # =================================================
                # COMBINE RUNS WITHIN EACH MONTH
                # =================================================

                values_by_month_ves = {
                    month: np.concatenate(
                        arrays
                    )
                    for (
                        month,
                        arrays
                    ) in monthly_arrays_ves.items()
                    if arrays
                }

                values_by_month_fs = {
                    month: np.concatenate(
                        arrays
                    )
                    for (
                        month,
                        arrays
                    ) in monthly_arrays_fs.items()
                    if arrays
                }

                # =================================================
                # COMBINE RUNS WITHIN EACH WEEK
                # =================================================

                values_by_week_ves = {
                    week: np.concatenate(
                        arrays
                    )
                    for (
                        week,
                        arrays
                    ) in weekly_arrays_ves.items()
                    if arrays
                }

                values_by_week_fs = {
                    week: np.concatenate(
                        arrays
                    )
                    for (
                        week,
                        arrays
                    ) in weekly_arrays_fs.items()
                    if arrays
                }

                # =================================================
                # ALL-TIME ARRAYS
                # =================================================

                values_ves = (
                    np.concatenate(
                        list(
                            values_by_month_ves.values()
                        )
                    )
                    if values_by_month_ves
                    else np.array([])
                )

                values_fs = (
                    np.concatenate(
                        list(
                            values_by_month_fs.values()
                        )
                    )
                    if values_by_month_fs
                    else np.array([])
                )

                # =================================================
                # 1. ALL-TIME HISTOGRAM
                # =================================================

                histogram_file = (
                    detector_output
                    / (
                        f"chi2_hist_{short_name}_"
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

                # =================================================
                # 2. MONTHLY 1D HISTOGRAMS
                # =================================================

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

                # =================================================
                # 3. WEEKLY VS MONTHLY HEATMAP
                # =================================================

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

                # =================================================
                # 4. WEEKLY QUANTILES
                # =================================================

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

                # =================================================
                # REPORT
                # =================================================

                print(
                    f"  WP{working_point} "
                    f"{short_name}:"
                )

                print(
                    f"      VES: "
                    f"{len(values_ves):,} entries"
                )

                print(
                    f"      FS : "
                    f"{len(values_fs):,} entries"
                )

                print(
                    f"      weeks VES: "
                    f"{len(values_by_week_ves)}"
                )

                print(
                    f"      weeks FS : "
                    f"{len(values_by_week_fs)}"
                )

                print(
                    f"      months VES: "
                    f"{len(values_by_month_ves)}"
                )

                print(
                    f"      months FS : "
                    f"{len(values_by_month_fs)}"
                )

                print(
                    f"      histogram -> "
                    f"{histogram_file}"
                )

                print(
                    f"      monthly hists -> "
                    f"{monthly_histogram_file}"
                )

                print(
                    f"      heatmap -> "
                    f"{heatmap_file}"
                )

                print(
                    f"      weekly quantiles -> "
                    f"{quantile_file}"
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

