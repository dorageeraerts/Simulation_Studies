#!/usr/bin/env python3

"""
MURAVES run overview: which runs (VES / FS) were taken in which period,
at which Working Point, for which detector.

Only the slow-control files are read (no ROOT files are opened).

Outputs (./run_periods/):

    run_periods_timeline.png

        x = date, one panel per detector,
        y = Working Point.

        Thick bars = blocks of consecutive runs with the same
        data type and WP. Blocks are labelled with:

            run range
            number of runs

        Dots = individual runs.

        Blue = VES, orange = FS.

    run_periods_weekly.png

        Weekly binned view.

    run_periods_monthly.png

        Monthly binned view.

    run_periods_daily.png

        Daily binned view, if --granularity day is selected.

        For all binned plots:

          - all detectors use the same x-axis scale
          - all detectors use the same WP/y-axis scale
          - bottom x-axis = week/month/day
          - EACH detector has its own top x-axis = run number
          - run-number labels are derived only from that detector
          - cells are split if both VES and FS occur at the same WP
          - number of runs is written inside each cell

    run_table.csv

        one row per run.

    run_blocks.csv

        one row per block.

A short summary is also printed to the terminal.
"""

from pathlib import Path

import argparse
import csv
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


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

OUTPUT_DIR = Path("./run_periods")

FILENAME_TEMPLATE = "MURAVES_AnalyzedData_run{run}.root"
SLOWCONTROL_TEMPLATE = "SLOWCONTROL_run{run}"


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


COLORS = {
    "VES": "tab:blue",
    "FS": "tab:orange",
}

VERBOSE = False


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
        None
        if values is None
        else [parse_run_range(v) for v in values]
    )


def iter_run_numbers(run_ranges):
    for start, stop in run_ranges:
        yield from range(start, stop + 1)


def warn(message):
    if VERBOSE:
        print(f"  [warn] {message}")


# ============================================================
# SLOW CONTROL
# ============================================================

def parse_slowcontrol_timestamp(fields):
    """
    Parse the timestamp stored in slow-control column/index 1.
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
                return datetime.strptime(candidate, fmt)
            except ValueError:
                pass

    # Also support numeric Unix timestamp.
    try:
        value = float(raw)

        if np.isfinite(value):

            if abs(value) >= 1e11:
                value /= 1000.0

            return datetime.fromtimestamp(value)

    except (ValueError, OverflowError, OSError):
        pass

    return None


def get_slowcontrol_info(slowcontrol_dir, detector, run):
    """
    Return (timestamp, working_point) from the first valid
    slow-control line, or None.
    """

    slowcontrol_file = (
        Path(slowcontrol_dir)
        / detector
        / SLOWCONTROL_TEMPLATE.format(run=run)
    )

    if not slowcontrol_file.is_file():
        warn(
            f"run {run}: slow-control file not found"
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
                    warn(
                        f"run {run}: malformed slow-control line"
                    )
                    return None

                timestamp = parse_slowcontrol_timestamp(
                    fields
                )

                if timestamp is None:
                    warn(
                        f"run {run}: could not parse timestamp "
                        f"'{fields[1]}'"
                    )
                    return None

                try:
                    working_point = float(
                        fields[5]
                    )

                except ValueError:
                    warn(
                        f"run {run}: invalid Working Point "
                        f"'{fields[5]}'"
                    )
                    return None

                if (
                    not np.isfinite(working_point)
                    or not working_point.is_integer()
                ):
                    warn(
                        f"run {run}: bad Working Point "
                        f"{working_point}"
                    )
                    return None

                return (
                    timestamp,
                    int(working_point),
                )

    except Exception as exc:
        warn(
            f"run {run}: could not read slow-control: {exc}"
        )
        return None

    warn(
        f"run {run}: no valid lines in slow-control file"
    )

    return None


def week_key(timestamp):
    """ISO calendar week, e.g. 2024-W02."""

    iso_year, iso_week, _ = (
        timestamp.isocalendar()
    )

    return f"{iso_year}-W{iso_week:02d}"


# ============================================================
# COLLECT RUNS / BUILD BLOCKS
# ============================================================

def collect_runs(detector, run_ranges):
    """
    Return a list of dictionaries, one per run with valid
    slow-control information.
    """

    data_dir = Path(
        DATA_DIR_TEMPLATE.format(
            detector=detector
        )
    )

    rows = []

    for data_type in ("VES", "FS"):

        n_requested = 0
        n_ok = 0

        for run in iter_run_numbers(
            run_ranges[data_type]
        ):

            n_requested += 1

            info = get_slowcontrol_info(
                SLOWCONTROL_DIR,
                detector,
                run,
            )

            if info is None:
                continue

            timestamp, wp = info

            root_exists = (
                data_dir
                / FILENAME_TEMPLATE.format(
                    run=run
                )
            ).is_file()

            rows.append(
                {
                    "detector": detector,
                    "data_type": data_type,
                    "run": run,
                    "timestamp": timestamp,
                    "wp": wp,
                    "root_exists": root_exists,
                }
            )

            n_ok += 1

        print(
            f"  {detector} {data_type}: "
            f"{n_ok}/{n_requested} runs "
            "with valid slow-control info"
        )

    rows.sort(
        key=lambda r: r["run"]
    )

    return rows


def build_blocks(rows):
    """
    Group runs into blocks of consecutive runs with the same
    data type and Working Point.
    """

    blocks = []

    for row in rows:

        last = blocks[-1] if blocks else None

        if (
            last is not None
            and last["data_type"] == row["data_type"]
            and last["wp"] == row["wp"]
        ):

            last["run_last"] = row["run"]
            last["n_runs"] += 1

            last["t_first"] = min(
                last["t_first"],
                row["timestamp"],
            )

            last["t_last"] = max(
                last["t_last"],
                row["timestamp"],
            )

        else:

            blocks.append(
                {
                    "detector": row["detector"],
                    "data_type": row["data_type"],
                    "wp": row["wp"],
                    "run_first": row["run"],
                    "run_last": row["run"],
                    "t_first": row["timestamp"],
                    "t_last": row["timestamp"],
                    "n_runs": 1,
                }
            )

    return blocks


# ============================================================
# OUTPUT TABLES
# ============================================================

def write_run_table(
    rows_by_detector,
    output_file,
):

    with open(
        output_file,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "detector",
                "data_type",
                "run",
                "timestamp",
                "iso_week",
                "working_point",
                "root_file_exists",
            ]
        )

        for rows in rows_by_detector.values():

            for r in rows:

                writer.writerow(
                    [
                        r["detector"],
                        r["data_type"],
                        r["run"],
                        r["timestamp"].strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        week_key(
                            r["timestamp"]
                        ),
                        r["wp"],
                        int(
                            r["root_exists"]
                        ),
                    ]
                )


def write_block_table(
    blocks_by_detector,
    output_file,
):

    with open(
        output_file,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "detector",
                "data_type",
                "working_point",
                "run_first",
                "run_last",
                "n_runs",
                "time_first",
                "time_last",
            ]
        )

        for blocks in blocks_by_detector.values():

            for b in blocks:

                writer.writerow(
                    [
                        b["detector"],
                        b["data_type"],
                        b["wp"],
                        b["run_first"],
                        b["run_last"],
                        b["n_runs"],
                        b["t_first"].strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        b["t_last"].strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                    ]
                )


def print_summary(blocks_by_detector):

    print()
    print("=" * 70)
    print(
        "SUMMARY "
        "(blocks of consecutive runs, same type and WP)"
    )
    print("=" * 70)

    for detector, blocks in blocks_by_detector.items():

        print()
        print(detector)

        for b in blocks:

            print(
                f"  {b['data_type']:3s}  "
                f"WP{b['wp']:<3d} "
                f"runs {b['run_first']}-{b['run_last']} "
                f"({b['n_runs']:4d} runs)  "
                f"{b['t_first']:%d-%m-%Y} -> "
                f"{b['t_last']:%d-%m-%Y}"
            )


# ============================================================
# PLOTTING
# ============================================================

def plot_overview(
    rows_by_detector,
    blocks_by_detector,
    x_axis,
    output_file,
    show_labels,
):
    """
    x_axis = "time" or "run".

    One panel per detector, y = Working Point.

    All detectors use the same y-axis scale.
    """

    detectors = list(
        rows_by_detector.keys()
    )

    # --------------------------------------------------------
    # Global WP scale
    # --------------------------------------------------------

    all_wps = sorted(
        {
            r["wp"]
            for rows in rows_by_detector.values()
            for r in rows
        }
    )

    if not all_wps:
        return

    ypos = {
        wp: i
        for i, wp in enumerate(all_wps)
    }

    fig, axes = plt.subplots(
        nrows=len(detectors),
        ncols=1,
        figsize=(
            15,
            3.6 * len(detectors) + 1.2,
        ),
        squeeze=False,
        sharey=True,
    )

    axes = axes.ravel()

    for ax, detector in zip(
        axes,
        detectors,
    ):

        rows = rows_by_detector[
            detector
        ]

        blocks = blocks_by_detector[
            detector
        ]

        if not rows:

            ax.text(
                0.5,
                0.5,
                "no runs found",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

            ax.set_title(
                detector
            )

            continue

        def xval(timestamp, run):
            return (
                timestamp
                if x_axis == "time"
                else run
            )

        # ----------------------------------------------------
        # Blocks
        # ----------------------------------------------------

        for b in blocks:

            y = ypos[b["wp"]]

            x0 = xval(
                b["t_first"],
                b["run_first"],
            )

            x1 = xval(
                b["t_last"],
                b["run_last"],
            )

            ax.hlines(
                y,
                x0,
                x1,
                color=COLORS[
                    b["data_type"]
                ],
                linewidth=9,
                alpha=0.45,
            )

            if show_labels:

                if (
                    b["run_first"]
                    == b["run_last"]
                ):
                    run_label = (
                        f"{b['run_first']}"
                    )

                else:
                    run_label = (
                        f"{b['run_first']}-"
                        f"{b['run_last']}"
                    )

                label = (
                    f"{run_label}\n"
                    f"({b['n_runs']} runs)"
                )

                xm = x0 + (x1 - x0) / 2

                ax.annotate(
                    label,
                    (xm, y),
                    xytext=(0, 9),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )

        # ----------------------------------------------------
        # Individual runs
        # ----------------------------------------------------

        for data_type in ("VES", "FS"):

            sel = [
                r
                for r in rows
                if r["data_type"]
                == data_type
            ]

            if not sel:
                continue

            ax.scatter(
                [
                    xval(
                        r["timestamp"],
                        r["run"],
                    )
                    for r in sel
                ],
                [
                    ypos[r["wp"]]
                    for r in sel
                ],
                s=5,
                color=COLORS[
                    data_type
                ],
                zorder=3,
            )

        # ----------------------------------------------------
        # Missing ROOT files
        # ----------------------------------------------------

        missing = [
            r
            for r in rows
            if not r["root_exists"]
        ]

        if missing:

            ax.scatter(
                [
                    xval(
                        r["timestamp"],
                        r["run"],
                    )
                    for r in missing
                ],
                [
                    ypos[r["wp"]]
                    for r in missing
                ],
                s=14,
                marker="x",
                color="black",
                linewidths=0.6,
                zorder=4,
            )

        ax.set_yticks(
            list(ypos.values())
        )

        ax.set_yticklabels(
            [
                f"WP{wp}"
                for wp in all_wps
            ]
        )

        ax.set_ylim(
            -0.7,
            len(all_wps) - 0.3 + 0.4,
        )

        ax.set_ylabel(
            "Working Point"
        )

        ax.set_title(
            detector
        )

        ax.grid(
            True,
            alpha=0.3,
            linestyle="--",
        )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS["VES"],
            lw=6,
            alpha=0.6,
            label="VES",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["FS"],
            lw=6,
            alpha=0.6,
            label="FS",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="black",
            lw=0,
            markersize=5,
            label="no ROOT file",
        ),
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.995),
    )

    fig.suptitle(
        "MURAVES run periods: VES / FS and Working Point",
        y=1.03,
    )

    fig.tight_layout()

    fig.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# BINNING HELPERS
# ============================================================

def period_start(
    timestamp,
    granularity,
):
    """
    First day of the period containing timestamp.

    week  -> Monday
    month -> first day of month
    day   -> same day
    """

    day = timestamp.date()

    if granularity == "week":

        day = day - timedelta(
            days=day.weekday()
        )

    elif granularity == "month":

        day = day.replace(
            day=1
        )

    return day


def next_period(
    day,
    granularity,
):

    if granularity == "day":

        return day + timedelta(
            days=1
        )

    if granularity == "week":

        return day + timedelta(
            days=7
        )

    if granularity == "month":

        if day.month == 12:

            return day.replace(
                year=day.year + 1,
                month=1,
                day=1,
            )

        return day.replace(
            month=day.month + 1,
            day=1,
        )

    raise ValueError(
        f"Unknown granularity: {granularity}"
    )


def period_label(
    day,
    granularity,
):

    if granularity == "week":

        iso_year, iso_week, _ = (
            day.isocalendar()
        )

        return (
            f"{iso_year}-W{iso_week:02d}"
        )

    if granularity == "month":

        return day.strftime(
            "%Y-%m"
        )

    return day.strftime(
        "%d-%m-%Y"
    )


def build_periods(
    first,
    last,
    granularity,
):

    periods = []

    current = first

    while current <= last:

        periods.append(
            current
        )

        current = next_period(
            current,
            granularity,
        )

    return periods


# ============================================================
# RUN NUMBER AXIS
# ============================================================

def get_detector_run_number_labels(
    rows,
    periods,
    period_index,
    granularity,
):
    """
    Determine the run number shown above each time bin
    for ONE detector.

    The first run chronologically occurring in each time bin
    is used as the representative run number.

    This is deliberately calculated separately for every
    detector.
    """

    first_run_by_period = {}

    for r in rows:

        p = period_start(
            r["timestamp"],
            granularity,
        )

        run = r["run"]

        if (
            p not in first_run_by_period
            or run
            < first_run_by_period[p]
        ):
            first_run_by_period[p] = run

    period_indices = []
    run_labels = []

    for p in periods:

        if p not in first_run_by_period:
            continue

        period_indices.append(
            period_index[p]
        )

        run_labels.append(
            str(
                first_run_by_period[p]
            )
        )

    return (
        period_indices,
        run_labels,
    )


def add_detector_run_axis(
    ax,
    rows,
    periods,
    period_index,
    granularity,
):
    """
    Add a separate top x-axis to ONE detector panel.

    The axis is linked to the same horizontal coordinate system
    as the detector's bottom time axis, but the run-number labels
    come only from this detector.

    Returns the new twinned axis.
    """

    run_ax = ax.twiny()

    # Keep exactly the same x limits as this detector panel.
    run_ax.set_xlim(
        ax.get_xlim()
    )

    (
        run_period_indices,
        run_labels,
    ) = get_detector_run_number_labels(
        rows,
        periods,
        period_index,
        granularity,
    )

    # Avoid overcrowding the top axis.
    max_run_labels = 25

    if (
        len(run_period_indices)
        > max_run_labels
    ):

        run_step = int(
            np.ceil(
                len(run_period_indices)
                / max_run_labels
            )
        )

        run_period_indices = (
            run_period_indices[
                ::run_step
            ]
        )

        run_labels = (
            run_labels[
                ::run_step
            ]
        )

    if run_period_indices:

        run_ax.set_xticks(
            np.array(
                run_period_indices
            ) + 0.5
        )

        run_ax.set_xticklabels(
            run_labels,
            fontsize=8,
        )

    else:

        run_ax.set_xticks([])

    # Requested label: simply "Run number".
    run_ax.set_xlabel(
        "Run number"
    )

    run_ax.tick_params(
        axis="x",
        which="both",
        pad=3,
    )

    return run_ax


# ============================================================
# BINNED VIEW
# ============================================================

def plot_binned(
    rows_by_detector,
    granularity,
    output_file,
):
    """
    Binned view.

    x = week / month / day
    y = Working Point

    Important:

      * identical x-axis scale for all detectors
      * identical WP/y-axis scale for all detectors
      * bottom axis = period
      * EACH detector has its own top axis = run number
      * run-number labels are calculated independently
        for each detector
      * each cell contains the number of runs
    """

    detectors = list(
        rows_by_detector.keys()
    )

    # --------------------------------------------------------
    # Global Working Point scale
    # --------------------------------------------------------

    all_wps = sorted(
        {
            r["wp"]
            for rows in rows_by_detector.values()
            for r in rows
        }
    )

    if not all_wps:
        return

    ypos = {
        wp: i
        for i, wp in enumerate(all_wps)
    }

    # --------------------------------------------------------
    # Build all periods globally.
    #
    # This makes the x-axis identical for all detectors.
    # --------------------------------------------------------

    all_period_starts = []

    for rows in rows_by_detector.values():

        for r in rows:

            all_period_starts.append(
                period_start(
                    r["timestamp"],
                    granularity,
                )
            )

    if not all_period_starts:
        return

    global_first = min(
        all_period_starts
    )

    global_last = max(
        all_period_starts
    )

    periods = build_periods(
        global_first,
        global_last,
        granularity,
    )

    period_index = {
        day: i
        for i, day in enumerate(periods)
    }

    n_periods = len(periods)

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        nrows=len(detectors),
        ncols=1,
        figsize=(
            15,
            4.1 * len(detectors) + 1.2,
        ),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    axes = axes.ravel()

    # Keep references to the individual run-number axes.
    run_axes = []

    # --------------------------------------------------------
    # Plot each detector
    # --------------------------------------------------------

    for ax, detector in zip(
        axes,
        detectors,
    ):

        rows = rows_by_detector[
            detector
        ]

        ax.set_title(
            detector
        )

        if not rows:

            ax.text(
                0.5,
                0.5,
                "no runs found",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

            # Still create an empty detector-specific
            # run-number axis so the layout remains consistent.
            run_ax = ax.twiny()
            run_ax.set_xlim(
                ax.get_xlim()
            )
            run_ax.set_xticks([])
            run_ax.set_xlabel(
                "Run number"
            )
            run_axes.append(run_ax)

            continue

        # ----------------------------------------------------
        # Count runs in each cell
        #
        # cells[(period, wp, type)] = n_runs
        # ----------------------------------------------------

        cells = {}

        for r in rows:

            p = period_start(
                r["timestamp"],
                granularity,
            )

            key = (
                p,
                r["wp"],
                r["data_type"],
            )

            cells[key] = (
                cells.get(key, 0) + 1
            )

        # ----------------------------------------------------
        # Determine whether VES and FS share a cell
        # ----------------------------------------------------

        types_in_cell = {}

        for (
            p,
            wp,
            data_type,
        ) in cells:

            types_in_cell.setdefault(
                (p, wp),
                set(),
            ).add(
                data_type
            )

        # ----------------------------------------------------
        # Draw cells
        # ----------------------------------------------------

        for (
            p,
            wp,
            data_type,
        ), n_runs in cells.items():

            x = period_index[p]
            y = ypos[wp]

            shared = (
                len(
                    types_in_cell[
                        (p, wp)
                    ]
                ) > 1
            )

            if shared:

                height = 0.4

                if data_type == "VES":
                    y0 = y - 0.4
                else:
                    y0 = y

            else:

                height = 0.8
                y0 = y - 0.4

            ax.add_patch(
                Rectangle(
                    (
                        x,
                        y0,
                    ),
                    1.0,
                    height,
                    facecolor=COLORS[
                        data_type
                    ],
                    edgecolor="none",
                    alpha=0.65,
                )
            )

            # Number of runs in the cell.
            fontsize = (
                7
                if granularity != "day"
                else 5
            )

            ax.text(
                x + 0.5,
                y0 + height / 2,
                str(n_runs),
                ha="center",
                va="center",
                fontsize=fontsize,
                color="black",
                zorder=5,
            )

        # ----------------------------------------------------
        # Axes
        # ----------------------------------------------------

        ax.set_xlim(
            0,
            n_periods,
        )

        ax.set_ylim(
            -0.6,
            len(all_wps) - 0.4,
        )

        ax.set_yticks(
            list(ypos.values())
        )

        ax.set_yticklabels(
            [
                f"WP{wp}"
                for wp in all_wps
            ]
        )

        ax.set_ylabel(
            "Working Point"
        )

        ax.grid(
            True,
            axis="y",
            alpha=0.3,
            linestyle="--",
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Add a separate run-number axis for THIS detector.
        # ----------------------------------------------------

        run_ax = add_detector_run_axis(
            ax=ax,
            rows=rows,
            periods=periods,
            period_index=period_index,
            granularity=granularity,
        )

        run_axes.append(
            run_ax
        )

    # ========================================================
    # BOTTOM X AXIS
    # ========================================================

    # The bottom axis is shared by all detector panels.
    #
    # Do not label every period if there are too many.

    max_bottom_labels = 30

    tick_step = max(
        1,
        int(
            np.ceil(
                n_periods
                / max_bottom_labels
            )
        ),
    )

    bottom_indices = np.arange(
        0,
        n_periods,
        tick_step,
    )

    bottom_labels = [
        period_label(
            periods[i],
            granularity,
        )
        for i in bottom_indices
    ]

    axes[-1].set_xticks(
        bottom_indices + 0.5
    )

    axes[-1].set_xticklabels(
        bottom_labels,
        rotation=60,
        ha="right",
        fontsize=8,
    )

    if granularity == "week":

        axes[-1].set_xlabel(
            "ISO week"
        )

    elif granularity == "month":

        axes[-1].set_xlabel(
            "Month"
        )

    else:

        axes[-1].set_xlabel(
            "Day"
        )

    # ========================================================
    # LEGEND
    # ========================================================

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS["VES"],
            lw=8,
            label="VES",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["FS"],
            lw=8,
            label="FS",
        ),
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.995),
    )

    # ========================================================
    # TITLE
    # ========================================================

    if granularity == "week":
        title_period = "ISO week"

    elif granularity == "month":
        title_period = "month"

    else:
        title_period = "day"

    fig.suptitle(
        f"MURAVES run periods per {title_period}",
        y=1.03,
    )

    # Leave enough vertical room for the individual
    # run-number axes above every detector panel.
    fig.tight_layout(
        rect=[0, 0, 1, 0.97]
    )

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
            "Plot which runs (VES/FS) were taken "
            "in which period, at which WP, "
            "for which detector."
        )
    )

    parser.add_argument(
        "--detectors",
        nargs="+",
        choices=sorted(
            RUN_RANGES.keys()
        ),
        default=sorted(
            RUN_RANGES.keys()
        ),
        help=(
            "Detectors to process "
            "(default: all)"
        ),
    )

    parser.add_argument(
        "--ves-ranges",
        nargs="+",
        metavar="START-END",
        help=(
            "Override default VES run ranges "
            "(all selected detectors)"
        ),
    )

    parser.add_argument(
        "--fs-ranges",
        nargs="+",
        metavar="START-END",
        help=(
            "Override default FS run ranges "
            "(all selected detectors)"
        ),
    )

    parser.add_argument(
        "--granularity",
        choices=[
            "week",
            "month",
            "day",
        ],
        default="week",
        help=(
            "Time bin of the selected binned plot: "
            "week (default), month or day"
        ),
    )

    parser.add_argument(
        "--no-labels",
        action="store_true",
        help=(
            "Do not annotate timeline blocks "
            "with run range and number of runs"
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Print a warning for every run "
            "that cannot be read"
        ),
    )

    args = parser.parse_args()

    return (
        args,
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

    global VERBOSE

    (
        args,
        ves_override,
        fs_override,
    ) = parse_arguments()

    VERBOSE = args.verbose

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("MURAVES RUN PERIODS")
    print("=" * 70)

    rows_by_detector = {}
    blocks_by_detector = {}

    # --------------------------------------------------------
    # Read all detectors
    # --------------------------------------------------------

    for detector in args.detectors:

        print()
        print(
            f"Reading slow-control: {detector}"
        )

        run_ranges = {
            "VES": (
                ves_override
                if ves_override is not None
                else RUN_RANGES[
                    detector
                ]["VES"]
            ),

            "FS": (
                fs_override
                if fs_override is not None
                else RUN_RANGES[
                    detector
                ]["FS"]
            ),
        }

        rows = collect_runs(
            detector,
            run_ranges,
        )

        rows_by_detector[
            detector
        ] = rows

        blocks_by_detector[
            detector
        ] = build_blocks(
            rows
        )

    # --------------------------------------------------------
    # Tables
    # --------------------------------------------------------

    print_summary(
        blocks_by_detector
    )

    run_table = (
        OUTPUT_DIR
        / "run_table.csv"
    )

    block_table = (
        OUTPUT_DIR
        / "run_blocks.csv"
    )

    write_run_table(
        rows_by_detector,
        run_table,
    )

    write_block_table(
        blocks_by_detector,
        block_table,
    )

    # --------------------------------------------------------
    # Timeline
    # --------------------------------------------------------

    timeline_file = (
        OUTPUT_DIR
        / "run_periods_timeline.png"
    )

    plot_overview(
        rows_by_detector,
        blocks_by_detector,
        "time",
        timeline_file,
        not args.no_labels,
    )

    # --------------------------------------------------------
    # Selected binned plot
    # --------------------------------------------------------

    if args.granularity == "week":

        binned_file = (
            OUTPUT_DIR
            / "run_periods_weekly.png"
        )

    elif args.granularity == "month":

        binned_file = (
            OUTPUT_DIR
            / "run_periods_monthly.png"
        )

    else:

        binned_file = (
            OUTPUT_DIR
            / "run_periods_daily.png"
        )

    plot_binned(
        rows_by_detector,
        args.granularity,
        binned_file,
    )

    # --------------------------------------------------------
    # Always create monthly plot as well.
    # --------------------------------------------------------

    monthly_file = (
        OUTPUT_DIR
        / "run_periods_monthly.png"
    )

    if args.granularity != "month":

        plot_binned(
            rows_by_detector,
            "month",
            monthly_file,
        )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print()

    print(
        f"Timeline       -> {timeline_file}"
    )

    print(
        f"Binned ({args.granularity}) "
        f"-> {binned_file}"
    )

    if args.granularity != "month":

        print(
            f"Monthly        -> {monthly_file}"
        )

    print(
        f"Run table      -> {run_table}"
    )

    print(
        f"Block table    -> {block_table}"
    )

    print()

    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()