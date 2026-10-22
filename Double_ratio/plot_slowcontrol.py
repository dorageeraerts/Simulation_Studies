#!/usr/bin/env python3

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# Configuration
# ============================================================

SLOWCONTROL_DIR = Path(
    "/pnfs/iihe/muraves/muraves_DATA/DECOMPRESSED"
)

DEFAULT_OUTPUT_DIR = Path("./slowcontrol_plots")


# SLOWCONTROL column indices
RUN_COL = 0
TIMESTAMP_COL = 1
TEMPERATURE_COL = 3
HUMIDITY_COL = 4
WP_COL = 5

TRIGGER_RATE_COL = 43
ACCIDENTAL_RATE_COL = 44

# Columns 0 ... 44 must exist
MIN_COLUMNS = 45

# SLOWCONTROL timestamp is in milliseconds
TIMESTAMP_DIVISOR = 1000.0


# Working points excluded from the PLOTS.
# They are still kept in slowcontrol_parsed.csv.
EXCLUDED_WORKING_POINTS = [6, 10]


# Smaller scatter points
SCATTER_SIZE = 8


# ============================================================
# Run classification
# ============================================================

VES_RANGES = [
    (4000, 10060),
    (13458, 14792),
]

FS_RANGES = [
    (10061, 13457),
]


def get_data_type(run):
    """
    Return VES, FS, or UNKNOWN depending on the run number.
    """

    for start, end in VES_RANGES:
        if start <= run <= end:
            return "VES"

    for start, end in FS_RANGES:
        if start <= run <= end:
            return "FS"

    return "UNKNOWN"


# ============================================================
# Parse SLOWCONTROL files
# ============================================================

def parse_slowcontrol_files(slowcontrol_dir):
    """
    Parse all SLOWCONTROL_run* files in the detector directory.
    """

    rows = []

    slowcontrol_files = sorted(
        slowcontrol_dir.glob("SLOWCONTROL_run*")
    )

    if not slowcontrol_files:
        raise RuntimeError(
            f"No SLOWCONTROL files found in {slowcontrol_dir}"
        )

    print(
        f"Found {len(slowcontrol_files)} SLOWCONTROL files "
        f"in {slowcontrol_dir}"
    )

    for file_index, filepath in enumerate(
        slowcontrol_files,
        start=1,
    ):

        if file_index % 1000 == 0:
            print(
                f"  Processing file "
                f"{file_index}/{len(slowcontrol_files)}..."
            )

        try:

            with filepath.open("r") as f:

                for line_number, line in enumerate(
                    f,
                    start=1,
                ):

                    line = line.strip()

                    if not line:
                        continue

                    # SLOWCONTROL files are whitespace-separated.
                    cols = line.split()

                    if len(cols) < MIN_COLUMNS:

                        print(
                            f"Warning: {filepath.name}, "
                            f"line {line_number}: "
                            f"only {len(cols)} columns, "
                            f"expected at least {MIN_COLUMNS}. "
                            f"Skipping."
                        )

                        continue

                    try:

                        run = int(
                            cols[RUN_COL]
                        )

                        timestamp = (
                            float(cols[TIMESTAMP_COL])
                            / TIMESTAMP_DIVISOR
                        )

                        temperature = float(
                            cols[TEMPERATURE_COL]
                        )

                        humidity = float(
                            cols[HUMIDITY_COL]
                        )

                        wp = float(
                            cols[WP_COL]
                        )

                        trigger_rate = float(
                            cols[TRIGGER_RATE_COL]
                        )

                        accidental_rate = float(
                            cols[ACCIDENTAL_RATE_COL]
                        )

                        dt = datetime.fromtimestamp(
                            timestamp
                        )

                    except (
                        ValueError,
                        OverflowError,
                        OSError,
                    ) as exc:

                        print(
                            f"Warning: could not parse "
                            f"{filepath.name}, "
                            f"line {line_number}: "
                            f"{exc}"
                        )

                        continue

                    rows.append(
                        {
                            "run": run,
                            "datetime": dt,
                            "temperature": temperature,
                            "humidity": humidity,
                            "wp": wp,
                            "trigger_rate": trigger_rate,
                            "accidental_rate": accidental_rate,
                            "type": get_data_type(run),
                        }
                    )

        except OSError as exc:

            print(
                f"Warning: could not read "
                f"{filepath}: {exc}"
            )

    df = pd.DataFrame(rows)

    if df.empty:

        raise RuntimeError(
            "No valid SLOWCONTROL data were parsed."
        )

    df = df.sort_values(
        ["datetime", "run"]
    ).reset_index(drop=True)

    return df


# ============================================================
# Load or parse cached data
# ============================================================

def load_or_parse_data(
    slowcontrol_dir,
    csv_path,
    reparse=False,
):
    """
    Load the existing parsed CSV if available.

    If --reparse is specified, ignore the existing CSV and
    parse all SLOWCONTROL files again.
    """

    if csv_path.exists() and not reparse:

        print()
        print("=" * 60)
        print("USING CACHED PARSED DATA")
        print("=" * 60)

        print(f"Loading: {csv_path}")

        print(
            "Existing slowcontrol_parsed.csv found."
        )

        print(
            "SLOWCONTROL files will NOT be reparsed."
        )

        df = pd.read_csv(
            csv_path,
            parse_dates=["datetime"],
        )

        if df.empty:

            raise RuntimeError(
                f"The existing CSV is empty: {csv_path}"
            )

        return df

    if reparse:

        print()
        print("=" * 60)
        print("FORCED REPARSE")
        print("=" * 60)

        print(
            "--reparse was specified."
        )

        print(
            "All SLOWCONTROL files will be parsed again."
        )

    else:

        print()
        print("=" * 60)
        print("NO CACHED CSV FOUND")
        print("=" * 60)

        print(
            "slowcontrol_parsed.csv does not exist."
        )

        print(
            "Parsing SLOWCONTROL files..."
        )

    df = parse_slowcontrol_files(
        slowcontrol_dir
    )

    # Save the complete parsed dataset.
    #
    # WP 6 and WP 10 are intentionally kept here.
    df.to_csv(
        csv_path,
        index=False,
    )

    print()
    print(
        f"Saved parsed data to:"
    )
    print(
        f"  {csv_path}"
    )

    return df


# ============================================================
# Print summary
# ============================================================

def print_summary(df):

    print()
    print("=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)

    print(
        f"Total data points: {len(df)}"
    )

    if "run" in df.columns:

        print(
            f"Run range: "
            f"{df['run'].min()} - "
            f"{df['run'].max()}"
        )

    print()

    print("Data types:")

    if "type" in df.columns:

        print(
            df["type"]
            .value_counts()
            .sort_index()
            .to_string()
        )

    print()

    print("Working points in cached CSV:")

    if "wp" in df.columns:

        print(
            sorted(
                df["wp"]
                .dropna()
                .unique()
                .tolist()
            )
        )

    print()

    print(
        "Working points excluded from plots: "
        f"{EXCLUDED_WORKING_POINTS}"
    )

    print()

    if "datetime" in df.columns:

        print(
            f"Time range: "
            f"{df['datetime'].min()} "
            f"to "
            f"{df['datetime'].max()}"
        )


# ============================================================
# Figure 1:
# Trigger rate vs DATE
#
# X axis = date
# Y axis = trigger rate
#
# Separate series:
#   VES + each WP
#   FS  + each WP
#
# No connecting lines.
# ============================================================

def plot_trigger_rate_vs_date(
    df,
    output_path,
):

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    for data_type, marker in [
        ("VES", "o"),
        ("FS", "s"),
    ]:

        type_df = df[
            df["type"] == data_type
        ]

        if type_df.empty:
            continue

        for wp in sorted(
            type_df["wp"]
            .dropna()
            .unique()
        ):

            subset = type_df[
                type_df["wp"] == wp
            ]

            ax.scatter(
                subset["datetime"],
                subset["trigger_rate"],
                marker=marker,
                s=SCATTER_SIZE,
                label=f"{data_type}, WP={wp:g}",
            )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Trigger rate"
    )

    ax.set_title(
        "Trigger Rate vs Date"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend(
        fontsize=8,
        loc="best",
    )

    fig.autofmt_xdate()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# Figure 2:
# Humidity vs DATE
#
# X axis = date
# Y axis = humidity
#
# Separate series:
#   VES + each WP
#   FS  + each WP
#
# No connecting lines.
# ============================================================

def plot_humidity_vs_date(
    df,
    output_path,
):

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    for data_type, marker in [
        ("VES", "o"),
        ("FS", "s"),
    ]:

        type_df = df[
            df["type"] == data_type
        ]

        if type_df.empty:
            continue

        for wp in sorted(
            type_df["wp"]
            .dropna()
            .unique()
        ):

            subset = type_df[
                type_df["wp"] == wp
            ]

            ax.scatter(
                subset["datetime"],
                subset["humidity"],
                marker=marker,
                s=SCATTER_SIZE,
                label=f"{data_type}, WP={wp:g}",
            )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Humidity (%)"
    )

    ax.set_title(
        "Humidity vs Date"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend(
        fontsize=8,
        loc="best",
    )

    fig.autofmt_xdate()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# Figure 3:
# Temperature vs DATE
#
# X axis = date
# Y axis = temperature
#
# Separate series:
#   VES + each WP
#   FS  + each WP
#
# No connecting lines.
# ============================================================

def plot_temperature_vs_date(
    df,
    output_path,
):

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    for data_type, marker in [
        ("VES", "o"),
        ("FS", "s"),
    ]:

        type_df = df[
            df["type"] == data_type
        ]

        if type_df.empty:
            continue

        for wp in sorted(
            type_df["wp"]
            .dropna()
            .unique()
        ):

            subset = type_df[
                type_df["wp"] == wp
            ]

            ax.scatter(
                subset["datetime"],
                subset["temperature"],
                marker=marker,
                s=SCATTER_SIZE,
                label=f"{data_type}, WP={wp:g}",
            )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Temperature"
    )

    ax.set_title(
        "Temperature vs Date"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend(
        fontsize=8,
        loc="best",
    )

    fig.autofmt_xdate()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# Figure 4:
# Temperature vs Working Point
#
# Individual measurements are scatter points.
# Mean +/- standard deviation are shown separately.
#
# WP 6 and WP 10 are excluded.
# ============================================================

def plot_temperature_vs_working_point(
    df,
    output_path,
):

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    for data_type, marker in [
        ("VES", "o"),
        ("FS", "s"),
    ]:

        type_df = df[
            df["type"] == data_type
        ]

        if type_df.empty:
            continue

        # ----------------------------------------------------
        # Individual measurements
        # ----------------------------------------------------

        ax.scatter(
            type_df["wp"],
            type_df["temperature"],
            marker=marker,
            s=SCATTER_SIZE,
            alpha=0.5,
            label=f"{data_type} measurements",
        )

        # ----------------------------------------------------
        # Mean +/- standard deviation
        # ----------------------------------------------------

        grouped = (
            type_df
            .groupby("wp")["temperature"]
            .agg(["mean", "std"])
            .reset_index()
        )

        grouped["std"] = grouped["std"].fillna(0)

        ax.errorbar(
            grouped["wp"],
            grouped["mean"],
            yerr=grouped["std"],
            marker=marker,
            linestyle="none",
            markersize=5,
            capsize=3,
            label=f"{data_type} mean +/- std",
        )

    ax.set_xlabel(
        "Working point"
    )

    ax.set_ylabel(
        "Temperature"
    )

    ax.set_title(
        "Temperature vs Working Point"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend(
        fontsize=8,
        loc="best",
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Plot MURAVES SLOWCONTROL data. "
            "Uses slowcontrol_parsed.csv as a cache "
            "unless --reparse is specified."
        )
    )

    parser.add_argument(
        "--detector",
        default="ROSSO",
        help="Detector name (default: ROSSO)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Base output directory "
            "(default: ./slowcontrol_plots)"
        ),
    )

    parser.add_argument(
        "--start-run",
        type=int,
        default=None,
        help="Only use runs >= START_RUN",
    )

    parser.add_argument(
        "--end-run",
        type=int,
        default=None,
        help="Only use runs <= END_RUN",
    )

    parser.add_argument(
        "--reparse",
        action="store_true",
        help=(
            "Force reparsing of all SLOWCONTROL files "
            "even if slowcontrol_parsed.csv already exists."
        ),
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    detector = args.detector

    slowcontrol_dir = (
        SLOWCONTROL_DIR / detector
    )

    detector_output = (
        args.output_dir / detector
    )

    detector_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        detector_output
        / "slowcontrol_parsed.csv"
    )

    # --------------------------------------------------------
    # Load cached CSV or parse SLOWCONTROL files
    # --------------------------------------------------------

    df = load_or_parse_data(
        slowcontrol_dir=slowcontrol_dir,
        csv_path=csv_path,
        reparse=args.reparse,
    )

    # --------------------------------------------------------
    # Ensure datetime is correctly interpreted
    # --------------------------------------------------------

    df["datetime"] = pd.to_datetime(
        df["datetime"]
    )

    # --------------------------------------------------------
    # Optional run filtering
    # --------------------------------------------------------

    if args.start_run is not None:

        df = df[
            df["run"] >= args.start_run
        ]

    if args.end_run is not None:

        df = df[
            df["run"] <= args.end_run
        ]

    # --------------------------------------------------------
    # Exclude WP 6 and WP 10 from the PLOTS
    #
    # The cached CSV remains unchanged.
    # --------------------------------------------------------

    df = df[
        ~df["wp"].isin(
            EXCLUDED_WORKING_POINTS
        )
    ].copy()

    if df.empty:

        raise RuntimeError(
            "No data remain after applying "
            "the run and working-point filters."
        )

    # --------------------------------------------------------
    # Sort by date
    # --------------------------------------------------------

    df = df.sort_values(
        ["datetime", "run"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(df)

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    trigger_plot = (
        detector_output
        / "trigger_rate_vs_date.png"
    )

    humidity_plot = (
        detector_output
        / "humidity_vs_date.png"
    )

    temperature_plot = (
        detector_output
        / "temperature_vs_date.png"
    )

    temperature_wp_plot = (
        detector_output
        / "temperature_vs_working_point.png"
    )

    # --------------------------------------------------------
    # Generate plots
    # --------------------------------------------------------

    plot_trigger_rate_vs_date(
        df,
        trigger_plot,
    )

    plot_humidity_vs_date(
        df,
        humidity_plot,
    )

    plot_temperature_vs_date(
        df,
        temperature_plot,
    )

    plot_temperature_vs_working_point(
        df,
        temperature_wp_plot,
    )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print(
        f"Plots saved in:"
    )

    print(
        f"  {detector_output}"
    )

    if args.show:

        print()
        print(
            "--show was specified. "
            "The plots are saved as PNG files above."
        )


if __name__ == "__main__":
    main()

