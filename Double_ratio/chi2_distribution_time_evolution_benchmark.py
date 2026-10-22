#!/usr/bin/env python3

"""
MURAVES chi2 monthly distributions: 3D views + VES benchmark comparison.

Branches:
    BestTrack_3p_ChiSquare_xy / _xz
    BestTrack_4p_ChiSquare_xy / _xz

For each detector (ROSSO, NERO, BLU) and each branch, two "selections":

    per_wp : one set of plots per Working Point
    all_wp : all Working Points merged together

and for each selection two figures:

    1. chi2_3d_*        VES | FS side by side (separate 3D axes, same z scale).
                        x = month (MM-YYYY), y = chi2, z = density.
                        One chi2 distribution (slice) per month.

    2. chi2_benchmark_* VES and FS are never taken in the same period, so FS
                        is compared month by month with the VES distribution
                        integrated over the FULL VES period (benchmark).

                        top    : 3D FS monthly slices, VES benchmark curve
                                 drawn inside every slice
                        middle : difference  FS(month) - VES(benchmark)
                                 (heatmap, month x chi2), or the pull
                                 (diff / stat. error) with --diff-mode pull
                        bottom : KS distance FS(month) vs benchmark, with the
                                 95% statistical-only threshold, and the
                                 range of VES(month) vs VES(benchmark) as a
                                 "natural fluctuation" control band.

Time definition:
    The timestamp is read from the slow-control file.
    Months are calendar months: YYYY-MM.

Output:
    ./chi2_distributions/time_evolution_3d/<detector>/{per_wp,all_wp}/
"""


from pathlib import Path
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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

OUTPUT_DIR = Path("./chi2_distributions/time_evolution_3d")

TREE_NAME = "AnalyzedData"

FILENAME_TEMPLATE = "MURAVES_AnalyzedData_run{run}.root"

SLOWCONTROL_TEMPLATE = "SLOWCONTROL_run{run}"


CHI2_MIN = 0.0
CHI2_MAX = 20.0
BIN_WIDTH = 0.1

# Fine bins are rebinned by this integer factor for plotting
# (0.1 * 2 = 0.2 wide bins -> smoother 3D slices, less noise per bin).
REBIN = 2

# Months with fewer entries than this are not drawn / not compared.
MIN_ENTRIES = 100

# 3D camera
ELEV = 28
AZIM = -62


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


VES_COLOR = "tab:blue"
FS_COLOR = "tab:orange"

# Selection key meaning "all working points merged".
ALL_WP = "ALL"


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


# ============================================================
# TIME / SLOW CONTROL
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

        print(
            "  [warn] slow-control file not found: "
            f"{slowcontrol_file}"
        )

        return None

    try:

        with open(slowcontrol_file, "r", errors="replace") as f:

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

                timestamp = parse_slowcontrol_timestamp(fields)

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
                        f"in {slowcontrol_file}: {fields[5]}"
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
                        f"in {slowcontrol_file}: {working_point}"
                    )

                    return None

                return timestamp, int(working_point)

    except Exception as exc:

        print(
            f"  [warn] could not read {slowcontrol_file}: {exc}"
        )

        return None

    print(f"  [warn] no valid lines in {slowcontrol_file}")

    return None


def month_key(timestamp):
    """Calendar month identifier, YYYY-MM."""

    return timestamp.strftime("%Y-%m")


def month_range(months):
    """
    Continuous list of calendar months between the first and
    last month present. Months without data are kept so gaps
    show up as gaps.
    """

    months = list(months)

    if not months:
        return []

    first, last = min(months), max(months)

    y, m = (int(v) for v in first.split("-"))
    ly, lm = (int(v) for v in last.split("-"))

    out = []

    while (y, m) <= (ly, lm):

        out.append(f"{y:04d}-{m:02d}")

        m += 1

        if m == 13:
            y += 1
            m = 1

    return out


def format_month_labels(months):

    return [
        datetime.strptime(m, "%Y-%m").strftime("%m-%Y")
        for m in months
    ]


def tick_indices(n, max_ticks=24):
    """Thin out month ticks when there are many months."""

    step = max(1, int(np.ceil(n / max_ticks)))

    return np.arange(0, n, step)


# ============================================================
# ROOT DATA
# ============================================================

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

                print(
                    f"  [warn] missing branches: {sorted(missing)}"
                )

                return None

            return tree.arrays(branches, library="np")

    except Exception as exc:

        print(f"  [warn] could not read {file_path}: {exc}")

        return None


def get_valid_chi2(values):

    values = np.asarray(values, dtype=float)

    valid = (
        np.isfinite(values)
        & (values >= CHI2_MIN)
        & (values <= CHI2_MAX)
    )

    return values[valid]


def make_bins():

    n_bins = int(np.ceil((CHI2_MAX - CHI2_MIN) / BIN_WIDTH))

    return np.linspace(CHI2_MIN, CHI2_MAX, n_bins + 1)


# ============================================================
# HISTOGRAM STORAGE / MANIPULATION
#
# Only per-month COUNTS per bin are stored (not raw values):
# summing counts is exact for merging runs / WPs / months,
# and uses far less memory.
#
# store[WP][branch][month] -> counts array (fine bins)
# ============================================================

def add_counts(store, wp, short_name, month, counts):

    per_branch = store.setdefault(wp, {}).setdefault(short_name, {})

    if month in per_branch:
        per_branch[month] += counts

    else:
        per_branch[month] = counts.copy()


def get_selection_counts(store, selection, short_name):
    """
    Return {month: counts} for one WP, or merged over all WPs
    when selection == ALL_WP.
    """

    if selection == ALL_WP:

        out = {}

        for wp_data in store.values():

            for month, c in wp_data.get(short_name, {}).items():

                if month in out:
                    out[month] = out[month] + c

                else:
                    out[month] = c.copy()

        return out

    return dict(store.get(selection, {}).get(short_name, {}))


def rebin_counts(counts, factor):

    return counts.reshape(-1, factor).sum(axis=1)


def density_and_error(counts, width):
    """
    Normalized density and its Poisson error:
        d = n / (N w),   err = sqrt(n) / (N w)
    """

    total = counts.sum()

    if total <= 0:

        nan = np.full(len(counts), np.nan)

        return nan, nan.copy(), 0

    dens = counts / (total * width)
    err = np.sqrt(counts) / (total * width)

    return dens, err, int(total)


def build_matrices(counts_by_month, months, width):
    """
    Rows = months, columns = (rebinned) chi2 bins.

    Returns density, error, entries per month.
    """

    n_bins = len(make_bins()) - 1
    n_bins //= REBIN

    dens = np.full((len(months), n_bins), np.nan)
    err = np.full((len(months), n_bins), np.nan)
    entries = np.zeros(len(months), dtype=int)

    for i, month in enumerate(months):

        counts = counts_by_month.get(month)

        if counts is None:
            continue

        counts = rebin_counts(counts, REBIN)

        d, e, n = density_and_error(counts, width)

        entries[i] = n

        dens[i] = d
        err[i] = e

    return dens, err, entries


def ks_distance(counts_a, counts_b):
    """
    Binned Kolmogorov-Smirnov distance and the 95% critical value
    (statistical only, two-sample, asymptotic).
    """

    n = counts_a.sum()
    m = counts_b.sum()

    if n <= 0 or m <= 0:
        return np.nan, np.nan

    cdf_a = np.cumsum(counts_a) / n
    cdf_b = np.cumsum(counts_b) / m

    ks = np.max(np.abs(cdf_a - cdf_b))

    crit = 1.358 * np.sqrt((n + m) / (n * m))

    return ks, crit


# ============================================================
# 3D HELPERS
# ============================================================

def plot_waterfall(ax, centers, matrix, color):
    """
    One filled chi2 distribution per month (slice at x = month index).
    """

    verts = []

    for i, row in enumerate(matrix):

        if not np.any(np.isfinite(row)):
            continue

        z = np.nan_to_num(row, nan=0.0)

        poly = [(i, centers[0], 0.0)]
        poly += [(i, c, v) for c, v in zip(centers, z)]
        poly.append((i, centers[-1], 0.0))

        verts.append(poly)

    if verts:

        ax.add_collection3d(
            Poly3DCollection(
                verts,
                facecolors=color,
                edgecolors=color,
                linewidths=0.7,
                alpha=0.45,
            )
        )


def style_3d_axes(ax, months, zmax, title):

    n = max(len(months), 1)

    idx = tick_indices(n)

    labels = format_month_labels(months)

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(CHI2_MIN, CHI2_MAX)
    ax.set_zlim(0, zmax)

    ax.set_xticks(idx)
    ax.set_xticklabels(
        [labels[i] for i in idx] if labels else [],
        rotation=35,
        ha="right",
        fontsize=7,
    )

    #ax.set_xlabel("Month (MM-YYYY)", labelpad=12)
    ax.set_ylabel(r"$\chi^2$")
    ax.set_zlabel("Density")

    ax.set_title(title)

    ax.set_box_aspect((2.2, 1.0, 0.7))

    ax.view_init(elev=ELEV, azim=AZIM)


def bin_centers_and_edges():

    edges = make_bins()[::REBIN]

    centers = 0.5 * (edges[:-1] + edges[1:])

    return edges, centers


# ============================================================
# FIGURE 1: 3D MONTHLY, VES | FS
# ============================================================

def plot_3d_monthly(
    counts_ves,
    counts_fs,
    detector,
    selection_label,
    branch_name,
    output_file,
):

    if not counts_ves and not counts_fs:
        return False

    width = BIN_WIDTH * REBIN

    _, centers = bin_centers_and_edges()

    months_ves = month_range(counts_ves)
    months_fs = month_range(counts_fs)

    dens_ves, _, _ = build_matrices(counts_ves, months_ves, width)
    dens_fs, _, _ = build_matrices(counts_fs, months_fs, width)

    finite = [
        m[np.isfinite(m)]
        for m in (dens_ves, dens_fs)
        if m.size and np.any(np.isfinite(m))
    ]

    if not finite:
        return False

    # Common z scale so VES and FS are visually comparable.
    zmax = 1.05 * max(np.max(f) for f in finite)

    fig = plt.figure(figsize=(18, 7.5))

    ax_ves = fig.add_subplot(1, 2, 1, projection="3d")
    ax_fs = fig.add_subplot(1, 2, 2, projection="3d")

    plot_waterfall(ax_ves, centers, dens_ves, VES_COLOR)
    plot_waterfall(ax_fs, centers, dens_fs, FS_COLOR)

    style_3d_axes(ax_ves, months_ves, zmax, "VES")
    style_3d_axes(ax_fs, months_fs, zmax, "FS")

    fig.suptitle(
        f"{branch_name} - {detector}, {selection_label}",
        y=0.97,
    )

    fig.savefig(output_file, dpi=150, bbox_inches="tight")

    plt.close(fig)

    return True


# ============================================================
# FIGURE 2: FS(month) vs VES(full-period) BENCHMARK
# ============================================================

def plot_benchmark_comparison(
    counts_bench,
    counts_ves,
    counts_fs,
    detector,
    selection_label,
    benchmark_label,
    branch_name,
    diff_mode,
    output_file,
):
    """
    counts_bench : fine-bin counts of VES integrated over all months
    counts_ves   : {month: counts} VES (only used for control band)
    counts_fs    : {month: counts} FS
    """

    if not counts_fs or counts_bench is None or counts_bench.sum() <= 0:
        return False

    width = BIN_WIDTH * REBIN

    edges, centers = bin_centers_and_edges()

    months = month_range(counts_fs)

    n = len(months)

    # --------------------------------------------------------
    # Benchmark and FS densities.
    # --------------------------------------------------------

    bench = rebin_counts(counts_bench, REBIN)

    bench_dens, bench_err, bench_n = density_and_error(bench, width)

    fs_dens, fs_err, fs_entries = build_matrices(
        counts_fs, months, width
    )

    if not np.any(np.isfinite(fs_dens)):
        return False

    # --------------------------------------------------------
    # Difference / pull.
    # --------------------------------------------------------

    diff = fs_dens - bench_dens[None, :]

    if diff_mode == "pull":

        sigma = np.sqrt(fs_err**2 + bench_err[None, :] ** 2)

        with np.errstate(divide="ignore", invalid="ignore"):
            diff = np.where(sigma > 0, diff / sigma, np.nan)

        vlim = 5.0
        cbar_label = (
            "(FS - VES bench.) / stat. error"
        )

    else:

        finite = np.abs(diff[np.isfinite(diff)])

        vlim = (
            np.percentile(finite, 99.5)
            if finite.size
            else 1.0
        )

        if not np.isfinite(vlim) or vlim <= 0:
            vlim = 1.0

        cbar_label = "Density(FS month) - Density(VES benchmark)"

    # --------------------------------------------------------
    # KS distances.
    # --------------------------------------------------------

    ks_fs = np.full(n, np.nan)
    crit_fs = np.full(n, np.nan)

    for i, month in enumerate(months):

        c = counts_fs.get(month)

        if c is None or c.sum() < MIN_ENTRIES:
            continue

        ks_fs[i], crit_fs[i] = ks_distance(
            rebin_counts(c, REBIN), bench
        )

    ks_ves = []

    for c in counts_ves.values():

        if c.sum() < MIN_ENTRIES:
            continue

        ks, _ = ks_distance(rebin_counts(c, REBIN), bench)

        if np.isfinite(ks):
            ks_ves.append(ks)

    # --------------------------------------------------------
    # Figure layout.
    # --------------------------------------------------------

    fig = plt.figure(figsize=(14, 16))

    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        width_ratios=[1, 0.025],
        height_ratios=[2.3, 1.4, 0.9],
        hspace=0.30,
        wspace=0.04,
    )

    ax3d = fig.add_subplot(gs[0, :], projection="3d")
    ax_diff = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 1])
    ax_ks = fig.add_subplot(gs[2, 0], sharex=ax_diff)

    # --------------------------------------------------------
    # Top: 3D FS slices + VES benchmark inside each slice.
    # --------------------------------------------------------

    plot_waterfall(ax3d, centers, fs_dens, FS_COLOR)

    for i in range(n):

        if not np.any(np.isfinite(fs_dens[i])):
            continue

        ax3d.plot(
            np.full(len(centers), i),
            centers,
            bench_dens,
            color=VES_COLOR,
            linewidth=0.9,
            alpha=0.9,
        )

    zmax = 1.05 * max(
        np.nanmax(fs_dens),
        np.nanmax(bench_dens),
    )

    style_3d_axes(
        ax3d,
        months,
        zmax,
        "FS per month, with VES benchmark curve in every slice",
    )

    ax3d.legend(
        handles=[
            Patch(
                facecolor=FS_COLOR,
                alpha=0.45,
                label="FS (month)",
            ),
            Line2D(
                [0],
                [0],
                color=VES_COLOR,
                label=(
                    f"VES benchmark, full period "
                    f"({benchmark_label}, N={bench_n:,})"
                ),
            ),
        ],
        loc="upper left",
    )

    # --------------------------------------------------------
    # Middle: difference heatmap.
    # --------------------------------------------------------

    mesh = ax_diff.pcolormesh(
        np.arange(n + 1),
        edges,
        diff.T,
        cmap="RdBu_r",
        vmin=-vlim,
        vmax=vlim,
        shading="flat",
    )

    ax_diff.set_facecolor("0.92")

    ax_diff.set_ylabel(r"$\chi^2$")

    ax_diff.set_ylim(CHI2_MIN, CHI2_MAX)

    ax_diff.set_title(
        "Difference: FS (month) - VES (benchmark)"
        "   [red = FS excess, blue = FS deficit]"
    )

    cbar = fig.colorbar(mesh, cax=cax)

    cbar.set_label(cbar_label)

    plt.setp(ax_diff.get_xticklabels(), visible=False)

    # --------------------------------------------------------
    # Bottom: KS distance per month.
    # --------------------------------------------------------

    x = np.arange(n) + 0.5

    ax_ks.bar(
        x,
        ks_fs,
        width=0.7,
        color=FS_COLOR,
        label="KS(FS month, VES benchmark)",
    )

    ax_ks.plot(
        x,
        crit_fs,
        linestyle="none",
        marker="_",
        markersize=14,
        markeredgewidth=2,
        color="black",
        label="95% threshold (stat. only)",
    )

    if ks_ves:

        ax_ks.axhspan(
            min(ks_ves),
            max(ks_ves),
            color=VES_COLOR,
            alpha=0.15,
            label="VES month vs VES benchmark (min-max)",
        )

        ax_ks.axhline(
            np.median(ks_ves),
            color=VES_COLOR,
            linestyle="--",
            linewidth=1.2,
        )

    ax_ks.set_ylabel("KS distance")

    ax_ks.set_xlim(0, n)

    ax_ks.grid(True, alpha=0.3, linestyle="--", axis="y")

    ax_ks.legend(fontsize=8, ncol=2, loc="upper left")

    idx = tick_indices(n)

    labels = format_month_labels(months)

    ax_ks.set_xticks(idx + 0.5)

    ax_ks.set_xticklabels(
        [labels[i] for i in idx],
        rotation=45,
        ha="right",
    )

    ax_ks.set_xlabel("Month (MM-YYYY)")

    fig.suptitle(
        f"{branch_name} - {detector}, {selection_label}",
        y=0.92,
    )

    fig.savefig(output_file, dpi=150, bbox_inches="tight")

    plt.close(fig)

    return True


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "3D monthly chi2 distributions and FS-vs-VES "
            "benchmark comparison."
        )
    )

    parser.add_argument(
        "--detector",
        choices=sorted(RUN_RANGES.keys()),
        help="Detector to process, e.g. ROSSO",
    )

    parser.add_argument(
        "--detectors",
        nargs="+",
        choices=sorted(RUN_RANGES.keys()),
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
        help=f"Maximum chi2 shown, default={CHI2_MAX}",
    )

    parser.add_argument(
        "--bin-width",
        type=float,
        default=BIN_WIDTH,
        help=f"Fine histogram bin width, default={BIN_WIDTH}",
    )

    parser.add_argument(
        "--rebin",
        type=int,
        default=REBIN,
        help=(
            "Merge this many fine bins for plotting "
            f"(must divide the number of bins), default={REBIN}"
        ),
    )

    parser.add_argument(
        "--min-entries",
        type=int,
        default=MIN_ENTRIES,
        help=(
            "Skip months with fewer entries than this, "
            f"default={MIN_ENTRIES}"
        ),
    )

    parser.add_argument(
        "--benchmark",
        choices=["same-wp", "all-wp"],
        default="same-wp",
        help=(
            "VES benchmark used in the per-WP plots: "
            "VES of the same WP (default) or VES of all WPs."
        ),
    )

    parser.add_argument(
        "--diff-mode",
        choices=["absolute", "pull"],
        default="absolute",
        help=(
            "Heatmap of FS-VES: absolute density difference "
            "(default) or difference / statistical error."
        ),
    )

    parser.add_argument("--elev", type=float, default=ELEV)

    parser.add_argument("--azim", type=float, default=AZIM)

    args = parser.parse_args()

    if args.detector and args.detectors:
        parser.error(
            "Use either --detector or --detectors, not both."
        )

    detectors = (
        [args.detector]
        if args.detector
        else args.detectors
        if args.detectors
        else ["ROSSO"]
    )

    if args.chi2_max <= CHI2_MIN:
        parser.error("--chi2-max must be > 0")

    if args.bin_width <= 0:
        parser.error("--bin-width must be > 0")

    if args.rebin < 1:
        parser.error("--rebin must be >= 1")

    return (
        args,
        detectors,
        parse_range_list(args.ves_ranges),
        parse_range_list(args.fs_ranges),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    global CHI2_MAX, BIN_WIDTH, REBIN, MIN_ENTRIES, ELEV, AZIM

    (
        args,
        detectors,
        ves_override,
        fs_override,
    ) = parse_arguments()

    CHI2_MAX = args.chi2_max
    BIN_WIDTH = args.bin_width
    REBIN = args.rebin
    MIN_ENTRIES = args.min_entries
    ELEV = args.elev
    AZIM = args.azim

    n_fine = len(make_bins()) - 1

    if n_fine % REBIN != 0:
        raise SystemExit(
            f"--rebin {REBIN} must divide the number of fine "
            f"bins ({n_fine})."
        )

    print("=" * 70)
    print("MURAVES CHI2: 3D MONTHLY + VES BENCHMARK")
    print("=" * 70)
    print(f"Detectors   : {', '.join(detectors)}")
    print(f"Chi2 range  : {CHI2_MIN} -> {CHI2_MAX}")
    print(
        f"Bin width   : {BIN_WIDTH} (plot bins: {BIN_WIDTH * REBIN:g})"
    )
    print(f"Min entries : {MIN_ENTRIES} per month")
    print(f"Benchmark   : {args.benchmark} (per-WP plots)")
    print(f"Diff mode   : {args.diff_mode}")
    print("=" * 70)

    bins = make_bins()

    for detector in detectors:

        print()
        print("#" * 70)
        print(f"DETECTOR: {detector}")
        print("#" * 70)

        data_dir = Path(DATA_DIR_TEMPLATE.format(detector=detector))

        detector_output = OUTPUT_DIR / detector

        (detector_output / "per_wp").mkdir(parents=True, exist_ok=True)
        (detector_output / "all_wp").mkdir(parents=True, exist_ok=True)

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

        # store[data_type][WP][branch][month] -> counts
        store = {"VES": {}, "FS": {}}

        # ====================================================
        # READ DATA
        # ====================================================

        for data_type in ("VES", "FS"):

            print()
            print(f"========== {detector} / {data_type} ==========")

            for run in iter_run_numbers(run_ranges[data_type]):

                file_path = data_dir / FILENAME_TEMPLATE.format(run=run)

                if not file_path.is_file():
                    print(f"Run {run}: file not found, skipping")
                    continue

                info = get_slowcontrol_info(
                    SLOWCONTROL_DIR, detector, run
                )

                if info is None:
                    print(f"Run {run}: no timestamp/WP, skipping")
                    continue

                timestamp, working_point = info

                month = month_key(timestamp)

                arrays = read_run(file_path)

                if arrays is None:
                    continue

                n_total = 0

                for short_name, branch_name in CHI2_BRANCHES.items():

                    values = get_valid_chi2(arrays[branch_name])

                    if len(values) == 0:
                        continue

                    counts, _ = np.histogram(values, bins=bins)

                    add_counts(
                        store[data_type],
                        working_point,
                        short_name,
                        month,
                        counts.astype(np.int64),
                    )

                    n_total += len(values)

                print(
                    f"Run {run}: {timestamp:%d-%m-%Y %H:%M:%S}  "
                    f"month={month}  WP={working_point}  "
                    f"entries (all branches)={n_total:,}"
                )

        # ====================================================
        # PLOTS
        # ====================================================

        print()
        print(f"Creating plots for {detector}...")

        working_points = sorted(
            set(store["VES"]) | set(store["FS"])
        )

        selections = [(wp, f"WP{wp}", f"WP{wp}", "per_wp")
                      for wp in working_points]

        selections.append(
            (ALL_WP, "all WPs", "allWP", "all_wp")
        )

        for selection, label, token, subdir in selections:

            outdir = detector_output / subdir

            for short_name, branch_name in CHI2_BRANCHES.items():

                counts_ves = get_selection_counts(
                    store["VES"], selection, short_name
                )

                counts_fs = get_selection_counts(
                    store["FS"], selection, short_name
                )

                if not counts_ves and not counts_fs:
                    continue

                # ---- Figure 1: 3D VES | FS ----------------------

                file_3d = outdir / (
                    f"chi2_3d_{short_name}_{token}_VES_FS.png"
                )

                done_3d = plot_3d_monthly(
                    counts_ves,
                    counts_fs,
                    detector,
                    label,
                    branch_name,
                    file_3d,
                )

                # ---- Figure 2: benchmark ------------------------

                if selection == ALL_WP or args.benchmark == "all-wp":
                    bench_selection = ALL_WP
                    bench_label = "all WPs"

                else:
                    bench_selection = selection
                    bench_label = label

                bench_months = get_selection_counts(
                    store["VES"], bench_selection, short_name
                )

                counts_bench = (
                    np.sum(list(bench_months.values()), axis=0)
                    if bench_months
                    else None
                )

                file_bench = outdir / (
                    f"chi2_benchmark_{short_name}_{token}.png"
                )

                done_bench = plot_benchmark_comparison(
                    counts_bench,
                    counts_ves,
                    counts_fs,
                    detector,
                    label,
                    bench_label,
                    branch_name,
                    args.diff_mode,
                    file_bench,
                )

                print(
                    f"  {label} {short_name}: "
                    f"VES months={len(counts_ves)}, "
                    f"FS months={len(counts_fs)}, "
                    f"3D={'ok' if done_3d else 'skipped'}, "
                    f"benchmark={'ok' if done_bench else 'skipped'}"
                )

        print()
        print(f"Output directory: {detector_output}")

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()