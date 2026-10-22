#!/usr/bin/env python3
"""
MURAVES chi2 distributions and per-run statistics.

Changes with respect to the previous version
--------------------------------------------
1.  Heatmaps and monthly histograms: theta = all only (no theta ranges).
2.  chi2_hist: the theta ranges are drawn on ONE figure (colour = theta
    range, linestyle = VES solid / FS dashed), forward = upper panel,
    backward = lower panel. The theta-all histogram is still made
    separately (colour = VES / FS). "theta =" removed from the title.
3.  Median + q25-q75 / q05-q95 per run and quantiles_per_run removed.
4.  Weekly quantiles are made in two variants:
      *_text.png : number of tracks written above the median+error bar,
                   no right y-axis / no bars
      *_bars.png : number of tracks as bars on the right y-axis,
                   no text
5.  "(to scale)" removed from the ISO-week x-axis label.
6.  "Month"/"ISO week" label of the top axis removed on the per-run plots
    (the ticks are kept).
7.  per_wp plots are no longer made.
8.  Output is sorted into subdirectories per plot type:
      hist/                      chi2 hists (theta all + theta ranges)
      hist/errorbars/            same hists with error bars
      monthly_hists/
      heatmaps/
      quantiles_weekly/          weekly quantiles, theta all
      quantiles_weekly/theta_ranges/   weekly quantiles per theta range
      per_run/                   mean / median / max per run
9.  Weekly quantiles: only the "text" variant is made (the "bars"
    variant is still available in plot_weekly_quantiles_comparison).

10. Weekly quantile plots: y-range 0-QUANTILE_YMAX (13). Extra "median
    only" version (no percentile bands, own y-range WEEKLY_MEDIAN_RANGE)
    in quantiles_weekly/median_only/ (+ theta_ranges/).

11. No upper chi2 cut on the tracks any more: xy and xz plots of the same
    fit contain exactly the same tracks (same N). Histograms/heatmaps are
    normalised to ALL tracks, so the part above CHI2_MAX is just not
    drawn (the area in the plotted range is then slightly < 1).
    Weekly quantiles/medians are computed on all tracks (no truncation).

12. Per-run mean/median/max: two panels (forward top, backward bottom),
    light background colour for the VES / FS run ranges.

13. Compact cache (one .npz per detector, ALL runs, float16 chi2 and
    uint16 angles*100). If it exists, ROOT and slow control are skipped.
    Use --rebuild-cache to force re-reading, --cache-dir to point to
    another directory. Each npz array is decompressed only once.

14. --run-selection all|good|both (default: both). Good-run selection is
    applied after the cache. Plots go to <detector>/all_runs/ and
    <detector>/good_runs/, and the title carries the label
    "[all runs]" / "[good runs only]".
15. DAC10-colored median-per-run plots are available with
    --only-median-per-run-dac, which skips all other plot types.

!! The angle branch names (ANGLE_BRANCHES) are a GUESS - check them.
"""

from pathlib import Path
import argparse
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import uproot
import select_runs as sr


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR_TEMPLATE = (
    "/pnfs/iihe/muraves/muraves_DATA/"
    "RECONSTRUCTED/{detector}/v1"
)

CACHE_DIR = Path("./chi2_distribution_cache")
CHI2_CLIP = 60000.0   # float16 max is 65504

SLOWCONTROL_DIR = "/pnfs/iihe/muraves/muraves_DATA/DECOMPRESSED"

OUTPUT_DIR = Path("./chi2_distributions/time_evolution_controlregion")

TREE_NAME = "AnalyzedData"
FILENAME_TEMPLATE = "MURAVES_AnalyzedData_run{run}.root"
SLOWCONTROL_TEMPLATE = "SLOWCONTROL_run{run}"

# Used for the distribution plots (hist, monthly hist, heatmaps, quantiles).
CHI2_MIN = 0.0
CHI2_MAX = 20.0
BIN_WIDTH = 0.1

CHI2_BRANCHES = {
    "3p_xy": "BestTrack_3p_ChiSquare_xy",
    "3p_xz": "BestTrack_3p_ChiSquare_xz",
    "4p_xy": "BestTrack_4p_ChiSquare_xy",
    "4p_xz": "BestTrack_4p_ChiSquare_xz",
}

FITS = ("3p", "4p")
PLANES = ["Z1", "Y1", "Z2", "Y2", "Z3", "Y3", "Z4", "Y4"]
DAC_FIRST_INDEX = 6

# !!! GUESSED NAMES - adapt to the real branch names in AnalyzedData !!!
ANGLE_BRANCHES = {
    "3p": {"theta": "Theta_3p", "phi": "Phi_3p"},
    "4p": {"theta": "Theta_4p", "phi": "Phi_4p"},
}

# "auto": per run, values are treated as radians if everything is <= 2*pi,
# otherwise as degrees. Can be forced with --angle-unit deg / rad.
ANGLE_UNIT = "auto"

# Forward-entering tracks: 90 <= phi <= 270 deg. Everything else = backward.
PHI_FWD_MIN = 90.0
PHI_FWD_MAX = 270.0

DIRECTIONS = [
    ("fwd", "Forward-entering (90° ≤ φ ≤ 270°)"),
    ("bwd", "Backward-entering (otherwise)"),
]

# (label, theta_min, theta_max) in degrees; theta_max=None -> open ended.
THETA_RANGES = [
    ("0-20", 0.0, 20.0),
    ("20-40", 20.0, 40.0),
    ("40-60", 40.0, 60.0),
    ("60+", 60.0, None),
]

# One colour per theta range (used in the combined theta histogram)
THETA_COLORS = {
    "0-20": "tab:blue",
    "20-40": "tab:green",
    "40-60": "tab:orange",
    "60+": "tab:red",
}

# Linestyle per data type in the combined theta histogram
TYPE_LINESTYLES = {"VES": "-", "FS": "--"}
TYPE_OPACITY = {"VES": 1.0, "FS": 0.5}

# y-axis upper limit of the weekly quantile plots
QUANTILE_YMAX = 13.0

# y-range of the weekly "median only" plots (per fit)
WEEKLY_MEDIAN_RANGE = {
    "3p": (0.0, 1.5),
    "4p": (0.0, 6.0),
}

# WPs shown together in the combined per-run plots.
PLOT_WPS = {15, 20, 25}

# y-range of the zoomed median-per-run plots
MEDIAN_ZOOM = {
    "3p": (0.0, 1.5),
    "4p": (0.0, 8.0),
}

RUN_MARKER_SIZE = 2

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

# Label shown in the plot titles ("all runs" / "good runs only").
# Set per selection in make_plots().
RUN_LABEL = ""

# Optional angular selection (degrees); configured from CLI in main().
THETA_MIN_CUT = None
THETA_MAX_CUT = None
PHI_MIN_CUT = None
PHI_MAX_CUT = None


# ============================================================
# COLORS AND MARKERS
# ============================================================

WP_COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
    "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
]

TYPE_MARKERS = {"VES": "o", "FS": "^"}

VES_COLOR = "tab:blue"
FS_COLOR = "tab:orange"
TYPE_COLORS = {"VES": VES_COLOR, "FS": FS_COLOR}


def make_wp_color_map(records):
    """One colour per WP present in the records (stable across plots)."""
    wps = sorted({rec["wp"] for rec in records})
    return {wp: WP_COLORS[i % len(WP_COLORS)] for i, wp in enumerate(wps)}


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
    if values is None:
        return None
    return [parse_run_range(v) for v in values]


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

    candidates = [raw, raw.replace("T", " "), raw.replace("/", "-")]

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
    slowcontrol_file = (
        Path(slowcontrol_dir)
        / detector
        / SLOWCONTROL_TEMPLATE.format(run=run)
    )

    if not slowcontrol_file.is_file():
        print(f"  [warn] slow-control file not found: {slowcontrol_file}")
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
                        f"from slow-control column/index 1: {fields[1]}"
                    )
                    return None

                try:
                    working_point = float(fields[5])
                except ValueError:
                    print(
                        f"  [warn] invalid Working Point in "
                        f"{slowcontrol_file}: {fields[5]}"
                    )
                    return None

                if not np.isfinite(working_point):
                    print(
                        f"  [warn] non-finite Working Point in "
                        f"{slowcontrol_file}"
                    )
                    return None

                if not working_point.is_integer():
                    print(
                        f"  [warn] non-integer Working Point in "
                        f"{slowcontrol_file}: {working_point}"
                    )
                    return None

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
                            f"  [warn] incomplete DAC10 values in "
                            f"{slowcontrol_file}"
                        )
                except ValueError:
                    print(
                        f"  [warn] invalid DAC10 values in "
                        f"{slowcontrol_file}"
                    )

                return timestamp, int(working_point), dac_per_plane

    except Exception as exc:
        print(f"  [warn] could not read {slowcontrol_file}: {exc}")
        return None

    print(f"  [warn] no valid lines in {slowcontrol_file}")
    return None


def get_good_run_set(detector):
    """Good runs from select_runs (applied after the cache, no ROOT needed)."""
    print()
    print(f"Selecting good runs for {detector}...")

    df = sr.load_parsing_overview(detector)
    ped = sr.load_pedestal_outliers(detector)
    mon = sr.load_monitoring_scan(detector)

    good_runs, details, cutflow, trigger_info = sr.select_runs(
        df,
        ped,
        mon,
        run_type="ADC",
        trigger_rate_n_sigma=3.0,
        label=f"[{detector}]",
    )

    good_runs = np.asarray(good_runs, dtype=int)
    print(f"  Good runs: {len(good_runs)}")
    return set(good_runs.tolist())


def month_key(timestamp):
    return timestamp.strftime("%Y-%m")


def week_key(timestamp):
    iso_year, iso_week, _ = timestamp.isocalendar()
    return f"{iso_year}-W{iso_week:02d}"


def month_label(month):
    return datetime.strptime(month, "%Y-%m").strftime("%m-%Y")


def week_to_date(week):
    """Monday of an ISO week string like '2025-W07'."""
    year = int(week[:4])
    wk = int(week.split("W")[1])
    return datetime.fromisocalendar(year, wk, 1)


def make_title(branch_name, detector, working_point, theta_label=None):
    """
    '<branch> - <detector>, WP<wp> [run label]' (theta part only for the
    weekly quantile plots of one theta range).
    """
    title = f"{branch_name} - {detector}, WP{working_point}"
    if theta_label is not None and theta_label != "all":
        title += f", {theta_label}°"
    if any(v is not None for v in (
        THETA_MIN_CUT, THETA_MAX_CUT, PHI_MIN_CUT, PHI_MAX_CUT
    )):
        parts = []
        if THETA_MIN_CUT is not None or THETA_MAX_CUT is not None:
            parts.append(f"theta {THETA_MIN_CUT if THETA_MIN_CUT is not None else '-'}"
                         f"–{THETA_MAX_CUT if THETA_MAX_CUT is not None else '-'}°")
        if PHI_MIN_CUT is not None or PHI_MAX_CUT is not None:
            parts.append(f"phi {PHI_MIN_CUT if PHI_MIN_CUT is not None else '-'}"
                         f"–{PHI_MAX_CUT if PHI_MAX_CUT is not None else '-'}°")
        title += " [" + ", ".join(parts) + "]"
    if RUN_LABEL:
        title += f" [{RUN_LABEL}]"
    return title


def fmt_n(n):
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}k"
    return str(n)


# ============================================================
# ROOT DATA
# ============================================================

def needed_branches():
    branches = list(CHI2_BRANCHES.values())
    for fit in FITS:
        branches.extend(ANGLE_BRANCHES[fit].values())
    return list(dict.fromkeys(branches))


def read_run(file_path):
    branches = needed_branches()

    try:
        with uproot.open(file_path) as f:
            if TREE_NAME not in f:
                print(f"  [warn] tree '{TREE_NAME}' not found")
                return None

            tree = f[TREE_NAME]
            tree_keys = set(tree.keys())
            missing = set(branches) - tree_keys

            if missing:
                print(f"  [warn] missing branches: {sorted(missing)}")
                hints = sorted(
                    k for k in tree_keys
                    if "theta" in k.lower() or "phi" in k.lower()
                )
                if hints:
                    print(
                        "  [hint] branches containing theta/phi: "
                        f"{hints[:20]}"
                    )
                    print(
                        "  [hint] fix ANGLE_BRANCHES at the top "
                        "of the script"
                    )
                return None

            return tree.arrays(branches, library="np")

    except Exception as exc:
        print(f"  [warn] could not read {file_path}: {exc}")
        return None


def build_fit_record(arrays, fit, angle_unit):
    """
    Keep only tracks reconstructed in BOTH projections
    (chi2_xy > 0 and chi2_xz > 0) with finite angles.
    Angles are stored in degrees, phi in [0, 360).
    """
    xy = np.asarray(arrays[CHI2_BRANCHES[f"{fit}_xy"]], dtype=float)
    xz = np.asarray(arrays[CHI2_BRANCHES[f"{fit}_xz"]], dtype=float)
    theta = np.asarray(arrays[ANGLE_BRANCHES[fit]["theta"]], dtype=float)
    phi = np.asarray(arrays[ANGLE_BRANCHES[fit]["phi"]], dtype=float)

    mask = (
        np.isfinite(xy) & np.isfinite(xz)
        & np.isfinite(theta) & np.isfinite(phi)
        & (xy > 0) & (xz > 0)
    )

    xy, xz, theta, phi = xy[mask], xz[mask], theta[mask], phi[mask]

    if len(xy) > 0:
        if angle_unit == "rad":
            is_rad = True
        elif angle_unit == "deg":
            is_rad = False
        else:
            is_rad = max(
                np.max(np.abs(theta)), np.max(np.abs(phi))
            ) <= 2 * np.pi + 0.01

        if is_rad:
            theta = np.degrees(theta)
            phi = np.degrees(phi)

    phi = np.mod(phi, 360.0)

    return {
        "xy": xy.astype(np.float32),
        "xz": xz.astype(np.float32),
        "theta": theta.astype(np.float32),
        "phi": phi.astype(np.float32),
    }


# ============================================================
# CACHE (compact per-track storage)
# ============================================================

def cache_path(cache_dir, detector, tag):
    return Path(cache_dir) / f"chi2_cache_{detector}{tag}.npz"


def cache_has_dac(path):
    with np.load(path, allow_pickle=False) as z:
        return "dac" in z.files


def save_cache(path, records):
    """Compact per-track storage: float16 chi2, uint16 angles*100."""
    path.parent.mkdir(parents=True, exist_ok=True)

    out = {
        "run": np.array([r["run"] for r in records], dtype=np.int32),
        "data_type": np.array([r["data_type"] for r in records], dtype="U3"),
        "wp": np.array([r["wp"] for r in records], dtype=np.int16),
        "timestamp": np.array(
            [r["timestamp"].isoformat() for r in records], dtype="U32"
        ),
        "dac": np.array(
            [
                [
                    r.get("dac", {}).get(plane, np.nan)
                    if r.get("dac") is not None else np.nan
                    for plane in PLANES
                ]
                for r in records
            ],
            dtype=np.float32,
        ),
    }

    for fit in FITS:
        lens = [len(r["fit"][fit]["xy"]) for r in records]
        out[f"{fit}_offsets"] = np.concatenate(
            [[0], np.cumsum(lens)]
        ).astype(np.int64)

        for key in ("xy", "xz"):
            arr = np.concatenate([r["fit"][fit][key] for r in records])
            out[f"{fit}_{key}"] = np.clip(arr, 0, CHI2_CLIP).astype(np.float16)

        for key in ("theta", "phi"):
            arr = np.concatenate([r["fit"][fit][key] for r in records])
            out[f"{fit}_{key}"] = np.clip(
                np.round(arr * 100.0), 0, 65535
            ).astype(np.uint16)

    np.savez_compressed(path, **out)
    print(f"  Cache written: {path} ({path.stat().st_size / 1e6:.1f} MB)")


def load_cache(path):
    """
    Returns (records, run_info) in the same format as the ROOT reader.
    Every array is read (decompressed) from the .npz exactly ONCE.
    """
    z = np.load(path, allow_pickle=False)

    runs = z["run"]
    data_types = z["data_type"]
    wps = z["wp"]
    timestamps = z["timestamp"]
    dacs = z["dac"] if "dac" in z.files else None

    data = {}
    for fit in FITS:
        data[fit] = {
            "offsets": z[f"{fit}_offsets"],
            "xy": z[f"{fit}_xy"],
            "xz": z[f"{fit}_xz"],
            "theta": z[f"{fit}_theta"],
            "phi": z[f"{fit}_phi"],
        }

    z.close()

    records, run_info = [], {}

    for i in range(len(runs)):
        run = int(runs[i])
        ts = datetime.fromisoformat(str(timestamps[i]))
        fit_records = {}

        for fit in FITS:
            d = data[fit]
            a, b = int(d["offsets"][i]), int(d["offsets"][i + 1])
            fit_records[fit] = {
                "xy": d["xy"][a:b].astype(np.float32),
                "xz": d["xz"][a:b].astype(np.float32),
                "theta": d["theta"][a:b].astype(np.float32) / 100.0,
                "phi": d["phi"][a:b].astype(np.float32) / 100.0,
            }

        rec = {
            "run": run,
            "data_type": str(data_types[i]),
            "wp": int(wps[i]),
            "timestamp": ts,
            "month": month_key(ts),
            "week": week_key(ts),
            "fit": fit_records,
        }
        records.append(rec)
        run_info[run] = {
            "timestamp": ts,
            "month": rec["month"],
            "week": rec["week"],
            "working_point": rec["wp"],
            "data_type": rec["data_type"],
            "dac": (
                {
                    plane: float(dacs[i, j])
                    for j, plane in enumerate(PLANES)
                    if np.isfinite(dacs[i, j])
                }
                if dacs is not None else None
            ),
        }

    del data
    return records, run_info


def filter_records(records, run_ranges):
    """Keep only runs inside the requested VES/FS ranges."""
    keep = []
    for rec in records:
        for lo, hi in run_ranges[rec["data_type"]]:
            if lo <= rec["run"] <= hi:
                keep.append(rec)
                break
    return keep


# ============================================================
# VALUE SELECTION
# ============================================================

def select_values(
    fit_rec,
    plane,
    direction=None,
    theta_range=None,
    max_value=None,
):
    """chi2 values of one projection after direction/theta/chi2 cuts."""
    chi = fit_rec[plane]

    if len(chi) == 0:
        return chi

    mask = np.ones(len(chi), dtype=bool)

    if direction is not None:
        phi = fit_rec["phi"]
        fwd = (phi >= PHI_FWD_MIN) & (phi <= PHI_FWD_MAX)
        mask &= fwd if direction == "fwd" else ~fwd

    if theta_range is not None:
        lo, hi = theta_range
        theta = fit_rec["theta"]
        mask &= theta >= lo
        if hi is not None:
            mask &= theta < hi

    # Optional angular-region selection, applied consistently to histograms
    # and per-run statistics. Angles are already converted to degrees.
    theta = fit_rec["theta"]
    phi = fit_rec["phi"]
    if THETA_MIN_CUT is not None:
        mask &= theta >= THETA_MIN_CUT
    if THETA_MAX_CUT is not None:
        mask &= theta <= THETA_MAX_CUT
    if PHI_MIN_CUT is not None and PHI_MAX_CUT is not None:
        # Support regions crossing phi=0, e.g. 300° to 40°.
        if PHI_MIN_CUT <= PHI_MAX_CUT:
            mask &= (phi >= PHI_MIN_CUT) & (phi <= PHI_MAX_CUT)
        else:
            mask &= (phi >= PHI_MIN_CUT) | (phi <= PHI_MAX_CUT)
    elif PHI_MIN_CUT is not None:
        mask &= phi >= PHI_MIN_CUT
    elif PHI_MAX_CUT is not None:
        mask &= phi <= PHI_MAX_CUT

    if max_value is not None:
        mask &= chi <= max_value

    return chi[mask]


def make_bins():
    n_bins = int(np.ceil((CHI2_MAX - CHI2_MIN) / BIN_WIDTH))
    return np.linspace(CHI2_MIN, CHI2_MAX, n_bins + 1)


def collect_periods(
    records,
    data_type,
    wp,
    fit,
    plane,
    direction,
    theta_range,
):
    """Returns (values_by_week, values_by_month) as dict period -> array."""
    weekly, monthly = {}, {}

    for rec in records:
        if rec["data_type"] != data_type or rec["wp"] != wp:
            continue

        vals = select_values(
            rec["fit"][fit],
            plane,
            direction=direction,
            theta_range=theta_range,
        )

        if len(vals) == 0:
            continue

        weekly.setdefault(rec["week"], []).append(vals)
        monthly.setdefault(rec["month"], []).append(vals)

    weekly = {k: np.concatenate(v) for k, v in weekly.items()}
    monthly = {k: np.concatenate(v) for k, v in monthly.items()}

    return weekly, monthly


def build_values(records, wp, fit, plane, theta_range):
    """
    week_vals[direction][data_type]  = dict week  -> array
    month_vals[direction][data_type] = dict month -> array
    alltime[direction][data_type]    = array
    """
    week_vals = {d: {} for d, _ in DIRECTIONS}
    month_vals = {d: {} for d, _ in DIRECTIONS}
    alltime = {d: {} for d, _ in DIRECTIONS}

    for direction, _ in DIRECTIONS:
        for data_type in ("VES", "FS"):
            weekly, monthly = collect_periods(
                records, data_type, wp, fit, plane, direction, theta_range
            )

            week_vals[direction][data_type] = weekly
            month_vals[direction][data_type] = monthly
            alltime[direction][data_type] = (
                np.concatenate(list(monthly.values()))
                if monthly
                else np.array([])
            )

    return week_vals, month_vals, alltime


def has_entries(alltime):
    return any(
        len(alltime[d][t]) > 0
        for d, _ in DIRECTIONS
        for t in ("VES", "FS")
    )


# ============================================================
# HISTOGRAM HELPER
# ============================================================

def draw_hist(
    ax, values, bins, color, label, linewidth, errorbars,
    linestyle="-", alpha=1.0,
):
    """Normalised step histogram, optionally with Poisson error bars."""
    if values is None or len(values) == 0:
        return

    counts, _ = np.histogram(values, bins=bins)
    n = len(values)  # all tracks, also those above CHI2_MAX

    if n == 0:
        return

    width = np.diff(bins)
    density = counts / (n * width)
    err = np.sqrt(counts) / (n * width)

    ax.step(
        bins,
        np.append(density, density[-1]),
        where="post",
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        label=f"{label} (N={n:,})",
        alpha=alpha,
    )

    if errorbars:
        centers = 0.5 * (bins[1:] + bins[:-1])
        ax.errorbar(
            centers,
            density,
            yerr=err,
            fmt="none",
            ecolor=color,
            elinewidth=0.9,
            capsize=1.5,
            alpha=0.8,
        )


# ============================================================
# 1D ALL-TIME HISTOGRAM (forward upper / backward lower)
# ============================================================

def plot_distribution_comparison(
    alltime,
    detector,
    working_point,
    branch_name,
    output_file,
    errorbars,
):
    """alltime[direction][data_type] = array of chi2 values (theta all)."""
    fig, axes = plt.subplots(
        2, 1, figsize=(9, 10), sharex=True, sharey=True
    )

    bins = make_bins()

    for ax, (direction, dir_title) in zip(axes, DIRECTIONS):
        for data_type in ("VES", "FS"):
            draw_hist(
                ax,
                alltime[direction].get(data_type),
                bins,
                TYPE_COLORS[data_type],
                data_type,
                1.8,
                errorbars,
            )

        ax.set_title(dir_title, fontsize=10)
        ax.set_ylabel("Density")
        ax.set_xlim(CHI2_MIN, CHI2_MAX)
        ax.grid(True, alpha=0.3, linestyle="--")
        if ax.get_legend_handles_labels()[0]:
            ax.legend()

    axes[-1].set_xlabel(r"$\chi^2$")

    fig.suptitle(make_title(branch_name, detector, working_point))

    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 1D ALL-TIME HISTOGRAM, ALL THETA RANGES ON ONE FIGURE
# ============================================================

def plot_theta_distribution_comparison(
    theta_alltime,
    detector,
    working_point,
    branch_name,
    output_file,
    errorbars,
):
    """
    theta_alltime[direction][theta_label][data_type] = array of chi2.
    Colour = theta range, linestyle = VES (solid) / FS (dashed).
    Forward upper panel, backward lower panel.
    """
    fig, axes = plt.subplots(
        2, 1, figsize=(9, 10), sharex=True, sharey=True
    )

    bins = make_bins()

    for ax, (direction, dir_title) in zip(axes, DIRECTIONS):
        for theta_label, _, _ in THETA_RANGES:
            for data_type in ("VES", "FS"):
                values = (
                    theta_alltime[direction]
                    .get(theta_label, {})
                    .get(data_type)
                )
                draw_hist(
                    ax,
                    values,
                    bins,
                    THETA_COLORS[theta_label],
                    f"θ {theta_label}° {data_type}",
                    1.6,
                    errorbars,
                    linestyle=TYPE_LINESTYLES[data_type],
                    alpha=TYPE_OPACITY[data_type],
                )

        ax.set_title(dir_title, fontsize=10)
        ax.set_ylabel("Density")
        ax.set_xlim(CHI2_MIN, CHI2_MAX)
        ax.grid(True, alpha=0.3, linestyle="--")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)

    axes[-1].set_xlabel(r"$\chi^2$")

    fig.suptitle(make_title(branch_name, detector, working_point))

    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MONTHLY HISTOGRAMS (forward upper / backward lower per month)
# ============================================================

def plot_monthly_histograms(
    month_vals,
    detector,
    working_point,
    branch_name,
    output_file,
):
    """month_vals[direction][data_type] = dict month -> array."""
    months = sorted(
        {
            m
            for d in month_vals
            for t in month_vals[d]
            for m in month_vals[d][t]
        }
    )

    if not months:
        return

    n_months = len(months)
    ncols = min(3, n_months)
    nblocks = int(np.ceil(n_months / ncols))

    fig, axes = plt.subplots(
        nrows=2 * nblocks,
        ncols=ncols,
        figsize=(5.0 * ncols, 2.8 * 2 * nblocks),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    bins = make_bins()
    width = np.diff(bins)

    ymax = 0.0
    for d in month_vals:
        for t in month_vals[d]:
            for values in month_vals[d][t].values():
                if len(values) == 0:
                    continue
                counts, _ = np.histogram(values, bins=bins)
                n = len(values)
                if n > 0:
                    ymax = max(ymax, np.max(counts / (n * width)))

    ymax = (ymax if ymax > 0 else 1.0) * 1.10

    used = set()

    for i, month in enumerate(months):
        row = 2 * (i // ncols)
        col = i % ncols

        for offset, (direction, dir_title) in enumerate(DIRECTIONS):
            ax = axes[row + offset, col]
            used.add((row + offset, col))

            for data_type in ("VES", "FS"):
                values = month_vals[direction].get(data_type, {}).get(month)
                draw_hist(
                    ax,
                    values,
                    bins,
                    TYPE_COLORS[data_type],
                    data_type,
                    1.4,
                    False,
                )

            if offset == 0:
                ax.set_title(f"{month_label(month)} - forward", fontsize=9)
            else:
                ax.set_title("backward", fontsize=9)

            ax.set_xlim(CHI2_MIN, CHI2_MAX)
            ax.set_ylim(0, ymax)
            ax.grid(True, alpha=0.25, linestyle="--")

            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=7)

    for r in range(axes.shape[0]):
        for c in range(axes.shape[1]):
            if (r, c) not in used:
                fig.delaxes(axes[r, c])

    fig.supxlabel(r"$\chi^2$")
    fig.supylabel("Density")
    fig.suptitle(
        make_title(branch_name, detector, working_point),
        y=0.995,
    )

    fig.tight_layout(rect=[0.03, 0.03, 1.0, 0.97])
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# HEATMAPS (one figure per data type)
# ============================================================

def build_heatmap_matrix(values_by_period, periods):
    bins = make_bins()
    matrix = []

    for period in periods:
        values = values_by_period.get(period)

        if values is not None and len(values) > 0:
            counts, _ = np.histogram(values, bins=bins)
            hist = counts / (len(values) * np.diff(bins))
        else:
            hist = np.full(len(bins) - 1, np.nan)

        matrix.append(hist)

    if matrix:
        matrix = np.asarray(matrix)
    else:
        matrix = np.empty((0, len(bins) - 1))

    return bins, matrix


def strided_ticks(labels, max_labels=60):
    step = max(1, int(np.ceil(len(labels) / max_labels)))
    idx = np.arange(0, len(labels), step)
    return idx, [labels[i] for i in idx]


def plot_heatmaps(
    week_vals,
    month_vals,
    data_type,
    detector,
    working_point,
    branch_name,
    output_file,
):
    """
    week_vals[direction] / month_vals[direction] = dict period -> array,
    for ONE data type. Rows: forward / backward. Columns: weekly / monthly.
    The number of tracks of each period is written in the tick labels.
    """
    weeks = sorted({p for d in week_vals for p in week_vals[d]})
    months = sorted({p for d in month_vals for p in month_vals[d]})

    if not weeks or not months:
        return

    bins = make_bins()

    mats = {}
    vmax = 0.0

    for direction, _ in DIRECTIONS:
        _, mats[("w", direction)] = build_heatmap_matrix(
            week_vals[direction], weeks
        )
        _, mats[("m", direction)] = build_heatmap_matrix(
            month_vals[direction], months
        )

    for m in mats.values():
        finite = m[np.isfinite(m)]
        if len(finite) > 0:
            vmax = max(vmax, np.max(finite))

    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=2, ncols=3, width_ratios=[1, 1, 0.03]
    )

    cbar_ax = fig.add_subplot(gs[:, 2])
    mesh = None

    for r, (direction, dir_title) in enumerate(DIRECTIONS):
        for c, (kind, periods, vals_by_dir) in enumerate(
            (
                ("w", weeks, week_vals),
                ("m", months, month_vals),
            )
        ):
            ax = fig.add_subplot(gs[r, c])
            matrix = mats[(kind, direction)]

            x = np.arange(len(periods) + 1)

            mesh = ax.pcolormesh(
                x,
                bins,
                matrix.T,
                shading="auto",
                cmap="viridis",
                vmin=0,
                vmax=vmax,
            )

            if kind == "w":
                names = list(periods)
            else:
                names = [month_label(p) for p in periods]

            n_labels = [
                f"N={fmt_n(len(vals_by_dir[direction].get(p, [])))}"
                for p in periods
            ]

            idx, tick_labels = strided_ticks(names)

            # period names at the bottom
            ax.set_xticks(idx + 0.5)
            ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)

            # number of tracks per period on top of the plot (vertical)
            ax_top = ax.secondary_xaxis("top")
            ax_top.set_xticks(idx + 0.5)
            ax_top.set_xticklabels(
                [n_labels[i] for i in idx], rotation=90, fontsize=7
            )
            ax.set_xlim(0, len(periods))
            ax.set_ylim(CHI2_MIN, CHI2_MAX)
            ax.set_ylabel(r"$\chi^2$")

            period_name = "Weekly" if kind == "w" else "Monthly"
            ax.set_title(f"{period_name} - {dir_title}", fontsize=10)

    cbar = fig.colorbar(mesh, cax=cbar_ax)
    cbar.set_label("Density")

    fig.suptitle(
        f"{make_title(branch_name, detector, working_point)} - {data_type}"
    )

    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# WEEKLY QUANTILES (real time axis + median uncertainty)
# ============================================================

def median_ci(sorted_values, z=1.0):
    """
    Distribution-free confidence interval of the median from order
    statistics (z=1 -> ~68 %). Works for skewed chi2 distributions.
    """
    n = len(sorted_values)
    half = z * np.sqrt(n) / 2.0

    lo_i = int(max(np.floor(n / 2.0 - half), 0))
    hi_i = int(min(np.ceil(n / 2.0 + half), n - 1))

    return sorted_values[lo_i], sorted_values[hi_i]


def week_quantile_table(values_by_week):
    """week -> dict(q10, q25, q50, q75, q90, lo, hi, n)"""
    table = {}

    for week, values in values_by_week.items():
        if len(values) == 0:
            continue

        s = np.sort(values)
        lo, hi = median_ci(s)

        q10, q25, q50, q75, q90 = np.percentile(
            s, [10, 25, 50, 75, 90]
        )

        table[week] = {
            "q10": q10, "q25": q25, "q50": q50,
            "q75": q75, "q90": q90,
            "lo": lo, "hi": hi, "n": len(s),
        }

    return table


def plot_weekly_quantiles_comparison(
    week_vals,
    detector,
    working_point,
    branch_name,
    theta_label,
    output_file,
    mode,
    ylim=None,
    median_only=False,
):
    """
    week_vals[direction][data_type] = dict week -> array.

    ylim        : (ymin, ymax); default (CHI2_MIN, QUANTILE_YMAX)
    median_only : only median + 68% CI (no percentile bands)
    x axis is the real calendar week: missing weeks leave a gap and
    lines are not connected across gaps.

    mode = "text": number of tracks written above median + error bar,
                   no right y-axis, no bars.
    mode = "bars": number of tracks as bars on the right y-axis,
                   no text.
    """
    if mode not in ("text", "bars"):
        raise ValueError(f"unknown mode: {mode}")

    if ylim is None:
        ylim = (CHI2_MIN, QUANTILE_YMAX)

    all_weeks = sorted(
        {
            w
            for d in week_vals
            for t in week_vals[d]
            for w in week_vals[d][t]
        }
    )

    if not all_weeks:
        return

    dates = {w: week_to_date(w) for w in all_weeks}
    d0 = min(dates.values())
    idx = {w: (dates[w] - d0).days // 7 for w in all_weeks}
    n_grid = max(idx.values()) + 1

    fig, axes = plt.subplots(
        2, 1, figsize=(15, 11), sharex=True, sharey=True
    )

    types_present = [
        t for t in ("VES", "FS")
        if any(week_vals[d].get(t) for d in week_vals)
    ]

    for ax, (direction, dir_title) in zip(axes, DIRECTIONS):
        ax_n = ax.twinx() if mode == "bars" else None
        n_max = 1

        for k, data_type in enumerate(types_present):
            table = week_quantile_table(
                week_vals[direction].get(data_type, {})
            )

            if not table:
                continue

            color = TYPE_COLORS[data_type]

            grid = {
                key: np.full(n_grid, np.nan)
                for key in ("q10", "q25", "q50", "q75", "q90", "lo", "hi")
            }

            xs, ns, his = [], [], []

            for week, row in table.items():
                i = idx[week]
                for key in grid:
                    grid[key][i] = row[key]
                xs.append(i)
                ns.append(row["n"])
                his.append(row["hi"])

            x = np.arange(n_grid)
            xs = np.asarray(xs)
            ns = np.asarray(ns)
            n_max = max(n_max, ns.max())

            if not median_only:
                ax.fill_between(
                    x, grid["q10"], grid["q90"],
                    color=color, alpha=0.12,
                    label=f"{data_type} 10-90 percentile",
                )
                ax.fill_between(
                    x, grid["q25"], grid["q75"],
                    color=color, alpha=0.25,
                    label=f"{data_type} 25-75 percentile",
                )
            ax.plot(
                x, grid["q50"],
                color=color,
                marker=TYPE_MARKERS[data_type],
                linewidth=1.5,
                markersize=3,
                label=f"{data_type} median (±68% CI)",
            )

            valid = np.isfinite(grid["q50"])
            med = grid["q50"][valid]

            ax.errorbar(
                x[valid],
                med,
                yerr=[
                    med - grid["lo"][valid],
                    grid["hi"][valid] - med,
                ],
                fmt="none",
                ecolor=color,
                elinewidth=1.2,
                capsize=2,
            )

            if mode == "bars":
                offset = (k - (len(types_present) - 1) / 2) * 0.4
                ax_n.bar(
                    xs + offset, ns, width=0.4,
                    color=color, alpha=0.30,
                )
            else:
                # number of tracks above the median + error bar
                for xi, ni, hi in zip(xs, ns, his):
                    if not np.isfinite(hi) or hi > ylim[1]:
                        continue
                    ax.annotate(
                        fmt_n(ni),
                        xy=(xi, hi),
                        xytext=(0, 4),
                        textcoords="offset points",
                        rotation=90,
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        color=color,
                    )

        if mode == "bars":
            ax_n.set_ylim(0, n_max * 5)
            ax_n.set_ylabel("Tracks per week")

            # keep the quantile curves in front of the bars
            ax.set_zorder(ax_n.get_zorder() + 1)
            ax.patch.set_visible(False)

        ax.set_ylim(*ylim)
        ax.set_ylabel(r"$\chi^2$")
        ax.set_title(dir_title, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")

        if ax.get_legend_handles_labels()[0]:
            ax.legend(ncol=2, loc="upper right", fontsize=8)

    step = max(1, int(np.ceil(n_grid / 25)))
    ticks = np.arange(0, n_grid, step)

    tick_labels = []
    for i in ticks:
        y, w, _ = (d0 + timedelta(weeks=int(i))).isocalendar()
        tick_labels.append(f"{y}-W{w:02d}")

    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels(tick_labels, rotation=45, ha="right")
    axes[-1].set_xlim(-1, n_grid)
    axes[-1].set_xlabel("ISO week")

    fig.suptitle(
        make_title(branch_name, detector, working_point, theta_label)
    )

    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# PER-RUN STATISTICS
# ============================================================

def calculate_run_statistics(run_values):
    runs = sorted(run_values)

    keys = ("mean", "median", "max")
    out = {k: [] for k in keys}

    for run in runs:
        values = run_values[run]

        if values is None or len(values) == 0:
            for k in keys:
                out[k].append(np.nan)
            continue

        out["mean"].append(np.mean(values))
        out["median"].append(np.median(values))
        out["max"].append(np.max(values))

    result = {k: np.asarray(v) for k, v in out.items()}
    result["runs"] = np.asarray(runs)

    return result


def build_series(records, short_name, direction=None):
    """
    One entry per (data type, WP) with per-run statistics.
    direction = "fwd" / "bwd" / None (all tracks).
    Uses all tracks reconstructed in xy AND xz (no chi2 upper cut).
    """
    fit, plane = short_name.split("_")
    grouped = {}

    for rec in records:
        vals = select_values(rec["fit"][fit], plane, direction=direction)
        if len(vals) == 0:
            continue
        grouped.setdefault(
            (rec["data_type"], rec["wp"]), {}
        )[rec["run"]] = vals

    series = []
    for (data_type, wp), run_values in sorted(
        grouped.items(), key=lambda kv: (kv[0][1], kv[0][0])
    ):
        series.append(
            {
                "data_type": data_type,
                "wp": wp,
                "stats": calculate_run_statistics(run_values),
            }
        )

    return series


# ============================================================
# SECOND X AXIS: MONTH OR ISO WEEK (ticks only, no axis label)
# ============================================================

def add_period_axis(ax, run_info, period_type):
    if not run_info:
        return None

    xlo, xhi = ax.get_xlim()
    runs = sorted(r for r in run_info if xlo <= r <= xhi)

    if not runs:
        return None

    period_to_runs = {}

    for run in runs:
        timestamp = run_info[run]["timestamp"]
        period = (
            month_key(timestamp)
            if period_type == "month"
            else week_key(timestamp)
        )
        period_to_runs.setdefault(period, []).append(run)

    positions, labels = [], []

    for period, period_runs in period_to_runs.items():
        positions.append(np.mean(period_runs))
        labels.append(
            month_label(period) if period_type == "month" else period
        )

    max_labels = 20

    if len(positions) > max_labels:
        indices = np.unique(
            np.linspace(0, len(positions) - 1, max_labels)
            .round()
            .astype(int)
        )
        positions = [positions[i] for i in indices]
        labels = [labels[i] for i in indices]

    ax_top = ax.secondary_xaxis("top")
    ax_top.set_xticks(positions)
    ax_top.set_xticklabels(labels, rotation=45, ha="left", fontsize=8)

    return ax_top


# ============================================================
# LEGEND (only what is plotted)
# ============================================================

def add_wp_type_legend(ax, wp_colors, wps, data_types, extra_handles=None):
    handles = []

    for wp in sorted(wps):
        handles.append(
            Line2D(
                [0], [0],
                color=wp_colors[wp],
                marker="o",
                linestyle="-",
                markersize=6,
                linewidth=1.5,
                label=f"WP{wp}",
            )
        )

    for data_type in ("VES", "FS"):
        if data_type not in data_types:
            continue

        handles.append(
            Line2D(
                [0], [0],
                color="black",
                marker=TYPE_MARKERS[data_type],
                linestyle="None",
                markersize=7,
                label=data_type,
            )
        )

    if extra_handles:
        handles.extend(extra_handles)

    if handles:
        ax.legend(handles=handles, ncol=2, fontsize=8, loc="best")


# ============================================================
# PER-RUN MEAN / MEDIAN / MAX (forward top, backward bottom)
# ============================================================

def plot_run_statistic(
    series_by_dir,
    run_info,
    detector,
    branch_name,
    statistic,
    output_file,
    period_type,
    wp_colors,
    run_ranges,
    title_extra="",
    ylim=None,
):
    """series_by_dir[direction] = series list. Forward top, backward bottom."""
    fig, axes = plt.subplots(
        2, 1, figsize=(15, 12), sharex=True, sharey=True
    )

    plotted_wps, plotted_types = set(), set()
    all_runs, y_max_values = [], []

    for ax, (direction, dir_title) in zip(axes, DIRECTIONS):
        # light background: VES / FS run ranges
        for dtype, ranges in run_ranges.items():
            for lo, hi in ranges:
                ax.axvspan(
                    lo - 0.5, hi + 0.5,
                    color=TYPE_COLORS[dtype], alpha=0.08, linewidth=0,
                )

        for s in series_by_dir[direction]:
            runs = s["stats"]["runs"]
            y = s["stats"][statistic]
            valid = np.isfinite(y)
            if len(runs) == 0 or not np.any(valid):
                continue

            ax.plot(
                runs[valid], y[valid],
                linestyle="None",
                marker=TYPE_MARKERS[s["data_type"]],
                markersize=RUN_MARKER_SIZE,
                color=wp_colors[s["wp"]],
                alpha=0.9,
            )
            plotted_wps.add(s["wp"])
            plotted_types.add(s["data_type"])
            all_runs.extend(runs[valid])
            y_max_values.append(np.max(y[valid]))

        ax.set_ylabel(r"$\chi^2$")
        ax.set_title(dir_title, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")

    if not plotted_wps:
        plt.close(fig)
        return

    xmin, xmax = min(all_runs), max(all_runs)
    xr = max(1, xmax - xmin)
    axes[0].set_xlim(xmin - 0.02 * xr, xmax + 0.02 * xr)  # shared x

    if ylim is not None:
        axes[0].set_ylim(*ylim)                            # shared y
    elif statistic == "max":
        top = max(y_max_values)
        axes[0].set_ylim(CHI2_MIN, top * 1.08 if top > 0 else 1.0)
    else:
        axes[0].set_ylim(bottom=CHI2_MIN)

    axes[-1].set_xlabel("Run number")

    labels = {"mean": "Mean", "median": "Median", "max": "Maximum"}
    run_label = f" [{RUN_LABEL}]" if RUN_LABEL else ""
    fig.suptitle(
        f"{branch_name} - {detector} - "
        f"{labels[statistic]} per run{run_label}{title_extra}"
    )

    bg_handles = [
        Patch(facecolor=TYPE_COLORS[t], alpha=0.15, label=f"{t} runs")
        for t in ("VES", "FS") if t in run_ranges and run_ranges[t]
    ]

    add_period_axis(axes[0], run_info, period_type)
    add_wp_type_legend(
        axes[0], wp_colors, plotted_wps, plotted_types,
        extra_handles=bg_handles,
    )

    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_median_per_run_dac(
    records,
    run_info,
    detector,
    short_name,
    branch_name,
    plane,
    output_file,
    period_type,
):
    """Plot one median-per-run series per plane, colored by WP and DAC10."""
    fit, projection = short_name.split("_")
    points = {}
    for rec in records:
        if rec["wp"] not in PLOT_WPS:
            continue
        values = rec["fit"][fit][projection]
        dac = run_info.get(rec["run"], {}).get("dac")
        if dac is None or plane not in dac or not np.isfinite(dac[plane]):
            continue
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        key = (rec["wp"], float(dac[plane]), rec["data_type"])
        points.setdefault(key, ([], []))
        points[key][0].append(rec["run"])
        points[key][1].append(float(np.median(values)))

    if not points:
        return

    combos = sorted({(wp, dac) for wp, dac, _ in points})
    cmap = plt.get_cmap("tab20" if len(combos) <= 20 else "turbo")
    color_map = {
        combo: cmap(i / max(1, len(combos) - 1))
        for i, combo in enumerate(combos)
    }
    fig, ax = plt.subplots(figsize=(15, 7))
    all_runs = []
    y_values = []
    for (wp, dac, data_type), (runs, medians) in points.items():
        order = np.argsort(runs)
        ax.plot(
            np.asarray(runs)[order], np.asarray(medians)[order],
            linestyle="None", marker=TYPE_MARKERS[data_type],
            markersize=RUN_MARKER_SIZE + 1,
            color=color_map[(wp, dac)], alpha=0.9,
        )
        all_runs.extend(runs)
        y_values.extend(medians)

    xmin, xmax = min(all_runs), max(all_runs)
    run_span = max(1, xmax - xmin)
    ax.set_xlim(xmin - 0.02 * run_span, xmax + 0.02 * run_span)
    ax.set_ylim(bottom=CHI2_MIN)
    ax.set_xlabel("Run number")
    ax.set_ylabel(r"$\chi^2$")
    run_label = f" [{RUN_LABEL}]" if RUN_LABEL else ""
    ax.set_title(
        f"{branch_name} - {detector} - Median per run - "
        f"{plane} DAC10 setting{run_label}"
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    add_period_axis(ax, run_info, period_type)

    handles = [
        Line2D(
            [0], [0], color=color_map[(wp, dac)], marker="o",
            linestyle="None", markersize=6,
            label=f"WP{wp}, DAC10={dac:g}",
        )
        for wp, dac in combos
    ]
    handles.extend(
        Line2D(
            [0], [0], color="black", marker=TYPE_MARKERS[data_type],
            linestyle="None", markersize=7, label=data_type,
        )
        for data_type in ("VES", "FS")
    )
    ax.legend(handles=handles, loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# READING RECORDS FROM ROOT FILES AND SLOW CONTROL
# ============================================================

def read_all_records(detector, data_dir, run_ranges, args):
    """Reads ALL runs in run_ranges (no good-run selection here)."""
    records = []
    run_info = {}

    for data_type in ("VES", "FS"):
        print()
        print(f"========== {detector} / {data_type} ==========")

        for run in iter_run_numbers(run_ranges[data_type]):
            file_path = Path(data_dir) / FILENAME_TEMPLATE.format(run=run)

            if not file_path.is_file():
                print(f"Run {run}: file not found, skipping")
                continue

            print(f"Run {run}: {file_path.name}")

            slowcontrol_info = get_slowcontrol_info(
                SLOWCONTROL_DIR, detector, run
            )

            if slowcontrol_info is None:
                print("  [warn] no timestamp/WP, skipping run")
                continue

            timestamp, working_point, dac_per_plane = slowcontrol_info
            month = month_key(timestamp)
            week = week_key(timestamp)

            print(
                f"  {timestamp.strftime('%d-%m-%Y %H:%M:%S')} | "
                f"{month} | {week} | WP{working_point}"
            )

            arrays = read_run(file_path)

            if arrays is None:
                continue

            run_info[run] = {
                "timestamp": timestamp,
                "month": month,
                "week": week,
                "working_point": working_point,
                "data_type": data_type,
                "dac": dac_per_plane,
            }

            fit_records = {
                fit: build_fit_record(arrays, fit, args.angle_unit)
                for fit in FITS
            }

            records.append(
                {
                    "run": run,
                    "data_type": data_type,
                    "wp": working_point,
                    "timestamp": timestamp,
                    "month": month,
                    "week": week,
                    "fit": fit_records,
                    "dac": dac_per_plane,
                }
            )

            for fit in FITS:
                print(
                    f"    {fit}: {len(fit_records[fit]['xy']):,} "
                    "tracks with xy>0 and xz>0"
                )

    return records, run_info


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Plot MURAVES chi2 distributions and per-run statistics."
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
        "--ves-ranges", nargs="+", metavar="START-END",
        help="Override default VES run ranges",
    )

    parser.add_argument(
        "--fs-ranges", nargs="+", metavar="START-END",
        help="Override default FS run ranges",
    )

    parser.add_argument(
        "--chi2-max", type=float, default=CHI2_MAX,
        help=f"Maximum chi2 for distribution plots, default={CHI2_MAX}",
    )

    parser.add_argument(
        "--bin-width", type=float, default=BIN_WIDTH,
        help=f"Histogram bin width, default={BIN_WIDTH}",
    )

    parser.add_argument(
        "--period-axis", choices=["month", "week"], default="month",
        help="Period shown on the second x-axis (default: month)",
    )

    parser.add_argument(
        "--angle-unit", choices=["auto", "deg", "rad"], default=ANGLE_UNIT,
        help="Unit of the theta/phi branches (default: auto-detect)",
    )

    parser.add_argument(
        "--no-theta", action="store_true",
        help="Skip all theta-range plots (faster)",
    )

    parser.add_argument(
        "--theta-min", type=float, default=None,
        help="Keep tracks with theta >= this angle in degrees",
    )
    parser.add_argument(
        "--theta-max", type=float, default=None,
        help="Keep tracks with theta <= this angle in degrees",
    )
    parser.add_argument(
        "--phi-min", type=float, default=None,
        help="Keep tracks with phi >= this angle in degrees (wrap-around supported)",
    )
    parser.add_argument(
        "--phi-max", type=float, default=None,
        help="Keep tracks with phi <= this angle in degrees (wrap-around supported)",
    )

    parser.add_argument(
        "--run-selection", choices=["all", "good", "both"], default="both",
        help=(
            "Plot all runs, only good runs (select_runs), or both into "
            "separate subdirectories (default: both)"
        ),
    )

    parser.add_argument(
        "--only-median-per-run-dac",
        action="store_true",
        help=(
            "Only create the DAC10-colored median-per-run plots; skip "
            "histograms, heatmaps, and weekly plots"
        ),
    )

    parser.add_argument(
        "--cache-dir", default=str(CACHE_DIR),
        help="Directory with/for the compact data cache (default: %(default)s)",
    )

    parser.add_argument(
        "--rebuild-cache", action="store_true",
        help="Ignore an existing cache and re-read the ROOT files",
    )

    args = parser.parse_args()

    if args.detector and args.detectors:
        parser.error("Use either --detector or --detectors, not both.")

    if args.detector:
        detectors = [args.detector]
    elif args.detectors:
        detectors = args.detectors
    else:
        detectors = ["ROSSO"]

    if args.chi2_max <= CHI2_MIN:
        parser.error("--chi2-max must be > 0")

    if args.bin_width <= 0:
        parser.error("--bin-width must be > 0")

    if args.theta_min is not None and args.theta_max is not None:
        if args.theta_max < args.theta_min:
            parser.error("--theta-max must be >= --theta-min")
    for name in ("phi_min", "phi_max"):
        value = getattr(args, name)
        if value is not None and not 0 <= value <= 360:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 360")

    return (
        args,
        detectors,
        parse_range_list(args.ves_ranges),
        parse_range_list(args.fs_ranges),
    )


# ============================================================
# ALL PLOTS FOR ONE SET OF RECORDS
# ============================================================

def make_plots(
    records,
    run_info,
    detector,
    run_ranges,
    args,
    out_base,
    wp_colors,
    run_label="",
):
    """Makes every plot for the given records into out_base/<plot type>/."""
    global RUN_LABEL
    RUN_LABEL = run_label

    if args.only_median_per_run_dac:
        per_run_dir = out_base / "per_run"
        per_run_dir.mkdir(parents=True, exist_ok=True)
        for short_name, branch_name in CHI2_BRANCHES.items():
            fit, projection = short_name.split("_")
            planes = [
                plane for plane in PLANES
                if plane[0] == ("Y" if projection == "xy" else "Z")
                and int(plane[1]) <= (3 if fit == "3p" else 4)
            ]
            for plane in planes:
                output_file = (
                    per_run_dir
                    / f"chi2_median_per_run_{short_name}_{plane}"
                      "_DAC10_VES_FS.png"
                )
                plot_median_per_run_dac(
                    records, run_info, detector, short_name, branch_name,
                    plane, output_file, args.period_axis,
                )
        print(f"Creating DAC10 median-per-run plots only for {detector} -> {out_base}")
        return

    dirs = {
        "hist": out_base / "hist",
        "hist_err": out_base / "hist" / "errorbars",
        "monthly": out_base / "monthly_hists",
        "heatmap": out_base / "heatmaps",
        "quant": out_base / "quantiles_weekly",
        "quant_theta": out_base / "quantiles_weekly" / "theta_ranges",
        "quant_median": out_base / "quantiles_weekly" / "median_only",
        "quant_median_theta": (
            out_base / "quantiles_weekly" / "median_only" / "theta_ranges"
        ),
        "per_run": out_base / "per_run",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    working_points = sorted({rec["wp"] for rec in records})

    # ====================================================
    # DISTRIBUTION PLOTS
    # ====================================================

    print()
    print(f"Creating distribution plots for {detector} -> {out_base}")

    for working_point in working_points:
        for short_name, branch_name in CHI2_BRANCHES.items():
            fit, plane = short_name.split("_")

            # ---------------- theta = all ----------------
            week_vals, month_vals, alltime = build_values(
                records, working_point, fit, plane, None
            )

            if not has_entries(alltime):
                print(
                    f"  WP{working_point} {short_name}: "
                    "no entries, skipping"
                )
                continue

            tag = f"{short_name}_WP{working_point}"

            # 1. all-time histogram: without / with error bars
            for errorbars in (False, True):
                suffix = "_errbars" if errorbars else ""
                hist_dir = dirs["hist_err"] if errorbars else dirs["hist"]

                plot_distribution_comparison(
                    alltime,
                    detector,
                    working_point,
                    branch_name,
                    hist_dir
                    / f"chi2_hist_{tag}_thetaall_VES_FS{suffix}.png",
                    errorbars,
                )

            # 2. monthly histograms (theta all)
            plot_monthly_histograms(
                month_vals,
                detector,
                working_point,
                branch_name,
                dirs["monthly"] / f"chi2_monthly_hists_{tag}_VES_FS.png",
            )

            # 3. heatmaps, separately for VES and FS (theta all)
            for data_type in ("VES", "FS"):
                if not any(
                    week_vals[d][data_type] for d, _ in DIRECTIONS
                ):
                    continue

                plot_heatmaps(
                    {d: week_vals[d][data_type] for d, _ in DIRECTIONS},
                    {d: month_vals[d][data_type] for d, _ in DIRECTIONS},
                    data_type,
                    detector,
                    working_point,
                    branch_name,
                    dirs["heatmap"]
                    / f"chi2_heatmap_weekly_monthly_{tag}_{data_type}.png",
                )

            # 4. weekly quantiles (theta all), number of tracks as text
            plot_weekly_quantiles_comparison(
                week_vals,
                detector,
                working_point,
                branch_name,
                "all",
                dirs["quant"]
                / f"chi2_quantiles_weekly_{tag}_thetaall_VES_FS_text.png",
                "text",
            )

            # 4b. weekly median only (no percentile bands)
            plot_weekly_quantiles_comparison(
                week_vals,
                detector,
                working_point,
                branch_name,
                "all",
                dirs["quant_median"]
                / f"chi2_median_weekly_{tag}_thetaall_VES_FS_text.png",
                "text",
                ylim=WEEKLY_MEDIAN_RANGE[fit],
                median_only=True,
            )

            if args.no_theta:
                continue

            # ---------------- theta ranges ----------------
            theta_alltime = {d: {} for d, _ in DIRECTIONS}

            for theta_label, lo, hi in THETA_RANGES:
                t_week, _, t_all = build_values(
                    records, working_point, fit, plane, (lo, hi)
                )

                for direction, _ in DIRECTIONS:
                    theta_alltime[direction][theta_label] = t_all[direction]

                if not has_entries(t_all):
                    print(
                        f"  WP{working_point} {short_name} "
                        f"theta {theta_label}: no entries, skipping"
                    )
                    continue

                theta_tag = "theta" + theta_label.replace("+", "plus")

                # weekly quantiles per theta range
                plot_weekly_quantiles_comparison(
                    t_week,
                    detector,
                    working_point,
                    branch_name,
                    theta_label,
                    dirs["quant_theta"]
                    / f"chi2_quantiles_weekly_{tag}_{theta_tag}"
                      "_VES_FS_text.png",
                    "text",
                )

                # weekly median only per theta range
                plot_weekly_quantiles_comparison(
                    t_week,
                    detector,
                    working_point,
                    branch_name,
                    theta_label,
                    dirs["quant_median_theta"]
                    / f"chi2_median_weekly_{tag}_{theta_tag}"
                      "_VES_FS_text.png",
                    "text",
                    ylim=WEEKLY_MEDIAN_RANGE[fit],
                    median_only=True,
                )

            # all theta ranges on ONE histogram figure
            for errorbars in (False, True):
                suffix = "_errbars" if errorbars else ""
                hist_dir = dirs["hist_err"] if errorbars else dirs["hist"]

                plot_theta_distribution_comparison(
                    theta_alltime,
                    detector,
                    working_point,
                    branch_name,
                    hist_dir
                    / f"chi2_hist_{tag}_thetaranges_VES_FS{suffix}.png",
                    errorbars,
                )

    # ====================================================
    # PER-RUN PLOTS (mean / median / max, forward + backward)
    # ====================================================

    print()
    print(f"Creating per-run statistics for {detector} -> {out_base}")

    for short_name, branch_name in CHI2_BRANCHES.items():
        fit = short_name.split("_")[0]
        zoom = MEDIAN_ZOOM[fit]

        series_by_dir = {}
        for direction, _ in DIRECTIONS:
            series_by_dir[direction] = [
                s for s in build_series(records, short_name, direction)
                if s["wp"] in PLOT_WPS
            ]

        if not any(series_by_dir.values()):
            continue

        for statistic in ("mean", "median", "max"):
            plot_run_statistic(
                series_by_dir,
                run_info,
                detector,
                branch_name,
                statistic,
                dirs["per_run"]
                / f"chi2_{statistic}_per_run_{short_name}_VES_FS.png",
                args.period_axis,
                wp_colors,
                run_ranges,
            )

        plot_run_statistic(
            series_by_dir,
            run_info,
            detector,
            branch_name,
            "median",
            dirs["per_run"]
            / f"chi2_median_per_run_{short_name}_VES_FS_zoom.png",
            args.period_axis,
            wp_colors,
            run_ranges,
            ylim=zoom,
        )

        print(f"  {short_name}: done")


# ============================================================
# MAIN
# ============================================================

def main():
    global CHI2_MAX
    global BIN_WIDTH
    global THETA_MIN_CUT, THETA_MAX_CUT, PHI_MIN_CUT, PHI_MAX_CUT

    (
        args,
        detectors,
        ves_override,
        fs_override,
    ) = parse_arguments()

    CHI2_MAX = args.chi2_max
    BIN_WIDTH = args.bin_width
    THETA_MIN_CUT, THETA_MAX_CUT = args.theta_min, args.theta_max
    PHI_MIN_CUT, PHI_MAX_CUT = args.phi_min, args.phi_max

    selections = (
        ["all", "good"] if args.run_selection == "both"
        else [args.run_selection]
    )

    print("=" * 70)
    print("MURAVES CHI2 DISTRIBUTIONS")
    print("=" * 70)
    print(f"Detectors   : {', '.join(detectors)}")
    print(f"Chi2 range  : {CHI2_MIN} -> {CHI2_MAX} (distribution plots)")
    print(f"Bin width   : {BIN_WIDTH}")
    print("Track selection: chi2_xy > 0 AND chi2_xz > 0")
    if any(v is not None for v in (
        THETA_MIN_CUT, THETA_MAX_CUT, PHI_MIN_CUT, PHI_MAX_CUT
    )):
        print(
            f"Angular cuts : theta=[{THETA_MIN_CUT}, {THETA_MAX_CUT}] deg, "
            f"phi=[{PHI_MIN_CUT}, {PHI_MAX_CUT}] deg"
        )
    print(f"Forward     : {PHI_FWD_MIN:.0f} <= phi <= {PHI_FWD_MAX:.0f} deg")
    print(
        "Theta       : all"
        + ("" if args.no_theta else f" + {[t[0] for t in THETA_RANGES]}")
    )
    print("WP = colour, VES = circle, FS = triangle")
    print(f"Period axis : {args.period_axis}")
    print(
        f"Cache dir   : {args.cache_dir}"
        + (" (rebuild)" if args.rebuild_cache else "")
    )
    print(f"Run selection: {', '.join(selections)}")
    print("=" * 70)

    for detector in detectors:
        print()
        print("#" * 70)
        print(f"DETECTOR: {detector}")
        print("#" * 70)

        data_dir = Path(DATA_DIR_TEMPLATE.format(detector=detector))

        detector_output = OUTPUT_DIR / detector
        detector_output.mkdir(parents=True, exist_ok=True)

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
        # READ DATA: from cache if available, otherwise from ROOT
        # ====================================================

        # ONE cache with all runs, independent of the run selection
        cfile = cache_path(args.cache_dir, detector, "")

        cache_is_usable = (
            cfile.is_file()
            and not args.rebuild_cache
            and (
                not args.only_median_per_run_dac
                or cache_has_dac(cfile)
            )
        )
        if cache_is_usable:
            print()
            print(f"Loading cached data: {cfile}")
            records, run_info = load_cache(cfile)
            print(f"  {len(records)} runs in cache")
        else:
            if (
                args.only_median_per_run_dac
                and cfile.is_file()
                and not cache_has_dac(cfile)
            ):
                print(
                    f"Cache {cfile} has no DAC10 values; rebuilding it "
                    "for the DAC10 plots."
                )
            records, run_info = read_all_records(
                detector, data_dir, run_ranges, args
            )
            if records:
                save_cache(cfile, records)

        # run-range selection applied AFTER the cache
        records = filter_records(records, run_ranges)

        if not records:
            print(f"No usable runs for {detector}, skipping.")
            continue

        # colours from ALL runs, so a WP has the same colour in the
        # all_runs and good_runs plots
        wp_colors = make_wp_color_map(records)

        good_run_set = (
            get_good_run_set(detector) if "good" in selections else None
        )

        # ====================================================
        # PLOTS, one subdirectory per selection
        # ====================================================

        for selection in selections:
            if selection == "good":
                sel_records = [
                    r for r in records if r["run"] in good_run_set
                ]
            else:
                sel_records = records

            print()
            print("-" * 70)
            print(f"{detector}: {selection} runs "
                  f"({len(sel_records)} runs)")
            print("-" * 70)

            if not sel_records:
                print(f"No runs for selection '{selection}', skipping.")
                continue

            sel_run_info = {
                r["run"]: run_info[r["run"]] for r in sel_records
            }

            make_plots(
                sel_records,
                sel_run_info,
                detector,
                run_ranges,
                args,
                detector_output / f"{selection}_runs",
                wp_colors,
                run_label=(
                    "good runs only" if selection == "good" else "all runs"
                ),
            )

        print()
        print(f"Output directory: {detector_output}")

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()