#!/usr/bin/env python3
"""
MURAVES cluster position diagnostics: histograms of hits vs cluster positions
in 1 mm binning, comparing unfiltered (all cluster positions/hits) vs. 
filtered (only hits associated to 3D tracks).
"""

from pathlib import Path
import argparse
import re
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import awkward as ak
import uproot

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR_TEMPLATE = (
    "/pnfs/iihe/muraves/muraves_DATA/"
    "RECONSTRUCTED/{detector}/v1"
)
SLOWCONTROL_DIR = "/pnfs/iihe/muraves/muraves_DATA/DECOMPRESSED"

CACHE_DIR = Path("./cluster_pos_cache")
OUTPUT_DIR = Path("./chi2_diagnostics/cluster_pos_analysis")

TREE_NAME = "AnalyzedData"
FILENAME_TEMPLATE = "MURAVES_AnalyzedData_run{run}.root"
SLOWCONTROL_TEMPLATE = "SLOWCONTROL_run{run}"

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

PLANES = ["Z1", "Z2", "Z3", "Z4", "Y1", "Y2", "Y3", "Y4"]
PLANE_INDEX = {p: i for i, p in enumerate(PLANES)}
FIT_PLANES = {
    "3p": ["Z1", "Z2", "Z3", "Y1", "Y2", "Y3"],
    "4p": ["Z1", "Z2", "Z3", "Z4", "Y1", "Y2", "Y3", "Y4"],
}
FITS = ("3p", "4p")
PROJ = {"Z": "xz", "Y": "xy"}

# 1 mm binning for cluster positions. 
# Assuming positions are typically in mm or cm; adjust range if needed (e.g., -500 to 500 mm).
# Change binning to match the true -0.5 to 0.5 scale of the raw data
#POS_EDGES = np.linspace(-0.6, 0.6, 121)  # 120 bins between -0.6 and 0.6
#NB_POS = len(POS_EDGES) - 1
#NKEY_POS = len(PLANES) * NB_POS
POS_EDGES = np.arange(-500.0, 501.0, 1.0)  # Exactly 1 mm binning from -500 to 500 mm
NB_POS = len(POS_EDGES) - 1
NKEY_POS = len(PLANES) * NB_POS


# ============================================================
# RUN / PERIOD PARSING
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
            raise argparse.ArgumentTypeError(f"Invalid run number: {value}") from exc
        return run, run
    try:
        start, stop = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid run range: {value}") from exc
    if stop < start:
        raise argparse.ArgumentTypeError(f"Invalid run range: {value}: end must be >= start.")
    return start, stop


def parse_range_list(values):
    if values is None:
        return None
    return [parse_run_range(v) for v in values]


def iter_run_numbers(run_ranges):
    for start, stop in run_ranges:
        yield from range(start, stop + 1)


DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_period(text):
    if "=" not in text:
        raise argparse.ArgumentTypeError(
            f"Period must be 'label=START-END[,START-END]' or "
            f"'label=YYYY-MM-DD:YYYY-MM-DD', got: {text}"
        )
    label, spec = text.split("=", 1)
    label = re.sub(r"[^\w.\-]+", "_", label.strip())
    runs, dates = [], []
    for part in spec.split(","):
        part = part.strip()
        if ":" in part and all(DATE_RE.match(x) for x in part.split(":", 1)):
            d0, d1 = part.split(":", 1)
            dates.append((datetime.strptime(d0, "%Y-%m-%d"),
                          datetime.strptime(d1, "%Y-%m-%d").replace(hour=23, minute=59, second=59)))
        else:
            runs.append(parse_run_range(part))
    if not runs and not dates:
        raise argparse.ArgumentTypeError(f"Empty period: {text}")
    desc = ", ".join(
        [f"runs {a}-{b}" for a, b in runs]
        + [f"{a:%Y-%m-%d} to {b:%Y-%m-%d}" for a, b in dates]
    )
    return {"label": label, "runs": runs, "dates": dates, "desc": desc}


def run_in_period(run, timestamp, period):
    if any(lo <= run <= hi for lo, hi in period["runs"]):
        return True
    return any(lo <= timestamp <= hi for lo, hi in period["dates"])


# ============================================================
# SLOW CONTROL
# ============================================================

def parse_slowcontrol_timestamp(fields):
    if len(fields) <= 1:
        return None
    raw = fields[1].strip()
    candidates = [raw, raw.replace("T", " "), raw.replace("/", "-")]
    formats = [
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S",
        "%d-%m-%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S",
        "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y",
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
    slowcontrol_file = Path(slowcontrol_dir) / detector / SLOWCONTROL_TEMPLATE.format(run=run)
    if not slowcontrol_file.is_file():
        return None
    try:
        with open(slowcontrol_file, "r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fields = line.split()
                if len(fields) <= 5:
                    return None
                timestamp = parse_slowcontrol_timestamp(fields)
                if timestamp is None:
                    return None
                try:
                    wp = float(fields[5])
                except ValueError:
                    return None
                if not np.isfinite(wp) or not wp.is_integer():
                    return None
                return timestamp, int(wp)
    except Exception:
        return None
    return None


def get_good_run_set(detector):
    import select_runs as sr
    print(f"\nSelecting good runs for {detector}...")
    df = sr.load_parsing_overview(detector)
    ped = sr.load_pedestal_outliers(detector)
    mon = sr.load_monitoring_scan(detector)
    good_runs, *_ = sr.select_runs(
        df, ped, mon, run_type="ADC", trigger_rate_n_sigma=3.0, label=f"[{detector}]",
    )
    good_runs = np.asarray(good_runs, dtype=int)
    print(f"  Good runs: {len(good_runs)}")
    return set(good_runs.tolist())


# ============================================================
# ROOT -> CLUSTER POSITION HISTOGRAMS
# ============================================================

def needed_branches():
    b = []
    for p in ("xy", "xz"):
        b += [f"BestTrack_3p_ChiSquare_{p}", f"BestTrack_4p_ChiSquare_{p}",
              f"BestTrack_3p_{p}_index", f"Track_3p_of_4p_index_{p}",
              f"chiSquare_4p_{p}", f"cluster_c4_index_{p}"]
    for k in (1, 2, 3):
        b += [f"TrackCluster_z{k}_index", f"TrackCluster_y{k}_index"]
    b += [f"ClusterPosition_{p}" for p in PLANES]
    b += [f"ClusterSize_{p}" for p in PLANES]  # <--- Added ClusterSize branches
    return b


def _pick(jag, idx):
    mask = ak.local_index(jag, axis=1) == idx[:, None]
    out = ak.fill_none(ak.firsts(jag[mask], axis=1), -1)
    return ak.to_numpy(out).astype(np.int64)


def _pick_match(values, flat_chi, best):
    mask = flat_chi == best[:, None]
    out = ak.fill_none(ak.firsts(values[mask], axis=1), -1)
    return ak.to_numpy(out).astype(np.int64)


def _fill_pos_histogram(counts, plane_idx, positions):
    ok = np.isfinite(positions)
    pos = positions[ok]
    b = np.clip(np.searchsorted(POS_EDGES, pos, side="right") - 1, 0, NB_POS - 1)
    key = plane_idx * NB_POS + b
    counts += np.bincount(key, minlength=NKEY_POS).astype(np.int64)


def extract_run_cluster_data(arr):
    """Extracts counts for unfiltered and filtered (3p and 4p) cluster positions."""
    counts_unfiltered = np.zeros(NKEY_POS, dtype=np.int64)
    for i, plane in enumerate(PLANES):
        pos_branch = f"ClusterPosition_{plane}"
        size_branch = f"ClusterSize_{plane}"
        if pos_branch in arr.fields:
            pos_jag = arr[pos_branch]
            
            if size_branch in arr.fields:
                size_jag = arr[size_branch]
                valid_clusters = (size_jag > 0)
                pos_jag = pos_jag[valid_clusters]
            
            # Convert normalized coordinates (-0.5 to 0.5) to millimeters (~ -500 to 500 mm)
            pos = ak.to_numpy(ak.flatten(pos_jag)).astype(float) * 1000.0
            pos = pos[np.isfinite(pos)]
            _fill_pos_histogram(counts_unfiltered, i, pos)

    counts_filtered_3p = np.zeros(NKEY_POS, dtype=np.int64)
    counts_filtered_4p = np.zeros(NKEY_POS, dtype=np.int64)

    for fit in FITS:
        chi_xz = ak.to_numpy(arr[f"BestTrack_{fit}_ChiSquare_xz"]).astype(float)
        chi_xy = ak.to_numpy(arr[f"BestTrack_{fit}_ChiSquare_xy"]).astype(float)
        sel = np.isfinite(chi_xz) & np.isfinite(chi_xy) & (chi_xz > 0) & (chi_xy > 0)
        if sel.sum() == 0:
            continue

        a = arr[sel]
        if fit == "3p":
            t3 = {p: ak.to_numpy(a[f"BestTrack_3p_{p}_index"]).astype(np.int64) for p in ("xz", "xy")}
        else:
            t3 = {p: ak.to_numpy(a[f"Track_3p_of_4p_index_{p}"]).astype(np.int64) for p in ("xz", "xy")}

        target_counts = counts_filtered_3p if fit == "3p" else counts_filtered_4p

        for letter, proj in PROJ.items():
            for k in (1, 2, 3):
                p_name = f"{letter}{k}"
                cl_idx = _pick(a[f"TrackCluster_{letter.lower()}{k}_index"], t3[proj])
                pos_jag = a[f"ClusterPosition_{p_name}"]
                valid_ev = cl_idx >= 0
                if valid_ev.any():
                    matched_pos = ak.to_numpy(pos_jag[valid_ev, cl_idx[valid_ev]]).astype(float) * 1000.0
                    _fill_pos_histogram(target_counts, PLANE_INDEX[p_name], matched_pos)
            
            if fit == "4p":
                p_name = f"{letter}4"
                cl_idx = _pick_match(a[f"cluster_c4_index_{proj}"], a[f"chiSquare_4p_{proj}"], chi_xz[sel])
                pos_jag = a[f"ClusterPosition_{p_name}"]
                valid_ev = cl_idx >= 0
                if valid_ev.any():
                    matched_pos = ak.to_numpy(pos_jag[valid_ev, cl_idx[valid_ev]]).astype(float) * 1000.0
                    _fill_pos_histogram(target_counts, PLANE_INDEX[p_name], matched_pos)

    nz_unf = np.flatnonzero(counts_unfiltered)
    nz_3p = np.flatnonzero(counts_filtered_3p)
    nz_4p = np.flatnonzero(counts_filtered_4p)

    return {
        "unfiltered": (nz_unf.astype(np.int32), counts_unfiltered[nz_unf].astype(np.int32)),
        "3p": (nz_3p.astype(np.int32), counts_filtered_3p[nz_3p].astype(np.int32)),
        "4p": (nz_4p.astype(np.int32), counts_filtered_4p[nz_4p].astype(np.int32)),
    }


def read_run(file_path):
    branches = needed_branches()
    try:
        with uproot.open(file_path) as f:
            if TREE_NAME not in f:
                return None
            tree = f[TREE_NAME]
            arr = tree.arrays([b for b in branches if b in tree.keys()], library="ak")
        return extract_run_cluster_data(arr)
    except Exception:
        return None


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def plot_cluster_position_histograms(results_unf, results_filt, fit_label, title, output_file):
    """Plots histogram of number of hits wrt cluster positions in 1 mm binning comparing unfiltered and filtered."""
    planes = FIT_PLANES[fit_label] if fit_label in FIT_PLANES else PLANES
    ncol = len(planes) // 2
    fig, axs = plt.subplots(2, ncol, figsize=(4.4 * ncol, 7.2), sharex=True, squeeze=False)
    centers = 0.5 * (POS_EDGES[:-1] + POS_EDGES[1:])

    for r, letter in enumerate("ZY"):
        pls = [p for p in planes if p[0] == letter]
        for c, plane in enumerate(pls):
            ax = axs[r, c]
            idx = PLANE_INDEX[plane]
            
            # Unfiltered histogram reconstruction
            h_unf = np.zeros(NB_POS, dtype=np.int64)
            keys_unf, cnts_unf = results_unf
            mask_unf = (keys_unf // NB_POS) == idx
            if mask_unf.any():
                bins = keys_unf[mask_unf] % NB_POS
                h_unf[bins] = cnts_unf[mask_unf]

            # Filtered histogram reconstruction
            h_filt = np.zeros(NB_POS, dtype=np.int64)
            keys_filt, cnts_filt = results_filt
            mask_filt = (keys_filt // NB_POS) == idx
            if mask_filt.any():
                bins = keys_filt[mask_filt] % NB_POS
                h_filt[bins] = cnts_filt[mask_filt]

            ax.step(centers, h_unf, where="mid", color="tab:blue", label="Unfiltered (All)")
            ax.step(centers, h_filt, where="mid", color="tab:orange", label=f"Filtered ({fit_label} tracks)")
            
            ax.set_title(f"Plane {plane}", fontsize=9)
            ax.grid(True, alpha=0.3, ls="--")
            ax.legend(fontsize=7, loc="upper right")
            if r == 1:
                ax.set_xlabel("Cluster Position (mm)")
            if c == 0:
                ax.set_ylabel("Number of Hits")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_file, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN SCRIPT EXECUTION ROUTINE
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Plot MURAVES cluster position histograms.")
    parser.add_argument("--detector", required=True, choices=list(RUN_RANGES.keys()))
    parser.add_argument("--runs", nargs="+", type=str, default=None)
    parser.add_argument("--good-runs", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    run_list = []
    if args.runs:
        for r_spec in parse_range_list(args.runs):
            run_list.extend(range(r_spec[0], r_spec[1] + 1))
    elif args.good_runs:
        good_set = get_good_run_set(args.detector)
        for dtype, ranges in RUN_RANGES[args.detector].items():
            for start, stop in ranges:
                for r in range(start, stop + 1):
                    if r in good_set:
                        run_list.append(r)
    else:
        # Default to all ranges
        for dtype, ranges in RUN_RANGES[args.detector].items():
            for start, stop in ranges:
                run_list.extend(range(start, stop + 1))

    print(f"Processing {len(run_list)} runs for {args.detector}...")

    total_unf_keys = np.array([], dtype=np.int32)
    total_unf_cnts = np.array([], dtype=np.int32)
    total_3p_keys = np.array([], dtype=np.int32)
    total_3p_cnts = np.array([], dtype=np.int32)
    total_4p_keys = np.array([], dtype=np.int32)
    total_4p_cnts = np.array([], dtype=np.int32)

    data_dir = Path(DATA_DIR_TEMPLATE.format(detector=args.detector))

    for run in run_list:
        file_path = data_dir / FILENAME_TEMPLATE.format(run=run)
        if not file_path.is_file():
            continue
        res = read_run(file_path)
        if res is None:
            continue
        
        # Simple accumulation helper
        def merge_hist(k1, c1, k2, c2):
            if len(k1) == 0:
                return k2, c2
            if len(k2) == 0:
                return k1, c1
            combined = {}
            for k, c in zip(k1, c1):
                combined[k] = combined.get(k, 0) + c
            for k, c in zip(k2, c2):
                combined[k] = combined.get(k, 0) + c
            keys = np.array(list(combined.keys()), dtype=np.int32)
            counts = np.array(list(combined.values()), dtype=np.int32)
            return keys, counts

        total_unf_keys, total_unf_cnts = merge_hist(total_unf_keys, total_unf_cnts, res["unfiltered"][0], res["unfiltered"][1])
        total_3p_keys, total_3p_cnts = merge_hist(total_3p_keys, total_3p_cnts, res["3p"][0], res["3p"][1])
        total_4p_keys, total_4p_cnts = merge_hist(total_4p_keys, total_4p_cnts, res["4p"][0], res["4p"][1])

    print("Generating plots...")
    plot_cluster_position_histograms(
        (total_unf_keys, total_unf_cnts), 
        (total_3p_keys, total_3p_cnts), 
        "3p", 
        f"{args.detector}: Cluster Positions (Unfiltered vs 3p Track Filtered)", 
        OUTPUT_DIR / f"{args.detector}_cluster_positions_3p.png"
    )
    plot_cluster_position_histograms(
        (total_unf_keys, total_unf_cnts), 
        (total_4p_keys, total_4p_cnts), 
        "4p", 
        f"{args.detector}: Cluster Positions (Unfiltered vs 4p Track Filtered)", 
        OUTPUT_DIR / f"{args.detector}_cluster_positions_4p.png"
    )
    print(f"Plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()