#!/usr/bin/env python3
"""
MURAVES per-strip chi2 / residual / hit analysis of 3D tracks, with excess finding.

For every 3D track (chi2_xz > 0 AND chi2_xy > 0) the strips of the clusters
that belong to the BEST track are collected per plane (Z1..Z4, Y1..Y4):

    3p : planes 1-3, chi2 = BestTrack_3p_ChiSquare_{xz|xy}
    4p : planes 1-4, chi2 = BestTrack_4p_ChiSquare_{xz|xy}
    (Z planes <-> xz track, Y planes <-> xy track;
     the 3p sample contains the 3p tracks of the 4p events as well)

Per (detector, selection, [period], WP, VES/FS, fit) it makes
  1) median chi2 per strip per plane (error bars = CI of the median), excess
     strips marked, with threshold and reference-only views
  2) median track RESIDUAL per strip per plane (error bars = CI of the median)
  3) hit distribution per strip per plane, excess strips marked, and a version
     where the bar colour is the median chi2 of the strip
  4) CSV tables: the list of strips with a chi2 excess, and the residual
     statistics of all strips.
and for two periods (or VES vs FS) comparison plots of the hit density, the
median chi2 (ratio B/A) and the median residual (difference B-A). Periods can
optionally be compared across VES/FS types with --compare-periods-any-type.

Directory layout
----------------
<OUTPUT_DIR>/<detector>/<all|good>_runs/full/WP<wp>/<VES|FS>/...
<OUTPUT_DIR>/<detector>/<all|good>_runs/periods/<label>/WP<wp>/<VES|FS>/...
(+ flagged_strips_summary.csv one level above the WP directories)
Period comparison plots are written below
<OUTPUT_DIR>/<detector>/<all|good>_runs/period_comparisons/.
They include per-WP comparisons and aggregate comparisons with all WPs combined.

Definition of an "excess" strip (high OR low) -- chi2 only
----------------------------------------------------------
Reference  M_ref = pooled median chi2 of all hits in the same plane
                   (--reference projection: pooled over all planes of the
                    same projection Z or Y of that fit).
f = 1 + --rel-excess (default 1.25).
Strip s is an excess HIGH when all of
  (a) N_s >= --min-hits (default 10; below that the median CI is useless)
  (b) lower edge of the z-sigma confidence interval of its median > M_ref
      (statistically significant; distribution-free CI from order
      statistics, default z = 3)
  (c) median_s > f * M_ref        (practically relevant)
  (d) optional (--min-robust-z): robust z-score of the strip median w.r.t. the
      other strips of the plane (MAD based) >= value
Strip s is an excess LOW when
  (a) same, (b) upper edge of the CI < M_ref, (c) median_s < M_ref / f,
  (d) robust z <= -value.
All strips with at least one hit are drawn; the table always contains the
robust z-score and the direction (high/low).

Residuals
---------
The residual of a plane is (cluster position - fitted track position) of the
cluster that belongs to the best track; every strip of that cluster gets the
residual of its cluster (the same way every strip gets the chi2 of the track).
ROOT branches:
    planes 1-3 : Residue_Track3p_{z|y}{1,2,3}[track]   (residual of the 3p fit,
                 indexed with the same 3p track as the strips; for the 4p
                 sample this is the 3p fit of the parent track, because the
                 file does not store residuals of the 4p fit)
    plane 4    : displacement_p4_{xz|xy}[candidate]    (plane-4 cluster minus
                 the extrapolation of the 3p track; flat over the 4p
                 candidates, so it is picked with the same chi2 match as the
                 plane-4 cluster)
Residuals are signed, so they are histogrammed in symmetric-log bins (100 per
decade of |residual|, from 1e-5 to 2, plus one central bin) and the same
distribution-free median CI as for chi2 is used. For two periods the residual
comparison shows the DIFFERENCE of the strip medians (B - A) and not the ratio
(the ratio of numbers scattering around zero is meaningless). The error bar on
the difference is the combination in quadrature of the two median errors, with
sigma_i = (ci_hi - ci_lo) / (2 z), drawn as +-z sigma.
Note: in a 3p fit of three points the three residuals are not independent
(r1 ~ r3 ~ -r2/2).

Data are stored per run as sparse per-strip histograms (chi2: log bins, 100 per
decade; residuals: symmetric-log bins) in a compact cache, so any run selection
/ period is just a sum of histograms. Medians come from the histograms
(interpolated inside the bin, ~2 % precision). A cache written by an older
version has no residual histograms: rerun once with --rebuild-cache.

Hit density vs strip POSITION
-----------------------------
In addition to the plots vs strip ID, hits_per_position_*.png shows the number
of hits per unit position (hits / strip pitch) vs StripsPosition_<plane>, with
the same hit selection (only strips of the clusters of the best track). For
every (plane, strip) the cache holds the sum of the positions and the number of
hits; the mean is the plotted position. Comparison plots
(hits_per_position_density_*.png) overlay two sets (unit area) with the ratio
B/A underneath. A cache written by an older version has no positions: rerun
once with --rebuild-cache.

Strip ID is what is plotted; "channel" in the electronics sense is a
different number (TriggerMaskChannels vs TriggerMaskStrips).
"""

from pathlib import Path
import argparse
import re
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import LinearSegmentedColormap
import awkward as ak
import uproot
import matplotlib.ticker


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR_TEMPLATE = (
    "/pnfs/iihe/muraves/muraves_DATA/"
    "RECONSTRUCTED/{detector}/v1"
)
SLOWCONTROL_DIR = "/pnfs/iihe/muraves/muraves_DATA/DECOMPRESSED"

CACHE_DIR = Path("./chi2_strip_cache")
OUTPUT_DIR = Path("./chi2_strip_analysis")

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

NSTRIP = 64
PLANES = ["Z1", "Z2", "Z3", "Z4", "Y1", "Y2", "Y3", "Y4"]
PLANE_INDEX = {p: i for i, p in enumerate(PLANES)}
FIT_PLANES = {
    "3p": ["Z1", "Z2", "Z3", "Y1", "Y2", "Y3"],
    "4p": ["Z1", "Z2", "Z3", "Z4", "Y1", "Y2", "Y3", "Y4"],
}
FITS = ("3p", "4p")
PROJ = {"Z": "xz", "Y": "xy"}   # plane letter -> track projection

# chi2 histogram: log bins, 100 per decade (2.3 % wide)
CHI2_EDGES = np.logspace(-4, 5, 9 * 100 + 1)
NB = len(CHI2_EDGES) - 1
NKEY = len(PLANES) * NSTRIP * NB

# residual histogram: symmetric-log bins, 100 per decade of |residual| from
# 1e-5 to 2, plus one central bin [-1e-5, 1e-5]; |residual| beyond 2 is clipped
_RES_POS = 10.0 ** np.arange(-5.0, np.log10(2.0) + 1e-9, 0.01)
RES_EDGES = np.concatenate([-_RES_POS[::-1], _RES_POS])
NBR = len(RES_EDGES) - 1
NKEY_R = len(PLANES) * NSTRIP * NBR

# strip positions: per (plane, strip) the sum of the positions and the number of
# hits are cached; mean = sum / hits (the position of a strip is a fixed
# function of its ID, so no histogram is needed)
NPOS = len(PLANES) * NSTRIP

# excess definition defaults (see docstring)
EXCESS_Z = 3.0
EXCESS_REL = 0.25
EXCESS_MIN_HITS = 10

TYPE_COLORS = {"VES": "tab:blue", "FS": "tab:orange"}

# text describing the runs of the comparison being drawn; written to
# embedded as PNG metadata ("Description") by
# finish_comparison_figure. Set by set_runs_metadata before each batch of plots.
RUNS_METADATA = {"text": None}


def compress_runs(run_numbers):
    """[5000,5001,5002,5005] -> '5000-5002, 5005'"""
    nums = sorted(set(int(n) for n in run_numbers))
    if not nums:
        return "none"
    parts, start, prev = [], nums[0], nums[0]
    for n in nums[1:] + [None]:
        if n is not None and n == prev + 1:
            prev = n
            continue
        parts.append(f"{start}" if start == prev else f"{start}-{prev}")
        if n is not None:
            start = prev = n
    return ", ".join(parts)


def format_runs(labels, run_groups):
    """text: for every compared set the data types, WPs and run ranges"""
    blocks = []
    for lab, group in zip(labels, run_groups):
        types = "/".join(sorted({r["data_type"] for r in group})) or "none"
        wps = ", ".join(f"WP{w}" for w in sorted({r["wp"] for r in group})) or "none"
        blocks.append(f"{lab}\n"
                      f"  data types: {types}\n"
                      f"  working points: {wps}\n"
                      f"  {len(group)} runs: {compress_runs(r['run'] for r in group)}")
    return "\n\n".join(blocks) + "\n"


def set_runs_metadata(labels, run_groups):
    """remember the runs of the plots drawn next (embedded in their PNGs)"""
    RUNS_METADATA["text"] = format_runs(labels, run_groups)


def write_runs_file(path, labels, run_groups):
    """runs_used.txt, written once per comparison (all WPs / data types)"""
    with open(path, "w") as f:
        f.write(format_runs(labels, run_groups))


def finish_comparison_figure(fig, handles, output_file):
    """legend, save (run list embedded as PNG text chunk) and close"""
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    meta = {"Description": RUNS_METADATA["text"]} if RUNS_METADATA["text"] else None
    fig.savefig(output_file, dpi=130, bbox_inches="tight", metadata=meta)
    plt.close(fig)


# ============================================================
# RUN / PERIOD PARSING (from the chi2 distribution script)
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
    """
    'label=5000-5200,5300-5350'          run ranges
    'label=2025-03-01:2025-04-15'        dates (inclusive, from slow control)
    """
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
# SLOW CONTROL (from the chi2 distribution script)
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
                    print(f"  [warn] malformed slow-control line in {slowcontrol_file}")
                    return None
                timestamp = parse_slowcontrol_timestamp(fields)
                if timestamp is None:
                    print(f"  [warn] could not parse timestamp: {fields[1]}")
                    return None
                try:
                    wp = float(fields[5])
                except ValueError:
                    print(f"  [warn] invalid Working Point in {slowcontrol_file}: {fields[5]}")
                    return None
                if not np.isfinite(wp) or not wp.is_integer():
                    print(f"  [warn] bad Working Point in {slowcontrol_file}: {wp}")
                    return None
                return timestamp, int(wp)
    except Exception as exc:
        print(f"  [warn] could not read {slowcontrol_file}: {exc}")
        return None
    print(f"  [warn] no valid lines in {slowcontrol_file}")
    return None


def get_good_run_set(detector):
    """Good runs from select_runs (imported only when needed)."""
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
# ROOT -> per-strip chi2 / residual histograms (vectorised with awkward)
# ============================================================

def needed_branches():
    b = []
    for p in ("xy", "xz"):
        b += [f"BestTrack_3p_ChiSquare_{p}", f"BestTrack_4p_ChiSquare_{p}",
              f"BestTrack_3p_{p}_index", f"Track_3p_of_4p_index_{p}",
              f"chiSquare_4p_{p}", f"cluster_c4_index_{p}"]
    for k in (1, 2, 3):
        b += [f"TrackCluster_z{k}_index", f"TrackCluster_y{k}_index"]
    b += [f"StripsID_{p}" for p in PLANES] + [f"ClusterSize_{p}" for p in PLANES]
    return b


def residual_branches():
    """optional branches: without them only the chi2 histograms are filled"""
    b = [f"displacement_p4_{p}" for p in ("xy", "xz")]
    for k in (1, 2, 3):
        b += [f"Residue_Track3p_z{k}", f"Residue_Track3p_y{k}"]
    return b


def position_branches():
    """optional branches: without them the position plots are skipped"""
    return [f"StripsPosition_{p}" for p in PLANES]


def _pick(jag, idx, dtype=np.int64, fill=-1):
    """jag (n, var), idx (n,) -> numpy (n,), `fill` where not available."""
    mask = ak.local_index(jag, axis=1) == idx[:, None]
    out = ak.fill_none(ak.firsts(jag[mask], axis=1), fill)
    return ak.to_numpy(out).astype(dtype)


def _pick_match(values, flat_chi, best, dtype=np.int64, fill=-1):
    """values / flat_chi (n, var) flat per-candidate arrays; best (n,) chi2 of the
    best candidate. Returns values at the (first) candidate whose chi2 equals
    best (`fill` if none).

    In the ROOT file the 4p arrays are flat (the outer 3p-track level of the
    JSON nesting is lost), so the best 4p candidate is located by its chi2."""
    mask = flat_chi == best[:, None]
    out = ak.fill_none(ak.firsts(values[mask], axis=1), fill)
    return ak.to_numpy(out).astype(dtype)


def _strip_hits(sid, csize, cl, extra=None):
    """strips of cluster cl[event]. In the ROOT file StripsID is flat over the
    clusters of the plane; ClusterSize gives the split.
    `extra` (optional): array with the same structure as sid (e.g.
    StripsPosition), returned for the same strips (None if the structure differs).
    returns (strip ids, event index of every strip, extra values or None)"""
    # first strip of the cluster = total size of all clusters before it
    start = ak.to_numpy(ak.sum(csize[ak.local_index(csize, axis=1) < cl[:, None]], axis=1)).astype(np.int64)
    size = _pick(csize, cl)
    ok = cl >= 0
    start = np.where(ok, start, 0)
    size = np.where(ok, size, 0)
    li = ak.local_index(sid, axis=1)
    m = (li >= start[:, None]) & (li < (start + size)[:, None])
    s = sid[m]
    cnt = ak.to_numpy(ak.num(s, axis=1))
    ids = ak.to_numpy(ak.flatten(s, axis=1)).astype(np.int64)
    ev = np.repeat(np.arange(len(cl)), cnt)
    e = None
    if extra is not None and bool(ak.all(ak.num(extra, axis=1) == ak.num(sid, axis=1))):
        e = ak.to_numpy(ak.flatten(extra[m], axis=1)).astype(float)
    return ids, ev, e


def _add(counts, plane, ids, chi_per_hit):
    ok = (ids >= 0) & (ids < NSTRIP)
    ids, chi = ids[ok], chi_per_hit[ok]
    b = np.clip(np.searchsorted(CHI2_EDGES, chi, side="right") - 1, 0, NB - 1)
    key = (PLANE_INDEX[plane] * NSTRIP + ids) * NB + b
    counts += np.bincount(key, minlength=NKEY).astype(np.int64)


def _add_res(rcounts, plane, ids, res_per_hit):
    ok = (ids >= 0) & (ids < NSTRIP) & np.isfinite(res_per_hit)
    ids, res = ids[ok], res_per_hit[ok]
    b = np.clip(np.searchsorted(RES_EDGES, res, side="right") - 1, 0, NBR - 1)
    key = (PLANE_INDEX[plane] * NSTRIP + ids) * NBR + b
    rcounts += np.bincount(key, minlength=NKEY_R).astype(np.int64)


def _add_pos(psum, pcnt, plane, ids, pos):
    """accumulate sum of positions and number of hits per (plane, strip)"""
    ok = (ids >= 0) & (ids < NSTRIP) & np.isfinite(pos)
    ids, pos = ids[ok], pos[ok]
    sl = slice(PLANE_INDEX[plane] * NSTRIP, (PLANE_INDEX[plane] + 1) * NSTRIP)
    psum[sl] += np.bincount(ids, weights=pos, minlength=NSTRIP)
    pcnt[sl] += np.bincount(ids, minlength=NSTRIP)


def _sparse(counts):
    nz = np.flatnonzero(counts)
    return nz.astype(np.int32), counts[nz].astype(np.int32)


def hits_from_arrays(arr, fit, with_res=True):
    """sparse (chi2 keys, chi2 counts, n_tracks, residual keys, residual counts)
    plus per-(plane,strip) position sum and hit count, for one fit of one run"""
    chi_xz = ak.to_numpy(arr[f"BestTrack_{fit}_ChiSquare_xz"]).astype(float)
    chi_xy = ak.to_numpy(arr[f"BestTrack_{fit}_ChiSquare_xy"]).astype(float)
    sel = np.isfinite(chi_xz) & np.isfinite(chi_xy) & (chi_xz > 0) & (chi_xy > 0)
    counts = np.zeros(NKEY, dtype=np.int64)
    rcounts = np.zeros(NKEY_R, dtype=np.int64)
    psum = np.zeros(NPOS)
    pcnt = np.zeros(NPOS)
    n_tracks = int(sel.sum())
    if n_tracks == 0:
        e = np.array([], dtype=np.int32)
        return e, e.copy(), 0, e.copy(), e.copy(), psum, pcnt

    a = arr[sel]
    cxz, cxy = chi_xz[sel], chi_xy[sel]
    chi = {"xz": cxz, "xy": cxy}

    if fit == "3p":
        t3 = {p: ak.to_numpy(a[f"BestTrack_3p_{p}_index"]).astype(np.int64) for p in ("xz", "xy")}
    else:
        t3 = {p: ak.to_numpy(a[f"Track_3p_of_4p_index_{p}"]).astype(np.int64) for p in ("xz", "xy")}

    def pos_of(plane):
        name = f"StripsPosition_{plane}"
        return a[name] if name in arr.fields else None

    for letter, proj in PROJ.items():
        for k in (1, 2, 3):
            plane = f"{letter}{k}"
            cl = _pick(a[f"TrackCluster_{letter.lower()}{k}_index"], t3[proj])
            ids, ev, pos = _strip_hits(a[f"StripsID_{plane}"], a[f"ClusterSize_{plane}"], cl,
                                       pos_of(plane))
            _add(counts, plane, ids, chi[proj][ev])
            if pos is not None:
                _add_pos(psum, pcnt, plane, ids, pos)
            if with_res and f"Residue_Track3p_{letter.lower()}{k}" in arr.fields:
                # residual of the 3p fit of this track in this plane
                res = _pick(a[f"Residue_Track3p_{letter.lower()}{k}"], t3[proj],
                            dtype=float, fill=np.nan)
                _add_res(rcounts, plane, ids, res[ev])
        if fit == "4p":
            plane = f"{letter}4"
            cl = _pick_match(a[f"cluster_c4_index_{proj}"], a[f"chiSquare_4p_{proj}"], chi[proj])
            ids, ev, pos = _strip_hits(a[f"StripsID_{plane}"], a[f"ClusterSize_{plane}"], cl,
                                       pos_of(plane))
            _add(counts, plane, ids, chi[proj][ev])
            if pos is not None:
                _add_pos(psum, pcnt, plane, ids, pos)
            if with_res and f"displacement_p4_{proj}" in arr.fields:
                # plane-4 cluster minus extrapolation of the 3p track
                res = _pick_match(a[f"displacement_p4_{proj}"], a[f"chiSquare_4p_{proj}"],
                                  chi[proj], dtype=float, fill=np.nan)
                _add_res(rcounts, plane, ids, res[ev])

    k1, c1 = _sparse(counts)
    k2, c2 = _sparse(rcounts)
    return k1, c1, n_tracks, k2, c2, psum, pcnt


def read_run(file_path):
    branches = needed_branches()
    res_branches = residual_branches()
    optional = res_branches + position_branches()
    try:
        with uproot.open(file_path) as f:
            if TREE_NAME not in f:
                print(f"  [warn] tree '{TREE_NAME}' not found")
                return None
            tree = f[TREE_NAME]
            have = set(tree.keys())
            missing = set(branches) - have
            if missing:
                print(f"  [warn] missing branches: {sorted(missing)}")
                return None
            present = [b for b in optional if b in have]
            if len(present) < len(optional):
                print(f"  [warn] optional branches missing: {sorted(set(optional) - have)}")
            with_res = any(b in have for b in res_branches)
            arr = tree.arrays(branches + present, library="ak")
        return {fit: hits_from_arrays(arr, fit, with_res) for fit in FITS}
    except Exception as exc:
        print(f"  [warn] could not read {file_path}: {exc}")
        return None


# ============================================================
# CACHE
# ============================================================

def cache_path(cache_dir, detector):
    return Path(cache_dir) / f"strip_chi2_cache_{detector}.npz"


def save_cache(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "run": np.array([r["run"] for r in records], dtype=np.int32),
        "data_type": np.array([r["data_type"] for r in records], dtype="U3"),
        "wp": np.array([r["wp"] for r in records], dtype=np.int16),
        "timestamp": np.array([r["timestamp"].isoformat() for r in records], dtype="U32"),
    }
    for fit in FITS:
        out[f"{fit}_ntracks"] = np.array([r["fit"][fit][2] for r in records], dtype=np.int64)
        out[f"{fit}_run_idx"] = np.concatenate(
            [np.full(len(r["fit"][fit][0]), i, dtype=np.int32) for i, r in enumerate(records)]
        )
        out[f"{fit}_key"] = np.concatenate([r["fit"][fit][0] for r in records])
        out[f"{fit}_cnt"] = np.concatenate([r["fit"][fit][1] for r in records])
        out[f"{fit}_rrun_idx"] = np.concatenate(
            [np.full(len(r["fit"][fit][3]), i, dtype=np.int32) for i, r in enumerate(records)]
        )
        out[f"{fit}_rkey"] = np.concatenate([r["fit"][fit][3] for r in records])
        out[f"{fit}_rcnt"] = np.concatenate([r["fit"][fit][4] for r in records])
        out[f"{fit}_possum"] = np.stack([r["fit"][fit][5] for r in records])   # (nruns, NPOS)
        out[f"{fit}_poscnt"] = np.stack([r["fit"][fit][6] for r in records])
    np.savez_compressed(path, **out)
    print(f"  Cache written: {path} ({path.stat().st_size / 1e6:.1f} MB)")


def load_cache(path):
    z = np.load(path, allow_pickle=False)
    data = {k: z[k] for k in z.files}
    z.close()
    runs = []
    for i in range(len(data["run"])):
        ts = datetime.fromisoformat(str(data["timestamp"][i]))
        runs.append({"run": int(data["run"][i]), "data_type": str(data["data_type"][i]),
                     "wp": int(data["wp"][i]), "timestamp": ts, "idx": i})
    return runs, data


def has_residuals(data, fit):
    return f"{fit}_rkey" in data and data[f"{fit}_rcnt"].sum() > 0


def has_positions(data, fit):
    return f"{fit}_poscnt" in data and data[f"{fit}_poscnt"].sum() > 0


def aggregate(data, fit, run_idx_set):
    """summed chi2 histogram H[plane, strip, bin] over a set of run indices"""
    mask = np.zeros(len(data["run"]), dtype=bool)
    mask[list(run_idx_set)] = True
    sel = mask[data[f"{fit}_run_idx"]]
    h = np.bincount(data[f"{fit}_key"][sel], weights=data[f"{fit}_cnt"][sel], minlength=NKEY)
    return h.reshape(len(PLANES), NSTRIP, NB)


def aggregate_res(data, fit, run_idx_set):
    """summed residual histogram R[plane, strip, bin] over a set of run indices"""
    mask = np.zeros(len(data["run"]), dtype=bool)
    mask[list(run_idx_set)] = True
    sel = mask[data[f"{fit}_rrun_idx"]]
    h = np.bincount(data[f"{fit}_rkey"][sel], weights=data[f"{fit}_rcnt"][sel], minlength=NKEY_R)
    return h.reshape(len(PLANES), NSTRIP, NBR)


def aggregate_pos(data, fit, run_idx_set):
    """(hits n[plane, strip], mean position[plane, strip]) summed over runs"""
    ii = sorted(run_idx_set)
    n = data[f"{fit}_poscnt"][ii].sum(axis=0)
    sm = data[f"{fit}_possum"][ii].sum(axis=0)
    pos = np.where(n > 0, sm / np.maximum(n, 1), np.nan)
    return n.reshape(len(PLANES), NSTRIP), pos.reshape(len(PLANES), NSTRIP)


# ============================================================
# STATISTICS FROM HISTOGRAMS
# ============================================================

def hist_quantile(H, q, edges=CHI2_EDGES, log=True):
    """H (m, nbins) counts, q scalar or (m,) -> (m,) quantile, interpolated inside
    the bin (log-interpolated for log=True, linear otherwise); NaN where there
    are no entries."""
    nb = len(edges) - 1
    cum = np.cumsum(H, axis=1)
    n = cum[:, -1]
    q = np.broadcast_to(np.asarray(q, dtype=float), n.shape)
    target = q * n
    b = np.array([np.searchsorted(c, t, side="left") for c, t in zip(cum, target)])
    b = np.clip(b, 0, nb - 1)
    rows = np.arange(len(H))
    prev = np.where(b > 0, cum[rows, np.maximum(b - 1, 0)], 0.0)
    c = H[rows, b]
    frac = np.where(c > 0, (target - prev) / np.where(c > 0, c, 1), 0.5)
    frac = np.clip(frac, 0, 1)
    if log:
        val = edges[b] * (edges[b + 1] / edges[b]) ** frac
    else:
        val = edges[b] + frac * (edges[b + 1] - edges[b])
    return np.where(n > 0, val, np.nan)


def plane_stats(H, z, edges=CHI2_EDGES, log=True):
    """per-strip stats of one plane H (NSTRIP, nbins)"""
    n = H.sum(axis=1)
    safe = np.maximum(n, 1)
    half = z / (2.0 * np.sqrt(safe))

    def quant(q):
        return hist_quantile(H, q, edges, log)

    return {
        "n": n,
        "q25": quant(0.25),
        "med": quant(0.5),
        "q75": quant(0.75),
        "ci_lo": quant(np.clip(0.5 - half, 0, 1)),
        "ci_hi": quant(np.clip(0.5 + half, 0, 1)),
    }


def reference_median(Hfit, plane, reference):
    """pooled median chi2 used as the 'general median'"""
    if reference == "projection":
        pl = [p for p in FIT_PLANES_CURRENT[0] if p[0] == plane[0]]
    else:
        pl = [plane]
    pooled = sum(Hfit[PLANE_INDEX[p]].sum(axis=0) for p in pl)[None, :]
    return float(hist_quantile(pooled, 0.5)[0])


FIT_PLANES_CURRENT = [None]   # set per fit (small helper for reference_median)


def analyse_fit(Hfit, fit, args):
    """-> dict plane -> stats (+ flags), list of rows for tables"""
    FIT_PLANES_CURRENT[0] = FIT_PLANES[fit]
    result, rows = {}, []
    for plane in FIT_PLANES[fit]:
        H = Hfit[PLANE_INDEX[plane]]
        st = plane_stats(H, args.excess_z)
        ref = reference_median(Hfit, plane, args.reference)

        # robust z of the strip median w.r.t. the other strips (information)
        valid = (st["n"] >= args.min_hits) & np.isfinite(st["med"])
        if valid.sum() >= 5:
            m = np.median(st["med"][valid])
            mad = 1.4826 * np.median(np.abs(st["med"][valid] - m))
            rz = (st["med"] - m) / mad if mad > 0 else np.full(NSTRIP, np.nan)
        else:
            rz = np.full(NSTRIP, np.nan)

        f = 1.0 + args.rel_excess
        judged = st["n"] >= args.min_hits
        rzf = np.nan_to_num(rz, nan=0.0)
        high = (
            judged & (st["ci_lo"] > ref) & (st["med"] > f * ref)
            & ((args.min_robust_z <= 0) | (rzf >= args.min_robust_z))
        )
        low = (
            judged & (st["ci_hi"] < ref) & (st["med"] < ref / f)
            & ((args.min_robust_z <= 0) | (rzf <= -args.min_robust_z))
        )
        flag = high | low
        st.update({"ref": ref, "flag": flag, "high": high, "low": low, "robust_z": rz})
        result[plane] = st

        for s in range(NSTRIP):
            rows.append({
                "plane": plane, "strip_id": s, "n_hits": int(st["n"][s]),
                "chi2_q25": st["q25"][s], "chi2_median": st["med"][s],
                "chi2_q75": st["q75"][s], "median_ci_lo": st["ci_lo"][s],
                "median_ci_hi": st["ci_hi"][s], "ref_median": ref,
                "ratio_to_ref": st["med"][s] / ref if ref > 0 else np.nan,
                "robust_z_vs_strips": rz[s], "excess": bool(flag[s]),
                "direction": "high" if high[s] else ("low" if low[s] else ""),
            })
    return result, rows


def analyse_residuals(Rfit, fit, args):
    """-> dict plane -> residual stats, list of rows for the residual table"""
    result, rows = {}, []
    for plane in FIT_PLANES[fit]:
        H = Rfit[PLANE_INDEX[plane]]
        st = plane_stats(H, args.excess_z, RES_EDGES, log=False)
        pooled = H.sum(axis=0)[None, :]
        st["ref"] = float(hist_quantile(pooled, 0.5, RES_EDGES, False)[0])   # plane median
        result[plane] = st
        for s in range(NSTRIP):
            rows.append({
                "plane": plane, "strip_id": s, "n_hits": int(st["n"][s]),
                "residual_branch": residual_branch_name(plane),
                "residual_q25": st["q25"][s], "residual_median": st["med"][s],
                "residual_q75": st["q75"][s], "median_ci_lo": st["ci_lo"][s],
                "median_ci_hi": st["ci_hi"][s], "plane_median": st["ref"],
            })
    return result, rows


def write_csv(path, rows, only_flagged=False):
    import csv
    if only_flagged:
        rows = [r for r in rows if r["excess"]]
    if not rows:
        with open(path, "w") as f:
            f.write("# none\n")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in r.items()})


# ============================================================
# PLOTS
# ============================================================

def branch_name(fit, plane):
    """real ROOT branch the chi2 of this plane comes from"""
    return f"BestTrack_{fit}_ChiSquare_{PROJ[plane[0]]}"


def residual_branch_name(plane):
    """real ROOT branch the residual of this plane comes from"""
    letter, k = plane[0], plane[1]
    if k == "4":
        return f"displacement_p4_{PROJ[letter]}"
    return f"Residue_Track3p_{letter.lower()}{k}"


def make_title(detector, wp, dtype, fit, run_label, period, ntracks):
    t = f"{detector}, WP{wp}, {dtype}, {fit} 3D tracks, {ntracks:,} tracks"
    if run_label:
        t += f" [{run_label}]"
    if period is not None:
        t += f"\nperiod '{period['label']}': {period['desc']}"
    return t


def residual_title(title, fit):
    if fit == "4p":
        return (title + "\nresiduals: planes 1-3 from the 3p fit of the track, "
                        "plane 4 = distance to the 3p extrapolation")
    return title


HIGH_COLOR, LOW_COLOR = "red", "tab:blue"


def _mark_flags(ax, x, st):
    for key, col, lab in (("high", HIGH_COLOR, "excess high"), ("low", LOW_COLOR, "excess low")):
        fl = st[key]
        if fl.any():
            ax.plot(x[fl], st["med"][fl], "o", mfc="none", mec=col, ms=9, mew=1.6,
                    label=f"{lab} ({fl.sum()})")
            for s_ in x[fl]:
                ax.annotate(str(s_), (s_, st["med"][s_]), xytext=(0, 8),
                            textcoords="offset points", ha="center", fontsize=7, color=col)


def _median_sigma(st):
    ok = (st["n"] > 0) & np.isfinite(st["med"])
    if not ok.any():
        return np.nan
    return float(np.nanstd(st["med"][ok]))


def plot_chi2_median(result, fit, title, output_file, args, *,
                     show_excess=True, show_sigma_lines=True):
    planes = FIT_PLANES[fit]
    ncol = len(planes) // 2
    fig, axs = plt.subplots(2, ncol, figsize=(4.4 * ncol, 7.2), sharex=True, squeeze=False)
    x = np.arange(NSTRIP)
    for r, letter in enumerate("ZY"):
        pls = [p for p in planes if p[0] == letter]
        lo_all, hi_all = [], []
        for c, plane in enumerate(pls):
            ax = axs[r, c]
            st = result[plane]
            ok = st["n"] > 0                       # every strip with hits is drawn
            ax.step(x[ok], st["med"][ok], where="mid", color="C0", lw=1.3)
            ax.errorbar(x[ok], st["med"][ok],
                        yerr=[st["med"][ok] - st["ci_lo"][ok], st["ci_hi"][ok] - st["med"][ok]],
                        fmt=".", ms=3, color="C0", lw=0.8)
            ax.axhline(st["ref"], color="green", ls="--", lw=1.2,
                       label=f"ref median {st['ref']:.3g}")
            if show_sigma_lines:
                sigma = _median_sigma(st)
                if np.isfinite(sigma):
                    lo = max(st["ref"] - 3.0 * sigma, np.nextafter(0.0, 1.0))
                    hi = st["ref"] + 3.0 * sigma
                    ax.axhline(lo, color="purple", ls=":", lw=1.2,
                               label=f"ref - 3 sigma ({lo:.3g})")
                    ax.axhline(hi, color="purple", ls=":", lw=1.2,
                               label=f"ref + 3 sigma ({hi:.3g})")
            if show_excess:
                _mark_flags(ax, x, st)
            ax.set_yscale("log")
            ax.set_xlim(-1, NSTRIP)
            ax.set_title(f"{plane}: {branch_name(fit, plane)}", fontsize=9)
            ax.grid(True, alpha=0.3, ls="--")
            ax.legend(fontsize=7, loc="upper right")
            if r == 1:
                ax.set_xlabel("strip ID")
            if c == 0:
                ax.set_ylabel("median chi2 per strip")
            if ok.any():
                lo_all.append(min(np.nanmin(st["med"][ok]), st["ref"]))
                hi_all.append(max(np.nanmax(st["med"][ok]), st["ref"]))
        if lo_all:   # common y-range per row, set by the medians (not by the error bars)
            for c in range(len(pls)):
                axs[r, c].set_ylim(min(lo_all) / 2.0, max(hi_all) * 3.0)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_file, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_residual_median(result, fit, title, output_file, min_hits=EXCESS_MIN_HITS):
    """median residual per strip (error bars = CI of the median).
    The y-range is set per panel from the medians of the strips with >= min_hits
    hits; error bars of low-statistics strips can therefore run off the panel."""
    planes = FIT_PLANES[fit]
    ncol = len(planes) // 2
    fig, axs = plt.subplots(2, ncol, figsize=(4.4 * ncol, 7.2), sharex=True, squeeze=False)
    x = np.arange(NSTRIP)
    for r, letter in enumerate("ZY"):
        for c, plane in enumerate(p for p in planes if p[0] == letter):
            ax = axs[r, c]
            st = result[plane]
            ok = st["n"] > 0
            ax.axhline(0.0, color="black", lw=0.8)
            if np.isfinite(st["ref"]):
                ax.axhline(st["ref"], color="green", ls="--", lw=1.2,
                           label=f"plane median {st['ref']:.2g}")
            ax.step(x[ok], st["med"][ok], where="mid", color="C0", lw=1.3)
            ax.errorbar(x[ok], st["med"][ok],
                        yerr=[st["med"][ok] - st["ci_lo"][ok], st["ci_hi"][ok] - st["med"][ok]],
                        fmt=".", ms=3, color="C0", lw=0.8)
            judged = (st["n"] >= min_hits) & np.isfinite(st["med"])
            if judged.any():        # y-range from the medians (not from the error bars)
                lo = min(float(np.min(st["med"][judged])), 0.0)
                hi = max(float(np.max(st["med"][judged])), 0.0)
                pad = 0.25 * (hi - lo) if hi > lo else 1e-3
                ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlim(-1, NSTRIP)
            ax.set_title(f"{plane}: {residual_branch_name(plane)}", fontsize=9)
            ax.grid(True, alpha=0.3, ls="--")
            ax.legend(fontsize=7, loc="upper right")
            if r == 1:
                ax.set_xlabel("strip ID")
            if c == 0:
                ax.set_ylabel("median residual per strip")
    fig.suptitle(residual_title(title, fit))
    fig.tight_layout()
    fig.savefig(output_file, dpi=130, bbox_inches="tight")
    plt.close(fig)

def rebin_linear(h, edges, new_edges):
    """counts of the (log-spaced) histogram h on edges -> counts on new_edges,
    via the cumulative distribution (assumes uniform density inside a bin)"""
    cum = np.concatenate([[0.0], np.cumsum(h)])
    return np.diff(np.interp(new_edges, edges, cum))


def residual_half_range(H, q=0.005):
    """symmetric x-range containing the central (1-2q) of the hits"""
    cum = np.concatenate([[0.0], np.cumsum(H)])
    if cum[-1] == 0:
        return np.nan
    lo, hi = np.interp([q, 1 - q], cum / cum[-1], RES_EDGES)
    return max(abs(lo), abs(hi))


def plot_residual_hist(Rfit, fit, title, output_file, nbins=101, q=0.005, logy=False):
    """Residual distribution per plane: x = residual, y = number of hits
    (strips of the track clusters), summed over all strips of the plane."""
    planes = FIT_PLANES[fit]
    ncol = len(planes) // 2
    fig, axs = plt.subplots(2, ncol, figsize=(4.4 * ncol, 7.2), squeeze=False)
    for r, letter in enumerate("ZY"):
        for c, plane in enumerate(p for p in planes if p[0] == letter):
            ax = axs[r, c]
            H = Rfit[PLANE_INDEX[plane]].sum(axis=0)
            n = H.sum()
            ax.set_title(f"{plane}: {residual_branch_name(plane)}", fontsize=9)
            ax.grid(True, alpha=0.3, ls="--")
            if c == 0:
                ax.set_ylabel("hits")
            if r == 1:
                ax.set_xlabel("residual")
            if n == 0:
                continue
            half = residual_half_range(H, q)
            edges = np.linspace(-half, half, nbins + 1)
            counts = rebin_linear(H, RES_EDGES, edges)
            ax.stairs(counts, edges, fill=True, color="C0", alpha=0.5)
            ax.stairs(counts, edges, color="C0", lw=1.0)
            q16, med, q84 = (float(hist_quantile(H[None, :], qq, RES_EDGES, False)[0])
                             for qq in (0.16, 0.5, 0.84))
            ax.axvline(0.0, color="black", lw=0.8)
            ax.axvline(med, color="green", ls="--", lw=1.2,
                       label=f"median {med:.2g}\nrobust sigma {(q84 - q16) / 2:.2g}")
            ax.plot([], [], " ", label=f"{int(n):,} hits ({counts.sum() / n:.1%} in range)")
            if logy:
                ax.set_yscale("log")
            ax.set_xlim(-half, half)
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(residual_title(title, fit))
    fig.tight_layout()
    fig.savefig(output_file, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_residual_hist_period_comparison(Rs, fit, title, output_file, nbins=101, q=0.005,
                                         min_counts=10):
    """Overlay of the residual distributions of two sets (fraction of hits per
    bin, so different statistics are comparable) + ratio panel B/A underneath.
    Rs: label -> R[plane, strip, bin]. Ratio only in bins with >= min_counts
    hits in both sets; error = ratio * sqrt(1/nA + 1/nB) (Poisson, on the
    rebinned counts, so approximate)."""
    labels = list(Rs)
    labA, labB = labels
    planes = FIT_PLANES[fit]
    ncol = len(planes) // 2
    fig, axs = plt.subplots(
        4, ncol, figsize=(4.4 * ncol, 10.2), squeeze=False,
        gridspec_kw={"height_ratios": [3, 1, 3, 1], "hspace": 0.28},
    )
    colors = ("tab:blue", "tab:orange")
    for r, letter in enumerate("ZY"):
        for c, plane in enumerate(p for p in planes if p[0] == letter):
            ax, axr = axs[2 * r, c], axs[2 * r + 1, c]
            Hs = {lab: Rs[lab][PLANE_INDEX[plane]].sum(axis=0) for lab in labels}
            halves = [residual_half_range(H, q) for H in Hs.values() if H.sum() > 0]
            ax.set_title(f"{plane}: {residual_branch_name(plane)}", fontsize=9, pad=7)
            ax.grid(True, alpha=0.3, ls="--")
            axr.grid(True, alpha=0.3, ls="--")
            axr.axhline(1.0, color="black", lw=1.0)
            if not halves:
                continue
            half = max(halves)
            edges = np.linspace(-half, half, nbins + 1)
            centers = 0.5 * (edges[1:] + edges[:-1])

            counts, fracs = {}, {}
            for lab, col in zip(labels, colors):
                H = Hs[lab]
                if H.sum() == 0:
                    continue
                counts[lab] = rebin_linear(H, RES_EDGES, edges)
                fracs[lab] = counts[lab] / H.sum()
                ax.stairs(fracs[lab], edges, fill=True, color=col, alpha=0.35)
                ax.stairs(fracs[lab], edges, color=col, lw=1.1)
            ax.axvline(0.0, color="black", lw=0.8)
            ax.set_xlim(-half, half)
            if c == 0:
                ax.set_ylabel("fraction of hits per bin")

            # --- ratio B / A
            if len(counts) == 2:
                ok = (counts[labA] >= min_counts) & (counts[labB] >= min_counts)
                if ok.any():
                    ratio = fracs[labB][ok] / fracs[labA][ok]
                    err = ratio * np.sqrt(1.0 / counts[labA][ok] + 1.0 / counts[labB][ok])
                    axr.errorbar(centers[ok], ratio, yerr=err, fmt=".", ms=3,
                                 color="0.25", lw=0.7, capsize=0)
                    m = float(np.clip(np.max(np.abs(np.log(ratio))),
                                      np.log(1.5), np.log(10)))
                    axr.set_ylim(np.exp(-m * 1.15), np.exp(m * 1.15))
            axr.axvline(0.0, color="black", lw=0.8)
            axr.set_xlim(-half, half)
            axr.set_yscale("log")
            axr.set_yticks([0.5, 1, 2])
            axr.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            axr.minorticks_off()
            if c == 0:
                axr.set_ylabel(f"{labB}\n/ {labA}", fontsize=8)
            if r == 1:
                axr.set_xlabel("residual")

    handles = [Patch(facecolor=col, alpha=0.5, edgecolor=col, label=lab)
               for lab, col in zip(labels, colors)]
    fig.suptitle(residual_title(title, fit))
    finish_comparison_figure(fig, handles, output_file)
    

def plot_hits_period_comparison(results, fit, title, output_file):
    """Overlay hits-per-strip (unit area) of two periods + ratio panel B/A."""
    periods = list(results)
    labA, labB = periods
    planes = FIT_PLANES[fit]
    ncol = len(planes) // 2
    fig, axs = plt.subplots(
        4, ncol, figsize=(4.4 * ncol, 10.2), sharex=True, squeeze=False,
        gridspec_kw={"height_ratios": [3, 1, 3, 1], "hspace": 0.28},
    )
    colors = ("tab:blue", "tab:orange")
    x = np.arange(NSTRIP)

    for r, letter in enumerate("ZY"):
        for c, plane in enumerate(p for p in planes if p[0] == letter):
            ax, axr = axs[2 * r, c], axs[2 * r + 1, c]

            # --- overlay of the two normalised distributions
            n = {lab: np.asarray(results[lab][plane]["n"], dtype=float) for lab in periods}
            dens = {}
            for lab, color in zip(periods, colors):
                total = n[lab].sum()
                dens[lab] = n[lab] / total if total > 0 else np.full(NSTRIP, np.nan)
                if total == 0:
                    continue
                ax.bar(x, dens[lab], width=1.0, color=color, alpha=0.4)
                ax.step(x, dens[lab], where="mid", color=color, lw=1.2)
            ax.set_title(f"{plane}: {branch_name(fit, plane)}", fontsize=9, pad=7)
            ax.set_xlim(-1, NSTRIP)
            ax.grid(True, alpha=0.3, ls="--")
            if c == 0:
                ax.set_ylabel("fraction of hits per strip")

            # --- ratio B / A (only strips with hits in both periods)
            ok = (n[labA] > 0) & (n[labB] > 0)
            axr.axhline(1.0, color="black", lw=1.0)
            if ok.any():
                ratio = dens[labB][ok] / dens[labA][ok]
                # Poisson error: relative error = sqrt(1/nA + 1/nB)
                err = ratio * np.sqrt(1.0 / n[labA][ok] + 1.0 / n[labB][ok])
                axr.errorbar(x[ok], ratio, yerr=err, fmt=".", ms=3,
                             color="0.25", lw=0.7, capsize=0)
                # symmetric log limits around 1, set by the ratios (not the error bars)
                m = float(np.clip(np.max(np.abs(np.log(ratio))), np.log(1.5), np.log(10)))
                axr.set_ylim(np.exp(-m * 1.15), np.exp(m * 1.15))
            axr.set_yscale("log")
            axr.set_yticks([0.5, 1, 2])
            axr.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            axr.minorticks_off()
            axr.grid(True, alpha=0.3, ls="--")
            if c == 0:
                axr.set_ylabel(f"{labB}\n/ {labA}", fontsize=8)
            if r == 1:
                axr.set_xlabel("strip ID")

    handles = [Patch(facecolor=col, alpha=0.5, edgecolor=col, label=lab)
               for lab, col in zip(periods, colors)]
    fig.suptitle(title)
    finish_comparison_figure(fig, handles, output_file)


def plot_chi2_median_period_comparison(results, fit, title, output_file):
    """Overlay median chi2 per strip for two periods + ratio panel B/A."""
    periods = list(results)
    labA, labB = periods
    planes = FIT_PLANES[fit]
    ncol = len(planes) // 2
    fig, axs = plt.subplots(
        4, ncol, figsize=(4.4 * ncol, 10.2), sharex=True, squeeze=False,
        gridspec_kw={"height_ratios": [3, 1, 3, 1], "hspace": 0.28},
    )
    colors = ("tab:blue", "tab:orange")
    x = np.arange(NSTRIP)

    for r, letter in enumerate("ZY"):
        for c, plane in enumerate(p for p in planes if p[0] == letter):
            ax, axr = axs[2 * r, c], axs[2 * r + 1, c]
            stats = {lab: results[lab][plane] for lab in periods}

            # --- overlay of the two strip-median distributions
            for lab, color in zip(periods, colors):
                st = stats[lab]
                ok = (st["n"] > 0) & np.isfinite(st["med"]) & (st["med"] > 0)
                if not ok.any():
                    continue
                ax.step(x[ok], st["med"][ok], where="mid", color=color, lw=1.3)
                ax.errorbar(
                    x[ok], st["med"][ok],
                    yerr=[
                        st["med"][ok] - st["ci_lo"][ok],
                        st["ci_hi"][ok] - st["med"][ok],
                    ],
                    fmt=".", ms=3, color=color, lw=0.7,
                )
                ax.axhline(
                    st["ref"], color=color, ls="--", lw=1.0,
                    label=f"{lab} ref {st['ref']:.3g}",
                )
            ax.set_yscale("log")
            ax.set_title(f"{plane}: {branch_name(fit, plane)}", fontsize=9, pad=7)
            ax.set_xlim(-1, NSTRIP)
            ax.grid(True, alpha=0.3, ls="--")
            if c == 0:
                ax.set_ylabel("median chi2 per strip")

            # --- ratio B / A (only strips with finite medians in both periods)
            st_a, st_b = stats[labA], stats[labB]
            ok = (
                (st_a["n"] > 0) & (st_b["n"] > 0)
                & np.isfinite(st_a["med"]) & np.isfinite(st_b["med"])
                & (st_a["med"] > 0) & (st_b["med"] > 0)
            )
            axr.axhline(1.0, color="black", lw=1.0)
            if ok.any():
                ratio = st_b["med"][ok] / st_a["med"][ok]
                low = st_b["ci_lo"][ok] / st_a["ci_hi"][ok]
                high = st_b["ci_hi"][ok] / st_a["ci_lo"][ok]
                finite_error = (
                    np.isfinite(low) & np.isfinite(high)
                    & (low > 0) & (high > 0)
                )
                yerr_low = np.where(finite_error, ratio - low, 0.0)
                yerr_high = np.where(finite_error, high - ratio, 0.0)
                axr.errorbar(
                    x[ok], ratio, yerr=[yerr_low, yerr_high],
                    fmt=".", ms=3, color="0.25", lw=0.7, capsize=0,
                )
                m = float(np.clip(
                    np.max(np.abs(np.log(ratio))),
                    np.log(1.5), np.log(10),
                ))
                axr.set_ylim(np.exp(-m * 1.15), np.exp(m * 1.15))
            axr.set_yscale("log")
            axr.set_yticks([0.5, 1, 2])
            axr.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            axr.minorticks_off()
            axr.grid(True, alpha=0.3, ls="--")
            if c == 0:
                axr.set_ylabel(f"{labB}\n/ {labA}", fontsize=8)
            if r == 1:
                axr.set_xlabel("strip ID")

    handles = [
        Line2D([0], [0], color=color, lw=1.5, label=lab)
        for lab, color in zip(periods, colors)
    ]
    fig.suptitle(title)
    finish_comparison_figure(fig, handles, output_file)


def plot_residual_period_comparison(results, fit, title, output_file,
                                    z=EXCESS_Z, min_hits=EXCESS_MIN_HITS):
    """Overlay median residual per strip for two periods + difference panel B-A.
    Difference error = z * sqrt(sigma_A^2 + sigma_B^2) with
    sigma_i = (ci_hi - ci_lo) / (2 z), i.e. +-z sigma. Only strips with at
    least `min_hits` hits in BOTH periods enter the difference panel."""
    periods = list(results)
    labA, labB = periods
    planes = FIT_PLANES[fit]
    ncol = len(planes) // 2
    fig, axs = plt.subplots(
        4, ncol, figsize=(4.4 * ncol, 10.2), sharex=True, squeeze=False,
        gridspec_kw={"height_ratios": [3, 1, 3, 1], "hspace": 0.28},
    )
    colors = ("tab:blue", "tab:orange")
    x = np.arange(NSTRIP)

    for r, letter in enumerate("ZY"):
        for c, plane in enumerate(p for p in planes if p[0] == letter):
            ax, axr = axs[2 * r, c], axs[2 * r + 1, c]
            stats = {lab: results[lab][plane] for lab in periods}

            # --- overlay of the two strip-median residuals
            ax.axhline(0.0, color="black", lw=0.8)
            lo_all, hi_all = [], []
            for lab, color in zip(periods, colors):
                st = stats[lab]
                ok = (st["n"] > 0) & np.isfinite(st["med"])
                if not ok.any():
                    continue
                ax.step(x[ok], st["med"][ok], where="mid", color=color, lw=1.3)
                ax.errorbar(
                    x[ok], st["med"][ok],
                    yerr=[st["med"][ok] - st["ci_lo"][ok], st["ci_hi"][ok] - st["med"][ok]],
                    fmt=".", ms=3, color=color, lw=0.7,
                )
                if np.isfinite(st["ref"]):
                    ax.axhline(st["ref"], color=color, ls="--", lw=1.0,
                               label=f"{lab} plane median {st['ref']:.2g}")
                judged = (st["n"] >= min_hits) & np.isfinite(st["med"])
                if judged.any():
                    lo_all.append(float(np.min(st["med"][judged])))
                    hi_all.append(float(np.max(st["med"][judged])))
            if lo_all:              # y-range from the medians (not from the error bars)
                lo, hi = min(min(lo_all), 0.0), max(max(hi_all), 0.0)
                pad = 0.25 * (hi - lo) if hi > lo else 1e-3
                ax.set_ylim(lo - pad, hi + pad)
            ax.set_title(f"{plane}: {residual_branch_name(plane)}", fontsize=9, pad=7)
            ax.set_xlim(-1, NSTRIP)
            ax.grid(True, alpha=0.3, ls="--")
            ax.legend(fontsize=7, loc="upper right")
            if c == 0:
                ax.set_ylabel("median residual per strip")

            # --- difference B - A (strips with >= min_hits in both periods)
            st_a, st_b = stats[labA], stats[labB]
            ok = (
                (st_a["n"] >= min_hits) & (st_b["n"] >= min_hits)
                & np.isfinite(st_a["med"]) & np.isfinite(st_b["med"])
                & np.isfinite(st_a["ci_lo"]) & np.isfinite(st_a["ci_hi"])
                & np.isfinite(st_b["ci_lo"]) & np.isfinite(st_b["ci_hi"])
            )
            axr.axhline(0.0, color="black", lw=1.0)
            if ok.any():
                diff = st_b["med"][ok] - st_a["med"][ok]
                wa = st_a["ci_hi"][ok] - st_a["ci_lo"][ok]      # = 2 z sigma_A
                wb = st_b["ci_hi"][ok] - st_b["ci_lo"][ok]      # = 2 z sigma_B
                err = 0.5 * np.hypot(wa, wb)                    # = z sqrt(sA^2 + sB^2)
                axr.errorbar(x[ok], diff, yerr=err, fmt=".", ms=3,
                             color="0.25", lw=0.7, capsize=0)
                m = max(float(np.max(np.abs(diff))), 2.0 * float(np.median(err)), 1e-6) * 1.3
                axr.set_ylim(-m, m)
            axr.grid(True, alpha=0.3, ls="--")
            if c == 0:
                axr.set_ylabel(f"{labB}\n- {labA}", fontsize=8)
            if r == 1:
                axr.set_xlabel("strip ID")

    handles = [
        Line2D([0], [0], color=color, lw=1.5, label=lab)
        for lab, color in zip(periods, colors)
    ]
    fig.suptitle(residual_title(title, fit))
    finish_comparison_figure(fig, handles, output_file)


def plot_hits(result, fit, title, output_file, args):
    """hits per strip; red = excess high, blue = excess low"""
    planes = FIT_PLANES[fit]
    ncol = len(planes) // 2
    fig, axs = plt.subplots(2, ncol, figsize=(4.4 * ncol, 7.2), sharex=True, squeeze=False)
    x = np.arange(NSTRIP)
    for r, letter in enumerate("ZY"):
        for c, plane in enumerate([p for p in planes if p[0] == letter]):
            ax = axs[r, c]
            st = result[plane]
            colors = np.where(st["high"], HIGH_COLOR, np.where(st["low"], LOW_COLOR, "C1"))
            ax.bar(x, st["n"], width=1.0, color=colors, alpha=0.85)
            ax.set_title(f"{plane}: {branch_name(fit, plane)}", fontsize=9)
            ax.set_xlim(-1, NSTRIP)
            ax.grid(True, alpha=0.3, ls="--")
            if r == 1:
                ax.set_xlabel("strip ID")
            if c == 0:
                ax.set_ylabel("hits (strips in track clusters)")
    fig.legend(handles=[Patch(color="C1", label="no excess"),
                        Patch(color=HIGH_COLOR, label="chi2 excess high"),
                        Patch(color=LOW_COLOR, label="chi2 excess low")],
               loc="lower center", ncol=3, fontsize=9)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_file, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_hits_chi2(result, fit, title, output_file, args):
    """hits per strip; bar colour = median chi2 of that strip relative to the
    reference median of the plane (blue below, white = reference, red above).
    Strips with an excess get a black outline + ID."""
    planes = FIT_PLANES[fit]
    ncol = len(planes) // 2
    fig, axs = plt.subplots(2, ncol, figsize=(4.4 * ncol + 1.2, 7.2), sharex=True, squeeze=False)
    x = np.arange(NSTRIP)
    lim = np.log10(3.0)                                  # colour scale: ref/3 ... ref*3
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    # Avoid a white centre: strips close to the reference remain visible.
    cmap = LinearSegmentedColormap.from_list(
        "chi2_ratio",
        [(0.0, "#2166ac"), (0.42, "#67a9cf"), (0.5, "#8c8c8c"),
         (0.58, "#ef8a62"), (1.0, "#b2182b")],
    )
    for r, letter in enumerate("ZY"):
        for c, plane in enumerate([p for p in planes if p[0] == letter]):
            ax = axs[r, c]
            st = result[plane]
            ok = st["n"] > 0
            ratio = np.where(ok, st["med"] / st["ref"], 1.0)
            col = cmap(norm(np.clip(np.log10(ratio), -lim, lim)))
            ax.bar(x[ok], st["n"][ok], width=1.0, color=col[ok],
                   edgecolor="#444444", linewidth=0.25)
            fl = st["flag"]
            if fl.any():
                ax.bar(x[fl], st["n"][fl], width=1.0, color="none", edgecolor="black", lw=1.4)
                for s_ in x[fl]:
                    ax.annotate(str(s_), (s_, st["n"][s_]), xytext=(0, 3),
                                textcoords="offset points", ha="center", fontsize=7)
            ax.set_title(f"{plane}: {branch_name(fit, plane)}", fontsize=9)
            ax.set_xlim(-1, NSTRIP)
            ax.grid(True, alpha=0.3, ls="--")
            if r == 1:
                ax.set_xlabel("strip ID")
            if c == 0:
                ax.set_ylabel("hits (strips in track clusters)")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 0.93, 1])
    cax = fig.add_axes([0.945, 0.15, 0.015, 0.7])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    ticks = [-lim, -lim / 2, 0, lim / 2, lim]
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{10 ** t:.2f}" for t in ticks])
    cb.set_label("median chi2 of the strip / reference median of the plane")
    fig.savefig(output_file, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _strip_width(pos_plane):
    """typical spacing of the strip positions of a plane (bar width / density unit)"""
    p = np.sort(pos_plane[np.isfinite(pos_plane)])
    d = np.diff(p)
    d = d[d > 0]
    return float(np.median(d)) if len(d) else 1.0


def plot_hits_position(n, pos, fit, title, output_file):
    """hit density vs strip position: hits per unit position (hits / strip pitch),
    only hits of the clusters of the best track. n, pos: [plane, strip]."""
    planes = FIT_PLANES[fit]
    ncol = len(planes) // 2
    fig, axs = plt.subplots(2, ncol, figsize=(4.4 * ncol, 7.2), squeeze=False)
    for r, letter in enumerate("ZY"):
        for c, plane in enumerate(p for p in planes if p[0] == letter):
            ax = axs[r, c]
            i = PLANE_INDEX[plane]
            w = _strip_width(pos[i])
            ok = (n[i] > 0) & np.isfinite(pos[i])
            ax.bar(pos[i][ok], n[i][ok] / w, width=w, color="C1", alpha=0.85)
            ax.set_title(f"{plane}: StripsPosition_{plane}", fontsize=9)
            ax.grid(True, alpha=0.3, ls="--")
            if r == 1:
                ax.set_xlabel("strip position")
            if c == 0:
                ax.set_ylabel("hits per unit position")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_file, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_hits_position_period_comparison(P, fit, title, output_file):
    """Overlay of the hit density vs strip position of two sets (unit area) +
    ratio B/A per strip underneath (x = position of set A).
    P: label -> (n[plane, strip], pos[plane, strip])."""
    labA, labB = list(P)
    planes = FIT_PLANES[fit]
    ncol = len(planes) // 2
    fig, axs = plt.subplots(
        4, ncol, figsize=(4.4 * ncol, 10.2), squeeze=False,
        gridspec_kw={"height_ratios": [3, 1, 3, 1], "hspace": 0.28},
    )
    colors = ("tab:blue", "tab:orange")
    for r, letter in enumerate("ZY"):
        for c, plane in enumerate(p for p in planes if p[0] == letter):
            ax, axr = axs[2 * r, c], axs[2 * r + 1, c]
            i = PLANE_INDEX[plane]
            dens = {}
            for lab, col in zip((labA, labB), colors):
                n, pos = P[lab][0][i].astype(float), P[lab][1][i]
                w = _strip_width(pos)
                ok = (n > 0) & np.isfinite(pos)
                total = n[ok].sum()
                if total == 0:
                    continue
                dens[lab] = (n / total / w, ok)
                ax.bar(pos[ok], dens[lab][0][ok], width=w, color=col, alpha=0.4)
                ax.step(pos[ok], dens[lab][0][ok], where="mid", color=col, lw=1.2)
            ax.set_title(f"{plane}: StripsPosition_{plane}", fontsize=9, pad=7)
            ax.grid(True, alpha=0.3, ls="--")
            if c == 0:
                ax.set_ylabel("fraction of hits per unit position")

            axr.axhline(1.0, color="black", lw=1.0)
            if len(dens) == 2:
                nA, nB = P[labA][0][i].astype(float), P[labB][0][i].astype(float)
                ok = dens[labA][1] & dens[labB][1]
                if ok.any():
                    ratio = dens[labB][0][ok] / dens[labA][0][ok]
                    err = ratio * np.sqrt(1.0 / nA[ok] + 1.0 / nB[ok])
                    axr.errorbar(P[labA][1][i][ok], ratio, yerr=err, fmt=".", ms=3,
                                 color="0.25", lw=0.7, capsize=0)
                    m = float(np.clip(np.max(np.abs(np.log(ratio))), np.log(1.5), np.log(10)))
                    axr.set_ylim(np.exp(-m * 1.15), np.exp(m * 1.15))
            axr.set_yscale("log")
            axr.set_yticks([0.5, 1, 2])
            axr.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            axr.minorticks_off()
            axr.grid(True, alpha=0.3, ls="--")
            if c == 0:
                axr.set_ylabel(f"{labB}\n/ {labA}", fontsize=8)
            if r == 1:
                axr.set_xlabel("strip position")

    handles = [Patch(facecolor=col, alpha=0.5, edgecolor=col, label=lab)
               for lab, col in zip((labA, labB), colors)]
    fig.suptitle(title)
    finish_comparison_figure(fig, handles, output_file)


# ============================================================
# ONE (detector, selection, [period]) -> all WP / VES / FS
# ============================================================

def process_set(runs, data, detector, out_base, run_label, period, args):
    """runs: list of run dicts (already filtered). Returns summary rows."""
    summary = []
    keys = sorted({(r["wp"], r["data_type"]) for r in runs})
    for wp, dtype in keys:
        idx = {r["idx"] for r in runs if r["wp"] == wp and r["data_type"] == dtype}
        wdir = out_base / f"WP{wp}" / dtype
        wdir.mkdir(parents=True, exist_ok=True)
        for fit in FITS:
            Hfit = aggregate(data, fit, idx)
            ntracks = int(sum(data[f"{fit}_ntracks"][i] for i in idx))
            if Hfit.sum() == 0:
                continue
            result, rows = analyse_fit(Hfit, fit, args)
            title = make_title(detector, wp, dtype, fit, run_label, period, ntracks)
            tag = f"{fit}_WP{wp}_{dtype}"
            plot_chi2_median(
                result, fit, title,
                wdir / f"chi2_median_per_strip_{tag}.png", args,
            )
            plot_chi2_median(
                result, fit, title,
                wdir / f"chi2_median_per_strip_refonly_{tag}.png", args,
                show_excess=False, show_sigma_lines=False,
            )
            plot_hits(result, fit, title, wdir / f"hits_per_strip_{tag}.png", args)
            plot_hits_chi2(result, fit, title, wdir / f"hits_per_strip_medianchi2_{tag}.png", args)
            if has_positions(data, fit):
                pn, ppos = aggregate_pos(data, fit, idx)
                plot_hits_position(pn, ppos, fit, title,
                                   wdir / f"hits_per_position_{tag}.png")
            else:
                print(f"    {fit}: cache has no position entries, rerun with --rebuild-cache")
            #write_csv(wdir / f"strip_stats_{tag}.csv", rows)
            write_csv(wdir / f"flagged_strips_{tag}.csv", rows, only_flagged=True)

            if has_residuals(data, fit):
                Rfit = aggregate_res(data, fit, idx)
                if Rfit.sum() > 0:
                    rres, rrows = analyse_residuals(Rfit, fit, args)
                    plot_residual_median(
                        rres, fit, title,
                        wdir / f"residual_median_per_strip_{tag}.png", args.min_hits,
                    )
                    plot_residual_hist(
                        Rfit, fit, title,
                        wdir / f"residual_hist_{tag}.png",
                    )
                    write_csv(wdir / f"residual_stats_{tag}.csv", rrows)
                else:
                    print(f"    WP{wp} {dtype} {fit}: no residual entries in this selection")
            else:
                print(f"    {fit}: cache has no residual entries, rerun with --rebuild-cache")

            for r in rows:
                if r["excess"]:
                    summary.append({"detector": detector, "WP": wp, "type": dtype, "fit": fit,
                                    **{k: r[k] for k in ("plane", "strip_id", "n_hits", "direction",
                                                         "chi2_median", "median_ci_lo",
                                                         "median_ci_hi",
                                                         "ref_median", "ratio_to_ref")}})
            nfl = sum(r["excess"] for r in rows)
            print(f"    WP{wp} {dtype} {fit}: {len(idx)} runs, {ntracks:,} tracks, "
                  f"{nfl} excess strips")
    return summary

def residual_comparison(data, fit, idx_sets, labels, args, title, output_file):
    """residual comparison (overlay + difference B-A + histogram overlay)"""
    if not has_residuals(data, fit):
        print(f"    {fit}: cache has no residual entries, rerun with --rebuild-cache")
        return False
    results, Rs = {}, {}
    for lab, idx in zip(labels, idx_sets):
        R = aggregate_res(data, fit, idx)
        if R.sum() == 0:
            print(f"    {fit}: no residual entries for '{lab}', residual comparison skipped")
            return False
        Rs[lab] = R
        results[lab], _ = analyse_residuals(R, fit, args)
    plot_residual_period_comparison(results, fit, title, output_file,
                                    z=args.excess_z, min_hits=args.min_hits)
    plot_residual_hist_period_comparison(
        Rs, fit, title,
        output_file.with_name(output_file.name.replace("residual_median_per_strip",
                                                       "residual_hist")),)
    return True


def position_comparison(data, fit, idx_sets, labels, title, output_file):
    """hit density vs strip position: overlay + ratio B/A"""
    if not has_positions(data, fit):
        print(f"    {fit}: cache has no position entries, rerun with --rebuild-cache")
        return False
    P = {}
    for lab, idx in zip(labels, idx_sets):
        n, pos = aggregate_pos(data, fit, idx)
        if n.sum() == 0:
            print(f"    {fit}: no position entries for '{lab}', position comparison skipped")
            return False
        P[lab] = (n, pos)
    plot_hits_position_period_comparison(P, fit, title, output_file)
    return True


def write_comparison_plots(run_groups, labels, data, args, output_dir,
                           detector, selection, data_types=None):
    """Write comparison plots for two run groups, aggregating all WPs."""
    if data_types is None:
        data_types = [None, None]
    set_runs_metadata(
        labels,
        [[r for r in g if dt is None or r["data_type"] == dt]
         for g, dt in zip(run_groups, data_types)],
    )
    for fit in FITS:
        comparison_results, ntracks, idx_sets = {}, [], []
        for label, group, dtype in zip(labels, run_groups, data_types):
            selected = [r for r in group if dtype is None or r["data_type"] == dtype]
            idx = {r["idx"] for r in selected}
            nt = int(sum(data[f"{fit}_ntracks"][i] for i in idx))
            Hfit = aggregate(data, fit, idx)
            if Hfit.sum() == 0:
                print(f"    {fit}: comparison side '{label}' has no hits, skipping")
                break
            comparison_results[label], _ = analyse_fit(Hfit, fit, args)
            ntracks.append(nt)
            idx_sets.append(idx)
        if len(comparison_results) != 2:
            continue
        comparison_title = (
            f"{detector}, {selection}, {fit} tracks: "
            f"{labels[0]} ({ntracks[0]:,}) vs {labels[1]} ({ntracks[1]:,})"
        )
        tag = f"{fit}" if data_types[0] != data_types[1] else f"{fit}_{data_types[0]}"
        plot_hits_period_comparison(
            comparison_results, fit, comparison_title,
            output_dir / f"hits_per_strip_density_{tag}.png",
        )
        plot_chi2_median_period_comparison(
            comparison_results, fit, comparison_title,
            output_dir / f"chi2_median_per_strip_refonly_{tag}.png",
        )
        residual_comparison(
            data, fit, idx_sets, labels, args, comparison_title,
            output_dir / f"residual_median_per_strip_{tag}.png",
        )
        position_comparison(
            data, fit, idx_sets, labels, comparison_title,
            output_dir / f"hits_per_position_density_{tag}.png",
        )
        print(f"    {fit}: comparison written to {output_dir}")


# ============================================================
# READING ROOT FILES
# ============================================================

def read_all_records(detector, data_dir, run_ranges):
    records = []
    for data_type in ("VES", "FS"):
        print(f"\n========== {detector} / {data_type} ==========")
        for run in iter_run_numbers(run_ranges[data_type]):
            fp = Path(data_dir) / FILENAME_TEMPLATE.format(run=run)
            if not fp.is_file():
                print(f"Run {run}: file not found, skipping")
                continue
            info = get_slowcontrol_info(SLOWCONTROL_DIR, detector, run)
            if info is None:
                print(f"Run {run}: no timestamp/WP, skipping")
                continue
            ts, wp = info
            res = read_run(fp)
            if res is None:
                continue
            print(f"Run {run}: {ts:%d-%m-%Y %H:%M} WP{wp} "
                  f"3p={res['3p'][2]:,} 4p={res['4p'][2]:,} tracks, "
                  f"residual hits 3p={int(res['3p'][4].sum()):,} "
                  f"4p={int(res['4p'][4].sum()):,}")
            records.append({"run": run, "data_type": data_type, "wp": wp,
                            "timestamp": ts, "fit": res})
    return records


# ============================================================
# ARGUMENTS / MAIN
# ============================================================

def parse_arguments():
    p = argparse.ArgumentParser(description="Per-strip chi2 / residual / hit analysis of 3D tracks.")
    p.add_argument("--detector", choices=sorted(RUN_RANGES))
    p.add_argument("--detectors", nargs="+", choices=sorted(RUN_RANGES))
    p.add_argument("--ves-ranges", nargs="+", metavar="START-END")
    p.add_argument("--fs-ranges", nargs="+", metavar="START-END")
    p.add_argument("--period", action="append", type=parse_period, metavar="LABEL=SPEC",
                   help="Period to investigate (repeatable). SPEC = run ranges "
                        "'5000-5200,5300-5350' or dates '2025-03-01:2025-04-15'. "
                        "Goes to <sel>_runs/periods/<LABEL>/ and is written on the figures.")
    p.add_argument("--compare-periods", nargs=2, metavar=("PERIOD_A", "PERIOD_B"),
                   help="Also compare these two named --period values with density=True "
                        "hit-per-strip histograms, chi2 medians (ratio B/A) and residual "
                        "medians (difference B-A).")
    p.add_argument("--compare-periods-any-type", action="store_true",
                   help="With --compare-periods, compare periods regardless of VES/FS type; "
                        "runs from all data types in each period are combined per WP.")
    p.add_argument("--compare-VES-FS", action="store_true",
                   help="Compare the selected VES and FS run ranges (or all configured "
                        "ranges if omitted), aggregating all WPs; only comparison "
                        "plots are produced.")
    p.add_argument("--skip-full", action="store_true",
                   help="Only make the period plots, not the full-range plots")
    p.add_argument("--periods-only", action="store_true",
                   help="Only make the requested period plots and optional comparison; "
                        "same effect as --skip-full")
    p.add_argument("--run-selection", choices=["all", "good", "both"], default="both")
    p.add_argument("--reference", choices=["plane", "projection"], default="plane",
                   help="'general median': pooled over the plane (default) or over all planes "
                        "of the same projection")
    p.add_argument("--excess-z", type=float, default=EXCESS_Z,
                   help=f"CI width of the strip median in sigma (default {EXCESS_Z})")
    p.add_argument("--rel-excess", type=float, default=EXCESS_REL,
                   help=f"min. relative deviation from the reference, f=1+value: median > f*ref (high) "
                        f"or < ref/f (low) (default {EXCESS_REL})")
    p.add_argument("--min-robust-z", type=float, default=0.0,
                   help="optional 4th criterion: strip median must also be >= this many "
                        "robust sigmas (MAD) above (high) / below (low) the other strips of the "
                        "plane (default 0 = off; try 3 for 'real outliers')")
    p.add_argument("--min-hits", type=int, default=EXCESS_MIN_HITS,
                   help=f"min. hits per strip to be judged for an excess (and to enter the "
                        f"residual difference panel / y-range); all strips are drawn "
                        f"(default {EXCESS_MIN_HITS})")
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    p.add_argument("--cache-dir", default=str(CACHE_DIR))
    p.add_argument("--rebuild-cache", action="store_true")
    args = p.parse_args()
    if args.detector and args.detectors:
        p.error("Use either --detector or --detectors, not both.")
    args.skip_full = args.skip_full or args.periods_only
    dets = [args.detector] if args.detector else (args.detectors or ["ROSSO"])
    if args.compare_periods:
        defined = {period["label"] for period in args.period or []}
        missing = [label for label in args.compare_periods if label not in defined]
        if missing:
            p.error("--compare-periods must refer to labels defined by --period: "
                    + ", ".join(missing))
    if args.compare_periods_any_type and not args.compare_periods:
        p.error("--compare-periods-any-type requires --compare-periods.")
    if args.compare_periods and args.compare_VES_FS:
        p.error("--compare-periods and --compare-VES-FS are mutually exclusive.")
    if args.compare_VES_FS and args.period:
        p.error("--compare-VES-FS compares the selected run ranges and cannot be used "
                "with --period.")
    return args, dets, parse_range_list(args.ves_ranges), parse_range_list(args.fs_ranges)


def main():
    args, detectors, ves_o, fs_o = parse_arguments()
    selections = ["all", "good"] if args.run_selection == "both" else [args.run_selection]
    out_root = Path(args.output_dir)

    print("=" * 70)
    print("MURAVES PER-STRIP CHI2 / RESIDUAL ANALYSIS")
    print(f"Detectors: {', '.join(detectors)} | selection: {', '.join(selections)}")
    print(f"Excess: CI(z={args.excess_z}) beyond ref AND median > ref*{1+args.rel_excess:g} "
          f"(high) or < ref/{1+args.rel_excess:g} (low) "
          f"AND N >= {args.min_hits}"
          + (f" AND robust_z >= {args.min_robust_z}" if args.min_robust_z > 0 else "")
          + f"; reference = {args.reference}")
    for per in args.period or []:
        print(f"Period '{per['label']}': {per['desc']}")
    print("=" * 70)

    for detector in detectors:
        print(f"\n{'#' * 70}\nDETECTOR: {detector}\n{'#' * 70}")
        run_ranges = {
            "VES": ves_o if ves_o is not None else RUN_RANGES[detector]["VES"],
            "FS": fs_o if fs_o is not None else RUN_RANGES[detector]["FS"],
        }
        cfile = cache_path(args.cache_dir, detector)
        if cfile.is_file() and not args.rebuild_cache:
            print(f"Loading cached data: {cfile}")
            runs, data = load_cache(cfile)
        else:
            data_dir = Path(DATA_DIR_TEMPLATE.format(detector=detector))
            records = read_all_records(detector, data_dir, run_ranges)
            if not records:
                print(f"No usable runs for {detector}, skipping.")
                continue
            save_cache(cfile, records)
            runs, data = load_cache(cfile)

        if not all(has_residuals(data, fit) for fit in FITS):
            print("  [warn] the cache has no residual histograms (written by an older "
                  "version): residual plots are skipped. Rerun with --rebuild-cache.")
        if not all(has_positions(data, fit) for fit in FITS):
            print("  [warn] the cache has no strip-position data (written by an older "
                  "version or StripsPosition_* missing): position plots are skipped. "
                  "Rerun with --rebuild-cache.")

        # run range selection applied AFTER the cache
        runs = [r for r in runs
                if any(lo <= r["run"] <= hi for lo, hi in run_ranges[r["data_type"]])]
        if not runs:
            print(f"No runs in the requested ranges for {detector}.")
            continue

        good = get_good_run_set(detector) if "good" in selections else None

        for selection in selections:
            sel_runs = [r for r in runs if selection == "all" or r["run"] in good]
            if not sel_runs:
                print(f"No runs for selection '{selection}'.")
                continue
            run_label = "good runs only" if selection == "good" else "all runs"
            base = out_root / detector / f"{selection}_runs"
            summary_all = []
            period_runs = {}

            if args.compare_VES_FS:
                ves_runs = [r for r in sel_runs if r["data_type"] == "VES"]
                fs_runs = [r for r in sel_runs if r["data_type"] == "FS"]
                compare_dir = base / "period_comparisons" / "VES_vs_FS"
                compare_dir.mkdir(parents=True, exist_ok=True)
                if ves_runs and fs_runs:
                    write_runs_file(compare_dir / "runs_used.txt",
                                    ["VES", "FS"], [ves_runs, fs_runs])
                    print(f"\n{detector} {selection}: comparing VES vs FS "
                          f"({len(ves_runs)} vs {len(fs_runs)} runs)")
                    write_comparison_plots(
                        [ves_runs, fs_runs], ["VES", "FS"], data, args,
                        compare_dir, detector, selection,
                        data_types=["VES", "FS"],
                    )
                else:
                    print(f"\n{detector} {selection}: no VES/FS pair available, "
                          "skipping comparison")
                continue

            if not args.skip_full:
                print(f"\n{detector} {selection}: full range ({len(sel_runs)} runs)")
                s = process_set(sel_runs, data, detector, base / "full", run_label, None, args)
                write_csv(base / "full" / "flagged_strips_summary.csv", s)
                summary_all += s

            for per in args.period or []:
                p_runs = [r for r in sel_runs if run_in_period(r["run"], r["timestamp"], per)]
                period_runs[per["label"]] = p_runs
                print(f"\n{detector} {selection}: period '{per['label']}' "
                      f"({len(p_runs)} runs)")
                if not p_runs:
                    print("  no runs in this period")
                    continue
                pdir = base / "periods" / per["label"]
                s = process_set(p_runs, data, detector, pdir, run_label, per, args)
                write_csv(pdir / "flagged_strips_summary.csv", s)

            if args.compare_periods and args.compare_periods_any_type:
                labels = args.compare_periods
                comparison_runs = [period_runs[label] for label in labels]
                if all(comparison_runs):
                    print(f"\n{detector} {selection}: comparing periods across data types "
                          f"'{labels[0]}' vs '{labels[1]}'")
                    compare_dir = base / "period_comparisons" / f"{labels[0]}_vs_{labels[1]}"
                    compare_dir.mkdir(parents=True, exist_ok=True)
                    write_runs_file(compare_dir / "runs_used.txt", labels, comparison_runs)
                    # Per-WP comparisons: pool FS and VES together within each period.
                    wps = sorted({r["wp"] for group in comparison_runs for r in group})
                    for wp in wps:
                        wp_groups = [[r for r in group if r["wp"] == wp]
                                     for group in comparison_runs]
                        if not all(wp_groups):
                            print(f"    WP{wp}: no runs in one of the periods, skipping")
                            continue
                        wdir = compare_dir / f"WP{wp}" / "any_data_type"
                        wdir.mkdir(parents=True, exist_ok=True)
                        set_runs_metadata(labels, wp_groups)
                        for fit in FITS:
                            comparison_results, ntracks, idx_sets = {}, [], []
                            for label, group in zip(labels, wp_groups):
                                idx = {r["idx"] for r in group}
                                nt = int(sum(data[f"{fit}_ntracks"][i] for i in idx))
                                Hfit = aggregate(data, fit, idx)
                                if Hfit.sum() == 0:
                                    print(f"    WP{wp} {fit}: period '{label}' has no {fit} hits, skipping comparison")
                                    break
                                comparison_results[label], _ = analyse_fit(Hfit, fit, args)
                                ntracks.append(nt)
                                idx_sets.append(idx)
                            if len(comparison_results) != 2:
                                continue
                            types_a = "/".join(sorted({r["data_type"] for r in wp_groups[0]}))
                            types_b = "/".join(sorted({r["data_type"] for r in wp_groups[1]}))
                            comparison_title = (
                                f"{detector}, WP{wp}, {fit} tracks: "
                                f"{labels[0]} [{types_a}] ({ntracks[0]:,}) vs "
                                f"{labels[1]} [{types_b}] ({ntracks[1]:,})"
                            )
                            tag = f"{fit}_WP{wp}_any_data_type"
                            plot_hits_period_comparison(
                                comparison_results, fit, comparison_title,
                                wdir / f"hits_per_strip_density_{tag}.png"
                            )
                            plot_chi2_median_period_comparison(
                                comparison_results, fit, comparison_title,
                                wdir / f"chi2_median_per_strip_refonly_{tag}.png"
                            )
                            residual_comparison(
                                data, fit, idx_sets, labels, args, comparison_title,
                                wdir / f"residual_median_per_strip_{tag}.png"
                            )
                            position_comparison(
                                data, fit, idx_sets, labels, comparison_title,
                                wdir / f"hits_per_position_density_{tag}.png"
                            )
                            print(f"    WP{wp} {fit}: cross-type comparison written to {wdir}")
                    # Also produce one comparison with all WPs and data types pooled.
                    all_wp_dir = compare_dir / "all_WPs_any_data_type"
                    all_wp_dir.mkdir(parents=True, exist_ok=True)
                    write_comparison_plots(
                        comparison_runs, labels, data, args, all_wp_dir,
                        detector, selection, data_types=[None, None],
                    )

            if args.compare_periods and not args.compare_periods_any_type:
                labels = args.compare_periods
                comparison_runs = [period_runs[label] for label in labels]
                if all(comparison_runs):
                    print(f"\n{detector} {selection}: comparing periods "
                          f"'{labels[0]}' vs '{labels[1]}'")
                    compare_dir = base / "period_comparisons" / f"{labels[0]}_vs_{labels[1]}"
                    compare_dir.mkdir(parents=True, exist_ok=True)
                    write_runs_file(compare_dir / "runs_used.txt", labels, comparison_runs)
                    for wp, dtype in sorted({(r["wp"], r["data_type"])
                        for p_runs in comparison_runs for r in p_runs
                    }):
                        missing = [lab for lab, p_runs in zip(labels, comparison_runs)
                                if not any(r["wp"] == wp and r["data_type"] == dtype for r in p_runs)]
                        if missing:
                            print(f"    WP{wp} {dtype}: no runs in period(s) {missing}, skipping comparison")
                            continue
                        wdir = compare_dir / f"WP{wp}" / dtype
                        wdir.mkdir(parents=True, exist_ok=True)
                        set_runs_metadata(
                            labels,
                            [[r for r in p_runs if r["wp"] == wp and r["data_type"] == dtype]
                             for p_runs in comparison_runs],
                        )
                        for fit in FITS:
                            comparison_results, ntracks, idx_sets = {}, [], []
                            for label, p_runs in zip(labels, comparison_runs):
                                idx = {r["idx"] for r in p_runs
                                    if r["wp"] == wp and r["data_type"] == dtype}
                                nt = int(sum(data[f"{fit}_ntracks"][i] for i in idx))
                                Hfit = aggregate(data, fit, idx)
                                if Hfit.sum() == 0:
                                    print(f"    WP{wp} {dtype} {fit}: period '{label}' has "
                                        f"{len(idx)} runs but no {fit} hits, skipping comparison")
                                    break
                                comparison_results[label], _ = analyse_fit(Hfit, fit, args)
                                ntracks.append(nt)
                                idx_sets.append(idx)
                            if len(comparison_results) != 2:
                                continue
                            comparison_title = (
                                f"{detector}, WP{wp}, {dtype}, {fit} tracks: "
                                f"{labels[0]} ({ntracks[0]:,}) vs {labels[1]} ({ntracks[1]:,})"
                            )
                            tag = f"{fit}_WP{wp}_{dtype}"
                            plot_hits_period_comparison(
                                comparison_results, fit, comparison_title,
                                wdir / f"hits_per_strip_density_{tag}.png"
                            )
                            plot_chi2_median_period_comparison(
                                comparison_results, fit, comparison_title,
                                wdir / f"chi2_median_per_strip_refonly_{tag}.png"
                            )
                            residual_comparison(
                                data, fit, idx_sets, labels, args, comparison_title,
                                wdir / f"residual_median_per_strip_{tag}.png",
                            )
                            position_comparison(
                                data, fit, idx_sets, labels, comparison_title,
                                wdir / f"hits_per_position_density_{tag}.png"
                            )
                            print(f"    WP{wp} {dtype} {fit}: comparison written")

                    # Also write a comparison with all WPs combined, while
                    # retaining separate VES and FS outputs.
                    for dtype in ("VES", "FS"):
                        aggregate_dir = compare_dir / dtype
                        aggregate_dir.mkdir(parents=True, exist_ok=True)
                        write_comparison_plots(
                            comparison_runs, labels, data, args, aggregate_dir,
                            detector, selection, data_types=[dtype, dtype],
                        )

        print(f"\nOutput directory: {out_root / detector}")

    print("\nDONE")


if __name__ == "__main__":
    main()