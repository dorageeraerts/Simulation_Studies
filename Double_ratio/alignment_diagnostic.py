#!/usr/bin/env python3
"""
MURAVES run diagnostics: WHY did the residuals / chi2 change between two periods?

One number per run and per plane (a few dozen "health indicators") is computed
from the AnalyzedData files, plotted against run number, and compared between
two periods. The comparison table is sorted by how much each indicator moved
compared to its own run-to-run scatter, so the first lines tell you where to look.

Indicators (all from the 3p best track unless stated; plane 4 uses the best 4p track)
-------------------------------------------------------------------------------
RESIDUAL (misalignment vs resolution)
  res_med        median residual [mm]                 -> translation of a plane
  res_sig        robust sigma (q84-q16)/2 [mm]        -> width (resolution / jitter)
  res_tail       fraction beyond 3 robust sigma       -> tails (noise, wrong cluster)
  res_vs_slope   d(res)/d(track slope) [mm]           -> z-shift / tilt of the plane
  res_vs_pos     d(res)/d(cluster position) [mm/m]    -> rotation / scale of the plane
CLUSTER (gain / noise / timing; belongs to the clusters used by the track)
  csize_mean     mean cluster size (strips)           -> gain / threshold / HV
  energy_med     median cluster energy                -> gain / HV / gas / temperature
  texp_med       median cluster time Texp (0 = none)  -> timing / latency drift
  texp_sig       robust sigma of Texp                 -> timing jitter
  texp_zero      fraction of clusters with Texp == 0  -> timing information lost
OCCUPANCY (all events)
  nclus          mean number of clusters per event    -> noise / hot channels
  hasclus        fraction of events with >= 1 cluster -> plane alive / efficiency proxy
GLOBAL
  trk_per_event, chi2_xz_med, chi2_xy_med, theta_med, theta_sd (acceptance changes),
  p4_match_frac_Z/Y (fraction of 3p tracks pointing at plane 4 that get a 4p track),
  TriggerRate, Temperature, WP (from the tree if non-zero, else from slow control)

Outputs in <output-dir>/<detector>/
  run_summary.csv                 one row per run, all indicators
  trends/<indicator>.png          indicator vs run number, periods shaded
  trends/global.png
  compare_<A>_vs_<B>/ranking.csv  all indicators ranked by shift / run-to-run scatter
  compare_<A>_vs_<B>/residual_vs_slope.png, residual_vs_position.png

Example
-------
  python muraves_run_diagnostics.py --detector ROSSO \
      --period A=4000-4500 --period B=9000-9500
  (compare periods of the same data type; VES vs FS differ by construction)
"""

import argparse
import csv
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import awkward as ak
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# CONFIGURATION (same conventions as the chi2 strip script)
# ----------------------------------------------------------------------------
DATA_DIR_TEMPLATE = "/pnfs/iihe/muraves/muraves_DATA/RECONSTRUCTED/{detector}/v1"
SLOWCONTROL_DIR = "/pnfs/iihe/muraves/muraves_DATA/DECOMPRESSED"
FILENAME_TEMPLATE = "MURAVES_AnalyzedData_run{run}.root"
SLOWCONTROL_TEMPLATE = "SLOWCONTROL_run{run}"
TREE_NAME = "AnalyzedData"

RUN_RANGES = {
    "NERO": {"VES": [(2500, 5600), (7894, 11990)], "FS": [(5636, 7893)]},
    "ROSSO": {"VES": [(4000, 10060), (13458, 14792)], "FS": [(10061, 13457)]},
    "BLU": {"VES": [(2800, 8818)], "FS": [(8819, 9925)]},
}

PLANES = ["Z1", "Z2", "Z3", "Z4", "Y1", "Y2", "Y3", "Y4"]
PROJ = {"Z": "xz", "Y": "xy"}                       # plane letter -> projection

# slow-control columns (tab/space separated), see reco_parameters.json
SC_TEMPERATURE, SC_WP, SC_RATE = 3, 5, 43

# binned profiles of the residual (kept per run so periods can be summed)
SLOPE_EDGES = np.linspace(-1.6, 1.6, 17)
POS_EDGES = np.linspace(-0.55, 0.55, 23)

# (name, label) of the per-plane indicators
PLANE_METRICS = [
    ("res_med", "median residual [mm]"),
    ("res_sig", "robust residual sigma [mm]"),
    ("res_tail", "fraction beyond 3 sigma"),
    ("res_vs_slope", "d(residual)/d(slope) [mm]"),
    ("res_vs_pos", "d(residual)/d(position) [mm/m]"),
    ("csize_mean", "mean cluster size (track clusters)"),
    ("energy_med", "median cluster energy (track clusters)"),
    ("texp_med", "median cluster Texp (track clusters)"),
    ("texp_sig", "robust sigma of Texp (track clusters)"),
    ("texp_zero", "fraction with Texp = 0 (track clusters)"),
    ("nclus", "mean clusters per event"),
    ("hasclus", "fraction of events with a cluster"),
]
GLOBAL_METRICS = ["trk_per_event", "chi2_xz_med", "chi2_xy_med", "theta_med", "theta_sd",
                  "p4_match_frac_Z", "p4_match_frac_Y", "TriggerRate", "Temperature", "WP"]

TYPE_COLOR = {"VES": "tab:blue", "FS": "tab:orange", "NA": "0.4"}
PERIOD_COLORS = ["#2ca02c", "#d62728", "#9467bd", "#8c564b"]


# ----------------------------------------------------------------------------
# ARGUMENTS
# ----------------------------------------------------------------------------
def parse_ranges(text):
    out = []
    for part in text.split(","):
        a, _, b = part.strip().partition("-")
        out.append((int(a), int(b or a)))
    return out


def parse_period(text):
    label, _, spec = text.partition("=")
    if not spec:
        raise argparse.ArgumentTypeError("period must be LABEL=START-END[,START-END]")
    return {"label": re.sub(r"[^\w.\-]+", "_", label.strip()), "runs": parse_ranges(spec)}


def parse_arguments():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--detector", required=True, choices=sorted(RUN_RANGES))
    p.add_argument("--runs", type=parse_ranges, default=None,
                   help="run ranges to read, e.g. 4000-4500,9000-9500 "
                        "(default: all configured VES and FS ranges)")
    p.add_argument("--period", action="append", type=parse_period, default=[],
                   metavar="LABEL=START-END", help="repeatable; shaded in the trends")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"),
                   help="labels of two periods to compare (default: the first two)")
    p.add_argument("--min-tracks", type=int, default=500,
                   help="runs with fewer 3p tracks are ignored in plots/ranking (default 500)")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--slowcontrol-dir", default=SLOWCONTROL_DIR)
    p.add_argument("--output-dir", default="./run_diagnostics")
    p.add_argument("--cache-dir", default="./run_diagnostics_cache")
    p.add_argument("--rebuild-cache", action="store_true")
    return p.parse_args()


# ----------------------------------------------------------------------------
# READING ONE RUN
# ----------------------------------------------------------------------------
def needed_branches():
    b = ["timestamp", "WorkingPoint", "Temperature", "TriggerRate", "Theta_3p"]
    for p in PROJ.values():
        b += [f"BestTrack_3p_ChiSquare_{p}", f"BestTrack_4p_ChiSquare_{p}",
              f"BestTrack_3p_{p}_index", f"Track_3p_of_4p_index_{p}", f"Slope_3p_{p}",
              f"Plane4th_isIntercepted_{p}", f"chiSquare_4p_{p}",
              f"cluster_c4_index_{p}", f"displacement_p4_{p}"]
    for pl in PLANES:
        b += [f"Nclusters_{pl}", f"ClusterSize_{pl}", f"ClusterEnergy_{pl}",
              f"ClusterPosition_{pl}", f"Cluster{pl}_Texp"]
    for k in (1, 2, 3):
        b += [f"TrackCluster_z{k}_index", f"TrackCluster_y{k}_index",
              f"Residue_Track3p_z{k}", f"Residue_Track3p_y{k}"]
    return b


def pick(jag, idx, dtype=np.int64, fill=-1):
    """element idx[event] of the jagged array jag[event] (fill if not available)"""
    mask = ak.local_index(jag, axis=1) == idx[:, None]
    return ak.to_numpy(ak.fill_none(ak.firsts(jag[mask], axis=1), fill)).astype(dtype)


def pick_match(values, flat_chi, best, dtype=np.int64, fill=-1):
    """value of the 4p candidate whose chi2 equals the best 4p chi2 of the event"""
    mask = flat_chi == best[:, None]
    return ak.to_numpy(ak.fill_none(ak.firsts(values[mask], axis=1), fill)).astype(dtype)


def robust(x):
    """median and (q84 - q16)/2"""
    q16, med, q84 = np.percentile(x, [16, 50, 84])
    return med, 0.5 * (q84 - q16)


def hist_sums(x, y, edges):
    c, _ = np.histogram(x, edges)
    s, _ = np.histogram(x, edges, weights=y)
    q, _ = np.histogram(x, edges, weights=y * y)
    return c, s, q


def fill_plane(out, prof, plane, res, slope, pos, size, energy, texp):
    """per-plane indicators from per-track arrays (NaN = no cluster)"""
    ok = np.isfinite(res)
    n = int(ok.sum())
    out[f"n_{plane}"] = n
    if n < 20:
        return
    r = res[ok] * 1e3                                  # m -> mm
    med, sig = robust(r)
    out[f"res_med_{plane}"], out[f"res_sig_{plane}"] = med, sig
    out[f"res_tail_{plane}"] = float(np.mean(np.abs(r - med) > 3 * sig)) if sig > 0 else np.nan

    core = np.abs(r - med) < 6 * sig if sig > 0 else np.ones(n, bool)
    sl, ps = slope[ok][core], pos[ok][core]
    rc = r[core]
    if len(rc) >= 50 and np.ptp(sl) > 0:
        out[f"res_vs_slope_{plane}"] = np.polyfit(sl, rc, 1)[0]
    if len(rc) >= 50 and np.ptp(ps) > 0:
        out[f"res_vs_pos_{plane}"] = np.polyfit(ps, rc, 1)[0]
    prof[plane] = {"slope": hist_sums(sl, rc, SLOPE_EDGES), "pos": hist_sums(ps, rc, POS_EDGES)}

    out[f"csize_mean_{plane}"] = float(np.nanmean(size[ok]))
    out[f"energy_med_{plane}"] = float(np.nanmedian(energy[ok]))
    t = texp[ok]
    out[f"texp_zero_{plane}"] = float(np.mean(t == 0))
    if (t > 0).sum() >= 20:
        tm, ts = robust(t[t > 0])
        out[f"texp_med_{plane}"], out[f"texp_sig_{plane}"] = tm, ts


def summarize(arr):
    """arr: awkward array of one run -> (dict of scalars, dict of profiles)"""
    out, prof = {}, {}
    n_ev = len(arr)
    out["n_events"] = n_ev
    for pl in PLANES:                                   # occupancy over ALL events
        ncl = ak.to_numpy(arr[f"Nclusters_{pl}"]).astype(float)
        out[f"nclus_{pl}"] = float(ncl.mean())
        out[f"hasclus_{pl}"] = float((ncl > 0).mean())

    chi3 = {p: ak.to_numpy(arr[f"BestTrack_3p_ChiSquare_{p}"]).astype(float) for p in PROJ.values()}
    chi4 = {p: ak.to_numpy(arr[f"BestTrack_4p_ChiSquare_{p}"]).astype(float) for p in PROJ.values()}
    sel3 = np.all([np.isfinite(c) & (c > 0) for c in chi3.values()], axis=0)
    sel4 = np.all([np.isfinite(c) & (c > 0) for c in chi4.values()], axis=0)
    out["n_3p"], out["n_4p"] = int(sel3.sum()), int(sel4.sum())
    out["trk_per_event"] = out["n_3p"] / max(n_ev, 1)
    if out["n_3p"] == 0:
        return out, prof
    out["chi2_xz_med"] = float(np.median(chi3["xz"][sel3]))
    out["chi2_xy_med"] = float(np.median(chi3["xy"][sel3]))
    theta = ak.to_numpy(arr["Theta_3p"]).astype(float)[sel3]
    out["theta_med"], out["theta_sd"] = float(np.median(theta)), float(np.std(theta))

    a = arr[sel3]
    t3 = {p: ak.to_numpy(a[f"BestTrack_3p_{p}_index"]).astype(np.int64) for p in PROJ.values()}
    for letter, proj in PROJ.items():
        slope = pick(a[f"Slope_3p_{proj}"], t3[proj], float, np.nan)
        for k in (1, 2, 3):
            pl = f"{letter}{k}"
            cl = pick(a[f"TrackCluster_{letter.lower()}{k}_index"], t3[proj])
            res = pick(a[f"Residue_Track3p_{letter.lower()}{k}"], t3[proj], float, np.nan)
            res = np.where(cl >= 0, res, np.nan)
            fill_plane(out, prof, pl, res, slope,
                       pick(a[f"ClusterPosition_{pl}"], cl, float, np.nan),
                       pick(a[f"ClusterSize_{pl}"], cl, float, np.nan),
                       pick(a[f"ClusterEnergy_{pl}"], cl, float, np.nan),
                       pick(a[f"Cluster{pl}_Texp"], cl, float, np.nan))
        # plane 4: fraction of 3p tracks pointing at plane 4 that obtained a 4p track
        flag = pick(a[f"Plane4th_isIntercepted_{proj}"], t3[proj]) == 1
        found = chi4[proj][sel3] > 0
        out[f"p4_match_frac_{letter}"] = float(found[flag].mean()) if flag.any() else np.nan

    if out["n_4p"] >= 20:                               # plane 4 from the 4p sample
        a4 = arr[sel4]
        t3p = {p: ak.to_numpy(a4[f"Track_3p_of_4p_index_{p}"]).astype(np.int64) for p in PROJ.values()}
        for letter, proj in PROJ.items():
            pl = f"{letter}4"
            best = chi4[proj][sel4]
            cl = pick_match(a4[f"cluster_c4_index_{proj}"], a4[f"chiSquare_4p_{proj}"], best)
            res = pick_match(a4[f"displacement_p4_{proj}"], a4[f"chiSquare_4p_{proj}"], best,
                             float, np.nan)
            res = np.where(cl >= 0, res, np.nan)        # = plane-4 cluster minus 3p extrapolation
            fill_plane(out, prof, pl, res,
                       pick(a4[f"Slope_3p_{proj}"], t3p[proj], float, np.nan),
                       pick(a4[f"ClusterPosition_{pl}"], cl, float, np.nan),
                       pick(a4[f"ClusterSize_{pl}"], cl, float, np.nan),
                       pick(a4[f"ClusterEnergy_{pl}"], cl, float, np.nan),
                       pick(a4[f"Cluster{pl}_Texp"], cl, float, np.nan))
    return out, prof


def slowcontrol(sc_dir, detector, run):
    """(temperature, WP, trigger rate) from the first slow-control line, NaN if unavailable"""
    f = Path(sc_dir) / detector / SLOWCONTROL_TEMPLATE.format(run=run)
    try:
        with open(f, errors="replace") as fh:
            for line in fh:
                fld = line.split()
                if fld:
                    return tuple(float(fld[i]) if len(fld) > i else np.nan
                                 for i in (SC_TEMPERATURE, SC_WP, SC_RATE))
    except (OSError, ValueError):
        pass
    return (np.nan,) * 3


def read_run(fp, detector, run, sc_dir):
    try:
        with uproot.open(fp) as f:
            tree = f[TREE_NAME]
            miss = set(needed_branches()) - set(tree.keys())
            if miss:
                print(f"  [warn] run {run}: missing branches {sorted(miss)[:4]}..., skipped")
                return None
            arr = tree.arrays(needed_branches(), library="ak")
    except Exception as exc:
        print(f"  [warn] run {run}: cannot read ({exc})")
        return None
    out, prof = summarize(arr)
    # tree values if filled, otherwise slow control
    temp, wp, rate = (float(np.median(ak.to_numpy(arr[k]))) for k in
                      ("Temperature", "WorkingPoint", "TriggerRate"))
    sc = slowcontrol(sc_dir, detector, run)
    out["Temperature"] = temp if temp != 0 else sc[0]
    out["WP"] = wp if wp != 0 else sc[1]
    out["TriggerRate"] = rate if rate != 0 else sc[2]
    ts = ak.to_numpy(arr["timestamp"])
    out["timestamp_ms"] = float(np.median(ts[ts > 0])) if (ts > 0).any() else np.nan
    return {"scalars": out, "prof": prof}


# ----------------------------------------------------------------------------
# CACHE + TABLE
# ----------------------------------------------------------------------------
def load_runs(args, wanted):
    cfile = Path(args.cache_dir) / f"run_diagnostics_{args.detector}.pkl"
    cache = {}
    if cfile.is_file() and not args.rebuild_cache:
        with open(cfile, "rb") as f:
            cache = pickle.load(f)
    data_dir = Path(args.data_dir or DATA_DIR_TEMPLATE.format(detector=args.detector))
    new = 0
    for run in wanted:
        if run in cache:
            continue
        fp = data_dir / FILENAME_TEMPLATE.format(run=run)
        if not fp.is_file():
            continue
        res = read_run(fp, args.detector, run, args.slowcontrol_dir)
        if res is None:
            continue
        cache[run] = res
        new += 1
        s = res["scalars"]
        print(f"run {run}: {s['n_events']:,} events, {s['n_3p']:,} 3p / {s['n_4p']:,} 4p tracks")
    if new:
        cfile.parent.mkdir(parents=True, exist_ok=True)
        with open(cfile, "wb") as f:
            pickle.dump(cache, f)
    return {r: cache[r] for r in wanted if r in cache}


def data_type_of(detector, run):
    for t, ranges in RUN_RANGES[detector].items():
        if any(lo <= run <= hi for lo, hi in ranges):
            return t
    return "NA"


def in_ranges(run, ranges):
    return any(lo <= run <= hi for lo, hi in ranges)


# ----------------------------------------------------------------------------
# PLOTS
# ----------------------------------------------------------------------------
def shade(ax, periods, ymin=None):
    for per, col in zip(periods, PERIOD_COLORS):
        for lo, hi in per["runs"]:
            ax.axvspan(lo - 0.5, hi + 0.5, color=col, alpha=0.15, lw=0)


def scatter_by_type(ax, df, col):
    for t, g in df.groupby("data_type"):
        ax.plot(g["run"], g[col], ".", ms=4, color=TYPE_COLOR.get(t, "0.4"), label=t)


def legend_handles(periods, df):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Line2D([0], [0], marker=".", ls="", color=TYPE_COLOR.get(t, "0.4"), label=t)
         for t in sorted(df["data_type"].unique())]
    h += [Patch(color=c, alpha=0.3, label=p["label"]) for p, c in zip(periods, PERIOD_COLORS)]
    return h


def plot_trends(df, periods, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    for name, label in PLANE_METRICS:
        fig, axs = plt.subplots(2, 4, figsize=(17, 6.5), sharex=True, squeeze=False)
        for i, pl in enumerate(PLANES):
            ax, col = axs[i // 4, i % 4], f"{name}_{pl}"
            ax.set_title(pl, fontsize=10)
            ax.grid(True, alpha=0.3, ls="--")
            shade(ax, periods)
            if col in df and df[col].notna().any():
                scatter_by_type(ax, df, col)
            if i % 4 == 0:
                ax.set_ylabel(label, fontsize=8)
            if i // 4 == 1:
                ax.set_xlabel("run")
        fig.suptitle(f"{name}: {label}")
        fig.legend(handles=legend_handles(periods, df), loc="lower center",
                   ncol=6, fontsize=9)
        fig.tight_layout(rect=[0, 0.04, 1, 0.97])
        fig.savefig(outdir / f"{name}.png", dpi=110)
        plt.close(fig)

    fig, axs = plt.subplots(2, 5, figsize=(20, 6.5), sharex=True, squeeze=False)
    for ax, col in zip(axs.flat, GLOBAL_METRICS):
        ax.set_title(col, fontsize=10)
        ax.grid(True, alpha=0.3, ls="--")
        shade(ax, periods)
        if col in df and df[col].notna().any():
            scatter_by_type(ax, df, col)
    fig.legend(handles=legend_handles(periods, df), loc="lower center", ncol=6, fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(outdir / "global.png", dpi=110)
    plt.close(fig)


def indicator_columns(df):
    return [c for c in df.columns
            if any(c.startswith(m + "_") for m, _ in PLANE_METRICS) or c in GLOBAL_METRICS]


def compare_table(df, A, B):
    rows = []
    for col in indicator_columns(df):
        a, b = df.loc[df["period"] == A, col].dropna(), df.loc[df["period"] == B, col].dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        pooled = np.sqrt(0.5 * (a.var() + b.var()))
        if not np.isfinite(pooled) or pooled == 0:
            continue
        diff = b.mean() - a.mean()
        err = np.sqrt(a.var() / len(a) + b.var() / len(b))
        rows.append({"indicator": col, "mean_A": a.mean(), "mean_B": b.mean(),
                     "diff_B_minus_A": diff, "run_scatter": pooled,
                     "shift_over_scatter": diff / pooled,
                     "significance": diff / err if err > 0 else np.nan,
                     "n_runs_A": len(a), "n_runs_B": len(b)})
    t = pd.DataFrame(rows)
    if len(t):
        t = t.reindex(t["shift_over_scatter"].abs().sort_values(ascending=False).index)
    return t


def plot_profiles(prof_by_period, labels, kind, edges, xlabel, title, outfile):
    fig, axs = plt.subplots(2, 4, figsize=(17, 6.5), sharex=True, squeeze=False)
    centers = 0.5 * (edges[1:] + edges[:-1])
    for i, pl in enumerate(PLANES):
        ax = axs[i // 4, i % 4]
        ax.set_title(pl, fontsize=10)
        ax.grid(True, alpha=0.3, ls="--")
        ax.axhline(0, color="k", lw=0.7)
        for lab, col in zip(labels, ("tab:blue", "tab:orange")):
            c = s = q = 0
            for pr in prof_by_period[lab]:
                if pl in pr:
                    cc, ss, qq = pr[pl][kind]
                    c, s, q = c + cc, s + ss, q + qq
            if np.isscalar(c):
                continue
            ok = c >= 20
            mean = np.where(ok, s / np.maximum(c, 1), np.nan)
            var = np.where(ok, q / np.maximum(c, 1) - mean ** 2, np.nan)
            ax.errorbar(centers[ok], mean[ok], yerr=np.sqrt(var[ok] / c[ok]), fmt=".-",
                        ms=4, lw=0.8, color=col, label=lab)
        if i % 4 == 0:
            ax.set_ylabel("mean residual [mm]")
        if i // 4 == 1:
            ax.set_xlabel(xlabel)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outfile, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    args = parse_arguments()
    det = args.detector
    ranges = args.runs or (RUN_RANGES[det]["VES"] + RUN_RANGES[det]["FS"])
    wanted = sorted({r for lo, hi in ranges for r in range(lo, hi + 1)})
    results = load_runs(args, wanted)
    if not results:
        print("No usable runs found.")
        return

    rows = []
    for run, res in results.items():
        row = {"run": run, "data_type": data_type_of(det, run), **res["scalars"]}
        row["period"] = next((p["label"] for p in args.period if in_ranges(run, p["runs"])), "")
        rows.append(row)
    full = pd.DataFrame(rows).sort_values("run").reset_index(drop=True)
    outdir = Path(args.output_dir) / det
    outdir.mkdir(parents=True, exist_ok=True)
    full.to_csv(outdir / "run_summary.csv", index=False, float_format="%.5g")

    df = full[full["n_3p"] >= args.min_tracks].copy()
    print(f"\n{len(full)} runs read, {len(df)} with >= {args.min_tracks} 3p tracks")
    plot_trends(df, args.period, outdir / "trends")
    print(f"Trend plots: {outdir / 'trends'}")

    labels = args.compare or [p["label"] for p in args.period[:2]]
    if len(labels) < 2:
        print("Give two --period to get the comparison.")
        return
    A, B = labels
    for lab in (A, B):
        g = df[df["period"] == lab]
        print(f"  period {lab}: {len(g)} runs, types {dict(g['data_type'].value_counts())}, "
              f"WP {sorted(g['WP'].dropna().unique().tolist())}")
    cdir = outdir / f"compare_{A}_vs_{B}"
    cdir.mkdir(parents=True, exist_ok=True)

    table = compare_table(df, A, B)
    table.to_csv(cdir / "ranking.csv", index=False, float_format="%.5g")
    print("\nIndicators ranked by |shift| / run-to-run scatter (top 20):")
    with pd.option_context("display.width", 200, "display.float_format", "{:.4g}".format):
        print(table.head(20)[["indicator", "mean_A", "mean_B", "diff_B_minus_A",
                              "shift_over_scatter", "significance"]].to_string(index=False))

    prof = {lab: [results[r]["prof"] for r in df.loc[df["period"] == lab, "run"]]
            for lab in (A, B)}
    plot_profiles(prof, (A, B), "slope", SLOPE_EDGES, "track slope",
                  f"{det}: residual vs track slope (z-shift / tilt shows as a slope)",
                  cdir / "residual_vs_slope.png")
    plot_profiles(prof, (A, B), "pos", POS_EDGES, "cluster position [m]",
                  f"{det}: residual vs cluster position (rotation / scale shows as a slope)",
                  cdir / "residual_vs_position.png")
    print(f"\nComparison written to {cdir}")


if __name__ == "__main__":
    main()