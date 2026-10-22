#!/usr/bin/env python3
"""
MURAVES chi2 morphing (quantile mapping): VES <-> FS, per detector and WP.

For every (detector, WP, fit [3p/4p], projection [xy/xz]) a monotonic map
    chi2' = F_target^-1( F_source(chi2) )
is derived and applied to the source tracks. No reweighting: every track
keeps its identity, only its chi2 value changes.

Two modes (--mode):
  control (default): the map is derived ONLY in a free-sky control region
        (CONTROL_THETA x CONTROL_PHI, same for all detectors), where VES and
        FS tracks are the same population, and then applied to ALL source
        tracks. Checks: consistency of the map inside the control region,
        held-out-run closure, angular coverage, and KS per angular bin.
  binned: maps per (direction x theta bin) from all tracks; assumes VES and
        FS tracks in the same bin are the same population.

By default only forward tracks (90 <= phi <= 270 deg) are used. For 4p the
map nodes are limited to chi2 <= --range-4p (default 100); quantiles are
still taken from the full samples, so no truncation bias is introduced.

Input : the compact cache written by the chi2 distribution script
        (chi2_cache_<DETECTOR>.npz). ROOT / slow control are not needed.
Output: <out>/<DETECTOR>/morph_maps_<SRC>_to_<TGT>.json   (the maps)
        <out>/<DETECTOR>/plots/*.png                       (validation)
        <out>/<DETECTOR>/validation_<SRC>_to_<TGT>.csv     (KS, efficiencies)

Use the maps downstream with load_maps() + apply_morph() (see bottom).
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# ============================================================
# CONFIGURATION
# ============================================================

CACHE_DIR = Path("./chi2_distribution_cache")
OUT_DIR = Path("./chi2_morphing")

FITS = ("3p", "4p")
PROJECTIONS = ("xy", "xz")

PHI_FWD_MIN, PHI_FWD_MAX = 90.0, 270.0           # forward-entering, deg
DEFAULT_THETA_EDGES = [0.0, 20.0, 40.0, 60.0, np.inf]

N_QUANTILES = 200        # max number of quantile nodes per map
MIN_TRACKS = 2000        # min tracks (source AND target) to derive a map
TRACKS_PER_NODE = 50     # at least this many tracks per quantile interval
CHI2_CLIP = 60000.0      # values at the cache clip are artefacts -> dropped
EFF_QUANTILES = (0.90, 0.95, 0.99)   # target quantiles used as test cuts

# free-sky control region (degrees), same for all detectors
CONTROL_THETA = (25.0, 32.0)
CONTROL_PHI = (170.0, 190.0)
MIN_TRACKS_CR = 1000     # min tracks in the control region (source AND target)
MIN_SUB = 300            # min tracks per control sub-region (consistency)
N_NULL = 20              # random half-splits for the consistency noise level
MIN_BIN_KS = 200         # min tracks per angular bin for the KS check

# Set per fit in the processing loops (see QuantileMap.fit):
FIT_XMAX = None          # map nodes are only placed up to this chi2 value
DIRS_USED = ("fwd",)     # directions that are processed (forward only default)


# ============================================================
# DATA
# ============================================================

def load_flat(path):
    """
    Flat per-track arrays from the compact cache, per fit.
    Every track carries run, data type and WP of its run.
    """
    z = np.load(path, allow_pickle=False)
    runs, dtypes, wps = z["run"], z["data_type"], z["wp"]
    out = {}
    for fit in FITS:
        lens = np.diff(z[f"{fit}_offsets"])
        out[fit] = {
            "run": np.repeat(runs, lens),
            "dtype": np.repeat(dtypes, lens),
            "wp": np.repeat(wps, lens),
            "xy": z[f"{fit}_xy"].astype(np.float32),
            "xz": z[f"{fit}_xz"].astype(np.float32),
            "theta": z[f"{fit}_theta"].astype(np.float32) / 100.0,
            "phi": z[f"{fit}_phi"].astype(np.float32) / 100.0,
        }
    z.close()
    return out


def get_good_run_set(detector):
    """Same good-run selection as the distribution script."""
    import select_runs as sr
    df = sr.load_parsing_overview(detector)
    ped = sr.load_pedestal_outliers(detector)
    mon = sr.load_monitoring_scan(detector)
    good, *_ = sr.select_runs(
        df, ped, mon, run_type="ADC",
        trigger_rate_n_sigma=3.0, label=f"[{detector}]",
    )
    return np.asarray(good, dtype=int)


# ============================================================
# THE MAP
# ============================================================

class QuantileMap:
    """
    Monotonic piecewise-linear map built from matched quantiles of a
    source and a reference sample.

    Outside the outer nodes the map is extrapolated multiplicatively
    (y = y_edge * x / x_edge), i.e. through the origin below and as a pure
    rescaling above. This is robust for the long chi2 tails.
    """

    def __init__(self, xs, ys, n_src=0, n_ref=0, level="bin"):
        self.xs = np.asarray(xs, float)
        self.ys = np.asarray(ys, float)
        self.n_src, self.n_ref, self.level = n_src, n_ref, level

    @classmethod
    def fit(cls, x_src, x_ref, level="bin", x_max=None):
        """
        Quantiles are ALWAYS computed from the full samples (truncating the
        samples would bias the map, since the cut removes a different
        fraction of source and target). x_max (default: module FIT_XMAX)
        only limits where nodes are placed: up to the quantile
        u_max = min(F_src(x_max), F_ref(x_max)). Beyond the last node the
        map is extrapolated by rescaling.
        """
        if x_max is None:
            x_max = FIT_XMAX
        n = min(len(x_src), len(x_ref))
        u_max = 0.995
        if x_max is not None:
            u_max = min(u_max, np.mean(x_src <= x_max),
                        np.mean(x_ref <= x_max))
            u_max = max(u_max, 0.05)
        n_q = int(min(N_QUANTILES, max(20, n * u_max // TRACKS_PER_NODE)))
        qs = np.linspace(0.005, u_max, n_q)
        # extra nodes in the upper tail, only as far as statistics allow
        tail = np.array([0.997, 0.999, 0.9995])
        tail = tail[(tail <= u_max) & (n * (1 - tail) >= 20)]
        qs = np.concatenate([qs, tail])

        q_src = np.quantile(x_src, qs)
        q_ref = np.quantile(x_ref, qs)

        # ties (float16 storage) -> average target over equal source values
        xs, inv = np.unique(q_src, return_inverse=True)
        ys = np.bincount(inv, q_ref) / np.bincount(inv)
        ys = np.maximum.accumulate(ys)
        return cls(xs, ys, len(x_src), len(x_ref), level)

    def __call__(self, x):
        x = np.asarray(x, float)
        y = np.interp(x, self.xs, self.ys)
        lo, hi = x < self.xs[0], x > self.xs[-1]
        if lo.any():
            y[lo] = self.ys[0] * x[lo] / self.xs[0] if self.xs[0] > 0 \
                else self.ys[0]
        if hi.any():
            y[hi] = self.ys[-1] * x[hi] / self.xs[-1]
        return np.maximum(y, 0.0)

    def to_dict(self):
        return {"q_src": self.xs.tolist(), "q_ref": self.ys.tolist(),
                "n_src": int(self.n_src), "n_ref": int(self.n_ref),
                "level": self.level}

    @classmethod
    def from_dict(cls, d):
        return cls(d["q_src"], d["q_ref"], d["n_src"], d["n_ref"], d["level"])


# ============================================================
# BINNING (direction x theta) AND FALLBACK CHAIN
# ============================================================

def is_forward(phi):
    return (phi >= PHI_FWD_MIN) & (phi <= PHI_FWD_MAX)


def bin_key(direction, lo, hi):
    return f"{direction}_{lo:g}-{hi:g}"


def split_bins(theta, phi, edges):
    fwd = is_forward(phi)
    out = {}
    for d, dm in (("fwd", fwd), ("bwd", ~fwd)):
        for lo, hi in zip(edges[:-1], edges[1:]):
            out[bin_key(d, lo, hi)] = dm & (theta >= lo) & (theta < hi)
    return out


def fit_binned_maps(src, ref, edges, min_tracks=MIN_TRACKS):
    """
    src / ref: dicts with chi2, theta, phi arrays.
    Returns dict key -> QuantileMap (keys: every bin, 'fwd', 'bwd', 'all'),
    or None if even the inclusive map lacks statistics.
    """
    ok = lambda a, b: a >= min_tracks and b >= min_tracks   # noqa: E731

    if not ok(len(src["chi2"]), len(ref["chi2"])):
        return None

    maps = {"all": QuantileMap.fit(src["chi2"], ref["chi2"], "all")}

    sf, rf = is_forward(src["phi"]), is_forward(ref["phi"])
    for d, sm, rm in (("fwd", sf, rf), ("bwd", ~sf, ~rf)):
        if ok(sm.sum(), rm.sum()):
            maps[d] = QuantileMap.fit(src["chi2"][sm], ref["chi2"][rm], "dir")
        else:
            maps[d] = maps["all"]

    sb = split_bins(src["theta"], src["phi"], edges)
    rb = split_bins(ref["theta"], ref["phi"], edges)
    for k in sb:
        if ok(sb[k].sum(), rb[k].sum()):
            maps[k] = QuantileMap.fit(
                src["chi2"][sb[k]], ref["chi2"][rb[k]], "bin")
        else:
            maps[k] = maps[k.split("_")[0]]
    return maps


def apply_binned(maps, chi2, theta, phi, edges):
    out = maps["all"](chi2)            # default for tracks outside all bins
    for k, m in split_bins(theta, phi, edges).items():
        if m.any():
            out[m] = maps[k](chi2[m])
    return out.astype(np.float32)


# ============================================================
# VALIDATION
# ============================================================

def efficiencies(src_vals, ref_vals):
    """Efficiency of cuts at target quantiles, before vs. target value."""
    cuts = np.quantile(ref_vals, EFF_QUANTILES)
    return cuts, np.array([np.mean(src_vals < c) for c in cuts])


def cut_effs(vals, cuts):
    return {c: float(np.mean(vals < c)) for c in cuts}


def cross_validate(src, ref, edges):
    """
    Map derived on even-indexed runs, tested on the odd-indexed runs
    (source morphed vs. target of the held-out runs).
    """
    def split(d, parity):
        runs = np.unique(d["run"])
        sel = np.isin(d["run"], runs[parity::2])
        return {k: v[sel] for k, v in d.items()}

    if len(np.unique(src["run"])) < 2 or len(np.unique(ref["run"])) < 2:
        return None

    maps = fit_binned_maps(split(src, 0), split(ref, 0), edges)
    if maps is None:
        return None
    s_test, r_test = split(src, 1), split(ref, 1)
    if len(s_test["chi2"]) < 100 or len(r_test["chi2"]) < 100:
        return None
    morphed = apply_binned(
        maps, s_test["chi2"], s_test["theta"], s_test["phi"], edges)
    return {
        "ks_before": ks_2samp(s_test["chi2"], r_test["chi2"]).statistic,
        "ks_after": ks_2samp(morphed, r_test["chi2"]).statistic,
        "eff_after": efficiencies(morphed, r_test["chi2"])[1],
        "eff_before": efficiencies(s_test["chi2"], r_test["chi2"])[1],
    }



def hist_ratio(ax, axr, s_vals, m_vals, r_vals, source, target, xmax,
               cuts, n_bins=25):
    """
    Density histograms (original / morphed / target) and ratio to target in
    EQUAL-POPULATION bins (defined by the target), up to xmax. Densities are
    normalised to ALL tracks, so a truncated range is not renormalised.
    Returns the plotted xmax. Writes the efficiency of fixed cuts as text.
    """
    if xmax is None:
        xmax = float(np.quantile(r_vals, 0.995))
    u_max = float(min(np.mean(r_vals <= xmax), 1.0))
    edges = np.unique(np.quantile(r_vals, np.linspace(0, u_max, n_bins + 1)))
    edges[0], edges[-1] = 0.0, xmax
    w = np.diff(edges)
    c = {k: np.histogram(v, bins=edges)[0]
         for k, v in (("o", s_vals), ("m", m_vals), ("t", r_vals))}
    n = {"o": len(s_vals), "m": len(m_vals), "t": len(r_vals)}
    dens = {k: c[k] / (n[k] * w) for k in c}

    for k, col, ls, lab in (
        ("o", "tab:blue", "--", f"{source} original (N={n['o']:,})"),
        ("m", "tab:green", "-", f"{source} morphed"),
        ("t", "tab:orange", "-", f"{target} (N={n['t']:,})"),
    ):
        ax.step(edges, np.append(dens[k], dens[k][-1]), where="post",
                color=col, ls=ls, lw=1.6, label=lab)

    centers = 0.5 * (edges[1:] + edges[:-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        for k, col, a in (("o", "tab:blue", 0.45), ("m", "tab:green", 1.0)):
            ratio = dens[k] / dens["t"]
            err = ratio * np.sqrt(1.0 / c[k] + 1.0 / c["t"])
            axr.errorbar(centers, ratio, yerr=err, xerr=w / 2, fmt=".",
                         color=col, alpha=a, ms=4, lw=0.8)
    axr.axhline(1, color="k", lw=0.8)
    axr.set_ylim(0.6, 1.4)
    axr.set_xlim(0, xmax)
    axr.set_ylabel("ratio to target")
    axr.set_xlabel(r"$\chi^2$ (equal-population bins)")
    ax.set_xlim(0, xmax)
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    if len(cuts):
        txt = "efficiency: target / orig / morphed\n" + "\n".join(
            f"  χ²<{c_:g}: {np.mean(r_vals < c_):.3f} / "
            f"{np.mean(s_vals < c_):.3f} / {np.mean(m_vals < c_):.3f}"
            for c_ in cuts)
        ax.text(0.97, 0.55, txt, transform=ax.transAxes, ha="right",
                va="top", fontsize=8, family="monospace",
                bbox=dict(fc="white", alpha=0.7, lw=0.3))
    for a in (ax, axr):
        a.grid(alpha=0.3, ls="--")
    plt.setp(ax.get_xticklabels(), visible=False)
    return xmax

def plot_validation(src, morphed, ref, maps, edges, title, out_file,
                    source, target, xmax=None, cuts=()):
    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[1.3, 1])
    ax = fig.add_subplot(gs[0, 0])
    axr = fig.add_subplot(gs[1, 0], sharex=ax)
    axm = fig.add_subplot(gs[:, 1])

    xmax = hist_ratio(ax, axr, src["chi2"], morphed, ref["chi2"],
                      source, target, xmax, cuts)

    # the maps themselves
    xx = np.linspace(0, xmax, 300)
    thetas = list(zip(edges[:-1], edges[1:]))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(thetas)))
    for (lo, hi), col in zip(thetas, colors):
        for d, ls in (("fwd", "-"), ("bwd", ":")):
            if d not in DIRS_USED:
                continue
            m = maps[bin_key(d, lo, hi)]
            lab = f"{d} θ {lo:g}-{hi:g}" + ("" if m.level == "bin"
                                           else f" [{m.level}]")
            axm.plot(xx, m(xx), color=col, ls=ls, lw=1.4, label=lab)
    axm.plot(xx, xx, "k", lw=0.8, alpha=0.6)
    axm.set_xlabel(rf"$\chi^2$ {source}")
    axm.set_ylabel(rf"morphed $\chi^2$ (→ {target})")
    axm.set_title("Maps per bin (dotted = backward; [..] = fallback)",
                  fontsize=9)
    axm.legend(fontsize=7, ncol=2)
    axm.grid(alpha=0.3, ls="--")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)



# ============================================================
# CONTROL-REGION MODE
# ============================================================

def in_region(theta, phi, cr):
    (t0, t1), (p0, p1) = cr
    return (theta >= t0) & (theta <= t1) & (phi >= p0) & (phi <= p1)


def sub_regions(cr):
    """Halves in theta and in phi of the control region."""
    (t0, t1), (p0, p1) = cr
    tm, pm = 0.5 * (t0 + t1), 0.5 * (p0 + p1)
    return {
        f"θ {t0:g}-{tm:g}": ((t0, tm), (p0, p1)),
        f"θ {tm:g}-{t1:g}": ((tm, t1), (p0, p1)),
        f"φ {p0:g}-{pm:g}": ((t0, t1), (p0, pm)),
        f"φ {pm:g}-{p1:g}": ((t0, t1), (pm, p1)),
    }


def max_rel_dev(m_a, m_b, x):
    """Max relative difference between two maps on the points x."""
    b = m_b(x)
    good = b > 0
    return float(np.max(np.abs(m_a(x[good]) - b[good]) / b[good]))


def control_consistency(src, ref, sm, rm, cr, full_map, rng):
    """
    Max relative deviation of the sub-region maps from the full control
    map, and the same for random half splits (pure statistical noise).
    """
    x = np.quantile(src["chi2"][sm], np.linspace(0.05, 0.95, 50))
    subs, devs = {}, {}
    for name, sub_cr in sub_regions(cr).items():
        s2, r2 = in_region(src["theta"], src["phi"], sub_cr), \
            in_region(ref["theta"], ref["phi"], sub_cr)
        if s2.sum() >= MIN_SUB and r2.sum() >= MIN_SUB:
            subs[name] = QuantileMap.fit(
                src["chi2"][s2], ref["chi2"][r2], "sub")
            devs[name] = max_rel_dev(subs[name], full_map, x)

    xs_cr, xr_cr = src["chi2"][sm], ref["chi2"][rm]
    null = []
    for _ in range(N_NULL):
        a = rng.random(len(xs_cr)) < 0.5
        b = rng.random(len(xr_cr)) < 0.5
        if min(a.sum(), (~a).sum(), b.sum(), (~b).sum()) < MIN_SUB:
            continue
        null.append(max_rel_dev(
            QuantileMap.fit(xs_cr[a], xr_cr[b], "sub"), full_map, x))
    return subs, devs, (float(np.median(null)) if null else np.nan)


def control_cross_validate(src, ref, sm, rm, cuts=()):
    """Map from CR tracks of even-indexed runs, tested on odd-indexed runs."""
    if len(np.unique(src["run"])) < 2 or len(np.unique(ref["run"])) < 2:
        return None

    def part(d, mask, parity):
        runs = np.unique(d["run"])
        sel = mask & np.isin(d["run"], runs[parity::2])
        return d["chi2"][sel]

    tr_s, tr_r = part(src, sm, 0), part(ref, rm, 0)
    te_s, te_r = part(src, sm, 1), part(ref, rm, 1)
    if min(len(tr_s), len(tr_r)) < MIN_SUB or min(len(te_s), len(te_r)) < 100:
        return None
    m = QuantileMap.fit(tr_s, tr_r, "control")
    te_m = m(te_s)
    return {
        "ks_before": ks_2samp(te_s, te_r).statistic,
        "ks_after": ks_2samp(te_m, te_r).statistic,
        "eff_after": efficiencies(te_m, te_r)[1],
        "cut_eff_target": cut_effs(te_r, cuts),
        "cut_eff_after": cut_effs(te_m, cuts),
    }


def angular_closure(src, ref, morphed, edges, cuts=()):
    """KS (morphed vs target, all tracks) per (direction, theta) bin."""
    sb = split_bins(src["theta"], src["phi"], edges)
    rb = split_bins(ref["theta"], ref["phi"], edges)
    rows = []
    for k in sb:
        ns, nr = sb[k].sum(), rb[k].sum()
        if ns < MIN_BIN_KS or nr < MIN_BIN_KS:
            continue
        row = {
            "bin": k, "n_source": int(ns), "n_target": int(nr),
            "ks_before": ks_2samp(src["chi2"][sb[k]],
                                  ref["chi2"][rb[k]]).statistic,
            "ks_after": ks_2samp(morphed[sb[k]],
                                 ref["chi2"][rb[k]]).statistic,
        }
        for c in cuts:
            row[f"eff_target_c{c:g}"] = float(np.mean(ref["chi2"][rb[k]] < c))
            row[f"eff_before_c{c:g}"] = float(np.mean(src["chi2"][sb[k]] < c))
            row[f"eff_after_c{c:g}"] = float(np.mean(morphed[sb[k]] < c))
        rows.append(row)
    return rows


def plot_control_validation(src, ref, morphed, sm, rm, m, subs, ang_rows,
                            title, out_file, source, target, cr,
                            xmax=None, cuts=()):
    fig = plt.figure(figsize=(19, 7))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1],
                          width_ratios=[1.2, 1, 1.1])
    ax = fig.add_subplot(gs[0, 0])
    axr = fig.add_subplot(gs[1, 0], sharex=ax)
    axm = fig.add_subplot(gs[:, 1])
    axk = fig.add_subplot(gs[:, 2])

    s_cr, r_cr, m_cr = src["chi2"][sm], ref["chi2"][rm], morphed[sm]
    xmax = hist_ratio(ax, axr, s_cr, m_cr, r_cr, source, target, xmax, cuts)
    ax.set_title("Control region (fit region)", fontsize=10)

    xx = np.linspace(0, xmax, 300)
    for (name, sm_), col in zip(subs.items(),
                                plt.cm.tab10(np.arange(len(subs)))):
        axm.plot(xx, sm_(xx), color=col, lw=1.1, label=name)
    axm.plot(xx, m(xx), "k", lw=2.2, label="full control region")
    axm.plot(xx, xx, color="gray", lw=0.8)
    axm.set_xlabel(rf"$\chi^2$ {source}")
    axm.set_ylabel(rf"morphed $\chi^2$ (→ {target})")
    axm.set_title("Map consistency inside the control region", fontsize=10)
    axm.legend(fontsize=8)
    axm.grid(alpha=0.3, ls="--")

    if ang_rows:
        xpos = np.arange(len(ang_rows))
        axk.bar(xpos - 0.2, [r["ks_before"] for r in ang_rows], 0.4,
                color="tab:blue", alpha=0.6, label="before")
        axk.bar(xpos + 0.2, [r["ks_after"] for r in ang_rows], 0.4,
                color="tab:green", label="after")
        axk.set_xticks(xpos)
        axk.set_xticklabels([r["bin"] for r in ang_rows], rotation=70,
                            ha="right", fontsize=7)
        axk.legend(fontsize=8)
    axk.set_ylabel("KS statistic (morphed vs target)")
    axk.set_title("All tracks, per angular bin (outside the CR a remaining\n"
                  "difference may be physical, e.g. volcano)", fontsize=9)
    axk.grid(alpha=0.3, ls="--")

    (t0, t1), (p0, p1) = cr
    fig.suptitle(f"{title}   [CR: θ {t0:g}-{t1:g}°, φ {p0:g}-{p1:g}°]")
    fig.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_detector_control(detector, args, edges):
    cr = (tuple(args.cr_theta), tuple(args.cr_phi))
    cache = Path(args.cache_dir) / f"chi2_cache_{detector}.npz"
    if not cache.is_file():
        print(f"[{detector}] cache not found: {cache}")
        return
    print(f"\n[{detector}] loading {cache}")
    data = load_flat(cache)
    good = get_good_run_set(detector) if args.good_runs_only else None
    global FIT_XMAX

    out_dir = Path(args.out_dir) / detector
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    tag = f"{args.source}_to_{args.target}"
    rng = np.random.default_rng(12345)
    #cuts = tuple(args.cuts)
    configured_cuts = tuple(args.cuts)

    all_maps = {"mode": "control", "detector": detector,
                "source": args.source, "target": args.target,
                "control_region": {"theta": list(cr[0]), "phi": list(cr[1])},
                "good_runs_only": bool(args.good_runs_only), "maps": {}}
    rows, ang_all, cut_rows = [], [], []

    for fit in FITS:
        d = data[fit]
        FIT_XMAX = args.range_4p if fit == "4p" else args.range_3p
        keep = (d["xy"] < CHI2_CLIP) & (d["xz"] < CHI2_CLIP)
        if good is not None:
            keep &= np.isin(d["run"], good)
        if not args.include_backward:
            keep &= is_forward(d["phi"])

        for wp in sorted(np.unique(d["wp"][keep])):
            sel = {t: keep & (d["wp"] == wp) & (d["dtype"] == t)
                   for t in (args.source, args.target)}
            for proj in PROJECTIONS:
                def sub(mask):
                    return {"chi2": d[proj][mask], "theta": d["theta"][mask],
                            "phi": d["phi"][mask], "run": d["run"][mask]}
                src, ref = sub(sel[args.source]), sub(sel[args.target])
                sm = in_region(src["theta"], src["phi"], cr)
                rm = in_region(ref["theta"], ref["phi"], cr)
                n_s, n_r = int(sm.sum()), int(rm.sum())
                if n_s < args.min_tracks_cr or n_r < args.min_tracks_cr:
                    print(f"  {fit}_{proj} WP{wp}: CR tracks "
                          f"{args.source}={n_s:,} {args.target}={n_r:,} "
                          f"-> skipped")
                    continue

                m = QuantileMap.fit(src["chi2"][sm], ref["chi2"][rm],
                                    "control")
                morphed = m(src["chi2"]).astype(np.float32)

                ks_b = ks_2samp(src["chi2"][sm], ref["chi2"][rm]).statistic
                ks_a = ks_2samp(morphed[sm], ref["chi2"][rm]).statistic

                #cuts, eff_b = efficiencies(src["chi2"][sm], ref["chi2"][rm])
                #_, eff_a = efficiencies(morphed[sm], ref["chi2"][rm])
                #cv = control_cross_validate(src, ref, sm, rm, cuts)
                quantile_cuts, eff_b = efficiencies(
                    src["chi2"][sm], ref["chi2"][rm]
                )
                _, eff_a = efficiencies(morphed[sm], ref["chi2"][rm])

                cv = control_cross_validate(
                    src, ref, sm, rm, configured_cuts
    )

                subs, devs, dev_null = control_consistency(
                    src, ref, sm, rm, cr, m, rng)
                dev_sub = max(devs.values()) if devs else np.nan
                ang = angular_closure(src, ref, morphed, edges, configured_cuts)

                for c in configured_cuts:
                    cut_rows.append({
                        "detector": detector, "fit": fit, "proj": proj,
                        "wp": int(wp), "cut": c,
                        "eff_target_cr": float(np.mean(ref["chi2"][rm] < c)),
                        "eff_source_before_cr":
                            float(np.mean(src["chi2"][sm] < c)),
                        "eff_source_after_cr":
                            float(np.mean(morphed[sm] < c)),
                        "eff_target_cr_heldout":
                            cv["cut_eff_target"][c] if cv else "",
                        "eff_source_after_cr_heldout":
                            cv["cut_eff_after"][c] if cv else "",
                    })

                (t0, t1) = cr[0]
                out_theta = float(np.mean(
                    (src["theta"] < t0) | (src["theta"] > t1)))

                print(f"  {fit}_{proj} WP{wp}: CR N {args.source}={n_s:,} "
                      f"{args.target}={n_r:,} | CR KS {ks_b:.3f} -> {ks_a:.3f}"
                      + (f" (held-out {cv['ks_after']:.3f})" if cv else "")
                      + f" | sub-region dev {dev_sub:.3f} vs noise "
                        f"{dev_null:.3f} | {out_theta:.0%} of tracks outside "
                        f"CR theta range")

                all_maps["maps"].setdefault(fit, {}).setdefault(
                    proj, {})[str(int(wp))] = {"control": m.to_dict()}

                for i, p in enumerate(EFF_QUANTILES):
                    rows.append({
                        "detector": detector, "fit": fit, "proj": proj,
                        "wp": int(wp), "n_cr_source": n_s, "n_cr_target": n_r,
                        "ks_cr_before": ks_b, "ks_cr_after": ks_a,
                        "ks_cr_after_heldout":
                            cv["ks_after"] if cv else "",
                        "subregion_max_rel_dev": dev_sub,
                        "noise_max_rel_dev": dev_null,
                        "frac_tracks_outside_cr_theta": out_theta,
                        "target_quantile": p, "cut_cr": quantile_cuts[i],
                        "eff_cr_before": eff_b[i], "eff_cr_after": eff_a[i],
                        "eff_cr_after_heldout":
                            cv["eff_after"][i] if cv else "",
                    })
                for r in ang:
                    ang_all.append({"detector": detector, "fit": fit,
                                    "proj": proj, "wp": int(wp), **r})

                plot_control_validation(
                    src, ref, morphed, sm, rm, m, subs, ang,
                    f"BestTrack_{fit}_ChiSquare_{proj} - {detector}, "
                    f"WP{wp}: {args.source} → {args.target}"
                    + (" [good runs only]" if args.good_runs_only else ""),
                    out_dir / "plots"
                    / f"morph_control_{tag}_{fit}_{proj}_WP{wp}.png",
                    args.source, args.target, cr,
                    xmax=FIT_XMAX, cuts=quantile_cuts)

    with open(out_dir / f"morph_maps_control_{tag}.json", "w") as f:
        json.dump(all_maps, f)
    for name, table in ((f"validation_control_{tag}.csv", rows),
                        (f"cut_efficiencies_control_{tag}.csv", cut_rows),
                        (f"angular_closure_control_{tag}.csv", ang_all)):
        if table:
            with open(out_dir / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(table[0]))
                w.writeheader()
                w.writerows(table)
    print(f"[{detector}] output: {out_dir}")


# ============================================================
# MAIN LOOP
# ============================================================

def process_detector_binned(detector, args, edges):
    cache = Path(args.cache_dir) / f"chi2_cache_{detector}.npz"
    if not cache.is_file():
        print(f"[{detector}] cache not found: {cache}")
        return

    print(f"\n[{detector}] loading {cache}")
    data = load_flat(cache)
    good = get_good_run_set(detector) if args.good_runs_only else None
    global FIT_XMAX

    out_dir = Path(args.out_dir) / detector
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    tag = f"{args.source}_to_{args.target}"

    all_maps = {"mode": "binned", "detector": detector,
                "source": args.source, "target": args.target,
                "theta_edges": [float(e) for e in edges],
                "phi_forward": [PHI_FWD_MIN, PHI_FWD_MAX],
                "good_runs_only": bool(args.good_runs_only), "maps": {}}
    rows = []

    for fit in FITS:
        d = data[fit]
        FIT_XMAX = args.range_4p if fit == "4p" else args.range_3p
        keep = np.ones(len(d["run"]), bool)
        if good is not None:
            keep &= np.isin(d["run"], good)
        if not args.include_backward:
            keep &= is_forward(d["phi"])
        # tracks in both projections are already guaranteed by the cache;
        # drop values sitting at the float16 clip
        keep &= (d["xy"] < CHI2_CLIP) & (d["xz"] < CHI2_CLIP)

        for wp in sorted(np.unique(d["wp"][keep])):
            sel = {t: keep & (d["wp"] == wp) & (d["dtype"] == t)
                   for t in (args.source, args.target)}
            n_s, n_t = sel[args.source].sum(), sel[args.target].sum()
            if n_s < MIN_TRACKS or n_t < MIN_TRACKS:
                print(f"  {fit} WP{wp}: {args.source}={n_s:,}, "
                      f"{args.target}={n_t:,} -> skipped (need both)")
                continue

            for proj in PROJECTIONS:
                def sub(mask):
                    return {"chi2": d[proj][mask], "theta": d["theta"][mask],
                            "phi": d["phi"][mask], "run": d["run"][mask]}
                src, ref = sub(sel[args.source]), sub(sel[args.target])

                maps = fit_binned_maps(src, ref, edges)
                if maps is None:
                    continue
                morphed = apply_binned(
                    maps, src["chi2"], src["theta"], src["phi"], edges)

                ks_b = ks_2samp(src["chi2"], ref["chi2"]).statistic
                ks_a = ks_2samp(morphed, ref["chi2"]).statistic
                cuts, eff_b = efficiencies(src["chi2"], ref["chi2"])
                _, eff_a = efficiencies(morphed, ref["chi2"])
                cv = cross_validate(src, ref, edges)

                n_fb = sum(m.level != "bin" for k, m in maps.items()
                           if k not in ("all", "fwd", "bwd"))
                print(f"  {fit}_{proj} WP{wp}: N {args.source}={n_s:,} "
                      f"{args.target}={n_t:,} | KS {ks_b:.3f} -> {ks_a:.3f}"
                      + (f" (held-out {cv['ks_after']:.3f})" if cv else "")
                      + f" | fallback bins: {n_fb}")

                all_maps["maps"].setdefault(fit, {}).setdefault(
                    proj, {})[str(int(wp))] = {k: m.to_dict()
                                               for k, m in maps.items()}
                for i, p in enumerate(EFF_QUANTILES):
                    rows.append({
                        "detector": detector, "fit": fit, "proj": proj,
                        "wp": int(wp), "n_source": int(n_s),
                        "n_target": int(n_t), "n_fallback_bins": n_fb,
                        "ks_before": ks_b, "ks_after": ks_a,
                        "ks_after_heldout": cv["ks_after"] if cv else "",
                        "target_quantile": p, "cut": cuts[i],
                        "eff_source_before": eff_b[i],
                        "eff_source_after": eff_a[i],
                        "eff_source_after_heldout":
                            cv["eff_after"][i] if cv else "",
                    })

                plot_validation(
                    src, morphed, ref, maps, edges,
                    f"BestTrack_{fit}_ChiSquare_{proj} - {detector}, WP{wp}: "
                    f"{args.source} → {args.target}"
                    + (" [good runs only]" if args.good_runs_only else ""),
                    out_dir / "plots"
                    / f"morph_{tag}_{fit}_{proj}_WP{wp}.png",
                    args.source, args.target, xmax=FIT_XMAX,
                    cuts=tuple(args.cuts))

    with open(out_dir / f"morph_maps_{tag}.json", "w") as f:
        json.dump(all_maps, f)
    if rows:
        with open(out_dir / f"validation_{tag}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"[{detector}] output: {out_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--detectors", nargs="+", default=["ROSSO"],
                   choices=["NERO", "ROSSO", "BLU"])
    p.add_argument("--mode", choices=["control", "binned"],
                   default="control",
                   help="control: derive map in the free-sky control region "
                        "(default); binned: per direction x theta bin")
    p.add_argument("--cr-theta", nargs=2, type=float,
                   default=list(CONTROL_THETA), metavar=("MIN", "MAX"),
                   help="control region theta range in deg (default 25 32)")
    p.add_argument("--cr-phi", nargs=2, type=float,
                   default=list(CONTROL_PHI), metavar=("MIN", "MAX"),
                   help="control region phi range in deg (default 170 190)")
    p.add_argument("--include-backward", action="store_true",
                   help="also process backward tracks (default: forward "
                        "only, 90 <= phi <= 270)")
    p.add_argument("--range-4p", type=float, default=100.0,
                   help="4p: map nodes only up to this chi2 (default 100; "
                        "0 = full range)")
    p.add_argument("--range-3p", type=float, default=0.0,
                   help="3p: map nodes only up to this chi2 (default 0 = "
                        "full range)")
    p.add_argument("--cuts", nargs="+", type=float, default=[1.0, 2.0, 5.0],
                   help="chi2 cut values for which efficiencies are "
                        "reported (default 1 2 5)")
    p.add_argument("--min-tracks-cr", type=int, default=MIN_TRACKS_CR,
                   help="min tracks in the control region per data type")
    p.add_argument("--source", choices=["VES", "FS"], default="VES",
                   help="data type that is morphed (default VES)")
    p.add_argument("--cache-dir", default=str(CACHE_DIR))
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--theta-edges", nargs="+", type=float,
                   help="theta bin edges in deg, e.g. 0 20 40 60 "
                        "(last bin is open ended). Default: 0 20 40 60")
    p.add_argument("--good-runs-only", action="store_true",
                   help="use select_runs good-run selection")
    args = p.parse_args()

    global DIRS_USED
    args.target = "FS" if args.source == "VES" else "VES"
    args.range_4p = args.range_4p or None
    args.range_3p = args.range_3p or None
    DIRS_USED = ("fwd", "bwd") if args.include_backward else ("fwd",)
    edges = (np.array(args.theta_edges + [np.inf]) if args.theta_edges
             else np.array(DEFAULT_THETA_EDGES))

    if args.cr_phi[0] > args.cr_phi[1] or args.cr_theta[0] > args.cr_theta[1]:
        p.error("control region: MIN must be <= MAX (no phi wrap-around)")

    run = (process_detector_control if args.mode == "control"
           else process_detector_binned)
    for det in args.detectors:
        run(det, args, edges)


# ============================================================
# DOWNSTREAM USE
# ============================================================

def load_maps(json_file):
    """Load the maps written by this script."""
    with open(json_file) as f:
        raw = json.load(f)
    if "theta_edges" in raw:
        raw["theta_edges"] = np.array(raw["theta_edges"], float)
    for fit in raw["maps"]:
        for proj in raw["maps"][fit]:
            for wp, bins in raw["maps"][fit][proj].items():
                raw["maps"][fit][proj][wp] = {
                    k: QuantileMap.from_dict(v) for k, v in bins.items()}
    return raw


def apply_morph(maps, fit, proj, wp, chi2, theta, phi):
    """
    Morph chi2 values of SOURCE-type tracks (same detector, same WP).
    chi2, theta, phi: arrays in the units/definitions of the cache (deg).
    """
    m = maps["maps"][fit][proj][str(int(wp))]
    chi2 = np.asarray(chi2, float)
    if maps.get("mode") == "control":
        # one map for all tracks; theta/phi are not needed
        return m["control"](chi2).astype(np.float32)
    return apply_binned(m, chi2, np.asarray(theta, float),
                        np.asarray(phi, float), maps["theta_edges"])


if __name__ == "__main__":
    main()