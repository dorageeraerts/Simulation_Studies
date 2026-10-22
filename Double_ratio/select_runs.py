"""
Select the list of good runs for a given hodoscope, applying the agreed quality criteria.

Criteria (in the order they are applied):

  1. Raw run must exist            -> 'missing_run == False'  ("subruns" is an empty list
                                      when the raw files were not found)
  2. All sub-runs present          -> number of subruns == N_SUBRUNS_REQUIRED
                                      (NOT AVAILABLE in the current pkl files, see note below)
  3. Parsing succeeded             -> 'is_run_ok == True'     ("status": "ok")
  4. Missing/shifted bit frequency -> mismatch test passed AND mismatch_prob < MISMATCH_PROB_MAX
                                      (same definition as dev/good_runs/good_runs_stat.py)
  5. Pedestal quality              -> number of affected channels <= MAX_AFFECTED_CHANNELS
                                      (from the pedestal deviation check summary)
  6. Accidental coincidences       -> accidental_rate == 0 (runs with accidental_rate > 0
                                      are excluded)
                                      (from the monitoring scan, data_scan_<color>.pkl)
  7. Trigger rate stability        -> median - N*sigma <= trigger_rate <= median + N*sigma,
                                      with N = TRIGGER_RATE_N_SIGMA (3) and sigma the ROBUST
                                      spread 1.4826*MAD, both computed on the runs that reach
                                      this cut. Applied LAST, so that the band is defined by
                                      the runs that are already good for every other reason.

NOTE on criterion 7: the standard deviation cannot be used here, because it is dominated by
the very runs the cut has to reject. For BLU about one run in five reports a trigger_rate of
O(5e5) Hz: the standard deviation then reaches 1.6e5 Hz against a median of 13 Hz, the band
covers the whole sample and its lower edge is negative, so a run with an abnormally LOW rate
can never be rejected. The median absolute deviation is built from medians, so those runs
cannot move it; 1.4826*MAD is the same number as the standard deviation for a gaussian sample,
which keeps the "median +/- N sigma" reading of the cut. N = 3 is used because the rate
distribution is a narrow core with long tails: at N = 1 the cut would throw away a third to a
half of perfectly good runs, while N = 3 reproduces the usual Tukey outlier fences.

The sample the band is computed on needs no cleaning: those O(5e5) Hz runs all have accidental
coincidences, so criterion 6 removes them before this cut ever sees them.

NOTE on criterion 2: 'read_summary_file' in muraves_lib.run_manager only exposes
'missing_run' (len(subruns) == 0), not the actual number of subruns, so the df_overview pkl
files do not carry that information. The cut is implemented but is skipped with
a warning unless the dataframe has a subrun-count column (see N_SUBRUNS_COLUMNS). To enable
it: add the subrun count to RunValidation/read_summary_file and regenerate the pkl files.

Inputs, all under DATA_PATH (/pnfs/iihe/muraves/muraves_DATA, --data-path to change it).
<version_parsing> and <version_pedestal> are the processing versions of the two production
steps and advance independently (--version-parsing, --version-pedestal):
  - PARSED/<COLOR>/<version_parsing>/df_overview_<COLOR>_<version_parsing>.pkl
        (built with read_summary_file)
  - PEDESTAL/<COLOR>/<version_pedestal>/<COLOR>/pedestal_deviation_check/
        <COLOR>_HISTchi2_outliers_SUMMARY.txt
  - RAW_GZ/<COLOR>/data_scan_<color>.pkl   (trigger rate, accidental rate, working point)

Output:
  - <COLOR>_type<TYPE>_SELECTED_RUNS.txt : one run number per line
  - <COLOR>_type<TYPE>_SELECTION_CUTFLOW.pdf : pie chart of the cut flow (--no-plot to skip)
  - <COLOR>_type<TYPE>_ACCIDENTAL_RATE_SELECTED.pdf : trigger rate vs run for the runs
        reaching criterion 6 (the survivors of criteria 1-5), split into the ones it keeps
        and the ones it rejects (accidental_rate > 0); no trigger rate band is drawn
  - <COLOR>_type<TYPE>_TRIGGER_RATE_SELECTED.pdf : trigger rate vs run for the runs reaching
        criterion 7, split into the selected ones and the ones the band rejects
  - optional csv with the per-run quantities used for the selection (--details)

Usage:
  python select_runs.py                          # all hodoscopes, type ADC
  python select_runs.py --hodoscope BLU --type BOTH --details
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display inside the container
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- configuration

ANALYSIS_DIR = Path(__file__).resolve().parent

# Production data on the storage element. Use --data-path to read a copy somewhere else
# (a local mirror, or the path a condor node sees).
DATA_PATH = Path("/pnfs/iihe/muraves/muraves_DATA")

# Processing versions of the two production steps. They advance independently, so each has
# its own option: --version-parsing and --version-pedestal.
VERSION_PARSING = "v0"
VERSION_PEDESTAL = "v1"

HODOSCOPES = ["BLU", "NERO", "ROSSO"]
TYPES = ["ADC", "PIEDISTALLI", "BOTH"]

# Quality thresholds
N_SUBRUNS_REQUIRED = 4      # criterion 2, may change in the future
MISMATCH_PROB_MAX = 0.05    # criterion 4, probability that a line contains a corrupted value
MAX_AFFECTED_CHANNELS = 10  # criterion 5, pedestal outlier channels per run
TRIGGER_RATE_N_SIGMA = 3.0  # criterion 7, half-width of the accepted band, in robust sigmas
MAX_ACCIDENTAL_RATE = 0.0   # criterion 6, runs above this accidental rate are rejected

# Number of events per file used to normalise the mismatch probability
# (same numbers as dev/good_runs/good_runs_stat.py)
N_EVENTS_PER_TYPE = {"ADC": 40000, "PIEDISTALLI": 50000}
N_LINES_PER_EVENT = 16
N_BITS_PER_LINE = 625

# Possible names of the subrun-count column (criterion 2), if it ever gets added
N_SUBRUNS_COLUMNS = ["n_subruns", "n_subrun", "subruns_counter", "number_of_subruns"]


# ---------------------------------------------------------------- loading

def load_parsing_overview(hodoscope, version=VERSION_PARSING, data_path=DATA_PATH):
    """
    Load the summary-file dataframe produced with run_manager.read_summary_file, from
    <data_path>/PARSED/<COLOR>/<version>/df_overview_<COLOR>_<version>.pkl
    """
    pkl_path = (Path(data_path) / "PARSED" / hodoscope / version /
                f"df_overview_{hodoscope}_{version}.pkl")
    if not pkl_path.exists():
        raise FileNotFoundError(f"Parsing overview not found: {pkl_path}")
    df = pd.read_pickle(pkl_path)
    print(f"[{hodoscope}] loaded {len(df)} summary entries from {pkl_path}")
    return df


def load_pedestal_outliers(hodoscope, version=VERSION_PEDESTAL, data_path=DATA_PATH):
    """
    Read the pedestal outlier summary and return a Series (index = run, value = number of
    affected channels), from
    <data_path>/PEDESTAL/<COLOR>/<version>/<COLOR>/pedestal_deviation_check/
        <COLOR>_HISTchi2_outliers_SUMMARY.txt

    The file starts with a few '#' comment lines and then holds one
    "<run> <number_of_affected_channels>" pair per line.
    """
    txt_path = (Path(data_path) / "PEDESTAL" / hodoscope / version / hodoscope /
                "pedestal_deviation_check" /
                f"{hodoscope}_HISTchi2_outliers_SUMMARY.txt")
    if not txt_path.exists():
        print(f"[{hodoscope}] [WARNING] pedestal summary not found: {txt_path}")
        return None

    runs, n_channels = [], []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 2:
                print(f"[{hodoscope}] [WARNING] skipping malformed pedestal line: {line!r}")
                continue
            try:
                runs.append(int(fields[0]))
                n_channels.append(int(fields[1]))
            except ValueError:
                print(f"[{hodoscope}] [WARNING] skipping malformed pedestal line: {line!r}")

    outliers = pd.Series(n_channels, index=pd.Index(runs, name="run"),
                         name="n_affected_channels")
    outliers = outliers[~outliers.index.duplicated(keep="last")]
    print(f"[{hodoscope}] loaded pedestal info for {len(outliers)} runs from {txt_path.name}")
    return outliers


def load_monitoring_scan(hodoscope, data_path=DATA_PATH):
    """
    Read <data_path>/RAW_GZ/<COLOR>/data_scan_<color>.pkl and return a dataframe indexed by
    run with the
    'trigger_rate' and 'accidental_rate' columns (criteria 6 and 7), or None if the file
    is not there. The working point temperature ('wp', column 5 of the slow-control file)
    comes along when the scan has it: it is not used by any criterion, only plotted.

    The scan holds one row per run; rows whose rates were not measured carry NaN.
    """
    pkl_path = Path(data_path) / "RAW_GZ" / hodoscope / f"data_scan_{hodoscope.lower()}.pkl"
    if not pkl_path.exists():
        print(f"[{hodoscope}] [WARNING] monitoring scan not found: {pkl_path}")
        return None

    df = pd.read_pickle(pkl_path)
    missing = [c for c in ("run", "trigger_rate", "accidental_rate") if c not in df.columns]
    if missing:
        print(f"[{hodoscope}] [WARNING] {pkl_path.name} has no {missing} column(s): "
              f"criteria 6 and 7 will be skipped.")
        return None

    columns = ["run", "trigger_rate", "accidental_rate"]
    if "wp" in df.columns:
        columns.append("wp")
    else:
        print(f"[{hodoscope}] [WARNING] {pkl_path.name} has no 'wp' column: the working "
              f"point temperature will not be plotted.")

    rates = df[columns].copy()
    if "wp" not in rates.columns:
        rates["wp"] = np.nan
    rates = rates[~rates["run"].duplicated(keep="last")].set_index("run")
    print(f"[{hodoscope}] loaded monitoring rates for {len(rates)} runs from {pkl_path.name}")
    return rates


# ---------------------------------------------------------------- quality flags

def add_quality_flags(df):
    """Add the derived quantities used by the selection to the summary dataframe."""
    df = df.copy()

    # Probability that a line contains at least one corrupted value.
    # 'mismatch_counter' is NaN when the summary file reports no mismatch at all,
    # which means zero mismatches, not "unknown".
    counter = pd.to_numeric(df["mismatch_counter"], errors="coerce")
    no_mismatch = df["has_mismatches"].ne(True)
    counter = counter.mask(counter.isna() & no_mismatch, 0.0)

    n_events = df["type"].map(N_EVENTS_PER_TYPE)
    df["mismatch_prob"] = (
        counter * N_BITS_PER_LINE / (n_events * N_LINES_PER_EVENT)
    )

    # A run passes the mismatch test if it has no mismatches at all, or if all of
    # its mismatches are recoverable.
    df["mismatch_test_passed"] = (
        df["has_mismatches"].eq(False)
        | (df["has_mismatches"].eq(True) & df["has_unrecoverable_mismatches"].eq(False))
    )
    return df


def robust_sigma(values):
    """
    Spread of 'values' from the median absolute deviation (MAD), scaled by 1.4826 so that it
    is the standard deviation for a gaussian sample.

    The standard deviation cannot be used for criterion 7: it is dominated by the very runs
    the cut has to reject. For BLU about one run in five reports a trigger rate of O(5e5) Hz,
    which brings the standard deviation to 1.6e5 Hz against a median of 13 Hz, so the band
    covers everything and its lower edge is negative. The MAD is built from medians, so a
    fifth of the sample can be arbitrarily wrong without moving it.
    """
    median = values.median()
    return 1.4826 * (values - median).abs().median()


def find_n_subruns_column(df):
    """Return the name of the subrun-count column, or None if the pkl does not carry it."""
    for column in N_SUBRUNS_COLUMNS:
        if column in df.columns:
            return column
    return None


# ---------------------------------------------------------------- selection

def select_runs(df, pedestal_outliers, monitoring_rates=None, run_type="ADC",
                n_subruns_required=N_SUBRUNS_REQUIRED,
                mismatch_prob_max=MISMATCH_PROB_MAX,
                max_affected_channels=MAX_AFFECTED_CHANNELS,
                trigger_rate_n_sigma=TRIGGER_RATE_N_SIGMA,
                max_accidental_rate=MAX_ACCIDENTAL_RATE,
                require_pedestal_info=False,
                require_monitoring_info=False,
                label=""):
    """
    Apply the quality criteria and return (good_runs, details, cutflow, trigger_rate_info).

    'good_runs'         is a sorted numpy array of run numbers.
    'details'           is the per-run dataframe of the quantities used for the selection.
    'cutflow'           is the dataframe of the cut flow (runs left / removed at each step),
                        used to draw the summary pie chart.
    'trigger_rate_info' is a dict with the runs of the monitoring scan, the runs entering
                        criterion 7 and the selected ones, their trigger rate and
                        the (median, robust sigma, band) used to cut, for
                        plot_trigger_rate; it is None when no monitoring information is
                        available.

    run_type: 'ADC' or 'PIEDISTALLI' selects that file type only.
              'BOTH' requires the run to pass the criteria for both file types.

    require_pedestal_info: if True, runs missing from the pedestal summary are rejected;
                           if False (default) they are kept, and only counted.

    require_monitoring_info: same, for the runs that have no trigger/accidental rate in the
                             monitoring scan (criteria 6 and 7).
    """
    assert run_type in TYPES, f"Unexpected type: {run_type}. Expected one of {TYPES}."

    df = add_quality_flags(df)
    if run_type != "BOTH":
        df = df[df["type"] == run_type]
    total_runs = df["run"].nunique()
    print(f"{label} starting from {total_runs} runs (type: {run_type})")

    # ---- criteria applied on the summary files (one row per run and file type)
    # Each entry is (label, condition to PASS the cut). The labels name the reason a run
    # is rejected, because that is what the cut flow and the pie chart report.
    cuts = [
        ("1. run N/A", df["missing_run"].eq(False)),
    ]

    n_subruns_column = find_n_subruns_column(df)
    if n_subruns_column is not None:
        cuts.append((
            f"2. not {n_subruns_required} subruns",
            df[n_subruns_column].eq(n_subruns_required),
        ))
    else:
        print(f"{label} [WARNING] no subrun-count column in the dataframe "
              f"(looked for {N_SUBRUNS_COLUMNS}): criterion 2 "
              f"('exactly {n_subruns_required} subruns') is SKIPPED. "
              f"read_summary_file only stores 'missing_run' (len(subruns) == 0).")

    cuts += [
        ("3. parsing failed", df["is_run_ok"].eq(True)),
        ("4a. unrecoverable mismatches", df["mismatch_test_passed"].eq(True)),
        (f"4b. mismatch_prob >= {mismatch_prob_max}", df["mismatch_prob"].lt(mismatch_prob_max)),
    ]

    # Sequential cut flow. A run survives only if ALL of its selected rows survive:
    # for run_type 'BOTH' this means both the ADC and the PIEDISTALLI file must pass.
    # All percentages are relative to the initial number of runs, so that the
    # "removed" column can be summed across the steps.
    print(f"{label} cut flow (percentages w.r.t. the {total_runs} initial runs):")
    cutflow = []
    mask = pd.Series(True, index=df.index)
    surviving = total_runs
    for name, cut in cuts:
        mask &= cut.fillna(False).astype(bool)
        previous, surviving = surviving, _surviving_runs(df, mask, run_type).size
        cutflow.append({"cut": name, "runs_left": surviving, "removed": previous - surviving})
        print(_cutflow_line(label, name, surviving, previous - surviving, total_runs))

    good_runs = _surviving_runs(df, mask, run_type)

    # ---- criterion 5: pedestal quality, one value per run (no file type)
    details = (
        df[df["run"].isin(good_runs)]
        .groupby("run")
        .agg(mismatch_prob=("mismatch_prob", "max"),
             mismatch_counter=("mismatch_counter", "max"))
        .reset_index()
    )

    if pedestal_outliers is None:
        print(f"{label} [WARNING] no pedestal summary available: criterion 5 is SKIPPED.")
        details["n_affected_channels"] = np.nan
    else:
        details["n_affected_channels"] = details["run"].map(pedestal_outliers)
        n_missing = int(details["n_affected_channels"].isna().sum())
        if n_missing:
            action = "rejected" if require_pedestal_info else "kept"
            print(f"{label} [INFO] {n_missing} runs are not in the pedestal summary: {action}.")

        pedestal_ok = details["n_affected_channels"].le(max_affected_channels)
        if not require_pedestal_info:
            pedestal_ok |= details["n_affected_channels"].isna()

        details = _apply_run_cut(details, "5. bad pedestal", pedestal_ok,
                                 cutflow, label, total_runs)

    # ---- criteria 6 and 7: monitoring rates, one value per run (no file type)
    trigger_rate_info = None
    if monitoring_rates is None:
        print(f"{label} [WARNING] no monitoring scan available: "
              f"criteria 6 and 7 are SKIPPED.")
        details["trigger_rate"] = np.nan
        details["accidental_rate"] = np.nan
        details["wp"] = np.nan
    else:
        details["trigger_rate"] = details["run"].map(monitoring_rates["trigger_rate"])
        details["accidental_rate"] = details["run"].map(monitoring_rates["accidental_rate"])
        details["wp"] = details["run"].map(monitoring_rates["wp"])
        n_missing = int(details["trigger_rate"].isna().sum())
        if n_missing:
            action = "rejected" if require_monitoring_info else "kept"
            print(f"{label} [INFO] {n_missing} runs have no trigger rate in the monitoring "
                  f"scan: {action}.")

        # Criterion 6: no accidental coincidences.
        accidental_ok = details["accidental_rate"].le(max_accidental_rate)
        if not require_monitoring_info:
            accidental_ok |= details["accidental_rate"].isna()
        reaching_accidental = details[["run", "trigger_rate", "wp"]].copy()
        details = _apply_run_cut(details, f"6. accidental_rate > {max_accidental_rate:g}",
                                 accidental_ok, cutflow, label, total_runs)
        accidental_rejected = reaching_accidental[
            ~reaching_accidental["run"].isin(details["run"])]

        # Criterion 7, applied last: median +/- n_sigma * robust_sigma, computed on the runs
        # that reach this cut. No sample cleaning is needed here: the runs that could distort
        # the band (BLU reports O(5e5) Hz for a fifth of its runs) all have accidental
        # coincidences and are already gone with criterion 6, and the MAD would not be moved
        # by them anyway.
        rates = details["trigger_rate"]
        median, sigma = rates.median(), robust_sigma(rates)
        low, high = median - trigger_rate_n_sigma * sigma, median + trigger_rate_n_sigma * sigma
        print(f"{label} [INFO] trigger rate: median = {median:.3f}, "
              f"robust sigma (1.4826*MAD) = {sigma:.3f}, "
              f"accepted band = [{low:.3f}, {high:.3f}] ({trigger_rate_n_sigma:g} sigma, "
              f"computed on the {int(rates.notna().sum())} runs reaching the cut)")
        if low < 0:
            print(f"{label} [WARNING] the lower edge of the band is negative: the cut cannot "
                  f"reject a run with an abnormally low trigger rate.")

        reaching_trigger = details[["run", "trigger_rate", "wp"]].copy()

        trigger_ok = rates.between(low, high)
        if not require_monitoring_info:
            trigger_ok |= rates.isna()
        details = _apply_run_cut(details, f"7. trigger rate outside "
                                          f"{trigger_rate_n_sigma:g} sigma",
                                 trigger_ok, cutflow, label, total_runs)
        trigger_rejected = reaching_trigger[~reaching_trigger["run"].isin(details["run"])]

        # Everything the two plots need: the runs each of the two cuts rejects, the runs
        # that survive the complete selection, and for each of them the trigger rate and the
        # working point temperature.
        trigger_rate_info = {
            "n_reaching_accidental": len(reaching_accidental),
            "n_reaching_trigger": len(reaching_trigger),
            "median": median, "sigma": sigma, "low": low, "high": high,
            "n_sigma": trigger_rate_n_sigma,
            "max_accidental_rate": max_accidental_rate,
        }
        for name, frame in (("reaching_trigger", reaching_trigger),
                            ("accidental_rejected", accidental_rejected),
                            ("trigger_rejected", trigger_rejected),
                            ("selected", details)):
            trigger_rate_info[f"{name}_run"] = frame["run"].to_numpy()
            trigger_rate_info[f"{name}_trigger_rate"] = frame["trigger_rate"].to_numpy()
            trigger_rate_info[f"{name}_wp"] = frame["wp"].to_numpy()

    good_runs = np.sort(details["run"].to_numpy())
    fraction = len(good_runs) / total_runs * 100 if total_runs else 0.0
    print(f"{label} SELECTED {len(good_runs)} runs out of {total_runs} ({fraction:.2f}%)")

    cutflow = pd.DataFrame(cutflow)
    cutflow["total_runs"] = total_runs
    cutflow["removed_percent"] = cutflow["removed"] / total_runs * 100 if total_runs else np.nan
    cutflow["runs_left_percent"] = cutflow["runs_left"] / total_runs * 100 if total_runs else np.nan
    return good_runs, details.reset_index(drop=True), cutflow, trigger_rate_info


def _apply_run_cut(details, name, keep, cutflow, label, total_runs):
    """Apply one per-run cut to 'details', append its line to 'cutflow' and print it."""
    previous = len(details)
    details = details[keep.fillna(False).astype(bool)]
    cutflow.append({"cut": name, "runs_left": len(details),
                    "removed": previous - len(details)})
    print(_cutflow_line(label, name, len(details), previous - len(details), total_runs))
    return details


def plot_cutflow_pie(cutflow, output_path, hodoscope="", run_type=""):
    """
    Pie chart of the selection: one slice per cut (the runs it removes) plus one
    slice for the selected runs. All slices are fractions of the initial number of
    runs, so they add up to 100%.

    Cuts that remove nothing get no slice (they would be invisible); they are
    listed under the chart instead.
    """
    total_runs = int(cutflow["total_runs"].iloc[0])
    selected = int(cutflow["runs_left"].iloc[-1])

    removing = cutflow[cutflow["removed"] > 0]
    empty_cuts = cutflow.loc[cutflow["removed"] == 0, "cut"].tolist()

    labels = list(removing["cut"]) + ["SELECTED"]
    sizes = list(removing["removed"]) + [selected]

    # Greys/reds for the rejected runs, green for the selected ones
    rejected_colors = plt.cm.Reds(np.linspace(0.35, 0.8, len(removing)))
    colors = list(rejected_colors) + ["#2ca02c"]
    explode = [0.0] * len(removing) + [0.05]

    def autopct(percent):
        return f"{percent:.2f}%\n({int(round(percent * total_runs / 100))})"

    fig, ax = plt.subplots(figsize=(9, 7))
    wedges, _, autotexts = ax.pie(
        sizes,
        colors=colors,
        explode=explode,
        startangle=90,
        counterclock=False,
        autopct=autopct,
        pctdistance=0.75,
        textprops={"fontsize": 9},
        wedgeprops={"edgecolor": "white", "linewidth": 1},
    )
    # Thin slices have no room for a label inside: move those outside, with a leader line
    # and a staggered radius so that consecutive small slices do not overlap.
    n_outside = 0
    for wedge, autotext, size in zip(wedges, autotexts, sizes):
        autotext.set_fontweight("bold")
        if size / sum(sizes) >= 0.06:
            autotext.set_color("white")
            continue
        angle = np.deg2rad(0.5 * (wedge.theta1 + wedge.theta2))
        x, y = np.cos(angle), np.sin(angle)
        radius = 1.25 + 0.22 * (n_outside % 2)
        n_outside += 1
        autotext.set_position((radius * x, radius * y))
        autotext.set_color("black")
        autotext.set_horizontalalignment("left" if x >= 0 else "right")
        ax.plot([1.02 * x, (radius - 0.04) * x], [1.02 * y, (radius - 0.04) * y],
                color="grey", linewidth=0.8, clip_on=False)

    ax.legend(wedges, [f"{name}  ({size} runs)" for name, size in zip(labels, sizes)],
              title="Removed by / selected", loc="center left",
              bbox_to_anchor=(1.15, 0.5), fontsize=9)
    ax.set_title(f"{hodoscope} - type {run_type}\nRun selection: "
                 f"{selected} good runs out of {total_runs} "
                 f"({selected / total_runs * 100:.2f}%)")
    ax.axis("equal")

    footnote = ("Slices are fractions of the initial number of runs. Cuts are applied in "
                "sequence, so each slice\nis what that cut removes on top of the previous ones.")
    if empty_cuts:
        footnote += "\nCuts that removed no additional run: " + ", ".join(empty_cuts) + "."
    fig.text(0.02, 0.02, footnote, fontsize=8, style="italic", va="bottom")

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_trigger_rate(trigger_rate_info, output_path, hodoscope="", run_type="",
                      population="accidental"):
    """
    Two populations are available:

      'accidental' the runs reaching criterion 6, i.e. the survivors of criteria 1-5, split
                   into the ones it keeps and the ones it rejects (accidental_rate > 0).
                   The trigger rate band is NOT drawn: it is only defined at criterion 7,
                   after this cut.
      'selected'   the runs reaching criterion 7, split into the ones the band keeps (the
                   selected runs) and the ones it rejects, with the band drawn.

    The rate can span several orders of magnitude (for BLU a fifth of the runs report
    O(1e5) Hz). In that case the figure gets a second panel: the whole range on a symlog
    scale on top, and a linear zoom on the core of the distribution below.
    """
    median, sigma = trigger_rate_info["median"], trigger_rate_info["sigma"]
    low, high = trigger_rate_info["low"], trigger_rate_info["high"]
    n_sigma = trigger_rate_info["n_sigma"]
    max_accidental = trigger_rate_info.get("max_accidental_rate", 0.0)

    kept_color, rejected_color = "#2f6f9f", "#d1662a"

    def series(prefix):
        return (np.asarray(trigger_rate_info[f"{prefix}_run"], dtype=float),
                np.asarray(trigger_rate_info[f"{prefix}_trigger_rate"], dtype=float),
                np.asarray(trigger_rate_info.get(f"{prefix}_wp", []), dtype=float))

    if population == "accidental":
        kept_runs, kept_rates, kept_wp = series("reaching_trigger")
        cut_runs, cut_rates, cut_wp = series("accidental_rejected")
        population_name = "the runs reaching criterion 6"
        kept_label = f"kept, on to criterion 7 ({kept_runs.size} runs)"
        cut_label = (f"rejected by criterion 6: accidental_rate > {max_accidental:g} "
                     f"({cut_runs.size} runs)")
        show_band = False
    elif population == "selected":
        kept_runs, kept_rates, kept_wp = series("selected")
        cut_runs, cut_rates, cut_wp = series("trigger_rejected")
        population_name = "the runs reaching criterion 7"
        kept_label = f"selected ({kept_runs.size} runs)"
        cut_label = (f"rejected by criterion 7: outside the band "
                     f"({cut_runs.size} runs)")
        show_band = True
    else:
        raise ValueError(f"Unexpected population: {population}. "
                         f"Expected 'accidental' or 'selected'.")

    def draw(ax):
        """Draw the band, the guides and the two populations of runs on one axis."""
        if show_band:
            ax.axhspan(low, high, color=kept_color, alpha=0.12, zorder=0,
                       label=f"median $\\pm$ {n_sigma:g}$\\sigma$ = [{low:.2f}, {high:.2f}]")
            ax.axhline(median, color="#1b4a6b", linewidth=1.5, zorder=3,
                       label=f"median = {median:.3f} ($\\sigma_{{\\mathrm{{rob}}}}$ = "
                             f"1.4826$\\cdot$MAD = {sigma:.3f})")
            for edge in (low, high):
                ax.axhline(edge, color=kept_color, linewidth=1.0, linestyle="--", zorder=2)

        ax.scatter(kept_runs, kept_rates, s=6, color=kept_color, alpha=0.5,
                   linewidths=0, zorder=4, label=kept_label)
        if cut_runs.size:
            ax.scatter(cut_runs, cut_rates, s=12, color=rejected_color, alpha=0.8,
                       linewidths=0, zorder=5, label=cut_label)

        ax.set_ylabel("trigger rate [Hz]")
        _style_axis(ax)

    def draw_wp(ax):
        """
        The working point temperature of the runs drawn above, in the same two colours.

        Only the runs with a trigger rate are drawn: a run whose rate is missing from the
        monitoring scan is absent from the panel above (it is kept by default, not measured),
        and drawing its working point here would show a run that has no counterpart above.
        """
        ax.scatter(kept_runs[kept_drawn], kept_wp[kept_drawn], s=6, color=kept_color,
                   alpha=0.5, linewidths=0, zorder=4)
        if cut_drawn.any():
            ax.scatter(cut_runs[cut_drawn], cut_wp[cut_drawn], s=12, color=rejected_color,
                       alpha=0.8, linewidths=0, zorder=5)
        ax.set_ylabel("working point\ntemperature [$^\\circ$C]")
        _style_axis(ax)

    def _style_axis(ax):
        ax.grid(axis="y", color="grey", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    runs = np.concatenate([kept_runs, cut_runs])
    rates = np.concatenate([kept_rates, cut_rates])
    finite = rates[np.isfinite(rates)]
    y_min = float(finite.min()) if finite.size else 0.0
    y_max = float(finite.max()) if finite.size else 1.0
    # A handful of runs far above the median would squash everything else onto a single line.
    needs_zoom = bool(finite.size) and median > 0 and y_max > 10 * median

    # The working point temperature gets a panel of its own, below the rate panels, for the
    # same runs and on the same x axis. It is not a criterion, only context.
    kept_drawn = np.isfinite(kept_rates) & np.isfinite(kept_wp)
    cut_drawn = np.isfinite(cut_rates) & np.isfinite(cut_wp)
    show_wp = bool(kept_drawn.any() or cut_drawn.any())

    heights = ([1.0, 1.4] if needs_zoom else [1.0]) + ([0.55] if show_wp else [])
    height = 5.0 + (2.5 if needs_zoom else 0.0) + (1.8 if show_wp else 0.0)
    axes = plt.subplots(len(heights), 1, figsize=(11, height), sharex=True, squeeze=False,
                        gridspec_kw={"height_ratios": heights, "hspace": 0.12})
    fig, axes = axes[0], axes[1][:, 0]

    if needs_zoom:
        ax_full, ax = axes[0], axes[1]
        draw(ax_full)
        ax_full.set_yscale("symlog", linthresh=max(2 * median, 1e-3))
        ax_full.set_ylim(min(0.0, 1.2 * y_min), 2.0 * y_max)
        _panel_label(ax_full, "full range (symlog)")
        draw(ax)
        # Robust range of the plotted rates: wide enough for the core and the near outliers,
        # immune to the O(1e5) Hz runs that make the second panel necessary in the first place.
        core = float(np.median(finite))
        spread = 1.4826 * float(np.median(np.abs(finite - core)))
        zoom_low, zoom_high = max(0.0, core - 10 * spread), core + 10 * spread
        if show_band:
            zoom_low, zoom_high = min(zoom_low, low), max(zoom_high, high)
        ax.set_ylim(zoom_low, zoom_high)
        _panel_label(ax, f"zoom on {zoom_low:.3g}-{zoom_high:.3g} Hz"
                         + (", where the band is" if show_band else ""))
    else:
        ax = axes[0]
        draw(ax)
        if finite.size:
            if show_band:
                y_min, y_max = min(y_min, low), max(y_max, high)
            padding = 0.05 * (y_max - y_min) or 1.0
            ax.set_ylim(y_min - padding, y_max + padding)
    title_ax = axes[0]

    if show_wp:
        # No panel caption here: the y label already says what the panel shows, and a caption
        # would sit on top of the 25 degree line, which is where most runs are.
        draw_wp(axes[-1])

    axes[-1].set_xlabel("run number")
    if show_band:
        title = (f"Criterion 7: trigger rate within {n_sigma:g} robust sigma(s) of the "
                 f"median - {population_name}")
    else:
        title = (f"Criterion 6: accidental coincidences - {population_name}, "
                 f"split into kept and rejected")
    title_ax.set_title(f"{hodoscope} - type {run_type}\n{title}")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)

    n_missing = int((~np.isfinite(rates)).sum())
    footnote = f"One point per run: {population_name} ({runs.size} runs)."
    if population == "accidental":
        footnote += (" Runs rejected by criteria 1-5 are not drawn.\nThe trigger rate band "
                     "is not drawn either: it is defined at criterion 7, on the runs this "
                     "cut keeps.")
    else:
        footnote += " Median and robust sigma are computed on these same runs."
    if n_missing:
        footnote += f"\n{n_missing} run{'s' if n_missing > 1 else ''} " \
                    f"ha{'ve' if n_missing > 1 else 's'} no trigger rate in the monitoring " \
                    f"scan and {'are' if n_missing > 1 else 'is'} not drawn."
    if low < 0:
        footnote += "\nThe lower edge of the band is negative: only high rates are rejected."
    fig.text(0.02, 0.02, footnote, fontsize=8, style="italic", va="bottom")

    if len(axes) > 1:
        # tight_layout does not honour the reserved band with shared-x panels: place them by
        # hand, reserving about 1.2 inches at the bottom for the x label and the footnote.
        fig.subplots_adjust(left=0.09, right=0.98, top=1 - 0.75 / height,
                            bottom=1.2 / height, hspace=0.12)
    else:
        fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return output_path


def _panel_label(ax, text):
    """Small grey caption inside the top-left corner of a panel."""
    ax.text(0.006, 0.97, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color="grey")


def _cutflow_line(label, name, surviving, removed, total_runs):
    """Format one line of the cut flow, with percentages w.r.t. the initial number of runs."""
    def percentage(n):
        return f"{n / total_runs * 100:6.2f}%" if total_runs else "   n/a"

    return (f"{label}   {name:<32} {surviving:>7} runs left ({percentage(surviving)}) "
            f"| removed {removed:>6} ({percentage(removed)})")


def _surviving_runs(df, mask, run_type):
    """
    Run numbers whose rows all pass 'mask'. For run_type 'BOTH' the run must also
    have both file types present in the dataframe.
    """
    passed = df.assign(passed=mask).groupby("run")["passed"].all()
    runs = passed[passed].index

    if run_type == "BOTH":
        both_types = df.groupby("run")["type"].nunique().ge(len(N_EVENTS_PER_TYPE))
        runs = runs.intersection(both_types[both_types].index)

    return np.sort(runs.to_numpy())


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hodoscope", default="ALL", choices=HODOSCOPES + ["ALL"],
                        help="hodoscope colour to process (default: ALL)")
    parser.add_argument("--type", dest="run_type", default="ADC", choices=TYPES,
                        help="file type the criteria are applied to (default: ADC)")
    parser.add_argument("--data-path", default=str(DATA_PATH),
                        help=f"root of the production data (default: {DATA_PATH})")
    parser.add_argument("--version-parsing", default=VERSION_PARSING,
                        help=f"processing version of the parsing step, used both for the "
                             f"directory and for the file name "
                             f"(default: {VERSION_PARSING})")
    parser.add_argument("--version-pedestal", default=VERSION_PEDESTAL,
                        help="processing version of the pedestal step "
                             f"(default: {VERSION_PEDESTAL})")
    parser.add_argument("--mismatch-prob-max", type=float, default=MISMATCH_PROB_MAX,
                        help=f"maximum mismatch probability (default: {MISMATCH_PROB_MAX})")
    parser.add_argument("--max-affected-channels", type=int, default=MAX_AFFECTED_CHANNELS,
                        help="maximum number of pedestal outlier channels "
                             f"(default: {MAX_AFFECTED_CHANNELS})")
    parser.add_argument("--n-subruns", type=int, default=N_SUBRUNS_REQUIRED,
                        help=f"required number of subruns (default: {N_SUBRUNS_REQUIRED})")
    parser.add_argument("--trigger-rate-n-sigma", type=float, default=TRIGGER_RATE_N_SIGMA,
                        help="half-width of the accepted trigger rate band, in robust sigmas "
                             "(1.4826*MAD) around the median "
                             f"(default: {TRIGGER_RATE_N_SIGMA})")
    parser.add_argument("--max-accidental-rate", type=float, default=MAX_ACCIDENTAL_RATE,
                        help="maximum accidental rate, runs above it are rejected "
                             f"(default: {MAX_ACCIDENTAL_RATE})")
    parser.add_argument("--require-pedestal-info", action="store_true",
                        help="reject runs that are not listed in the pedestal summary "
                             "(default: keep them)")
    parser.add_argument("--require-monitoring-info", action="store_true",
                        help="reject runs whose trigger/accidental rate is missing from the "
                             "monitoring scan (default: keep them)")
    parser.add_argument("--details", action="store_true",
                        help="also write a csv with the quantities used for the selection")
    parser.add_argument("--no-plot", action="store_true",
                        help="do not save the cut flow pie chart")
    parser.add_argument("--output-dir", default=str(ANALYSIS_DIR),
                        help="where to write the run lists (default: this directory)")
    args = parser.parse_args()

    hodoscopes = HODOSCOPES if args.hodoscope == "ALL" else [args.hodoscope]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for hodoscope in hodoscopes:
        label = f"[{hodoscope}]"
        print(f"\n{'=' * 70}\n{hodoscope}\n{'=' * 70}")

        df = load_parsing_overview(hodoscope, version=args.version_parsing,
                                   data_path=args.data_path)
        pedestal_outliers = load_pedestal_outliers(hodoscope, version=args.version_pedestal,
                                                   data_path=args.data_path)
        monitoring_rates = load_monitoring_scan(hodoscope, data_path=args.data_path)

        good_runs, details, cutflow, trigger_rate_info = select_runs(
            df,
            pedestal_outliers,
            monitoring_rates,
            run_type=args.run_type,
            n_subruns_required=args.n_subruns,
            mismatch_prob_max=args.mismatch_prob_max,
            max_affected_channels=args.max_affected_channels,
            trigger_rate_n_sigma=args.trigger_rate_n_sigma,
            max_accidental_rate=args.max_accidental_rate,
            require_pedestal_info=args.require_pedestal_info,
            require_monitoring_info=args.require_monitoring_info,
            label=label,
        )

        output_path = output_dir / f"{hodoscope}_type{args.run_type}_SELECTED_RUNS.txt"
        with open(output_path, "w") as f:
            f.writelines(f"{run}\n" for run in good_runs)
        print(f"{label} run list written to {output_path}")

        if args.details:
            details_path = output_dir / f"{hodoscope}_type{args.run_type}_SELECTED_RUNS.csv"
            details.to_csv(details_path, index=False)
            print(f"{label} details written to {details_path}")

        if not args.no_plot:
            plot_path = output_dir / f"{hodoscope}_type{args.run_type}_SELECTION_CUTFLOW.pdf"
            plot_cutflow_pie(cutflow, plot_path, hodoscope=hodoscope, run_type=args.run_type)
            print(f"{label} cut flow pie chart written to {plot_path}")

            if trigger_rate_info is not None:
                plots = (("accidental", "ACCIDENTAL_RATE_SELECTED"),
                         ("selected", "TRIGGER_RATE_SELECTED"))
                for population, name in plots:
                    rate_path = (output_dir /
                                 f"{hodoscope}_type{args.run_type}_{name}.pdf")
                    plot_trigger_rate(trigger_rate_info, rate_path, hodoscope=hodoscope,
                                      run_type=args.run_type, population=population)
                    print(f"{label} {name} plot written to {rate_path}")


if __name__ == "__main__":
    main()