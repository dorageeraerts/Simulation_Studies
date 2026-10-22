#!/usr/bin/env python3

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import io
import json
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot
import xarray as xr
import cdsapi


# ============================================================
# CONFIGURATION
# ============================================================

RECONSTRUCTED_DIR = Path("/pnfs/iihe/muraves/muraves_DATA/RECONSTRUCTED")
SLOWCONTROL_DIR = Path("/pnfs/iihe/muraves/muraves_DATA/DECOMPRESSED")

TREE_NAME = "AnalyzedData"
TIMESTAMP_BRANCH = "timestamp"

SLOWCONTROL_TIMESTAMP_INDEX = 1
SLOWCONTROL_WP_INDEX = 5
SLOWCONTROL_TRIGGER_RATE_INDEX = 43

VESUVIUS_LAT = 40.82
VESUVIUS_LON = 14.43

# Small ERA5 box around Vesuvius.
ERA5_AREA = [40.875, 14.375, 40.75, 14.50]  # N, W, S, E

OUTPUT_DIR = Path("atmospheric_spaceweather_correlations")

# Where cached auxiliary data (NMDB, Kp, F10.7) is stored so repeat runs
# don't hit the network again.
CACHE_DIR = OUTPUT_DIR / "cache"

NM_STATION = "OULU"

# Number of ERA5 monthly requests to submit to the CDS queue at the same
# time. CDS queues requests server-side regardless of size, so this is
# the main lever you actually have over wall-clock time. Keep this
# modest -- CDS enforces a per-user concurrent-request limit (commonly
# a handful), and submitting more than that just makes some requests
# fail/wait anyway.
ERA5_MAX_PARALLEL_REQUESTS = 4

# If the slow-control timestamps are known to be local Italian time,
# change this to "Europe/Rome". If they are already UTC, leave None.
SLOWCONTROL_TIMEZONE = None

# Plot marker: VES = circle, FS = square.
WP_MARKERS = {"VES": "o", "FS": "s"}

# Fixed colours for WP. The marker changes VES/FS, the colour stays tied to WP.
WP_COLORS = {
    6: "tab:blue",
    10: "tab:orange",
    15: "tab:green",
    20: "tab:red",
    25: "tab:purple",
    30: "tab:brown",
    35: "tab:pink",
}

# Default run ranges.
DEFAULT_RANGES = {
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


# ============================================================
# GENERAL HELPERS
# ============================================================

def parse_range_list(values):
    """Convert ["4000-4010", "5000-5010"] -> [(4000, 4010), (5000, 5010)]."""
    if values is None:
        return None

    ranges = []
    for value in values:
        if "-" not in value:
            raise ValueError(f"Invalid range '{value}'. Use e.g. 4000-4010.")

        a, b = value.split("-", 1)
        a, b = int(a), int(b)

        if b < a:
            raise ValueError(f"Invalid range '{value}': end < start.")

        ranges.append((a, b))

    return ranges


def runs_from_ranges(ranges):
    """Expand [(1,3),(10,12)] -> [1,2,3,10,11,12]."""
    runs = []
    for first, last in ranges:
        runs.extend(range(first, last + 1))
    return runs


def parse_timestamp(value):
    """Parse timestamps robustly: strings, bytes, unix seconds/ms."""
    if isinstance(value, bytes):
        value = value.decode(errors="ignore")

    value = str(value).strip()

    try:
        x = float(value)
        if x > 1e12:
            return pd.to_datetime(x, unit="ms")
        if x > 1e9:
            return pd.to_datetime(x, unit="s")
    except Exception:
        pass

    try:
        ts = pd.Timestamp(value)
        if not pd.isna(ts):
            return ts
    except Exception:
        pass

    return pd.NaT


def timestamp_to_utc(ts):
    """Convert timestamp to timezone-aware UTC."""
    ts = pd.Timestamp(ts)

    if ts.tzinfo is None:
        if SLOWCONTROL_TIMEZONE is not None:
            ts = ts.tz_localize(SLOWCONTROL_TIMEZONE)
        else:
            ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    return ts


def to_utc_series(series):
    """Vectorized tz-normalisation of a Series/array of Timestamps to UTC."""
    ts = pd.to_datetime(series)
    if ts.dt.tz is None:
        return ts.dt.tz_localize("UTC")
    return ts.dt.tz_convert("UTC")


# ============================================================
# FAST INTERVAL AVERAGING
#
# The original script recomputed a boolean mask over the *entire*
# auxiliary time series (ERA5 t2m/sp, Kp, F10.7, NMDB) for every single
# MURAVES run via df.iterrows(). That's O(n_runs * n_timesteps) per
# variable. IntervalAverager instead precomputes prefix sums once per
# variable (O(n_timesteps)) and looks up each run's mean with two
# binary searches (O(log n_timesteps) per run), which is the main
# processing-time fix in this version of the script.
# ============================================================

class IntervalAverager:
    """
    Precomputes prefix sums of a time series so that the mean value
    over an arbitrary [start, end] time window can be looked up in
    O(log n) instead of re-scanning the whole series.

    `times` must be sorted ascending, timezone-aware (UTC) Timestamps.
    NaNs in `values` are ignored (excluded from both sum and count).
    """

    def __init__(self, times, values):
        times = pd.to_datetime(pd.Series(times), utc=True)
        order = np.argsort(times.values)

        self.times_ns = times.values[order].astype("datetime64[ns]").astype(np.int64)
        values = np.asarray(values, dtype=float)[order]

        valid = np.isfinite(values)
        safe_values = np.where(valid, values, 0.0)

        # Prefix sums, index i = sum/count of the first i points.
        self.cum_sum = np.concatenate([[0.0], np.cumsum(safe_values)])
        self.cum_count = np.concatenate([[0], np.cumsum(valid.astype(np.int64))])

        self.values_sorted = values  # kept for nearest-point fallback

    def mean_in_window(self, start, end):
        """Mean of the series within [start, end] inclusive, or NaN."""
        start_ns = np.datetime64(pd.Timestamp(start).tz_convert("UTC")).astype("datetime64[ns]").astype(np.int64)
        end_ns = np.datetime64(pd.Timestamp(end).tz_convert("UTC")).astype("datetime64[ns]").astype(np.int64)

        lo = np.searchsorted(self.times_ns, start_ns, side="left")
        hi = np.searchsorted(self.times_ns, end_ns, side="right")

        count = self.cum_count[hi] - self.cum_count[lo]
        if count == 0:
            return np.nan

        total = self.cum_sum[hi] - self.cum_sum[lo]
        return total / count

    def nearest_value(self, start):
        """Value at the timestamp nearest to `start` (fallback when a
        run has no auxiliary samples inside its window)."""
        if len(self.times_ns) == 0:
            return np.nan

        start_ns = np.datetime64(pd.Timestamp(start).tz_convert("UTC")).astype("datetime64[ns]").astype(np.int64)
        idx = np.searchsorted(self.times_ns, start_ns)
        idx = np.clip(idx, 0, len(self.times_ns) - 1)

        # Check the neighbour too, in case it's closer.
        if idx > 0 and abs(self.times_ns[idx - 1] - start_ns) < abs(self.times_ns[idx] - start_ns):
            idx -= 1

        return self.values_sorted[idx]

    def mean_or_nearest(self, start, end):
        value = self.mean_in_window(start, end)
        if np.isnan(value):
            return self.nearest_value(start)
        return value


def average_series_over_runs(df, averager, use_nearest_fallback=False):
    """Vectorized-ish helper: returns a list of per-run averages using
    an IntervalAverager, in the same row order as df."""
    starts = to_utc_series(df["timestamp_start"])
    ends = to_utc_series(df["timestamp_end"])

    results = []
    for start, end in zip(starts, ends):
        if use_nearest_fallback:
            results.append(averager.mean_or_nearest(start, end))
        else:
            results.append(averager.mean_in_window(start, end))

    return results


# ============================================================
# MURAVES SLOW CONTROL
# ============================================================

def read_slowcontrol(detector, run):
    """Read the slow-control file for one run -> timestamp/WP/trigger_rate."""
    filename = SLOWCONTROL_DIR / detector / f"SLOWCONTROL_run{run}"

    if not filename.exists():
        return None

    timestamp = None
    wp = None
    trigger_rate = None

    try:
        with filename.open("r", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                fields = line.split()

                if len(fields) <= max(
                    SLOWCONTROL_TIMESTAMP_INDEX,
                    SLOWCONTROL_WP_INDEX,
                    SLOWCONTROL_TRIGGER_RATE_INDEX,
                ):
                    continue

                if timestamp is None:
                    candidate = parse_timestamp(fields[SLOWCONTROL_TIMESTAMP_INDEX])
                    if not pd.isna(candidate):
                        timestamp = candidate

                if wp is None:
                    try:
                        wp = int(float(fields[SLOWCONTROL_WP_INDEX]))
                    except Exception:
                        pass

                if trigger_rate is None:
                    try:
                        trigger_rate = float(fields[SLOWCONTROL_TRIGGER_RATE_INDEX])
                    except Exception:
                        pass

                if timestamp is not None and wp is not None and trigger_rate is not None:
                    break

    except Exception as exc:
        print(f"[WARNING] Could not read {filename}: {exc}")
        return None

    if timestamp is None:
        return None

    return {"timestamp_start": timestamp, "wp": wp, "trigger_rate": trigger_rate}


# ============================================================
# MURAVES ROOT DATA
# ============================================================

def read_root_run(detector, run):
    """Read ROOT file -> entries, duration, track rate. No cuts applied."""
    filename = RECONSTRUCTED_DIR / detector / "v1" / f"MURAVES_AnalyzedData_run{run}.root"

    if not filename.exists():
        return None

    try:
        with uproot.open(filename) as root_file:
            tree = root_file[TREE_NAME]
            n_entries = tree.num_entries

            if n_entries < 2:
                return None

            timestamps = tree[TIMESTAMP_BRANCH].array(library="np")

    except Exception as exc:
        print(f"[WARNING] Could not read {filename}: {exc}")
        return None

    timestamps = np.asarray(timestamps, dtype=float)
    timestamps = timestamps[np.isfinite(timestamps)]

    if len(timestamps) < 2:
        return None

    # MURAVES ROOT timestamps are assumed milliseconds.
    t_min, t_max = np.min(timestamps), np.max(timestamps)
    duration_seconds = (t_max - t_min) / 1000.0

    if duration_seconds <= 0:
        return None

    track_rate = n_entries / duration_seconds

    return {
        "track_entries": int(n_entries),
        "duration_seconds": float(duration_seconds),
        "track_rate": float(track_rate),
    }


# ============================================================
# COLLECT MURAVES RUNS
# ============================================================

def collect_runs(detector, run_ranges):
    rows = []
    print(f"[INFO] Collecting {detector}...")

    for run_type in ["VES", "FS"]:
        ranges = run_ranges.get(run_type, [])
        runs = runs_from_ranges(ranges)

        for run in runs:
            slow = read_slowcontrol(detector, run)
            root = read_root_run(detector, run)

            if slow is None or root is None:
                continue

            start = timestamp_to_utc(slow["timestamp_start"])
            end = start + pd.Timedelta(seconds=root["duration_seconds"])

            rows.append({
                "detector": detector,
                "run": run,
                "run_type": run_type,
                "wp": slow["wp"],
                "trigger_rate": slow["trigger_rate"],
                "track_entries": root["track_entries"],
                "duration_seconds": root["duration_seconds"],
                "track_rate": root["track_rate"],
                "timestamp_start": start,
                "timestamp_end": end,
            })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        return df

    df = df.sort_values(["timestamp_start", "detector", "run"]).reset_index(drop=True)
    print(f"[INFO] Found {len(df)} runs.")

    return df


# ============================================================
# ERA5
# ============================================================

def _era5_month_bounds(period, start, end):
    year, month = period.year, period.month
    month_start = pd.Timestamp(year=year, month=month, day=1)
    month_end = period.end_time.normalize()

    first_day = max(start.normalize(), month_start)
    last_day = min(end.normalize(), month_end)

    days = [f"{day:02d}" for day in range(first_day.day, last_day.day + 1)]

    return first_day, last_day, days


def _retrieve_era5_month(client, period, days, outfile):
    year, month = period.year, period.month

    request = {
        "product_type": ["reanalysis"],
        "variable": ["2m_temperature", "surface_pressure"],
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": days,
        "time": [f"{hour:02d}:00" for hour in range(24)],
        # NetCDF instead of GRIB: xarray reads it with the "netcdf4"
        # engine, which is a plain Python package. GRIB would need
        # cfgrib *and* the ecCodes C library installed underneath it,
        # which is a much heavier and more fragile dependency to get
        # working in a conda env.
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": ERA5_AREA,
    }

    client.retrieve("reanalysis-era5-single-levels", request, str(outfile))
    return outfile


def download_era5(start, end, max_parallel=ERA5_MAX_PARALLEL_REQUESTS):
    """
    Download ERA5 hourly 2-m temperature and surface pressure
    month-by-month as GRIB.

    NOTE ON SPEED: the CDS backend queues every request server-side.
    A single month's request commonly waits minutes to well over an
    hour before it even starts processing, independent of how small
    the request is. That queueing is normal and outside this script's
    control. The one lever we do have is submitting several months'
    requests at once instead of strictly sequentially, so their queue
    waits overlap. `max_parallel` controls how many are in flight at
    once; keep it at or below your CDS account's concurrent-request
    limit or extra requests will simply queue behind the others anyway.

    Already-downloaded months are reused (cached on disk), so re-running
    the script for an overlapping period is fast.
    """

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    if start.tzinfo is not None:
        start = start.tz_convert("UTC").tz_localize(None)
    if end.tzinfo is not None:
        end = end.tz_convert("UTC").tz_localize(None)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client()

    months = pd.period_range(start=start.to_period("M"), end=end.to_period("M"), freq="M")

    monthly_files = [None] * len(months)
    to_fetch = []  # (index, period, days, outfile)

    for i, period in enumerate(months):
        first_day, last_day, days = _era5_month_bounds(period, start, end)
        outfile = OUTPUT_DIR / f"era5_{period.year}_{period.month:02d}.nc"

        if outfile.exists():
            print(f"[INFO] Using cached {outfile}")
            monthly_files[i] = outfile
            continue

        print(f"[INFO] Queuing ERA5 {period.year}-{period.month:02d}: {first_day.date()} -> {last_day.date()}")
        to_fetch.append((i, period, days, outfile))

    if to_fetch:
        print(
            f"[INFO] Submitting {len(to_fetch)} ERA5 month request(s) "
            f"to CDS (up to {max_parallel} in flight at once). "
            f"This step is normally the slowest part of the script "
            f"because of CDS server-side queueing -- that's expected."
        )

        with ThreadPoolExecutor(max_workers=max(1, max_parallel)) as pool:
            futures = {
                pool.submit(_retrieve_era5_month, client, period, days, outfile): i
                for i, period, days, outfile in to_fetch
            }

            for future in as_completed(futures):
                i = futures[future]
                outfile = future.result()
                print(f"[INFO] Finished ERA5 {months[i]}: {outfile}")
                monthly_files[i] = outfile

    return monthly_files


def load_era5(monthly_files):
    """Open all monthly NetCDF files and concatenate along time."""
    if not monthly_files:
        raise RuntimeError("No ERA5 NetCDF files available.")

    print("[INFO] Loading ERA5 NetCDF files...")

    datasets = []

    for filename in monthly_files:
        print(f"[INFO] Reading {filename}")

        try:
            # NetCDF is read with the "netcdf4" engine, a plain Python
            # package -- no cfgrib/ecCodes system install required.
            ds = xr.open_dataset(filename, engine="netcdf4")

        except Exception as exc:
            raise RuntimeError(
                f"Could not read ERA5 NetCDF file {filename}.\n"
                f"Make sure the 'netcdf4' Python package is installed "
                f"(pip install netCDF4).\n"
                f"Original error: {exc}"
            ) from exc

        # The CDS NetCDF converter sometimes names the time dimension
        # "valid_time" instead of "time" -- normalize it so the rest
        # of the pipeline doesn't have to care which one it got.
        if "time" not in ds.coords and "valid_time" in ds.coords:
            ds = ds.rename({"valid_time": "time"})

        datasets.append(ds)

    try:
        combined = xr.concat(datasets, dim="time")
        combined = combined.sortby("time")

        times = combined["time"].values
        _, unique_indices = np.unique(times, return_index=True)
        combined = combined.isel(time=np.sort(unique_indices))

        combined = combined.sel(latitude=VESUVIUS_LAT, longitude=VESUVIUS_LON, method="nearest")

        combined = combined.load()

    finally:
        for ds in datasets:
            ds.close()

    print("[INFO] ERA5 loaded.")

    return combined


def add_era5(df, era5):
    """Add mean ERA5 temperature and pressure over each MURAVES run interval."""
    era5_times = pd.to_datetime(era5["time"].values, utc=True)

    t2m_averager = IntervalAverager(era5_times, era5["t2m"].values)
    sp_averager = IntervalAverager(era5_times, era5["sp"].values)

    t2m_kelvin = average_series_over_runs(df, t2m_averager, use_nearest_fallback=True)
    sp_pa = average_series_over_runs(df, sp_averager, use_nearest_fallback=True)

    # ERA5: t2m = Kelvin, sp = Pa.
    df["temperature_C"] = [v - 273.15 if np.isfinite(v) else np.nan for v in t2m_kelvin]
    df["pressure_hPa"] = [v / 100.0 if np.isfinite(v) else np.nan for v in sp_pa]

    return df


# ============================================================
# GFZ Kp / F10.7
# ============================================================

def _gfz_cache_file(start, end, index):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{index}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
    return CACHE_DIR / f"gfz_{tag}.csv"


def get_gfz_index(start, end, index):
    """Retrieve a GFZ index (Kp or Fobs) via the official JSON API, cached on disk."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")

    cache_file = _gfz_cache_file(start, end, index)
    if cache_file.exists():
        print(f"[INFO] Using cached GFZ {index} ({cache_file})")
        cached = pd.read_csv(cache_file, parse_dates=["timestamp"])
        cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True)
        return cached

    url = (
        "https://kp.gfz.de/app/json/"
        f"?start={start.strftime('%Y-%m-%dT%H:%M:%SZ')}"
        f"&end={end.strftime('%Y-%m-%dT%H:%M:%SZ')}"
        f"&index={index}"
    )

    if index == "Kp":
        url += "&status=def"

    print(f"[INFO] Downloading GFZ {index}...")

    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
    except Exception as exc:
        raise RuntimeError(f"GFZ request failed for {index}: {exc}") from exc

    result = None

    try:
        obj = pd.read_json(io.BytesIO(data))
        if obj.shape[1] >= 2:
            obj = obj.iloc[:, :2]
            obj.columns = ["timestamp", "value"]
            obj["timestamp"] = pd.to_datetime(obj["timestamp"], utc=True, errors="coerce")
            obj["value"] = pd.to_numeric(obj["value"], errors="coerce")
            obj = obj.dropna(subset=["timestamp", "value"])
            if len(obj) > 0:
                result = obj
    except Exception:
        pass

    if result is None:
        try:
            obj = json.loads(data.decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Could not decode GFZ JSON for {index}: {exc}") from exc

        if isinstance(obj, dict):
            time_values = next(
                (obj[k] for k in ["time", "times", "timestamp", "timestamps"] if k in obj), None
            )
            data_values = next((obj[k] for k in ["value", "values", index] if k in obj), None)

            if time_values is not None and data_values is not None:
                result = pd.DataFrame({"timestamp": time_values, "value": data_values})
                result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
                result["value"] = pd.to_numeric(result["value"], errors="coerce")
                result = result.dropna(subset=["timestamp", "value"])

    if result is None:
        raise RuntimeError(f"Unrecognised GFZ JSON format for {index}.")

    result.to_csv(cache_file, index=False)

    return result


def prepare_gfz_data(start, end):
    """Download (or load cached) Kp and observed F10.7."""
    kp = get_gfz_index(start, end, "Kp").rename(columns={"value": "kp"})[["timestamp", "kp"]]
    f107 = get_gfz_index(start, end, "Fobs").rename(columns={"value": "f107"})[["timestamp", "f107"]]
    return kp, f107


def add_gfz(df, kp, f107):
    """Average GFZ Kp and F10.7 values over each run."""
    kp_averager = IntervalAverager(kp["timestamp"], kp["kp"])
    f107_averager = IntervalAverager(f107["timestamp"], f107["f107"])

    df["kp"] = average_series_over_runs(df, kp_averager)
    df["f107"] = average_series_over_runs(df, f107_averager)

    return df


# ============================================================
# NMDB OULU
# ============================================================

def _nmdb_cache_file(year):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"nmdb_{NM_STATION}_{year}.csv"


def get_nmdb_year(year):
    """Download one full year of hourly Oulu neutron monitor data, cached on disk."""
    cache_file = _nmdb_cache_file(year)

    if cache_file.exists():
        print(f"[INFO] Using cached NMDB {NM_STATION} {year} ({cache_file})")
        cached = pd.read_csv(cache_file, parse_dates=["timestamp"])
        cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True)
        return cached

    url = (
        "https://www.nmdb.eu/nest/draw_graph.php"
        "?formchk=1"
        f"&stations[]={NM_STATION}"
        "&tabchoice=revori"
        "&dtype=corr_for_efficiency"
        "&tresolution=60"
        "&force=1"
        "&yunits=0"
        "&date_choice=bydate"
        "&start_day=1&start_month=1"
        f"&start_year={year}"
        "&start_hour=0&start_min=0"
        "&end_day=31&end_month=12"
        f"&end_year={year}"
        "&end_hour=23&end_min=59"
        "&output=ascii"
    )

    print(f"[INFO] Downloading NMDB OULU {year}...")

    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            text = response.read().decode("utf-8", errors="ignore")
    except Exception as exc:
        raise RuntimeError(f"NMDB request failed for {year}: {exc}") from exc

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 3:
            continue

        try:
            timestamp = pd.to_datetime(parts[0] + " " + parts[1], utc=True)
            value = float(parts[-1])
        except Exception:
            continue

        if not np.isfinite(value):
            continue

        rows.append({"timestamp": timestamp, "neutron_monitor": value})

    result = pd.DataFrame(rows)

    if len(result) == 0:
        raise RuntimeError(f"No NMDB data parsed for {year}.")

    result = result.sort_values("timestamp").reset_index(drop=True)
    result.to_csv(cache_file, index=False)

    return result


def prepare_nmdb(start, end):
    """Load only the years required by the MURAVES data (cached per year)."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    years = range(start.year, end.year + 1)
    frames = [get_nmdb_year(year) for year in years]

    nmdb = pd.concat(frames, ignore_index=True)
    nmdb = nmdb[(nmdb["timestamp"] >= start) & (nmdb["timestamp"] <= end)].copy()

    return nmdb


def add_nmdb(df, nmdb):
    """Average neutron monitor rate over each run."""
    averager = IntervalAverager(nmdb["timestamp"], nmdb["neutron_monitor"])
    df["neutron_monitor"] = average_series_over_runs(df, averager)
    return df


# ============================================================
# PLOTTING
# ============================================================

def make_plot(df, x, y, xlabel, ylabel, filename, title):
    """One scatter plot. Colour = WP, marker = VES/FS."""
    fig, ax = plt.subplots(figsize=(8, 6))
    plotted = False

    for wp in sorted(df["wp"].dropna().unique()):
        try:
            wp_int = int(wp)
        except Exception:
            continue

        for run_type in ["VES", "FS"]:
            sub = df[(df["wp"] == wp) & (df["run_type"] == run_type)].copy()

            if len(sub) == 0:
                continue

            sub = sub[
                np.isfinite(pd.to_numeric(sub[x], errors="coerce"))
                & np.isfinite(pd.to_numeric(sub[y], errors="coerce"))
            ]

            if len(sub) == 0:
                continue

            color = WP_COLORS.get(wp_int, "black")

            ax.scatter(
                sub[x], sub[y],
                marker=WP_MARKERS[run_type],
                color=color,
                label=f"WP {wp_int} - {run_type}",
                alpha=0.7,
                s=35,
            )
            plotted = True

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    if plotted:
        ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()

    output = OUTPUT_DIR / filename
    fig.savefig(output, dpi=200)
    plt.close(fig)

    print(f"[INFO] Saved {output}")


def make_all_plots(df):
    """Produce exactly the requested 10 plots."""
    make_plot(df, "temperature_C", "track_rate", "Temperature [°C]", "Track rate [Hz]",
              "01_track_rate_vs_temperature.png", "MURAVES track rate vs temperature")

    make_plot(df, "pressure_hPa", "track_rate", "Surface pressure [hPa]", "Track rate [Hz]",
              "02_track_rate_vs_pressure.png", "MURAVES track rate vs pressure")

    make_plot(df, "temperature_C", "trigger_rate", "Temperature [°C]", "Trigger rate [Hz]",
              "03_trigger_rate_vs_temperature.png", "MURAVES trigger rate vs temperature")

    make_plot(df, "pressure_hPa", "trigger_rate", "Surface pressure [hPa]", "Trigger rate [Hz]",
              "04_trigger_rate_vs_pressure.png", "MURAVES trigger rate vs pressure")

    make_plot(df, "kp", "track_rate", "Kp", "Track rate [Hz]",
              "05_track_rate_vs_kp.png", "MURAVES track rate vs Kp")

    make_plot(df, "kp", "trigger_rate", "Kp", "Trigger rate [Hz]",
              "06_trigger_rate_vs_kp.png", "MURAVES trigger rate vs Kp")

    make_plot(df, "f107", "track_rate", "F10.7 [sfu]", "Track rate [Hz]",
              "07_track_rate_vs_f107.png", "MURAVES track rate vs F10.7")

    make_plot(df, "f107", "trigger_rate", "F10.7 [sfu]", "Trigger rate [Hz]",
              "08_trigger_rate_vs_f107.png", "MURAVES trigger rate vs F10.7")

    make_plot(df, "neutron_monitor", "track_rate", "Oulu neutron monitor rate", "Track rate [Hz]",
              "09_track_rate_vs_neutron_monitor.png", "MURAVES track rate vs neutron monitor")

    make_plot(df, "neutron_monitor", "trigger_rate", "Oulu neutron monitor rate", "Trigger rate [Hz]",
              "10_trigger_rate_vs_neutron_monitor.png", "MURAVES trigger rate vs neutron monitor")


# ============================================================
# COMMAND LINE
# ============================================================

def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Correlate MURAVES track/trigger rates with atmospheric and space-weather conditions."
    )

    parser.add_argument("--detector", choices=["NERO", "ROSSO", "BLU"], help="Process one detector.")
    parser.add_argument("--detectors", nargs="+", choices=["NERO", "ROSSO", "BLU"], help="Process multiple detectors.")
    parser.add_argument("--ves-ranges", nargs="+", help="VES run ranges, e.g. 4000-4010 5000-5010")
    parser.add_argument("--fs-ranges", nargs="+", help="FS run ranges, e.g. 10061-10070")
    parser.add_argument(
        "--era5-workers",
        type=int,
        default=ERA5_MAX_PARALLEL_REQUESTS,
        help=(
            "How many ERA5 monthly requests to submit to the CDS queue "
            "in parallel (default: %(default)s). Raising this can shrink "
            "wall-clock time when several months are missing from the "
            "cache, since CDS queue waits then overlap instead of "
            "stacking sequentially -- but don't set it above your CDS "
            "account's concurrent-request limit."
        ),
    )

    return parser


# ============================================================
# MAIN
# ============================================================

def main():
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.detector is not None:
        detectors = [args.detector]
    elif args.detectors is not None:
        detectors = args.detectors
    else:
        detectors = ["NERO", "ROSSO", "BLU"]

    if args.ves_ranges is not None or args.fs_ranges is not None:
        ves_ranges = parse_range_list(args.ves_ranges) if args.ves_ranges is not None else []
        fs_ranges = parse_range_list(args.fs_ranges) if args.fs_ranges is not None else []

        ranges_by_detector = {
            detector: {"VES": ves_ranges, "FS": fs_ranges} for detector in detectors
        }
    else:
        ranges_by_detector = {detector: DEFAULT_RANGES[detector] for detector in detectors}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Collect MURAVES data
    # --------------------------------------------------------

    frames = [
        collect_runs(detector, ranges_by_detector[detector])
        for detector in detectors
    ]
    frames = [f for f in frames if len(f) > 0]

    if not frames:
        raise RuntimeError("No MURAVES runs were found.")

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("timestamp_start").reset_index(drop=True)

    print(f"[INFO] Total MURAVES runs: {len(df)}")

    # --------------------------------------------------------
    # 2. Determine global time range
    # --------------------------------------------------------

    global_start = pd.Timestamp(df["timestamp_start"].min())
    global_end = pd.Timestamp(df["timestamp_end"].max())

    global_start = global_start.tz_localize("UTC") if global_start.tzinfo is None else global_start.tz_convert("UTC")
    global_end = global_end.tz_localize("UTC") if global_end.tzinfo is None else global_end.tz_convert("UTC")

    print("[INFO] Global time range:")
    print(f"        {global_start}")
    print(f"        {global_end}")

    # --------------------------------------------------------
    # 3. ERA5
    # --------------------------------------------------------

    monthly_era5_files = download_era5(global_start, global_end, max_parallel=args.era5_workers)
    era5 = load_era5(monthly_era5_files)
    df = add_era5(df, era5)

    # --------------------------------------------------------
    # 4. GFZ Kp and F10.7
    # --------------------------------------------------------

    kp, f107 = prepare_gfz_data(global_start, global_end)
    df = add_gfz(df, kp, f107)

    # --------------------------------------------------------
    # 5. Oulu neutron monitor
    # --------------------------------------------------------

    nmdb = prepare_nmdb(global_start, global_end)
    df = add_nmdb(df, nmdb)

    # --------------------------------------------------------
    # 6. Save CSV
    # --------------------------------------------------------

    csv_file = OUTPUT_DIR / "muraves_atmospheric_spaceweather.csv"

    df["timestamp_start"] = df["timestamp_start"].astype(str)
    df["timestamp_end"] = df["timestamp_end"].astype(str)

    df.to_csv(csv_file, index=False)
    print(f"[INFO] Saved CSV: {csv_file}")

    # --------------------------------------------------------
    # 7. Make exactly 10 plots
    # --------------------------------------------------------

    make_all_plots(df)

    print("\n[INFO] Analysis complete.")


if __name__ == "__main__":
    main()