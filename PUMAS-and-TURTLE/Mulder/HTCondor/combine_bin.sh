#!/bin/bash
# combine_bin.sh — combine the per-job-replica values for ONE angular bin.
# Pure bash + awk, no Python needed.
#
# Usage: ./combine_bin.sh <az> <el> [indir] [outdir]
#
# CHANGED FOR CHUNKING: each job now covers a whole az-chunk (az_start to
# az_end), so a job's output file has one line PER AZ BIN in that chunk,
# not one line per file like before:
#   az el flux_ave flux_err n_replicas
# We can no longer glob filenames by exact az (the filename only has
# az_start-az_end), so we glob broadly by el/job and let awk pick out just
# the row(s) whose az column matches the bin we actually want.
#
# We treat each matching row's flux_ave as one independent sample and report:
#   flux_mean = mean over job replicas
#   flux_sem  = std over job replicas (ddof=1) / sqrt(n_jobs)
# (n_jobs and total replica count are reported too, for sanity-checking)

set -e

AZ=$1
EL=$2
INDIR=${3:-relativeApproach/condor_parts}
OUTDIR=${4:-relativeApproach/combined_bins}
TOL=${AZ_MATCH_TOL:-1e-6}   # float-compare tolerance when matching az within a file

mkdir -p "$OUTDIR"

# ADJUST if MuravesFluxmeter_HTC.py's own output filename convention changed:
# az_start-az_end replaces the old single "az${AZ}" segment.
pattern="${INDIR}/flux_5m_02bin_rho*_discrete_200T_az*_el${EL}_job*.txt"
files=( $pattern )

if [ ! -e "${files[0]}" ]; then
    echo "No job files found for el=$EL (pattern: $pattern)" >&2
    exit 1
fi

outfile="${OUTDIR}/combined_az${AZ}_el${EL}.txt"

awk -v az="$AZ" -v el="$EL" -v tol="$TOL" '
    ($1 > az - tol && $1 < az + tol) {
        n++
        flux[n]  = $3
        err1[n]  = $4
        nrep    += $5
        sum     += $3
    }
    END {
        if (n == 0) { exit 1 }
        mean = sum / n
        ss = 0
        for (i = 1; i <= n; i++) {
            d = flux[i] - mean
            ss += d * d
        }
        sem = (n > 1) ? sqrt(ss / (n - 1)) / sqrt(n) : err1[1]
        printf "%s %s %.6e %.6e %d %d\n", az, el, mean, sem, n, nrep
    }
' "${files[@]}" > "$outfile"

if [ ! -s "$outfile" ]; then
    echo "No matching rows for az=$AZ el=$EL across ${#files[@]} job file(s)" >&2
    rm -f "$outfile"
    exit 1
fi

cat "$outfile"
echo "-> wrote $outfile" >&2