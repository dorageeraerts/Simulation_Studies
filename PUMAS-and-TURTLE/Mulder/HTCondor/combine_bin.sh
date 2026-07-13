#!/bin/bash
# combine_bin.sh — combine the per-job-replica values for ONE angular bin.
set -e

AZ=$1
EL=$2
INDIR=${3:-relativeApproach/condor_parts}
OUTDIR=${4:-relativeApproach/combined_bins}

# Scale tolerance to bin width rather than a fixed constant, so it stays
# safely smaller than the spacing between adjacent bins regardless of DPHI.
DPHI_FOR_TOL=${DPHI:-0.2}
TOL=${AZ_MATCH_TOL:-$(awk -v d="$DPHI_FOR_TOL" 'BEGIN{printf "%.9g", d/10}')}

mkdir -p "$OUTDIR"

pattern="${INDIR}/flux_5m_02bin_rho*_discrete_200T_az*_el${EL}_job*.txt"
files=( $pattern )

if [ ! -e "${files[0]}" ]; then
    echo "No job files found for el=$EL (pattern: $pattern)" >&2
    exit 1
fi

AZ_INT=$(printf "%.1f" "$AZ")
outfile="${OUTDIR}/combined_az${AZ_INT}_el${EL}.txt"

awk -v az="$AZ" -v el="$EL" -v tol="$TOL" '
    ($1 > az - tol && $1 < az + tol) {
        n++
        flux[n] = $3
        err1[n] = $4
        nrep += $5
        sum += $3
    }
    END {
        if (n == 0) { exit 1 }
        mean = sum / n
        ss = 0
        for (i = 1; i <= n; i++) { d = flux[i] - mean; ss += d * d }
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