#!/bin/bash
# combine_bin.sh — combine the per-job-replica values for ONE angular bin.
# Writes its one-line result to stdout; caller is responsible for collecting it.
#
# Input columns (per replica): az el flux_rock flux_rock_err flux_open transmission transmission_err N_events
# Output columns: az el flux_rock_mean flux_rock_sem flux_open transmission_mean transmission_sem N_events_total
set -e

AZ=$1
EL=$2
DPHI=$3
INDIR=${4:-relativeApproach/condor_parts}

# Scale tolerance to bin width rather than a fixed constant, so it stays
# safely smaller than the spacing between adjacent bins regardless of DPHI.
DPHI_FOR_TOL=${DPHI:-0.2}
TOL=${AZ_MATCH_TOL:-$(awk -v d="$DPHI_FOR_TOL" 'BEGIN{printf "%.9g", d/10}')}

pattern="${INDIR}/flux_5m_*bin_rho*_discrete_100T_az*_el${EL}_job*.txt"
files=( $pattern )

if [ ! -e "${files[0]}" ]; then
    echo "No job files found for el=$EL (pattern: $pattern)" >&2
    exit 1
fi

if ! result=$(awk -v az="$AZ" -v el="$EL" -v tol="$TOL" '
    ($1 > az - tol && $1 < az + tol) {
        n++
        flux_rock[n]     = $3
        flux_open_val    = $5   # deterministic - same every replica, just keep last
        trans[n]         = $6
        nevents_sum      += $8
        sum_flux_rock    += $3
        sum_trans        += $6
    }
    END {
        if (n == 0) { exit 1 }

        mean_flux_rock = sum_flux_rock / n
        mean_trans     = sum_trans / n

        ss_flux_rock = 0
        ss_trans     = 0
        for (i = 1; i <= n; i++) {
            d1 = flux_rock[i] - mean_flux_rock
            ss_flux_rock += d1 * d1
            d2 = trans[i] - mean_trans
            ss_trans += d2 * d2
        }

        sem_flux_rock = (n > 1) ? sqrt(ss_flux_rock / (n - 1)) / sqrt(n) : 0
        sem_trans     = (n > 1) ? sqrt(ss_trans / (n - 1)) / sqrt(n) : 0

        printf "%s %s %.6e %.6e %.6e %.6e %.6e %d\n", \
            az, el, mean_flux_rock, sem_flux_rock, flux_open_val, mean_trans, sem_trans, nevents_sum
    }
' "${files[@]}"); then
    echo "No matching rows for az=$AZ el=$EL across ${#files[@]} job file(s)" >&2
    exit 1
fi

echo "$result"