#!/bin/bash
# generate_bins.sh <PHI_MIN> <PHI_MAX> <EL_MIN> <EL_MAX> <DPHI> <DEL> <RHO> <N_E> <NEVENTS> <N_AZ_PER_JOB>
#
# Writes:
#   run_config.txt - all run parameters, reused by every downstream script
#   bins.txt       - one row per individual az/el bin (used by combine_all_bins.sh)
#   chunks.txt     - one row per az-chunk per el: "az_start az_end el"
#                    (used by generate_job_list.sh to build Condor jobs)
set -e

PHI_MIN=$1; PHI_MAX=$2; EL_MIN=$3; EL_MAX=$4
DPHI=$5; DEL=$6; RHO=$7; N_E=$8; NEVENTS=$9; N_AZ_PER_JOB=${10}

if [ -z "$N_AZ_PER_JOB" ]; then
    echo "Usage: $0 PHI_MIN PHI_MAX EL_MIN EL_MAX DPHI DEL RHO N_E NEVENTS N_AZ_PER_JOB" >&2
    exit 1
fi

cat > run_config.txt <<EOF
PHI_MIN=$PHI_MIN
PHI_MAX=$PHI_MAX
EL_MIN=$EL_MIN
EL_MAX=$EL_MAX
DPHI=$DPHI
DEL=$DEL
RHO=$RHO
N_E=$N_E
NEVENTS=$NEVENTS
N_AZ_PER_JOB=$N_AZ_PER_JOB
EOF

> bins.txt
> chunks.txt

awk -v phi_min="$PHI_MIN" -v phi_max="$PHI_MAX" -v el_min="$EL_MIN" -v el_max="$EL_MAX" \
    -v dphi="$DPHI" -v del="$DEL" -v n_az="$N_AZ_PER_JOB" '
BEGIN {
    for (el = el_min; el < el_max - 1e-9; el += del) {
        chunk_start = phi_min
        count = 0
        for (az = phi_min; az < phi_max - 1e-9; az += dphi) {
            printf "%.6g %.6g\n", az, el >> "bins.txt"
            count++
            if (count == n_az) {
                az_end = az + dphi
                printf "%.6g %.6g %.6g\n", chunk_start, az_end, el >> "chunks.txt"
                chunk_start = az_end
                count = 0
            }
        }
        # flush a final partial chunk (fewer than n_az az bins left over)
        if (count > 0) {
            az_end = chunk_start + count * dphi
            printf "%.6g %.6g %.6g\n", chunk_start, az_end, el >> "chunks.txt"
        }
    }
}'

echo "Wrote $(wc -l < bins.txt) bins and $(wc -l < chunks.txt) chunks to bins.txt / chunks.txt"