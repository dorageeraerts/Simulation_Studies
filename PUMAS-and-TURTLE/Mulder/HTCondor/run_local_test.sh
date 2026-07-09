#!/bin/bash
# run_local_test.sh - sequential (no Condor) smoke test on a small grid.
# Run this after generate_bins.sh (with a small test range) to sanity-check
# the pipeline before scaling up to Condor.
set -e

if [ ! -f chunks.txt ] || [ ! -f run_config.txt ]; then
    echo "chunks.txt / run_config.txt not found - run generate_bins.sh first" >&2
    exit 1
fi

source muraves_mulder_config.sh
source run_config.txt

CLUSTER_ID="local_test_$(date +%Y%m%d_%H%M%S)"

while read -r AZ_START AZ_END EL; do
    for ((j=1; j<=N_JOBS_PER_BIN; j++)); do
        ./run_bin_job.sh "$AZ_START" "$AZ_END" "$EL" "$j" "$CLUSTER_ID" "$DPHI" "$DEL" "$RHO" "$N_E" "$NEVENTS"
    done
done < chunks.txt

export CLUSTER_ID
./combine_all_bins.sh