#!/bin/bash
set -euo pipefail

shopt -s nullglob

CLUSTER_ID=${CLUSTER_ID:?"CLUSTER_ID must be exported before calling this script"}

source muraves_mulder_config.sh

if [ -f run_config.txt ]; then
    source run_config.txt
else
    echo "run_config.txt not found" >&2
    exit 1
fi

INDIR="${OUTPUT_HOST}/${CLUSTER_ID}"
COMBDIR="${INDIR}/combined_bins"

if [ ! -d "$COMBDIR" ]; then
    echo "ERROR: $COMBDIR does not exist" >&2
    exit 1
fi

PHI_MIN_ABS=$(awk -v phi="$PHI_MIN" -v off="$PHI_OFFSET" 'BEGIN {printf "%.0f", phi + off}')
PHI_MAX_ABS=$(awk -v phi="$PHI_MAX" -v off="$PHI_OFFSET" 'BEGIN {printf "%.0f", phi + off}')

FINAL="${INDIR}/flux_all_bins_phi${PHI_MIN_ABS}-${PHI_MAX_ABS}_el${EL_MIN}-${EL_MAX}_dphi${DPHI}_del${DEL}_rho${RHO}_nE${N_E}_nEvents${NEVENTS}.txt"

echo "Merging files from:"
echo "  $COMBDIR"
echo "Output:"
echo "  $FINAL"

NFILES=$(find "$COMBDIR" -name 'combined_az*_el*.txt' | wc -l)

if [ "$NFILES" -eq 0 ]; then
    echo "ERROR: no combined bin files found" >&2
    exit 1
fi

echo "Found $NFILES combined bin files"

# Safe for thousands/millions of files:
# find feeds filenames incrementally, no ARG_MAX problem
find "$COMBDIR" -name 'combined_az*_el*.txt' -print0 \
    | xargs -0 -r sort -k1,1n -k2,2n -m > "$FINAL"


EXPECTED_BINS=$(wc -l < bins.txt)
ACTUAL_BINS=$(wc -l < "$FINAL")

echo "Final combined file:"
echo "  $FINAL"
echo "Bins:"
echo "  $ACTUAL_BINS / $EXPECTED_BINS"

if [ "$ACTUAL_BINS" -ne "$EXPECTED_BINS" ]; then
    echo "WARNING: bin count mismatch!" >&2
    exit 1
fi

echo "Merge successful"

# Uncomment only after checking the output:
# rm -rf "$COMBDIR"
# find "$INDIR" -maxdepth 1 -name 'flux_5m_*_job*.txt' -delete