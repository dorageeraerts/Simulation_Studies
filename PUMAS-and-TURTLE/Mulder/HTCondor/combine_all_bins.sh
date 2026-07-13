#!/bin/bash
# Run AFTER all Condor jobs have finished. submit_all_bins.sh already waits
# (condor_wait) and calls this automatically with CLUSTER_ID exported -
# only run it manually if CLUSTER_ID is set to match a real run.
#
# 1) combines the per-job-replica files for each bin into one line per bin
# 2) concatenates + sorts all bins into one final file, named after the
#    actual run parameters (phi range, el range, d-phi, d-el, rho, n-E, nEvents)
#set -e

set -euo pipefail
shopt -s nullglob   # so an empty combined_bins/ glob doesn't pass a literal string to cat

CLUSTER_ID=${CLUSTER_ID:?"CLUSTER_ID must be exported before calling combine_all_bins.sh"}
source muraves_mulder_config.sh   # OUTPUT_HOST base path

if [ -f run_config.txt ]; then
    source run_config.txt   # KEY=VALUE lines written by generate_bins.sh
else
    echo "run_config.txt not found -- run ./generate_bins.sh first" >&2
    exit 1
fi

# ADJUST if your job output actually lands somewhere else: this assumes
# run_bin_job.sh copies each job's finished output to OUTPUT_HOST/CLUSTER_ID
# (its node-local scratch dir is cleaned up after copying).
INDIR="${OUTPUT_HOST}/${CLUSTER_ID}"
echo "$INDIR" >&2
COMBDIR="${INDIR}/combined_bins"

if [ ! -d "$INDIR" ]; then
    echo "ERROR: $INDIR does not exist - no job output was produced (check Condor logs / mamba errors)" >&2
    exit 1
fi

PHI_MIN_ABS=$(awk -v phi="$PHI_MIN" -v off="$PHI_OFFSET" 'BEGIN {printf "%.0f", phi + off}')
PHI_MAX_ABS=$(awk -v phi="$PHI_MAX" -v off="$PHI_OFFSET" 'BEGIN {printf "%.0f", phi + off}')

FINAL="${INDIR}/flux_all_bins_phi${PHI_MIN_ABS}-${PHI_MAX_ABS}_el${EL_MIN}-${EL_MAX}_dphi${DPHI}_del${DEL}_rho${RHO}_nE${N_E}_nEvents${NEVENTS}.txt"
#FINAL="${INDIR}/flux_all_bins_phi${PHI_MIN}-${PHI_MAX}_el${EL_MIN}-${EL_MAX}_dphi${DPHI}_del${DEL}_rho${RHO}_nE${N_E}_nEvents${NEVENTS}.txt"

mkdir -p "$COMBDIR"

MISSING=0
while read -r az el; do
    [ -z "$az" ] && continue
    az_abs=$(awk -v az="$az" -v off="$PHI_OFFSET" 'BEGIN {printf "%.6f", az + off}')
    if ! ./combine_bin.sh "$az_abs" "$el" "$INDIR" "$COMBDIR"; then
        echo "WARNING: bin az=$az el=$el missing or incomplete - skipped" >&2
        MISSING=$((MISSING + 1))
    fi
done < bins.txt

#COMBINED_FILES=( "${COMBDIR}"/combined_az*_el*.txt )
#if [ "${#COMBINED_FILES[@]}" -gt 0 ]; then
#    cat "${COMBINED_FILES[@]}" | sort -k1,1n -k2,2n > "$FINAL"
#else
#    : > "$FINAL"
#fi

if find "$COMBDIR" -name 'combined_az*_el*.txt' -print -quit | grep -q .; then
    find "$COMBDIR" -name 'combined_az*_el*.txt' -print0 \
        | xargs -0 -r cat \
        | sort -k1,1n -k2,2n > "$FINAL"
else
    : > "$FINAL"
fi

EXPECTED_BINS=$(wc -l < bins.txt)
ACTUAL_BINS=$(wc -l < "$FINAL")

echo "Final combined file: $FINAL ($ACTUAL_BINS / $EXPECTED_BINS bins)"

if [ ! -s "$FINAL" ] || [ "$MISSING" -gt 0 ] || [ "$ACTUAL_BINS" -ne "$EXPECTED_BINS" ]; then
    echo "WARNING: run incomplete ($MISSING bin(s) missing) - leaving intermediate files in $INDIR for debugging, NOT cleaning up" >&2
    exit 1
fi

echo "All $EXPECTED_BINS bins present - cleaning up intermediate files, keeping only $FINAL"
rm -rf "$COMBDIR"
find "$INDIR" -maxdepth 1 -name 'flux_5m_*_job*.txt' -delete