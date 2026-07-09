#!/bin/bash
# Run AFTER all Condor jobs have finished. submit_all_bins.sh already waits
# (condor_wait) and calls this automatically with CLUSTER_ID exported -
# only run it manually if CLUSTER_ID is set to match a real run.
#
# 1) combines the per-job-replica files for each bin into one line per bin
# 2) concatenates + sorts all bins into one final file, named after the
#    actual run parameters (phi range, el range, d-phi, d-el, rho, n-E, nEvents)
set -e

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
COMBDIR="${INDIR}/combined_bins"
FINAL="${INDIR}/flux_all_bins_phi${PHI_MIN}-${PHI_MAX}_el${EL_MIN}-${EL_MAX}_dphi${DPHI}_del${DEL}_rho${RHO}_nE${N_E}_nEvents${NEVENTS}.txt"

mkdir -p "$COMBDIR"

while read -r az el; do
    [ -z "$az" ] && continue
    ./combine_bin.sh "$az" "$el" "$INDIR" "$COMBDIR" \
        || echo "WARNING: bin az=$az el=$el missing or incomplete - skipped" >&2
done < bins.txt

cat "${COMBDIR}"/combined_az*_el*.txt 2>/dev/null | sort -k1,1n -k2,2n > "$FINAL"
echo "Final combined file: $FINAL ($(wc -l < "$FINAL") bins)"