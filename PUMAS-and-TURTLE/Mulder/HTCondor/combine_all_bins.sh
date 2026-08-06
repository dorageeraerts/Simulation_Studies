#!/bin/bash
# Run manually once you've confirmed Condor has finished (e.g. via
# `condor_q <CLUSTER_ID>`). Picks up CLUSTER_ID / RUN_TAG / LOGBASE from
# run_state/<CLUSTER_ID>.env (written by submit_all_bins.sh), or from
# run_state/latest.env if no cluster id is given.
#
# 1) combines the per-job-replica files for each bin into one line per bin,
#    in a single pass over all job files (see combine_all_bins.awk)
# 2) sorts all bins into one final file, named after the actual run
#    parameters (phi range, el range, d-phi, d-el, rho, n-E, nEvents)
# 3) if any bins are missing: maps them back to the chunks.txt row (=
#    Condor job) each one belongs to, writes missing_chunks_<CLUSTER_ID>.txt
#    / missing_jobs_<CLUSTER_ID>.txt, and submits those jobs again so their
#    output lands back in THIS run's output dir. Once that resubmission
#    finishes, just run ./combine_all_bins.sh <CLUSTER_ID> again.

set -euo pipefail

source muraves_mulder_config.sh   # OUTPUT_HOST base path (and default LOGBASE, if set there)

RUN_STATE_DIR="run_state"

if [ -n "${1:-}" ]; then
    STATE_FILE="${RUN_STATE_DIR}/${1}.env"                  # ./combine_all_bins.sh 3382819 (explicit wins)
elif [ -f "${RUN_STATE_DIR}/latest.env" ]; then
    STATE_FILE="${RUN_STATE_DIR}/latest.env"                # ./combine_all_bins.sh, no args -> most recent run
elif [ -n "${CLUSTER_ID:-}" ]; then
    STATE_FILE="${RUN_STATE_DIR}/${CLUSTER_ID}.env"         # last-resort fallback if somehow no latest.env
else
    echo "ERROR: no run state available - pass a cluster id explicitly." >&2
    exit 1
fi

if [ ! -f "$STATE_FILE" ]; then
    echo "ERROR: no run-state file found ($STATE_FILE)." >&2
    echo "Pass a cluster id explicitly: ./combine_all_bins.sh <CLUSTER_ID>" >&2
    exit 1
fi

source "$STATE_FILE"   # sets CLUSTER_ID, RUN_TAG, LOGBASE (overrides config default, if any),
                        # and (if the run-state file is new enough) N_JOBS_PER_BIN, MAX_IDLE
echo "Using CLUSTER_ID=${CLUSTER_ID} RUN_TAG=${RUN_TAG} LOGBASE=${LOGBASE} (from $STATE_FILE)" >&2

mv "${LOGBASE}/${RUN_TAG}" "${LOGBASE}/${CLUSTER_ID}" 2>/dev/null || true

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

if [ ! -d "$INDIR" ]; then
    echo "ERROR: $INDIR does not exist - no job output was produced (check Condor logs / mamba errors)" >&2
    exit 1
fi

PHI_MIN_ABS=$(awk -v phi="$PHI_MIN" -v off="$PHI_OFFSET" 'BEGIN {printf "%.0f", phi + off}')
PHI_MAX_ABS=$(awk -v phi="$PHI_MAX" -v off="$PHI_OFFSET" 'BEGIN {printf "%.0f", phi + off}')

FINAL="${INDIR}/flux_all_bins_phi${PHI_MIN_ABS}-${PHI_MAX_ABS}_el${EL_MIN}-${EL_MAX}_dphi${DPHI}_del${DEL}_rho${RHO}_nE${N_E}_nEvents${NEVENTS}_$(date +%d%m%y).txt"

UNSORTED="${INDIR}/.flux_all_bins_unsorted.txt"
EXPECTED_BINS_FILE="${INDIR}/.expected_bins.txt"
MISSING_BINS_FILE="${INDIR}/.missing_bins.txt"
STDERR_LOG="${INDIR}/.combine_stderr.log"
rm -f "$MISSING_BINS_FILE"
trap 'rm -f "$UNSORTED" "$EXPECTED_BINS_FILE" "$MISSING_BINS_FILE" "$STDERR_LOG"' EXIT

# Build the expected-bins list: "abs_az el local_az". local_az (== bins.txt's
# own value, before PHI_OFFSET) is kept so a missing bin can be mapped back
# to the chunks.txt row / job that should be resubmitted.
awk -v off="$PHI_OFFSET" '
    NF == 0 { next }
    { printf "%.6f %s %s\n", $1 + off, $2, $1 }
' bins.txt > "$EXPECTED_BINS_FILE"

# Gather all job-replica files once. Adjust the glob if your job output
# naming ever changes -- must match what run_bin_job.sh actually writes.
shopt -s nullglob
DATA_FILES=( "${INDIR}"/flux_5m_*bin_rho*_discrete_100T_az*_job*.txt )
shopt -u nullglob

if [ ${#DATA_FILES[@]} -eq 0 ]; then
    echo "ERROR: no job output files found in $INDIR" >&2
    exit 1
fi

# Single pass over every file, bucketed by bin, instead of one pass per bin.
awk -v dphi="$DPHI" -v missing_file="$MISSING_BINS_FILE" \
    -f combine_all_bins.awk "$EXPECTED_BINS_FILE" "${DATA_FILES[@]}" \
    > "$UNSORTED" 2> "$STDERR_LOG"

# Surface the per-bin warnings exactly like before, and pull out the count.
grep -v '^MISSING_COUNT=' "$STDERR_LOG" >&2 || true
MISSING=$(grep '^MISSING_COUNT=' "$STDERR_LOG" | tail -1 | cut -d= -f2)
MISSING=${MISSING:-0}

sort -k1,1n -k2,2n "$UNSORTED" > "$FINAL"

EXPECTED_BINS=$(wc -l < bins.txt)
ACTUAL_BINS=$(wc -l < "$FINAL")

echo "Final combined file: $FINAL ($ACTUAL_BINS / $EXPECTED_BINS bins)"

if [ ! -s "$FINAL" ] || [ "$MISSING" -gt 0 ] || [ "$ACTUAL_BINS" -ne "$EXPECTED_BINS" ]; then
    echo "WARNING: run incomplete ($MISSING bin(s) missing) - leaving intermediate files in $INDIR for debugging, NOT cleaning up" >&2

    if [ -s "$MISSING_BINS_FILE" ] && [ -f chunks.txt ]; then
        MISSING_CHUNKS_FILE="missing_chunks_${CLUSTER_ID}.txt"
        MISSING_JOBS_FILE="missing_jobs_${CLUSTER_ID}.txt"

        # Map each missing (local_az, el) bin back to the chunks.txt row
        # (az_start, az_end, el) it belongs to, deduplicated - a chunk is
        # one Condor job covering multiple az bins at one el in a single
        # output file, so if one bin in it is missing the whole job needs
        # to be resubmitted.
        awk '
            FNR == NR { cstart[NR]=$1; cend[NR]=$2; cel[NR]=$3; nchunks=NR; next }
            {
                az=$1; el=$2
                for (i=1; i<=nchunks; i++) {
                    if (az >= cstart[i] - 1e-6 && az < cend[i] - 1e-6 && \
                        (cel[i]-el < 1e-6 && el-cel[i] < 1e-6)) {
                        keyc = cstart[i] SUBSEP cend[i] SUBSEP cel[i]
                        if (!(keyc in seen)) { seen[keyc]=1; print cstart[i], cend[i], cel[i] }
                        break
                    }
                }
            }
        ' chunks.txt "$MISSING_BINS_FILE" > "$MISSING_CHUNKS_FILE"

        N_MISSING_CHUNKS=$(wc -l < "$MISSING_CHUNKS_FILE")
        if [ "$N_MISSING_CHUNKS" -gt 0 ]; then
            echo "Mapped $MISSING missing bin(s) to $N_MISSING_CHUNKS chunk(s)/job(s) -> $MISSING_CHUNKS_FILE" >&2

            ./generate_job_list.sh "${N_JOBS_PER_BIN:-1}" "$MISSING_CHUNKS_FILE" "$MISSING_JOBS_FILE"

            RESUBMIT_TAG="${RUN_TAG}_resubmit_$(date +%Y%m%d_%H%M%S)"
            mkdir -p "${LOGBASE}/${RESUBMIT_TAG}"

            # __CLUSTER_ARG__ is filled with the ORIGINAL numeric cluster
            # ID (not '$(Cluster)') so run_bin_job.sh writes back into
            # this same run's output dir ($INDIR) instead of a fresh one.
            RESUBMIT_CONDOR="generated_resubmit_${CLUSTER_ID}.condor"
            sed -e "s|__DPHI__|${DPHI}|g" \
                -e "s|__DEL__|${DEL}|g" \
                -e "s|__RHO__|${RHO}|g" \
                -e "s|__N_E__|${N_E}|g" \
                -e "s|__NEVENTS__|${NEVENTS}|g" \
                -e "s|__MAX_IDLE__|${MAX_IDLE:-1000}|g" \
                -e "s|__RUN_TAG__|${RESUBMIT_TAG}|g" \
                -e "s|__CLUSTER_ARG__|${CLUSTER_ID}|g" \
                -e "s|__JOBS_FILE__|${MISSING_JOBS_FILE}|g" \
                mulder.sub > "$RESUBMIT_CONDOR"

            echo "Submitting resubmission for missing bins (output will land back in $INDIR)..." >&2
            SUBMIT_OUT=$(condor_submit "$RESUBMIT_CONDOR")
            echo "$SUBMIT_OUT"
            NEW_CLUSTER_ID=$(echo "$SUBMIT_OUT" | sed -n 's/.* submitted to cluster \([0-9][0-9]*\)\..*/\1/p')
            echo "Resubmitted as cluster ${NEW_CLUSTER_ID:-unknown} (writing into original dir $INDIR)." >&2
            echo "Once it finishes (condor_q ${NEW_CLUSTER_ID:-<id>}), just run ./combine_all_bins.sh ${CLUSTER_ID} again." >&2
        else
            echo "WARNING: could not map any missing bin to a chunks.txt row - check chunks.txt/bins.txt are the ones used for this run" >&2
        fi
    else
        echo "No chunks.txt found (or no missing bins recorded) - skipping auto-resubmission." >&2
    fi

    exit 1
fi

echo "All $EXPECTED_BINS bins present - cleaning up intermediate files, keeping only $FINAL"
find "$INDIR" -maxdepth 1 -name 'flux_5m_*_job*.txt' -delete