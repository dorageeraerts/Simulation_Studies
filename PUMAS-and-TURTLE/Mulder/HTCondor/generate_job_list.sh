#!/bin/bash
# generate_job_list.sh [N_JOBS_PER_BIN] [CHUNKS_FILE] [OUTPUT_FILE]
#
# N_JOBS_PER_BIN defaults to muraves_mulder_config.sh - pass an argument
# only to override for a one-off test run.
# CHUNKS_FILE defaults to chunks.txt, OUTPUT_FILE defaults to all_jobs.txt.
# Pass these explicitly to build a job list from a different chunk set,
# e.g. combine_all_bins.sh calls this with its missing_chunks_<ID>.txt to
# build missing_jobs_<ID>.txt for resubmission.
#
# Expands chunks x N_JOBS_PER_BIN replicas -> OUTPUT_FILE, one row per job:
#   az_start az_end el job label
# az_start/az_end/el are LOCAL (pre-PHI_OFFSET) values. label is
# az<abs_start>-<abs_end>_el<el> (absolute az, to match what
# MuravesFluxmeter_HTC.py / run_bin_job.sh actually name output files
# with), used by mulder.sub to name .log/.out/.err files per job.
set -e

source muraves_mulder_config.sh
N_JOBS_PER_BIN=${1:-$N_JOBS_PER_BIN}
CHUNKS_FILE=${2:-chunks.txt}
OUTPUT_FILE=${3:-all_jobs.txt}

if [ ! -f "$CHUNKS_FILE" ]; then
    echo "$CHUNKS_FILE not found - run generate_bins.sh first" >&2
    exit 1
fi

> "$OUTPUT_FILE"
while read -r AZ_START AZ_END EL; do
    AZ_START_ABS=$(awk -v a="$AZ_START" -v o="$PHI_OFFSET" 'BEGIN{printf "%.6g", a+o}')
    AZ_END_ABS=$(awk -v a="$AZ_END" -v o="$PHI_OFFSET" 'BEGIN{printf "%.6g", a+o}')
    LABEL="az${AZ_START_ABS}-${AZ_END_ABS}_el${EL}"
    for ((j=1; j<=N_JOBS_PER_BIN; j++)); do
        echo "$AZ_START $AZ_END $EL $j $LABEL" >> "$OUTPUT_FILE"
    done
done < "$CHUNKS_FILE"

echo "Wrote $(wc -l < "$OUTPUT_FILE") jobs to $OUTPUT_FILE"