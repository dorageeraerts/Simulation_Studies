#!/bin/bash
# run_bin_job.sh - one MC replica job for one az-chunk at a fixed elevation.
# Called by Condor as:
#   run_bin_job.sh <az_start> <az_end> <el> <job_index> <cluster_id> <dphi> <del> <rho> <n_E> <nEvents>
set -e

AZ_START=$1
AZ_END=$2
EL=$3
JOB_ID=$4
CLUSTER_ID=$5
DPHI=$6
DEL=$7
RHO=$8
N_E=$9
NEVENTS=${10}

source muraves_mulder_config.sh   # OUTPUT_HOST is the static base here

OUTPUT_DIR="${OUTPUT_HOST}/${CLUSTER_ID}"
EL_MAX=$(awk -v a="$EL" -v d="$DEL" 'BEGIN{printf "%.6g", a+d}')
TAG="az${AZ_START}-${AZ_END}_el${EL}_job${JOB_ID}"

# Write to node-local scratch first, then copy the finished result to the
# shared filesystem in one shot. Binding straight to OUTPUT_DIR over
# NFS/AFS for every job's whole runtime would put a lot of small-write
# network traffic on the shared filesystem when many jobs run at once, and
# risks partial files if a job is killed mid-run.
# $_CONDOR_SCRATCH_DIR is Condor's own per-job scratch dir (local disk,
# auto-cleaned on job exit). Fall back to /tmp for local (non-Condor) runs.
SCRATCH_BASE="${_CONDOR_SCRATCH_DIR:-/tmp/${TAG}_$$}"
LOCAL_OUT="${SCRATCH_BASE}/output"
mkdir -p "$LOCAL_OUT"

singularity exec \
    --bind "${LOCAL_OUT}:${OUTPUT_CONTAINER}" \
    --bind "${DATA_PATH_HOST}:${DATA_PATH_CONTAINER}" \
    --pwd "${DATA_PATH_CONTAINER}" \
    "${SIF_PATH}" \
    mamba run -n muraves-mulder python MuravesFluxmeter_HTC.py \
    --phi-min "$AZ_START" --phi-max "$AZ_END" \
    --el-min "$EL" --el-max "$EL_MAX" \
    --d-phi "$DPHI" --d-el "$DEL" \
    --rho "$RHO" --n-E "$N_E" --nEvents "$NEVENTS" \
    --tag "$TAG" \
    --input-path "$DATA_PATH_CONTAINER" \
    --output-path "$OUTPUT_CONTAINER"

mkdir -p "$OUTPUT_DIR"
cp -r "${LOCAL_OUT}/." "$OUTPUT_DIR/"

# Clean up our own /tmp fallback (Condor cleans $_CONDOR_SCRATCH_DIR itself)
if [ -z "${_CONDOR_SCRATCH_DIR:-}" ]; then
    rm -rf "$SCRATCH_BASE"
fi