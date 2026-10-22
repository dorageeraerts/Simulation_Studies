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

source muraves_mulder_config.sh   # OUTPUT_HOST, SIF_PATH, PHI_OFFSET, ... defined here

OUTPUT_DIR="${OUTPUT_HOST}/${CLUSTER_ID}"
EL_MAX=$(awk -v a="$EL" -v d="$DEL" 'BEGIN{printf "%.6g", a+d}')

# Filename uses ABSOLUTE az (local + PHI_OFFSET) to match what
# MuravesFluxmeter_HTC.py actually reports inside the file.
AZ_START_ABS=$(awk -v a="$AZ_START" -v o="$PHI_OFFSET" 'BEGIN{printf "%.6g", a+o}')
AZ_END_ABS=$(awk -v a="$AZ_END" -v o="$PHI_OFFSET" 'BEGIN{printf "%.6g", a+o}')
TAG="az${AZ_START_ABS}-${AZ_END_ABS}_el${EL}_job${JOB_ID}"

# Write to node-local scratch first, then copy the finished result to the
# shared filesystem in one shot (less NFS traffic, no partial files if killed).
# $_CONDOR_SCRATCH_DIR is Condor's per-job scratch dir (auto-cleaned).
# Fall back to /tmp for local (non-Condor) runs.
SCRATCH_BASE="${_CONDOR_SCRATCH_DIR:-/tmp/${TAG}_$$}"
LOCAL_OUT="${SCRATCH_BASE}/output"
LOCAL_HOME="${SCRATCH_BASE}/home"   # per-job home (isolates caches)
mkdir -p "$LOCAL_OUT" "$LOCAL_HOME"

PYTHON=/opt/conda/envs/muraves-mulder/bin/python

# Common singularity options. --cleanenv keeps host PATH/PYTHONPATH/etc. out.
SING_OPTS=(
    --cleanenv
    --home "${LOCAL_HOME}:/home/$(id -un)"
    --bind "${LOCAL_OUT}:${OUTPUT_CONTAINER}"
    --bind "${DATA_PATH_HOST}:${DATA_PATH_CONTAINER}"
    --pwd "${DATA_PATH_CONTAINER}"
)

# Preflight: fail immediately (no retries) if the image/env is broken.
echo "Using SIF: ${SIF_PATH}"
singularity exec "${SING_OPTS[@]}" "${SIF_PATH}" "$PYTHON" -c \
    "import sys, pyarrow, mulder; print('python:', sys.executable); print('pyarrow', pyarrow.__version__, pyarrow.__file__)" \
    || { echo "ERROR: pyarrow/mulder import failed in ${SIF_PATH}" >&2; exit 1; }

# Retry loop: for transient failures only (NFS hiccups, lock collisions).
N_TRIES=3
for attempt in $(seq 1 $N_TRIES); do
    if singularity exec "${SING_OPTS[@]}" "${SIF_PATH}" \
        "$PYTHON" MuravesFluxmeter_transmission.py \
        --phi-min "$AZ_START" --phi-max "$AZ_END" \
        --el-min "$EL" --el-max "$EL_MAX" \
        --d-phi "$DPHI" --d-el "$DEL" \
        --rho "$RHO" --n-E "$N_E" --nEvents "$NEVENTS" \
        --tag "$TAG" \
        --input-path "$DATA_PATH_CONTAINER" \
        --output-path "$OUTPUT_CONTAINER"
    then
        break
    fi
    echo "WARNING: simulation failed (attempt $attempt/$N_TRIES) for $TAG" >&2
    if [ "$attempt" -eq "$N_TRIES" ]; then
        echo "ERROR: giving up on $TAG after $N_TRIES attempts" >&2
        exit 1
    fi
    sleep $((RANDOM % 10 + 5))   # jittered backoff, avoid re-colliding
done

# Verify the run actually produced output before declaring success.
PRODUCED=$(find "$LOCAL_OUT" -maxdepth 1 -name "flux_5m_*_${TAG}.txt" -size +0c)
if [ -z "$PRODUCED" ]; then
    echo "ERROR: no non-empty output file found in $LOCAL_OUT for $TAG" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
cp -r "${LOCAL_OUT}/." "$OUTPUT_DIR/"

# Clean up our own /tmp fallback (Condor cleans $_CONDOR_SCRATCH_DIR itself)
if [ -z "${_CONDOR_SCRATCH_DIR:-}" ]; then
    rm -rf "$SCRATCH_BASE"
fi