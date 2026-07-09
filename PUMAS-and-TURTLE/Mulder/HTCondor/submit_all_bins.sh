#!/bin/bash
# submit_all_bins.sh <N_JOBS_PER_BIN> <MAX_IDLE>
#
# Builds all_jobs.txt, fills in the Condor submit template, submits it,
# parses the cluster ID straight out of condor_submit's own output, blocks
# (condor_wait) until every job in that cluster finishes, then runs
# combine_all_bins.sh with CLUSTER_ID exported so it looks in the right
# output directory.
set -e

N_JOBS_PER_BIN=${1:?"Usage: $0 N_JOBS_PER_BIN MAX_IDLE"}
MAX_IDLE=${2:?"Usage: $0 N_JOBS_PER_BIN MAX_IDLE"}

if [ ! -f run_config.txt ]; then
    echo "run_config.txt not found - run generate_bins.sh first" >&2
    exit 1
fi
source run_config.txt   # DPHI DEL RHO N_E NEVENTS ...

mkdir -p logs
./generate_job_list.sh "$N_JOBS_PER_BIN"

sed -e "s|__DPHI__|${DPHI}|g" \
    -e "s|__DEL__|${DEL}|g" \
    -e "s|__RHO__|${RHO}|g" \
    -e "s|__N_E__|${N_E}|g" \
    -e "s|__NEVENTS__|${NEVENTS}|g" \
    -e "s|__MAX_IDLE__|${MAX_IDLE}|g" \
    submit.condor > generated_submit.condor

SUBMIT_OUT=$(condor_submit generated_submit.condor)
echo "$SUBMIT_OUT"

# condor_submit prints e.g. "N job(s) submitted to cluster 1234."
CLUSTER_ID=$(echo "$SUBMIT_OUT" | sed -n 's/.* submitted to cluster \([0-9][0-9]*\)\..*/\1/p')
if [ -z "$CLUSTER_ID" ]; then
    echo "ERROR: could not parse cluster ID from condor_submit output" >&2
    exit 1
fi
echo "Cluster ID: $CLUSTER_ID"

condor_wait "logs/cluster_${CLUSTER_ID}.log"

export CLUSTER_ID
./combine_all_bins.sh