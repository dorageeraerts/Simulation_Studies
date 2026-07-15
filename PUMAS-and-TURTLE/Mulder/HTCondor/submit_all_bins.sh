#!/bin/bash
# submit_all_bins.sh [N_JOBS_PER_BIN] [MAX_IDLE]
#
# Both default to muraves_mulder_config.sh - pass arguments only to override
# for a one-off run.
#
# Builds all_jobs.txt, fills in the Condor submit template, submits it,
# parses the cluster ID straight out of condor_submit's own output, blocks
# (condor_wait) until every job in that cluster finishes, then runs
# combine_all_bins.sh with CLUSTER_ID exported so it looks in the right
# output directory.
set -e

source muraves_mulder_config.sh
N_JOBS_PER_BIN=${1:-$N_JOBS_PER_BIN}
MAX_IDLE=${2:-$MAX_IDLE}

./generate_bins.sh

if [ ! -f run_config.txt ]; then
    echo "run_config.txt not found - run generate_bins.sh first" >&2
    exit 1
fi
source run_config.txt   # DPHI DEL RHO N_E NEVENTS ...

#mkdir -p logs
./generate_job_list.sh "$N_JOBS_PER_BIN"

LOGBASE="/user/dgeeraer/MURAVES/output_mulder/output_htcondor"
#LOCAL_LOGBASE="/tmp/condor_logs_dgeeraer"
RUN_TAG="staging_$$_$(date +%s)"

mkdir -p "${LOGBASE}/${RUN_TAG}"
#mkdir -p "${LOCAL_LOGBASE}/${RUN_TAG}" #write log on local disk so condor_wait "sees" it properly

sed -e "s|__DPHI__|${DPHI}|g" \
    -e "s|__DEL__|${DEL}|g" \
    -e "s|__RHO__|${RHO}|g" \
    -e "s|__N_E__|${N_E}|g" \
    -e "s|__NEVENTS__|${NEVENTS}|g" \
    -e "s|__MAX_IDLE__|${MAX_IDLE}|g" \
    -e "s|__RUN_TAG__|${RUN_TAG}|g" \
    mulder.sub > generated_submit.condor

SUBMIT_OUT=$(condor_submit generated_submit.condor)
echo "$SUBMIT_OUT"

# condor_submit prints e.g. "N job(s) submitted to cluster 1234."
CLUSTER_ID=$(echo "$SUBMIT_OUT" | sed -n 's/.* submitted to cluster \([0-9][0-9]*\)\..*/\1/p')
if [ -z "$CLUSTER_ID" ]; then
    echo "ERROR: could not parse cluster ID from condor_submit output" >&2
    exit 1
fi
echo "Cluster ID: $CLUSTER_ID"

#condor_wait "${LOGBASE}/${RUN_TAG}/${CLUSTER_ID}.log"
#condor_wait "${LOCAL_LOGBASE}/${RUN_TAG}/${CLUSTER_ID}.log"

#LOGFILE="${LOCAL_LOGBASE}/${RUN_TAG}/${CLUSTER_ID}.log"

# Wait for the schedd to actually materialize the log file before
# handing off to condor_wait - with large clusters (thousands of jobs),
# there can be a delay between condor_submit returning and the log
# file appearing on disk.
#echo "Waiting for log file to appear: ${LOGFILE}"
#for i in $(seq 1 60); do
    #[ -f "$LOGFILE" ] && break
    #sleep 1
#done

#if [ ! -f "$LOGFILE" ]; then
    #echo "ERROR: log file never appeared after 60s: ${LOGFILE}" >&2
    #exit 1
#fi

#condor_wait "$LOGFILE"

# Now that everything has genuinely finished, bring the local log(s) into
# the same (Ceph) directory as the .out/.err files, then rename the whole
# thing from RUN_TAG to the real CLUSTER_ID.
#mv "${LOCAL_LOGBASE}/${RUN_TAG}"/* "${LOGBASE}/${RUN_TAG}/" 2>/dev/null || true
#rmdir "${LOCAL_LOGBASE}/${RUN_TAG}" 2>/dev/null || true

# rename log directory to cluster ID

# Poll condor_q instead of tailing (condor_wait) a log file - works regardless of
# whether the submit host and schedd host are the same machine, and
# avoids any local-disk vs shared-filesystem visibility issues.
#echo "Waiting for cluster ${CLUSTER_ID} to finish..."
#while true; do
    #N_LEFT=$(condor_q "$CLUSTER_ID" -totals 2>/dev/null | grep "Total for query" | awk '{print $4}')
    #[ -z "$N_LEFT" ] && N_LEFT=0
    #if [ "$N_LEFT" -eq 0 ]; then
        #break
    #fi
    #sleep 60
#done
#echo "All jobs done."

#mv "${LOGBASE}/${RUN_TAG}" "${LOGBASE}/${CLUSTER_ID}"

export CLUSTER_ID
#./combine_all_bins.sh