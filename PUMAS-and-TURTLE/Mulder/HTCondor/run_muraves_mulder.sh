#!/bin/bash
set -e
chmod +x *.sh

source muraves_mulder_config.sh

./generate_bins.sh "$PHI_MIN" "$PHI_MAX" "$EL_MIN" "$EL_MAX" "$DPHI" "$DEL" "$RHO" "$N_E" "$NEVENTS" "$N_AZ_PER_JOB"

./submit_all_bins.sh "$N_JOBS_PER_BIN" "$MAX_IDLE"