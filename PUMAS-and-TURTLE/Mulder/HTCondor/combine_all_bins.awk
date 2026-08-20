#!/usr/bin/awk -f
# combine_all_bins.awk
#
# Single-pass replacement for combine_bin.sh's per-bin loop.
# Reads every job file exactly once and buckets rows by (az,el) bin.
#
# Usage:
#   awk -v dphi="$DPHI" [-v missing_file=<path>] -f combine_all_bins.awk \
#       expected_bins.txt data_file1 data_file2 ...
#
# expected_bins.txt: "abs_az el local_az" per row. local_az is the
# pre-PHI_OFFSET value (same units as chunks.txt) so a missing bin can be
# mapped back to the chunk/job that produced it, for resubmission.
# If missing_file is given, each missing bin's "local_az el" is appended
# to it (in addition to the existing human-readable stderr warning).
#
# --- Fix: n=1 error -----------------------------------------------------
# When only ONE job/replica lands in a bin (N_JOBS_PER_BIN=1, or a
# resubmission only recovers one of several expected replicas), the
# across-replica sample standard error is undefined for n=1, and this
# used to just write 0 - silently claiming a perfect measurement. Now
# each contributing line's own error (columns 4 and 7 - the simulation's
# own reported statistical error for that single run) is tracked, and
# used directly whenever a bin ends up with exactly one contributing job,
# instead of a variance-based SEM.
# -------------------------------------------------------------------------

function idx_of(az) { return sprintf("%.0f", az / dphi) }
function el_key(el) { return sprintf("%.6f", el + 0) }   # normalize numeric formatting

BEGIN {
    if (dphi == "") { print "ERROR: must pass -v dphi=<DPHI>" > "/dev/stderr"; exit 1 }
}

FNR == NR {
    if (NF < 3) next
    e_az = $1; e_el = $2; e_local = $3
    key = idx_of(e_az) SUBSEP el_key(e_el)
    expected[key] = e_az
    expected_el[key] = e_el
    expected_local[key] = e_local
    next
}

{
    az = $1; el = $2
    key = idx_of(az) SUBSEP el_key(el)

    n[key]++
    sum_fr[key]    += $3
    sumsq_fr[key]  += $3*$3
    sum_tr[key]    += $6
    sumsq_tr[key]  += $6*$6
    flux_open[key]  = $5
    nev[key]       += $8
    az_val[key]     = az
    el_val[key]     = el
    # remember this line's own error - if n[key] ends up at 1, this is
    # what gets reported instead of a bogus 0.
    last_fr_err[key] = $4
    last_tr_err[key] = $7
}

END {
    missing = 0
    for (k in expected) {
        if (!(k in n)) {
            printf "WARNING: bin az=%s el=%s missing or incomplete - skipped\n", \
                expected[k], expected_el[k] > "/dev/stderr"
            if (missing_file != "") {
                printf "%s %s\n", expected_local[k], expected_el[k] >> missing_file
            }
            missing++
            continue
        }

        ni = n[k]
        mfr = sum_fr[k]/ni
        mtr = sum_tr[k]/ni

        if (ni > 1) {
            vfr = (sumsq_fr[k] - ni*mfr*mfr)/(ni-1)
            vtr = (sumsq_tr[k] - ni*mtr*mtr)/(ni-1)
            if (vfr < 0) vfr = 0
            if (vtr < 0) vtr = 0
            semfr = sqrt(vfr)/sqrt(ni)
            semtr = sqrt(vtr)/sqrt(ni)
        } else {
            # only one job contributed to this bin: use the simulation's
            # own reported MC error for that single run, not a
            # variance-based SEM (undefined, and wrongly 0, for n=1).
            semfr = last_fr_err[k]
            semtr = last_tr_err[k]
        }

        printf "%s %s %.6e %.6e %.6e %.6e %.6e %d\n", \
            az_val[k], el_val[k], mfr, semfr, flux_open[k], mtr, semtr, nev[k]
    }
    printf "MISSING_COUNT=%d\n", missing > "/dev/stderr"
}