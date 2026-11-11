#!/bin/bash
# Submit every digi configuration listed at the bottom, one condor cluster
# each. Comment a line out to skip that run.
#
#   ./submit_all_digi.sh              submit
#   DRY_RUN=1 ./submit_all_digi.sh    print the condor_submit commands only
#   FORCE=1 ./submit_all_digi.sh      submit even into non-empty output dirs
#
# Each run's settings go to condor_submit as -append lines, which override
# what digi.submit defines (plain VAR=value would not: it is applied before
# the file is read). They end up in each job's Arguments, fixed at submit
# time, so nothing depends on what any file says when the jobs later start.
#
# A run whose output directory already holds files is skipped, so a rerun
# cannot end up interleaved with chunks from an older one. Clear or move the
# directory first, or use FORCE=1.
#
# Reco reads these outputs: submit the matching reco runs
# (../reco/submit_all_reco.sh) only once a digi run has finished, and leave
# configs/ alone while digi jobs are queued.

cd "$(dirname "$0")" || exit 1

# SAMPLE is set above each block of runs at the bottom, and CONE_STATUS
# (generator statuses the coning cones around) where it is not 1.
CONE_STATUS=1
# Must match OUTPUT_ROOT in digi.submit; only used for the non-empty check.
OUTPUT_ROOT=/ospool/uc-shared/project/futurecolliders/larsonma

# submit <BIB> <coning> <timing> <digi tag, or -> <max files> <memory> <disk>
submit() {
    if [ $# -ne 7 ]; then
        echo "submit: expected 7 columns, got $#: $*" >&2; exit 1
    fi
    local bib=$1 coning=$2 timing=$3 digi_tag=$4 max_files=$5 memory=$6 disk=$7
    if [ "$digi_tag" = "-" ]; then digi_tag=""; fi
    # Raw hit times since 2026-10-16 (digi.sh enforces it too).
    if [[ "$digi_tag" != *rawtime* ]]; then
        echo "submit: digi tag '${digi_tag:--}' must contain rawtime" >&2; exit 1
    fi
    # Same naming as digi.sh (bib_tag / cone_tag / name_tag in common.sh).
    local bib_name
    case "$bib" in
        0)    bib_name=NoBIB   ;;
        17)   bib_name=1pcBIB  ;;
        81)   bib_name=5pcBIB  ;;
        167)  bib_name=nomBIB  ;;
        1667) bib_name=fullBIB ;;
        *)    bib_name="${bib}bkgBIB" ;;
    esac
    local cone=uncone
    if [ "$coning" = "1" ]; then cone=coned; fi
    local name="digi_${bib_name}_${cone}${digi_tag:+_$digi_tag}"
    local out_dir="${OUTPUT_ROOT}/${SAMPLE}/${name}"
    local batch="${SAMPLE}/${name}"

    local files="first ${max_files} sim files"
    if [ "$max_files" = "0" ]; then files="all sim files"; fi
    echo ">>> ${batch}: BIB=${bib}, timing=${timing}, coneStatus=${CONE_STATUS}, ${files}, ${memory} / ${disk} disk per job"
    if [ -n "$(ls -A "$out_dir" 2>/dev/null)" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "    SKIPPED: $out_dir already has files in it"
        return
    fi
    # A run still in the queue may have written nothing yet. And digi only
    # queues the sim files that exist at submit time (list_sim_ids.sh), so
    # submitting while the sim runs would silently leave the rest out.
    local queued
    queued=$(condor_q -af JobBatchName 2>/dev/null)
    if grep -qx "$batch" <<< "$queued"; then
        echo "    SKIPPED: $batch is already in the queue"
        return
    fi
    # TEMP disabled: a condor_rm'd (status X) sim job still counts here.
    # if grep -qx "${SAMPLE}/sim" <<< "$queued"; then
    #     echo "    SKIPPED: ${SAMPLE}/sim still has jobs in the queue"
    #     return
    # fi

    local cmd=(condor_submit
        -batch-name "${batch}"
        -append "SAMPLE = ${SAMPLE}"
        -append "BIB_LEVEL = ${bib}"
        -append "DO_CONING = ${coning}"
        -append "TIMING = ${timing}"
        -append "CONE_STATUS = ${CONE_STATUS}"
        -append "DIGI_TAG = ${digi_tag}"
        -append "MAX_FILES = ${max_files}"
        -append "request_memory = ${memory}"
        -append "request_disk = ${disk}"
        digi.submit)
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '   '; printf ' %q' "${cmd[@]}"; echo
    else
        "${cmd[@]}"
    fi
}

# BIB: OverlayFullNumberBackground, 1667 = full BIB, 0 = none. Timing names
# are defined in configs/.../TrackerDigi/timing_presets.py; the digi tag must
# contain "<timing>_timing" exactly when timing is extended or loose (digi.sh
# enforces it), and "rawtime" always: digi writes raw (not ToF-corrected) hit times
# since 2026-10-16, as the 4D seeding / CKF expects, and older dirs without
# the tag hold corrected times. MAX_FILES = the first N sim files, lowest id first, 0 = all: 25 and
# 1250 select the same sim files the earlier runs used, so the samples stay
# comparable.
# Memory/disk: full-BIB nominal jobs peaked at ~10 GB with ~1.7 GB outputs;
# the old NoBIB digi at 0.9 GB and 7 MB disk. Extended keeps far more
# tracker hits (~10x expected), so its values are a guess: check the first
# jobs' MemoryUsage / DiskUsage and adjust.
#
#       BIB   coning  timing    digi tag         max files  memory   disk

#SAMPLE=muon_pT2_1000GeV_theta88_92

# full BIB, uncone. The existing digi_fullBIB_uncone was all made with nominal
# windows (checked in its logs), so this line is skipped until it is cleared.
#submit  1667  0       nominal   -                25         "16 GB"  "8 GB"
# Submitted 2026-10-06:
#submit  1667  0       extended  extended_timing  25         "32 GB"  "40 GB"

# full BIB, coned. Submitted 2026-10-06:
#submit  1667  1       nominal   -                1250       "16 GB"  "8 GB"
# New sample (no coned extended-timing digi existed before):
#submit  1667  1       extended  extended_timing  1250       "32 GB"  "40 GB"

SAMPLE=muon_pT0_50GeV_theta10_170
# Raw-time versions of every earlier digi of this sample (those are ToF-corrected).

# full BIB, coned: the first 1250 sim chunks (2500 events).
#submit  1667  1       nominal   rawtime                  1250  "16 GB"  "8 GB"
# No BIB, uncone: every sim chunk (5000 = all 10k events).
#submit  0     0       nominal   rawtime                  0     "2 GB"   "2 GB"
# full BIB, uncone: the first 5 sim files, nominal and extended timing.
#submit  1667  0       nominal   rawtime                  5     "16 GB"  "8 GB"
#submit  1667  0       extended  extended_timing_rawtime  5     "32 GB"  "40 GB"

SAMPLE=muon_0_50GeV_theta10_170_d0PCA_300_10kEv_gen
# no BIB displaced muons, vertex at the PCA (pgun --d0AtPCA, t0 = |v|/c). Hits arrive
# late by d0 + sqrt(r^2 - d0^2) - r, past the nominal IT/OT window for d0 >~ 95 mm,
# so extended timing is still needed. The 2026-10-16 digi_NoBIB_coned_extended_timing
# is mixed ToF-corrected / raw: do not use.
#submit  0     0       extended  extended_timing_rawtime  0     "2 GB"   "2 GB"
#submit  0     1       extended  extended_timing_rawtime  1250  "2 GB"   "2 GB"
#submit  167   1       extended  extended_timing_rawtime  1250  "16 GB"  "8 GB"

SAMPLE=stau_4000GeV_30ns
# Stau pairs, m = 4000 GeV, tau = 30 ns (sim/sim.submit: 1000 chunks, 2 events
# each). The staus are not generator status 1, so the coning needs their
# status. beta ~ 0.6: hits arrive late (~1 ns at the IT, ~3 ns at the OT),
# outside the nominal windows, so loose timing.
CONE_STATUS=1,2,22
# 16 GB held ~10% of jobs (RSS p50 15.2 GB, max 18.6; disk 12.6-14 GB), 2026-10-22
#submit  1667  1       loose     loose_timing_rawtime     0     "24 GB"  "16 GB"
# same without BIB, to isolate the BIB effects on the stau tracking (coned too,
# so BIB is the only difference)
submit  0     1       loose     loose_timing_rawtime     0     "4 GB"   "4 GB"
CONE_STATUS=1
