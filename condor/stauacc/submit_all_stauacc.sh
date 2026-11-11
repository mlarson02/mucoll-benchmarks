#!/bin/bash
# Stau analysis stage 1 over every reco directory listed at the bottom, one
# condor cluster each, one job per per-job reco file found there. Comment a
# line out to skip it.
#
#   ./submit_all_stauacc.sh              submit
#   DRY_RUN=1 ./submit_all_stauacc.sh    print the condor_submit commands only
#   FORCE=1 ./submit_all_stauacc.sh      submit even into non-empty output dirs
#
# Output: $OUTPUT_ROOT/$SAMPLE/acc_<reco dir>/, read by listing that directory
# in OutsideInTrackingStudies/Stau/edm4hepStauTrackAnalyzer.py's file_names.

cd "$(dirname "$0")" || exit 1
mkdir -p logs lists

SAMPLE=stau_4000GeV_30ns
OUTPUT_ROOT=/ospool/uc-shared/project/futurecolliders/larsonma

# submit <reco dir>
submit() {
    local reco_dir=$1
    local in_dir="${OUTPUT_ROOT}/${SAMPLE}/${reco_dir}"
    local out_dir="${OUTPUT_ROOT}/${SAMPLE}/acc_${reco_dir}"
    local list="lists/${reco_dir}.txt"

    # Indices of the per-job reco files (not hadd'd _reco.edm4hep.root outputs).
    ls "$in_dir" 2>/dev/null | sed -n "s/^${SAMPLE}_reco\([0-9][0-9]*\)\.edm4hep\.root$/\1/p" | sort -n > "$list"
    local n
    n=$(wc -l < "$list")
    echo ">>> ${reco_dir}: ${n} files -> ${out_dir}"
    if [ "$n" -eq 0 ]; then
        echo "    SKIPPED: no per-job reco files in $in_dir"
        return
    fi
    if [ -n "$(ls -A "$out_dir" 2>/dev/null)" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "    SKIPPED: $out_dir already has files in it"
        return
    fi

    # -queue must come last: condor_submit reads the rest of the command line
    # as the queue statement.
    local cmd=(condor_submit
        -batch-name "${SAMPLE}/acc_${reco_dir}"
        -append "SAMPLE = ${SAMPLE}"
        -append "RECO_DIR = ${reco_dir}"
        stauacc.submit
        -queue "PID from ${list}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '   '; printf ' %q' "${cmd[@]}"; echo
    else
        "${cmd[@]}"
    fi
}

submit reco_fullBIB_coned_loose_timing_rawtime_VXDNominal_InOut_TwoWay_No4D
submit reco_fullBIB_coned_loose_timing_rawtime_VXDNominal_InOut_TwoWay_Beta0p6
#submit reco_fullBIB_coned_loose_timing_rawtime_VXDNominal_InOut_TwoWay

# Same without BIB (skipped until the reco has run)
#submit reco_NoBIB_coned_loose_timing_rawtime_VXDNominal_InOut_TwoWay
#submit reco_NoBIB_coned_loose_timing_rawtime_VXDNominal_InOut_TwoWay_No4D
#submit reco_NoBIB_coned_loose_timing_rawtime_VXDNominal_InOut_TwoWay_Beta0p6
