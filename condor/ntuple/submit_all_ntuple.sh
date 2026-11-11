#!/bin/bash
# Run the track ntupler over every reco directory listed at the bottom, one
# condor cluster each, one job per per-job reco file found there. Comment a
# line out to skip it.
#
#   ./submit_all_ntuple.sh              submit
#   DRY_RUN=1 ./submit_all_ntuple.sh    print the condor_submit commands only
#   FORCE=1 ./submit_all_ntuple.sh      submit even into non-empty output dirs
#
# Output: $OUTPUT_ROOT/$SAMPLE/ntuples_<reco dir without "reco_">/, read by
# OutsideInTrackingStudies/trackPlotter.C.

cd "$(dirname "$0")" || exit 1
mkdir -p logs lists

SAMPLE=muon_pT2_1000GeV_theta88_92
OUTPUT_ROOT=/ospool/uc-shared/project/futurecolliders/larsonma

# submit <reco dir>
submit() {
    local reco_dir=$1
    local in_dir="${OUTPUT_ROOT}/${SAMPLE}/${reco_dir}"
    local out_dir="${OUTPUT_ROOT}/${SAMPLE}/ntuples_${reco_dir#reco_}"
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
        -batch-name "ntuple_${reco_dir#reco_}"
        -append "SAMPLE = ${SAMPLE}"
        -append "RECO_DIR = ${reco_dir}"
        ntuple.submit
        -queue "PID from ${list}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '   '; printf ' %q' "${cmd[@]}"; echo
    else
        "${cmd[@]}"
    fi
}

# full BIB, coned
submit reco_fullBIB_coned_VXD_InOut_TwoWay
submit reco_fullBIB_coned_IT_OutIn
submit reco_fullBIB_coned_OT_OutIn

# full BIB, uncone
#submit reco_fullBIB_uncone_VXD_InOut_TwoWay
#submit reco_fullBIB_uncone_IT_OutIn
#submit reco_fullBIB_uncone_OT_OutIn
#submit reco_fullBIB_uncone_extended_timing_VXD_InOut_TwoWay
#submit reco_fullBIB_uncone_extended_timing_IT_OutIn
#submit reco_fullBIB_uncone_extended_timing_OT_OutIn
#submit reco_fullBIB_uncone_VXDNominal_InOut
#submit reco_fullBIB_uncone_ITOTDisks_OutIn
