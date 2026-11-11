#!/bin/bash
# Stau analysis stage 1 for HTCondor (see stauacc.submit): one reco file ->
# edm4hepStauTrackAnalyzer.process_file()'s result, <sample>_acc<N>.pkl.
set -eo pipefail

sample=""
output_root=""   # /ospool root
reco_dir=""      # e.g. reco_fullBIB_coned_loose_timing_rawtime_VXDNominal_InOut_TwoWay_Beta0p6
proc_id=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sample)     sample="$2";      shift 2 ;;
        --outputRoot) output_root="$2"; shift 2 ;;
        --recoDir)    reco_dir="$2";    shift 2 ;;
        --pId)        proc_id="$2";     shift 2 ;;
        *) echo "unknown arg $1"; exit 2 ;;
    esac
done
for v in sample output_root reco_dir proc_id; do
    if [ -z "${!v}" ]; then echo "missing --$v"; exit 2; fi
done

# shellcheck disable=SC1091
source "$(dirname "$0")/common.sh"

in_file="${sample}_reco${proc_id}.edm4hep.root"
out_file="${sample}_acc${proc_id}.pkl"
out_dir="${output_root}/${sample}/acc_${reco_dir}"
coned=0
if [[ "$reco_dir" == *_coned* ]]; then coned=1; fi
echo "<<< in:  ${output_root}/${sample}/${reco_dir}/${in_file}"
echo "<<< out: ${out_dir}/${out_file} (coned=${coned})"

stage_in "${output_root}/${sample}/${reco_dir}/${in_file}" "$in_file"
python3 stauAccumulator.py --input "$in_file" --output "$out_file" --coned "$coned" --index "$proc_id"
stage_out "$out_file" "$out_dir"

rm -f "$in_file" "$out_file"
rm -rf mucoll-benchmarks common.sh "pelican-${PELICAN_VERSION}" stauAccumulator.py edm4hepStauTrackAnalyzer.py __pycache__
echo ">>> stau acc chunk $proc_id done"
