#!/bin/bash
# Track ntupler wrapper for HTCondor. See ntuple.submit.
set -eo pipefail

sample=""
output_root=""   # /ospool root
reco_dir=""      # e.g. reco_fullBIB_coned_VXD_InOut_TwoWay
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
out_file="${sample}_ntuple${proc_id}.root"
out_dir="${output_root}/${sample}/ntuples_${reco_dir#reco_}"
coned=0
if [[ "$reco_dir" == *_coned* ]]; then coned=1; fi
echo "<<< in:  ${output_root}/${sample}/${reco_dir}/${in_file}"
echo "<<< out: ${out_dir}/${out_file} (coned=${coned})"

stage_in "${output_root}/${sample}/${reco_dir}/${in_file}" "$in_file"
python3 trackNtupler.py --input "$in_file" --output "$out_file" --coned "$coned"
stage_out "$out_file" "$out_dir"

rm -f "$in_file" "$out_file"
rm -rf mucoll-benchmarks common.sh "pelican-${PELICAN_VERSION}" trackNtupler.py
echo ">>> ntuple chunk $proc_id done"
