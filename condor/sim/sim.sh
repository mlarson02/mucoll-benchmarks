#!/bin/bash
# ddsim wrapper for HTCondor. See sim.submit for how it's called.
set -eo pipefail

# ---- args --------------------------------------------------------------
sample=""
gen_file=""
n_events=""
chunks=""
proc_id=""
out_dir=""
pdg_file=""   # optional particle table (.tbl), e.g. for staus

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sample)   sample="$2";   shift 2 ;;
        --genFile)  gen_file="$2"; shift 2 ;;
        --nEvents)  n_events="$2"; shift 2 ;;
        --chunks)   chunks="$2";   shift 2 ;;
        --pId)      proc_id="$2";  shift 2 ;;
        --outDir)   out_dir="$2";  shift 2 ;;
        # "=" form: an empty PDG_FILE macro must not leave a bare --pdgFile.
        --pdgFile=*) pdg_file="${1#*=}"; shift ;;
        *) echo "unknown arg $1"; exit 2 ;;
    esac
done

for v in sample gen_file n_events chunks proc_id out_dir; do
    if [ -z "${!v}" ]; then echo "missing --$v"; exit 2; fi
done

# ---- env ---------------------------------------------------------------
# common.sh is transferred alongside this script.
# shellcheck disable=SC1091
source "$(dirname "$0")/common.sh"

# ---- compute chunk -----------------------------------------------------
events_per_chunk=$(( n_events / chunks ))
events_to_skip=$(( proc_id * events_per_chunk ))
out_name="${sample}_sim${proc_id}.edm4hep.root"

echo "<<< running sim: chunk $proc_id / $chunks, skip=$events_to_skip nEv=$events_per_chunk"
echo "<<< input:  $(basename "$gen_file")"
echo "<<< output: $out_name"

# ---- run ddsim ---------------------------------------------------------
extra=()
if [ -n "$pdg_file" ]; then
    echo "<<< pdg file: $(basename "$pdg_file")"
    extra+=(--physics.pdgfile "$(basename "$pdg_file")")
fi
ddsim \
    --steeringFile mucoll-benchmarks/simulation/steer_baseline.py \
    --inputFiles "$(basename "$gen_file")" \
    --outputFile "$out_name" \
    --numberOfEvents "$events_per_chunk" \
    --skipNEvents "$events_to_skip" \
    "${extra[@]}"

# ---- stage output out over OSDF ---------------------------------------
stage_out "$out_name" "$out_dir"

# ---- housekeeping so condor doesn't drag inputs back ------------------
rm -f "$(basename "$gen_file")"
if [ -n "$pdg_file" ]; then rm -f "$(basename "$pdg_file")"; fi
rm -f "$out_name"
rm -rf mucoll-benchmarks common.sh "pelican-${PELICAN_VERSION}"

echo ">>> sim chunk $proc_id done"
