#!/bin/bash
# Digitization wrapper for HTCondor. See digi.submit for the caller.
set -eo pipefail

# ---- args --------------------------------------------------------------
sample=""
output_root=""   # /ospool root; sim in/digi out dirs derived from it below
proc_id=""
bib="0"          # OverlayFullNumberBackground; 0 disables BIB overlay
do_coning="0"    # 1 -> pass --doTrackerConing
n_threads="1"
out_tag=""       # optional label appended to the output directory name
# Tracker time windows (overlay + digitiser), by name in
# TrackerDigi/timing_presets.py. Required, with no default: the windows used
# to be edited into the config files, and since configs/ is transferred when
# a job starts rather than when it is submitted, those edits changed jobs
# already in the queue. As an argument it is frozen at condor_submit time.
timing=""
cone_status="1"  # generator statuses to cone around, comma separated

# Read straight off CVMFS: remote OSG sites do not mount /ospool, but the
# public OSDF namespace is exposed under /cvmfs/public-uc.osgstorage.org.
# Requires HAS_CVMFS_public_uc_osgstorage_org in digi.submit Requirements.
BIB_MUPLUS="/cvmfs/public-uc.osgstorage.org/ospool/uc-shared/public/futurecolliders/gregorypenn/BIB_v9/sim_mp/"
BIB_MUMINUS="/cvmfs/public-uc.osgstorage.org/ospool/uc-shared/public/futurecolliders/gregorypenn/BIB_v9/sim_mm/"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sample)     sample="$2";     shift 2 ;;
        --outputRoot) output_root="$2";shift 2 ;;
        --pId)        proc_id="$2";    shift 2 ;;
        --BIB)        bib="$2";        shift 2 ;;
        --doConing)   do_coning="$2";  shift 2 ;;
        --nThreads)   n_threads="$2";  shift 2 ;;
        # Both spellings are accepted; digi.submit uses the "=" form because
        # HTCondor's Arguments syntax drops an empty macro token entirely,
        # which would leave a bare --outTag swallowing the next argument.
        --outTag=*)   out_tag="${1#*=}"; shift ;;
        --outTag)     out_tag="$2";    shift 2 ;;
        --timing=*)   timing="${1#*=}"; shift ;;
        --timing)     timing="$2";     shift 2 ;;
        --coneStatus=*) cone_status="${1#*=}"; shift ;;
        --bibMuPlus)  BIB_MUPLUS="$2"; shift 2 ;;
        --bibMuMinus) BIB_MUMINUS="$2";shift 2 ;;
        *) echo "unknown arg $1"; exit 2 ;;
    esac
done

for v in sample output_root proc_id timing; do
    if [ -z "${!v}" ]; then echo "missing --$v"; exit 2; fi
done
# The directory name has to say which windows are inside it; checked before
# anything is staged in, so a mismatch fails every job in seconds.
for t in extended loose; do
    if [ "$timing" = "$t" ] && [[ "$out_tag" != *${t}_timing* ]]; then
        echo "--timing $t needs '${t}_timing' in --outTag (got '$out_tag')"; exit 2
    fi
    if [ "$timing" != "$t" ] && [[ "$out_tag" == *${t}_timing* ]]; then
        echo "--outTag '$out_tag' says ${t}_timing but --timing is '$timing'"; exit 2
    fi
done
# Digi writes raw hit times since 2026-10-16; older digi dirs hold ToF-corrected
# ones, so new outputs must be named apart from them.
if [[ "$out_tag" != *rawtime* ]]; then
    echo "--outTag must contain 'rawtime' (got '$out_tag')"; exit 2
fi

# ---- env ---------------------------------------------------------------
# shellcheck disable=SC1091
source "$(dirname "$0")/common.sh"

# Checked against the preset table itself, so the two cannot drift apart.
if ! python3 -c "
import sys
sys.path.insert(0, '${MUCOLL_CONFIG}/${MUCOLL_CONFIG_NAME}')
from TrackerDigi.timing_presets import TIMING_PRESETS
sys.exit(0 if '${timing}' in TIMING_PRESETS else 1)
"; then
    echo "unknown --timing '${timing}'; see TrackerDigi/timing_presets.py"
    exit 2
fi
echo "<<< timing: ${timing}"
echo "<<< cone around generator status: ${cone_status}"

# ---- resolve in/out dirs ----------------------------------------------
sim_dir="${output_root}/${sample}/sim"
out_dir="${output_root}/${sample}/digi_$(bib_tag "$bib")_$(cone_tag "$do_coning")$(name_tag "$out_tag")"
echo "<<< sim_dir: $sim_dir"
echo "<<< out_dir: $out_dir"

# ---- fetch input over OSDF ---------------------------------------------
in_file="${sample}_sim${proc_id}.edm4hep.root"
stage_in "$sim_dir/$in_file" "$in_file"

# ---- assemble k4run args ----------------------------------------------
out_name="${sample}_digi${proc_id}.edm4hep.root"
histo_name="${sample}_digi${proc_id}_hist.root"

k4args=(
    "$MUCOLL_CONFIG/$MUCOLL_CONFIG_NAME/digi_steer.py"
    --inputFiles "$in_file"
    --outputFile "$out_name"
    --histoFile  "$histo_name"
    --numThreads "$n_threads"
    --timing     "$timing"
)

if [ "$do_coning" = "1" ]; then
    k4args+=(--doTrackerConing --coneAroundStatus ${cone_status//,/ })
fi

if [ "$bib" != "0" ]; then
    k4args+=(
        --doOverlayFull
        --OverlayFullNumberBackground "$bib"
        --OverlayFullPathToMuPlus  "$BIB_MUPLUS"
        --OverlayFullPathToMuMinus "$BIB_MUMINUS"
    )
fi

echo "<<< running: k4run ${k4args[*]}"
k4run "${k4args[@]}"

# ---- stage output out over OSDF ---------------------------------------
stage_out "$out_name" "$out_dir"
if [ -f "$histo_name" ]; then stage_out "$histo_name" "$out_dir"; fi

rm -f "$in_file" "$out_name" "$histo_name"
rm -rf mucoll-benchmarks common.sh "pelican-${PELICAN_VERSION}"

echo ">>> digi chunk $proc_id done (BIB=$bib coning=$do_coning timing=$timing)"
