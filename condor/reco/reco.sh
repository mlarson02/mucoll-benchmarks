#!/bin/bash
# Reconstruction wrapper for HTCondor. See reco.submit.
#
# NOTE: --doTrackerConing is store_true with default False in reco_args.py,
# so passing it for --doConing 1 is what actually selects the "...Coned"
# collections as the tracking input (Tracking/mergers.py).
#
# With an overlay on, reco_steer.py keeps the tracker hits and drops the
# calorimeter hits (--dropTrackerHits / --keepCaloHits override). Do not
# toggle those by editing this script between submissions: like the configs,
# the executable is read when each job starts, so the edit would reach jobs
# already queued. Wire them through reco.submit's Arguments, as --seeding /
# --ckfDirection / --twoWay are.
set -eo pipefail

# ---- args --------------------------------------------------------------
sample=""
output_root=""   # /ospool root; digi in/reco out dirs derived from it below
proc_id=""
# --BIB carries the digi's OverlayFullNumberBackground. Reco itself only
# cares whether overlay was on (use_bib below); the level is needed solely
# to name the digi input / reco output directories.
bib="0"
do_coning="1"
# Optional labels appended to the directory names. --digiTag selects which
# digi variant to read (it must match the --outTag the digi job ran with);
# --recoTag labels this job's own output directory; --recoLabel is appended
# after it, to tell apart reco configurations of the same digi input.
digi_tag=""
reco_tag=""
reco_label=""
# Threads inside the tracking algorithms, and Gaudi event-loop threads (k4run
# --numThreads). reco.submit passes both; request_cpus is their product.
tracking_threads="8"
event_threads="1"   # jobs submitted before 2026-10-19 do not pass it
# CKF configuration. Required, with no defaults: these used to be constants in
# Tracking/CKF_tracking.py, and since the configs directory is transferred when
# a job starts rather than when it is submitted, editing them between
# submissions silently changed jobs still in the queue. As arguments they are
# frozen into the job at condor_submit time.
seeding=""         # name in Tracking/seeding_presets.py
ckf_direction=""   # InOut | OutIn
two_way=""         # 1 | 0
seed_dedup=""      # 1 | 0 (SeedDeduplication)
seed_dedup_min_shared="3"  # 1 | 2 | 3 shared hits to skip a seed; jobs before 2026-10-19 do not pass it
hit_time="1"       # 1 | 0: 4D tracking (seed time cuts + CKF hit time); jobs before 2026-10-25 do not pass it
time_beta="1.0"    # beta assumed for the hit times (TimeBetaHypothesis), 1.0 = no shift; jobs before 2026-10-25 do not pass it
seed_zmax=""       # max |z| of seed space points [mm]
seed_impact_max="" # max seed |d0| [mm]
seed_cr=""         # collision region half-length in z [mm]
seed_min_pt=""     # min seed pT [GeV]
# Jobs before 2026-10-26 do not pass these four; the defaults are what they ran.
seed_dz_max="0"         # max |dz| of a seed doublet [mm], 0 = no cut
seed_dr_max_top="0"     # max middle-top dr of a seed [mm], 0 = SeedFinding_DeltaRMax
seed_max_per_spm="1"    # seeds per middle SP minus one (SeedFilter_MaxSeedsPerSpM)
seed_compat_weight="0"  # SeedFilter_CompatSeedWeight
# Jobs before 2026-11-10 do not pass these two; the defaults are what they ran.
seed_zbins="0"          # 1 = the preset's SEEDING_Z_BINS (Tracking/seeding_presets.py)
seed_rmax="1600"        # max r of seed space points [mm] (SeedFinding_RMax)
filter_d0=""       # max SiTrack |d0| [mm]
filter_z0=""       # max SiTrack |z0| [mm]

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sample)     sample="$2";     shift 2 ;;
        --outputRoot) output_root="$2";shift 2 ;;
        --pId)        proc_id="$2";    shift 2 ;;
        --BIB)        bib="$2";        shift 2 ;;
        --doConing)   do_coning="$2";  shift 2 ;;
        # Both spellings are accepted; reco.submit uses the "=" form because
        # HTCondor's Arguments syntax drops an empty macro token entirely,
        # which would leave a bare --digiTag swallowing the next argument.
        --digiTag=*)  digi_tag="${1#*=}"; shift ;;
        --digiTag)    digi_tag="$2";   shift 2 ;;
        --recoTag=*)  reco_tag="${1#*=}"; shift ;;
        --recoTag)    reco_tag="$2";   shift 2 ;;
        --recoLabel=*) reco_label="${1#*=}"; shift ;;
        --recoLabel)  reco_label="$2"; shift 2 ;;
        --trackingThreads) tracking_threads="$2"; shift 2 ;;
        --eventThreads)    event_threads="$2";    shift 2 ;;
        --seeding=*)  seeding="${1#*=}"; shift ;;
        --seeding)    seeding="$2";    shift 2 ;;
        --ckfDirection=*) ckf_direction="${1#*=}"; shift ;;
        --ckfDirection)   ckf_direction="$2"; shift 2 ;;
        --twoWay=*)   two_way="${1#*=}"; shift ;;
        --twoWay)     two_way="$2";    shift 2 ;;
        --seedDedup=*) seed_dedup="${1#*=}"; shift ;;
        --seedDedup)  seed_dedup="$2"; shift 2 ;;
        --seedDedupMinShared=*) seed_dedup_min_shared="${1#*=}"; shift ;;
        --seedDedupMinShared)   seed_dedup_min_shared="$2"; shift 2 ;;
        --hitTime=*)  hit_time="${1#*=}"; shift ;;
        --hitTime)    hit_time="$2";   shift 2 ;;
        --timeBeta=*) time_beta="${1#*=}"; shift ;;
        --timeBeta)   time_beta="$2";  shift 2 ;;
        --seedZMax=*) seed_zmax="${1#*=}"; shift ;;
        --seedZMax)   seed_zmax="$2";  shift 2 ;;
        --seedImpactMax=*) seed_impact_max="${1#*=}"; shift ;;
        --seedImpactMax)   seed_impact_max="$2"; shift 2 ;;
        --seedCollisionRegion=*) seed_cr="${1#*=}"; shift ;;
        --seedCollisionRegion)   seed_cr="$2"; shift 2 ;;
        --seedMinPt=*) seed_min_pt="${1#*=}"; shift ;;
        --seedMinPt)   seed_min_pt="$2"; shift 2 ;;
        --seedDeltaZMax=*) seed_dz_max="${1#*=}"; shift ;;
        --seedDeltaZMax)   seed_dz_max="$2"; shift 2 ;;
        --seedDeltaRMaxTop=*) seed_dr_max_top="${1#*=}"; shift ;;
        --seedDeltaRMaxTop)   seed_dr_max_top="$2"; shift 2 ;;
        --seedMaxPerSpM=*) seed_max_per_spm="${1#*=}"; shift ;;
        --seedMaxPerSpM)   seed_max_per_spm="$2"; shift 2 ;;
        --seedCompatWeight=*) seed_compat_weight="${1#*=}"; shift ;;
        --seedCompatWeight)   seed_compat_weight="$2"; shift 2 ;;
        --seedZBins=*) seed_zbins="${1#*=}"; shift ;;
        --seedZBins)   seed_zbins="$2"; shift 2 ;;
        --seedRMax=*)  seed_rmax="${1#*=}"; shift ;;
        --seedRMax)    seed_rmax="$2"; shift 2 ;;
        --filterTracksD0=*) filter_d0="${1#*=}"; shift ;;
        --filterTracksD0)   filter_d0="$2"; shift 2 ;;
        --filterTracksZ0=*) filter_z0="${1#*=}"; shift ;;
        --filterTracksZ0)   filter_z0="$2"; shift 2 ;;
        *) echo "unknown arg $1"; exit 2 ;;
    esac
done

for v in sample output_root proc_id; do
    if [ -z "${!v}" ]; then echo "missing --$v"; exit 2; fi
done
# Checked before anything is staged in, so a typo in a preset fails every job
# in seconds instead of after each one has pulled its digi input.
case "$ckf_direction" in
    InOut|OutIn) ;;
    *) echo "--ckfDirection must be InOut or OutIn (got '$ckf_direction')"; exit 2 ;;
esac
case "$two_way" in
    0|1) ;;
    *) echo "--twoWay must be 0 or 1 (got '$two_way')"; exit 2 ;;
esac
case "$seed_dedup" in
    0|1) ;;
    *) echo "--seedDedup must be 0 or 1 (got '$seed_dedup')"; exit 2 ;;
esac
case "$hit_time" in
    0|1) ;;
    *) echo "--hitTime must be 0 or 1 (got '$hit_time')"; exit 2 ;;
esac
if ! awk -v b="$time_beta" 'BEGIN { exit !(b ~ /^[0-9]*\.?[0-9]+$/ && b > 0 && b <= 1) }'; then
    echo "--timeBeta must be a number in (0, 1] (got '$time_beta')"; exit 2
fi
case "$seed_dedup_min_shared" in
    1|2|3) ;;
    *) echo "--seedDedupMinShared must be 1, 2 or 3 (got '$seed_dedup_min_shared')"; exit 2 ;;
esac
if [ -z "$seeding" ]; then echo "missing --seeding"; exit 2; fi
if ! [[ "$seed_zmax" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "--seedZMax must be a positive number in mm (got '$seed_zmax')"; exit 2
fi
if ! [[ "$seed_impact_max" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "--seedImpactMax must be a positive number in mm (got '$seed_impact_max')"; exit 2
fi
if ! [[ "$seed_cr" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "--seedCollisionRegion must be a positive number in mm (got '$seed_cr')"; exit 2
fi
if ! [[ "$seed_min_pt" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "--seedMinPt must be a positive number in GeV (got '$seed_min_pt')"; exit 2
fi
if ! [[ "$seed_dz_max" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "--seedDeltaZMax must be a number >= 0 in mm (got '$seed_dz_max')"; exit 2
fi
if ! [[ "$seed_dr_max_top" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "--seedDeltaRMaxTop must be a number >= 0 in mm (got '$seed_dr_max_top')"; exit 2
fi
if ! [[ "$seed_max_per_spm" =~ ^[0-9]+$ ]]; then
    echo "--seedMaxPerSpM must be an integer >= 0 (got '$seed_max_per_spm')"; exit 2
fi
if ! [[ "$seed_compat_weight" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "--seedCompatWeight must be a number >= 0 (got '$seed_compat_weight')"; exit 2
fi
case "$seed_zbins" in
    0|1) ;;
    *) echo "--seedZBins must be 0 or 1 (got '$seed_zbins')"; exit 2 ;;
esac
if ! [[ "$seed_rmax" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "--seedRMax must be a positive number in mm (got '$seed_rmax')"; exit 2
fi
if ! [[ "$filter_d0" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "--filterTracksD0 must be a positive number in mm (got '$filter_d0')"; exit 2
fi
if ! [[ "$filter_z0" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "--filterTracksZ0 must be a positive number in mm (got '$filter_z0')"; exit 2
fi

# ---- env ---------------------------------------------------------------
# shellcheck disable=SC1091
source "$(dirname "$0")/common.sh"

# The seeding name is checked against the preset table itself rather than a
# copy of its keys here, so the two cannot drift apart.
if ! python3 -c "
import sys
sys.path.insert(0, '${MUCOLL_CONFIG}/${MUCOLL_CONFIG_NAME}')
from Tracking.seeding_presets import SEEDING_PRESETS
sys.exit(0 if '${seeding}' in SEEDING_PRESETS else 1)
"; then
    echo "unknown --seeding '${seeding}'; see Tracking/seeding_presets.py"
    exit 2
fi
echo "<<< ckf: seeding=${seeding} direction=${ckf_direction} twoWay=${two_way} seedDedup=${seed_dedup} seedDedupMinShared=${seed_dedup_min_shared} hitTime=${hit_time} timeBeta=${time_beta} seedZMax=${seed_zmax} seedImpactMax=${seed_impact_max} seedCollisionRegion=${seed_cr} seedMinPt=${seed_min_pt} seedDeltaZMax=${seed_dz_max} seedDeltaRMaxTop=${seed_dr_max_top} seedMaxPerSpM=${seed_max_per_spm} seedCompatWeight=${seed_compat_weight} seedZBins=${seed_zbins} seedRMax=${seed_rmax} filterTracksD0=${filter_d0} filterTracksZ0=${filter_z0}"

# ---- use the k4ActsTracking build shipped with the job ----------------
# reco.submit transfers the local build (K4ACTS_BUILD) in place of the
# k4ActsTracking installed in the image. Put it first on the library and
# Python paths, so Gaudi's plugin lookup and the Configurables both resolve
# to it; k4run additionally preloads the plugin library below.
k4acts_dir="${PWD}/k4acts"
mkdir -p "$k4acts_dir"
for f in libk4ActsTracking.so libk4ActsTrackingPlugins.so k4ActsTrackingPlugins.components genConfDir; do
    if [ ! -e "$f" ]; then echo "missing shipped k4ActsTracking file: $f"; exit 3; fi
    mv "$f" "$k4acts_dir/"
done
export LD_LIBRARY_PATH="${k4acts_dir}:${LD_LIBRARY_PATH}"
export PYTHONPATH="${k4acts_dir}/genConfDir:${PYTHONPATH}"
echo "<<< k4ActsTracking configurables: $(python3 -c 'import inspect; from Configurables import CKFTrackingAlg; print(inspect.getfile(CKFTrackingAlg))')"

# ---- resolve in/out dirs ----------------------------------------------
tag="$(bib_tag "$bib")_$(cone_tag "$do_coning")"
digi_dir="${output_root}/${sample}/digi_${tag}$(name_tag "$digi_tag")"
out_dir="${output_root}/${sample}/reco_${tag}$(name_tag "$reco_tag")$(name_tag "$reco_label")"
use_bib=0
if [ "$bib" != "0" ]; then use_bib=1; fi
echo "<<< digi_dir: $digi_dir"
echo "<<< out_dir:  $out_dir"

# ---- fetch input over OSDF ---------------------------------------------
in_file="${sample}_digi${proc_id}.edm4hep.root"
stage_in "$digi_dir/$in_file" "$in_file"

# ---- assemble k4run args ----------------------------------------------
out_name="${sample}_reco${proc_id}.edm4hep.root"
histo_name="${sample}_reco${proc_id}_hist.root"

k4args=(
    "$MUCOLL_CONFIG/$MUCOLL_CONFIG_NAME/reco_steer.py"
    --inputFiles "$in_file"
    --outputFile "$out_name"
    --histoFile  "$histo_name"
    --TrackingThreads "$tracking_threads"
    --numThreads "$event_threads"
    --seeding "$seeding"
    --ckfDirection "$ckf_direction"
    --seedZMax "$seed_zmax"
    --seedImpactMax "$seed_impact_max"
    --seedCollisionRegion "$seed_cr"
    --seedMinPt "$seed_min_pt"
    --seedDeltaZMax "$seed_dz_max"
    --seedDeltaRMaxTop "$seed_dr_max_top"
    --seedMaxPerSpM "$seed_max_per_spm"
    --seedCompatWeight "$seed_compat_weight"
    --seedRMax "$seed_rmax"
    --filterTracksD0 "$filter_d0"
    --filterTracksZ0 "$filter_z0"
)
# Passed both ways, never left to the default.
if [ "$two_way" = "1" ]; then
    k4args+=(--twoWayCKF)
else
    k4args+=(--no-twoWayCKF)
fi
if [ "$seed_dedup" = "1" ]; then
    k4args+=(--seedDeduplication --seedDedupMinShared "$seed_dedup_min_shared")
else
    k4args+=(--no-seedDeduplication)
fi
if [ "$hit_time" = "1" ]; then
    k4args+=(--hitTimeInTracking)
else
    k4args+=(--no-hitTimeInTracking)
fi
k4args+=(--timeBetaHypothesis "$time_beta")
if [ "$seed_zbins" = "1" ]; then
    k4args+=(--seedZBins)
fi

if [ "$do_coning" = "1" ]; then
    k4args+=(--doTrackerConing)
fi

if [ "$use_bib" = "1" ]; then
    # Tells reco to drop the BIB-heavy hit collections from the output --
    # calorimeter always, tracker only when this is not a coned run.
    k4args+=(--doOverlayFull)
fi

echo "<<< running: k4run ${k4args[*]}"
LD_PRELOAD="${k4acts_dir}/libk4ActsTrackingPlugins.so" k4run "${k4args[@]}"

# ---- stage output out over OSDF ---------------------------------------
stage_out "$out_name" "$out_dir"
if [ -f "$histo_name" ]; then stage_out "$histo_name" "$out_dir"; fi

rm -f "$in_file" "$out_name" "$histo_name"
rm -rf mucoll-benchmarks common.sh "pelican-${PELICAN_VERSION}" "$k4acts_dir"

echo ">>> reco chunk $proc_id done (useBIB=$use_bib coning=$do_coning seeding=$seeding direction=$ckf_direction twoWay=$two_way seedDedup=$seed_dedup seedDedupMinShared=$seed_dedup_min_shared hitTime=$hit_time timeBeta=$time_beta seedZMax=$seed_zmax seedImpactMax=$seed_impact_max seedCollisionRegion=$seed_cr seedMinPt=$seed_min_pt seedDeltaZMax=$seed_dz_max seedDeltaRMaxTop=$seed_dr_max_top seedMaxPerSpM=$seed_max_per_spm seedCompatWeight=$seed_compat_weight seedZBins=$seed_zbins seedRMax=$seed_rmax filterTracksD0=$filter_d0 filterTracksZ0=$filter_z0)"
