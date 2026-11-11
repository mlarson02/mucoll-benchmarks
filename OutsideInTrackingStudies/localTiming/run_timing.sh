#!/bin/bash
# Single-event, single-thread reco timing, full BIB uncone (2026-10-31):
# v2 / v3 outside-in (DZ400 SpM0) vs VXDNominal inside-out, with and without two-way.
# Same arguments as condor/reco/reco.sh. Run inside the job container, e.g. detached:
#   nohup bash run_timing.sh > run_timing.out 2>&1 &
# Logs: <config>.log here; summary printed at the end of run_timing.out.

# Start from a clean environment, like a condor job: sourcing /opt/setup_mucoll.sh
# in an already set-up shell doubles its long path variables past the kernel's
# 128 kB per-variable limit, and every command then fails "Argument list too long".
if [ -z "$RUN_TIMING_CLEAN" ]; then
    exec env -i RUN_TIMING_CLEAN=1 HOME="$HOME" USER="$USER" TERM="${TERM:-dumb}" \
        PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin bash "$0"
fi

source /opt/setup_mucoll.sh
MB=/scratch/larsonma/tutorial2024/ACTSGaudi/TemplateWorkspace/mucoll-benchmarks
B=/scratch/larsonma/tutorial2024/ACTSGaudi/TemplateWorkspace/build/packages/k4ActsTracking/k4ActsTracking
pushd $MB >/dev/null; source setup_config.sh . MAIA_v0; popd >/dev/null
export LD_LIBRARY_PATH=$B:$LD_LIBRARY_PATH PYTHONPATH=$B/genConfDir:$PYTHONPATH
IN=/ospool/uc-shared/project/futurecolliders/larsonma/muon_pT0_50GeV_theta10_170/digi_fullBIB_uncone_rawtime/muon_pT0_50GeV_theta10_170_digi0.edm4hep.root
cd "$(dirname "$(realpath "$0")")" || exit 1

common=(-n 1 --inputFiles $IN --TrackingThreads 1 --numThreads 1 --seedZMax 2300 --seedImpactMax 3
        --seedCollisionRegion 6 --seedMinPt 1.5 --filterTracksD0 10 --filterTracksZ0 10
        --no-seedDeduplication --hitTimeInTracking --timeBetaHypothesis 1.0 --doOverlayFull)
run() { n=$1; shift; ( time LD_PRELOAD=$B/libk4ActsTrackingPlugins.so k4run $MUCOLL_CONFIG/$MUCOLL_CONFIG_NAME/reco_steer.py \
        "${common[@]}" --outputFile $n.edm4hep.root --histoFile ${n}_hist.root "$@" ) > $n.log 2>&1 & }

run v2_OutIn_DZ400_SpM0     --seeding ITOTDisks_Minimal_v2 --ckfDirection OutIn --twoWayCKF --seedDeltaZMax 400 --seedMaxPerSpM 0
run v3_OutIn_DZ400_SpM0     --seeding ITOTDisks_Minimal_v3 --ckfDirection OutIn --twoWayCKF --seedDeltaZMax 400 --seedMaxPerSpM 0
run VXDNominal_InOut        --seeding VXDNominal --ckfDirection InOut --no-twoWayCKF
run VXDNominal_InOut_TwoWay --seeding VXDNominal --ckfDirection InOut --twoWayCKF
wait
grep -H -E "^Reconstructor:E[a-z.]* +INFO TimeElapsed|^real" *.log
