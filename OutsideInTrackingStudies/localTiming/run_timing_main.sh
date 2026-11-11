#!/bin/bash
# Single-event, single-thread reco timing with mucoll-benchmarks / MAIAConfig main
# (/scratch/larsonma/MainProfiling), tracker only (Pandora/FastJet commented out in
# its recoAlgList.py), 100% BIB, 4D tracking (the main default), main's VXD seeding.
# Same input as run_timing.sh: the tracker-only full-BIB uncone rawtime digi (nominal
# windows and raw hit times, as main's digi).
# Run inside the job container, e.g. detached:
#   nohup bash run_timing_main.sh > run_timing_main.out 2>&1 &

# Start from a clean environment (see run_timing.sh).
if [ -z "$RUN_TIMING_CLEAN" ]; then
    exec env -i RUN_TIMING_CLEAN=1 HOME="$HOME" USER="$USER" TERM="${TERM:-dumb}" \
        PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin bash "$0"
fi

source /opt/setup_mucoll.sh
MB=/scratch/larsonma/MainProfiling/mucoll-benchmarks
pushd $MB >/dev/null; source setup_config.sh . MAIA_v0; popd >/dev/null
# k4ActsTracking: your build, as in run_timing.sh (has every property main's CKF
# config sets). Set B="" to use the one installed in the image instead.
B=/scratch/larsonma/tutorial2024/ACTSGaudi/TemplateWorkspace/build/packages/k4ActsTracking/k4ActsTracking
PRELOAD=""
if [ -n "$B" ]; then
    export LD_LIBRARY_PATH=$B:$LD_LIBRARY_PATH PYTHONPATH=$B/genConfDir:$PYTHONPATH
    PRELOAD=$B/libk4ActsTrackingPlugins.so
fi
IN=/ospool/uc-shared/project/futurecolliders/larsonma/muon_pT0_50GeV_theta10_170/digi_fullBIB_uncone_rawtime/muon_pT0_50GeV_theta10_170_digi0.edm4hep.root
cd "$(dirname "$(realpath "$0")")" || exit 1

echo "<<< MUCOLL_CONFIG=$MUCOLL_CONFIG  k4ActsTracking build: ${B:-image}"
( time LD_PRELOAD=$PRELOAD k4run $MUCOLL_CONFIG/$MUCOLL_CONFIG_NAME/reco_steer.py -n 1 \
    --inputFiles $IN --outputFile main_VXD.edm4hep.root --histoFile main_VXD_hist.root \
    --TrackingThreads 1 --numThreads 1 --doOverlayFull ) > main_VXD.log 2>&1

grep -H -E "^Reconstructor:E[a-z.]* +INFO TimeElapsed|^real" main_VXD.log
