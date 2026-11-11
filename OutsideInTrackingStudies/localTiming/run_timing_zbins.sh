#!/bin/bash
# Single-event, single-thread reco timing, full BIB uncone (2026-11-10), with the
# z-binned seedings: VXDMinimal (main's seeding + VXD disk 7 + IT disk 0, main's
# zMax / z edges / minPt), VXDNominal with SEEDING_Z_BINS, RMax 600, minPt 1 GeV,
# and v2 / v3 outside-in DZ400 SpM0 with SEEDING_Z_BINS (minPt stays 1.5: ACTS
# needs RMax 1600 <= 2 x helix radius). Two runs at a time (ap23 64 GiB cap).
# Same arguments as condor/reco/reco.sh. Run inside the job container, e.g.:
#   nohup bash run_timing_zbins.sh > run_timing_zbins.out 2>&1 &

# Start from a clean environment (see run_timing.sh).
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

common=(-n 1 --inputFiles $IN --TrackingThreads 1 --numThreads 1 --seedImpactMax 3
        --seedCollisionRegion 6 --filterTracksD0 10 --filterTracksZ0 10
        --no-seedDeduplication --hitTimeInTracking --timeBetaHypothesis 1.0 --doOverlayFull)
run() { n=$1; shift; ( time LD_PRELOAD=$B/libk4ActsTrackingPlugins.so k4run $MUCOLL_CONFIG/$MUCOLL_CONFIG_NAME/reco_steer.py \
        "${common[@]}" --outputFile $n.edm4hep.root --histoFile ${n}_hist.root "$@" ) > $n.log 2>&1 & }

vxdmin=(--seeding VXDMinimal --ckfDirection InOut --seedZMax 600 --seedRMax 450 --seedMinPt 1.0)
vxdnom=(--seeding VXDNominal --ckfDirection InOut --seedZMax 2300 --seedRMax 600 --seedMinPt 1.0 --seedZBins)
outin=(--ckfDirection OutIn --twoWayCKF --seedZMax 2300 --seedMinPt 1.5 --seedDeltaZMax 400 --seedMaxPerSpM 0 --seedZBins)

run VXDMinimal_InOut              "${vxdmin[@]}" --no-twoWayCKF
run VXDMinimal_InOut_TwoWay       "${vxdmin[@]}" --twoWayCKF
wait
run VXDNominal_InOut_ZBins        "${vxdnom[@]}" --no-twoWayCKF
run VXDNominal_InOut_TwoWay_ZBins "${vxdnom[@]}" --twoWayCKF
wait
run v2_OutIn_DZ400_SpM0_ZBins     --seeding ITOTDisks_Minimal_v2 "${outin[@]}"
run v3_OutIn_DZ400_SpM0_ZBins     --seeding ITOTDisks_Minimal_v3 "${outin[@]}"
wait
grep -H -E "^Reconstructor:E[a-z.]* +INFO Time(Elapsed| User)|^real" VXDMinimal_*.log VXDNominal_*_ZBins.log *_ZBins.log | sort -u
