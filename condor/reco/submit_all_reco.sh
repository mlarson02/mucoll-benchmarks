#!/bin/bash
# Submit every reco configuration listed at the bottom, one condor cluster
# each. Comment a line out to skip that run.
#
#   ./submit_all_reco.sh              submit
#   DRY_RUN=1 ./submit_all_reco.sh    print the condor_submit commands only
#   FORCE=1 ./submit_all_reco.sh      submit even into non-empty output dirs
#   FILL_MISSING=1 ./submit_all_reco.sh   into a non-empty output dir, submit
#                                     only the chunks with digi input but no
#                                     reco file (reruns failed jobs)
#
# Each run's settings go to condor_submit as -append lines, which override
# what reco.submit defines. (Plain VAR=value on the command line would not:
# it is applied before the file is read, so e.g. reco.submit's empty DIGI_TAG
# would silently win.) They end up in each job's Arguments, fixed at submit
# time, so nothing depends on what any file says when the jobs later start.
#
# A run whose output directory already holds files is skipped, so a rerun
# cannot end up interleaved with chunks from an older one. Clear or move the
# directory first, or use FORCE=1. FILL_MISSING=1 instead submits only the
# missing chunks, with the line's settings; truncated files are not detected,
# so move those out first.

cd "$(dirname "$0")" || exit 1

# SAMPLE is set above each block of runs at the bottom.
BIB_LEVEL=1667   # full BIB, unless a block below resets it
# Must match OUTPUT_ROOT in reco.submit; only used for the non-empty check.
OUTPUT_ROOT=/ospool/uc-shared/project/futurecolliders/larsonma
# Memory per job, by coning and digi timing; overrides request_memory in
# reco.submit. Measured at 8 tracking threads (2026-10-19): coned mean ~4 GB,
# max 11.6 GB; uncone 19-24 GB (ITOTDisks-type killed at ~27 GB on 2026-10-13).
# Now 1 tracking thread x 2 event threads (reco.submit): coned peaked at 14-19 GB
# (59 jobs held at 12 GB, 2026-10-19). The extended-timing
# values are a guess for ~10x the tracker hits: check MemoryUsage and adjust.
MEMORY_CONED="24 GB"
MEMORY_UNCONE="32 GB"
MEMORY_CONED_EXTENDED="32 GB"
MEMORY_UNCONE_EXTENDED="64 GB"
# BIB_LEVEL=0, any coning/timing (guess; NoBIB digi peaked < 2 GB).
MEMORY_NOBIB="6 GB"

# submit key=value ..., every key required, any order (see the key list below).
submit() {
    local coning="" digi_tag="" chunks="" seeding="" direction="" two_way="" seed_dedup="" zmax="" d0max="" cr="" minpt="" trk_d0="" trk_z0="" reco_tag=""
    local dedup_shared=3   # optional key
    local hit_time=1       # optional key
    local time_beta=1.0    # optional key
    local dz_max=0         # optional key
    local dr_max_top=0     # optional key
    local max_spm=1        # optional key
    local compat_weight=0  # optional key
    local zbins=0          # optional key
    local rmax=1600        # optional key
    local arg
    for arg in "$@"; do
        case "$arg" in
            coning=*)  coning="${arg#*=}" ;;
            digi=*)    digi_tag="${arg#*=}" ;;
            chunks=*)  chunks="${arg#*=}" ;;
            seeding=*) seeding="${arg#*=}" ;;
            dir=*)     direction="${arg#*=}" ;;
            twoway=*)  two_way="${arg#*=}" ;;
            dedup=*)   seed_dedup="${arg#*=}" ;;
            dedupshared=*) dedup_shared="${arg#*=}" ;;
            hittime=*) hit_time="${arg#*=}" ;;
            beta=*)    time_beta="${arg#*=}" ;;
            zmax=*)    zmax="${arg#*=}" ;;
            d0max=*)   d0max="${arg#*=}" ;;
            cr=*)      cr="${arg#*=}" ;;
            minpt=*)   minpt="${arg#*=}" ;;
            dzmax=*)   dz_max="${arg#*=}" ;;
            drmaxtop=*) dr_max_top="${arg#*=}" ;;
            maxspm=*)  max_spm="${arg#*=}" ;;
            compat=*)  compat_weight="${arg#*=}" ;;
            zbins=*)   zbins="${arg#*=}" ;;
            rmax=*)    rmax="${arg#*=}" ;;
            trkd0=*)   trk_d0="${arg#*=}" ;;
            trkz0=*)   trk_z0="${arg#*=}" ;;
            tag=*)     reco_tag="${arg#*=}" ;;
            *) echo "submit: unknown argument '$arg'" >&2; exit 1 ;;
        esac
    done
    for arg in coning digi_tag chunks seeding direction two_way seed_dedup zmax d0max cr minpt trk_d0 trk_z0 reco_tag; do
        if [ -z "${!arg}" ]; then echo "submit: missing value for $arg in: $*" >&2; exit 1; fi
    done
    if [ "$digi_tag" = "-" ]; then digi_tag=""; fi
    # A SeedDedup reco tag must run with seed deduplication on, and vice versa.
    local r_dedup=0
    [[ "$reco_tag" == *SeedDedup* ]] && r_dedup=1
    if [ "$r_dedup" != "$seed_dedup" ]; then
        echo "submit: reco tag '$reco_tag' and seed dedup '$seed_dedup' disagree (tag must contain SeedDedup iff dedup is 1)" >&2; exit 1
    fi
    # dedupshared other than 3 needs dedup=1, and the tag must say Shared<N>.
    case "$dedup_shared" in
        3) if [[ "$reco_tag" == *Shared* ]]; then
               echo "submit: reco tag '$reco_tag' contains Shared but dedupshared is 3" >&2; exit 1
           fi ;;
        1|2) if [ "$seed_dedup" != "1" ] || [[ "$reco_tag" != *"Shared${dedup_shared}"* ]]; then
               echo "submit: dedupshared=$dedup_shared needs dedup=1 and 'Shared${dedup_shared}' in the reco tag" >&2; exit 1
           fi ;;
        *) echo "submit: dedupshared must be 1, 2 or 3 (got '$dedup_shared')" >&2; exit 1 ;;
    esac
    # hittime=0 (3D tracking) iff the tag says No4D.
    case "$hit_time" in
        0|1) ;;
        *) echo "submit: hittime must be 0 or 1 (got '$hit_time')" >&2; exit 1 ;;
    esac
    local r_no4d=0
    [[ "$reco_tag" == *No4D* ]] && r_no4d=1
    if [ "$r_no4d" = "$hit_time" ]; then
        echo "submit: reco tag '$reco_tag' and hittime=$hit_time disagree (tag must contain No4D iff hittime is 0)" >&2; exit 1
    fi
    # beta other than 1 (4D tracking for a slow particle) needs hittime=1, and the
    # tag must say Beta<beta> with p for the point, e.g. Beta0p6.
    if ! awk -v b="$time_beta" 'BEGIN { exit !(b ~ /^[0-9]*\.?[0-9]+$/ && b > 0 && b <= 1) }'; then
        echo "submit: beta must be a number in (0, 1] (got '$time_beta')" >&2; exit 1
    fi
    if awk -v b="$time_beta" 'BEGIN { exit !(b == 1) }'; then
        if [[ "$reco_tag" == *Beta* ]]; then
            echo "submit: reco tag '$reco_tag' contains Beta but beta is 1" >&2; exit 1
        fi
    else
        local beta_tag
        beta_tag="Beta$(awk -v b="$time_beta" 'BEGIN { printf "%g", b }' | tr . p)"
        if [ "$hit_time" != "1" ] || [[ "$reco_tag" != *"$beta_tag"* ]]; then
            echo "submit: beta=$time_beta needs hittime=1 and '$beta_tag' in the reco tag" >&2; exit 1
        fi
    fi
    # zbins=1 (the preset's SEEDING_Z_BINS) iff the tag says ZBins.
    case "$zbins" in
        0|1) ;;
        *) echo "submit: zbins must be 0 or 1 (got '$zbins')" >&2; exit 1 ;;
    esac
    local r_zbins=0
    [[ "$reco_tag" == *ZBins* ]] && r_zbins=1
    if [ "$r_zbins" != "$zbins" ]; then
        echo "submit: reco tag '$reco_tag' and zbins=$zbins disagree (tag must contain ZBins iff zbins is 1)" >&2; exit 1
    fi
    # dzmax / drmaxtop / maxspm / compat / rmax other than the default must be named
    # in the tag as DZ<v> / DRTop<v> / SpM<v> / Compat<v> / RMax<v> (point as p), and not otherwise.
    local spec key def val mark
    for spec in "dzmax 0 $dz_max DZ" "drmaxtop 0 $dr_max_top DRTop" "maxspm 1 $max_spm SpM" "compat 0 $compat_weight Compat" "rmax 1600 $rmax RMax"; do
        read -r key def val mark <<< "$spec"
        if ! [[ "$val" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
            echo "submit: $key must be a number >= 0 (got '$val')" >&2; exit 1
        fi
        if awk -v a="$val" -v b="$def" 'BEGIN { exit !(a == b) }'; then
            if [[ "$reco_tag" =~ ${mark}[0-9] ]]; then
                echo "submit: reco tag '$reco_tag' contains ${mark}<N> but $key is the default $def" >&2; exit 1
            fi
        elif [[ "$reco_tag" != *"${mark}${val//./p}"* ]]; then
            echo "submit: $key=$val needs '${mark}${val//./p}' in the reco tag" >&2; exit 1
        fi
    done
    # An extended/loose_timing reco tag must read the same-timing digi, and vice versa.
    local t r_t d_t
    for t in extended loose; do
        r_t=0; d_t=0
        [[ "$reco_tag" == *${t}_timing* ]] && r_t=1
        [[ "$digi_tag" == *${t}_timing* ]] && d_t=1
        if [ "$r_t" != "$d_t" ]; then
            echo "submit: reco tag '$reco_tag' and digi tag '${digi_tag:--}' disagree on ${t}_timing" >&2; exit 1
        fi
    done
    # The 4D seeding / CKF expects raw hit times (rawtime digi, since 2026-10-16);
    # the reco tag says so too, keeping new outputs apart from old ones.
    if [[ "$digi_tag" != *rawtime* || "$reco_tag" != *rawtime* ]]; then
        echo "submit: digi tag '${digi_tag:--}' and reco tag '$reco_tag' must both contain rawtime" >&2; exit 1
    fi
    # The reco tag (= output directory) must name the seeding and direction used.
    if [[ "_$reco_tag" != *"_${seeding}_${direction}"* ]]; then
        echo "submit: reco tag '$reco_tag' does not contain '${seeding}_${direction}'" >&2; exit 1
    fi
    local cone=uncone memory=$MEMORY_UNCONE
    if [ "$coning" = "1" ]; then cone=coned; memory=$MEMORY_CONED; fi
    # Loose uses the extended values too (a guess).
    if [[ "$digi_tag" == *extended_timing* || "$digi_tag" == *loose_timing* ]]; then
        memory=$MEMORY_UNCONE_EXTENDED
        if [ "$coning" = "1" ]; then memory=$MEMORY_CONED_EXTENDED; fi
    fi
    if [ "$BIB_LEVEL" = "0" ]; then memory=$MEMORY_NOBIB; fi
    # Same naming as reco.sh and digi.sh (bib_tag / cone_tag in common.sh);
    # nomBIB = 167 = 10% of full BIB (1667).
    local bib_name
    case "$BIB_LEVEL" in
        0)    bib_name=NoBIB   ;;
        17)   bib_name=1pcBIB  ;;
        81)   bib_name=5pcBIB  ;;
        167)  bib_name=nomBIB  ;;
        1667) bib_name=fullBIB ;;
        *)    bib_name="${BIB_LEVEL}bkgBIB" ;;
    esac
    local digi_name="digi_${bib_name}_${cone}${digi_tag:+_$digi_tag}"
    local out_dir="${OUTPUT_ROOT}/${SAMPLE}/reco_${bib_name}_${cone}_${reco_tag}"
    local batch="${SAMPLE}/reco_${bib_name}_${cone}_${reco_tag}"
    local bib_pct
    bib_pct=$(awk -v b="$BIB_LEVEL" 'BEGIN { printf "%.0f", 100 * b / 1667 }')

    echo ">>> ${batch}: BIB=${BIB_LEVEL} (${bib_pct}% of full) seeding=${seeding} ${direction} twoWay=${two_way} seedDedup=${seed_dedup} dedupShared=${dedup_shared} hitTime=${hit_time} beta=${time_beta} zMax=${zmax} d0max=${d0max} CR=${cr} minPt=${minpt} dzMax=${dz_max} drMaxTop=${dr_max_top} maxSpM=${max_spm} compat=${compat_weight} zBins=${zbins} rMax=${rmax} trkD0=${trk_d0} trkZ0=${trk_z0}, ${chunks} jobs x ${memory}"
    local fill=0
    if [ -n "$(ls -A "$out_dir" 2>/dev/null)" ] && [ "${FORCE:-0}" != "1" ]; then
        if [ "${FILL_MISSING:-0}" != "1" ]; then
            echo "    SKIPPED: $out_dir already has files in it"
            return
        fi
        fill=1
    fi
    # The same run still in the queue has written nothing yet, so the check
    # above cannot see it. And reco on a digi run that has not finished would
    # silently use whatever subset of files exists so far (submit_all_digi.sh
    # names each digi cluster <sample>/<output directory>).
    local queued
    queued=$(condor_q -af JobBatchName 2>/dev/null)
    if grep -qx "$batch" <<< "$queued"; then
        echo "    SKIPPED: $batch is already in the queue"
        return
    fi
    if grep -qx "${SAMPLE}/${digi_name}" <<< "$queued"; then
        echo "    SKIPPED: ${SAMPLE}/${digi_name} still has jobs in the queue"
        return
    fi
    if [ -z "$(ls -A "${OUTPUT_ROOT}/${SAMPLE}/${digi_name}" 2>/dev/null)" ]; then
        echo "    SKIPPED: no digi input in ${SAMPLE}/${digi_name}"
        return
    fi

    # Chunk indices to queue (reco.submit: Queue CHUNK from CHUNK_LIST).
    mkdir -p chunk_lists
    local chunk_list="chunk_lists/${batch//\//__}.txt"
    if [ "$fill" = "1" ]; then
        local i
        for ((i = 0; i < chunks; i++)); do
            [ -e "${OUTPUT_ROOT}/${SAMPLE}/${digi_name}/${SAMPLE}_digi${i}.edm4hep.root" ] || continue
            [ -e "${out_dir}/${SAMPLE}_reco${i}.edm4hep.root" ] || echo "$i"
        done > "$chunk_list"
        if [ ! -s "$chunk_list" ]; then
            echo "    SKIPPED: no missing chunks in $out_dir"
            return
        fi
        echo "    FILL_MISSING: $(wc -l < "$chunk_list") of ${chunks} chunks ($chunk_list)"
    else
        seq 0 $((chunks - 1)) > "$chunk_list"
    fi

    local cmd=(condor_submit
        -batch-name "${batch}"
        -append "SAMPLE = ${SAMPLE}"
        -append "CHUNK_LIST = ${chunk_list}"
        -append "BIB_LEVEL = ${BIB_LEVEL}"
        -append "DO_CONING = ${coning}"
        -append "DIGI_TAG = ${digi_tag}"
        -append "RECO_TAG = ${reco_tag}"
        -append "SEEDING = ${seeding}"
        -append "CKF_DIRECTION = ${direction}"
        -append "TWO_WAY = ${two_way}"
        -append "SEED_DEDUP = ${seed_dedup}"
        -append "SEED_DEDUP_MIN_SHARED = ${dedup_shared}"
        -append "HIT_TIME = ${hit_time}"
        -append "TIME_BETA = ${time_beta}"
        -append "SEED_ZMAX = ${zmax}"
        -append "SEED_IMPACT_MAX = ${d0max}"
        -append "SEED_CR = ${cr}"
        -append "SEED_MIN_PT = ${minpt}"
        -append "SEED_DZ_MAX = ${dz_max}"
        -append "SEED_DR_MAX_TOP = ${dr_max_top}"
        -append "SEED_MAX_PER_SPM = ${max_spm}"
        -append "SEED_COMPAT_WEIGHT = ${compat_weight}"
        -append "SEED_ZBINS = ${zbins}"
        -append "SEED_R_MAX = ${rmax}"
        -append "FILTER_D0 = ${trk_d0}"
        -append "FILTER_Z0 = ${trk_z0}"
        -append "request_memory = ${memory}"
        reco.submit)
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '   '; printf ' %q' "${cmd[@]}"; echo
    else
        "${cmd[@]}"
    fi
}

# Keys (all required, any order):
#   coning   1 = coned digi input, 0 = uncone
#   digi     digi tag (directory suffix); must contain rawtime (raw hit times
#            since 2026-10-16), and extended/loose_timing iff tag does
#   chunks   highest digi index + 1; jobs for missing indices fail at stage-in
#   seeding  name in configs/.../Tracking/seeding_presets.py
#   dir      InOut | OutIn (CKF first-pass direction)
#   twoway   1 = two-way CKF
#   dedup    1 = SeedDeduplication (skip seeds whose hits are all on an earlier
#            track); tag must then contain SeedDedup. Added 2026-10-17, so 0 is
#            what every earlier run did.
#   dedupshared  (optional, default 3) hits a seed must share with an earlier
#            track to be skipped; 1 or 2 needs dedup=1 and Shared<N> in the tag.
#            Added 2026-10-19.
#   hittime  (optional, default 1) 4D tracking: seed time cuts and hit time in
#            the CKF, both assuming beta = 1. 0 turns both off (3D tracking);
#            the tag must then contain No4D. Added 2026-10-25.
#   beta     (optional, default 1.0) beta the 4D tracking assumes: hit times are
#            shifted by -|x| (1/beta - 1) / c (TimeBetaHypothesis; needs a
#            k4ActsTracking build with it). Not 1 needs hittime=1 and
#            Beta<beta> in the tag, point as p, e.g. Beta0p6. Added 2026-10-25.
#   zmax    max |z| of seed space points [mm]: 600 for barrel/VXD seeding,
#            2300 for disk seeding (last disks at |z| = 2190)
#   d0max    seed max |d0| [mm] (SeedFinding_ImpactMax); default 3
#   cr       collision-region half-length in z [mm]; default 6. Before
#            2026-10-12 d0max / cr were fixed in CKF_tracking.py.
#   minpt    min seed pT [GeV]
#   dzmax    (optional, default 0 = off) max |dz| of a seed doublet [mm]
#            (SeedFinding_DeltaZMax; needs the k4ActsTracking build with it).
#            Not 0 needs DZ<v> in the tag. Added 2026-10-26.
#   drmaxtop (optional, default 0 = DeltaRMax) max middle-top dr of a seed [mm]
#            (SeedFinding_DeltaRMaxTop), e.g. so a 4th hit beyond the next layer
#            can count as compatible. Not 0 needs DRTop<v> in the tag. Added 2026-10-26.
#   maxspm   (optional, default 1) seeds kept per middle SP minus one
#            (SeedFilter_MaxSeedsPerSpM). Not 1 needs SpM<v> in the tag. Added 2026-10-26.
#   compat   (optional, default 0) SeedFilter_CompatSeedWeight, added to a seed's
#            weight per compatible top SP (a 4th hit). Not 0 needs Compat<v> in
#            the tag. Added 2026-10-26.
#   zbins    (optional, default 0) 1 = the preset's SEEDING_Z_BINS z edges
#            (Tracking/seeding_presets.py; need zmax >= 2300); tag must then
#            contain ZBins. Added 2026-11-10.
#   rmax     (optional, default 1600) max r of seed space points [mm]
#            (SeedFinding_RMax; ACTS needs rmax <= 2 x helix radius at minpt, e.g.
#            1333 at 1 GeV). Not 1600 needs RMax<v> in the tag. Added 2026-11-10.
#   trkd0    SiTrack max |d0| [mm]
#   trkz0    SiTrack max |z0| [mm]
#   tag      reco tag (output directory suffix); must contain rawtime and
#            <seeding>_<dir>

SAMPLE=muon_pT2_1000GeV_theta88_92

# Digi inputs re-made with the --timing presets (2026-10-06): the coned and
# extended-timing sections below need resubmitting once that digi finishes
# (the old outputs used mixed or overlay-capped windows); clear their reco
# directories first. digi_fullBIB_uncone was already nominal, so the uncone
# sections do not depend on the rerun.

# full BIB, uncone (digi_fullBIB_uncone: 21 files, 3 missing)
#submit coning=0 digi=- chunks=24 seeding=VXD dir=InOut twoway=1 dedup=0 zmax=600 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=VXD_InOut_TwoWay
#submit coning=0 digi=- chunks=24 seeding=IT dir=OutIn twoway=1 dedup=0 zmax=600 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=IT_OutIn
#submit coning=0 digi=- chunks=24 seeding=OT dir=OutIn twoway=1 dedup=0 zmax=600 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=OT_OutIn

# full BIB, uncone, extended timing (digi_fullBIB_uncone_extended_timing, sim ids 0-26) -- RERUN
# Submitted 2026-10-06:
#submit coning=0 digi=extended_timing chunks=27 seeding=VXD dir=InOut twoway=1 dedup=0 zmax=600 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=extended_timing_VXD_InOut_TwoWay
#submit coning=0 digi=extended_timing chunks=27 seeding=IT dir=OutIn twoway=1 dedup=0 zmax=600 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=extended_timing_IT_OutIn
#submit coning=0 digi=extended_timing chunks=27 seeding=OT dir=OutIn twoway=1 dedup=0 zmax=600 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=extended_timing_OT_OutIn

# full BIB, coned (digi_fullBIB_coned, sim ids 0-1338) -- RERUN
#submit coning=1 digi=- chunks=1339 seeding=OT dir=OutIn twoway=1 dedup=0 zmax=600 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=OT_OutIn
#submit coning=1 digi=- chunks=1339 seeding=VXD dir=InOut twoway=1 dedup=0 zmax=600 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=VXD_InOut_TwoWay
#submit coning=1 digi=- chunks=1339 seeding=IT dir=OutIn twoway=1 dedup=0 zmax=600 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=IT_OutIn
#submit coning=1 digi=- chunks=1339 seeding=VXD dir=InOut twoway=0 dedup=0 zmax=600 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=VXD_InOut
# Held back for now, to limit the output storage:
#submit coning=1 digi=- chunks=1339 seeding=IT dir=InOut twoway=1 dedup=0 zmax=600 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=IT_InOut

#submit coning=1 digi=- chunks=1339 seeding=VXDNominal dir=InOut twoway=0 dedup=0 zmax=2300 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=VXDNominal_InOut
#submit coning=1 digi=- chunks=1339 seeding=ITOTDisks dir=OutIn twoway=1 dedup=0 zmax=2300 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=ITOTDisks_OutIn


# full BIB, uncone, same two configurations (digi_fullBIB_uncone: 21 files, 3 missing)
# Submitted 2026-10-06:
#submit coning=0 digi=- chunks=24 seeding=VXDNominal dir=InOut twoway=0 dedup=0 zmax=2300 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=VXDNominal_InOut
#submit coning=0 digi=- chunks=24 seeding=ITOTDisks dir=OutIn twoway=1 dedup=0 zmax=2300 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=ITOTDisks_OutIn

# Submitted 2026-10-06:
#submit coning=0 digi=extended_timing chunks=27 seeding=VXDNominal dir=InOut twoway=0 dedup=0 zmax=2300 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=extended_timing_VXDNominal_InOut
#submit coning=0 digi=extended_timing chunks=27 seeding=ITOTDisks dir=OutIn twoway=1 dedup=0 zmax=2300 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=extended_timing_ITOTDisks_OutIn

# full BIB, coned, extended timing (digi_fullBIB_coned_extended_timing; new,
# only if that digi run is enabled in ../digi/submit_all_digi.sh)
#submit coning=1 digi=extended_timing chunks=1339 seeding=OT dir=OutIn twoway=1 dedup=0 zmax=600 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=extended_timing_OT_OutIn
#submit coning=1 digi=extended_timing chunks=1339 seeding=VXD dir=InOut twoway=1 dedup=0 zmax=600 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=extended_timing_VXD_InOut_TwoWay
#submit coning=1 digi=extended_timing chunks=1339 seeding=IT dir=OutIn twoway=1 dedup=0 zmax=600 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=extended_timing_IT_OutIn
#submit coning=1 digi=extended_timing chunks=1339 seeding=VXD dir=InOut twoway=0 dedup=0 zmax=600 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=extended_timing_VXD_InOut

# Not full BIB, so not expressible above without BIB_LEVEL=0:
# digi_NoBIB_uncone: CHUNKS 5000 (3404 files, 1596 missing)

SAMPLE=muon_pT0_50GeV_theta10_170
# All on the raw-time digi (digi_*_rawtime in ../digi/submit_all_digi.sh).

# full BIB, uncone (first 5 sim files): nominal VXD seeding vs
# IT+OT outside-in, all IT+OT layers vs the minimal outermost subset.
#submit coning=0 digi=rawtime chunks=3 seeding=VXDNominal dir=InOut twoway=0 dedup=0 zmax=2300 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=rawtime_VXDNominal_InOut
#submit coning=0 digi=rawtime chunks=3 seeding=ITOTDisks dir=OutIn twoway=1 dedup=0 zmax=2300 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=rawtime_ITOTDisks_OutIn
# No BIB, uncone (digi_NoBIB_uncone_rawtime: all 5000 sim chunks)
BIB_LEVEL=0
#submit coning=0 digi=rawtime chunks=5000 seeding=VXDNominal dir=InOut twoway=0 dedup=0 zmax=2300 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=rawtime_VXDNominal_InOut
#submit coning=0 digi=rawtime chunks=5000 seeding=ITOTDisks dir=OutIn twoway=1 dedup=0 zmax=2300 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=rawtime_ITOTDisks_OutIn
BIB_LEVEL=1667
# full BIB, coned (digi_fullBIB_coned_rawtime: first 1250 sim chunks)

# First two: rerun of the jobs that started on the broken 2026-10-19 11:59
# build (no DoTwoWayCKF); needs FILL_MISSING=1 for the partly done SeedDedup run.
#submit coning=1 digi=rawtime chunks=1250 seeding=VXDNominal dir=InOut twoway=0 dedup=1 zmax=2300 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=rawtime_VXDNominal_InOut_SeedDedup
#submit coning=1 digi=rawtime chunks=1250 seeding=VXDNominal dir=InOut twoway=1 dedup=1 zmax=2300 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=rawtime_VXDNominal_InOut_TwoWay_SeedDedup
#submit coning=1 digi=rawtime chunks=1250 seeding=ITOTDisks_Minimal_v2 dir=OutIn twoway=1 dedup=1 zmax=2300 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=rawtime_ITOTDisks_Minimal_v2_OutIn_SeedDedup
# same, skipping seeds with 2 of 3 hits on an earlier track
#submit coning=1 digi=rawtime chunks=1250 seeding=VXDNominal dir=InOut twoway=0 dedup=1 dedupshared=2 zmax=2300 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=rawtime_VXDNominal_InOut_SeedDedupShared2
#submit coning=1 digi=rawtime chunks=1250 seeding=VXDNominal dir=InOut twoway=1 dedup=1 dedupshared=2 zmax=2300 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=rawtime_VXDNominal_InOut_TwoWay_SeedDedupShared2
#submit coning=0 digi=rawtime chunks=3 seeding=ITOTDisks_Minimal dir=OutIn twoway=1 dedup=1 dedupshared=3 zmax=2300 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=rawtime_ITOTDisks_Minimal_OutIn_SeedDedup


# full BIB, uncone
#submit coning=0 digi=extended_timing_rawtime chunks=3 seeding=VXDNominal dir=InOut twoway=0 dedup=0 zmax=2300 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=extended_timing_rawtime_VXDNominal_InOut
#submit coning=0 digi=extended_timing_rawtime chunks=3 seeding=ITOTDisks dir=OutIn twoway=1 dedup=0 zmax=2300 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=extended_timing_rawtime_ITOTDisks_OutIn


#submit coning=0 digi=rawtime chunks=3 seeding=VXDNominal dir=InOut twoway=1 dedup=0 zmax=2300 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=rawtime_VXDNominal_InOut

#submit coning=0 digi=rawtime chunks=3 seeding=ITOTDisks dir=OutIn twoway=1 dedup=0 zmax=2300 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=rawtime_ITOTDisks_OutIn

# chunk 1 also hit the broken build: FILL_MISSING=1 once the other two finish
#submit coning=0 digi=rawtime chunks=3 seeding=ITOTDisks_Minimal_v2 dir=OutIn twoway=1 dedup=0 zmax=2300 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=rawtime_ITOTDisks_Minimal_v2_OutIn

#submit coning=0 digi=rawtime chunks=3 seeding=VXDNominal dir=InOut twoway=1 dedup=1 zmax=2300 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=rawtime_VXDNominal_InOut_SeedDedup
#submit coning=0 digi=rawtime chunks=3 seeding=ITOTDisks dir=OutIn twoway=1 dedup=1 zmax=2300 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=rawtime_ITOTDisks_OutIn_SeedDedup
#submit coning=0 digi=rawtime chunks=3 seeding=ITOTDisks_Minimal_v2 dir=OutIn twoway=1 dedup=1 zmax=2300 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=rawtime_ITOTDisks_Minimal_v2_OutIn_SeedDedup
# same, skipping seeds with 2 of 3 hits on an earlier track
#submit coning=0 digi=rawtime chunks=3 seeding=VXDNominal dir=InOut twoway=1 dedup=1 dedupshared=2 zmax=2300 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=rawtime_VXDNominal_InOut_SeedDedupShared2
#submit coning=0 digi=rawtime chunks=3 seeding=ITOTDisks dir=OutIn twoway=1 dedup=1 dedupshared=2 zmax=2300 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=rawtime_ITOTDisks_OutIn_SeedDedupShared2
#submit coning=0 digi=rawtime chunks=3 seeding=ITOTDisks_Minimal_v2 dir=OutIn twoway=1 dedup=1 dedupshared=2 zmax=2300 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=rawtime_ITOTDisks_Minimal_v2_OutIn_SeedDedupShared2

#submit coning=0 digi=rawtime chunks=3 seeding=ITOTDisks_Minimal dir=OutIn twoway=1 dedup=0 zmax=2300 d0max=150 cr=300 minpt=5 trkd0=150 trkz0=300 tag=rawtime_ITOTDisks_Minimal_OutIn_d0max150_CR300_minPt5
# v2 / v3 are in the d0max / CR scan below (d0max3_CR6 = the default).
#submit coning=0 digi=rawtime chunks=5 seeding=ITOTDisks_Minimal_v2 dir=OutIn twoway=1 dedup=0 zmax=2300 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=rawtime_ITOTDisks_Minimal_v2_OutIn
#submit coning=0 digi=rawtime chunks=5 seeding=ITOTDisks_Minimal_v3 dir=OutIn twoway=1 dedup=0 zmax=2300 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=rawtime_ITOTDisks_Minimal_v3_OutIn
#submit coning=0 digi=rawtime chunks=3 seeding=ITOTDisks_Minimal_v4 dir=OutIn twoway=1 dedup=0 zmax=2300 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=rawtime_ITOTDisks_Minimal_v4_OutIn

# Seed d0max / CR scan, for seed and track multiplicity.
#submit coning=0 digi=rawtime chunks=3 seeding=ITOTDisks_Minimal_v2 dir=OutIn twoway=1 dedup=0 zmax=2300 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=rawtime_ITOTDisks_Minimal_v2_OutIn_d0max3_CR6
#submit coning=0 digi=rawtime chunks=3 seeding=ITOTDisks_Minimal_v2 dir=OutIn twoway=1 dedup=0 zmax=2300 d0max=10 cr=15 minpt=1.5 trkd0=10 trkz0=10 tag=rawtime_ITOTDisks_Minimal_v2_OutIn_d0max10_CR15
#submit coning=0 digi=rawtime chunks=3 seeding=ITOTDisks_Minimal_v2 dir=OutIn twoway=1 dedup=0 zmax=2300 d0max=20 cr=20 minpt=1.5 trkd0=10 trkz0=10 tag=rawtime_ITOTDisks_Minimal_v2_OutIn_d0max20_CR20
#submit coning=0 digi=rawtime chunks=3 seeding=ITOTDisks_Minimal_v2 dir=OutIn twoway=1 dedup=0 zmax=2300 d0max=30 cr=30 minpt=1.5 trkd0=10 trkz0=10 tag=rawtime_ITOTDisks_Minimal_v2_OutIn_d0max30_CR30
#submit coning=0 digi=rawtime chunks=3 seeding=ITOTDisks_Minimal_v3 dir=OutIn twoway=1 dedup=0 zmax=2300 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=rawtime_ITOTDisks_Minimal_v3_OutIn_d0max3_CR6
#submit coning=0 digi=rawtime chunks=3 seeding=ITOTDisks_Minimal_v3 dir=OutIn twoway=1 dedup=0 zmax=2300 d0max=10 cr=15 minpt=1.5 trkd0=10 trkz0=10 tag=rawtime_ITOTDisks_Minimal_v3_OutIn_d0max10_CR15
#submit coning=0 digi=rawtime chunks=3 seeding=ITOTDisks_Minimal_v3 dir=OutIn twoway=1 dedup=0 zmax=2300 d0max=20 cr=20 minpt=1.5 trkd0=10 trkz0=10 tag=rawtime_ITOTDisks_Minimal_v3_OutIn_d0max20_CR20
#submit coning=0 digi=rawtime chunks=3 seeding=ITOTDisks_Minimal_v3 dir=OutIn twoway=1 dedup=0 zmax=2300 d0max=30 cr=30 minpt=1.5 trkd0=10 trkz0=10 tag=rawtime_ITOTDisks_Minimal_v3_OutIn_d0max30_CR30

# Seed filter scan (2026-10-26), v2 outside-in, all at |dz| <= 400 mm per doublet
# (drops most disk layer-skipping seeds). Compare with rawtime_ITOTDisks_Minimal_v2_OutIn.
# maxspm=0: 1 seed per middle SP instead of 2. Seed weight = -|d0| [mm] + compat
# per compatible top SP (max 2): compat=200 ranks 4th-hit seeds first, compat=1
# trades a 4th hit against 1 mm of d0. Needs the build with SeedFinding_DeltaZMax.
# drmaxtop=650: a compatible top must be a top candidate, and with dr <= 400 a
# barrel middle reaches only the next layer (ITB2 554 -> OTB1 1153 is 599 mm).
#submit coning=0 digi=rawtime chunks=3 seeding=ITOTDisks_Minimal_v2 dir=OutIn twoway=1 dedup=0 zmax=2300 d0max=3 cr=6 minpt=1.5 dzmax=400 trkd0=10 trkz0=10 tag=rawtime_ITOTDisks_Minimal_v2_OutIn_DZ400
#submit coning=0 digi=rawtime chunks=3 seeding=ITOTDisks_Minimal_v2 dir=OutIn twoway=1 dedup=0 zmax=2300 d0max=3 cr=6 minpt=1.5 dzmax=400 maxspm=0 trkd0=10 trkz0=10 tag=rawtime_ITOTDisks_Minimal_v2_OutIn_DZ400_SpM0
#submit coning=0 digi=rawtime chunks=3 seeding=ITOTDisks_Minimal_v2 dir=OutIn twoway=1 dedup=0 zmax=2300 d0max=3 cr=6 minpt=1.5 dzmax=400 drmaxtop=650 trkd0=10 trkz0=10 tag=rawtime_ITOTDisks_Minimal_v2_OutIn_DZ400_DRTop650
#submit coning=0 digi=rawtime chunks=3 seeding=ITOTDisks_Minimal_v2 dir=OutIn twoway=1 dedup=0 zmax=2300 d0max=3 cr=6 minpt=1.5 dzmax=400 drmaxtop=650 compat=200 trkd0=10 trkz0=10 tag=rawtime_ITOTDisks_Minimal_v2_OutIn_DZ400_DRTop650_Compat200
#submit coning=0 digi=rawtime chunks=3 seeding=ITOTDisks_Minimal_v2 dir=OutIn twoway=1 dedup=0 zmax=2300 d0max=3 cr=6 minpt=1.5 dzmax=400 drmaxtop=650 maxspm=0 compat=200 trkd0=10 trkz0=10 tag=rawtime_ITOTDisks_Minimal_v2_OutIn_DZ400_DRTop650_SpM0_Compat200
#submit coning=0 digi=rawtime chunks=3 seeding=ITOTDisks_Minimal_v2 dir=OutIn twoway=1 dedup=0 zmax=2300 d0max=3 cr=6 minpt=1.5 dzmax=400 drmaxtop=650 maxspm=0 compat=1 trkd0=10 trkz0=10 tag=rawtime_ITOTDisks_Minimal_v2_OutIn_DZ400_DRTop650_SpM0_Compat1

#submit coning=0 digi=rawtime chunks=3 seeding=ITOTDisks_Minimal_v3 dir=OutIn twoway=1 dedup=0 zmax=2300 d0max=3 cr=6 minpt=1.5 dzmax=400 trkd0=10 trkz0=10 tag=rawtime_ITOTDisks_Minimal_v3_OutIn_DZ400
#submit coning=0 digi=rawtime chunks=3 seeding=ITOTDisks_Minimal_v3 dir=OutIn twoway=1 dedup=0 zmax=2300 d0max=3 cr=6 minpt=1.5 dzmax=400 maxspm=0 trkd0=10 trkz0=10 tag=rawtime_ITOTDisks_Minimal_v3_OutIn_DZ400_SpM0
#submit coning=0 digi=rawtime chunks=3 seeding=ITOTDisks_Minimal_v3 dir=OutIn twoway=1 dedup=0 zmax=2300 d0max=3 cr=6 minpt=1.5 dzmax=400 drmaxtop=650 trkd0=10 trkz0=10 tag=rawtime_ITOTDisks_Minimal_v3_OutIn_DZ400_DRTop650
#submit coning=0 digi=rawtime chunks=3 seeding=ITOTDisks_Minimal_v3 dir=OutIn twoway=1 dedup=0 zmax=2300 d0max=3 cr=6 minpt=1.5 dzmax=400 drmaxtop=650 compat=200 trkd0=10 trkz0=10 tag=rawtime_ITOTDisks_Minimal_v3_OutIn_DZ400_DRTop650_Compat200
#submit coning=0 digi=rawtime chunks=3 seeding=ITOTDisks_Minimal_v3 dir=OutIn twoway=1 dedup=0 zmax=2300 d0max=3 cr=6 minpt=1.5 dzmax=400 drmaxtop=650 maxspm=0 compat=200 trkd0=10 trkz0=10 tag=rawtime_ITOTDisks_Minimal_v3_OutIn_DZ400_DRTop650_SpM0_Compat200
#submit coning=0 digi=rawtime chunks=3 seeding=ITOTDisks_Minimal_v3 dir=OutIn twoway=1 dedup=0 zmax=2300 d0max=3 cr=6 minpt=1.5 dzmax=400 drmaxtop=650 maxspm=0 compat=1 trkd0=10 trkz0=10 tag=rawtime_ITOTDisks_Minimal_v3_OutIn_DZ400_DRTop650_SpM0_Compat1

# Efficiency of the scan's best seed reduction (SpM0: -51% seeds; DZ400 -5%;
# DRTop650 / compat did not change the count): full BIB, coned, 1250 chunks.
# Submitted 2026-10-31 (held until the 103126 image reached CVMFS):
#submit coning=1 digi=rawtime chunks=1250 seeding=ITOTDisks_Minimal_v2 dir=OutIn twoway=1 dedup=0 zmax=2300 d0max=3 cr=6 minpt=1.5 dzmax=400 maxspm=0 trkd0=10 trkz0=10 tag=rawtime_ITOTDisks_Minimal_v2_OutIn_DZ400_SpM0
#submit coning=1 digi=rawtime chunks=1250 seeding=ITOTDisks_Minimal_v3 dir=OutIn twoway=1 dedup=0 zmax=2300 d0max=3 cr=6 minpt=1.5 dzmax=400 maxspm=0 trkd0=10 trkz0=10 tag=rawtime_ITOTDisks_Minimal_v3_OutIn_DZ400_SpM0

# No BIB, uncone (digi_NoBIB_uncone_rawtime, 5000 chunks): efficiency vs eta of the
# z-binned seedings (2026-11-10). VXDMinimal = main's seeding + VXD disk 7 + IT disk 0
# with main's zMax 600 / z edges / minPt 1 GeV. Baselines: reco_NoBIB_uncone_rawtime_
# VXDNominal_InOut and _ITOTDisks_OutIn.
BIB_LEVEL=0
submit coning=0 digi=rawtime chunks=5000 seeding=VXDMinimal dir=InOut twoway=0 dedup=0 zmax=600 d0max=3 cr=6 minpt=1.0 rmax=450 trkd0=10 trkz0=10 tag=rawtime_VXDMinimal_InOut_RMax450_MinPt1
submit coning=0 digi=rawtime chunks=5000 seeding=VXDMinimal dir=InOut twoway=1 dedup=0 zmax=600 d0max=3 cr=6 minpt=1.0 rmax=450 trkd0=10 trkz0=10 tag=rawtime_VXDMinimal_InOut_TwoWay_RMax450_MinPt1
submit coning=0 digi=rawtime chunks=5000 seeding=VXDNominal dir=InOut twoway=0 dedup=0 zmax=2300 d0max=3 cr=6 minpt=1.0 rmax=600 zbins=1 trkd0=10 trkz0=10 tag=rawtime_VXDNominal_InOut_ZBins_RMax600_MinPt1
submit coning=0 digi=rawtime chunks=5000 seeding=ITOTDisks_Minimal_v2 dir=OutIn twoway=1 dedup=0 zmax=2300 d0max=3 cr=6 minpt=1.5 dzmax=400 maxspm=0 zbins=1 trkd0=10 trkz0=10 tag=rawtime_ITOTDisks_Minimal_v2_OutIn_DZ400_SpM0_ZBins
submit coning=0 digi=rawtime chunks=5000 seeding=ITOTDisks_Minimal_v3 dir=OutIn twoway=1 dedup=0 zmax=2300 d0max=3 cr=6 minpt=1.5 dzmax=400 maxspm=0 zbins=1 trkd0=10 trkz0=10 tag=rawtime_ITOTDisks_Minimal_v3_OutIn_DZ400_SpM0_ZBins
BIB_LEVEL=1667

SAMPLE=muon_0_50GeV_theta10_170_d0PCA_300_10kEv_gen
BIB_LEVEL=167   # nomBIB = 10% of full BIB, as the digi

# 10% BIB (nomBIB), coned, extended timing (digi_nomBIB_coned_extended_timing_rawtime, ids 0-1249, 2 events each);
# same settings as the local test run of 2026-10-15
#submit coning=1 digi=extended_timing_rawtime chunks=1250 seeding=ITOTDisks_Minimal_v2 dir=OutIn twoway=1 dedup=0 zmax=2300 d0max=300 cr=300 minpt=1.5 trkd0=300 trkz0=300 tag=extended_timing_rawtime_ITOTDisks_Minimal_v2_OutIn_d0_300_CR_300

BIB_LEVEL=0     # no BIB
# No BIB, coned, extended timing (digi_NoBIB_coned_extended_timing_rawtime, ids 0-1249); same settings as above
#submit coning=1 digi=extended_timing_rawtime chunks=1250 seeding=ITOTDisks_Minimal_v2 dir=OutIn twoway=1 dedup=0 zmax=2300 d0max=300 cr=300 minpt=1.5 trkd0=300 trkz0=300 tag=extended_timing_rawtime_ITOTDisks_Minimal_v2_OutIn_d0_300_CR_300

SAMPLE=stau_4000GeV_30ns
BIB_LEVEL=1667
# full BIB, coned, loose timing (digi_fullBIB_coned_loose_timing_rawtime, ids 0-999)
#submit coning=1 digi=loose_timing_rawtime chunks=1250 seeding=VXDNominal dir=InOut twoway=1 dedup=0 zmax=2300 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=loose_timing_rawtime_VXDNominal_InOut_TwoWay
# same, 3D tracking (no seed time cuts, no CKF hit time): the 4D CKF assumes
# beta = 1 and loses the beta ~ 0.6 staus. Memory is a guess: more seeds
# survive without the time cuts.
#submit coning=1 digi=loose_timing_rawtime chunks=1250 seeding=VXDNominal dir=InOut twoway=1 dedup=0 hittime=0 zmax=2300 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=loose_timing_rawtime_VXDNominal_InOut_TwoWay_No4D
# same, 4D tracking for the stau beta (truth 0.60): hit times shifted by
# -|x| (1/beta - 1) / c. Needs the k4ActsTracking build with TimeBetaHypothesis.
#submit coning=1 digi=loose_timing_rawtime chunks=1250 seeding=VXDNominal dir=InOut twoway=1 dedup=0 beta=0.6 zmax=2300 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=loose_timing_rawtime_VXDNominal_InOut_TwoWay_Beta0p6

# Same three without BIB (digi_NoBIB_coned_loose_timing_rawtime), to isolate the
# BIB effects. Skipped until that digi run has finished. NoBIB reco keeps the
# per-hit truth, so the analyzer's seed efficiency works here.
BIB_LEVEL=0
#submit coning=1 digi=loose_timing_rawtime chunks=1250 seeding=VXDNominal dir=InOut twoway=1 dedup=0 zmax=2300 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=loose_timing_rawtime_VXDNominal_InOut_TwoWay
#submit coning=1 digi=loose_timing_rawtime chunks=1250 seeding=VXDNominal dir=InOut twoway=1 dedup=0 hittime=0 zmax=2300 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=loose_timing_rawtime_VXDNominal_InOut_TwoWay_No4D
#submit coning=1 digi=loose_timing_rawtime chunks=1250 seeding=VXDNominal dir=InOut twoway=1 dedup=0 beta=0.6 zmax=2300 d0max=3 cr=6 minpt=1.5 trkd0=10 trkz0=10 tag=loose_timing_rawtime_VXDNominal_InOut_TwoWay_Beta0p6
BIB_LEVEL=1667
