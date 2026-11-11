#!/bin/bash
# Merge the per-job reco outputs in each reco_* directory of a sample:
#   <dir>/<sample>_reco<N>.edm4hep.root  ->  <dir>/<sample>_reco.edm4hep.root
#
# A directory whose inputs add up to more than --max-gb is merged into
# several parts instead, <sample>_reco_part<K>.edm4hep.root, each under the
# limit. The coned directories (~2 TB each) need this: once a TTree passes
# ROOT's TTree::fgMaxTreeSize (100 GB), hadd switches to a "_1.root"
# continuation file and leaves a truncated output behind.
#
# Merging doubles the storage of a directory until its inputs are deleted;
# --cleanup deletes them once every part of that directory merged cleanly.
#
# Run inside the container (needs hadd).
#
# Usage: ./hadd_reco_outputs.sh [OPTIONS] [DIR_GLOB ...]
#   DIR_GLOB         reco directories to merge (default: 'reco_*'),
#                    e.g. 'reco_fullBIB_uncone*'
#   --base DIR       sample directory holding them
#                    (default: .../muon_pT2_1000GeV_theta88_92)
#   --dry-run        print what would be merged, write nothing
#   --skip-existing  leave directories that already have a merged output
#   --cleanup        delete a directory's per-job inputs after it merged OK
#   --jobs N         hadd processes run in parallel (default: 2)
#   --threads N      pass -j N to each hadd (default: off). Its workers write
#                    partial merges to $TMPDIR and merge those again, so it
#                    needs temp space up to the part size and doubles the I/O;
#                    -fk merges are I/O-bound, so expect a modest gain at best.
#                    Up to jobs x threads processes run at once.
#   --max-gb N       size limit per merged file, in GB (default: 90)
#   --exclude GLOB   skip matching directories (on top of EXCLUDE below)

set -euo pipefail

BASE=/ospool/uc-shared/project/futurecolliders/larsonma/muon_pT2_1000GeV_theta88_92
# Directories never merged, e.g. while their jobs are still running.
EXCLUDE=(
    reco_fullBIB_coned_IT_InOut   # jobs still running
)
DRY_RUN=0
SKIP_EXISTING=0
CLEANUP=0
JOBS=2
THREADS=0
MAX_GB=90
PATTERNS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base)          BASE="$2"; shift ;;
        --dry-run)       DRY_RUN=1 ;;
        --skip-existing) SKIP_EXISTING=1 ;;
        --cleanup)       CLEANUP=1 ;;
        --jobs)          JOBS="$2"; shift ;;
        --threads)       THREADS="$2"; shift ;;
        --max-gb)        MAX_GB="$2"; shift ;;
        --exclude)       EXCLUDE+=("$2"); shift ;;
        -*)              echo "Unknown option: $1" >&2; exit 1 ;;
        *)               PATTERNS+=("$1") ;;
    esac
    shift
done
[[ ${#PATTERNS[@]} -eq 0 ]] && PATTERNS=("reco_*")
MAX_BYTES=$(awk -v g="$MAX_GB" 'BEGIN { printf "%d", g * 1e9 }')

if [[ $DRY_RUN -eq 0 ]] && ! command -v hadd > /dev/null 2>&1; then
    echo "[error] hadd not found on PATH: run this inside the container." >&2
    exit 1
fi

INPUT_GLOB='*_reco[0-9]*.edm4hep.root'

gb() { awk -v b="$1" 'BEGIN { printf "%.0f", b / 1e9 }'; }

# hadd_one <output> <input>...
# Writes to a temporary name and renames on success, so a killed or failed
# merge never leaves something that looks like a finished output.
hadd_one() {
    local out=$1; shift
    local tmp="${out%.edm4hep.root}.tmp.edm4hep.root"
    local log="${out}.haddlog"
    local -a jflag=()
    if [[ $THREADS -gt 0 ]]; then jflag=(-j "$THREADS"); fi
    echo "  [hadd] $(basename "$out")  ($# files)"
    # -fk: keep the inputs' compression, so baskets are copied, not recompressed.
    if hadd -fk "${jflag[@]}" "$tmp" "$@" > "$log" 2>&1; then
        if [[ -e "${tmp%.root}_1.root" ]]; then
            rm -f "$tmp" "${tmp%.root}"_[0-9]*.root
            echo "  [error] $(basename "$out") passed ROOT's 100 GB TTree limit; rerun with a lower --max-gb" >&2
            return 1
        fi
        mv -f "$tmp" "$out"
        rm -f "$log"
        echo "  [ok]   $(basename "$out")"
    else
        rm -f "$tmp"
        echo "  [error] hadd failed for $(basename "$out")  (log: $log)" >&2
        tail -n 5 "$log" | sed 's/^/          /' >&2
        return 1
    fi
}

# merge_dir <dir>: returns non-zero if any part failed.
merge_dir() {
    local dir=$1 name
    name=$(basename "$dir")

    local -a files=() sizes=()
    local size path
    while read -r size path; do
        files+=("$path"); sizes+=("$size")
    done < <(find "$dir" -maxdepth 1 -name "$INPUT_GLOB" -printf '%s %p\n' | sort -k2 -V)

    if [[ ${#files[@]} -eq 0 ]]; then
        echo "[skip] $name: no per-job reco files"
        return 0
    fi

    # <sample>_reco<N>.edm4hep.root -> <sample>_reco
    local base
    base=$(basename "${files[0]}" | sed -E 's/[0-9]+\.edm4hep\.root$//')

    # Consecutive groups of inputs, each at most MAX_BYTES.
    local -a part=()
    local i k=0 acc=0 total=0
    for i in "${!files[@]}"; do
        if (( acc > 0 && acc + sizes[i] > MAX_BYTES )); then
            k=$((k + 1)); acc=0
        fi
        part[i]=$k
        acc=$((acc + sizes[i])); total=$((total + sizes[i]))
    done
    local nparts=$((k + 1))

    local -a outs=()
    local p
    if [[ $nparts -eq 1 ]]; then
        outs=("$dir/$base.edm4hep.root")
    else
        for ((p = 0; p < nparts; p++)); do outs+=("$dir/${base}_part$p.edm4hep.root"); done
    fi

    local -a existing=()
    mapfile -t existing < <(find "$dir" -maxdepth 1 \( -name "$base.edm4hep.root" \
        -o -name "${base}_part[0-9]*.edm4hep.root" \) ! -name '*.tmp.edm4hep.root' | sort -V)

    if [[ ${#existing[@]} -gt 0 && $SKIP_EXISTING -eq 1 ]]; then
        # Inputs newer than the merge mean jobs finished after it was made.
        if [[ -n "$(find "$dir" -maxdepth 1 -name "$INPUT_GLOB" -newer "${existing[0]}" -print -quit)" ]]; then
            echo "[skip] $name: merged already, but INPUTS ARE NEWER - stale, rerun without --skip-existing"
        else
            echo "[skip] $name: merged already"
        fi
        return 0
    fi

    echo "[merge] $name: ${#files[@]} files, $(gb "$total") GB -> $nparts output(s)"
    if [[ -n "$(find "$dir" -maxdepth 1 -name "$INPUT_GLOB" -mmin -30 -print -quit)" ]]; then
        echo "  [warn] inputs written in the last 30 min: jobs may still be running"
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        local n
        for ((p = 0; p < nparts; p++)); do
            n=0
            for i in "${!files[@]}"; do if [[ ${part[i]} -eq $p ]]; then n=$((n + 1)); fi; done
            echo "  would write $(basename "${outs[p]}")  ($n files)"
        done
        return 0
    fi

    # A previous merge may have been split differently; clear it so no
    # outdated part is left next to the new ones.
    if [[ ${#existing[@]} -gt 0 ]]; then
        echo "  removing previous merge: ${#existing[@]} file(s)"
        rm -f "${existing[@]}"
    fi
    rm -f "$dir/$base"*.tmp.edm4hep.root

    local failed=0 q pid
    local -a in=() pids=()
    for ((p = 0; p < nparts; p += JOBS)); do
        pids=()
        for ((q = p; q < p + JOBS && q < nparts; q++)); do
            in=()
            for i in "${!files[@]}"; do if [[ ${part[i]} -eq $q ]]; then in+=("${files[i]}"); fi; done
            hadd_one "${outs[q]}" "${in[@]}" &
            pids+=($!)
        done
        for pid in "${pids[@]}"; do wait "$pid" || failed=$((failed + 1)); done
    done

    if [[ $failed -gt 0 ]]; then
        # Remove the parts that did succeed too: an incomplete set would look
        # merged to a later --skip-existing run.
        rm -f "${outs[@]}"
        echo "  [error] $name: $failed part(s) failed; partial merge removed, inputs kept" >&2
        return 1
    fi
    if [[ $CLEANUP -eq 1 ]]; then
        rm -f "${files[@]}"
        echo "  cleaned up ${#files[@]} input files ($(gb "$total") GB)"
    fi
}

mapfile -t DIRS < <(for pat in "${PATTERNS[@]}"; do
    for d in "$BASE"/$pat; do [[ -d "$d" ]] && echo "$d"; done
done | sort -u)

if [[ ${#DIRS[@]} -eq 0 ]]; then
    echo "No directories matching ${PATTERNS[*]} in $BASE"
    exit 0
fi

errors=0
for d in "${DIRS[@]}"; do
    excluded=0
    for ex in "${EXCLUDE[@]}"; do
        # shellcheck disable=SC2053  # $ex is a glob on purpose
        if [[ "$(basename "$d")" == $ex ]]; then excluded=1; fi
    done
    if [[ $excluded -eq 1 ]]; then
        echo "[skip] $(basename "$d"): excluded"
        continue
    fi
    merge_dir "$d" || errors=$((errors + 1))
done

echo ""
if [[ $errors -eq 0 ]]; then
    echo "Done."
else
    echo "[warn] $errors directory merge(s) had failures."
    exit 1
fi
