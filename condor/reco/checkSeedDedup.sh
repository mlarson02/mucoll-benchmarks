#!/bin/bash
# Sum the "Seed deduplication: skipped X of Y seeds" lines from reco job logs.
# Usage: ./checkSeedDedup.sh <out_dir name> [cluster ID]
#   e.g. ./checkSeedDedup.sh reco_fullBIB_uncone_rawtime_VXDNominal_InOut_SeedDedup 42476425
DIR=${1:?usage: $0 <out_dir name> [cluster ID]}
CLUSTER=$2
LOGS=$(dirname "$0")/logs

files=$(grep -l "out_dir: .*/${DIR}\$" "$LOGS"/out.${CLUSTER:-*}.* 2>/dev/null | sort -V)
[ -z "$files" ] && { echo "no jobs found writing to $DIR"; exit 1; }

printf "%-22s %10s %14s %8s  %s\n" job skipped seeds frac status
for f in $files; do
    job=${f##*/out.}
    status=running
    grep -q "^>>> reco chunk .* done" "$f" && status=done
    line=$(grep -h "Seed deduplication: skipped" "$f" "${f/out./err.}" 2>/dev/null | head -1)
    if [ -n "$line" ]; then
        read skipped seeds <<< "$(sed -E 's/.*skipped ([0-9]+) of ([0-9]+) seeds.*/\1 \2/' <<< "$line")"
        awk -v j="$job" -v s="$skipped" -v n="$seeds" -v st="$status" \
            'BEGIN { printf "%-22s %10d %14d %7.4f%%  %s\n", j, s, n, (n ? 100*s/n : 0), st }'
        echo "$skipped $seeds" >> /dev/shm/seeddedup.$$
    else
        printf "%-22s %10s %14s %8s  %s\n" "$job" - - - "$status (no dedup line)"
    fi
done

[ -f /dev/shm/seeddedup.$$ ] && awk '{ s += $1; n += $2; j++ }
    END { printf "TOTAL (%d jobs)          %10d %14d %7.4f%%\n", j, s, n, (n ? 100*s/n : 0) }' /dev/shm/seeddedup.$$
rm -f /dev/shm/seeddedup.$$
