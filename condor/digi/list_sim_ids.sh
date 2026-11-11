#!/bin/bash
# Print the ProcId of every sim output that actually exists, one per line.
# digi.submit calls this via `queue PROCID from ... |` so a submission only
# ever queues jobs whose input is present (sim chunks that failed or were
# held simply drop out).
#
# usage: list_sim_ids.sh <outputRoot> <sample> [maxFiles]
#
# maxFiles: emit only the first N ids (lowest chunk id first). Omit or
# pass 0 for all of them.
set -eo pipefail

output_root="$1"
sample="$2"
max_files="${3:-0}"

if [ -z "$output_root" ] || [ -z "$sample" ]; then
    echo "usage: list_sim_ids.sh <outputRoot> <sample> [maxFiles]" >&2
    exit 2
fi

sim_dir="${output_root}/${sample}/sim"
if [ ! -d "$sim_dir" ]; then
    echo "list_sim_ids.sh: no such dir: $sim_dir" >&2
    exit 3
fi

all_ids="$(ls "$sim_dir" \
    | sed -n "s/^${sample}_sim\([0-9]\+\)\.edm4hep\.root\$/\1/p" \
    | sort -n)"

if [ "$max_files" -gt 0 ] 2>/dev/null; then
    printf '%s\n' "$all_ids" | head -n "$max_files"
else
    printf '%s\n' "$all_ids"
fi
