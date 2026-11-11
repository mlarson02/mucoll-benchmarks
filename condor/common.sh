#!/bin/bash
# Sourced by sim.sh / digi.sh / reco.sh inside the mucoll container.
# Sets up /opt/setup_mucoll.sh and the mucoll-benchmarks MAIA config that
# was transferred with the job. FIXME: eventually replace this stanza with
# a sourced setup.sh from a cloned+built TemplateWorkspace overlay.

# NB: intentionally NOT using `set -u` — the mucoll spack setup.sh
# references ACLOCAL_PATH (and other) unset vars.
set -eo pipefail

echo "<<< host: $HOSTNAME"
echo "<<< singularity image: ${APPTAINER_NAME:-${SINGULARITY_NAME:-unknown}}"

echo "<<< source /opt/setup_mucoll.sh"
# shellcheck disable=SC1091
source /opt/setup_mucoll.sh

# The submit file transfers the config subtree (configs/, setup_config.sh,
# simulation/steer_baseline.py). Reconstruct the mucoll-benchmarks layout
# so setup_config.sh can find configs/MAIAConfig/MAIAConfig relative to it.
mkdir -p mucoll-benchmarks/simulation
[ -e setup_config.sh ] && mv setup_config.sh mucoll-benchmarks/
[ -e configs ] && mv configs mucoll-benchmarks/
[ -e steer_baseline.py ] && mv steer_baseline.py mucoll-benchmarks/simulation/

pushd mucoll-benchmarks >/dev/null
# pipefail off: setup_config.sh runs `find ... | head -n 1`, and find can
# die of SIGPIPE once head has exited; with pipefail that killed ~28% of jobs
# (exit 141, empty stderr, mostly on osris.org nodes).
set +o pipefail
# shellcheck disable=SC1091
source setup_config.sh . MAIA_v0
set -o pipefail
popd >/dev/null

echo "<<< which ddsim: $(which ddsim)"
echo "<<< which k4run: $(which k4run)"
echo "<<< MUCOLL_CONFIG=${MUCOLL_CONFIG}"
echo "<<< MUCOLL_GEO=${MUCOLL_GEO}"

# ---- pelican / OSDF staging -------------------------------------------
# Remote OSG sites do not mount /ospool, so inputs and outputs move over
# OSDF via pelican rather than plain cp.
PELICAN_VERSION="${PELICAN_VERSION:-7.15.3}"
PELICAN_FEDERATION="${PELICAN_FEDERATION:-osg-htc.org}"

setup_pelican() {
    if command -v pelican >/dev/null 2>&1; then
        echo "<<< pelican already on PATH: $(command -v pelican)"
        return 0
    fi
    echo "<<< bootstrapping pelican ${PELICAN_VERSION}"
    wget -q "https://dl.pelicanplatform.org/${PELICAN_VERSION}/pelican_Linux_x86_64.tar.gz"
    tar -xzmf pelican_Linux_x86_64.tar.gz
    rm -f pelican_Linux_x86_64.tar.gz
    export PATH="${PWD}/pelican-${PELICAN_VERSION}:${PATH}"
    echo "<<< pelican: $(command -v pelican)"
}

# stage_in <absolute /ospool path> <local filename>
stage_in() {
    local remote="$1" dest="$2"
    echo "<<< pelican get ${remote} -> ${dest}"
    pelican object get -d "pelican://${PELICAN_FEDERATION}${remote}" "$dest"
}

# stage_out <local filename> <absolute /ospool destination dir>
stage_out() {
    local src="$1" remote_dir="$2"
    echo "<<< pelican put ${src} -> ${remote_dir}/"
    pelican object put -d "$src" "pelican://${PELICAN_FEDERATION}${remote_dir}/${src}"
}

# ---- output directory naming ------------------------------------------
# Computed here rather than in the .submit files: HTCondor's $CHOICE is a
# numeric-index function, not a key=value lookup, so it cannot do this.

# bib_tag <OverlayFullNumberBackground>
bib_tag() {
    case "$1" in
        0)    echo "NoBIB"   ;;
        17)   echo "1pcBIB"  ;;
        81)   echo "5pcBIB"  ;;
        167)  echo "nomBIB"  ;;
        1667) echo "fullBIB" ;;
        *)    echo "$1bkgBIB" ;;
    esac
}

# cone_tag <0|1>
cone_tag() {
    if [ "$1" = "1" ]; then echo "coned"; else echo "uncone"; fi
}

# name_tag <label> -> "_<label>", or nothing when the label is empty.
# Appends an optional user label to an output directory name so that two
# runs with the same BIB/coning settings can live side by side, e.g.
# digi_fullBIB_coned_v2. Anything outside [A-Za-z0-9._-] is stripped so a
# stray space or slash in the label can never rewrite the path.
name_tag() {
    local t="${1//[^[:alnum:]._-]/}"
    if [ -n "$t" ]; then echo "_$t"; fi
}

setup_pelican
