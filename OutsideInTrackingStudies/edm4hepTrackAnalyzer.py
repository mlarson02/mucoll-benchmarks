from podio.reading import get_reader  # for the generic reader
from podio.root_io import Reader  # for TTree (i.e. ROOTReader)
import edm4hep
import ROOT
from collections import defaultdict
ROOT.gInterpreter.LoadFile("podio/LinkNavigator.h")
ROOT.gInterpreter.LoadFile("podio/LinkCollection.h")
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm, TwoSlopeNorm
import glob
import os
import re
import sys
import random
from math import *
import time
import zlib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator, NullLocator

# Minor ticks on every plot (categorical / integer x axes turn them off).
plt.rcParams.update({"xtick.minor.visible": True, "ytick.minor.visible": True})

import numpy as np

system_to_relname = {
    1: "VXDBarrel", 2: "VXDEndcap",
    3: "ITBarrel",  4: "ITEndcap",
    5: "OTBarrel",  6: "OTEndcap",
}

def cell_layer(cell_id):
    """Layer field of a tracker cellID ("system:5,side:-2,layer:6,module:11,sensor:8"),
    bits 7-12. Works on ints and numpy arrays."""
    return (cell_id >> 7) & 0x3F

def draw_radius_circles(ax, radii, color, **kwargs):
    for r in radii:
        circle = patches.Circle((0, 0), r, fill=False, color=color, linewidth=1.2, **kwargs)
        ax.add_patch(circle)


vb_r = [30, 51, 74, 102]
vb_l = 130
ve_z = [80, 120, 200, 280]
ve_h = [0, 15, 35, 50]

ib_r = [127, 340, 554]
ib_l = [963.2, 963.2, 1384.6]
ie_z = [524, 808, 1093, 1377, 1661, 1946, 2190]
ie_h = np.arange(100, 300, (300-100)/7)

ob_r = [819, 1153, 1486]
ob_l = 2528.4
oe_z = [1310, 1617, 1883, 2190]

def index_links(link_collection, direction="from_to"):
    """
    Returns a dict keyed by objectID.index with list of (other_obj, weight).
    direction="from_to" : key=from, values=to
    direction="to_from" : key=to,   values=from
    """
    out = defaultdict(list)
    for link in link_collection:
        a = link.getFrom()
        b = link.getTo()
        w = link.getWeight() if hasattr(link, "getWeight") else 1.0

        if direction == "from_to":
            out[a.getObjectID().index].append((b, w))
        else:
            out[b.getObjectID().index].append((a, w))
    return out


BField = 5  # FIXME change this for MuColl_v1 or MAIA

# PDG id of the signal particle, matched on |PDG| so that both gun signs work:
# --pdg 13 is mu-, --pdg -13 is mu+ (the latter is run_chain.sh's default, and
# what the physics-validation gen files use). Matching on == 13 silently gives a
# zero denominator, hence 0% efficiency, for a mu+ sample.
SIGNAL_PDG = abs(int(os.environ.get("SIGNAL_PDG", 13)))

def is_signal_pdg(pdg):
    """True when `pdg` is the signal species, either charge sign."""
    return abs(int(pdg)) == SIGNAL_PDG

# --- z = 0 module-seam diagnostic -------------------------------------------
# The IT and OT barrels place their modules with an even nz, so a module edge
# falls exactly on z = 0 (z_incr == module length == 30.1 mm, no overlap). The
# vertex barrel uses an odd nmodules = 5, so z = 0 is mid-module there. These
# two knobs drive a zoomed r-z scatter and a fine z profile to make the
# resulting hole visible.
ZSEAM_COLLECT_HALF = 25.0   # collect every merged hit with |z| < this [mm]
ZSEAM_ZOOM_HALF    = 1.0    # x-range of the zoomed scatter [mm]
ZSEAM_MAX_HITS     = 2000000  # safety cap so a BIB sample cannot blow up memory

ETA_ZERO = False  # set True for eta~0 tracks: zooms r-z plots to z in [-500, 500] mm
RZ_ZLIM = 500 if ETA_ZERO else 2100

# Whether the input reco was produced from a coned-digi sample. Controls the
# subsystem-collection name suffix in the [DIAG] block ("VXDBarrelHitsConed"
# vs "VXDBarrelHits"); MergedTrackerHits (used by the rest of the analyzer) is
# the same in both cases. Precedence: USE_CONED env var > per-file autodetect.
def _autodetect_use_coned(fname):
    return "coned" in os.path.basename(fname).lower()

_env_use_coned = os.environ.get("USE_CONED")
if _env_use_coned is not None:
    USE_CONED = _env_use_coned not in ("0", "false", "False", "")
else:
    USE_CONED = None  # deferred; set per-file inside the loop below

GENERATE_ID_DIAG = True  # set True to generate the cellID (system, side, layer) verification plot + summary

# Worker processes for the event loop, one input file per task (N_WORKERS env
# var overrides; 1 = run in this process). Memory sets this, not cores: a
# worker holds one whole event, 5+ GB at 100% BIB uncone, and this machine
# caps each user at 64 GiB across all their processes (16 workers got OOM-killed).
N_WORKERS = int(os.environ.get("N_WORKERS", 1))

# Per-signal-muon efficiency debug printout, plus the truth composition of
# every AllTracks track when the muon has no AllTracks link. Slow on BIB: it
# walks every hit link in Python.
EFF_DEBUG = False

# Per-track AllTracks printout (print_alltracks) for the first PRINT_ALLTRACKS
# events of each file; 0 = off. Use N_WORKERS=1 to keep files' output apart.
PRINT_ALLTRACKS = int(os.environ.get("PRINT_ALLTRACKS", 0))

# Seed pT plots: the muon's seed is one with at least SEED_MATCH_MIN_HITS of its
# 3 hits from the muon (3: no BIB hit). If there is none, or the hits were not
# written out, it is the seed closest to the muon in (eta, phi) within
# SEED_MATCH_MAX_DR.
SEED_MATCH_MIN_HITS = 3
SEED_MATCH_MAX_DR = 0.05

# --- No-hits mode -----------------------------------------------------------
# The 100% BIB uncone reco output is written without any tracker-hit
# collections: no MergedTrackerHits, no MergedTrackerHitsRelations, no
# per-subdetector hit collections, no SimTrackerHits. What survives is
# AllTracks / DedupedTracks / SiTracks / SeedTracks, their truth links
# (AllTracksRelations, DedupedTracksRelations, SiTrackRelations) and
# MCParticles — enough for track multiplicity, efficiency and perigee
# parameters, but nothing that needs a hit position or a cellID.
#
# In this mode the analyzer must never dereference a hit object: the tracks
# still carry their trackerHits relations, but those point at a collection podio
# cannot resolve, so the hit objects come back unavailable and reading a
# position or a cellID off one is not safe. The relation *size* is a different
# matter — _AllTracks_trackerHits stores the ObjectIDs, so len() still gives the
# exact per-track hit count, and SiTracks / DedupedTracks are subset collections
# of AllTracks so the same objects (and the same relations) back them. Whether
# podio lets len() through for an unresolvable collection id is version
# dependent, so it is probed on the first event rather than assumed;
# NHITS_SOURCE records the outcome.
#
# There is no second source for the count: Track::subdetectorHitNumbers would
# be the obvious one (it is a vector member, stored inline, and FilterTracksAlg
# indexes it by system id 1-6) but the ACTS -> EDM4hep converter leaves it
# empty, so it reads 0 for every track. That also means the per-subdetector
# split (pixel / inner / outer, per-layer) is genuinely unrecoverable from a
# no-hits file — only totals survive.
#
# Precedence: NO_HITS env var ("1"/"true" to force on, "0"/"false" to force
# off) > per-file autodetect from the first event's collection list.
NO_HITS_REQUIRED_COLLECTIONS = ("MergedTrackerHits", "MergedTrackerHitsRelations")

_env_no_hits = os.environ.get("NO_HITS")
if _env_no_hits is not None:
    NO_HITS = _env_no_hits not in ("0", "false", "False", "")
else:
    NO_HITS = None  # deferred; set per-file inside the loop below

# Where track_nhits() reads per-track hit counts from. Only consulted in
# no-hits mode; "trackerHits" is what a normal file always uses.
NHITS_SOURCE = "trackerHits"


def _detect_mode(event):
    """Return (no_hits, nhits_source) for the file `event` was read from.

    Called on the first event of each file, so it costs no extra read.
    """
    try:
        available = set(event.getAvailableCollections())
    except AttributeError:
        available = {c for c in NO_HITS_REQUIRED_COLLECTIONS if event.get(c) is not None}
    if all(c in available for c in NO_HITS_REQUIRED_COLLECTIONS):
        return False, "trackerHits"

    # Hit collections are gone: check whether the trackerHits relation size
    # is still readable. A CKF track always has hits, so one track
    # reporting a non-zero count settles it.
    for cname in ("AllTracks", "SeedTracks"):
        coll = event.get(cname)
        if coll is None:
            continue
        for ti, trk in enumerate(coll):
            if ti >= 50:
                break
            try:
                if len(trk.getTrackerHits()) > 0:
                    return True, "trackerHits"
            except Exception:
                pass
    return True, "unavailable"


def track_nhits(track):
    """Hits on a track, without dereferencing the hit objects.

    0 when the relation size did not survive the drop (NHITS_SOURCE ==
    "unavailable"), in which case every hit-count distribution is left empty.
    """
    if not NO_HITS or NHITS_SOURCE == "trackerHits":
        return len(track.getTrackerHits())
    return 0


def track_has_hits(track):
    """The 'this is a real track, not an empty stub' test used by the
    multiplicity and efficiency counters.

    With no hit count available at all the test cannot be made, so every track
    in the collection counts and the multiplicities are raw collection sizes.
    """
    if NO_HITS and NHITS_SOURCE == "unavailable":
        return True
    return track_nhits(track) > 0



def truth_d0_z0_helix(mcp, Bz_T=5.0, xref=0.0, yref=0.0, zref=0.0,
                      p_in_GeV=True, pos_in_mm=True):
    """
    Helical d0, z0 analogous to CMS L1 code, for an EDM4hep MCParticle.

    Ported from /scratch/larsonma/tutorial2024/SLCIO-Analyzer/slcio_analyzer.py.
    xref,yref,zref: reference point (beamspot). Usually (0,0,0).
    Returns (d0, z0) in the same length units as the input positions (mm if pos_in_mm).
    """
    v = mcp.getVertex()
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    p = mcp.getMomentum()
    px, py, pz = float(p[0]), float(p[1]), float(p[2])
    q = float(mcp.getCharge())

    pt = sqrt(px*px + py*py)
    if pt == 0 or q == 0 or Bz_T == 0:
        # straight-line fallback
        d0 = ((vx - xref) * py - (vy - yref) * px) / (pt if pt != 0 else 1.0)
        pt2 = px*px + py*py
        tpar = -(((vx - xref) * px + (vy - yref) * py) / pt2) if pt2 != 0 else 0.0
        z0 = (vz - zref) + pz * tpar + zref
        return d0, z0

    phi  = atan2(py, px)
    tanL = pz / pt

    c_conv  = 0.299792458 if p_in_GeV else 1.0
    r2_inv  = q * c_conv * Bz_T / pt / 2.0   # 1/(2R) in 1/m

    scale   = 1e-3 if pos_in_mm else 1.0
    delx_m  = (vx - xref) * scale
    dely_m  = (vy - yref) * scale
    vz_m    = (vz - zref) * scale

    inv2r2  = 1.0 / (2.0 * r2_inv)            # R in m

    x0p = delx_m - inv2r2 * sin(phi)
    y0p = dely_m + inv2r2 * cos(phi)
    rp  = sqrt(x0p*x0p + y0p*y0p)

    d0_m = q * rp - inv2r2

    delphi = phi - atan2(-r2_inv * x0p, r2_inv * y0p)
    if delphi < -pi: delphi += 2.0 * pi
    if delphi >  pi: delphi -= 2.0 * pi

    z0_m = vz_m + tanL * delphi * inv2r2

    if pos_in_mm:
        d0 = d0_m / scale
        z0 = (z0_m / scale) + zref
    else:
        d0 = d0_m
        z0 = z0_m + zref
    return d0, z0


def plot_tag(fname):
    """Short name identifying one input file in plot paths and legend labels.

    The condor reco output gives every variant the same basename and tells them
    apart by directory (.../reco_fullBIB_uncone_VXD_InOut/<sample>_reco.edm4hep.root),
    so the parent directory is the distinguishing part and is used whenever the
    path has one; a directory entry is named after itself. A bare filename in
    the run directory falls back to the basename.
    """
    if os.path.isdir(fname):
        return os.path.basename(os.path.normpath(fname))
    parent = os.path.dirname(fname)
    if parent:
        return os.path.basename(os.path.normpath(parent))
    return os.path.basename(fname).replace(".edm4hep.root", "")


def make_plot_tags(fnames):
    """Map each input file to a unique plot_tag().

    Two files sharing a parent directory would otherwise write into the same
    plot directory and carry the same legend label, so those get the basename
    appended to tell them apart.
    """
    tags = [plot_tag(f) for f in fnames]
    unique = []
    for tag, fname in zip(tags, fnames):
        if tags.count(tag) > 1:
            unique.append(f"{tag}_{os.path.basename(fname).replace('.edm4hep.root', '')}")
        else:
            unique.append(tag)
    return dict(zip(fnames, unique))


# Per-job condor reco outputs, <sample>_reco<N>.edm4hep.root. Matches neither
# the hadd outputs (<sample>_reco.edm4hep.root, <sample>_reco_part<K>...) nor
# hadd's .tmp files, so a directory that has also been merged is not read twice.
PER_JOB_GLOB = "*_reco[0-9]*.edm4hep.root"

# Chain at most this many files per entry (None = all of them), e.g. for a
# quick look at a ~1000-file coned directory.
MAX_FILES_PER_INPUT = None


def _natural_key(path):
    """Sort key putting _reco2 before _reco10."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path)]


def expand_input(entry):
    """The files behind one file_names entry, read as a single sample.

    An entry is a file, a directory (its per-job reco outputs, chained: the
    TChain-like way to read a condor run without hadd-ing it first), or a glob.
    A directory whose per-job files were already merged and removed
    (hadd_reco_outputs.sh --cleanup) is read through its merged outputs.
    """
    if os.path.isdir(entry):
        files = glob.glob(os.path.join(entry, PER_JOB_GLOB))
        if not files:
            files = [f for pattern in ("*_reco.edm4hep.root", "*_reco_part[0-9]*.edm4hep.root")
                     for f in glob.glob(os.path.join(entry, pattern))
                     if not f.endswith(".tmp.edm4hep.root")]
    elif any(c in entry for c in "*?["):
        files = [f for f in glob.glob(entry) if not f.endswith(".tmp.edm4hep.root")]
    else:
        files = [entry]
    files = sorted(files, key=_natural_key)
    if MAX_FILES_PER_INPUT is not None:
        files = files[:MAX_FILES_PER_INPUT]
    if not files:
        raise FileNotFoundError(f"no input files for {entry}")
    return files


def open_reader(files):
    """podio's ROOT reader chains a list of files into one event sequence (a
    TChain underneath). get_reader takes a single file only, but it also
    handles the non-TTree formats, so it is kept for that case."""
    return get_reader(files[0]) if len(files) == 1 else Reader(files)


# Efficiency vs theta: 12 bins of pi/12 rad. TH_BRL is the theta of |eta| = 0.6,
# the barrel / endcap boundary drawn on the eta plots.
THETA_BINS = np.linspace(0, np.pi, 13)
# Efficiency vs truth displacement: (key, bins [mm], x label, file tag). Values
# outside the bins are folded into the edge bins.
DISPLACEMENT_EFF = [
    ("eff_truth_d0", np.linspace(-300, 300, 61), r"Truth Muon $d_0$ [mm]", "d0"),
    ("eff_truth_z0", np.linspace(-300, 300, 61), r"Truth Muon $z_0$ [mm]", "z0"),
    ("eff_truth_vrxy", np.linspace(0, 1000, 51), r"Truth Muon production vertex $r_{xy}$ [mm]", "vrxy"),
]
EFF2D_MIN_MUONS = 5   # efficiency maps leave bins with fewer muons blank
TH_BRL = 2 * np.arctan(np.exp(-0.6))
# Efficiency vs pT up to 50 GeV: log bins, ~10 per decade, from 0.5 GeV (a log axis
# cannot start at 0); pT < 0.5 GeV is clipped into the first bin.
PT_BINS_0_50 = np.logspace(np.log10(0.5), np.log10(50), 21)
# Seed pT: from the 1.5 GeV seed pT cut (default --seedMinPt) to 50 GeV.
SEED_PT_BINS = np.logspace(np.log10(1.5), np.log10(50), 16)


def compute_efficiency(truth_vals, matched_flags, bins):
    truth_arr = np.array(truth_vals)
    matched_arr = np.array(matched_flags, dtype=float)
    total, _ = np.histogram(truth_arr, bins=bins)
    matched_hist, _ = np.histogram(truth_arr, bins=bins, weights=matched_arr)
    centers = 0.5 * (bins[:-1] + bins[1:])
    with np.errstate(divide='ignore', invalid='ignore'):
        eff = np.where(total > 0, matched_hist / total, np.nan)
        err = np.where(total > 0, np.sqrt(eff * (1 - eff) / np.maximum(total, 1)), np.nan)
    mask = total > 0
    return centers[mask], eff[mask], err[mask]


def eff_ticks(eta_x=False):
    """Efficiency y axis on the current plot: 0 to 1.05, ticks (and grid) every
    0.1, minor every 0.02. With eta_x, x ticks every 0.5, minor every 0.1."""
    ax = plt.gca()
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.02))
    if eta_x:
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))


def unit_weights(values, bins):
    """Per-entry weights that normalise a histogram of `values` to unit sum over
    `bins`, like ROOT's Scale(1/Integral()): entries outside the bins don't count."""
    v = np.asarray(values, dtype=float)
    n = np.count_nonzero((v >= bins[0]) & (v <= bins[-1]))
    return np.full(len(v), 1.0 / n if n else 0.0)


# pT thresholds [GeV] for the fake-rate-vs-threshold plots
FAKE_PT_THRESHOLDS = np.arange(0, 15.25, 0.5)


def rate_above(values, thresholds, n_events):
    """Entries per event with value > each threshold, and its Poisson error
    sqrt(N) / n_events. Zeros become NaN so a log y-axis just skips them."""
    v = np.sort(np.asarray(values, dtype=float))
    n = (len(v) - np.searchsorted(v, thresholds, side='right')).astype(float)
    n[n == 0] = np.nan
    return n / n_events, np.sqrt(n) / n_events


# A SiTrack is a duplicate when more than this fraction of its own hits are on
# one other SiTrack ranked above it (more hits, then lower chi2): a group of N
# overlapping tracks gives N - 1 duplicates.
DUP_SHARED_FRAC = 0.5


# Seed |d0| / max doublet |zOrigin| plots: bins [mm] and the current CKF_tracking.py
# defaults of SeedFinding_ImpactMax / SeedFinding_CollisionRegion, drawn for reference.
SEED_D0_BINS = np.linspace(0, 100, 51)
SEED_ZORIGIN_BINS = np.linspace(0, 150, 76)
SEED_D0_REF, SEED_ZORIGIN_REF = 3.0, 6.0


def seed_cut_label(name, v, cut):
    """Legend text: 95% / 99% quantiles and the fraction below the reference cut."""
    q95, q99 = np.percentile(v, [95, 99])
    return f"{name}: 95% < {q95:.2g}, 99% < {q99:.2g} mm, {100*np.mean(v < cut):.1f}% < {cut:g}"


def rate_per_event(values, bins, n_events, clip=False):
    """Bin centres, entries per event in each bin, and the Poisson error
    sqrt(N) / n_events. clip=True folds values outside the bins into the edge bins."""
    v = np.asarray(values, dtype=float)
    if clip:
        v = np.clip(v, bins[0], bins[-1])
    n, _ = np.histogram(v, bins=bins)
    n_ev = max(n_events, 1)
    return 0.5 * (bins[:-1] + bins[1:]), n / n_ev, np.sqrt(n) / n_ev


def hist2d_lognorm(x, y, bins, weights=None, cmap="viridis"):
    """Plot a 2D log-color histogram, computing vmin/vmax from actual non-zero bins.
    Returns the image object, or None if all bins are empty."""
    h, _, _ = np.histogram2d(x, y, bins=bins, weights=weights)
    nz = h[h > 0]
    if len(nz) == 0:
        return None
    norm = LogNorm(vmin=nz.min(), vmax=h.max())
    _, _, _, img = plt.hist2d(x, y, bins=bins, weights=weights, norm=norm, cmap=cmap)
    return img


# --- C++ helpers for the whole-collection loops -----------------------------
# A getter called from Python costs ~1 us, and a 100% BIB event has millions of
# hits and hit links, so the loops over entire collections run in C++ and hand
# back numpy arrays. Templated on the collection type; _cpp() instantiates them
# for the C++ class of the collection passed in.
_CPP_HELPERS_OK = ROOT.gInterpreter.Declare(r"""
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <unordered_map>
#include <utility>
#include <vector>
#include "podio/ObjectID.h"
#include "edm4hep/Constants.h"
#include "edm4hep/TrackState.h"

namespace ana {

// Unique across collections: collectionID in bits 31-62, index in bits 0-30.
// MergedTrackerHits is a subset collection over the six subdetector hit
// collections, so the index alone is not unique.
inline std::int64_t objKey(const podio::ObjectID& id) {
  return (static_cast<std::int64_t>(id.collectionID) << 31) | static_cast<std::int64_t>(id.index);
}

template <class HitColl>
void fillHits(const HitColl& hits, double* x, double* y, double* z,
              std::int64_t* cellID, std::int64_t* key) {
  std::size_t i = 0;
  for (const auto& h : hits) {
    const auto p = h.getPosition();
    x[i] = p.x;
    y[i] = p.y;
    z[i] = p.z;
    cellID[i] = static_cast<std::int64_t>(h.getCellID());
    key[i] = objKey(h.getObjectID());
    ++i;
  }
}

// objKey of every hit linked to a SimTrackerHit of a |PDG| == pdg particle.
template <class LinkColl>
std::vector<std::int64_t> signalHitKeys(const LinkColl& links, int pdg) {
  std::vector<std::int64_t> keys;
  for (const auto& link : links) {
    const auto sim = link.getTo();
    if (!sim.isAvailable()) continue;
    const auto mcp = sim.getParticle();
    if (mcp.isAvailable() && std::abs(mcp.getPDG()) == pdg) {
      keys.push_back(objKey(link.getFrom().getObjectID()));
    }
  }
  return keys;
}

// Positions in the collection of the |PDG| == pdg particles.
template <class MCColl>
std::vector<int> signalParticles(const MCColl& mcps, int pdg) {
  std::vector<int> pos;
  int i = 0;
  for (const auto& p : mcps) {
    if (std::abs(p.getPDG()) == pdg) pos.push_back(i);
    ++i;
  }
  return pos;
}

// sigma(omega), sigma(d0), sigma(z0) of a track state: square roots of the
// diagonal of its covariance (NaN where the variance is not positive).
inline std::array<double, 3> stateSigmas(const edm4hep::TrackState& ts) {
  const auto sigma = [&ts](edm4hep::TrackParams p) {
    const double var = ts.covMatrix.getValue(p, p);
    return var > 0 ? std::sqrt(var) : std::nan("");
  };
  return {sigma(edm4hep::TrackParams::omega), sigma(edm4hep::TrackParams::d0),
          sigma(edm4hep::TrackParams::z0)};
}

// Per track: hit count (0 without reading the relation when countHits is
// false), chi2, ndf, number of track states, and the parameters of one state:
// the AtIP one when atIP, else state stateIdx (NaN when the track has no such
// state).
template <class TrackColl>
void fillTracks(const TrackColl& tracks, int stateIdx, bool atIP, bool countHits,
                std::int64_t* nhits, double* chi2, std::int64_t* ndf, std::int64_t* nstates,
                double* omega, double* tanLambda, double* phi, double* d0, double* z0) {
  const double nan = std::nan("");
  std::size_t i = 0;
  for (const auto& t : tracks) {
    nhits[i] = countHits ? static_cast<std::int64_t>(t.getTrackerHits().size()) : 0;
    chi2[i] = t.getChi2();
    ndf[i] = t.getNdf();
    const auto states = t.getTrackStates();
    nstates[i] = static_cast<std::int64_t>(states.size());
    int idx = stateIdx;
    if (atIP) {
      idx = -1;
      for (std::size_t k = 0; k < states.size(); ++k) {
        if (states[k].location == edm4hep::TrackState::AtIP) {
          idx = static_cast<int>(k);
          break;
        }
      }
    }
    if (idx >= 0 && states.size() > static_cast<std::size_t>(idx)) {
      const auto& ts = states[idx];
      omega[i] = ts.omega;
      tanLambda[i] = ts.tanLambda;
      phi[i] = ts.phi;
      d0[i] = ts.D0;
      z0[i] = ts.Z0;
    } else {
      omega[i] = tanLambda[i] = phi[i] = d0[i] = z0[i] = nan;
    }
    ++i;
  }
}

// Per track: how many of its hits have an objKey in keys[0..nkeys) (sorted).
template <class TrackColl>
void countKeyedHits(const TrackColl& tracks, const std::int64_t* keys, std::size_t nkeys,
                    std::int64_t* out) {
  std::size_t i = 0;
  for (const auto& t : tracks) {
    std::int64_t n = 0;
    for (const auto& h : t.getTrackerHits()) {
      if (std::binary_search(keys, keys + nkeys, objKey(h.getObjectID()))) ++n;
    }
    out[i++] = n;
  }
}

// Per track: hits in the pixel (systems 1-2), inner (3-4) and outer (5-6) trackers.
template <class TrackColl>
void subsystemHitCounts(const TrackColl& tracks, std::int64_t* pixel, std::int64_t* inner, std::int64_t* outer) {
  std::size_t i = 0;
  for (const auto& t : tracks) {
    pixel[i] = inner[i] = outer[i] = 0;
    for (const auto& h : t.getTrackerHits()) {
      const auto sys = h.getCellID() & 0x1F;
      if (sys == 1 || sys == 2) ++pixel[i];
      else if (sys == 3 || sys == 4) ++inner[i];
      else if (sys == 5 || sys == 6) ++outer[i];
    }
    ++i;
  }
}

// Per seed, from its 3 hits ordered by radius, the two quantities the ACTS seeder
// cuts on: |transverse impact parameter| of the circle through the hits
// (SeedFinding_ImpactMax), and the r = 0 intercept zOrigin = zM - rM*dz/dr of the
// inner (bottom-middle) and outer (middle-top) doublets (SeedFinding_CollisionRegion).
// NaN when a seed does not have 3 hits at increasing radii.
template <class TrackColl>
void seedImpact(const TrackColl& seeds, double* d0, double* z0In, double* z0Out) {
  const double nan = std::nan("");
  std::size_t i = 0;
  for (const auto& s : seeds) {
    d0[i] = z0In[i] = z0Out[i] = nan;
    std::vector<std::array<double, 4>> p;  // x, y, z, r
    for (const auto& h : s.getTrackerHits()) {
      const auto q = h.getPosition();
      p.push_back({q.x, q.y, q.z, std::hypot(q.x, q.y)});
    }
    if (p.size() == 3) {
      std::sort(p.begin(), p.end(), [](const auto& a, const auto& b) { return a[3] < b[3]; });
      if (p[1][3] > p[0][3] && p[2][3] > p[1][3]) {
        z0In[i] = p[1][2] - p[1][3] * (p[1][2] - p[0][2]) / (p[1][3] - p[0][3]);
        z0Out[i] = p[1][2] - p[1][3] * (p[2][2] - p[1][2]) / (p[2][3] - p[1][3]);
      }
      const double ax = p[0][0], ay = p[0][1], bx = p[1][0], by = p[1][1], cx = p[2][0], cy = p[2][1];
      const double D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
      if (D != 0) {
        const double a2 = ax * ax + ay * ay, b2 = bx * bx + by * by, c2 = cx * cx + cy * cy;
        const double ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / D;
        const double uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / D;
        d0[i] = std::abs(std::hypot(ux, uy) - std::hypot(ax - ux, ay - uy));
      }
    }
    ++i;
  }
}

// Per seed, its layer combination: the (system << 6 | layer) code of each of its
// 3 hits, ordered by radius, packed 12 bits each (inner hit highest). -1 unless
// the seed has 3 hits.
template <class TrackColl>
void seedLayers(const TrackColl& seeds, std::int64_t* out) {
  std::size_t i = 0;
  for (const auto& s : seeds) {
    std::vector<std::pair<double, std::int64_t>> h;  // r, code
    for (const auto& hit : s.getTrackerHits()) {
      const auto q = hit.getPosition();
      const auto id = static_cast<std::int64_t>(hit.getCellID());
      h.push_back({std::hypot(q.x, q.y), ((id & 0x1F) << 6) | ((id >> 7) & 0x3F)});
    }
    out[i] = -1;
    if (h.size() == 3) {
      std::sort(h.begin(), h.end());
      out[i] = (h[0].second << 24) | (h[1].second << 12) | h[2].second;
    }
    ++i;
  }
}

// Per track: 1 if more than frac of its hits are on one other track ranked above
// it (more hits, then lower chi2, then earlier in the collection), else 0.
template <class TrackColl>
void flagDuplicates(const TrackColl& tracks, double frac, std::int64_t* out) {
  std::vector<std::vector<std::int64_t>> keys;
  std::vector<double> chi2;
  for (const auto& t : tracks) {
    std::vector<std::int64_t> k;
    for (const auto& h : t.getTrackerHits()) k.push_back(objKey(h.getObjectID()));
    keys.push_back(std::move(k));
    chi2.push_back(t.getChi2());
  }
  const std::size_t n = keys.size();
  auto above = [&](std::size_t j, std::size_t i) {
    if (keys[j].size() != keys[i].size()) return keys[j].size() > keys[i].size();
    if (chi2[j] != chi2[i]) return chi2[j] < chi2[i];
    return j < i;
  };
  std::unordered_map<std::int64_t, std::vector<std::size_t>> byHit;
  for (std::size_t i = 0; i < n; ++i)
    for (auto k : keys[i]) byHit[k].push_back(i);
  std::unordered_map<std::size_t, std::size_t> shared;
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = 0;
    shared.clear();
    for (auto k : keys[i])
      for (auto j : byHit[k])
        if (j != i) ++shared[j];
    for (const auto& [j, c] : shared) {
      if (c > frac * keys[i].size() && above(j, i)) {
        out[i] = 1;
        break;
      }
    }
  }
}

}  // namespace ana
""")
if not _CPP_HELPERS_OK:
    raise RuntimeError("the ana:: C++ helpers failed to compile (see the cling errors above)")

_CPP_FUNCS = {}


def _cpp(name, coll):
    """ana::<name>, instantiated for the C++ type of collection `coll`."""
    tname = getattr(type(coll), "__cpp_name__", None)
    fn = _CPP_FUNCS.get((name, tname))
    if fn is None:
        fn = getattr(ROOT.ana, name)
        if tname is not None:
            try:
                fn = fn[tname]
            except Exception:
                pass  # let cppyy deduce the type from the call's arguments
        _CPP_FUNCS[(name, tname)] = fn
    return fn


def obj_key(obj):
    """Python twin of ana::objKey."""
    oid = obj.getObjectID()
    return (int(oid.collectionID) << 31) | int(oid.index)


def hit_arrays(hits):
    """(x, y, z, cellID, obj_key) arrays over every hit in `hits`."""
    n = len(hits)
    x, y, z = np.empty(n), np.empty(n), np.empty(n)
    cell_id, key = np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64)
    if n:
        _cpp("fillHits", hits)(hits, x, y, z, cell_id, key)
    return x, y, z, cell_id, key


def signal_hit_keys(links):
    """obj_key()s of the hits linked to a signal-particle SimTrackerHit."""
    if len(links) == 0:
        return set()
    return {int(k) for k in _cpp("signalHitKeys", links)(links, SIGNAL_PDG)}


def signal_particles(mcps):
    """Positions in `mcps` of the signal particles, in collection order."""
    if len(mcps) == 0:
        return []
    return [int(p) for p in _cpp("signalParticles", mcps)(mcps, SIGNAL_PDG)]


def track_arrays(tracks, state_idx=-1, at_ip=False):
    """Per-track arrays from ana::fillTracks. `nhits` and `has_hits` follow
    track_nhits() / track_has_hits(). The state parameters are read from the
    AtIP state when at_ip, else from state state_idx (-1: none)."""
    n = len(tracks)
    nhits, ndf, nstates = (np.zeros(n, dtype=np.int64) for _ in range(3))
    chi2, omega, tan_lambda, phi, d0, z0 = (np.empty(n) for _ in range(6))
    if n:
        _cpp("fillTracks", tracks)(tracks, state_idx, at_ip, not NO_HITS or NHITS_SOURCE == "trackerHits",
                                   nhits, chi2, ndf, nstates, omega, tan_lambda, phi, d0, z0)
    has_hits = np.ones(n, dtype=bool) if NO_HITS and NHITS_SOURCE == "unavailable" else nhits > 0
    return dict(nhits=nhits, has_hits=has_hits, chi2=chi2, ndf=ndf, nstates=nstates,
                omega=omega, tanLambda=tan_lambda, phi=phi, d0=d0, z0=z0)


def keyed_hit_counts(tracks, keys):
    """Per track, how many of its hits have an obj_key() in the set `keys`."""
    out = np.zeros(len(tracks), dtype=np.int64)
    if len(out) and keys:
        k = np.sort(np.fromiter(keys, dtype=np.int64, count=len(keys)))
        _cpp("countKeyedHits", tracks)(tracks, k, len(k), out)
    return out


def subsystem_hit_counts(tracks):
    """Per track: (pixel, inner, outer) hit counts (ana::subsystemHitCounts)."""
    n = len(tracks)
    pixel, inner, outer = (np.zeros(n, dtype=np.int64) for _ in range(3))
    if n:
        _cpp("subsystemHitCounts", tracks)(tracks, pixel, inner, outer)
    return pixel, inner, outer


def seed_impact(seeds):
    """Per seed: |d0| of the circle through its hits and max |zOrigin| of its two
    doublets, i.e. what SeedFinding_ImpactMax / SeedFinding_CollisionRegion cut on
    (ana::seedImpact)."""
    n = len(seeds)
    d0, z0_in, z0_out = (np.full(n, np.nan) for _ in range(3))
    if n:
        _cpp("seedImpact", seeds)(seeds, d0, z0_in, z0_out)
    return d0, np.maximum(np.abs(z0_in), np.abs(z0_out))


def seed_layers(seeds):
    """Per seed, its packed layer combination (ana::seedLayers), -1 without 3 hits."""
    out = np.full(len(seeds), -1, dtype=np.int64)
    if len(seeds):
        _cpp("seedLayers", seeds)(seeds, out)
    return out


_SYS_SHORT = {1: "VXDB", 2: "VXDE", 3: "ITB", 4: "ITE", 5: "OTB", 6: "OTE"}


def seed_layers_label(combo):
    """'VXDB0-VXDB2-VXDB4' for a packed seed_layers() code, inner hit first."""
    codes = [(combo >> s) & 0xFFF for s in (24, 12, 0)]
    return "-".join(f"{_SYS_SHORT.get(c >> 6, c >> 6)}{c & 0x3F}" for c in codes)


# Layer-combination plots show the combinations with the most seeds per event.
SEED_LAYERS_TOP_N = 25


def seed_layer_rates(combos, n_events):
    """{packed combination: seeds per event}, for the seeds with 3 hits."""
    v = np.asarray(combos, dtype=np.int64)
    u, n = np.unique(v[v >= 0], return_counts=True)
    return dict(zip(u.tolist(), (n / max(n_events, 1)).tolist()))


def print_seed_layer_table(combos, n_events, name, outpath, n_with_state=None):
    """Every layer combination with its seeds, seeds / event, fraction and
    cumulative fraction, plus seeds without 3 hits, to stdout and outpath. The
    rows sum to the number of seeds in the collection; n_with_state (seeds with
    a track state, what the seed multiplicity plots count) is compared too."""
    v = np.asarray(combos, dtype=np.int64)
    u, n = np.unique(v[v >= 0], return_counts=True)
    order = np.argsort(-n, kind="stable")
    ne = max(n_events, 1)
    total = len(v)
    lines = [f"Seeds by layer combination ({name}): {total} seeds in {n_events} events "
             f"= {total / ne:.6g} / event",
             f"{'combination':<24} {'seeds':>12} {'per event':>12} {'fraction':>9} {'cumul.':>9}"]
    cum = 0
    for c, k in zip(u[order].tolist(), n[order].tolist()):
        cum += k
        lines.append(f"{seed_layers_label(c):<24} {k:>12d} {k / ne:>12.6g} {k / max(total, 1):>9.4%} {cum / max(total, 1):>9.4%}")
    n_bad = int((v < 0).sum())
    lines.append(f"{'(not 3 hits)':<24} {n_bad:>12d} {n_bad / ne:>12.6g} {n_bad / max(total, 1):>9.4%}")
    s = int(n.sum()) + n_bad
    lines.append(f"{'SUM':<24} {s:>12d} {s / ne:>12.6g}  " + ("OK, = all seeds" if s == total else f"MISMATCH vs {total}"))
    if n_with_state is not None:
        lines.append(f"Seeds with a track state (seed multiplicity plots): {n_with_state} "
                     f"= {n_with_state / ne:.6g} / event; {total - n_with_state} seeds without one")
    text = "\n".join(lines)
    print(text)
    with open(outpath, "w") as f:
        f.write(text + "\n")


def seed_system_rates(combos, n_events):
    """{system id: seeds / event with >= 1 hit in that system}, from packed
    seed_layers() codes. A seed counts once in each system it touches."""
    v = np.asarray(combos, dtype=np.int64)
    v = v[v >= 0]
    systems = [((v >> s) & 0xFFF) >> 6 for s in (24, 12, 0)]
    ne = max(n_events, 1)
    return {sid: float(((systems[0] == sid) | (systems[1] == sid) | (systems[2] == sid)).sum()) / ne
            for sid in _SYS_SHORT}


def plot_seed_systems(series, title, outpath):
    """Grouped bars of seeds / event with >= 1 hit in each subsystem.
    series: (label, color, {system id: rate})."""
    sids = list(_SYS_SHORT)
    x = np.arange(len(sids))
    w = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, (label, color, r) in enumerate(series):
        vals = [r.get(s, 0) for s in sids]
        ax.bar(x + (k - (len(series) - 1) / 2) * w, vals, width=w, color=color, label=label)
        print(f"{title} [{label}]: " + ", ".join(f"{_SYS_SHORT[s]}={v:.6g}" for s, v in zip(sids, vals)))
    ax.set_xticks(x)
    ax.set_xticklabels([_SYS_SHORT[s] for s in sids])
    ax.set_yscale('log')
    ax.set_ylabel("Seeds / event with $\\geq$ 1 hit in subsystem")
    ax.set_title(title)
    ax.grid(True, axis='y', linestyle=":", alpha=0.7)
    ax.legend(fontsize=7)
    fig.tight_layout()
    plt.savefig(outpath)
    plt.close(fig)


def plot_seed_layers(series, title, outpath):
    """Horizontal bars of seeds per event per layer combination, for the
    SEED_LAYERS_TOP_N combinations with the highest rate in any series.
    series: (label, color, {combination: rate})."""
    top = sorted({c for _, _, r in series for c in r},
                 key=lambda c: -max(r.get(c, 0) for _, _, r in series))[:SEED_LAYERS_TOP_N]
    if not top:
        return
    fig, ax = plt.subplots(figsize=(10, 0.35 * len(top) * max(1, len(series) / 2) + 2))
    h = 0.8 / len(series)
    y = np.arange(len(top))
    for k, (label, color, r) in enumerate(series):
        ax.barh(y + (k - (len(series) - 1) / 2) * h, [r.get(c, 0) for c in top], height=h,
                color=color, label=label)
    # One band per combination: alternating shading, a solid line between bands.
    for k in range(len(top)):
        if k % 2:
            ax.axhspan(k - 0.5, k + 0.5, color='0.93', zorder=0)
        ax.axhline(k + 0.5, color='black', linewidth=0.8)
    ax.set_ylim(-0.5, len(top) - 0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([seed_layers_label(c) for c in top], fontsize=7)
    ax.tick_params(axis='y', which='minor', left=False)
    ax.invert_yaxis()
    ax.set_xscale('log')
    ax.set_xlabel("Seeds / event")
    ax.set_title(title)
    ax.grid(True, axis='x', linestyle=":", alpha=0.7)
    ax.legend(fontsize=7)
    fig.tight_layout()
    plt.savefig(outpath)
    plt.close(fig)


def duplicate_flags(tracks):
    """Per track, whether it is a duplicate (see DUP_SHARED_FRAC), from ana::flagDuplicates."""
    out = np.zeros(len(tracks), dtype=np.int64)
    if len(out):
        _cpp("flagDuplicates", tracks)(tracks, DUP_SHARED_FRAC, out)
    return out.astype(bool)


TRACKSTATE_AT_IP = 1   # edm4hep::TrackState::AtIP


def print_alltracks(tag, evt, mcps, tracks, links, signal_keys):
    """PRINT_ALLTRACKS dump of one event: the signal particles, then one line per
    AllTracks track. pT/eta/phi/d0/z0 come from the state whose location is AtIP
    (its index in the state list is printed), not from a fixed state index."""
    print(f"[AllTracks] {tag} evt={evt}: {len(tracks)} tracks")
    for pos in signal_particles(mcps):
        p = mcps[pos].getMomentum()
        v = ROOT.TVector3(p[0], p[1], p[2])
        print(f"[AllTracks]   truth mcp#{pos} pdg={mcps[pos].getPDG()} "
              f"pT={v.Perp():.3f} eta={v.Eta():+.3f} phi={v.Phi():+.3f}")
    truth = {}   # track obj_key -> (MCParticle index, PDG, link weight)
    for link in links:
        mcp = link.getTo()
        truth[obj_key(link.getFrom())] = (mcp.getObjectID().index, mcp.getPDG(), link.getWeight())
    n_mu = keyed_hit_counts(tracks, signal_keys) if signal_keys else None
    for ti, t in enumerate(tracks):
        chi2, ndf = t.getChi2(), t.getNdf()
        states = t.getTrackStates()
        line = (f"[AllTracks]   trk#{ti:<3d} nhits={track_nhits(t):>2d} nholes={t.getNholes()} "
                f"chi2={chi2:7.2f} ndf={ndf:>2d} chi2/ndf={chi2 / ndf if ndf > 0 else float('nan'):6.2f} "
                f"nstates={len(states)}")
        if not NO_HITS:
            sys_ids = [int(h.getCellID()) & 0x1F for h in t.getTrackerHits()]
            line += (f" VXD/IT/OT={sum(s in (1, 2) for s in sys_ids)}/"
                     f"{sum(s in (3, 4) for s in sys_ids)}/{sum(s in (5, 6) for s in sys_ids)}")
        if n_mu is not None:
            line += f" muHits={n_mu[ti]}"
        ip = next((k for k, s in enumerate(states) if s.location == TRACKSTATE_AT_IP), None)
        if ip is None:
            line += " | no AtIP state"
        else:
            s = states[ip]
            pt = 0.2998 * BField / abs(s.omega * 1000.) if s.omega != 0 else float("inf")
            line += (f" | pT={pt:.3f} eta={asinh(s.tanLambda):+.3f} phi={s.phi:+.3f} "
                     f"d0={s.D0:+.4f} z0={s.Z0:+.4f}")
        m = truth.get(obj_key(t))
        line += f" | truth mcp#{m[0]} pdg={m[1]} w={m[2]:.2f}" if m else " | no truth link"
        print(line)


def ip_state(track):
    """The track's AtIP state, or None. Found by location: ACTS's EDM4hep
    converter writes it first, then one state per hit, inside-out."""
    for s in track.getTrackStates():
        if s.location == TRACKSTATE_AT_IP:
            return s
    return None


def ip_residuals(ts, truth):
    """Residuals and pulls of an AtIP state against truth = (pT, omega, d0, z0)
    of the matched particle. sigma(omega/d0/z0) are the state's covariance
    diagonal; sigma(pT) = pT * sigma(omega) / |omega|, since pT ~ 1/|omega|.
    A pull is NaN when its sigma is missing."""
    pt_true, omega_true, d0_true, z0_true = truth
    pt = 0.2998 * BField / abs(ts.omega * 1000.)
    sig = ROOT.ana.stateSigmas(ts)
    sig_omega, sig_d0, sig_z0 = sig[0], sig[1], sig[2]
    return dict(
        pt_resolution=(pt - pt_true) / pt_true,
        d0_residual=ts.D0 - d0_true,
        z0_residual=ts.Z0 - z0_true,
        pt_pull=(pt - pt_true) / (pt * sig_omega / abs(ts.omega)),
        omega_pull=(ts.omega - omega_true) / sig_omega,
        d0_pull=(ts.D0 - d0_true) / sig_d0,
        z0_pull=(ts.Z0 - z0_true) / sig_z0,
    )


# Residual / pull histograms of the muon-matched tracks, from ip_residuals():
# (key, plot file name, title, x label, bins, is a pull).
RESIDUAL_PLOTS = [
    ("pt_resolution", "pt_residual", r"$p_T$ Residual",
     r"$(p_T^{\mathrm{reco}} - p_T^{\mathrm{truth}}) / p_T^{\mathrm{truth}}$", np.linspace(-1, 1, 41), False),
    ("d0_residual", "d0_residual", r"$d_0$ Residual",
     r"$d_0^{\mathrm{reco}} - d_0^{\mathrm{truth}}$ [mm]", np.linspace(-0.05, 0.05, 41), False),
    ("z0_residual", "z0_residual", r"$z_0$ Residual",
     r"$z_0^{\mathrm{reco}} - z_0^{\mathrm{truth}}$ [mm]", np.linspace(-0.05, 0.05, 41), False),
    ("pt_pull", "pt_pull", r"$p_T$ Pull",
     r"$(p_T^{\mathrm{reco}} - p_T^{\mathrm{truth}}) / \sigma(p_T)$", np.linspace(-5, 5, 51), True),
    ("omega_pull", "qoverpt_pull", r"$q/p_T$ Pull",
     r"$(\omega^{\mathrm{reco}} - \omega^{\mathrm{truth}}) / \sigma(\omega)$,  $\omega \propto q/p_T$",
     np.linspace(-5, 5, 51), True),
    ("d0_pull", "d0_pull", r"$d_0$ Pull",
     r"$(d_0^{\mathrm{reco}} - d_0^{\mathrm{truth}}) / \sigma(d_0)$", np.linspace(-5, 5, 51), True),
    ("z0_pull", "z0_pull", r"$z_0$ Pull",
     r"$(z_0^{\mathrm{reco}} - z_0^{\mathrm{truth}}) / \sigma(z_0)$", np.linspace(-5, 5, 51), True),
]
# Collections they are filled for: (accumulator suffix, plot file suffix, title).
RESIDUAL_STAGES = [
    ("sitracks", "sitracks", "SiTracks"),
    ("alltracks", "alltracks", "AllTracks"),
]


def residual_hist(ax, vals, bins, label, **style):
    """Normalised step histogram of residuals or pulls. Mean and std are over
    every finite value, including the ones outside the axis range."""
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return
    n_out = int(np.sum((vals < bins[0]) | (vals > bins[-1])))
    ax.hist(vals, bins=bins, weights=np.full(len(vals), 1. / len(vals)), histtype='step', linewidth=1.5,
            label=f"{label}  μ={np.mean(vals):.3g} σ={np.std(vals):.3g}"
                  + (f" ({n_out} outside)" if n_out else ""), **style)


def unit_gaussian(ax, bins):
    """N(0, 1) as a fraction per bin, the reference a pull should follow."""
    x = np.linspace(bins[0], bins[-1], 200)
    ax.plot(x, (bins[1] - bins[0]) * np.exp(-x**2 / 2) / np.sqrt(2 * np.pi),
            color="gray", linestyle=":", label="N(0, 1)")

# Each entry: a file, a glob, or a directory whose per-job outputs are chained
# (see expand_input).
#file_names = [
#    "/ospool/uc-shared/project/futurecolliders/larsonma/muon_pT2_1000GeV_theta88_92/reco_fullBIB_uncone_VXD_InOut_TwoWay",
#    "/ospool/uc-shared/project/futurecolliders/larsonma/muon_pT2_1000GeV_theta88_92/reco_fullBIB_uncone_IT_OutIn",
#    "/ospool/uc-shared/project/futurecolliders/larsonma/muon_pT2_1000GeV_theta88_92/reco_fullBIB_uncone_OT_OutIn",
#    "/ospool/uc-shared/project/futurecolliders/larsonma/muon_pT2_1000GeV_theta88_92/reco_fullBIB_uncone_extended_timing_VXD_InOut_TwoWay",
#    "/ospool/uc-shared/project/futurecolliders/larsonma/muon_pT2_1000GeV_theta88_92/reco_fullBIB_uncone_extended_timing_IT_OutIn",
#    "/ospool/uc-shared/project/futurecolliders/larsonma/muon_pT2_1000GeV_theta88_92/reco_fullBIB_uncone_extended_timing_OT_OutIn",
#]
# Bare file names: run from this directory.
file_names = [
    #"muon_pT0_50GeV_theta10_170_NoBIB_uncone_VXDNominal_InOut_reco.edm4hep.root",
    #"muon_pT0_50GeV_theta10_170_NoBIB_uncone_VXDNominal_InOut_TwoWay_reco.edm4hep.root",
    #"muon_pT0_50GeV_theta10_170_NoBIB_uncone_VXDNominal_InOut_TwoWay_noinflate_reco.edm4hep.root",
    #"muon_pT0_50GeV_theta10_170_NoBIB_uncone_OutInMinimalSeeding_TwoWay_reco.edm4hep.root",
    #"muon_pT0_50GeV_theta10_170_NoBIB_uncone_OutInMinimalSeeding_TwoWay_noinflate_reco.edm4hep.root",
    #"muon_pT0_50GeV_theta10_170_NoBIB_uncone_OutInMinimalSeeding_TwoWay_NoBrStop_reco.edm4hep.root",
    #"muon_pT0_50GeV_theta10_170_NoBIB_uncone_OutInMinimalSeeding_v2_TwoWay_reco.edm4hep.root",
    #"muon_pT0_50GeV_theta10_170_NoBIB_uncone_ITOTDisks_OutIn_reco.edm4hep.root",
    #"muon_pT0_50GeV_theta10_170_NoBIB_uncone_VXDNominal_uvfix_reco.edm4hep.root",


    #"muon_pT0_50GeV_theta10_170_NoBIB_uncone_ITOTDisks_OutIn_uvfix_reco.edm4hep.root",
    #"muon_pT0_50GeV_theta10_170_NoBIB_uncone_ITOTDisks_Minimal_v3_reco.edm4hep.root",
    #"muon_pT0_50GeV_theta10_170_NoBIB_uncone_OutInMinimalSeeding_v2_TwoWay_reco.edm4hep.root",
    #"muon_pT0_50GeV_theta10_170_NoBIB_uncone_ITOTDisks_Minimal_v4_reco.edm4hep.root",
    #"muon_pT0_50GeV_theta10_170_NoBIB_uncone_ITOTDisks_Minimal_v2_d0max10_CR15_reco.edm4hep.root",
    #"muon_pT0_50GeV_theta10_170_NoBIB_uncone_ITOTDisks_Minimal_v2_d0max20_CR30_reco.edm4hep.root",
    #"muon_pT0_50GeV_theta10_170_NoBIB_uncone_ITOTDisks_Minimal_v2_d0max30_CR45_reco.edm4hep.root",
    #"muon_pT0_50GeV_theta10_170_NoBIB_uncone_ITOTDisks_Minimal_v3_d0max10_CR15_reco.edm4hep.root",
    #"muon_pT0_50GeV_theta10_170_NoBIB_uncone_ITOTDisks_Minimal_v3_d0max20_CR30_reco.edm4hep.root",
    #"muon_pT0_50GeV_theta10_170_NoBIB_uncone_ITOTDisks_Minimal_v3_d0max30_CR45_reco.edm4hep.root",
    #"muon_pT0_50GeV_theta10_170_NoBIB_uncone_ITOTDisks_Minimal_v4_d0max10_CR15_reco.edm4hep.root",
    #"muon_pT0_50GeV_theta10_170_NoBIB_uncone_ITOTDisks_Minimal_v4_d0max20_CR30_reco.edm4hep.root",
    #"muon_pT0_50GeV_theta10_170_NoBIB_uncone_ITOTDisks_Minimal_v4_d0max30_CR45_reco.edm4hep.root",



    #"muon_pT0_50GeV_theta10_170_NoBIB_uncone_ITOTDisks_Minimal_d0max100_CR150_reco.edm4hep.root",
    #"muon_pT0_50GeV_theta10_170_NoBIB_uncone_ITOTDisks_Minimal_v2_d0max100_CR150_reco.edm4hep.root",
    #"muon_pT0_50GeV_theta10_170_NoBIB_uncone_ITOTDisks_Minimal_v3_d0max100_CR150_reco.edm4hep.root"

    #"/ospool/uc-shared/project/futurecolliders/larsonma/muon_pT0_50GeV_theta10_170/reco_fullBIB_uncone_rawtime_VXDNominal_InOut",
    #"/ospool/uc-shared/project/futurecolliders/larsonma/muon_pT0_50GeV_theta10_170/reco_fullBIB_uncone_rawtime_ITOTDisks_Minimal_v3_OutIn",
    #"/ospool/uc-shared/project/futurecolliders/larsonma/muon_pT0_50GeV_theta10_170/reco_fullBIB_uncone_rawtime_ITOTDisks_OutIn",
    #"/ospool/uc-shared/project/futurecolliders/larsonma/muon_pT0_50GeV_theta10_170/reco_fullBIB_uncone_rawtime_ITOTDisks_Minimal_v2_OutIn",
    #"/ospool/uc-shared/project/futurecolliders/larsonma/muon_pT0_50GeV_theta10_170/reco_fullBIB_uncone_rawtime_VXDNominal_InOut_SeedDedupShared2",
    #"/ospool/uc-shared/project/futurecolliders/larsonma/muon_pT0_50GeV_theta10_170/reco_fullBIB_uncone_rawtime_VXDNominal_InOut_SeedDedup",
    #"/ospool/uc-shared/project/futurecolliders/larsonma/muon_pT0_50GeV_theta10_170/reco_fullBIB_uncone_rawtime_ITOTDisks_Minimal_v2_OutIn_SeedDedupShared2",
    #"/ospool/uc-shared/project/futurecolliders/larsonma/muon_pT0_50GeV_theta10_170/reco_fullBIB_uncone_rawtime_ITOTDisks_Minimal_v2_OutIn_SeedDedup",


    #"/ospool/uc-shared/project/futurecolliders/larsonma/muon_pT0_50GeV_theta10_170/reco_fullBIB_uncone_rawtime_ITOTDisks_Minimal_v2_OutIn_SeedDedup",
    #"/ospool/uc-shared/project/futurecolliders/larsonma/muon_pT0_50GeV_theta10_170/reco_fullBIB_uncone_rawtime_ITOTDisks_Minimal_v2_OutIn_DZ400_DRTop650_SpM0_Compat200",
    #"/ospool/uc-shared/project/futurecolliders/larsonma/muon_pT0_50GeV_theta10_170/reco_fullBIB_uncone_rawtime_ITOTDisks_Minimal_v2_OutIn_DZ400_DRTop650_SpM0_Compat1",
    #"/ospool/uc-shared/project/futurecolliders/larsonma/muon_pT0_50GeV_theta10_170/reco_fullBIB_uncone_rawtime_ITOTDisks_Minimal_v2_OutIn_DZ400_SpM0",
    #"/ospool/uc-shared/project/futurecolliders/larsonma/muon_pT0_50GeV_theta10_170/reco_fullBIB_uncone_rawtime_ITOTDisks_Minimal_v2_OutIn_DZ400_DRTop650_Compat200",
   # "/ospool/uc-shared/project/futurecolliders/larsonma/muon_pT0_50GeV_theta10_170/reco_fullBIB_uncone_rawtime_ITOTDisks_Minimal_v2_OutIn_DZ400_DRTop650",

    "/ospool/uc-shared/project/futurecolliders/larsonma/muon_pT0_50GeV_theta10_170/reco_fullBIB_coned_rawtime_VXDNominal_InOut_SeedDedup",
    "/ospool/uc-shared/project/futurecolliders/larsonma/muon_pT0_50GeV_theta10_170/reco_fullBIB_coned_rawtime_VXDNominal_InOut_TwoWay_SeedDedup",
    "/ospool/uc-shared/project/futurecolliders/larsonma/muon_pT0_50GeV_theta10_170/reco_fullBIB_coned_rawtime_ITOTDisks_Minimal_v2_OutIn_SeedDedup",

]

OVERLAY_DIR = "SeedMult_Efficiencies"
#OVERLAY_LABELS = ["ITOT Min v2", "ITOT Min v2 DZ400, DRTop650, SpM0, Compat200", "ITOT Min v2 DZ400, DRTop650, SpM0, Compat1", "ITOT Min v2 DZ400, SpM0", "ITOT Min v2 DZ400, DRTop650, SpM1, Compat200", "ITOT Min v2 DZ400, DRTop650, SpM1, Compat1"]
OVERLAY_LABELS = ["VXD Nominal In-Out", "VXD Nominal In-Out Two-Way", "ITOTDisks_Minimal_v2 Out-In SeedDedup"]
#OVERLAY_LABELS = ["Full IT+OT Seeding, Two-Way,uvfix, Inflate100", "OutInMinimal_v3 Seeding, Seed: d0max 3, CR 6", "OutInMinimal_v2 Seeding, Seed: d0max 3, CR 6", "OutInMinimal_v4 Seeding, Seed: d0max 3, CR 6", "OutInMinimal_v2 Seeding, Seed: d0max 10, CR 15", "OutInMinimal_v2 Seeding, Seed: d0max 20, CR 30", "OutInMinimal_v2 Seeding, Seed: d0max 30, CR 45", "OutInMinimal_v3 Seeding, Seed: d0max 10, CR 15", "OutInMinimal_v3 Seeding, Seed: d0max 20, CR 30", "OutInMinimal_v3 Seeding, Seed: d0max 30, CR 45", "OutInMinimal_v4 Seeding, Seed: d0max 10, CR 15", "OutInMinimal_v4 Seeding, Seed: d0max 20, CR 30", "OutInMinimal_v4 Seeding, Seed: d0max 30, CR 45"]
#OVERLAY_LABELS = ["Minimal", "Minimal_v2", "Minimal_v3"]
#OVERLAY_LABELS = ["ITOTDisks_Minimal_v3, d0 300 mm, CR 500 mm"]

# Example:
#OVERLAY_LABELS = ["VXD Barrel Seeding", "IT Barrel Seeding", "OT Barrel Seeding"]
#OVERLAY_LABELS = ["VXD Barrel Seeding, Inside Out",  "IT Barrel Seeding, Inside-Out", "IT Barrel Seeding, Outside-In", "OT Barrel Seeding, Outside-In"]
#OVERLAY_LABELS = ["VXD Seeding, NOMINAL CONFIG NO CONING", "OT Outside-In", "OT Out-In No Br Stop No Calo Extrap"]
#OVERLAY_LABELS = ["VXD Barrel Seeding", "IT Barrel Seeding", "OT Barrel Seeding"]
#OVERLAY_LABELS = ["VXDNominal One-Way", "VXDNominal, Two-Way, Inflate100", "VXD Nominal, Two-Way, NoInflate", "OutInMinimal Seeding, Two-Way, Inflate100", "OutInMinimal Seeding, Two-Way, NoInflate", "OutInMinimal Seeding, Two-Way, NoBrStop, Inflate100", "OutInMinimal_v2 Seeding, Two-Way, NoBrStop, Inflate100", "Full IT+OT Seeding, Two-Way, Inflate100", "Full IT+OT Seeding, Two-Way,uvfix, Inflate100"]
#OVERLAY_LABELS = ["VXDNominal One-Way", "VXDNominal, Two-Way, Inflate100", "OutInMinimal_v2 Seeding, Two-Way, Inflate100", "Full IT+OT Seeding, Two-Way, Inflate100", "VXDNominal, Two-Way, uv fix, Inflate100", "Full IT+OT Seeding, Two-Way,uvfix, Inflate100", "OutInMinimal_v3 Seeding, Two-Way, Inflate100"]
#                  "VXD Barrel Seeding, Ext. Timing", "IT Barrel Seeding, Ext. Timing", "OT Barrel Seeding, Ext. Timing",  ]
#OVERLAY_LABELS = ["VXD Barrel, InOut", "IT Barrel, OutIn", "OT Barrel, OutIn"]
#
#OVERLAY_LABELS = ["OT Seeding, doOutsideInCKF", "OT Seeding, propagateBackward", "OT Seeding, Seed Direc Fix doOutsideInCKF"]

# Two-way NHits printout + vs-eta plot (twoway_nhits_compare): file_names[0]
# must be the one-way reco and the others two-way reco of the same events.
# Only for such a set.
TWOWAY_NHITS_COMPARE = False
# Per-file plot directory / legend name, from the directory the file sits in.
PLOT_TAGS = make_plot_tags(file_names)


# --- Event loop --------------------------------------------------------------
# Each input file is read by one worker process (process_file); the per-file
# results of an entry are combined by merge_results() and drawn by
# plot_entry(). Every worker re-imports this file, so module-level code must
# stay cheap: the work happens under main().

# Tracker r-z occupancy histogram. Bins are sized to resolve individual layers
# and to keep VXD / IT / OT regions distinguishable. The histogram accumulates
# raw hit counts across all events; divided by event count at plot time.
OCC_Z_EDGES = np.linspace(-2300, 2300, 461)   # 10 mm bins along z
OCC_R_EDGES = np.linspace(0, 1600, 161)       # 10 mm bins along r

# ID verification: a uniform random sample of this many MergedTrackerHits per
# subdetector feeds the cellID scatter; the (system, side, layer) -> count
# table is integrated over all events.
ID_DIAG_TARGET_HITS_PER_SYSTEM = 5000   # ~30k total across the 6 subdetectors

_SYS_NAMES = {
    1: "VXD Barrel", 2: "VXD Endcap",
    3: "IT Barrel",  4: "IT Endcap",
    5: "OT Barrel",  6: "OT Endcap",
}

# Per-event / per-matched-track lists, concatenated over files in file order.
_LIST_KEYS = (
    "pixel_nhits_sitracks", "inner_nhits_sitracks", "outer_nhits_sitracks", "total_nhits_sitracks",
    "pixel_nhits_alltracks", "inner_nhits_alltracks", "outer_nhits_alltracks", "total_nhits_alltracks",
    "track_chi2_reduced_sitracks", "track_chi2_reduced_alltracks",
    "track_eta_sitracks", "track_matched_pt_sitracks", "track_percent_muon_hits_sitracks",
    "track_pt_resolution_sitracks", "track_reco_pt_sitracks", "track_phi_sitracks",
    "track_d0_sitracks", "track_z0_sitracks",
    "track_d0_residual_sitracks", "track_z0_residual_sitracks",
    "track_pt_resolution_alltracks", "track_d0_residual_alltracks", "track_z0_residual_alltracks",
    "track_pt_pull_sitracks", "track_omega_pull_sitracks", "track_d0_pull_sitracks", "track_z0_pull_sitracks",
    "track_pt_pull_alltracks", "track_omega_pull_alltracks", "track_d0_pull_alltracks", "track_z0_pull_alltracks",
    "track_delta_qoverp_sq_sitracks", "num_bib_hits_sitracks",
    "all_tracks_mult_per_event", "deduped_tracks_mult_per_event", "sitracks_mult_per_event",
    "fake_mult_per_event", "dup_mult_per_event",
    "muon_matched_sitracks_per_event",
    "muon_si_nhits_per_event",   # per event: {muon MCP index: (truth pT, truth eta, SiTrack NHits)}
    "eff_truth_eta", "eff_truth_pt", "eff_truth_phi", "eff_truth_theta_deg",
    "eff_truth_d0", "eff_truth_z0", "eff_truth_vrxy", "eff_truth_vz",
    "eff_matched_si", "eff_matched_alltracks",
    "seed_eff_eta", "seed_eff_pt", "seed_eff_matched",
    "seed_nseeds_per_event", "seed_pt_best_per_muon",
    "layer_nhits_per_event",   # per event: {(system, layer): hits}
)
# One value per AllTracks track or per seed (millions on BIB): numpy arrays.
_ARRAY_KEYS = (
    "alltracks_raw_nhits", "alltracks_raw_chi2_reduced", "alltracks_raw_pt",
    "alltracks_raw_d0", "alltracks_raw_z0", "seed_pt_all", "seed_eta_all",
    "fake_pt", "fake_eta", "dup_pt", "dup_eta",
    "seed_d0_all", "seed_zorigin_all", "seed_d0_mu", "seed_zorigin_mu", "seed_mu_truth_pt",
    "seed_layers_all", "seed_layers_mu",
    # per fake (not muon-linked) track: pixel / inner / outer hit counts
    "fake_pixel_nhits_sitracks", "fake_inner_nhits_sitracks", "fake_outer_nhits_sitracks",
    "fake_pixel_nhits_alltracks", "fake_inner_nhits_alltracks", "fake_outer_nhits_alltracks",
)
# Counters, summed over files.
_SUM_KEYS = ("n_events", "matched_tracks_sitracks", "matched_tracks_alltracks",
             "n_seed_hit_matched", "n_seed_dr_matched")


def _bottom_k(old, new, k=ID_DIAG_TARGET_HITS_PER_SYSTEM):
    """Join two (u, r, z, side, layer) samples and keep the k smallest u.

    Every hit carries a uniform random u, so the k smallest u over all hits
    seen are a uniform random sample of k of them, and samples taken
    separately (per event, per file) merge into one.
    """
    if old is not None:
        new = tuple(np.concatenate([a, b]) for a, b in zip(old, new))
    if len(new[0]) <= k:
        return new
    keep = np.argpartition(new[0], k)[:k]
    return tuple(a[keep] for a in new)


def _id_diag_add(acc, rng, r, z, cell_id):
    """cellID (system, side, layer) counts and per-system hit sample for one
    event's merged hits."""
    system = cell_id & 0x1F           # system (subdetector), bits 0-4
    side = (cell_id >> 5) & 0x3       # side (±z), bits 5-6
    layer = cell_layer(cell_id)       # layer, bits 7-12
    counts = np.bincount(cell_id & 0x1FFF, minlength=1 << 13)   # (system, side, layer) bits
    for code in np.nonzero(counts)[0].tolist():
        n = int(counts[code])
        acc["id_diag_counts"][(code & 0x1F, (code >> 5) & 0x3, cell_layer(code))] += n
        acc["id_diag_seen_per_sys"][code & 0x1F] += n
    u = rng.random(len(cell_id))
    sample = acc["id_diag_sample"]
    for s in np.nonzero(np.bincount(system, minlength=32))[0].tolist():
        old = sample.get(s)
        # Once a system's sample is full, only hits below its largest u can enter.
        cut = old[0].max() if old is not None and len(old[0]) >= ID_DIAG_TARGET_HITS_PER_SYSTEM else 1.0
        m = (system == s) & (u < cut)
        if m.any():
            sample[s] = _bottom_k(old, (u[m], r[m], z[m], side[m], layer[m]))


def process_file(task):
    """Event loop over one input file; returns its accumulators.

    Runs in a worker process. NO_HITS / NHITS_SOURCE are set from the file's
    own first event.
    """
    global NO_HITS, NHITS_SOURCE
    fname, file_idx, use_coned = task
    t_start = time.time()
    rng = np.random.default_rng(zlib.crc32(fname.encode()))   # reproducible ID-diag sample

    acc = {k: [] for k in _LIST_KEYS}
    acc.update({k: 0 for k in _SUM_KEYS})
    acc.update(
        fname=fname, use_coned=use_coned, no_hits=None, detected_no_hits=None, nhits_source=None,
        diag_lines=[],
        layer_hits_sitracks={sys_id: defaultdict(int) for sys_id in range(1, 7)},
        id_diag_counts=defaultdict(int),        # (system, side, layer) -> count
        id_diag_seen_per_sys=defaultdict(int),  # system -> total hits seen
        id_diag_sample={},                      # system -> (u, r, z, side, layer), see _bottom_k
        occupancy_hist=np.zeros((len(OCC_Z_EDGES) - 1, len(OCC_R_EDGES) - 1), dtype=np.int64),
        zseam_chunks=[],     # per event: (z, r, system) of the merged hits with |z| < ZSEAM_COLLECT_HALF
        event_displays=[],   # (event index in this file, hit-map data) for draw_event_rz/xy
        all_rz_displays=[],  # same, for draw_all_rz
    )
    raw = {k: [] for k in _ARRAY_KEYS}   # per-event arrays, joined at the end
    zseam_n = 0

    reader = open_reader([fname])
    i = 0
    for event in reader.get("events"):
        if i == 0:
            # Dropped-hit-collection autodetect, per file (see the no-hits
            # mode notes). The env var only forces the mode; the hit-count
            # source is always measured.
            detected, NHITS_SOURCE = _detect_mode(event)
            NO_HITS = detected if _env_no_hits is None else _env_no_hits not in ("0", "false", "False", "")
            acc.update(no_hits=NO_HITS, detected_no_hits=detected, nhits_source=NHITS_SOURCE)

        rz_muon_r = []
        rz_muon_z = []
        rz_bib_r = []
        rz_bib_z = []

        xy_muon_x = []
        xy_muon_y = []
        xy_bib_x = []
        xy_bib_y = []

        hits_in_sitracks_muon_track = set()

        rz_muon_sitracks = []

        chi2_sitracks_list = []

        hitRelations = None if NO_HITS else event.get("MergedTrackerHitsRelations")
        hitCollection = None if NO_HITS else event.get("MergedTrackerHits")

        # Diagnostic: verify subdetector collections are present and non-empty
        # (first events of each entry's first file; printed by finish_entry).
        if file_idx == 0 and i < 3 and not NO_HITS:
            _suffix = "Coned" if use_coned else ""
            for _cname in [
                f"VXDBarrelHits{_suffix}", f"VXDEndcapHits{_suffix}",
                f"ITBarrelHits{_suffix}",  f"ITEndcapHits{_suffix}",
                f"OTBarrelHits{_suffix}",  f"OTEndcapHits{_suffix}",
            ]:
                _col = event.get(_cname)
                acc["diag_lines"].append(f"[DIAG] Event {i} {_cname}: {len(_col) if _col is not None else 'MISSING'} hits")
            acc["diag_lines"].append(f"[DIAG] Event {i} MergedTrackerHits: {len(hitCollection) if hitCollection is not None else 'MISSING'} hits")

        # Position, cellID and obj_key of every merged tracker hit, read in C++
        # (ana::fillHits). Everything that looks at all hits uses these arrays.
        _keys = _cids = None
        _layer_nhits = {}   # (system, layer) -> hits in this event, both endcap sides together
        if hitCollection is not None and len(hitCollection) > 0:
            _xs, _ys, _zs, _cids, _keys = hit_arrays(hitCollection)
            _rs = np.hypot(_xs, _ys)

            _sl = np.bincount((_cids & 0x1F) | (cell_layer(_cids) << 5))
            _layer_nhits = {(c & 0x1F, c >> 5): int(_sl[c]) for c in np.nonzero(_sl)[0].tolist()}

            # Tracker r-z occupancy: bin every merged tracker hit into the global histogram.
            _h2d, _, _ = np.histogram2d(_zs, _rs, bins=[OCC_Z_EDGES, OCC_R_EDGES])
            acc["occupancy_hist"] += _h2d.astype(np.int64)

            # z = 0 seam diagnostic: keep every hit near z = 0, unsampled, with
            # its subdetector so the barrels can be separated at plot time.
            if zseam_n < ZSEAM_MAX_HITS:
                _near = np.abs(_zs) < ZSEAM_COLLECT_HALF
                if _near.any():
                    acc["zseam_chunks"].append((_zs[_near], _rs[_near], _cids[_near] & 0x1F))
                    zseam_n += int(_near.sum())

            # ID verification: decode cellID for every merged tracker hit and accumulate
            if GENERATE_ID_DIAG:
                _id_diag_add(acc, rng, _rs, _zs, _cids)
        if hitCollection is not None:
            acc["layer_nhits_per_event"].append(_layer_nhits)

        # Hits whose SimTrackerHit belongs to a signal particle (C++ over every
        # link). Empty in no-hits mode, where no hit is ever looked up.
        signal_keys = signal_hit_keys(hitRelations) if hitRelations is not None else set()

        mcParticles = event.get("MCParticles")
        trackMCParticlesLinks = event.get("SiTrackRelations")
        allTracksMCParticlesLinks = event.get("AllTracksRelations")
        seedTracks = event.get("SeedTracks")
        allTracks = event.get("AllTracks")
        dedupedTracks = event.get("DedupedTracks")
        siTracks = event.get("SiTracks")

        # Seed stage of the efficiency: a seed with all its hits from the signal
        # muon (single-muon events). Needs the hit links, so unused in no-hits mode.
        _seed_mu_hits = keyed_hit_counts(seedTracks, signal_keys)
        has_muon_seed = bool((_seed_mu_hits >= SEED_MATCH_MIN_HITS).any())

        if i < PRINT_ALLTRACKS:
            print_alltracks(os.path.basename(fname), i, mcParticles, allTracks,
                            allTracksMCParticlesLinks, signal_keys)

        _at = track_arrays(allTracks, at_ip=True)
        acc["all_tracks_mult_per_event"].append(int(_at["has_hits"].sum()))
        acc["deduped_tracks_mult_per_event"].append(int(track_arrays(dedupedTracks, -1)["has_hits"].sum()))

        # Raw AllTracks per-track diagnostics — no MC matching, every CKF output track.
        _sel = _at["has_hits"]
        raw["alltracks_raw_nhits"].append(_at["nhits"][_sel & (_at["nhits"] > 0)])
        _ndf_ok = _sel & (_at["ndf"] > 0)
        raw["alltracks_raw_chi2_reduced"].append(_at["chi2"][_ndf_ok] / _at["ndf"][_ndf_ok])
        _has_state = _sel & ~np.isnan(_at["omega"])
        _omega = _at["omega"][_has_state]
        raw["alltracks_raw_pt"].append(0.2998 * BField / np.abs(_omega[_omega != 0] * 1000.))
        raw["alltracks_raw_d0"].append(_at["d0"][_has_state])
        raw["alltracks_raw_z0"].append(_at["z0"][_has_state])
        _si = track_arrays(siTracks, at_ip=True)
        acc["sitracks_mult_per_event"].append(int(_si["has_hits"].sum()))

        mcp_to_SiTracks = defaultdict(list)
        mcp_to_AllTracks = defaultdict(list)

        # Hit -> SimTrackerHit map, only for the EFF_DEBUG composition dump.
        hit_to_toobj = defaultdict(list)   # hit -> (to, w)
        if EFF_DEBUG:
            for link in (hitRelations or []):
                w = link.getWeight() if hasattr(link, "getWeight") else 1.0
                sim = link.getTo()
                # Coned outputs no longer store the sim hits the links point to.
                if sim.isAvailable():
                    hit_to_toobj[link.getFrom().getObjectID().index].append((sim, w))

        for siTrackLink in trackMCParticlesLinks:
            mcp     = siTrackLink.getTo()
            siTrack = siTrackLink.getFrom()
            w       = siTrackLink.getWeight()

            mcp_id = mcp.getObjectID().index
            mcp_to_SiTracks[mcp_id].append((siTrack, w))

        for allTrackLink in allTracksMCParticlesLinks:
            mcp      = allTrackLink.getTo()
            allTrack = allTrackLink.getFrom()
            w        = allTrackLink.getWeight()

            mcp_id = mcp.getObjectID().index
            mcp_to_AllTracks[mcp_id].append((allTrack, w))

        mcpid = None
        mcp_tlv = None
        SiTrackpT = None
        muon_si_keys = set()   # obj_key()s of the SiTracks linked to a signal muon
        muon_all_idx = set()   # AllTracks positions of the tracks linked to a signal muon
        muon_si_nhits = {}     # signal-muon MCP index -> (truth pT, truth eta, NHits of its SiTrack)
        for _pos in signal_particles(mcParticles):
            mcp = mcParticles[_pos]
            mcp_p = mcp.getMomentum()
            mcp_tlv = ROOT.TLorentzVector()
            mcp_tlv.SetPxPyPzE(mcp_p[0], mcp_p[1], mcp_p[2], mcp.getEnergy())
            mcpid = mcp.getObjectID().index
            muon_si_keys.update(obj_key(t) for t, _ in mcp_to_SiTracks.get(mcpid, []))
            muon_all_idx.update(t.getObjectID().index for t, _ in mcp_to_AllTracks.get(mcpid, []))
            # Truth d0/z0 from the helix through the production vertex, w.r.t.
            # the origin (the CKF's perigee); also the efficiency x-axes.
            d0_truth, z0_truth = truth_d0_z0_helix(mcp, Bz_T=BField)
            # Truth (pT, omega, d0, z0) for ip_residuals(), only if this MCP
            # has a matched track in any selection stage. omega in 1/mm, signed
            # by the charge like the reco one.
            truth = None
            if mcp_to_SiTracks.get(mcpid) or mcp_to_AllTracks.get(mcpid):
                pt_truth = mcp_tlv.Perp()
                truth = (pt_truth, mcp.getCharge() * 0.2998 * BField / (1000. * pt_truth), d0_truth, z0_truth)
            hits_in_sitracks_muon_track = set()
            rz_muon_r = []
            rz_muon_z = []
            rz_bib_r = []
            rz_bib_z = []
            xy_muon_x = []
            xy_muon_y = []
            xy_bib_x = []
            xy_bib_y = []
            rz_muon_sitracks = []
            if len(mcp_to_SiTracks.get(mcpid, [])) == 1:
                acc["matched_tracks_sitracks"] += 1
            if len(mcp_to_AllTracks.get(mcpid, [])) >= 1:
                acc["matched_tracks_alltracks"] += 1
            acc["eff_truth_eta"].append(mcp_tlv.Eta())
            acc["eff_truth_pt"].append(mcp_tlv.Perp())
            acc["eff_truth_phi"].append(mcp_tlv.Phi())
            acc["eff_truth_theta_deg"].append(np.degrees(mcp_tlv.Theta()))
            acc["eff_truth_d0"].append(d0_truth)
            acc["eff_truth_z0"].append(z0_truth)
            acc["eff_truth_vrxy"].append(float(np.hypot(mcp.getVertex()[0], mcp.getVertex()[1])))
            acc["eff_truth_vz"].append(float(mcp.getVertex()[2]))
            acc["eff_matched_si"].append(any(track_has_hits(t) for t, _ in mcp_to_SiTracks.get(mcpid, [])))
            acc["eff_matched_alltracks"].append(any(track_has_hits(t) for t, _ in mcp_to_AllTracks.get(mcpid, [])))
            if not NO_HITS:
                acc["seed_eff_eta"].append(mcp_tlv.Eta())
                acc["seed_eff_pt"].append(mcp_tlv.Perp())
                acc["seed_eff_matched"].append(has_muon_seed)

            # --- [eff-debug] one-line per-event summary + drill-down on unmatched events.
            # nLinksMu is 0 or 1 (TrackTruthAlg emits at most one link per MCP).
            # nHits/nHitRel tell us whether the merger dropped sim-relations wholesale.
            if EFF_DEBUG:
                n_links_mu_all = len(mcp_to_AllTracks.get(mcpid, []))
                n_links_mu_si  = len(mcp_to_SiTracks.get(mcpid, []))
                n_hits         = len(hitCollection) if hitCollection is not None else -1
                n_hit_rels     = len(hitRelations) if hitRelations is not None else -1
                # Per-system hit counts (system id = cellID & 0x1F; 1=VXDB 2=VXDE 3=ITB 4=ITE 5=OTB 6=OTE).
                _per_sys = np.bincount(_cids & 0x1F, minlength=32) if _cids is not None else np.zeros(32, dtype=int)
                _sys_str = f"VXDB={_per_sys[1]} VXDE={_per_sys[2]} ITB={_per_sys[3]} ITE={_per_sys[4]} OTB={_per_sys[5]} OTE={_per_sys[6]}"
                print(f"[eff-debug] {os.path.basename(fname)} evt={i} mcpid={mcpid} "
                      f"nMuMCPs={len(signal_particles(mcParticles))} "
                      f"nSeeds={len(seedTracks)} nAllTracks={len(allTracks)} nLinksMu(All)={n_links_mu_all} "
                      f"nLinksMu(Si)={n_links_mu_si} nHits={n_hits} nHitRel={n_hit_rels} [{_sys_str}]")

                # The composition dump needs the hit -> SimTrackerHit map, so it
                # is only possible when the hit relations were written out.
                if n_links_mu_all == 0 and not NO_HITS:
                    # Rebuild the per-track truth-fraction breakdown that TrackTruthAlg
                    # would have seen — same map, same denominator — so we can tell
                    # WHY no link was made (no sim-relation for the hits, or the muon
                    # simply isn't the plurality on any track, or 'unavailable' MCP).
                    print(f"[eff-debug]   UNMATCHED: dumping AllTracks composition")
                    for _ti, _tk in enumerate(allTracks):
                        _tk_hits = _tk.getTrackerHits()
                        if len(_tk_hits) == 0:
                            continue
                        _n_with_sim = 0
                        _mcp_count = defaultdict(int)  # mcp_index -> hits
                        _mcp_pdg   = {}
                        for _hh in _tk_hits:
                            _sims = hit_to_toobj.get(_hh.getObjectID().index, [])
                            if _sims:
                                _n_with_sim += 1
                            for _sim, _ in _sims:
                                _p = _sim.getParticle()
                                if _p.isAvailable():
                                    _pi = _p.getObjectID().index
                                    _mcp_count[_pi] += 1
                                    _mcp_pdg[_pi] = _p.getPDG()
                        _top = sorted(_mcp_count.items(), key=lambda kv: -kv[1])[:3]
                        _top_str = ", ".join(f"idx={pi}(pdg={_mcp_pdg[pi]}):{c}" for pi, c in _top) or "none"
                        print(f"[eff-debug]     trk#{_ti} nHits={len(_tk_hits)} "
                              f"nHitsWithSim={_n_with_sim} topMCPs=[{_top_str}]")

            # --- AllTracks (muon-matched) chi2/Ndf collection (minimal — just for the chi2 overlay) ---
            for allTrack, _ in mcp_to_AllTracks.get(mcpid, []):
                _at_n = track_nhits(allTrack)
                if track_has_hits(allTrack) and allTrack.getNdf() > 0:
                    acc["track_chi2_reduced_alltracks"].append(allTrack.getChi2() / float(allTrack.getNdf()))
                _ts = ip_state(allTrack)
                if track_has_hits(allTrack) and _ts is not None:
                    for k, v in ip_residuals(_ts, truth).items():
                        acc[f"track_{k}_alltracks"].append(v)
                # Per-track Pixel/Inner/Outer/Total hit counts, same subsystem grouping as SiTracks.
                # The subsystem split needs a cellID per hit, so in no-hits
                # mode only the total is available.
                if NO_HITS:
                    if _at_n > 0:
                        acc["total_nhits_alltracks"].append(_at_n)
                elif _at_n > 0:
                    _at_pixel = _at_inner = _at_outer = 0
                    for _hh in allTrack.getTrackerHits():
                        _sys = int(_hh.getCellID()) & 0x1F
                        if _sys == 1 or _sys == 2:
                            _at_pixel += 1
                        elif _sys == 3 or _sys == 4:
                            _at_inner += 1
                        elif _sys == 5 or _sys == 6:
                            _at_outer += 1
                    acc["pixel_nhits_alltracks"].append(_at_pixel)
                    acc["inner_nhits_alltracks"].append(_at_inner)
                    acc["outer_nhits_alltracks"].append(_at_outer)
                    acc["total_nhits_alltracks"].append(_at_pixel + _at_inner + _at_outer)

            # --- SiTracks ---
            for siTrack, w in mcp_to_SiTracks.get(mcpid, []):
                hit_matched_counter = 0   # must reset per track; n_hits_si below is per-track
                ts = ip_state(siTrack)   # track state documentation: https://edm4hep.web.cern.ch/classedm4hep_1_1_track_state.html
                if ts is None:
                    continue
                omega = ts.omega
                SiTrackpT = 0.2998 * BField / fabs(omega * 1000.)
                chi2_reduced_si = 0
                if(siTrack.getNdf() > 0):
                    chi2_reduced_si = siTrack.getChi2() / float(siTrack.getNdf())
                chi2_sitracks_list.append(chi2_reduced_si)
                theta_si = np.pi/2 - np.arctan(ts.tanLambda)
                eta_si = -np.log(np.tan(theta_si/2))
                phi_si = ts.phi
                d0_si = ts.D0
                z0_si = ts.Z0
                LC_pixel_nhit_si = 0
                LC_inner_nhit_si = 0
                LC_outer_nhit_si = 0
                # Everything in this loop reads hit positions and cellIDs,
                # so it only runs when the hit collection was written out.
                for hit in ([] if NO_HITS else siTrack.getTrackerHits()):
                    position = hit.getPosition()
                    r = sqrt(position[0] ** 2 + position[1] ** 2)
                    hkey = obj_key(hit)
                    hits_in_sitracks_muon_track.add(hkey)
                    # decode system and layer from cellID: "system:5,side:-2,layer:6,module:11,sensor:8"
                    cell_id = int(hit.getCellID())
                    system = cell_id & 0x1F
                    layer_si = cell_layer(cell_id)
                    if 1 <= system <= 6:
                        acc["layer_hits_sitracks"][system][layer_si] += 1
                    if system == 1 or system == 2:
                        LC_pixel_nhit_si += 1
                    elif system == 3 or system == 4:
                        LC_inner_nhit_si += 1
                    elif system == 5 or system == 6:
                        LC_outer_nhit_si += 1
                    if hkey in signal_keys:
                        rz_muon_r.append(r)
                        rz_muon_z.append(position[2])
                        xy_muon_x.append(position[0])
                        xy_muon_y.append(position[1])
                        rz_muon_sitracks.append((r, position[2]))
                        hit_matched_counter += 1
                    else:
                        rz_bib_r.append(r)
                        rz_bib_z.append(position[2])
                        xy_bib_x.append(position[0])
                        xy_bib_y.append(position[1])
                n_hits_si = track_nhits(siTrack)
                if track_has_hits(siTrack):
                    p_magnitude = sqrt(mcp_p[0]**2 + mcp_p[1]**2 + mcp_p[2]**2)
                    charge = mcp.getCharge()
                    truthqoverp = float(charge) / p_magnitude
                    qoverp_si = (ts.omega * sin(theta_si) * 1000.) / (0.2998 * BField)
                    muon_si_nhits[mcpid] = (mcp_tlv.Perp(), mcp_tlv.Eta(), n_hits_si)
                    acc["track_chi2_reduced_sitracks"].append(chi2_reduced_si)
                    acc["track_eta_sitracks"].append(eta_si)
                    acc["track_matched_pt_sitracks"].append(mcp_tlv.Perp())
                    acc["track_reco_pt_sitracks"].append(SiTrackpT)
                    acc["track_phi_sitracks"].append(phi_si)
                    acc["track_d0_sitracks"].append(d0_si)
                    acc["track_z0_sitracks"].append(z0_si)
                    for k, v in ip_residuals(ts, truth).items():
                        acc[f"track_{k}_sitracks"].append(v)
                    acc["track_delta_qoverp_sq_sitracks"].append((qoverp_si - truthqoverp)**2)
                    # Muon/BIB split and the subsystem breakdown both need
                    # the hits themselves; only the total survives without
                    # them, from the trackerHits relation size.
                    if NO_HITS:
                        if n_hits_si > 0:
                            acc["total_nhits_sitracks"].append(n_hits_si)
                    else:
                        acc["track_percent_muon_hits_sitracks"].append(float(hit_matched_counter) / n_hits_si)
                        acc["num_bib_hits_sitracks"].append(n_hits_si - hit_matched_counter)
                        acc["pixel_nhits_sitracks"].append(LC_pixel_nhit_si)
                        acc["inner_nhits_sitracks"].append(LC_inner_nhit_si)
                        acc["outer_nhits_sitracks"].append(LC_outer_nhit_si)
                        acc["total_nhits_sitracks"].append(LC_pixel_nhit_si + LC_inner_nhit_si + LC_outer_nhit_si)

        # Event hit maps, drawn later by plot_entry: hit positions only, so
        # nothing to draw in no-hits mode (hits_in_sitracks_muon_track stays
        # empty there anyway).
        n_si_matched = len(mcp_to_SiTracks.get(mcpid, [])) if mcpid is not None else 0
        want_display = (not NO_HITS and n_si_matched == 1
                        and len(hits_in_sitracks_muon_track) > 0 and len(rz_bib_r) > 0)
        want_all_rz = not NO_HITS and n_si_matched >= 1 and len(acc["all_rz_displays"]) < 10
        if want_display or want_all_rz:
            # Muon hits the track did not pick up.
            _missed = np.zeros(0, dtype=bool)
            if _keys is not None and signal_keys:
                _missed = np.isin(_keys, np.fromiter(signal_keys, dtype=np.int64, count=len(signal_keys)))
                _missed &= ~np.isin(_keys, np.fromiter(hits_in_sitracks_muon_track, dtype=np.int64,
                                                       count=len(hits_in_sitracks_muon_track)))
            disp = dict(
                rz_muon_r=rz_muon_r, rz_muon_z=rz_muon_z, rz_bib_r=rz_bib_r, rz_bib_z=rz_bib_z,
                xy_muon_x=xy_muon_x, xy_muon_y=xy_muon_y, xy_bib_x=xy_bib_x, xy_bib_y=xy_bib_y,
                rz_muon_notintrack_r=_rs[_missed].tolist() if _missed.any() else [],
                rz_muon_notintrack_z=_zs[_missed].tolist() if _missed.any() else [],
                xy_muon_notintrack_x=_xs[_missed].tolist() if _missed.any() else [],
                xy_muon_notintrack_y=_ys[_missed].tolist() if _missed.any() else [],
                rz_muon_sitracks=rz_muon_sitracks, chi2_sitracks_list=chi2_sitracks_list,
                truth_pt=mcp_tlv.Perp() if mcp_tlv is not None else None, reco_pt=SiTrackpT,
            )
            if want_display:
                acc["event_displays"].append((i, disp))
            if want_all_rz:
                acc["all_rz_displays"].append((i, disp))

        if mcpid is not None:
            acc["muon_matched_sitracks_per_event"].append(len(mcp_to_SiTracks.get(mcpid, [])))
        # Every event, so entries can be paired event by event.
        acc["muon_si_nhits_per_event"].append(muon_si_nhits)

        # Fake SiTracks: with hits, but not truth-linked to a signal muon. pT and
        # eta at the IP state, as for the muon-matched SiTracks.
        _si_keys = np.fromiter((obj_key(t) for t in siTracks), dtype=np.int64, count=len(siTracks))
        _fake = _si["has_hits"] & ~np.isin(_si_keys, np.fromiter(muon_si_keys, dtype=np.int64,
                                                                 count=len(muon_si_keys)))
        acc["fake_mult_per_event"].append(int(_fake.sum()))
        # Pixel / inner / outer hits of fake SiTracks and fake AllTracks (AllTracks
        # not linked to a signal muon, picked by position, no per-track Python).
        # Needs the hit cellIDs, so not in no-hits mode.
        if not NO_HITS:
            _at_fake = _at["has_hits"].copy()
            _at_fake[list(muon_all_idx)] = False
            for stage, tracks, sel in (("sitracks", siTracks, _fake), ("alltracks", allTracks, _at_fake)):
                for part, n in zip(("pixel", "inner", "outer"), subsystem_hit_counts(tracks)):
                    raw[f"fake_{part}_nhits_{stage}"].append(n[sel].astype(np.int16))
        _fake &= ~np.isnan(_si["omega"]) & (_si["omega"] != 0)
        raw["fake_pt"].append(0.2998 * BField / np.abs(_si["omega"][_fake] * 1000.))
        raw["fake_eta"].append(-np.log(np.tan((np.pi / 2 - np.arctan(_si["tanLambda"][_fake])) / 2)))

        # Duplicate SiTracks (see DUP_SHARED_FRAC), with pT and eta at the IP
        # state. Needs the hit objects, so not in no-hits mode.
        if not NO_HITS:
            _dup = duplicate_flags(siTracks) & _si["has_hits"]
            acc["dup_mult_per_event"].append(int(_dup.sum()))
            _dup &= ~np.isnan(_si["omega"]) & (_si["omega"] != 0)
            raw["dup_pt"].append(0.2998 * BField / np.abs(_si["omega"][_dup] * 1000.))
            raw["dup_eta"].append(np.arcsinh(_si["tanLambda"][_dup]))

        # Seed track pT and eta of every seed
        _sd = track_arrays(seedTracks, 0)
        _ok = (_sd["nstates"] > 0) & (_sd["omega"] != 0)
        theta_s = np.pi/2 - np.arctan(_sd["tanLambda"][_ok])
        eta_s = -np.log(np.tan(theta_s / 2))
        phi_s = _sd["phi"][_ok]
        pt_s = 0.2998 * BField / np.abs(_sd["omega"][_ok] * 1000.)
        raw["seed_pt_all"].append(pt_s)
        raw["seed_eta_all"].append(eta_s)
        acc["seed_nseeds_per_event"].append(len(pt_s))
        # Seed |d0| and max |zOrigin| (the seeder's ImpactMax / CollisionRegion
        # quantities) for every seed, and for the muon's 3/3 seeds with its truth pT.
        if not NO_HITS:
            _d0s, _zos = seed_impact(seedTracks)
            raw["seed_d0_all"].append(_d0s)
            raw["seed_zorigin_all"].append(_zos)
            _mu = _seed_mu_hits >= SEED_MATCH_MIN_HITS
            raw["seed_d0_mu"].append(_d0s[_mu])
            raw["seed_zorigin_mu"].append(_zos[_mu])
            raw["seed_mu_truth_pt"].append(np.full(int(_mu.sum()), mcp_tlv.Perp() if mcp_tlv is not None else np.nan))
            _lay = seed_layers(seedTracks)
            raw["seed_layers_all"].append(_lay)
            raw["seed_layers_mu"].append(_lay[_mu])
        # The muon's seed: one with >= SEED_MATCH_MIN_HITS muon hits (of several,
        # the one with the most, then the closest to the truth direction); if
        # there is none, the closest seed within SEED_MATCH_MAX_DR.
        if mcp_tlv is not None and len(pt_s):
            dphi = (phi_s - mcp_tlv.Phi() + np.pi) % (2 * np.pi) - np.pi
            dr = np.hypot(eta_s - mcp_tlv.Eta(), dphi)
            dr[np.isnan(dr)] = np.inf
            n_muon_hits = _seed_mu_hits[_ok]
            cand = np.nonzero(n_muon_hits >= SEED_MATCH_MIN_HITS)[0]
            j = None
            if len(cand):
                j = cand[np.lexsort((dr[cand], -n_muon_hits[cand]))[0]]
                acc["n_seed_hit_matched"] += 1
            elif dr.min() < SEED_MATCH_MAX_DR:
                j = int(np.argmin(dr))
                acc["n_seed_dr_matched"] += 1
            if j is not None:
                acc["seed_pt_best_per_muon"].append((mcp_tlv.Perp(), float(pt_s[j])))

        i += 1
        # Free this event and its per-hit arrays before the next event is
        # read; otherwise the worker briefly holds two events.
        del event
        _xs = _ys = _zs = _rs = _cids = _keys = _at = _sd = _si = None

    for k in _ARRAY_KEYS:
        acc[k] = np.concatenate(raw[k]) if raw[k] else np.empty(0)
    acc["n_events"] = i
    acc["seconds"] = time.time() - t_start
    return acc


def merge_results(file_results):
    """Combine the per-file accumulators of one entry, in file order."""
    m = {k: [] for k in _LIST_KEYS}
    m.update({k: 0 for k in _SUM_KEYS})
    m.update(
        layer_hits_sitracks={sys_id: defaultdict(int) for sys_id in range(1, 7)},
        id_diag_counts=defaultdict(int), id_diag_seen_per_sys=defaultdict(int), id_diag_sample={},
        occupancy_hist=np.zeros((len(OCC_Z_EDGES) - 1, len(OCC_R_EDGES) - 1), dtype=np.int64),
        event_displays=[], all_rz_displays=[],
    )
    arrays = {k: [] for k in _ARRAY_KEYS}
    zseam, zseam_n = [], 0
    for res in file_results:
        offset = m["n_events"]   # entry-wide index of this file's first event
        for k in _LIST_KEYS:
            m[k].extend(res[k])
        for k in _SUM_KEYS:
            m[k] += res[k]
        for k in _ARRAY_KEYS:
            arrays[k].append(res[k])
        for sys_id, layers in res["layer_hits_sitracks"].items():
            for layer, n in layers.items():
                m["layer_hits_sitracks"][sys_id][layer] += n
        for key, n in res["id_diag_counts"].items():
            m["id_diag_counts"][key] += n
        for s, n in res["id_diag_seen_per_sys"].items():
            m["id_diag_seen_per_sys"][s] += n
        for s, sample in res["id_diag_sample"].items():
            m["id_diag_sample"][s] = _bottom_k(m["id_diag_sample"].get(s), sample)
        m["occupancy_hist"] += res["occupancy_hist"]
        # Same cap rule as one sequential pass: an event's near-z=0 hits are
        # added while the running total is below ZSEAM_MAX_HITS.
        for chunk in res["zseam_chunks"]:
            if zseam_n < ZSEAM_MAX_HITS:
                zseam.append(chunk)
                zseam_n += len(chunk[0])
        m["event_displays"] += [(offset + li, d) for li, d in res["event_displays"]]
        m["all_rz_displays"] += [(offset + li, d) for li, d in res["all_rz_displays"]]
    m["all_rz_displays"] = m["all_rz_displays"][:10]
    for k in _ARRAY_KEYS:
        m[k] = np.concatenate(arrays[k]) if arrays[k] else np.empty(0)
    for j, k in enumerate(("zseam_z", "zseam_r", "zseam_sys")):
        m[k] = np.concatenate([c[j] for c in zseam]) if zseam else np.empty(0)
    return m


# --- Event hit maps ------------------------------------------------------------

def _draw_detector_rz(ax_non):
    """Tracker layer outlines in r-z."""
    for layer in vb_r:
        ax_non.plot([-vb_l/2, vb_l/2], [layer, layer], color="#348ec2")
        ax_non.plot([-vb_l/2, vb_l/2], [layer+2, layer+2], color="#348ec2")

    for j, layer in enumerate(ve_z):
        ax_non.plot([layer, layer], [ve_h[j], 125], alpha=0.5, color="#348ec2")
        ax_non.plot([-layer, -layer], [ve_h[j], 125], alpha=0.5, color="#348ec2")
        ax_non.plot([layer+2, layer+2], [ve_h[j], 125], alpha=0.5, color="#348ec2")
        ax_non.plot([-layer+2, -layer+2], [ve_h[j], 125], alpha=0.5, color="#348ec2")

    for j, layer in enumerate(ib_r):
        ax_non.plot([-ib_l[j]/2, ib_l[j]/2], [layer, layer], color="#34b18f")

    for j, layer in enumerate(ie_z):
        if j == 0:
            ax_non.plot([layer, layer], [ie_h[j], 410], alpha=0.5, color="#34b18f")
            ax_non.plot([-layer, -layer], [ie_h[j], 410], alpha=0.5, color="#34b18f")
        else:
            ax_non.plot([layer, layer], [ie_h[j], 570], alpha=0.5, color="#34b18f")
            ax_non.plot([-layer, -layer], [ie_h[j], 570], alpha=0.5, color="#34b18f")

    for layer in ob_r:
        ax_non.plot([-ob_l/2, ob_l/2], [layer, layer], color="#dd7e33")
    for j, layer in enumerate(oe_z):
        ax_non.plot([layer, layer], [600, 1430], alpha=0.5, color="#dd7e33")
        ax_non.plot([-layer, -layer], [600, 1430], alpha=0.5, color="#dd7e33")


def _scatter_event_hits(ax_non, d, view):
    """Muon / non-muon / missed-muon hits of one event; view is "rz" or "xy"."""
    def pts(name):
        return (d[f"rz_{name}_z"], d[f"rz_{name}_r"]) if view == "rz" else (d[f"xy_{name}_x"], d[f"xy_{name}_y"])

    if len(pts("muon")[0]) > 0:
        ax_non.scatter(*pts("muon"), s=30, color='steelblue', alpha=0.6, label="Muon-matched hits in track")
    if len(pts("bib")[0]) > 0:
        ax_non.scatter(*pts("bib"), s=30, color='tab:orange', alpha=0.6, label="Non-muon hits")
    if len(pts("muon_notintrack")[0]) > 0:
        ax_non.scatter(*pts("muon_notintrack"), s=40, marker='^', color='red', alpha=0.7, label="Muon hits (not in track)")


def _hit_map_title(view_name, d):
    title_non = f"SiTracks {view_name} Hit Map"
    if d["chi2_sitracks_list"]:
        title_non += f"\nχ²/Ndf = {np.mean(d['chi2_sitracks_list']):.2f}"
    if d["truth_pt"] is not None:
        title_non += f"\nTruth Muon $p_{{T}}$ = {d['truth_pt']:.1f} GeV"
    if d["reco_pt"] is not None:
        title_non += f"\nReconstructed Track $p_{{T}}$ = {d['reco_pt']:.1f} GeV"
    return title_non


def _sorted_fit_points(d):
    sorted_rz = sorted(d["rz_muon_sitracks"], key=lambda p: float(p[0])**2 + float(p[1])**2)
    z_fit = np.array([float(p[1]) for p in sorted_rz], dtype=float)
    r_fit = np.array([float(p[0]) for p in sorted_rz], dtype=float)
    return z_fit, r_fit


def draw_event_rz(d, i, plotdir, pdf):
    """r-z hit map of an event with exactly one muon-matched SiTrack."""
    fig, ax_non = plt.subplots(1, 1, figsize=(6, 5))
    _draw_detector_rz(ax_non)
    _scatter_event_hits(ax_non, d, "rz")
    if len(d["rz_muon_sitracks"]) > 0:
        z_fit, r_fit = _sorted_fit_points(d)
        a, b = np.polyfit(z_fit, r_fit, 1)
        ax_non.set_xlim(-RZ_ZLIM, RZ_ZLIM)
        ax_non.set_ylim(0, 1600)
        x0, x1 = ax_non.get_xlim()
        z_line = np.linspace(x0, x1, 200)
        ax_non.plot(z_line, a * z_line + b, linewidth=1, label=f"fit (N={len(z_fit)})")
    ax_non.set_xlabel("z [mm]")
    ax_non.set_ylabel("r [mm]")
    ax_non.set_title(_hit_map_title("r–z", d))
    ax_non.legend(loc="best", markerscale=2)
    ax_non.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    plt.savefig(plotdir + f"/EventRZ_{i}.pdf")
    pdf.savefig(fig)
    plt.close(fig)


def draw_event_xy(d, i, plotdir):
    """x-y hit map of the same events as draw_event_rz."""
    if len(d["xy_bib_x"]) == 0:
        return
    fig, ax_non = plt.subplots(1, 1, figsize=(6, 5))
    ax_non.set_xlim(-1400, 1400)
    ax_non.set_ylim(-1400, 1400)
    draw_radius_circles(ax_non, vb_r, color="#348ec2")
    draw_radius_circles(ax_non, ib_r, color="#34b18f")
    draw_radius_circles(ax_non, ob_r, color="#dd7e33")
    _scatter_event_hits(ax_non, d, "xy")
    ax_non.set_xlabel("x [mm]")
    ax_non.set_ylabel("y [mm]")
    ax_non.set_title(_hit_map_title("x–y", d))
    ax_non.legend(loc="best", markerscale=2)
    ax_non.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    ax_non.set_aspect("equal")
    fig.tight_layout()
    plt.savefig(plotdir + f"/EventXY_{i}.pdf")
    plt.close(fig)


def draw_all_rz(d, i, plotdir, pdf):
    """r-z hit map for the first 10 events with any muon-matched SiTrack."""
    fig, ax_non = plt.subplots(1, 1, figsize=(6, 5))
    _draw_detector_rz(ax_non)
    _scatter_event_hits(ax_non, d, "rz")
    if len(d["rz_muon_sitracks"]) > 0:
        z_fit, r_fit = _sorted_fit_points(d)
        a = np.dot(z_fit, r_fit) / np.dot(z_fit, z_fit)
        ax_non.set_xlim(-RZ_ZLIM, RZ_ZLIM)
        ax_non.set_ylim(0, 1600)
        z_end = z_fit[np.argmax(np.abs(z_fit))]
        z_line = np.linspace(0, z_end, 200)
        ax_non.plot(z_line, a * z_line, linewidth=1, label=f"fit (N={len(z_fit)})")
    ax_non.set_xlim(-RZ_ZLIM, RZ_ZLIM)
    ax_non.set_ylim(0, 1600)
    ax_non.set_xlabel("z [mm]")
    ax_non.set_ylabel("r [mm]")
    ax_non.set_title(_hit_map_title("r–z", d))
    ax_non.legend(loc="best", markerscale=2)
    ax_non.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    plt.savefig(plotdir + f"/All_EventRZ_{i}.pdf")
    pdf.savefig(fig)
    plt.close(fig)


# --- Per-entry plots -----------------------------------------------------------

def plot_entry(plotdir, acc, no_hits):
    """Plots for one entry from merge_results() output; returns what the
    cross-file overlays need."""
    # Event hit maps need hit positions, so there is nothing to put in these
    # files in no-hits mode.
    if not no_hits:
        with PdfPages(os.path.join(plotdir, "rz_hit_maps_all_events.pdf")) as rz_pdf:
            for ev, d in acc["event_displays"]:
                draw_event_rz(d, ev, plotdir, rz_pdf)
                draw_event_xy(d, ev, plotdir)
        with PdfPages(os.path.join(plotdir, "all_rz_hit_maps_muon_matched.pdf")) as all_rz_pdf:
            for ev, d in acc["all_rz_displays"]:
                draw_all_rz(d, ev, plotdir, all_rz_pdf)

    i = acc["n_events"]
    matched_tracks_sitracks = acc["matched_tracks_sitracks"]
    matched_tracks_alltracks = acc["matched_tracks_alltracks"]
    layer_hits_sitracks = acc["layer_hits_sitracks"]
    id_diag_counts = acc["id_diag_counts"]
    id_diag_seen_per_sys = acc["id_diag_seen_per_sys"]
    id_diag_sample = acc["id_diag_sample"]
    occupancy_hist = acc["occupancy_hist"]
    zseam_z = acc["zseam_z"]
    zseam_r = acc["zseam_r"]
    zseam_sys = acc["zseam_sys"]
    pixel_nhits_sitracks = acc["pixel_nhits_sitracks"]
    inner_nhits_sitracks = acc["inner_nhits_sitracks"]
    outer_nhits_sitracks = acc["outer_nhits_sitracks"]
    total_nhits_sitracks = acc["total_nhits_sitracks"]
    pixel_nhits_alltracks = acc["pixel_nhits_alltracks"]
    inner_nhits_alltracks = acc["inner_nhits_alltracks"]
    outer_nhits_alltracks = acc["outer_nhits_alltracks"]
    total_nhits_alltracks = acc["total_nhits_alltracks"]
    track_chi2_reduced_sitracks = acc["track_chi2_reduced_sitracks"]
    track_chi2_reduced_alltracks = acc["track_chi2_reduced_alltracks"]
    track_eta_sitracks = acc["track_eta_sitracks"]
    track_matched_pt_sitracks = acc["track_matched_pt_sitracks"]
    track_percent_muon_hits_sitracks = acc["track_percent_muon_hits_sitracks"]
    track_pt_resolution_sitracks = acc["track_pt_resolution_sitracks"]
    track_reco_pt_sitracks = acc["track_reco_pt_sitracks"]
    track_phi_sitracks = acc["track_phi_sitracks"]
    track_d0_sitracks = acc["track_d0_sitracks"]
    track_z0_sitracks = acc["track_z0_sitracks"]
    track_delta_qoverp_sq_sitracks = acc["track_delta_qoverp_sq_sitracks"]
    num_bib_hits_sitracks = acc["num_bib_hits_sitracks"]
    all_tracks_mult_per_event = acc["all_tracks_mult_per_event"]
    deduped_tracks_mult_per_event = acc["deduped_tracks_mult_per_event"]
    sitracks_mult_per_event = acc["sitracks_mult_per_event"]
    muon_matched_sitracks_per_event = acc["muon_matched_sitracks_per_event"]
    eff_truth_eta = acc["eff_truth_eta"]
    eff_truth_pt = acc["eff_truth_pt"]
    eff_truth_phi = acc["eff_truth_phi"]
    eff_truth_theta_deg = acc["eff_truth_theta_deg"]
    eff_matched_si = acc["eff_matched_si"]
    eff_matched_alltracks = acc["eff_matched_alltracks"]
    alltracks_raw_nhits = acc["alltracks_raw_nhits"]
    alltracks_raw_chi2_reduced = acc["alltracks_raw_chi2_reduced"]
    alltracks_raw_pt = acc["alltracks_raw_pt"]
    alltracks_raw_d0 = acc["alltracks_raw_d0"]
    alltracks_raw_z0 = acc["alltracks_raw_z0"]
    seed_pt_all = acc["seed_pt_all"]
    seed_eta_all = acc["seed_eta_all"]
    seed_nseeds_per_event = acc["seed_nseeds_per_event"]
    seed_pt_best_per_muon = acc["seed_pt_best_per_muon"]
    layer_nhits_per_event = acc["layer_nhits_per_event"]
    fake_mult_per_event = acc["fake_mult_per_event"]
    fake_pt = acc["fake_pt"]
    fake_eta = acc["fake_eta"]

    if len(muon_matched_sitracks_per_event) > 0:
        max_mult = max(muon_matched_sitracks_per_event) if muon_matched_sitracks_per_event else 5
        bins_mm = np.arange(-0.5, max_mult + 1.5, 1)
        plt.figure()
        plt.hist(muon_matched_sitracks_per_event, bins=bins_mm,
                 weights=unit_weights(muon_matched_sitracks_per_event, bins_mm),
                 histtype='step', linewidth=1.5, label="SiTracks")
        plt.xlabel("Number of muon-matched tracks per event")
        plt.ylabel("Fraction of events")
        plt.yscale('log')
        plt.legend()
        plt.title("Muon-Matched Track Multiplicity per Event")
        plt.xticks(range(max_mult + 2))
        plt.gca().xaxis.set_minor_locator(NullLocator())
        plt.tight_layout()
        plt.savefig(plotdir + "/muon_matched_track_mult_per_event.pdf")
        plt.close()

    # --- Cutflow: # tracks surviving each reconstruction stage ---
    if len(sitracks_mult_per_event) > 0:
        stage_names = ["AllTracks", "DedupedTracks", "SiTracks"]
        stage_lists = [all_tracks_mult_per_event, deduped_tracks_mult_per_event,
                       sitracks_mult_per_event]
        stage_totals = [sum(lst) for lst in stage_lists]
        stage_means  = [np.mean(lst) if lst else 0.0 for lst in stage_lists]
        stage_stds   = [np.std(lst)  if lst else 0.0 for lst in stage_lists]

        print("Track cutflow (totals across all events; %% relative to prev / AllTracks):")
        prev = stage_totals[0]
        first = stage_totals[0] if stage_totals[0] > 0 else 1
        for n, t, m in zip(stage_names, stage_totals, stage_means):
            f_prev  = 100 * t / prev  if prev  > 0 else 0.0
            f_first = 100 * t / first if first > 0 else 0.0
            print(f"  {n:<20s} total={t:>8d}  mean/ev={m:>7.2f}   "
                  f"vs prev: {f_prev:5.1f}%   vs AllTracks: {f_first:5.1f}%")
            prev = t

        # Same per-event counts as the multiplicity plots.
        print("Tracks per event (track collections: tracks with hits; SeedTracks: seeds with a track state; "
              "Fake SiTracks: SiTracks not matched to the muon):")
        for n, counts in [("AllTracks", all_tracks_mult_per_event),
                          ("DedupedTracks", deduped_tracks_mult_per_event),
                          ("SiTracks", sitracks_mult_per_event),
                          ("Fake SiTracks", fake_mult_per_event),
                          ("SeedTracks", seed_nseeds_per_event)]:
            print(f"  {n:<20s} mean={np.mean(counts):>10.2f}  median={np.median(counts):>10.1f}  "
                  f"std dev={np.std(counts):>10.2f}")

        fig, axs = plt.subplots(1, 2, figsize=(13, 5))
        x = np.arange(len(stage_names))
        axs[0].bar(x, stage_totals, color='steelblue', edgecolor='black')
        for xi, t in zip(x, stage_totals):
            axs[0].text(xi, t, f"{t}", ha='center', va='bottom', fontsize=9)
        axs[0].set_xticks(x)
        axs[0].set_xticklabels(stage_names, rotation=20, ha='right')
        axs[0].xaxis.set_minor_locator(NullLocator())
        axs[0].set_ylabel("Total tracks (all events)")
        axs[0].set_title("Track Cutflow: total surviving per stage")
        axs[0].grid(True, axis='y', linestyle=':', alpha=0.6)

        axs[1].bar(x, stage_means, yerr=stage_stds, color='salmon', edgecolor='black', capsize=4)
        for xi, m in zip(x, stage_means):
            axs[1].text(xi, m, f"{m:.2f}", ha='center', va='bottom', fontsize=9)
        axs[1].set_xticks(x)
        axs[1].set_xticklabels(stage_names, rotation=20, ha='right')
        axs[1].xaxis.set_minor_locator(NullLocator())
        axs[1].set_ylabel("Mean tracks / event (±RMS)")
        axs[1].set_title("Track Cutflow: per-event average")
        axs[1].grid(True, axis='y', linestyle=':', alpha=0.6)
        fig.tight_layout()
        plt.savefig(plotdir + "/track_cutflow.pdf")
        plt.close(fig)

        all_counts = (all_tracks_mult_per_event + deduped_tracks_mult_per_event
                      + sitracks_mult_per_event)
        all_counts_pos = [v for v in all_counts if v > 0]
        lo = max(1, min(all_counts_pos)) if all_counts_pos else 1
        hi = max(all_counts) if all_counts else 1
        bins_mult = np.logspace(np.log10(lo), np.log10(hi + 1), 51)
        plt.figure()
        for arr, lbl in [
            (all_tracks_mult_per_event,      "AllTracks"),
            (deduped_tracks_mult_per_event,   "DedupedTracks"),
            (sitracks_mult_per_event,         "SiTracks"),
        ]:
            if arr:
                plt.hist(arr, bins=bins_mult, weights=unit_weights(arr, bins_mult),
                         histtype='step', linewidth=1.5, label=f"{lbl}  (mean={np.mean(arr):.3g})")
        plt.xscale('log')
        plt.xlabel("Track multiplicity per event")
        plt.ylabel("Fraction of events")
        plt.legend()
        plt.title("Track Multiplicity per Event")
        plt.tight_layout()
        plt.savefig(plotdir + "/track_multiplicity_per_event.pdf")
        plt.close()

    # --- Fake SiTracks (with hits, not matched to the muon): rate vs pT threshold, pT and eta
    if len(fake_pt) > 0 and i > 0:
        fake_label = f"{np.mean(fake_mult_per_event):.3g} fake tracks / event"
        bins_fpt = np.logspace(-1, 4, 51)
        plt.figure()
        rate, err = rate_above(fake_pt, FAKE_PT_THRESHOLDS, i)
        plt.errorbar(FAKE_PT_THRESHOLDS, rate, yerr=err, fmt='o', markersize=3, label=fake_label)
        plt.yscale('log')
        plt.xlim(FAKE_PT_THRESHOLDS[0], FAKE_PT_THRESHOLDS[-1])
        plt.xlabel(r"$p_T$ threshold [GeV]")
        plt.ylabel(r"Fake tracks with $p_T >$ threshold / event / 0.5 GeV")
        plt.title(r"Fake Track Rate vs $p_T$ Threshold (SiTracks not matched to the muon)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plotdir + "/fake_rate_vs_pt_threshold.pdf")
        plt.close()

        plt.figure()
        plt.hist(fake_pt, bins=bins_fpt, weights=np.ones(len(fake_pt)) / len(fake_pt),
                 histtype='step', linewidth=1.5, label=f"{fake_label}, median $p_T$ = {np.median(fake_pt):.3g} GeV")
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r"Track $p_T$ [GeV]")
        plt.ylabel("Fraction of fake tracks")
        plt.title(r"Fake Track $p_T$ (SiTracks not matched to the muon)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plotdir + "/fake_pt.pdf")
        plt.close()

        plt.figure()
        plt.hist(fake_eta, bins=np.linspace(-3, 3, 61), weights=np.ones(len(fake_eta)) / len(fake_eta),
                 histtype='step', linewidth=1.5, label=fake_label)
        plt.xlabel(r"Track $\eta$")
        plt.ylabel("Fraction of fake tracks")
        plt.title(r"Fake Track $\eta$ (SiTracks not matched to the muon)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plotdir + "/fake_eta.pdf")
        plt.close()

    # Duplicate SiTracks per event vs reco pT and eta (pT outside the bins clipped into the edge bins)
    if acc["dup_mult_per_event"]:
        dup_per_event = np.mean(acc["dup_mult_per_event"])
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, key, bins, xlabel, logx in [
            (axs[0], "dup_pt", PT_BINS_0_50, r"Track $p_T$ [GeV]", True),
            (axs[1], "dup_eta", np.linspace(-2.5, 2.5, 26), r"Track $\eta$", False),
        ]:
            c, r, err = rate_per_event(acc[key], bins, i, clip=logx)
            ax.errorbar(c, r, yerr=err, fmt='o', color='steelblue', capsize=3,
                        label=f"SiTracks ({dup_per_event:.3g} duplicates / event)")
            if logx:
                ax.set_xscale('log')
            ax.set_xlim(bins[0], bins[-1])
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Duplicate tracks / event / bin")
            ax.legend(fontsize=8)
            ax.grid(True, linestyle=":", alpha=0.7)
        fig.suptitle(f"Duplicate SiTracks (> {100*DUP_SHARED_FRAC:.0f}% of hits shared with a higher-ranked track)")
        fig.tight_layout()
        plt.savefig(plotdir + "/duplicate_rate.pdf")
        plt.close(fig)

    # Seed |d0| and max doublet |zOrigin| (what ImpactMax / CollisionRegion cut on).
    # Only seeds that passed the run's cuts are stored, so run with loose cuts.
    if len(acc["seed_d0_mu"]):
        fig, axs = plt.subplots(2, 2, figsize=(13, 9))
        for col, (mu_key, all_key, bins, xlabel, cut) in enumerate([
            ("seed_d0_mu", "seed_d0_all", SEED_D0_BINS, r"Seed $|d_0|$ [mm]", SEED_D0_REF),
            ("seed_zorigin_mu", "seed_zorigin_all", SEED_ZORIGIN_BINS, r"Seed max doublet $|z_{\mathrm{Origin}}|$ [mm]", SEED_ZORIGIN_REF),
        ]):
            ax = axs[0, col]
            for key, color, name in [(all_key, 'gray', "all seeds"), (mu_key, 'steelblue', "muon seeds (3/3)")]:
                v = np.asarray(acc[key], dtype=float)
                v = v[np.isfinite(v)]
                if len(v):
                    ax.hist(np.clip(v, bins[0], bins[-1]), bins=bins, weights=np.full(len(v), 1. / len(v)),
                            histtype='step', linewidth=1.5, color=color, label=seed_cut_label(name, v, cut))
            ax.axvline(cut, color='red', linestyle='--', linewidth=1)
            ax.set_yscale('log')
            ax.set_xlabel(xlabel + " (last bin: overflow)")
            ax.set_ylabel("Fraction of seeds")
            ax.legend(fontsize=7)
            ax = axs[1, col]
            v, pt = np.asarray(acc[mu_key], dtype=float), np.asarray(acc["seed_mu_truth_pt"], dtype=float)
            ok = np.isfinite(v) & np.isfinite(pt)
            plt.sca(ax)
            img = hist2d_lognorm(np.clip(pt[ok], PT_BINS_0_50[0], PT_BINS_0_50[-1]),
                                 np.clip(v[ok], bins[0], bins[-1]), [PT_BINS_0_50, bins])
            if img is not None:
                plt.colorbar(img, ax=ax, label="Muon seeds")
            ax.axhline(cut, color='red', linestyle='--', linewidth=1)
            ax.set_xscale('log')
            ax.set_xlabel(r"Truth Muon $p_T$ [GeV]")
            ax.set_ylabel(xlabel)
        fig.suptitle("Seed impact parameters (red: current ImpactMax / CollisionRegion)")
        fig.tight_layout()
        plt.savefig(plotdir + "/seed_impact.pdf")
        plt.close(fig)

    # Hits per event in each tracker layer, from the cellID (system, layer).
    if layer_nhits_per_event:
        print("Hits per event in each layer (MergedTrackerHits; an endcap layer counts both sides):")
        for sys_id, layer in sorted({k for ev in layer_nhits_per_event for k in ev}):
            counts = [ev.get((sys_id, layer), 0) for ev in layer_nhits_per_event]
            print(f"  {system_to_relname.get(sys_id, f'system{sys_id}'):<10s} layer {layer:>2d}  "
                  f"mean={np.mean(counts):>10.2f}  median={np.median(counts):>10.1f}  "
                  f"std dev={np.std(counts):>10.2f}")

    eff_sitracks  = matched_tracks_sitracks  / i if i > 0 else 0.0
    eff_alltracks = matched_tracks_alltracks / i if i > 0 else 0.0
    print(f"AllTracks (>=1 truth-matched) efficiency: {matched_tracks_alltracks}/{i} ({100*eff_alltracks:.1f}%)")
    print(f"SiTracks efficiency:       {matched_tracks_sitracks}/{i} ({100*eff_sitracks:.1f}%)")

    # --- AllTracks raw distributions with the AllTracks→SiTracks filter cut lines ---
    # Cuts in track_filter_cfg: MinPt=0.5 GeV, MaxD0=100, MaxZ0=100, plus an
    # NHits>=8 reference line on the nhits panel.
    if len(alltracks_raw_nhits) or len(alltracks_raw_pt) or len(alltracks_raw_d0):
        FILT_MIN_PT = 0.5    # GeV
        FILT_MAX_D0 = 100    # mm
        FILT_MAX_Z0 = 100    # mm
        REF_NHITS_MIN = 8
        fig, axs = plt.subplots(2, 2, figsize=(12, 9))

        # NHits per AllTrack
        if len(alltracks_raw_nhits):
            nhit_max = max(int(np.max(alltracks_raw_nhits)), 20)
            bins_nhit = np.arange(-0.5, nhit_max + 1.5, 1)
            axs[0, 0].hist(alltracks_raw_nhits, bins=bins_nhit, color='steelblue', edgecolor='black',
                           label=f"N={len(alltracks_raw_nhits)}\nmedian={np.median(alltracks_raw_nhits):.1f}\nmean={np.mean(alltracks_raw_nhits):.2f}")
            axs[0, 0].axvline(REF_NHITS_MIN - 0.5, color='red', linestyle='--',
                              label=f"reference: NHits>={REF_NHITS_MIN}")
            axs[0, 0].set_yscale('log')
            axs[0, 0].set_xlabel("NHits per AllTrack")
            axs[0, 0].set_ylabel("Tracks")
            axs[0, 0].set_title("AllTracks: NHits per track (raw)")
            axs[0, 0].legend(fontsize=8)
        else:
            axs[0, 0].set_title("AllTracks: NHits per track\n(no hit counts in this file)", fontsize=9)
            axs[0, 0].set_xticks([])
            axs[0, 0].set_yticks([])

        # chi2/Ndf per AllTrack
        if len(alltracks_raw_chi2_reduced):
            bins_chi2 = np.logspace(-4, 6, 60)
            axs[0, 1].hist(alltracks_raw_chi2_reduced, bins=bins_chi2,
                           color='steelblue', edgecolor='black',
                           label=f"N={len(alltracks_raw_chi2_reduced)}\nmedian={np.median(alltracks_raw_chi2_reduced):.2e}")
            axs[0, 1].set_xscale('log')
            axs[0, 1].set_yscale('log')
            axs[0, 1].set_xlabel(r"$\chi^2$ / Ndf")
            axs[0, 1].set_ylabel("Tracks")
            axs[0, 1].set_title("AllTracks: chi2/Ndf (raw)")
            axs[0, 1].legend(fontsize=8)

        # Reco pT per AllTrack
        if len(alltracks_raw_pt):
            pt_arr = np.array(alltracks_raw_pt)
            pt_min = max(pt_arr.min() * 0.5, 1e-4)
            pt_max = pt_arr.max() * 1.5
            bins_pt = np.logspace(np.log10(pt_min), np.log10(pt_max), 60)
            axs[1, 0].hist(pt_arr, bins=bins_pt, color='steelblue', edgecolor='black',
                           label=f"N={len(pt_arr)}\nmedian={np.median(pt_arr):.2g} GeV")
            axs[1, 0].axvline(FILT_MIN_PT, color='red', linestyle='--',
                              label=f"filter cut: pT>={FILT_MIN_PT} GeV")
            axs[1, 0].set_xscale('log')
            axs[1, 0].set_yscale('log')
            axs[1, 0].set_xlabel(r"Reco $p_T$ [GeV]")
            axs[1, 0].set_ylabel("Tracks")
            axs[1, 0].set_title("AllTracks: reco $p_T$ (raw)")
            axs[1, 0].legend(fontsize=8)

        # D0 and Z0 overlay
        if len(alltracks_raw_d0) and len(alltracks_raw_z0):
            d0_arr = np.array(alltracks_raw_d0)
            z0_arr = np.array(alltracks_raw_z0)
            ext = max(np.max(np.abs(d0_arr)), np.max(np.abs(z0_arr)), FILT_MAX_D0 * 2)
            bins_dz = np.linspace(-ext, ext, 81)
            axs[1, 1].hist(d0_arr, bins=bins_dz, histtype='step', linewidth=1.7,
                           color='steelblue',
                           label=f"D0  median={np.median(d0_arr):.1f}, |D0|>100: {np.sum(np.abs(d0_arr) > FILT_MAX_D0)}/{len(d0_arr)}")
            axs[1, 1].hist(z0_arr, bins=bins_dz, histtype='step', linewidth=1.7,
                           color='salmon',
                           label=f"Z0  median={np.median(z0_arr):.1f}, |Z0|>100: {np.sum(np.abs(z0_arr) > FILT_MAX_Z0)}/{len(z0_arr)}")
            axs[1, 1].axvline(-FILT_MAX_D0, color='red', linestyle='--', alpha=0.7,
                              label=f"filter cut: |D0|,|Z0|<{FILT_MAX_D0} mm")
            axs[1, 1].axvline( FILT_MAX_D0, color='red', linestyle='--', alpha=0.7)
            axs[1, 1].set_yscale('log')
            axs[1, 1].set_xlabel(r"$d_0$ / $z_0$ [mm]")
            axs[1, 1].set_ylabel("Tracks")
            axs[1, 1].set_title("AllTracks: $d_0$ and $z_0$ (raw)")
            axs[1, 1].legend(fontsize=7)

        fig.suptitle("AllTracks raw distributions with AllTracks→SiTracks filter cuts", fontsize=13)
        fig.tight_layout()
        plt.savefig(plotdir + "/alltracks_raw_filter_diagnostic.pdf")
        plt.close(fig)

        # Cut-survivor summary (non-exclusive)
        d0_arr_full = np.array(alltracks_raw_d0)
        z0_arr_full = np.array(alltracks_raw_z0)
        pt_arr_full = np.asarray(alltracks_raw_pt)
        nhit_arr_full = np.array(alltracks_raw_nhits)
        N = max(len(alltracks_raw_nhits), d0_arr_full.size, pt_arr_full.size)
        fail_pt    = int(np.sum(pt_arr_full < FILT_MIN_PT)) if pt_arr_full.size else 0
        fail_d0    = int(np.sum(np.abs(d0_arr_full) > FILT_MAX_D0)) if d0_arr_full.size else 0
        fail_z0    = int(np.sum(np.abs(z0_arr_full) > FILT_MAX_Z0)) if z0_arr_full.size else 0
        print(f"[AllTracks raw] N={N}")
        print(f"  fail pT < {FILT_MIN_PT} GeV: {fail_pt}/{N} ({100*fail_pt/max(N,1):.1f}%)")
        print(f"  fail |D0| > {FILT_MAX_D0} mm:  {fail_d0}/{N} ({100*fail_d0/max(N,1):.1f}%)")
        print(f"  fail |Z0| > {FILT_MAX_Z0} mm:  {fail_z0}/{N} ({100*fail_z0/max(N,1):.1f}%)")
        if nhit_arr_full.size:
            fail_nhits = int(np.sum(nhit_arr_full < REF_NHITS_MIN))
            print(f"  fail NHits < {REF_NHITS_MIN} : {fail_nhits}/{nhit_arr_full.size} "
                  f"({100*fail_nhits/max(nhit_arr_full.size,1):.1f}%)")
        else:
            print(f"  fail NHits < {REF_NHITS_MIN} : n/a (no hit counts in this file)")
    # The pixel/inner/outer panels are empty in no-hits mode (no cellIDs to
    # group by), so the total drives the guard and empty panels are labelled
    # rather than plotted.
    for _outname, _panels in [
        ("all_hits_combined_sitracks.pdf", [
            (pixel_nhits_sitracks,  'skyblue',    "Pixel Hits (SiTracks)",  10, (0, 10)),
            (inner_nhits_sitracks,  'lightgreen', "Inner Hits (SiTracks)",   5, (0,  5)),
            (outer_nhits_sitracks,  'salmon',     "Outer Hits (SiTracks)",   5, (0,  5)),
            (total_nhits_sitracks,  'green',      "Total Hits (SiTracks)",  15, (0, 15)),
        ]),
        ("all_hits_combined_alltracks.pdf", [
            (pixel_nhits_alltracks, 'skyblue',    "Pixel Hits (AllTracks)",  10, (0, 10)),
            (inner_nhits_alltracks, 'lightgreen', "Inner Hits (AllTracks)",   5, (0,  5)),
            (outer_nhits_alltracks, 'salmon',     "Outer Hits (AllTracks)",   5, (0,  5)),
            (total_nhits_alltracks, 'green',      "Total Hits (AllTracks)",  15, (0, 15)),
        ]),
    ]:
        if not any(len(d) > 0 for d, *_ in _panels):
            continue
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        for ax, (data, color, title, nbins, rng) in zip(axs.flat, _panels):
            if len(data) == 0:
                ax.set_title(f"{title}\n(no hit-level info in this file)", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            ax.hist(data, bins=nbins, range=rng, color=color, edgecolor='black',
                    label=f"Mean: {np.mean(data):.2f}\nMedian: {np.median(data):.1f}")
            ax.set_title(title)
            ax.set_xlabel("Number of hits")
            ax.set_ylabel("Tracks")
            ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(plotdir, _outname))
        plt.close(fig)

    # === Track Chi2 Reduced (draw whichever series has data) ===
    if (len(track_chi2_reduced_sitracks) > 0
            or len(track_chi2_reduced_alltracks) > 0):
        plt.figure()
        bins = np.logspace(-2, 6, 60)
        if len(track_chi2_reduced_alltracks) > 0:
            plt.hist(track_chi2_reduced_alltracks, bins=bins, weights=np.ones(len(track_chi2_reduced_alltracks)) / len(track_chi2_reduced_alltracks), histtype='step', linewidth=1.7, color='purple', label="AllTracks")
        if len(track_chi2_reduced_sitracks) > 0:
            plt.hist(track_chi2_reduced_sitracks, bins=bins, weights=np.ones(len(track_chi2_reduced_sitracks)) / len(track_chi2_reduced_sitracks), color='steelblue', edgecolor='black', alpha=0.6, label="SiTracks")
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.title("Matched Muon Tracks Chi2 Reduced")
        plt.xlabel("Chi2 Reduced")
        plt.ylabel("Fraction of tracks")
        plt.tight_layout()
        plt.savefig(plotdir + "/chi2_reduced.pdf")
        plt.close()

    # === SiTracks-only summary plots ===
    if len(track_chi2_reduced_sitracks) > 0:
        pt_max_sitracks = 2 * max(track_matched_pt_sitracks)
        xbins_pt = np.linspace(0, pt_max_sitracks, 51)
        ybins_pt = np.linspace(0, pt_max_sitracks, 51)
        reco_pt_sitracks = [truth * (1 + res) for truth, res in zip(track_matched_pt_sitracks, track_pt_resolution_sitracks)]
        plt.figure()
        img = hist2d_lognorm(track_matched_pt_sitracks, reco_pt_sitracks, [xbins_pt, ybins_pt])
        if img is not None:
            plt.colorbar(img, label="Counts")
        plt.plot([0, pt_max_sitracks], [0, pt_max_sitracks], 'r--', linewidth=1, label="truth = reco")
        plt.xlabel("Truth Muon $p_T$ [GeV]")
        plt.ylabel("Reconstructed Track $p_T$ [GeV]")
        plt.legend()
        plt.title("Truth vs Reconstructed $p_T$ (SiTracks)")
        plt.tight_layout()
        plt.savefig(plotdir + "/truth_vs_reco_pt_sitracks.pdf")
        plt.close()

        xbins = np.logspace(-2, 4, 60)
        ybins = np.linspace(-3, 3, 61)
        plt.figure()
        img = hist2d_lognorm(track_chi2_reduced_sitracks, track_eta_sitracks, [xbins, ybins])
        if img is not None:
            plt.colorbar(img, label="Counts")
        plt.xscale("log")
        plt.xlabel("Chi2 Reduced")
        plt.ylabel("Track η")
        plt.title("Chi2 vs η (SiTracks)")
        plt.tight_layout()
        plt.savefig(plotdir + "/chi2_vs_eta_sitracks_hist2d.pdf")
        plt.close()

        xbins = np.linspace(0, 2500, 26)
        ybins = np.logspace(-2, 4, 60)
        plt.figure()
        img = hist2d_lognorm(track_matched_pt_sitracks, track_chi2_reduced_sitracks, [xbins, ybins])
        if img is not None:
            plt.colorbar(img, label="Counts")
        plt.yscale("log")
        plt.xlabel("Truth Muon $p_T$ [GeV]")
        plt.ylabel("Chi2 Reduced")
        plt.title("$p_T$ vs Chi2 Reduced (SiTracks)")
        plt.tight_layout()
        plt.savefig(plotdir + "/pt_vs_chi2_sitracks_hist2d.pdf")
        plt.close()

    # The panels below all need the per-hit muon/BIB classification, which only
    # exists when the hit relations were written out.
    if len(track_chi2_reduced_sitracks) > 0 and len(track_percent_muon_hits_sitracks) > 0:
        xbins = np.linspace(0, 2500, 26)
        ybins = np.linspace(0, 1, 21)
        plt.figure()
        img = hist2d_lognorm(track_matched_pt_sitracks, track_percent_muon_hits_sitracks, [xbins, ybins])
        if img is not None:
            plt.colorbar(img, label="Counts")
        plt.xlabel("Truth Muon $p_T$ [GeV]")
        plt.ylabel("Fraction of Muon Hits")
        plt.ylim(0, 1)
        plt.title("$p_T$ vs Fraction of Muon Hits (SiTracks)")
        plt.tight_layout()
        plt.savefig(plotdir + "/pt_vs_muonhitfraction_sitracks_hist2d.pdf")
        plt.close()

        xbins = np.linspace(0, 1, 21)
        ybins = np.logspace(-2, 4, 60)
        plt.figure()
        img = hist2d_lognorm(track_percent_muon_hits_sitracks, track_chi2_reduced_sitracks, [xbins, ybins])
        if img is not None:
            plt.colorbar(img, label="Counts")
        plt.yscale("log")
        plt.xlabel("Fraction of Muon Hits")
        plt.ylabel("Chi2 Reduced")
        plt.xlim(0, 1)
        plt.title("Fraction of Muon Hits vs Chi2 Reduced (SiTracks)")
        plt.tight_layout()
        plt.savefig(plotdir + "/muonhitfraction_vs_chi2_sitracks_hist2d.pdf")
        plt.close()

        xbins = np.linspace(0, 10, 11)
        ybins = np.logspace(-2, 4, 60)
        plt.figure()
        img = hist2d_lognorm(num_bib_hits_sitracks, track_chi2_reduced_sitracks, [xbins, ybins])
        if img is not None:
            plt.colorbar(img, label="Counts")
        plt.yscale("log")
        plt.xlabel("# BIB Hits in Muon Matched Track")
        plt.ylabel("Chi2 Reduced")
        plt.xlim(0, 10)
        plt.title("# BIB Hits vs Chi2 Reduced (SiTracks)")
        plt.tight_layout()
        plt.savefig(plotdir + "/numberBIBhits_vs_chi2_sitracks_hist2d.pdf")
        plt.close()

        xbins = np.linspace(0, 1, 21)
        ybins = np.linspace(-1, 1, 81)
        w_pt_si = np.ones(len(track_pt_resolution_sitracks)) / len(track_pt_resolution_sitracks)
        plt.figure()
        img = hist2d_lognorm(track_percent_muon_hits_sitracks, track_pt_resolution_sitracks, [xbins, ybins], weights=w_pt_si)
        if img is not None:
            plt.colorbar(img, label="Fraction of tracks")
        plt.xlabel("Fraction of Muon Hits")
        plt.ylabel(r"$p_T$ Resolution $(p_T^{\mathrm{reco}} - p_T^{\mathrm{truth}}) / p_T^{\mathrm{truth}}$")
        plt.xlim(0, 1)
        plt.title(r"$p_T$ Resolution vs Fraction of Muon Hits (SiTracks)")
        plt.tight_layout()
        plt.savefig(plotdir + "/pt_resolution_vs_muonhitfraction_sitracks_hist2d.pdf")
        plt.close()

        min_dqp_si = max(min(track_delta_qoverp_sq_sitracks), 1e-12)
        xbins = np.linspace(0, 1, 21)
        ybins = np.logspace(np.log10(min_dqp_si), np.log10(max(track_delta_qoverp_sq_sitracks) * 1.1), 81)
        w_si = np.ones(len(track_delta_qoverp_sq_sitracks)) / len(track_delta_qoverp_sq_sitracks)
        plt.figure()
        img = hist2d_lognorm(track_percent_muon_hits_sitracks, track_delta_qoverp_sq_sitracks, [xbins, ybins], weights=w_si)
        if img is not None:
            plt.colorbar(img, label="Fraction of tracks")
        plt.xlabel("Fraction of Muon Hits")
        plt.ylabel(r"$\sigma(q/p)^2 = (q/p_{\mathrm{reco}} - q/p_{\mathrm{truth}})^2$ [GeV$^{-2}$]")
        plt.xlim(0, 1)
        plt.yscale("log")
        plt.title(r"$\sigma(q/p)^2$ vs Fraction of Muon Hits (SiTracks)")
        plt.tight_layout()
        plt.savefig(plotdir + "/delta_qoverp_sq_vs_muonhitfraction_sitracks_hist2d.pdf")
        plt.close()

    # === Seed track plots ===
    if len(seed_pt_all) > 0:
        bins_pt_s = np.logspace(np.log10(1), np.log10(5000), 81)
        plt.figure()
        plt.hist(seed_pt_all, bins=bins_pt_s, color='steelblue', edgecolor='black',
                 label=f"N={len(seed_pt_all)}\nMean: {np.mean(seed_pt_all):.1f} GeV")
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r"Seed $p_T$ [GeV]")
        plt.ylabel("Seeds")
        plt.legend(fontsize=9)
        plt.title(r"Seed Track $p_T$ Distribution (all seeds)")
        plt.tight_layout()
        plt.savefig(plotdir + "/seed_pt.pdf")
        plt.close()

        bins_eta_s = np.linspace(-3, 3, 61)
        plt.figure()
        plt.hist(seed_eta_all, bins=bins_eta_s, color='steelblue', edgecolor='black')
        plt.xlabel(r"Seed $\eta$")
        plt.ylabel("Seeds")
        plt.title(r"Seed Track $\eta$ Distribution (all seeds)")
        plt.tight_layout()
        plt.savefig(plotdir + "/seed_eta.pdf")
        plt.close()

    if len(seed_nseeds_per_event) > 0 and max(seed_nseeds_per_event) > 0:
        min_ns    = max(1, min(v for v in seed_nseeds_per_event if v > 0))
        max_ns    = max(seed_nseeds_per_event)
        mean_ns   = np.mean(seed_nseeds_per_event)
        median_ns = np.median(seed_nseeds_per_event)
        lo = max(0.5, min_ns / 1.2)
        hi = max_ns * 1.2
        bins_ns = np.logspace(np.log10(lo), np.log10(hi), 51) if max_ns > 1 else np.arange(-0.5, 5.5, 1)
        plt.figure()
        plt.hist(seed_nseeds_per_event, bins=bins_ns, weights=unit_weights(seed_nseeds_per_event, bins_ns),
                 color='steelblue', edgecolor='black',
                 label=f'mean={mean_ns:.2e},  median={median_ns:.2e}')
        if max_ns > 1:
            plt.xscale('log')
        plt.xlabel("Number of seeds per event")
        plt.ylabel("Fraction of events")
        plt.title("Seed Multiplicity per Event")
        plt.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(plotdir + "/seed_mult_per_event.pdf")
        plt.close()

        # Seeds per event vs the seed's own pT and eta (pT outside the bins clipped into the edge bins)
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, vals, bins, xlabel, logx in [
            (axs[0], seed_pt_all, SEED_PT_BINS, r"Seed $p_T$ [GeV]", True),
            (axs[1], seed_eta_all, np.linspace(-2.5, 2.5, 26), r"Seed $\eta$", False),
        ]:
            c, r, err = rate_per_event(vals, bins, i, clip=logx)
            ax.errorbar(c, r, yerr=err, fmt='o', color='steelblue', capsize=3,
                        label=f"all seeds ({mean_ns:.3g} / event)")
            if logx:
                ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(bins[0], bins[-1])
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Seeds / event / bin")
            ax.legend(fontsize=8)
            ax.grid(True, linestyle=":", alpha=0.7)
        fig.suptitle("Average Seed Multiplicity per Event vs Seed $p_T$ and $\\eta$")
        fig.tight_layout()
        plt.savefig(plotdir + "/seed_mult_vs_pt_eta.pdf")
        plt.close(fig)

    # Seeds per event by layer combination (hits ordered by radius, inner first)
    for key, name, color in [("seed_layers_all", "all seeds", "steelblue"),
                             ("seed_layers_mu", "muon seeds, 3/3 muon hits", "darkorange")]:
        if len(acc[key]):
            stem = plotdir + f"/{key.replace('seed_layers', 'seed_mult_vs_layers')}"
            plot_seed_layers([(name, color, seed_layer_rates(acc[key], i))],
                             f"Seeds per Event by Layer Combination ({name})", stem + ".pdf")
            print_seed_layer_table(acc[key], i, name, stem + ".txt",
                                   int(np.sum(seed_nseeds_per_event)) if key == "seed_layers_all" else None)
            plot_seed_systems([(name, color, seed_system_rates(acc[key], i))],
                              f"Seeds per Event with a Hit in Each Subsystem ({name})",
                              plotdir + f"/{key.replace('seed_layers', 'seed_mult_vs_systems')}.pdf")

    if len(eff_truth_pt) > 0:
        print(f"Muons with a matched seed: {len(seed_pt_best_per_muon)}/{len(eff_truth_pt)} "
              f"({acc['n_seed_hit_matched']} by >= {SEED_MATCH_MIN_HITS} of 3 muon hits, "
              f"{acc['n_seed_dr_matched']} by closest seed within Delta R < {SEED_MATCH_MAX_DR:g})")

    if len(seed_pt_best_per_muon) > 0:
        truth_pts_s = [x[0] for x in seed_pt_best_per_muon]
        seed_pts_s  = [x[1] for x in seed_pt_best_per_muon]
        resolutions_s = [(s - t) / t for t, s in seed_pt_best_per_muon]

        pt_max_s = max(max(truth_pts_s), max(seed_pts_s)) * 1.2
        xbins_s = np.linspace(0, pt_max_s, 51)
        ybins_s = np.linspace(0, pt_max_s, 51)
        plt.figure()
        img = hist2d_lognorm(truth_pts_s, seed_pts_s, [xbins_s, ybins_s])
        if img is not None:
            plt.colorbar(img, label="Counts")
        plt.plot([0, pt_max_s], [0, pt_max_s], 'r--', linewidth=1, label="truth = seed")
        plt.xlabel(r"Truth Muon $p_T$ [GeV]")
        plt.ylabel(r"Best-matched Seed $p_T$ [GeV]")
        plt.legend()
        plt.title(rf"Truth vs Seed $p_T$ ($\geq${SEED_MATCH_MIN_HITS}/3 muon hits, else closest $\Delta R < {SEED_MATCH_MAX_DR:g}$)")
        plt.tight_layout()
        plt.savefig(plotdir + "/truth_vs_seed_pt_hist2d.pdf")
        plt.close()

        bins_res_s = np.linspace(-2, 2, 81)
        plt.figure()
        plt.hist(resolutions_s, bins=bins_res_s, color='steelblue', edgecolor='black',
                 label=f"Mean: {np.mean(resolutions_s):.3f}\nRMS: {np.std(resolutions_s):.3f}")
        plt.xlabel(r"$(p_T^{\mathrm{seed}} - p_T^{\mathrm{truth}}) / p_T^{\mathrm{truth}}$")
        plt.ylabel("Counts")
        plt.legend(fontsize=9)
        plt.title(rf"Seed $p_T$ Resolution ($\geq${SEED_MATCH_MIN_HITS}/3 muon hits, else closest $\Delta R < {SEED_MATCH_MAX_DR:g}$)")
        plt.tight_layout()
        plt.savefig(plotdir + "/seed_pt_resolution.pdf")
        plt.close()

    # === 1D histogram plots (muon-matched tracks) ===

    # --- Hit multiplicity: total (SiTracks) ---
    if len(total_nhits_sitracks) > 0:
        bins_nhit = np.arange(-0.5, 20.5, 1)
        plt.figure()
        plt.hist(total_nhits_sitracks, bins=bins_nhit, color='steelblue', edgecolor='black',
                 label=f"Mean: {np.mean(total_nhits_sitracks):.2f}\nMedian: {np.median(total_nhits_sitracks):.1f}")
        plt.yscale('log')
        plt.xlim(0, 20)
        plt.xlabel("N hits")
        plt.ylabel("Tracks")
        plt.legend(fontsize=9)
        plt.title("Hit Multiplicity (SiTracks, Muon Matched)")
        plt.tight_layout()
        plt.savefig(plotdir + "/hit_mult_sitracks.pdf")
        plt.close()

    # --- Hit multiplicity: muon-matched vs BIB overlaid (SiTracks) ---
    if len(total_nhits_sitracks) > 0 and len(num_bib_hits_sitracks) > 0:
        muon_hits_si = [nhit - nbib for nhit, nbib in zip(total_nhits_sitracks, num_bib_hits_sitracks)]
        plt.figure()
        plt.hist(muon_hits_si,          bins=bins_nhit, color='steelblue', alpha=0.6, edgecolor='black',
                 label=f"Muon-matched hits  Mean: {np.mean(muon_hits_si):.2f}, Median: {np.median(muon_hits_si):.1f}")
        plt.hist(num_bib_hits_sitracks, bins=bins_nhit, color='salmon',    alpha=0.6, edgecolor='black',
                 label=f"Non-muon hits  Mean: {np.mean(num_bib_hits_sitracks):.2f}, Median: {np.median(num_bib_hits_sitracks):.1f}")
        plt.yscale('log')
        plt.xlim(0, 20)
        plt.xlabel("N hits")
        plt.ylabel("Tracks")
        plt.legend(fontsize=9)
        plt.title("Hit Multiplicity Overlay (SiTracks, Muon Matched)")
        plt.tight_layout()
        plt.savefig(plotdir + "/hit_mult_overlay_sitracks.pdf")
        plt.close()

    # --- Track pT (SiTracks, log-spaced bins, 0-5 TeV) ---
    if len(track_reco_pt_sitracks) > 0:
        bins_pt = np.logspace(np.log10(1), np.log10(5000), 101)
        plt.figure()
        w_pt_si = np.ones(len(track_reco_pt_sitracks)) / len(track_reco_pt_sitracks)
        plt.hist(track_reco_pt_sitracks, bins=bins_pt, weights=w_pt_si, color='salmon', alpha=0.6, edgecolor='black', label="SiTracks")
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(1, 5000)
        plt.xlabel(r"$p_T$ [GeV]")
        plt.ylabel("Fraction of tracks")
        plt.legend()
        plt.title(r"Track $p_T$ (Muon Matched)")
        plt.tight_layout()
        plt.savefig(plotdir + "/track_pt_overlay.pdf")
        plt.close()

    # --- Track eta (SiTracks) ---
    if len(track_eta_sitracks) > 0:
        bins_eta = np.linspace(-3, 3, 61)
        plt.figure()
        w_eta_si = np.ones(len(track_eta_sitracks)) / len(track_eta_sitracks)
        plt.hist(track_eta_sitracks, bins=bins_eta, weights=w_eta_si, color='salmon', alpha=0.6, edgecolor='black', label="SiTracks")
        plt.yscale('log')
        plt.xlabel(r"$\eta$")
        plt.ylabel("Fraction of tracks")
        plt.legend()
        plt.title(r"Track $\eta$ (Muon Matched)")
        plt.tight_layout()
        plt.savefig(plotdir + "/track_eta_overlay.pdf")
        plt.close()

    # --- Track phi (SiTracks) ---
    if len(track_phi_sitracks) > 0:
        bins_phi = np.linspace(-np.pi, np.pi, 65)
        plt.figure()
        w_phi_si = np.ones(len(track_phi_sitracks)) / len(track_phi_sitracks)
        plt.hist(track_phi_sitracks, bins=bins_phi, weights=w_phi_si, color='salmon', alpha=0.6, edgecolor='black', label="SiTracks")
        plt.yscale('log')
        plt.xlabel(r"$\phi$")
        plt.ylabel("Fraction of tracks")
        plt.legend()
        plt.title(r"Track $\phi$ (Muon Matched)")
        plt.tight_layout()
        plt.savefig(plotdir + "/track_phi_overlay.pdf")
        plt.close()

    # --- Track d0 (SiTracks) ---
    if len(track_d0_sitracks) > 0:
        d0_max = max(abs(min(track_d0_sitracks)), abs(max(track_d0_sitracks))) * 1.1
        bins_d0 = np.linspace(-d0_max, d0_max, 81)
        plt.figure()
        w_d0_si = np.ones(len(track_d0_sitracks)) / len(track_d0_sitracks)
        plt.hist(track_d0_sitracks, bins=bins_d0, weights=w_d0_si, color='salmon', alpha=0.6, edgecolor='black', label="SiTracks")
        plt.xlabel(r"$d_0$ [mm]")
        plt.ylabel("Fraction of tracks")
        plt.legend()
        plt.title(r"Track $d_0$ (Muon Matched)")
        plt.tight_layout()
        plt.savefig(plotdir + "/track_d0_overlay.pdf")
        plt.close()

    # --- Track z0 (SiTracks) ---
    if len(track_z0_sitracks) > 0:
        z0_max = max(abs(min(track_z0_sitracks)), abs(max(track_z0_sitracks))) * 1.1
        bins_z0 = np.linspace(-z0_max, z0_max, 81)
        plt.figure()
        w_z0_si = np.ones(len(track_z0_sitracks)) / len(track_z0_sitracks)
        plt.hist(track_z0_sitracks, bins=bins_z0, weights=w_z0_si, color='salmon', alpha=0.6, edgecolor='black', label="SiTracks")
        plt.xlabel(r"$z_0$ [mm]")
        plt.ylabel("Fraction of tracks")
        plt.legend()
        plt.title(r"Track $z_0$ (Muon Matched)")
        plt.tight_layout()
        plt.savefig(plotdir + "/track_z0_overlay.pdf")
        plt.close()

    # --- # BIB hits per track (SiTracks) ---
    if len(num_bib_hits_sitracks) > 0:
        bib_max = max(num_bib_hits_sitracks) if num_bib_hits_sitracks else 20
        bins_bib = np.arange(-0.5, max(bib_max, 20) + 1.5, 1)
        plt.figure()
        w = np.ones(len(num_bib_hits_sitracks)) / len(num_bib_hits_sitracks)
        plt.hist(num_bib_hits_sitracks, bins=bins_bib, weights=w, histtype='step', linewidth=1.7,
                 color='salmon', label=f"SiTracks  mean={np.mean(num_bib_hits_sitracks):.2f}")
        plt.yscale('log')
        plt.xlabel("# non-muon (BIB) hits per track")
        plt.ylabel("Fraction of tracks")
        plt.legend(fontsize=9)
        plt.title("# BIB Hits per Track (Muon Matched)")
        plt.tight_layout()
        plt.savefig(plotdir + "/num_bib_hits_overlay.pdf")
        plt.close()

    # --- Muon-hit fraction per track (SiTracks) ---
    if len(track_percent_muon_hits_sitracks) > 0:
        bins_mhf = np.linspace(0, 1, 21)
        plt.figure()
        w = np.ones(len(track_percent_muon_hits_sitracks)) / len(track_percent_muon_hits_sitracks)
        plt.hist(track_percent_muon_hits_sitracks, bins=bins_mhf, weights=w, histtype='step', linewidth=1.7,
                 color='salmon', label=f"SiTracks  mean={np.mean(track_percent_muon_hits_sitracks):.3f}")
        plt.yscale('log')
        plt.xlabel("Muon-matched hit fraction")
        plt.ylabel("Fraction of tracks")
        plt.legend(fontsize=9)
        plt.title("Muon-Matched Hit Fraction per Track")
        plt.tight_layout()
        plt.savefig(plotdir + "/muonhitfrac_overlay.pdf")
        plt.close()

    # --- ID verification: cellID-decoded (system, side, layer) ---
    if GENERATE_ID_DIAG and id_diag_counts:
        # Summary table to stdout
        print("[ID DIAG] cellID decoding: (system, side, layer) -> hit count (all events)")
        print(f"[ID DIAG] system map (EDM4hep): 1=VXDBarrel 2=VXDEndcap 3=ITBarrel 4=ITEndcap 5=OTBarrel 6=OTEndcap")
        for key in sorted(id_diag_counts.keys()):
            sys_id, side_id, layer_id = key
            #print(f"  system={sys_id:2d}  side={side_id}  layer={layer_id:3d}  count={id_diag_counts[key]}")
        unique_sys   = sorted({k[0] for k in id_diag_counts})
        unique_pairs = sorted({(k[0], k[2]) for k in id_diag_counts})
        print(f"[ID DIAG] unique system IDs present: {unique_sys}")
        print(f"[ID DIAG] unique (system, layer) pairs: {unique_pairs}")
        print(f"[ID DIAG] random sample sizes per system: "
              + ", ".join(f"sys={s}:{len(id_diag_sample[s][0])}/{id_diag_seen_per_sys[s]}"
                          for s in sorted(id_diag_sample)))

        _sample_sys = sorted(id_diag_sample)
        if _sample_sys:
            r_arr     = np.concatenate([id_diag_sample[s][1] for s in _sample_sys])
            z_arr     = np.concatenate([id_diag_sample[s][2] for s in _sample_sys])
            sys_arr   = np.concatenate([np.full(len(id_diag_sample[s][0]), s) for s in _sample_sys])
            side_arr  = np.concatenate([id_diag_sample[s][3] for s in _sample_sys])
            layer_arr = np.concatenate([id_diag_sample[s][4] for s in _sample_sys])

            # Panel 1: colored by system (subdetector)
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            for sys_id in sorted(np.unique(sys_arr)):
                m = sys_arr == sys_id
                axs[0].scatter(z_arr[m], r_arr[m], s=4, alpha=0.5,
                               label=f"sys={sys_id} ({system_to_relname.get(sys_id, '?')})")
            axs[0].set_xlabel("z [mm]"); axs[0].set_ylabel("r [mm]")
            axs[0].set_title("Hits in r–z, colored by cellID `system`")
            axs[0].legend(fontsize=7, markerscale=3, loc='best')
            axs[0].grid(True, linestyle=':', alpha=0.5)

            # Panel 2: colored by layer
            for layer_id in sorted(np.unique(layer_arr)):
                m = layer_arr == layer_id
                axs[1].scatter(z_arr[m], r_arr[m], s=4, alpha=0.5, label=f"layer={layer_id}")
            axs[1].set_xlabel("z [mm]"); axs[1].set_ylabel("r [mm]")
            axs[1].set_title("Hits in r–z, colored by cellID `layer`")
            axs[1].legend(fontsize=6, markerscale=3, ncol=2, loc='best')
            axs[1].grid(True, linestyle=':', alpha=0.5)

            # Panel 3: colored by combined (system, side) — proxy for ACTS-style 'volume'
            combo_arr = sys_arr * 10 + side_arr
            for combo in sorted(np.unique(combo_arr)):
                m = combo_arr == combo
                axs[2].scatter(z_arr[m], r_arr[m], s=4, alpha=0.5,
                               label=f"sys.side={combo // 10}.{combo % 10}")
            axs[2].set_xlabel("z [mm]"); axs[2].set_ylabel("r [mm]")
            axs[2].set_title("Hits in r–z, colored by (system, side)")
            axs[2].legend(fontsize=7, markerscale=3, loc='best')
            axs[2].grid(True, linestyle=':', alpha=0.5)

            fig.suptitle(f"cellID ID verification  ({len(r_arr)} hits, randomly sampled per subdetector)", fontsize=13)
            fig.tight_layout()
            plt.savefig(plotdir + "/id_verification_rz.pdf")
            plt.close(fig)

    # --- z = 0 module-seam check -------------------------------------------
    # The IT/OT barrels tile their modules with an even nz, so a module edge
    # lands exactly on z = 0; the VXD barrel (odd nmodules = 5) has a module
    # centred there instead. A track at theta = 90 deg crosses that seam in all
    # six IT+OT barrel layers at once. These two panels show it directly: a
    # zoomed r-z scatter, and a fine z profile per barrel subsystem where the
    # seam appears as an empty bin at z = 0.
    if len(zseam_z):
        _zz = np.array(zseam_z)
        _zr = np.array(zseam_r)
        _zs_sys = np.array(zseam_sys)
        _barrels = [(1, "VXD Barrel (odd nmodules -> module centred on z=0)"),
                    (3, "IT Barrel (even nz -> module EDGE on z=0)"),
                    (5, "OT Barrel (even nz -> module EDGE on z=0)")]

        # Panel A: zoomed r-z scatter, +-ZSEAM_ZOOM_HALF around z = 0.
        _m_zoom = np.abs(_zz) <= ZSEAM_ZOOM_HALF
        fig, ax = plt.subplots(figsize=(9, 6))
        if _m_zoom.any():
            for _sid in sorted(np.unique(_zs_sys[_m_zoom])):
                _m = _m_zoom & (_zs_sys == _sid)
                ax.scatter(_zz[_m], _zr[_m], s=6, alpha=0.6,
                           label=f"sys={_sid} ({system_to_relname.get(_sid, '?')}), {_m.sum()} hits")
            ax.legend(fontsize=7, markerscale=3, loc='best')
        else:
            ax.text(0.5, 0.5, f"no hits with |z| < {ZSEAM_ZOOM_HALF} mm",
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.axvline(0.0, color='red', linestyle='--', linewidth=1.2, alpha=0.8,
                   label="z = 0 (module seam in IT/OT)")
        ax.set_xlim(-ZSEAM_ZOOM_HALF, ZSEAM_ZOOM_HALF)
        ax.set_xlabel("z [mm]")
        ax.set_ylabel("r [mm]")
        ax.set_title(f"Merged tracker hits within $\\pm${ZSEAM_ZOOM_HALF:g} mm of z = 0  "
                     f"({int(_m_zoom.sum())} hits, {i} events)")
        ax.grid(True, linestyle=':', alpha=0.5)
        fig.tight_layout()
        plt.savefig(plotdir + "/zseam_rz_zoom.pdf")
        plt.close(fig)

        # Panel B: fine z profile per barrel subsystem over the collected window.
        # 0.1 mm bins resolve a seam far narrower than the 30.1 mm module pitch.
        _zbins = np.arange(-ZSEAM_COLLECT_HALF, ZSEAM_COLLECT_HALF + 0.1, 0.1)
        fig, axs = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
        print(f"[ZSEAM] hits within |z| < {ZSEAM_COLLECT_HALF} mm, "
              f"and within |z| < {ZSEAM_ZOOM_HALF} mm, per subsystem:")
        for ax, (_sid, _title) in zip(axs, _barrels):
            _m = _zs_sys == _sid
            _n_near = int((_m & (np.abs(_zz) <= ZSEAM_ZOOM_HALF)).sum())
            print(f"[ZSEAM]   sys={_sid} ({system_to_relname.get(_sid, '?'):10s}): "
                  f"{int(_m.sum()):7d} within collect window, {_n_near:7d} within zoom window")
            if _m.any():
                ax.hist(_zz[_m], bins=_zbins, histtype='step', linewidth=1.2, color='steelblue')
            ax.axvline(0.0, color='red', linestyle='--', linewidth=1.0, alpha=0.8)
            ax.set_ylabel("Hits / 0.1 mm")
            ax.set_title(f"{_title}   [{int(_m.sum())} hits, {_n_near} with |z| < {ZSEAM_ZOOM_HALF:g} mm]",
                         fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.5)
        axs[-1].set_xlabel("z [mm]")
        fig.suptitle(f"Hit z profile around the module seam at z = 0  ({i} events)", fontsize=13)
        fig.tight_layout()
        plt.savefig(plotdir + "/zseam_z_profile.pdf")
        plt.close(fig)

    # --- Truth muon kinematics: pT, eta, phi, theta ------------------------
    # Sanity check on what the gun actually produced, independent of any
    # reconstruction. If these are empty the PDG filter (SIGNAL_PDG) is wrong.
    if eff_truth_pt:
        fig, axs = plt.subplots(2, 2, figsize=(11, 8))
        _panels = [
            (axs[0][0], eff_truth_pt, None, r"Truth $p_T$ [GeV]", "Truth $p_T$", True),
            (axs[0][1], eff_truth_eta, np.linspace(-3, 3, 61), r"Truth $\eta$", "Truth $\\eta$", False),
            (axs[1][0], eff_truth_phi, np.linspace(-np.pi, np.pi, 61), r"Truth $\phi$ [rad]", "Truth $\\phi$", False),
            (axs[1][1], eff_truth_theta_deg, np.linspace(0, 180, 181), r"Truth $\theta$ [deg]", "Truth $\\theta$", False),
        ]
        for ax, vals, bins, xlabel, title, logx in _panels:
            arr = np.array(vals)
            if logx:
                _lo = max(arr[arr > 0].min(), 1e-3) if (arr > 0).any() else 1e-3
                bins = np.logspace(np.log10(_lo * 0.9), np.log10(arr.max() * 1.1), 61)
                ax.set_xscale('log')
            ax.hist(arr, bins=bins, histtype='step', linewidth=1.5, color='steelblue')
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Truth particles")
            ax.set_title(f"{title}   mean={arr.mean():.3g}, median={np.median(arr):.3g}, N={arr.size}",
                         fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.6)
        fig.suptitle(f"Truth kinematics, |PDG| = {SIGNAL_PDG}  ({i} events)", fontsize=13)
        fig.tight_layout()
        plt.savefig(plotdir + "/truth_kinematics.pdf")
        plt.close(fig)
        print(f"[TRUTH] |PDG|={SIGNAL_PDG}: {len(eff_truth_pt)} particles in {i} events; "
              f"pT [{min(eff_truth_pt):.2f}, {max(eff_truth_pt):.2f}] GeV, "
              f"theta [{min(eff_truth_theta_deg):.2f}, {max(eff_truth_theta_deg):.2f}] deg")
    else:
        print(f"[TRUTH] WARNING: no MCParticles with |PDG| = {SIGNAL_PDG} found in {i} events. "
              f"Every efficiency will read 0%. Set the SIGNAL_PDG env var to the gun's species.")

    # --- Tracker r-z occupancy: hits per event in each (z, r) bin ---
    if occupancy_hist.sum() > 0 and i > 0:
        occ_per_event = occupancy_hist.astype(np.float64) / float(i)
        nz_mask       = occ_per_event > 0
        if nz_mask.any():
            vmin = occ_per_event[nz_mask].min()
            vmax = occ_per_event.max()
        else:
            vmin, vmax = 1e-3, 1.0
        fig, ax = plt.subplots(figsize=(12, 6))
        Z_edges, R_edges = OCC_Z_EDGES, OCC_R_EDGES
        # ROOT-style rainbow palette (kRainbow ≈ matplotlib 'jet'); empty bins white.
        occ_cmap = plt.get_cmap('jet').copy()
        occ_cmap.set_under('white')
        occ_cmap.set_bad('white')
        # pcolormesh uses (x_edges, y_edges, C[ny, nx]); occupancy is indexed [z, r]
        # so transpose to put r on the y-axis and z on the x-axis.
        img = ax.pcolormesh(Z_edges, R_edges, occ_per_event.T,
                            norm=LogNorm(vmin=vmin, vmax=vmax), cmap=occ_cmap, shading='auto')
        fig.colorbar(img, ax=ax)
        ax.set_xlabel("z [mm]")
        ax.set_ylabel("r [mm]")
        ax.set_title(f"Tracker r–z occupancy (hits / event, {i} events)")
        ax.set_xlim(Z_edges[0], Z_edges[-1])
        ax.set_ylim(R_edges[0], R_edges[-1])
        fig.tight_layout()
        plt.savefig(plotdir + "/tracker_occupancy_rz.pdf")
        plt.close(fig)

    # --- Truth muon eta distribution ---
    if len(eff_truth_eta) > 0:
        plt.figure()
        plt.hist(eff_truth_eta, bins=50, range=(-3, 3), color='steelblue', edgecolor='black',
                 label=f"Mean: {np.mean(eff_truth_eta):.2f}\nMedian: {np.median(eff_truth_eta):.2f}")
        plt.xlabel(r"Truth Muon $\eta$")
        plt.ylabel("Muons")
        plt.legend(fontsize=9)
        plt.title("Truth Muon $\\eta$ Distribution")
        plt.tight_layout()
        plt.savefig(plotdir + "/truth_muon_eta.pdf")
        plt.close()

    # --- Tracking efficiency plots ---
    if len(eff_truth_eta) > 0:
        eta_arr = np.array(eff_truth_eta)
        si_arr = np.array(eff_matched_si, dtype=float)
        alltracks_arr = np.array(eff_matched_alltracks, dtype=float)

        int_si        = si_arr.mean()
        int_alltracks = alltracks_arr.mean() if alltracks_arr.size else float('nan')
        barrel_mask = np.abs(eta_arr) < 0.6
        endcap_mask = ~barrel_mask
        barrel_si        = si_arr[barrel_mask].mean()        if barrel_mask.sum() > 0 else float('nan')
        barrel_alltracks = alltracks_arr[barrel_mask].mean() if barrel_mask.sum() > 0 and alltracks_arr.size else float('nan')
        endcap_si        = si_arr[endcap_mask].mean()        if endcap_mask.sum() > 0 else float('nan')
        endcap_alltracks = alltracks_arr[endcap_mask].mean() if endcap_mask.sum() > 0 and alltracks_arr.size else float('nan')

        # Efficiency vs eta
        eta_bins = np.linspace(-2.5, 2.5, 26)
        c_eta, eff_si_eta, err_si_eta = compute_efficiency(eff_truth_eta, eff_matched_si, eta_bins)
        c_eta_a, eff_alltracks_eta, err_alltracks_eta = compute_efficiency(eff_truth_eta, eff_matched_alltracks, eta_bins)
        plt.figure()
        plt.errorbar(c_eta_a, eff_alltracks_eta, yerr=err_alltracks_eta, fmt='D-', color='purple',
                     label=f"AllTracks (int: {100*int_alltracks:.1f}%, brl: {100*barrel_alltracks:.1f}%, EC: {100*endcap_alltracks:.1f}%)",
                     capsize=3)
        plt.errorbar(c_eta, eff_si_eta, yerr=err_si_eta, fmt='o-', color='steelblue',
                     label=f"SiTracks (int: {100*int_si:.1f}%, brl: {100*barrel_si:.1f}%, EC: {100*endcap_si:.1f}%)",
                     capsize=3)
        plt.axvline(-0.6, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        plt.axvline( 0.6, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        plt.text(-1.55, 0.03, r"Endcap ($|\eta|>0.6$)", fontsize=7, color='gray', ha='center')
        plt.text( 0.0,  0.03, r"Barrel ($|\eta|<0.6$)", fontsize=7, color='gray', ha='center')
        plt.text( 1.55, 0.03, r"Endcap ($|\eta|>0.6$)", fontsize=7, color='gray', ha='center')
        plt.xlabel(r"Truth Muon $\eta$")
        plt.ylabel("Tracking Efficiency")
        plt.xlim(-2.5, 2.5)
        eff_ticks(eta_x=True)
        plt.legend(fontsize=8, loc='lower center')
        plt.title(r"Tracking Efficiency vs Truth $\eta$")
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.tight_layout()
        plt.savefig(plotdir + "/efficiency_vs_eta.pdf")
        plt.close()

        # Efficiency vs theta
        theta_rad = np.radians(eff_truth_theta_deg)
        c_th, eff_si_th, err_si_th = compute_efficiency(theta_rad, eff_matched_si, THETA_BINS)
        c_th_a, eff_alltracks_th, err_alltracks_th = compute_efficiency(theta_rad, eff_matched_alltracks, THETA_BINS)
        plt.figure()
        plt.errorbar(c_th_a, eff_alltracks_th, yerr=err_alltracks_th, fmt='D-', color='purple',
                     label=f"AllTracks (int: {100*int_alltracks:.1f}%, brl: {100*barrel_alltracks:.1f}%, EC: {100*endcap_alltracks:.1f}%)",
                     capsize=3)
        plt.errorbar(c_th, eff_si_th, yerr=err_si_th, fmt='o-', color='steelblue',
                     label=f"SiTracks (int: {100*int_si:.1f}%, brl: {100*barrel_si:.1f}%, EC: {100*endcap_si:.1f}%)",
                     capsize=3)
        plt.axvline(TH_BRL, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        plt.axvline(np.pi - TH_BRL, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        plt.text(TH_BRL / 2, 0.03, r"Endcap ($|\eta|>0.6$)", fontsize=7, color='gray', ha='center')
        plt.text(np.pi / 2, 0.03, r"Barrel ($|\eta|<0.6$)", fontsize=7, color='gray', ha='center')
        plt.text(np.pi - TH_BRL / 2, 0.03, r"Endcap ($|\eta|>0.6$)", fontsize=7, color='gray', ha='center')
        plt.xlabel(r"Truth Muon $\theta$ [rad]")
        plt.ylabel("Tracking Efficiency")
        plt.xlim(0, np.pi)
        eff_ticks()
        plt.legend(fontsize=8, loc='lower center')
        plt.title(r"Tracking Efficiency vs Truth $\theta$")
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.tight_layout()
        plt.savefig(plotdir + "/efficiency_vs_theta.pdf")
        plt.close()

        # Efficiency vs truth d0, z0 and production vertex r_xy, and the truth
        # distributions themselves (values outside the axes in the edge bins)
        for key, bins, xlabel, tag in DISPLACEMENT_EFF:
            x = np.clip(acc[key], bins[0], bins[-1])
            plt.figure()
            for flags, fmt, color, name in [(eff_matched_alltracks, 'D-', 'purple', "AllTracks"),
                                            (eff_matched_si, 'o-', 'steelblue', "SiTracks")]:
                c, e, err = compute_efficiency(x, flags, bins)
                plt.errorbar(c, e, yerr=err, fmt=fmt, color=color, capsize=3,
                             label=f"{name} ({100*np.mean(flags):.1f}%)")
            plt.xlabel(xlabel)
            plt.ylabel("Tracking Efficiency")
            plt.xlim(bins[0], bins[-1])
            plt.ylim(0, 1.05)
            plt.legend(fontsize=8, loc='lower center')
            plt.title(f"Tracking Efficiency vs {xlabel.replace('Truth Muon ', '').split(' [')[0]}")
            plt.grid(True, linestyle=":", alpha=0.7)
            plt.tight_layout()
            plt.savefig(plotdir + f"/efficiency_vs_{tag}.pdf")
            plt.close()

            v = np.asarray(acc[key], dtype=float)
            plt.figure()
            plt.hist(x, bins=bins, color='steelblue', edgecolor='black', alpha=0.7,
                     label=f"N={len(v)}, range [{np.min(v):.3g}, {np.max(v):.3g}] mm")
            plt.xlabel(xlabel + " (edge bins: overflow)")
            plt.ylabel("Muons")
            plt.title(xlabel.split(' [')[0])
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(plotdir + f"/truth_{tag}.pdf")
            plt.close()

        # 2D truth maps (generator check) and efficiency vs (d0, z0)
        d0t, z0t = np.asarray(acc["eff_truth_d0"], dtype=float), np.asarray(acc["eff_truth_z0"], dtype=float)
        rxy, vz = np.asarray(acc["eff_truth_vrxy"], dtype=float), np.asarray(acc["eff_truth_vz"], dtype=float)
        for xv, yv, xb, yb, xl, yl, tag in [
            (d0t, z0t, DISPLACEMENT_EFF[0][1], DISPLACEMENT_EFF[1][1],
             r"Truth Muon $d_0$ [mm]", r"Truth Muon $z_0$ [mm]", "d0_vs_z0"),
            (rxy, vz, DISPLACEMENT_EFF[2][1], np.linspace(-300, 300, 61),
             r"Truth Muon production vertex $r_{xy}$ [mm]", r"Truth Muon production vertex $z$ [mm]", "vrxy_vs_vz"),
        ]:
            plt.figure(figsize=(7, 5.5))
            img = hist2d_lognorm(np.clip(xv, xb[0], xb[-1]), np.clip(yv, yb[0], yb[-1]), [xb, yb])
            if img is not None:
                plt.colorbar(img, label="Muons")
            plt.xlabel(xl + " (edge bins: overflow)")
            plt.ylabel(yl + " (edge bins: overflow)")
            plt.title(f"Truth {tag.replace('_vs_', ' vs ')} (N={len(xv)})")
            plt.tight_layout()
            plt.savefig(plotdir + f"/truth_{tag}.pdf")
            plt.close()
        # Efficiency maps vs (d0, z0) and vs production vertex (r_xy, z); the
        # nozzle (tungsten for |z| > 6 cm, r < ~10 + 0.18 (|z| - 60) mm) would
        # show as a low-efficiency wedge at small r_xy and large |z|.
        for xv, yv, xb, yb, xl, yl, tag in [
            (d0t, z0t, np.linspace(-300, 300, 21), np.linspace(-300, 300, 21),
             r"Truth Muon $d_0$ [mm]", r"Truth Muon $z_0$ [mm]", "d0_vs_z0"),
            (rxy, vz, np.linspace(0, 300, 21), np.linspace(-300, 300, 41),
             r"Production vertex $r_{xy}$ [mm]", r"Production vertex $z$ [mm]", "vrxy_vs_vz"),
        ]:
            xs, ys = np.clip(xv, xb[0], xb[-1]), np.clip(yv, yb[0], yb[-1])
            tot, _, _ = np.histogram2d(xs, ys, bins=[xb, yb])
            for flags, name in [(eff_matched_alltracks, "alltracks"), (eff_matched_si, "sitracks")]:
                hit, _, _ = np.histogram2d(xs, ys, bins=[xb, yb], weights=np.asarray(flags, dtype=float))
                with np.errstate(divide="ignore", invalid="ignore"):
                    eff = np.where(tot >= EFF2D_MIN_MUONS, hit / tot, np.nan)
                plt.figure(figsize=(7, 5.5))
                m = plt.pcolormesh(xb, yb, np.ma.masked_invalid(eff).T, vmin=0, vmax=1, cmap="viridis")
                plt.colorbar(m, label="Tracking Efficiency")
                plt.xlabel(xl + " (edge bins: overflow)")
                plt.ylabel(yl + " (edge bins: overflow)")
                plt.title(f"Efficiency, {'AllTracks' if name == 'alltracks' else 'SiTracks'} "
                          f"(blank: < {EFF2D_MIN_MUONS} muons)")
                plt.tight_layout()
                plt.savefig(plotdir + f"/efficiency_{tag}_{name}.pdf")
                plt.close()

        # Efficiency vs pT (log-spaced bins, 2-2500 GeV, ~10 bins/decade)
        pt_bins = np.logspace(np.log10(2), np.log10(2500), 33)
        c_pt, eff_si_pt, err_si_pt = compute_efficiency(eff_truth_pt, eff_matched_si, pt_bins)
        c_pt_a, eff_alltracks_pt, err_alltracks_pt = compute_efficiency(eff_truth_pt, eff_matched_alltracks, pt_bins)
        plt.figure()
        plt.errorbar(c_pt_a, eff_alltracks_pt, yerr=err_alltracks_pt, fmt='D-', color='purple',
                     label=f"AllTracks (int: {100*int_alltracks:.1f}%, brl: {100*barrel_alltracks:.1f}%, EC: {100*endcap_alltracks:.1f}%)",
                     capsize=3)
        plt.errorbar(c_pt, eff_si_pt, yerr=err_si_pt, fmt='o-', color='steelblue',
                     label=f"SiTracks (int: {100*int_si:.1f}%, brl: {100*barrel_si:.1f}%, EC: {100*endcap_si:.1f}%)",
                     capsize=3)
        plt.xscale('log')
        plt.xlabel(r"Truth Muon $p_T$ [GeV]")
        plt.ylabel("Tracking Efficiency")
        plt.xlim(2, 2500)
        eff_ticks()
        plt.legend(fontsize=8, loc='lower center')
        plt.title(r"Tracking Efficiency vs Truth $p_T$")
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.tight_layout()
        plt.savefig(plotdir + "/efficiency_vs_pt.pdf")
        plt.close()

        # Efficiency vs pT, 0-50 GeV
        pt_clipped = np.clip(eff_truth_pt, PT_BINS_0_50[0], None)
        c_pt, eff_si_pt, err_si_pt = compute_efficiency(pt_clipped, eff_matched_si, PT_BINS_0_50)
        c_pt_a, eff_alltracks_pt, err_alltracks_pt = compute_efficiency(pt_clipped, eff_matched_alltracks, PT_BINS_0_50)
        plt.figure()
        plt.errorbar(c_pt_a, eff_alltracks_pt, yerr=err_alltracks_pt, fmt='D-', color='purple',
                     label=f"AllTracks (int: {100*int_alltracks:.1f}%, brl: {100*barrel_alltracks:.1f}%, EC: {100*endcap_alltracks:.1f}%)",
                     capsize=3)
        plt.errorbar(c_pt, eff_si_pt, yerr=err_si_pt, fmt='o-', color='steelblue',
                     label=f"SiTracks (int: {100*int_si:.1f}%, brl: {100*barrel_si:.1f}%, EC: {100*endcap_si:.1f}%)",
                     capsize=3)
        plt.xscale('log')
        plt.xlabel(r"Truth Muon $p_T$ [GeV]")
        plt.ylabel("Tracking Efficiency")
        plt.xlim(PT_BINS_0_50[0], PT_BINS_0_50[-1])
        eff_ticks()
        plt.legend(fontsize=8, loc='lower center')
        plt.title(r"Tracking Efficiency vs Truth $p_T$")
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.tight_layout()
        plt.savefig(plotdir + "/efficiency_vs_pt_0_50.pdf")
        plt.close()

        # Efficiency by stage: the muon has a seed (all 3 hits its own), an
        # AllTracks track, a SiTracks track
        for xkey, skey, bins, xlabel, logx, tag in [
            ("eff_truth_eta", "seed_eff_eta", np.linspace(-2.5, 2.5, 26), r"Truth Muon $\eta$", False, "eta"),
            ("eff_truth_pt", "seed_eff_pt", PT_BINS_0_50, r"Truth Muon $p_T$ [GeV]", True, "pt_0_50"),
        ]:
            plt.figure()
            if acc["seed_eff_matched"]:
                xs = np.clip(acc[skey], bins[0], None) if logx else acc[skey]
                c, e, err = compute_efficiency(xs, acc["seed_eff_matched"], bins)
                plt.errorbar(c, e, yerr=err, fmt='s-', color='darkorange', capsize=3,
                             label=f"Seed, 3/3 muon hits ({100*np.mean(acc['seed_eff_matched']):.1f}%)")
            x = np.clip(acc[xkey], bins[0], None) if logx else acc[xkey]
            for flags, fmt, color, name in [(eff_matched_alltracks, 'D-', 'purple', "AllTracks"),
                                            (eff_matched_si, 'o-', 'steelblue', "SiTracks")]:
                c, e, err = compute_efficiency(x, flags, bins)
                plt.errorbar(c, e, yerr=err, fmt=fmt, color=color, capsize=3,
                             label=f"{name} ({100*np.mean(flags):.1f}%)")
            if logx:
                plt.xscale('log')
            plt.xlabel(xlabel)
            plt.ylabel("Efficiency")
            plt.xlim(bins[0], bins[-1])
            eff_ticks()
            plt.legend(fontsize=8, loc='lower center')
            plt.title("Efficiency by Stage: Seed, AllTracks, SiTracks")
            plt.grid(True, linestyle=":", alpha=0.7)
            plt.tight_layout()
            plt.savefig(plotdir + f"/efficiency_stages_vs_{tag}.pdf")
            plt.close()

    # --- Average hits per muon-matched track vs. layer number ---
    n_sitracks = len(total_nhits_sitracks)
    if n_sitracks > 0:
        fig, axs = plt.subplots(3, 2, figsize=(11, 12))
        for ax, (sys_id, sys_name) in zip(axs.flat, _SYS_NAMES.items()):
            lh_si = layer_hits_sitracks[sys_id]
            all_layers = sorted(lh_si)
            if not all_layers:
                ax.set_title(sys_name)
                continue
            x      = np.array(all_layers)
            avg_si = np.array([lh_si.get(l, 0) / n_sitracks for l in all_layers])
            width  = 0.6
            ax.bar(x, avg_si, width=width, label="SiTracks", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Avg hits / muon track")
            ax.set_title(sys_name)
            ax.legend()
            ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
        fig.suptitle("Average Hits per Muon Track vs. Layer (Muon Matched)", fontsize=13)
        fig.tight_layout()
        plt.savefig(plotdir + "/avg_hits_per_layer.pdf")
        plt.close(fig)

    # --- pT / d0 / z0 residuals and pulls (muon matched), one page per collection ---
    for suffix, file_suffix, title_tag in RESIDUAL_STAGES:
        if not any(len(acc[f"track_{key}_{suffix}"]) for key, *_ in RESIDUAL_PLOTS):
            continue
        fig, axes = plt.subplots(2, 4, figsize=(20, 9))
        axes[0, 3].axis("off")   # residuals on the top row, pulls below
        for ax, (key, _, title, xlabel, bins, pull) in zip([*axes[0, :3], *axes[1]], RESIDUAL_PLOTS):
            residual_hist(ax, acc[f"track_{key}_{suffix}"], bins, "", color="salmon")
            if pull:
                unit_gaussian(ax, bins)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Fraction of tracks")
            ax.set_yscale("log")
            ax.set_title(title)
            ax.legend(fontsize=7)
        fig.suptitle(f"Residuals and Pulls ({title_tag}, Muon Matched)", fontsize=13)
        fig.tight_layout()
        plt.savefig(plotdir + f"/residuals_pulls_{file_suffix}.pdf")
        plt.close(fig)

    return {
        # hit counts by subdetector
        "pixel_nhits_sitracks":         list(pixel_nhits_sitracks),
        "inner_nhits_sitracks":         list(inner_nhits_sitracks),
        "outer_nhits_sitracks":         list(outer_nhits_sitracks),
        "total_nhits_sitracks":         list(total_nhits_sitracks),
        # same breakdown at the AllTracks stage (pre-dedup, pre-filter), so the
        # overlay can show what the CKF produced before the selection.
        "pixel_nhits_alltracks":        list(pixel_nhits_alltracks),
        "inner_nhits_alltracks":        list(inner_nhits_alltracks),
        "outer_nhits_alltracks":        list(outer_nhits_alltracks),
        "total_nhits_alltracks":        list(total_nhits_alltracks),
        # muon vs BIB hit counts
        "num_bib_hits_sitracks":        list(num_bib_hits_sitracks),
        # chi2
        "track_chi2_reduced_sitracks":  list(track_chi2_reduced_sitracks),
        # pT
        "track_reco_pt_sitracks":       list(track_reco_pt_sitracks),
        "track_matched_pt_sitracks":    list(track_matched_pt_sitracks),
        # kinematic parameters
        "track_eta_sitracks":           list(track_eta_sitracks),
        "track_phi_sitracks":           list(track_phi_sitracks),
        "track_d0_sitracks":            list(track_d0_sitracks),
        "track_z0_sitracks":            list(track_z0_sitracks),
        # hit quality
        "track_percent_muon_hits_sitracks":list(track_percent_muon_hits_sitracks),
        # pT / d0 / z0 residuals and pulls, SiTracks and AllTracks
        **{f"track_{key}_{suffix}": list(acc[f"track_{key}_{suffix}"])
           for key, *_ in RESIDUAL_PLOTS for suffix, *_ in RESIDUAL_STAGES},
        # efficiency
        "eff_truth_eta":                list(eff_truth_eta),
        "eff_truth_pt":                 list(eff_truth_pt),
        "eff_truth_theta_deg":          list(eff_truth_theta_deg),
        "eff_truth_d0":                 list(acc["eff_truth_d0"]),
        "eff_truth_z0":                 list(acc["eff_truth_z0"]),
        "eff_truth_vrxy":               list(acc["eff_truth_vrxy"]),
        "seed_eff_eta":                 list(acc["seed_eff_eta"]),
        "seed_eff_pt":                  list(acc["seed_eff_pt"]),
        "seed_eff_matched":             list(acc["seed_eff_matched"]),
        "eff_matched_si":               list(eff_matched_si),
        "eff_matched_alltracks":        list(eff_matched_alltracks),
        # event-level multiplicity
        "muon_matched_sitracks_per_event": list(muon_matched_sitracks_per_event),
        "muon_si_nhits_per_event":         list(acc["muon_si_nhits_per_event"]),
        "all_tracks_mult_per_event":       list(all_tracks_mult_per_event),
        "deduped_tracks_mult_per_event":   list(deduped_tracks_mult_per_event),
        "sitracks_mult_per_event":         list(sitracks_mult_per_event),
        # per-layer hit counts (for avg hits per layer plots)
        "layer_hits_sitracks": {sid: dict(layer_hits_sitracks[sid]) for sid in range(1, 7)},
        "n_matched_sitracks": len(total_nhits_sitracks),
        # seed tracks
        "seed_pt_all":             seed_pt_all,
        "seed_eta_all":            seed_eta_all,
        "seed_nseeds_per_event":   list(seed_nseeds_per_event),
        "seed_pt_best_per_muon":   list(seed_pt_best_per_muon),
        # hit occupancy, for the nominal vs extended-timing ratio plots
        "n_events":                i,
        "occupancy_hist":          occupancy_hist,
        "layer_nhits_mean": {k: float(np.mean([ev.get(k, 0) for ev in layer_nhits_per_event]))
                             for k in {k for ev in layer_nhits_per_event for k in ev}},
        # fake SiTracks
        "fake_mult_per_event":     list(fake_mult_per_event),
        "fake_pt":                 fake_pt,
        "fake_eta":                fake_eta,
        # pixel / inner / outer / total hits of fake SiTracks and AllTracks
        **{f"fake_{part}_nhits_{stage}": acc[f"fake_{part}_nhits_{stage}"]
           for part in ("pixel", "inner", "outer") for stage in ("sitracks", "alltracks")},
        **{f"fake_total_nhits_{stage}": sum(acc[f"fake_{part}_nhits_{stage}"].astype(np.int16)
                                            for part in ("pixel", "inner", "outer"))
           for stage in ("sitracks", "alltracks")},
        # duplicate SiTracks
        "dup_mult_per_event":      list(acc["dup_mult_per_event"]),
        "dup_pt":                  acc["dup_pt"],
        "dup_eta":                 acc["dup_eta"],
        # seed |d0| / max doublet |zOrigin|
        **{k: acc[k] for k in ("seed_d0_all", "seed_zorigin_all", "seed_d0_mu", "seed_zorigin_mu", "seed_mu_truth_pt",
                               "seed_layers_all", "seed_layers_mu")},
    }


# Bins with fewer nominal hits than this (summed over all events) are left
# blank in the occupancy ratio map rather than drawn as noise.
OCC_RATIO_MIN_COUNTS = 50


def plot_timing_ratios(base_name, nom, ext):
    """Extended / nominal timing window, per r-z bin and per layer.

    Side-by-side occupancy maps are a poor way to see this: on a log colour
    scale spanning ~4 decades, a factor 1.4 is ~4% of the colour bar, and each
    map's autoscaled range moves with its own data. A ratio shows it directly.
    """
    if not nom.get("n_events") or not ext.get("n_events"):
        return
    occ_nom = nom["occupancy_hist"] / nom["n_events"]
    occ_ext = ext["occupancy_hist"] / ext["n_events"]

    # --- r-z ratio map ---
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(nom["occupancy_hist"] >= OCC_RATIO_MIN_COUNTS, occ_ext / occ_nom, np.nan)
    finite = ratio[np.isfinite(ratio)]
    if finite.size:
        lo, hi = np.percentile(finite, [1, 99])
        lo, hi = min(lo, 0.99), max(hi, 1.01)   # keep 1 strictly inside the range
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("white")
        fig, ax = plt.subplots(figsize=(12, 6))
        img = ax.pcolormesh(OCC_Z_EDGES, OCC_R_EDGES, np.ma.masked_invalid(ratio).T,
                            norm=TwoSlopeNorm(vmin=lo, vcenter=1.0, vmax=hi),
                            cmap=cmap, shading="auto")
        fig.colorbar(img, ax=ax, label="extended / nominal (hits per event)")
        ax.set_xlabel("z [mm]")
        ax.set_ylabel("r [mm]")
        ax.set_title(f"Occupancy ratio, extended / nominal timing  "
                     f"(total {occ_ext.sum() / occ_nom.sum():.2f}; "
                     f"{ext['n_events']} / {nom['n_events']} events; "
                     f"bins with < {OCC_RATIO_MIN_COUNTS} nominal hits blank)")
        fig.tight_layout()
        plt.savefig(os.path.join(OVERLAY_DIR, f"occupancy_ratio_rz_{base_name}.pdf"))
        plt.close(fig)

    # --- per-layer ratio ---
    keys = sorted(k for k in nom["layer_nhits_mean"] if nom["layer_nhits_mean"][k] > 0)
    if not keys:
        return
    vals = [ext["layer_nhits_mean"].get(k, 0.0) / nom["layer_nhits_mean"][k] for k in keys]
    print(f"[TIMING] hits per event per layer, extended / nominal ({base_name}):")
    for (sys_id, layer), v in zip(keys, vals):
        print(f"  {system_to_relname.get(sys_id, f'system{sys_id}'):<10s} layer {layer:>2d}  "
              f"{nom['layer_nhits_mean'][(sys_id, layer)]:>11.1f} -> "
              f"{ext['layer_nhits_mean'].get((sys_id, layer), 0.0):>11.1f}   x{v:.3f}")
    fig, ax = plt.subplots(figsize=(14, 5))
    sys_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    x = np.arange(len(keys))
    ax.bar(x, vals, color=[sys_colors[(s - 1) % len(sys_colors)] for s, _ in keys], edgecolor="black")
    ax.axhline(1.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{system_to_relname.get(s, s)} {l}" for s, l in keys], rotation=60, ha="right", fontsize=8)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylabel("hits per event, extended / nominal")
    ax.set_ylim(min(0.9, min(vals) - 0.05), max(vals) + 0.1)
    ax.set_title("Hits per event per layer: extended / nominal timing (an endcap layer counts both sides)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    plt.savefig(os.path.join(OVERLAY_DIR, f"layer_hits_ratio_{base_name}.pdf"))
    plt.close(fig)


# === Cross-file overlay plots ===
def plot_overlays(all_files_hit_data):
    if len(all_files_hit_data) <= 1:
        return
    os.makedirs(OVERLAY_DIR, exist_ok=True)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if OVERLAY_LABELS is not None:
        if len(OVERLAY_LABELS) != len(file_names):
            raise ValueError(
                f"OVERLAY_LABELS has {len(OVERLAY_LABELS)} entries but file_names has "
                f"{len(file_names)}; they must match."
            )
        labels = list(OVERLAY_LABELS)
    else:
        labels = [PLOT_TAGS[f] for f in file_names]

    # Inputs whose tags differ only by "extended_timing_" are the same seeding
    # with a wider timing window: they share a colour, and the extended-timing
    # one is drawn dashed (hatched in the bar plot).
    tags = [PLOT_TAGS[f] for f in file_names]
    base_tags = [t.replace("extended_timing_", "") for t in tags]
    is_timing_variant = [b != t and b in tags for t, b in zip(tags, base_tags)]
    color_of = {}
    for b in base_tags:
        color_of.setdefault(b, colors[len(color_of) % len(colors)])

    def _col(idx):
        return color_of[base_tags[idx]]

    def _style(idx):
        """Colour and line style for input idx in line plots."""
        return dict(color=_col(idx), linestyle="--" if is_timing_variant[idx] else "-")

    def _has(key):
        return any(len(all_files_hit_data[f].get(key, [])) > 0 for f in file_names)

    # ── 1. Muon-matched track multiplicity per event (SiTracks) ──────────
    if _has("muon_matched_sitracks_per_event"):
        all_counts_flat = [v for f in file_names
                           for v in all_files_hit_data[f]["muon_matched_sitracks_per_event"]]
        max_mult = max(all_counts_flat) if all_counts_flat else 5
        bins_mm = np.arange(-0.5, max_mult + 1.5, 1)
        plt.figure()
        for idx, fname in enumerate(file_names):
            arr = all_files_hit_data[fname]["muon_matched_sitracks_per_event"]
            if arr:
                plt.hist(arr, bins=bins_mm, weights=unit_weights(arr, bins_mm), histtype='step', linewidth=1.5,
                         **_style(idx), label=labels[idx])
        plt.xlabel("Number of muon-matched tracks per event")
        plt.ylabel("Fraction of events")
        plt.yscale('log')
        plt.xticks(range(max_mult + 2))
        plt.gca().xaxis.set_minor_locator(NullLocator())
        plt.title("Muon-Matched Track Multiplicity per Event (SiTracks)")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(OVERLAY_DIR, "muon_matched_track_mult_per_event_overlay.pdf"))
        plt.close()

    # ── 2. Track multiplicity per event ──────────────────────────────────────
    if _has("sitracks_mult_per_event"):
        _mult_keys = [
            "all_tracks_mult_per_event",
            "deduped_tracks_mult_per_event",
            "sitracks_mult_per_event",
        ]
        _mult_labels = ["AllTracks", "DedupedTracks", "SiTracks"]
        all_counts_flat = [v for f in file_names for k in _mult_keys
                           for v in all_files_hit_data[f][k] if v > 0]
        if all_counts_flat:
            lo = max(1, min(all_counts_flat))
            hi = max(v for f in file_names for k in _mult_keys for v in all_files_hit_data[f][k])
            bins_mult = np.logspace(np.log10(lo), np.log10(hi + 1), 51)
            if len(file_names) == 1:
                plt.figure(figsize=(8, 5))
                for key, coll_lbl in zip(_mult_keys, _mult_labels):
                    arr = all_files_hit_data[file_names[0]][key]
                    if arr:
                        plt.hist(arr, bins=bins_mult, weights=unit_weights(arr, bins_mult),
                                 histtype='step', linewidth=1.5, label=f"{coll_lbl}  (mean={np.mean(arr):.3g})")
                plt.xscale('log')
                plt.xlabel("Track multiplicity per event")
                plt.ylabel("Fraction of events")
                plt.title("Track Multiplicity per Event")
                plt.legend(fontsize=7)
                plt.tight_layout()
                plt.savefig(os.path.join(OVERLAY_DIR, "track_multiplicity_per_event_overlay.pdf"))
                plt.close()
            else:
                fig, axs = plt.subplots(1, len(_mult_keys), figsize=(4 * len(_mult_keys), 5))
                for ax, key, coll_lbl in zip(axs, _mult_keys, _mult_labels):
                    # Each panel's bins span that collection's own range.
                    pos = [v for f in file_names for v in all_files_hit_data[f][key] if v > 0]
                    if not pos:
                        ax.set_title(coll_lbl)
                        continue
                    bins_key = np.logspace(np.log10(min(pos) / 1.2), np.log10(max(pos) * 1.2), 51)
                    for idx, fname in enumerate(file_names):
                        arr = all_files_hit_data[fname][key]
                        if arr:
                            ax.hist(arr, bins=bins_key, weights=unit_weights(arr, bins_key),
                                    histtype='step', linewidth=1.5,
                                    **_style(idx), label=f"{labels[idx]}  (mean={np.mean(arr):.3g})")
                    ax.set_xscale('log')
                    ax.set_xlabel("Track multiplicity per event")
                    ax.set_ylabel("Fraction of events")
                    ax.set_title(coll_lbl)
                    ax.legend(fontsize=7)
                fig.suptitle("Track Multiplicity per Event", fontsize=12)
                fig.tight_layout()
                plt.savefig(os.path.join(OVERLAY_DIR, "track_multiplicity_per_event_overlay.pdf"))
                plt.close(fig)

    # ── 3. Chi2 reduced ──────────────────────────────────────────────────────
    for key, fname_suffix, title_tag in [
        ("track_chi2_reduced_sitracks", "sitracks", "SiTracks"),
    ]:
        if not _has(key):
            continue
        bins_chi2 = np.linspace(0, 3, 16)
        plt.figure()
        for idx, fname in enumerate(file_names):
            arr = all_files_hit_data[fname][key]
            if not arr:
                continue
            w = np.ones(len(arr)) / len(arr)
            plt.hist(arr, bins=bins_chi2, weights=w, histtype='step', linewidth=1.5,
                     **_style(idx), label=labels[idx])
        plt.yscale('log')
        plt.xlabel("Chi2 Reduced")
        plt.ylabel("Fraction of tracks")
        plt.title(f"Chi2 Reduced ({title_tag}, Muon Matched)")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(OVERLAY_DIR, f"chi2_reduced_{fname_suffix}_overlay.pdf"))
        plt.close()

    # ── 4. Hit multiplicity: pixel / inner / outer / total (2×2 panels) ──────
    # Muon-matched tracks, then fake (not muon-linked) tracks on a log scale.
    for suffix, pixel_key, inner_key, outer_key, total_key, title_tag, ylabel, logy in [
        ("sitracks", "pixel_nhits_sitracks",  "inner_nhits_sitracks", "outer_nhits_sitracks", "total_nhits_sitracks", "SiTracks",
         "Fraction of muon matched tracks", False),
        ("alltracks", "pixel_nhits_alltracks", "inner_nhits_alltracks", "outer_nhits_alltracks", "total_nhits_alltracks", "AllTracks",
         "Fraction of muon matched tracks", False),
        *[(f"fake_{st}", f"fake_pixel_nhits_{st}", f"fake_inner_nhits_{st}", f"fake_outer_nhits_{st}",
           f"fake_total_nhits_{st}", f"Fake {tag}", "Fraction of fake tracks", True)
          for st, tag in (("sitracks", "SiTracks"), ("alltracks", "AllTracks"))],
    ]:
        if not any(len(all_files_hit_data[f].get(total_key, [])) > 0 for f in file_names):
            continue
        fig, axs = plt.subplots(2, 2, figsize=(14, 9))
        for ax, key, title, xmax in [
            (axs[0, 0], pixel_key, f"Pixel Hits ({title_tag})", 10),
            (axs[0, 1], inner_key, f"Inner Hits ({title_tag})",  5),
            (axs[1, 0], outer_key, f"Outer Hits ({title_tag})",  5),
            (axs[1, 1], total_key, f"Total Hits ({title_tag})", 20),
        ]:
            # One bin per integer hit count, wide enough for every input.
            xmax = max([xmax] + [int(np.max(all_files_hit_data[f][key])) for f in file_names
                                 if len(all_files_hit_data[f].get(key, []))])
            xs = np.arange(xmax + 1)
            bins = np.arange(-0.5, xmax + 1.5, 1)
            for idx, fname in enumerate(file_names):
                arr = all_files_hit_data[fname].get(key, [])
                if len(arr) == 0:
                    continue
                w = np.ones(len(arr)) / len(arr)
                # Legend carries per-file N, mean and median NHits so overlays can
                # be compared without eyeballing the histogram shape.
                _mean = float(np.mean(arr))
                _med  = float(np.median(arr))
                ax.hist(arr, bins=bins, weights=w, histtype='step', linewidth=1.5,
                        **_style(idx),
                        label=f"{labels[idx]}  (N={len(arr)}, mean={_mean:.2f}, med={_med:.1f})")
            ax.set_title(title)
            ax.set_xlabel("Number of hits")
            ax.set_ylabel(ylabel)
            if logy:
                ax.set_yscale('log')
            ax.set_xlim(-0.5, xmax + 0.5)
            ax.set_xticks(xs)
            for b in xs[:-1] + 0.5:
                ax.axvline(b, color='0.85', linewidth=0.6, zorder=0)
            ax.legend(fontsize=7)
        fig.tight_layout()
        plt.savefig(os.path.join(OVERLAY_DIR, f"all_hits_combined_{suffix}_overlay.pdf"))
        plt.close(fig)

    # ── 5. Hit multiplicity: muon-matched vs BIB (2 panels) ──────────────────
    for suffix, total_key, bib_key, title_tag in [
        ("sitracks", "total_nhits_sitracks",  "num_bib_hits_sitracks", "SiTracks"),
    ]:
        # Needs the muon/BIB split, so files written without hit relations
        # contribute nothing here.
        if not _has(total_key) or not _has(bib_key):
            continue
        bins_nhit = np.arange(-0.5, 20.5, 1)
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        ax_muon, ax_bib = axs
        for idx, fname in enumerate(file_names):
            total = all_files_hit_data[fname][total_key]
            bib   = all_files_hit_data[fname][bib_key]
            if not total or not bib:
                continue
            muon_hits = [n - b for n, b in zip(total, bib)]
            w_m = np.ones(len(muon_hits)) / len(muon_hits)
            w_b = np.ones(len(bib)) / len(bib)
            ax_muon.hist(muon_hits, bins=bins_nhit, weights=w_m, histtype='step',
                         linewidth=1.5, **_style(idx), label=labels[idx])
            ax_bib.hist(bib, bins=bins_nhit, weights=w_b, histtype='step',
                        linewidth=1.5, **_style(idx), label=labels[idx])
        for ax, title in [
            (ax_muon, f"Muon-matched hits ({title_tag})"),
            (ax_bib,  f"Non-muon (BIB) hits ({title_tag})"),
        ]:
            ax.set_yscale('log')
            ax.set_xlim(0, 20)
            ax.set_xlabel("N hits")
            ax.set_ylabel("Fraction of tracks")
            ax.set_title(title)
            ax.legend(fontsize=7)
        fig.tight_layout()
        plt.savefig(os.path.join(OVERLAY_DIR, f"hit_mult_muon_bib_{suffix}_overlay.pdf"))
        plt.close(fig)

    # ── 6. Track pT (SiTracks) ───────────────────────────────────────────
    if _has("track_reco_pt_sitracks"):
        bins_pt = np.logspace(np.log10(1), np.log10(5000), 51)
        plt.figure()
        for idx, fname in enumerate(file_names):
            arr = all_files_hit_data[fname]["track_reco_pt_sitracks"]
            if not arr:
                continue
            w = np.ones(len(arr)) / len(arr)
            plt.hist(arr, bins=bins_pt, weights=w, histtype='step', linewidth=1.5,
                     **_style(idx), label=labels[idx])
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(1, 5000)
        plt.xlabel(r"$p_T$ [GeV]")
        plt.ylabel("Fraction of tracks")
        plt.title(r"Track $p_T$ (SiTracks, Muon Matched)")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(OVERLAY_DIR, "track_pt_overlay.pdf"))
        plt.close()

    # ── 7. Track eta (SiTracks) ──────────────────────────────────────────
    if _has("track_eta_sitracks"):
        bins_eta = np.linspace(-3, 3, 31)
        plt.figure()
        for idx, fname in enumerate(file_names):
            arr = all_files_hit_data[fname]["track_eta_sitracks"]
            if not arr:
                continue
            w = np.ones(len(arr)) / len(arr)
            plt.hist(arr, bins=bins_eta, weights=w, histtype='step', linewidth=1.5,
                     **_style(idx), label=labels[idx])
        plt.yscale('log')
        plt.xlabel(r"$\eta$")
        plt.ylabel("Fraction of tracks")
        plt.title(r"Track $\eta$ (SiTracks, Muon Matched)")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(OVERLAY_DIR, "track_eta_overlay.pdf"))
        plt.close()

    # ── 8. Track phi (SiTracks) ──────────────────────────────────────────
    if _has("track_phi_sitracks"):
        bins_phi = np.linspace(-np.pi, np.pi, 33)
        plt.figure()
        for idx, fname in enumerate(file_names):
            arr = all_files_hit_data[fname]["track_phi_sitracks"]
            if not arr:
                continue
            w = np.ones(len(arr)) / len(arr)
            plt.hist(arr, bins=bins_phi, weights=w, histtype='step', linewidth=1.5,
                     **_style(idx), label=labels[idx])
        plt.yscale('log')
        plt.xlabel(r"$\phi$")
        plt.ylabel("Fraction of tracks")
        plt.title(r"Track $\phi$ (SiTracks, Muon Matched)")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(OVERLAY_DIR, "track_phi_overlay.pdf"))
        plt.close()

    # ── 9. Track d0 (SiTracks) ───────────────────────────────────────────
    if _has("track_d0_sitracks"):
        all_d0 = [v for f in file_names
                  for v in all_files_hit_data[f]["track_d0_sitracks"]]
        if all_d0:
            d0_max = max(abs(min(all_d0)), abs(max(all_d0))) * 1.1
            bins_d0 = np.linspace(-d0_max, d0_max, 41)
            plt.figure()
            for idx, fname in enumerate(file_names):
                arr = all_files_hit_data[fname]["track_d0_sitracks"]
                if not arr:
                    continue
                w = np.ones(len(arr)) / len(arr)
                plt.hist(arr, bins=bins_d0, weights=w, histtype='step', linewidth=1.5,
                        **_style(idx), label=labels[idx])
            plt.xlabel(r"$d_0$ [mm]")
            plt.ylabel("Fraction of tracks")
            plt.title(r"Track $d_0$ (SiTracks, Muon Matched)")
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(os.path.join(OVERLAY_DIR, "track_d0_overlay.pdf"))
            plt.close()

    # ── 10. Track z0 (SiTracks) ──────────────────────────────────────────
    if _has("track_z0_sitracks"):
        all_z0 = [v for f in file_names
                  for v in all_files_hit_data[f]["track_z0_sitracks"]]
        if all_z0:
            z0_max = max(abs(min(all_z0)), abs(max(all_z0))) * 1.1
            bins_z0 = np.linspace(-z0_max, z0_max, 41)
            plt.figure()
            for idx, fname in enumerate(file_names):
                arr = all_files_hit_data[fname]["track_z0_sitracks"]
                if not arr:
                    continue
                w = np.ones(len(arr)) / len(arr)
                plt.hist(arr, bins=bins_z0, weights=w, histtype='step', linewidth=1.5,
                        **_style(idx), label=labels[idx])
            plt.xlabel(r"$z_0$ [mm]")
            plt.ylabel("Fraction of tracks")
            plt.title(r"Track $z_0$ (SiTracks, Muon Matched)")
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(os.path.join(OVERLAY_DIR, "track_z0_overlay.pdf"))
            plt.close()

    # ── 11. Truth muon eta ───────────────────────────────────────────────────
    if _has("eff_truth_eta"):
        plt.figure()
        for idx, fname in enumerate(file_names):
            arr = all_files_hit_data[fname]["eff_truth_eta"]
            if not arr:
                continue
            plt.hist(arr, bins=25, range=(-3, 3), histtype='step', linewidth=1.5,
                     **_style(idx), label=labels[idx])
        plt.xlabel(r"Truth Muon $\eta$")
        plt.ylabel("Muons")
        plt.title("Truth Muon $\\eta$ Distribution")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(OVERLAY_DIR, "truth_muon_eta_overlay.pdf"))
        plt.close()

    # ── 12. Tracking efficiency vs eta (SiTracks) ────────────────────────────
    if _has("eff_truth_eta"):
        eta_bins = np.linspace(-2.5, 2.5, 26)
        plt.figure()
        for idx, fname in enumerate(file_names):
            eta     = all_files_hit_data[fname]["eff_truth_eta"]
            matched = all_files_hit_data[fname]["eff_matched_si"]
            if not eta:
                continue
            c, eff, err = compute_efficiency(eta, matched, eta_bins)
            int_eff = float(np.mean(matched))
            plt.errorbar(c, eff, yerr=err, fmt='o', **_style(idx), capsize=3,
                        label=f"{labels[idx]}  ({100*int_eff:.1f}%)")
        plt.axvline(-0.6, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        plt.axvline( 0.6, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        plt.xlabel(r"Truth Muon $\eta$")
        plt.ylabel("Tracking Efficiency")
        plt.xlim(-2.5, 2.5)
        eff_ticks(eta_x=True)
        plt.title(r"Tracking Efficiency vs $\eta$ (SiTracks)")
        plt.legend(fontsize=7, loc='lower center')
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(OVERLAY_DIR, "efficiency_vs_eta_overlay.pdf"))
        plt.close()

    # ── 12a. Finer-binned efficiency vs eta (SiTracks), printed bin by bin ──
    # theta 10-170 deg ends at |eta| = 2.44, inside the outer bin of both.
    if _has("eff_truth_eta"):
        for width, edge, tag in [(0.05, 2.45, "0p05"), (0.1, 2.5, "0p1")]:
            eta_bins = np.linspace(-edge, edge, int(round(2 * edge / width)) + 1)
            print(f"[EFF] SiTracks efficiency vs truth eta, {width:g} bins:")
            plt.figure()
            for idx, fname in enumerate(file_names):
                eta     = all_files_hit_data[fname]["eff_truth_eta"]
                matched = all_files_hit_data[fname]["eff_matched_si"]
                if not eta:
                    continue
                c, eff, err = compute_efficiency(eta, matched, eta_bins)
                int_eff = float(np.mean(matched))
                plt.errorbar(c, eff, yerr=err, fmt='o', markersize=3, **_style(idx), capsize=2,
                             label=f"{labels[idx]}  ({100*int_eff:.1f}%)")
                total, _ = np.histogram(eta, bins=eta_bins)
                n_matched, _ = np.histogram(eta, bins=eta_bins, weights=np.asarray(matched, dtype=float))
                print(f"  {labels[idx]}  (integrated {100*int_eff:.2f}%):")
                for lo, hi, n, k in zip(eta_bins[:-1], eta_bins[1:], total, n_matched):
                    if n:
                        e = k / n
                        print(f"    {lo:+.2f} to {hi:+.2f}  {int(k):>6d} / {int(n):>6d}  "
                              f"eff = {e:.4f} +- {np.sqrt(e * (1 - e) / n):.4f}")
                    else:
                        print(f"    {lo:+.2f} to {hi:+.2f}  {0:>6d} / {0:>6d}  eff = n/a")
            plt.axvline(-0.6, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
            plt.axvline( 0.6, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
            plt.xlabel(r"Truth Muon $\eta$")
            plt.ylabel("Tracking Efficiency")
            plt.xlim(-edge, edge)
            eff_ticks(eta_x=True)
            plt.title(rf"Tracking Efficiency vs $\eta$ (SiTracks, {width:g} bins)")
            plt.legend(fontsize=7, loc='lower center')
            plt.grid(True, linestyle=":", alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(OVERLAY_DIR, f"efficiency_vs_eta_bin{tag}_overlay.pdf"))
            plt.close()

    # ── 12c. Tracking efficiency vs truth d0, z0, production r_xy (SiTracks) ─
    for key, bins, xlabel, tag in DISPLACEMENT_EFF:
        if not _has(key):
            continue
        plt.figure()
        for idx, fname in enumerate(file_names):
            x       = all_files_hit_data[fname][key]
            matched = all_files_hit_data[fname]["eff_matched_si"]
            if not len(x):
                continue
            c, eff, err = compute_efficiency(np.clip(x, bins[0], bins[-1]), matched, bins)
            plt.errorbar(c, eff, yerr=err, fmt='o', **_style(idx), capsize=3,
                        label=f"{labels[idx]}  ({100*float(np.mean(matched)):.1f}%)")
        plt.xlabel(xlabel)
        plt.ylabel("Tracking Efficiency")
        plt.xlim(bins[0], bins[-1])
        plt.ylim(0, 1.05)
        plt.title(f"Tracking Efficiency vs {xlabel.replace('Truth Muon ', '').split(' [')[0]} (SiTracks)")
        plt.legend(fontsize=7, loc='lower center')
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(OVERLAY_DIR, f"efficiency_vs_{tag}_overlay.pdf"))
        plt.close()

        plt.figure()
        for idx, fname in enumerate(file_names):
            x = np.asarray(all_files_hit_data[fname][key], dtype=float)
            if not len(x):
                continue
            plt.hist(np.clip(x, bins[0], bins[-1]), bins=bins, weights=np.full(len(x), 1. / len(x)),
                     histtype='step', linewidth=1.5, **_style(idx), label=labels[idx])
        plt.xlabel(xlabel + " (edge bins: overflow)")
        plt.ylabel("Fraction of muons")
        plt.title(xlabel.split(' [')[0])
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(OVERLAY_DIR, f"truth_{tag}_overlay.pdf"))
        plt.close()

    # ── 12b. Tracking efficiency vs theta (SiTracks) ─────────────────────────
    if _has("eff_truth_theta_deg"):
        plt.figure()
        for idx, fname in enumerate(file_names):
            theta   = np.radians(all_files_hit_data[fname]["eff_truth_theta_deg"])
            matched = all_files_hit_data[fname]["eff_matched_si"]
            if not len(theta):
                continue
            c, eff, err = compute_efficiency(theta, matched, THETA_BINS)
            int_eff = float(np.mean(matched))
            plt.errorbar(c, eff, yerr=err, fmt='o', **_style(idx), capsize=3,
                        label=f"{labels[idx]}  ({100*int_eff:.1f}%)")
        plt.axvline(TH_BRL, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        plt.axvline(np.pi - TH_BRL, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        plt.xlabel(r"Truth Muon $\theta$ [rad]")
        plt.ylabel("Tracking Efficiency")
        plt.xlim(0, np.pi)
        eff_ticks()
        plt.title(r"Tracking Efficiency vs $\theta$ (SiTracks)")
        plt.legend(fontsize=7, loc='lower center')
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(OVERLAY_DIR, "efficiency_vs_theta_overlay.pdf"))
        plt.close()

    # ── 13. Tracking efficiency vs pT (SiTracks) ─────────────────────────────
    if _has("eff_truth_pt"):
        pt_bins = np.logspace(np.log10(2), np.log10(2500), 33)
        plt.figure()
        for idx, fname in enumerate(file_names):
            pt      = all_files_hit_data[fname]["eff_truth_pt"]
            matched = all_files_hit_data[fname]["eff_matched_si"]
            if not pt:
                continue
            c, eff, err = compute_efficiency(pt, matched, pt_bins)
            int_eff = float(np.mean(matched))
            plt.errorbar(c, eff, yerr=err, fmt='o', **_style(idx), capsize=3,
                        label=f"{labels[idx]}  ({100*int_eff:.1f}%)")
        plt.xscale('log')
        plt.xlabel(r"Truth Muon $p_T$ [GeV]")
        plt.ylabel("Tracking Efficiency")
        plt.xlim(2, 2500)
        eff_ticks()
        plt.title(r"Tracking Efficiency vs $p_T$ (SiTracks)")
        plt.legend(fontsize=7, loc='lower center')
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(OVERLAY_DIR, "efficiency_vs_pt_overlay.pdf"))
        plt.close()

    # ── 13b. Tracking efficiency vs pT, 0-50 GeV (SiTracks) ─────────────────
    if _has("eff_truth_pt"):
        plt.figure()
        for idx, fname in enumerate(file_names):
            pt      = all_files_hit_data[fname]["eff_truth_pt"]
            matched = all_files_hit_data[fname]["eff_matched_si"]
            if not pt:
                continue
            c, eff, err = compute_efficiency(np.clip(pt, PT_BINS_0_50[0], None), matched, PT_BINS_0_50)
            int_eff = float(np.mean(matched))
            plt.errorbar(c, eff, yerr=err, fmt='o', **_style(idx), capsize=3,
                        label=f"{labels[idx]}  ({100*int_eff:.1f}%)")
        plt.xscale('log')
        plt.xlabel(r"Truth Muon $p_T$ [GeV]")
        plt.ylabel("Tracking Efficiency")
        plt.xlim(PT_BINS_0_50[0], PT_BINS_0_50[-1])
        eff_ticks()
        plt.title(r"Tracking Efficiency vs $p_T$ (SiTracks)")
        plt.legend(fontsize=7, loc='lower center')
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(OVERLAY_DIR, "efficiency_vs_pt_0_50_overlay.pdf"))
        plt.close()

    # ── 13c. Seed-stage and AllTracks-stage efficiency vs eta and pT ─────────
    for flag_key, eta_key, pt_key, stage, stem in [
        ("seed_eff_matched", "seed_eff_eta", "seed_eff_pt", "Seed, 3/3 muon hits", "seed"),
        ("eff_matched_alltracks", "eff_truth_eta", "eff_truth_pt", "AllTracks", "alltracks"),
    ]:
        if not _has(flag_key):
            continue
        for xkey, bins, xlabel, logx, tag in [
            (eta_key, np.linspace(-2.5, 2.5, 26), r"Truth Muon $\eta$", False, "eta"),
            (pt_key, PT_BINS_0_50, r"Truth Muon $p_T$ [GeV]", True, "pt_0_50"),
        ]:
            plt.figure()
            for idx, fname in enumerate(file_names):
                x       = all_files_hit_data[fname][xkey]
                matched = all_files_hit_data[fname][flag_key]
                if not x:
                    continue
                c, eff, err = compute_efficiency(np.clip(x, bins[0], None) if logx else x, matched, bins)
                plt.errorbar(c, eff, yerr=err, fmt='o', **_style(idx), capsize=3,
                            label=f"{labels[idx]}  ({100*float(np.mean(matched)):.1f}%)")
            if logx:
                plt.xscale('log')
            plt.xlabel(xlabel)
            plt.ylabel("Efficiency")
            plt.xlim(bins[0], bins[-1])
            eff_ticks()
            plt.title(f"Efficiency vs {xlabel.replace('Truth Muon ', '').replace(' [GeV]', '')} ({stage})")
            plt.legend(fontsize=7, loc='lower center')
            plt.grid(True, linestyle=":", alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(OVERLAY_DIR, f"{stem}_efficiency_vs_{tag}_overlay.pdf"))
            plt.close()

    # ── 14. Average hits per layer ────────────────────────────────────────────
    for suffix, hits_key, n_key, title_tag in [
        ("sitracks", "layer_hits_sitracks", "n_matched_sitracks", "SiTracks"),
    ]:
        has_data = any(
            any(all_files_hit_data[f][hits_key][sid] for sid in range(1, 7))
            for f in file_names
        )
        if not has_data:
            continue
        fig, axs = plt.subplots(3, 2, figsize=(13, 12))
        bar_width = 0.8 / max(len(file_names), 1)
        for ax, (sys_id, sys_name) in zip(axs.flat, _SYS_NAMES.items()):
            all_layers = sorted(set(
                l for f in file_names for l in all_files_hit_data[f][hits_key][sys_id]
            ))
            if not all_layers:
                ax.set_title(sys_name)
                continue
            x = np.array(all_layers, dtype=float)
            for idx, fname in enumerate(file_names):
                n  = all_files_hit_data[fname][n_key]
                lh = all_files_hit_data[fname][hits_key][sys_id]
                avg = np.array([lh.get(l, 0) / n if n else 0.0 for l in all_layers])
                offset = (idx - len(file_names) / 2 + 0.5) * bar_width
                ax.bar(x + offset, avg, width=bar_width, label=labels[idx],
                       alpha=0.8, color=_col(idx),
                       hatch="//" if is_timing_variant[idx] else None)
            ax.set_xticks(list(range(int(min(x)), int(max(x)) + 1)))
            ax.set_xlabel("Layer")
            ax.set_ylabel("Avg hits / muon track")
            ax.set_title(sys_name)
            ax.legend(fontsize=6)
            ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
        fig.suptitle(f"Avg Hits per Muon Track vs. Layer ({title_tag})", fontsize=13)
        fig.tight_layout()
        plt.savefig(os.path.join(OVERLAY_DIR, f"avg_hits_per_layer_{suffix}_overlay.pdf"))
        plt.close(fig)

    # ── 15. Seed pT distribution ──────────────────────────────────────────────
    if _has("seed_pt_all"):
        bins_pt_s = np.logspace(np.log10(1), np.log10(5000), 41)
        plt.figure()
        for idx, fname in enumerate(file_names):
            arr = all_files_hit_data[fname]["seed_pt_all"]
            if len(arr) == 0:
                continue
            w = np.ones(len(arr)) / len(arr)
            plt.hist(arr, bins=bins_pt_s, weights=w, histtype='step', linewidth=1.5,
                     **_style(idx), label=labels[idx])
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r"Seed $p_T$ [GeV]")
        plt.ylabel("Fraction of seeds")
        plt.title(r"Seed Track $p_T$ Distribution (all seeds)")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(OVERLAY_DIR, "seed_pt_overlay.pdf"))
        plt.close()

    # ── 16. Seed multiplicity per event ──────────────────────────────────────
    if _has("seed_nseeds_per_event"):
        all_ns = [v for f in file_names for v in all_files_hit_data[f]["seed_nseeds_per_event"] if v > 0]
        if all_ns:
            lo = max(0.5, min(all_ns) / 1.2)
            hi = max(all_ns) * 1.2
            bins_ns = np.logspace(np.log10(lo), np.log10(hi), 26)
            plt.figure()
            for idx, fname in enumerate(file_names):
                arr = [v for v in all_files_hit_data[fname]["seed_nseeds_per_event"] if v > 0]
                if not arr:
                    continue
                mean_ns   = np.mean(arr)
                median_ns = np.median(arr)
                label = f'{labels[idx]}\nmean={mean_ns:.2e},  median={median_ns:.2e}'
                plt.hist(arr, bins=bins_ns, weights=unit_weights(arr, bins_ns), histtype='step', linewidth=1.5,
                         **_style(idx), label=label)
            plt.xscale('log')
            plt.xlabel("Number of seeds per event")
            plt.ylabel("Fraction of events")
            plt.title("Seed Multiplicity per Event")
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(os.path.join(OVERLAY_DIR, "seed_mult_per_event_overlay.pdf"))
            plt.close()

        # Seeds per event vs the seed's own pT and eta
        for key, bins, xlabel, logx, tag in [
            ("seed_pt_all", SEED_PT_BINS, r"Seed $p_T$ [GeV]", True, "pt"),
            ("seed_eta_all", np.linspace(-2.5, 2.5, 26), r"Seed $\eta$", False, "eta"),
        ]:
            plt.figure()
            for idx, fname in enumerate(file_names):
                d = all_files_hit_data[fname]
                if not d["n_events"]:
                    continue
                c, r, err = rate_per_event(d[key], bins, d["n_events"], clip=logx)
                plt.errorbar(c, r, yerr=err, fmt='o', **_style(idx), capsize=3,
                             label=f"{labels[idx]}  ({np.mean(d['seed_nseeds_per_event']):.3g} / event)")
            if logx:
                plt.xscale('log')
            plt.yscale('log')
            plt.xlim(bins[0], bins[-1])
            plt.xlabel(xlabel)
            plt.ylabel("Seeds / event / bin")
            plt.title(f"Average Seed Multiplicity per Event vs {xlabel.replace(' [GeV]', '')}")
            plt.legend(fontsize=7)
            plt.grid(True, linestyle=":", alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(OVERLAY_DIR, f"seed_mult_vs_{tag}_overlay.pdf"))
            plt.close()

    # Seeds per event by layer combination
    for key, name in [("seed_layers_all", "all seeds"), ("seed_layers_mu", "muon seeds, 3/3 muon hits")]:
        if _has(key):
            plot_seed_layers([(labels[idx], _col(idx), seed_layer_rates(all_files_hit_data[f][key],
                                                                         all_files_hit_data[f]["n_events"]))
                              for idx, f in enumerate(file_names)],
                             f"Seeds per Event by Layer Combination ({name})",
                             os.path.join(OVERLAY_DIR, f"{key.replace('seed_layers', 'seed_mult_vs_layers')}_overlay.pdf"))
            plot_seed_systems([(labels[idx], _col(idx), seed_system_rates(all_files_hit_data[f][key],
                                                                          all_files_hit_data[f]["n_events"]))
                               for idx, f in enumerate(file_names)],
                              f"Seeds per Event with a Hit in Each Subsystem ({name})",
                              os.path.join(OVERLAY_DIR, f"{key.replace('seed_layers', 'seed_mult_vs_systems')}_overlay.pdf"))

    # ── 17. Seed pT resolution (best-matched seed vs truth muon) ─────────────
    if _has("seed_pt_best_per_muon"):
        bins_res_s = np.linspace(-2, 2, 41)
        plt.figure()
        for idx, fname in enumerate(file_names):
            pairs = all_files_hit_data[fname]["seed_pt_best_per_muon"]
            if not pairs:
                continue
            res = [(s - t) / t for t, s in pairs]
            w = np.ones(len(res)) / len(res)
            plt.hist(res, bins=bins_res_s, weights=w, histtype='step', linewidth=1.5,
                     **_style(idx),
                     label=f"{labels[idx]}  μ={np.mean(res):.3f} σ={np.std(res):.3f}")
        plt.xlabel(r"$(p_T^{\mathrm{seed}} - p_T^{\mathrm{truth}}) / p_T^{\mathrm{truth}}$")
        plt.ylabel("Fraction of seeds")
        plt.title(rf"Seed $p_T$ Resolution ($\geq${SEED_MATCH_MIN_HITS}/3 muon hits, else closest $\Delta R < {SEED_MATCH_MAX_DR:g}$)")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(OVERLAY_DIR, "seed_pt_resolution_overlay.pdf"))
        plt.close()

    # ── 18. pT / d0 / z0 residuals and pulls (muon matched), per collection ──
    for suffix, file_suffix, title_tag in RESIDUAL_STAGES:
        for key, file_stem, title, xlabel, bins, pull in RESIDUAL_PLOTS:
            k = f"track_{key}_{suffix}"
            if not _has(k):
                continue
            plt.figure()
            for idx, fname in enumerate(file_names):
                residual_hist(plt.gca(), all_files_hit_data[fname][k], bins, labels[idx], **_style(idx))
            if pull:
                unit_gaussian(plt.gca(), bins)
            plt.xlabel(xlabel)
            plt.ylabel("Fraction of tracks")
            plt.yscale('log')
            plt.title(f"{title} ({title_tag}, Muon Matched)")
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(os.path.join(OVERLAY_DIR, f"{file_stem}_{file_suffix}_overlay.pdf"))
            plt.close()

    # ── 19. Muon-matched hit fraction — one plot per selection stage ─────────
    for key, fname_suffix, title_tag in [
        ("track_percent_muon_hits_sitracks", "sitracks",    "SiTracks"),
    ]:
        if not _has(key):
            continue
        bins_mhf = np.linspace(0, 1, 11)
        plt.figure()
        for idx, fname in enumerate(file_names):
            arr = all_files_hit_data[fname][key]
            if not arr:
                continue
            w = np.ones(len(arr)) / len(arr)
            plt.hist(arr, bins=bins_mhf, weights=w, histtype='step', linewidth=1.5,
                     **_style(idx),
                     label=f"{labels[idx]}  mean={np.mean(arr):.3f}")
        plt.xlabel("Muon-matched hit fraction")
        plt.ylabel("Fraction of tracks")
        plt.yscale('log')
        plt.title(f"Muon-Matched Hit Fraction ({title_tag})")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(OVERLAY_DIR, f"muonhitfrac_{fname_suffix}_overlay.pdf"))
        plt.close()

    # ── 22. Extended / nominal timing: occupancy and per-layer hit ratios ───
    tag_to_entry = {PLOT_TAGS[f]: f for f in file_names}
    for idx, fname in enumerate(file_names):
        if is_timing_variant[idx]:
            plot_timing_ratios(base_tags[idx],
                               all_files_hit_data[tag_to_entry[base_tags[idx]]],
                               all_files_hit_data[fname])

    # ── 23. Fake SiTracks (not matched to the muon): rate vs pT threshold, pT and eta ──
    if _has("fake_pt"):
        plt.figure()
        for idx, fname in enumerate(file_names):
            d = all_files_hit_data[fname]
            if len(d["fake_pt"]) == 0:
                continue
            rate, err = rate_above(d["fake_pt"], FAKE_PT_THRESHOLDS, d["n_events"])
            # Timing variants: open markers instead of the dashed lines used elsewhere
            plt.errorbar(FAKE_PT_THRESHOLDS, rate, yerr=err, fmt='o', markersize=3, color=_col(idx),
                         mfc='none' if is_timing_variant[idx] else _col(idx),
                         label=f"{labels[idx]}  ({np.mean(d['fake_mult_per_event']):.3g} / event)")
        plt.yscale('log')
        plt.xlim(FAKE_PT_THRESHOLDS[0], FAKE_PT_THRESHOLDS[-1])
        plt.xlabel(r"$p_T$ threshold [GeV]")
        plt.ylabel(r"Fake tracks with $p_T >$ threshold / event / 0.5 GeV")
        plt.title(r"Fake Track Rate vs $p_T$ Threshold (SiTracks not matched to the muon)")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(OVERLAY_DIR, "fake_rate_vs_pt_threshold_overlay.pdf"))
        plt.close()

        for key, bins, xlabel, title, outname, logx, logy in [
            ("fake_pt", np.logspace(-1, 4, 51), r"Track $p_T$ [GeV]",
             r"Fake Track $p_T$ (SiTracks not matched to the muon)", "fake_pt_overlay.pdf", True, True),
            ("fake_eta", np.linspace(-3, 3, 31), r"Track $\eta$",
             r"Fake Track $\eta$ (SiTracks not matched to the muon)", "fake_eta_overlay.pdf", False, False),
        ]:
            plt.figure()
            for idx, fname in enumerate(file_names):
                d = all_files_hit_data[fname]
                arr = d[key]
                if len(arr) == 0:
                    continue
                plt.hist(arr, bins=bins, weights=np.full(len(arr), 1.0 / len(arr)), histtype='step',
                         linewidth=1.5, **_style(idx),
                         label=f"{labels[idx]}  ({np.mean(d['fake_mult_per_event']):.3g} / event)")
            if logx:
                plt.xscale('log')
            if logy:
                plt.yscale('log')
            plt.xlabel(xlabel)
            plt.ylabel("Fraction of fake tracks")
            plt.title(title)
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(os.path.join(OVERLAY_DIR, outname))
            plt.close()

    # ── 24. Duplicate SiTracks per event vs reco pT and eta ──────────────────
    if _has("dup_mult_per_event"):
        for key, bins, xlabel, logx, tag in [
            ("dup_pt", PT_BINS_0_50, r"Track $p_T$ [GeV]", True, "pt"),
            ("dup_eta", np.linspace(-2.5, 2.5, 26), r"Track $\eta$", False, "eta"),
        ]:
            plt.figure()
            for idx, fname in enumerate(file_names):
                d = all_files_hit_data[fname]
                if not d["dup_mult_per_event"]:
                    continue
                c, r, err = rate_per_event(d[key], bins, d["n_events"], clip=logx)
                plt.errorbar(c, r, yerr=err, fmt='o', **_style(idx), capsize=3,
                             label=f"{labels[idx]}  ({np.mean(d['dup_mult_per_event']):.3g} / event)")
            if logx:
                plt.xscale('log')
            plt.xlim(bins[0], bins[-1])
            plt.xlabel(xlabel)
            plt.ylabel("Duplicate tracks / event / bin")
            plt.title(f"Duplicate SiTracks (> {100*DUP_SHARED_FRAC:.0f}% of hits shared)")
            plt.legend(fontsize=7)
            plt.grid(True, linestyle=":", alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(OVERLAY_DIR, f"duplicate_rate_vs_{tag}_overlay.pdf"))
            plt.close()

    # ── 25. Seed |d0| and max doublet |zOrigin| (ImpactMax / CollisionRegion) ──────
    for key, bins, xlabel, cut, tag in [
        ("seed_d0_mu", SEED_D0_BINS, r"Seed $|d_0|$ [mm]", SEED_D0_REF, "d0"),
        ("seed_zorigin_mu", SEED_ZORIGIN_BINS, r"Seed max doublet $|z_{\mathrm{Origin}}|$ [mm]", SEED_ZORIGIN_REF, "zorigin"),
        ("seed_d0_all", SEED_D0_BINS, r"Seed $|d_0|$ [mm]", SEED_D0_REF, "d0_allseeds"),
        ("seed_zorigin_all", SEED_ZORIGIN_BINS, r"Seed max doublet $|z_{\mathrm{Origin}}|$ [mm]", SEED_ZORIGIN_REF, "zorigin_allseeds"),
    ]:
        if not _has(key):
            continue
        plt.figure()
        for idx, fname in enumerate(file_names):
            v = np.asarray(all_files_hit_data[fname].get(key, []), dtype=float)
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            plt.hist(np.clip(v, bins[0], bins[-1]), bins=bins, weights=np.full(len(v), 1. / len(v)),
                     histtype='step', linewidth=1.5, **_style(idx), label=seed_cut_label(labels[idx], v, cut))
        plt.axvline(cut, color='red', linestyle='--', linewidth=1)
        plt.yscale('log')
        plt.xlabel(xlabel + " (last bin: overflow)")
        plt.ylabel("Fraction of seeds")
        plt.title(("Muon seeds (3/3 muon hits)" if key.endswith("_mu") else "All seeds")
                  + ", red: current cut")
        plt.legend(fontsize=6)
        plt.tight_layout()
        plt.savefig(os.path.join(OVERLAY_DIR, f"seed_{tag}_overlay.pdf"))
        plt.close()


def twoway_nhits_compare(all_files_hit_data):
    """NHits of each muon's matched SiTrack, two-way entry vs the one-way
    file_names[0], muon by muon: a printout, and the mean change vs truth eta
    (every muon matched in both, unchanged ones included).

    Events are paired by index, so every entry must be reco of the same events
    in the same order; a muon whose truth pT differs is counted as misaligned
    and skipped.
    """
    labels = list(OVERLAY_LABELS) if OVERLAY_LABELS is not None else [PLOT_TAGS[f] for f in file_names]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    eta_bins = np.linspace(-2.5, 2.5, 26)
    centers = 0.5 * (eta_bins[:-1] + eta_bins[1:])
    ref = all_files_hit_data[file_names[0]]["muon_si_nhits_per_event"]
    print(f"[TWOWAY] Muon-matched SiTrack NHits, two-way vs {labels[0]} (same event, same muon):")
    plt.figure()
    for idx in range(1, len(file_names)):
        other = all_files_hit_data[file_names[idx]]["muon_si_nhits_per_event"]
        if len(other) != len(ref):
            print(f"  WARNING: {labels[idx]} has {len(other)} events, {labels[0]} has {len(ref)}; "
                  f"pairing the first {min(len(other), len(ref))}.")
        deltas, etas = [], []
        n_only_ref = n_only_other = n_misaligned = 0
        for ev_ref, ev_other in zip(ref, other):
            n_only_other += len(ev_other.keys() - ev_ref.keys())
            for mcpid, (pt_ref, eta_ref, n_ref) in ev_ref.items():
                if mcpid not in ev_other:
                    n_only_ref += 1
                    continue
                pt_other, _, n_other = ev_other[mcpid]
                if not np.isclose(pt_other, pt_ref, rtol=1e-6):
                    n_misaligned += 1
                    continue
                deltas.append(n_other - n_ref)
                etas.append(eta_ref)
        deltas = np.array(deltas, dtype=int)
        etas = np.array(etas)
        n = len(deltas)
        up = deltas[deltas > 0]
        print(f"  {labels[idx]}:")
        if n == 0:
            print("    no muon has a matched SiTrack in both")
        else:
            print(f"    {n} muons matched in both; NHits incremented for {len(up)} ({100. * len(up) / n:.1f}%), "
                  f"mean increment {np.mean(up) if len(up) else 0.:.2f} hits; "
                  f"decreased {int((deltas < 0).sum())}, unchanged {int((deltas == 0).sum())}")
        print(f"    matched only in one-way: {n_only_ref}, only in two-way: {n_only_other}"
              + (f", misaligned (truth pT differs, skipped): {n_misaligned}" if n_misaligned else ""))
        if n == 0:
            continue
        # Mean change per truth-eta bin, error = standard error of the mean.
        ibin = np.digitize(etas, eta_bins) - 1
        mean = np.full(len(centers), np.nan)
        err = np.zeros(len(centers))
        for b in range(len(centers)):
            d = deltas[ibin == b]
            if len(d):
                mean[b] = d.mean()
                err[b] = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.
        plt.errorbar(centers, mean, yerr=err, fmt='o', color=colors[idx % len(colors)], capsize=3,
                     label=f"{labels[idx]}  (mean {deltas.mean():+.2f})")
    plt.axhline(0, color='gray', linewidth=0.8)
    plt.xlabel(r"Truth Muon $\eta$")
    plt.ylabel(f"Mean NHits change vs {labels[0]}")
    plt.xlim(-2.5, 2.5)
    plt.title(r"Hits Added by Two-Way CKF vs $\eta$ (SiTracks, Muon Matched)")
    plt.legend(fontsize=7)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()
    os.makedirs(OVERLAY_DIR, exist_ok=True)
    plt.savefig(os.path.join(OVERLAY_DIR, "twoway_nhits_added_vs_eta.pdf"))
    plt.close()


# --- Driver ----------------------------------------------------------------------

def finish_entry(infile, file_results):
    """Report, merge and plot one entry once all of its files are processed."""
    n_files = len(file_results)
    print("Processing:", infile, f"({n_files} file{'s' if n_files != 1 else ''})")
    print(f"  USE_CONED = {file_results[0]['use_coned']} "
          f"(from {'env var' if _env_use_coned is not None else 'filename autodetect'})")
    read = [r for r in file_results if r["n_events"] > 0] or file_results
    no_hits, detected, nhits_source = read[0]["no_hits"], read[0]["detected_no_hits"], read[0]["nhits_source"]
    if len({(r["no_hits"], r["nhits_source"]) for r in read}) > 1:
        print("  WARNING: the files of this entry differ in which hit collections they have; "
              "each was read in its own mode and the plots mix them.")
    if no_hits:
        print(f"  NO_HITS = True (from {'env var' if _env_no_hits is not None else 'collection autodetect'}): "
              f"{' / '.join(NO_HITS_REQUIRED_COLLECTIONS)} absent — hit-level plots disabled "
              f"(subsystem split, muon/BIB split, per-layer, event displays, occupancy).")
        print(f"  per-track hit counts from: {nhits_source}"
              + ("  (none available: hit-count plots will be empty and the "
                 "multiplicity counters count every track)"
                 if nhits_source == "unavailable" else ""))
    elif detected:
        print("  WARNING: NO_HITS forced off by the env var, but this file has no "
              f"{' / '.join(NO_HITS_REQUIRED_COLLECTIONS)} — the hit-level code will fail on it.")
    for r in file_results:
        for line in r["diag_lines"]:
            print(line)
    plotdir = PLOT_TAGS[infile] + "_plots"
    print(f"  plots -> {plotdir}/")
    os.makedirs(plotdir, exist_ok=True)
    return plot_entry(plotdir, merge_results(file_results), bool(no_hits))


def main():
    # Checked before any file is read; plot_overlays would only fail at the end.
    if OVERLAY_LABELS is not None and len(OVERLAY_LABELS) != len(file_names):
        raise SystemExit(f"OVERLAY_LABELS has {len(OVERLAY_LABELS)} entries but file_names has "
                         f"{len(file_names)}; they must match.")
    # Expanded up front, so a missing or empty input fails at startup rather
    # than after the entries before it have run.
    input_files = {entry: expand_input(entry) for entry in file_names}
    tasks = []   # (entry, file index, process_file argument), entry by entry
    for entry in file_names:
        # Per-entry coning autodetect (env var USE_CONED overrides).
        use_coned = USE_CONED if _env_use_coned is not None else _autodetect_use_coned(entry)
        tasks += [(entry, k, (f, k, use_coned)) for k, f in enumerate(input_files[entry])]
    n_workers = max(1, min(N_WORKERS, len(tasks)))
    print(f"{len(tasks)} files in {len(file_names)} entries, "
          f"{n_workers} worker process{'es' if n_workers != 1 else ''}", flush=True)

    results = {entry: [None] * len(input_files[entry]) for entry in file_names}
    pending = {entry: len(input_files[entry]) for entry in file_names}
    all_files_hit_data = {}
    t_start = time.time()
    n_done = 0

    def collect(entry, k, res):
        nonlocal n_done
        n_done += 1
        results[entry][k] = res
        pending[entry] -= 1
        print(f"[{n_done}/{len(tasks)}] {PLOT_TAGS[entry]}: {os.path.basename(res['fname'])}, "
              f"{res['n_events']} events in {res['seconds']:.0f} s "
              f"({time.time() - t_start:.0f} s elapsed)", flush=True)
        # An entry is plotted as soon as its last file is in, while the
        # workers carry on with the remaining files.
        if pending[entry] == 0:
            all_files_hit_data[entry] = finish_entry(entry, results.pop(entry))

    if n_workers == 1:
        for entry, k, task in tasks:
            collect(entry, k, process_file(task))
    else:
        # spawn: each worker starts a fresh interpreter instead of a fork of
        # this process's ROOT state.
        pool = ProcessPoolExecutor(max_workers=n_workers, mp_context=multiprocessing.get_context("spawn"))
        try:
            futures = {pool.submit(process_file, task): (entry, k) for entry, k, task in tasks}
            for fut in as_completed(futures):
                collect(*futures[fut], fut.result())
        except BrokenProcessPool:
            sys.exit(f"A worker process was killed, most likely for running out of memory with "
                     f"N_WORKERS={n_workers} (check dmesg). Rerun with a lower N_WORKERS.")
        finally:
            pool.shutdown(cancel_futures=True)

    plot_overlays({entry: all_files_hit_data[entry] for entry in file_names})
    if TWOWAY_NHITS_COMPARE and len(file_names) > 1:
        twoway_nhits_compare(all_files_hit_data)


if __name__ == "__main__":
    main()
