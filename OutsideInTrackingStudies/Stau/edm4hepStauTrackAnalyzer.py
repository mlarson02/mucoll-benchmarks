# Stau version of ../edm4hepTrackAnalyzer.py (muons), for the stau_<mass>_<tau>
# samples: two staus per event (|PDG| 1000015 / 2000015, status 22, beta ~ 0.6,
# pT ~ TeV). The efficiencies count only staus in the acceptance: |eta| < ACC_ETA_MAX
# and decaying outside the tracker volume (see stau_in_acceptance).
from podio.reading import get_reader  # for the generic reader
from podio.root_io import Reader  # for TTree (i.e. ROOTReader)
import edm4hep
import ROOT
from collections import defaultdict
ROOT.gInterpreter.LoadFile("podio/LinkNavigator.h")
ROOT.gInterpreter.LoadFile("podio/LinkCollection.h")
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

# matplotlib is only needed for the plots. Stage 1 of the condor path
# (stauAccumulator.py) only runs process_file(), and its image may lack it.
try:
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import LogNorm, TwoSlopeNorm
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.ticker import MultipleLocator, NullLocator

    # Minor ticks on every plot (categorical / integer x axes turn them off).
    plt.rcParams.update({"xtick.minor.visible": True, "ytick.minor.visible": True})
except ImportError:
    plt = None

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

# Signal: staus, either chirality and charge. Zero-length copies (endpoint =
# vertex) are skipped, as in LLPStudies/MarkLLPCode/analysis/analysis.py.
STAU_PDGS = (1000015, 2000015)
SIGNAL_PDG = "1000015|2000015"   # label only

def is_signal_pdg(pdg):
    """True when `pdg` is a stau, either charge sign."""
    return abs(int(pdg)) in STAU_PDGS

# Efficiency acceptance: |eta| < ACC_ETA_MAX (barrel), and the stau decays
# outside the tracker volume, r > TRACKER_R or |z| > TRACKER_Z [mm] (OT barrel
# outer layer r = 1486, last disks |z| = 2190).
ACC_ETA_MAX = float(os.environ.get("ACC_ETA_MAX", 0.8))
TRACKER_R = 1500.
TRACKER_Z = 2200.

# Selected staus per event (select_staus): produced at the origin (|x|, |y|, |z|
# < ORIGIN_TOL mm), longest decay length first.
STAUS_PER_EVENT = 2
ORIGIN_TOL = 1e-3

# Velocity fit (fit_beta): time resolution per system [ns], as HitTimeResolutions.
C_MM_PER_NS = 299.792458
HIT_TIME_SIGMA = {1: 0.03, 2: 0.03, 3: 0.06, 4: 0.06, 5: 0.06, 6: 0.06}
BETA_FIT_MIN_HITS = 4

# Geometric truth match, since coned + BIB reco keeps no per-hit truth: a reco
# hit is the stau's when within GEO_TOL_VXD (VXD) / GEO_TOL_OTHER (IT, OT) [mm]
# of its truth straight line from the vertex. At 3 TeV the helix leaves that
# line by r^2 / 2R: < 3 um in the VXD (5 um resolution), ~80 um at the IT and
# ~0.6 mm at the outer OT, where the strips also measure z to ~0.1 mm only.
# "Seedable": stau hits on >= GEO_MIN_VXD_STATIONS VXD stations, a station
# being a doublet's two layers (layer // 2), as a seed takes one of them.
GEO_TOL_VXD = 0.05
GEO_TOL_OTHER = 1.0
GEO_MIN_VXD_STATIONS = 3
GEO_RESID_RANGE = 0.2   # mm: VXD hit distances from the line kept for the tolerance check

# Tiered efficiency, as in mira_analysis_code/efficiency/efficiency_from_slcio.py:
# a matched SiTrack with chi2/ndf <= TIER_CHI2NDF_MAX and >= the tier's VXD / IT /
# OT hits. A VXD hit on a doublet layer counts 1/2, so the tier counts layers:
# in MAIA the doublets are VXD barrel layers 0/1 and every VXD endcap layer;
# barrel layers 2, 4, 6 are single and count 1.
# (key, label, min VXD, min IT, min OT)
TIER_CHI2NDF_MAX = 3.0
VXD_BARREL_DOUBLET_LAYERS = (0, 1)

def vxd_tier_weight(system, layer):
    """Tier weight of a VXD hit: 1/2 on a doublet layer, 1 on a single one."""
    return 0.5 if system == 2 or (system == 1 and layer in VXD_BARREL_DOUBLET_LAYERS) else 1.0

# Mass reconstruction, as in mira_analysis_code/efficiency/lifetime_study.py:
# m = p sqrt(1/beta^2 - 1), p from pT and tan(lambda), beta from fit_beta. Per
# event the stau-matched SiTracks that pass the VB+IB+OB tier, |eta| <
# MASS_ETA_MAX, chi2/ndf < MASS_CHI2NDF_MAX, pT < MASS_PT_MAX [GeV] and
# 0 < beta < 1, ordered by pT: the leading and subleading track (NaN if none).
MASS_ETA_MAX = 0.8
MASS_CHI2NDF_MAX = 3.0
MASS_PT_MAX = 10000.0
# What plot_entry hands plot_overlays for the stau overlays
STAU_OVERLAY_KEYS = (
    "eff_truth_eta", "eff_truth_pt", "eff_matched_si", "eff_matched_alltracks",
    "eff_tier_vb", "eff_tier_ib", "eff_tier_ob", "stau_purity_si", "stau_purity_all",
    "geo_seed_best", "geo_nstations_vxd", "geo_vxd_resid",
    "stau_vel_reco_beta", "stau_vel_truth_beta", "stau_vel_in_acc",
    "lead_pt", "lead_beta", "lead_mass", "sublead_pt", "sublead_beta", "sublead_mass",
)
# (quantity, bins, x label) of the mass plots, as in Mira's plots
STAU_MASS_PLOTS = [
    ("mass", np.linspace(0, 6000, 61), "Reconstructed mass [GeV]"),
    ("beta", np.linspace(0.3, 1.05, 76), r"Reconstructed $\beta$"),
    ("pt", np.linspace(0, 10000, 61), r"Reconstructed $p_T$ [GeV]"),
]

TIERS = [
    ("vb", "VB: ≥3 VXD", 3, 0, 0),
    ("ib", "VB+IB: + ≥2 IT", 3, 2, 0),
    ("ob", "VB+IB+OB: + ≥2 OT", 3, 2, 2),
]

# A track is a stau's only when its TrackTruthAlg link weight is above
# PURITY_MIN. The weight is the track purity: hits from that particle / all hits
# on the track, BIB hits (no truth) included; each particle is linked to the
# one track with the highest such fraction. Applied wherever tracks are matched
# to staus (efficiencies, tiers, track plots, beta fit, fakes).
PURITY_MIN = float(os.environ.get("PURITY_MIN", 0.5))

# Per-event r-z / x-y hit maps (EventRZ_*, EventXY_*): only with per-hit truth
# (coned + BIB outputs drop it, and then every hit counts as non-stau), at most
# this many per entry.
MAX_EVENT_DISPLAYS = 50

def stau_decays_outside(mcp):
    e = mcp.getEndpoint()
    return hypot(e[0], e[1]) > TRACKER_R or abs(e[2]) > TRACKER_Z

def stau_in_acceptance(eta, mcp):
    return abs(eta) < ACC_ETA_MAX and stau_decays_outside(mcp)

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
N_WORKERS = int(os.environ.get("N_WORKERS", 4))

# Per-signal-stau efficiency debug printout, plus the truth composition of
# every AllTracks track when the stau has no AllTracks link. Slow on BIB: it
# walks every hit link in Python.
EFF_DEBUG = False

# Per-track AllTracks printout (print_alltracks) for the first PRINT_ALLTRACKS
# events of each file; 0 = off. Use N_WORKERS=1 to keep files' output apart.
PRINT_ALLTRACKS = int(os.environ.get("PRINT_ALLTRACKS", 0))

# Seed pT plots: the stau's seed is one with at least SEED_MATCH_MIN_HITS of its
# 3 hits from the stau (3: no BIB hit). If there is none, or the hits were not
# written out, it is the seed closest to the stau in (eta, phi) within
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

# Per-file results of process_file() saved by stage 1 of the condor path
# (stauAccumulator.py, condor/stauacc/), <sample>_acc<N>.pkl. A file_names
# entry that is a directory of these is merged and plotted without reading
# any reco file.
ACC_GLOB = "*_acc[0-9]*.pkl"


def _acc_encode(o):
    """process_file() output as plain Python objects: numpy arrays become
    {"__nd__": [dtype, shape, bytes]}, numpy scalars Python numbers, so the
    pickle loads whatever numpy version the reading side has."""
    if isinstance(o, np.ndarray):
        if o.dtype == object:
            return [_acc_encode(v) for v in o]
        return {"__nd__": [o.dtype.str, list(o.shape), o.tobytes()]}
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, dict):
        return {_acc_encode(k): _acc_encode(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(_acc_encode(v) for v in o)
    return o


def _acc_decode(o):
    if isinstance(o, dict):
        if len(o) == 1 and "__nd__" in o:
            dtype, shape, buf = o["__nd__"]
            return np.frombuffer(buf, dtype=np.dtype(dtype)).reshape(shape).copy()
        return {k: _acc_decode(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(_acc_decode(v) for v in o)
    return o


def save_acc(acc, path):
    import pickle
    with open(path, "wb") as f:
        pickle.dump(_acc_encode(acc), f, protocol=4)


def load_acc(path):
    import pickle
    with open(path, "rb") as f:
        return _acc_decode(pickle.load(f))

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
    (hadd_reco_outputs.sh --cleanup) is read through its merged outputs. A
    directory of stage-1 results (ACC_GLOB) is loaded instead of processed.
    """
    if os.path.isdir(entry):
        files = glob.glob(os.path.join(entry, PER_JOB_GLOB))
        if not files:
            files = glob.glob(os.path.join(entry, ACC_GLOB))
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


def unreadable_reason(fname):
    """Why podio cannot read `fname`, or None if it can.

    A reco job whose upload was cut off leaves a truncated file. When ROOT can
    recover some of its keys (the metadata trees), the file opens but has no
    "events" tree, and podio's reader then segfaults, killing the worker; so
    such files are caught here with plain ROOT first."""
    try:
        f = ROOT.TFile.Open(fname)
    except OSError:
        return "cannot be opened"
    try:
        if not f or f.IsZombie():
            return "cannot be opened"
        if f.TestBit(ROOT.TFile.kRecovered):
            return "truncated (ROOT had to recover it)"
        events = f.Get("events")
        if not events or events.GetEntries() == 0:
            return "no events tree"
        return None
    finally:
        if f:
            f.Close()


# Efficiency vs theta / eta: 12 bins over the acceptance, |eta| < ACC_ETA_MAX
# (TH_ACC its theta). TH_BRL is the theta of |eta| = 0.6, the barrel / endcap
# boundary drawn on the plots.
TH_ACC = 2 * np.arctan(np.exp(-ACC_ETA_MAX))
THETA_BINS = np.linspace(TH_ACC, np.pi - TH_ACC, 13)
ETA_EFF_BINS = np.linspace(-ACC_ETA_MAX, ACC_ETA_MAX, int(round(20 * ACC_ETA_MAX)) + 1)   # 0.1 wide
# Truth stau pT [GeV]: m = 4 TeV at sqrt(s) = 10 TeV gives p ~ 3 TeV.
STAU_PT_BINS = np.linspace(0, 3500, 36)
# Efficiency vs truth displacement: (key, bins [mm], x label, file tag). Values
# outside the bins are folded into the edge bins.
DISPLACEMENT_EFF = [
    ("eff_truth_d0", np.linspace(-300, 300, 61), r"Truth Stau $d_0$ [mm]", "d0"),
    ("eff_truth_z0", np.linspace(-300, 300, 61), r"Truth Stau $z_0$ [mm]", "z0"),
    ("eff_truth_vrxy", np.linspace(0, 1000, 51), r"Truth Stau production vertex $r_{xy}$ [mm]", "vrxy"),
]
EFF2D_MIN_STAUS = 5   # efficiency maps leave bins with fewer staus blank
TH_BRL = 2 * np.arctan(np.exp(-0.6))
# Track pT (duplicates): log bins, ~10 per decade, from 0.5 GeV (a log axis
# cannot start at 0); pT < 0.5 GeV is clipped into the first bin.
PT_BINS_0_50 = np.logspace(np.log10(0.5), np.log10(5000), 41)
# Seed pT: from the 1.5 GeV seed pT cut (default --seedMinPt) to 50 GeV.
SEED_PT_BINS = np.logspace(np.log10(1.5), np.log10(5000), 36)


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

// objKey of every hit linked to a |PDG| == pdg particle (first) and that
// particle's index in its collection (second).
template <class LinkColl>
std::pair<std::vector<std::int64_t>, std::vector<int>> signalHitKeysByParticle(const LinkColl& links, int pdg) {
  std::pair<std::vector<std::int64_t>, std::vector<int>> out;
  for (const auto& link : links) {
    const auto sim = link.getTo();
    if (!sim.isAvailable()) continue;
    const auto mcp = sim.getParticle();
    if (mcp.isAvailable() && std::abs(mcp.getPDG()) == pdg) {
      out.first.push_back(objKey(link.getFrom().getObjectID()));
      out.second.push_back(mcp.getObjectID().index);
    }
  }
  return out;
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

// Distance of point p from the straight line (x0, u), u a unit vector, counting
// only the half-line ahead of x0 (behind it: infinity). For the back-to-back
// staus, the line of one extended backwards would run through the other.
inline double aheadLineDistance(const double* line, double px, double py, double pz) {
  const double dx = px - line[0], dy = py - line[1], dz = pz - line[2];
  const double t = dx * line[3] + dy * line[4] + dz * line[5];
  if (t <= 0) return INFINITY;
  const double qx = dx - t * line[3], qy = dy - t * line[4], qz = dz - t * line[5];
  return std::sqrt(qx * qx + qy * qy + qz * qz);
}

// Per seed: of the nLines lines (x0, y0, z0, ux, uy, uz each), the one with the
// most of the seed's hits within tolerance of it (tolVxd for VXD hits, tolOther
// for the rest; first line on a tie) in bestLine, and that number of hits in
// nOnLine. bestLine = -1, nOnLine = 0 when no hit is on any line.
template <class TrackColl>
void seedLineMatch(const TrackColl& seeds, const double* lines, int nLines, double tolVxd,
                   double tolOther, std::int64_t* bestLine, std::int64_t* nOnLine) {
  std::size_t i = 0;
  for (const auto& s : seeds) {
    bestLine[i] = -1;
    nOnLine[i] = 0;
    const auto hits = s.getTrackerHits();
    for (int l = 0; l < nLines; ++l) {
      std::int64_t n = 0;
      for (const auto& hit : hits) {
        const auto q = hit.getPosition();
        const auto sys = static_cast<std::int64_t>(hit.getCellID()) & 0x1F;
        const double tol = (sys == 1 || sys == 2) ? tolVxd : tolOther;
        if (aheadLineDistance(lines + 6 * l, q.x, q.y, q.z) < tol) ++n;
      }
      if (n > nOnLine[i]) {
        nOnLine[i] = n;
        bestLine[i] = l;
      }
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
    """obj_key()s of the hits linked to a stau SimTrackerHit."""
    if len(links) == 0:
        return set()
    return {int(k) for pdg in STAU_PDGS for k in _cpp("signalHitKeys", links)(links, pdg)}


def stau_hit_keys(links):
    """{MCParticle index: obj_key()s of the hits linked to that stau}."""
    out = defaultdict(set)
    if len(links) == 0:
        return out
    for pdg in STAU_PDGS:
        res = _cpp("signalHitKeysByParticle", links)(links, pdg)
        for k, p in zip(res.first, res.second):
            out[int(p)].add(int(k))
    return out


def signal_particles(mcps):
    """Positions in `mcps` of the moving staus (endpoint != vertex), in collection order."""
    if len(mcps) == 0:
        return []
    pos = sorted(int(p) for pdg in STAU_PDGS for p in _cpp("signalParticles", mcps)(mcps, pdg))
    out = []
    for p in pos:
        v, e = mcps[p].getVertex(), mcps[p].getEndpoint()
        if (e[0] - v[0], e[1] - v[1], e[2] - v[2]) != (0, 0, 0):
            out.append(p)
    return out


def select_staus(mcps, moving):
    """Up to STAUS_PER_EVENT of the moving staus produced at the origin, the
    longest decay length first: the two pair-produced staus, not their
    stau -> stau (+ photon) copies, which start away from the origin."""
    at_origin = []
    for p in moving:
        v, e = mcps[p].getVertex(), mcps[p].getEndpoint()
        if abs(v[0]) < ORIGIN_TOL and abs(v[1]) < ORIGIN_TOL and abs(v[2]) < ORIGIN_TOL:
            at_origin.append((-sqrt((e[0] - v[0])**2 + (e[1] - v[1])**2 + (e[2] - v[2])**2), p))
    return [p for _, p in sorted(at_origin)[:STAUS_PER_EVENT]]


def has_stau_daughter(mcp):
    return any(is_signal_pdg(d.getPDG()) for d in mcp.getDaughters())


def fit_beta(hits):
    """beta from a weighted fit of the raw hit times, t = |x| / (beta c), through
    the origin (the collision at t = 0; no vertex smearing in sim). sigma_t per
    hit from CKF_tracking.py HitTimeResolutions. Straight line from the IP: the
    staus have pT ~ TeV. None with fewer than BETA_FIT_MIN_HITS hits."""
    if len(hits) < BETA_FIT_MIN_HITS:
        return None
    sxt = sxx = 0.0
    for h in hits:
        p = h.getPosition()
        d = sqrt(p[0]**2 + p[1]**2 + p[2]**2)
        w = 1.0 / HIT_TIME_SIGMA.get(int(h.getCellID()) & 0x1F, 0.06)**2
        sxt += w * d * h.getTime()
        sxx += w * d * d
    if sxt <= 0:
        return None
    return sxx / sxt / C_MM_PER_NS   # slope dt/dx = sxt / sxx = 1 / (beta c)


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


def seed_line_match(seeds, lines):
    """Per seed: the index in `lines` (x0, y0, z0, ux, uy, uz) of the stau line
    most of its hits lie on (GEO_TOL_*, ana::seedLineMatch; -1 for none), and
    how many of its hits do."""
    best = np.full(len(seeds), -1, dtype=np.int64)
    n_on = np.zeros(len(seeds), dtype=np.int64)
    if len(seeds) and len(lines):
        arr = np.ascontiguousarray(lines, dtype=np.float64).ravel()
        _cpp("seedLineMatch", seeds)(seeds, arr, len(lines), GEO_TOL_VXD, GEO_TOL_OTHER, best, n_on)
    return best, n_on


def geo_stau_hits(line, xs, ys, zs, cids):
    """Reco hits on a stau's truth line (GEO_TOL_*), ahead of its vertex: VXD
    hits, distinct VXD stations (doublet = one), IT hits, OT hits, and the
    distances of the VXD hits within GEO_RESID_RANGE of it (BIB included)."""
    x0, y0, z0, ux, uy, uz = line
    dx, dy, dz = xs - x0, ys - y0, zs - z0
    t = dx * ux + dy * uy + dz * uz
    d = np.sqrt((dx - t * ux) ** 2 + (dy - t * uy) ** 2 + (dz - t * uz) ** 2)
    sysid = cids & 0x1F
    vxd = (sysid == 1) | (sysid == 2)
    on = (t > 0) & (d < np.where(vxd, GEO_TOL_VXD, GEO_TOL_OTHER))
    v = on & vxd
    stations = len(set(zip(sysid[v].tolist(), (cell_layer(cids[v]) // 2).tolist())))
    n_it = int((on & ((sysid == 3) | (sysid == 4))).sum())
    n_ot = int((on & ((sysid == 5) | (sysid == 6))).sum())
    resid = d[(t > 0) & vxd & (d < GEO_RESID_RANGE)]
    return int(v.sum()), stations, n_it, n_ot, resid


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


# Residual / pull histograms of the stau-matched tracks, from ip_residuals():
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
# (see expand_input). A directory of condor stage-1 results (acc_<reco dir>,
# condor/stauacc/) is loaded instead: same plots, no reco file read.
file_names = [
    #"/ospool/uc-shared/project/futurecolliders/larsonma/stau_4000GeV_30ns/reco_fullBIB_coned_loose_timing_rawtime_VXDNominal_InOut_TwoWay_No4D",
    #"/ospool/uc-shared/project/futurecolliders/larsonma/stau_4000GeV_30ns/reco_fullBIB_coned_loose_timing_rawtime_VXDNominal_InOut_TwoWay_Beta0p6"

    "/ospool/uc-shared/project/futurecolliders/larsonma/stau_4000GeV_30ns/acc_reco_fullBIB_coned_loose_timing_rawtime_VXDNominal_InOut_TwoWay_Beta0p6"
]
# The same two runs from condor stage 1 (condor/stauacc/submit_all_stauacc.sh):
#file_names = [
#    "/ospool/uc-shared/project/futurecolliders/larsonma/stau_4000GeV_30ns/acc_reco_fullBIB_coned_loose_timing_rawtime_VXDNominal_InOut_TwoWay_No4D",
#    "/ospool/uc-shared/project/futurecolliders/larsonma/stau_4000GeV_30ns/acc_reco_fullBIB_coned_loose_timing_rawtime_VXDNominal_InOut_TwoWay_Beta0p6",
#]
# Optional custom legend labels for cross-file overlay plots.
# - Leave as None to fall back to the filename-derived defaults.
# - Otherwise must be a list of strings the same length as file_names.
#OVERLAY_LABELS = ["4 TeV, 30 ns, No 4D Tracking", "4 TeV, 30 ns, Beta=0.6 4D Tracking"]
OVERLAY_LABELS = ["4 TeV, 30 ns, Beta=0.6 4D Tracking"]
OVERLAY_DIR = "overlay_plots_stau"  # output directory for cross-file comparison plots

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
    "track_eta_sitracks", "track_matched_pt_sitracks", "track_percent_stau_hits_sitracks",
    "track_pt_resolution_sitracks", "track_reco_pt_sitracks", "track_phi_sitracks",
    "track_d0_sitracks", "track_z0_sitracks",
    "track_d0_residual_sitracks", "track_z0_residual_sitracks",
    "track_pt_resolution_alltracks", "track_d0_residual_alltracks", "track_z0_residual_alltracks",
    "track_pt_pull_sitracks", "track_omega_pull_sitracks", "track_d0_pull_sitracks", "track_z0_pull_sitracks",
    "track_pt_pull_alltracks", "track_omega_pull_alltracks", "track_d0_pull_alltracks", "track_z0_pull_alltracks",
    "track_delta_qoverp_sq_sitracks", "num_bib_hits_sitracks",
    "all_tracks_mult_per_event", "deduped_tracks_mult_per_event", "sitracks_mult_per_event",
    "fake_mult_per_event", "dup_mult_per_event",
    "stau_matched_sitracks_per_event",
    "stau_si_nhits_per_event",   # per event: {stau MCP index: (truth pT, truth eta, SiTrack NHits)}
    "eff_truth_eta", "eff_truth_pt", "eff_truth_phi", "eff_truth_theta_deg",
    "eff_truth_d0", "eff_truth_z0", "eff_truth_vrxy", "eff_truth_vz",
    "eff_matched_si", "eff_matched_alltracks",
    "seed_eff_eta", "seed_eff_pt", "seed_eff_matched",
    "seed_nseeds_per_event", "seed_pt_best_per_stau",
    # every selected stau (select_staus), in or out of the acceptance
    "stau_truth_beta", "stau_truth_eta", "stau_truth_pt", "stau_decay_r", "stau_decay_z",
    "stau_in_acc", "stau_has_stau_daughter",
    # per SiTrack matched to a selected stau: fit_beta() and its truth
    "stau_vel_reco_beta", "stau_vel_truth_beta", "stau_vel_nhits", "stau_vel_in_acc",
    # per stau in the acceptance, parallel to eff_truth_*: has a SiTrack passing the tier (TIERS)
    "eff_tier_vb", "eff_tier_ib", "eff_tier_ob",
    # per stau in the acceptance, parallel to eff_truth_*: geometric truth match (GEO_TOL_*)
    "eff_geo_seed", "geo_seed_best", "geo_nhits_vxd", "geo_nstations_vxd", "geo_nhits_it", "geo_nhits_ot",
    "geo_vxd_resid",   # per VXD hit within GEO_RESID_RANGE of a stau line in acceptance: its distance
    # per stau in the acceptance: its SiTracks / AllTracks link weight (purity), NaN without a link
    "stau_purity_si", "stau_purity_all",
    # per event: leading / subleading mass candidate (MASS_*), NaN when there is none
    "lead_pt", "lead_beta", "lead_mass", "sublead_pt", "sublead_beta", "sublead_mass",
    "layer_nhits_per_event",   # per event: {(system, layer): hits}
)
# One value per AllTracks track or per seed (millions on BIB): numpy arrays.
_ARRAY_KEYS = (
    "alltracks_raw_nhits", "alltracks_raw_chi2_reduced", "alltracks_raw_pt",
    "alltracks_raw_d0", "alltracks_raw_z0", "seed_pt_all", "seed_eta_all",
    "fake_pt", "fake_eta", "dup_pt", "dup_eta",
    "seed_d0_all", "seed_zorigin_all", "seed_d0_mu", "seed_zorigin_mu", "seed_mu_truth_pt",
    "seed_layers_all", "seed_layers_mu",
    # per fake (not stau-linked) track: pixel / inner / outer hit counts
    "fake_pixel_nhits_sitracks", "fake_inner_nhits_sitracks", "fake_outer_nhits_sitracks",
    "fake_pixel_nhits_alltracks", "fake_inner_nhits_alltracks", "fake_outer_nhits_alltracks",
)
# Counters, summed over files.
_SUM_KEYS = ("n_events", "matched_tracks_sitracks", "matched_tracks_alltracks",
             "n_seed_hit_matched", "n_seed_dr_matched", "n_staus_moving", "n_staus_selected",
             "n_sel_stau_to_stau", "n_sel_stau_to_stau_daughter_only",
             "n_tier_tracks_vb", "n_tier_tracks_ib", "n_tier_tracks_ob",
             "n_ip_omega_zero_sitracks", "n_ip_omega_zero_alltracks")


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

    # A file podio cannot read (e.g. a reco job whose upload was cut off, leaving
    # a truncated file) is skipped with a warning instead of stopping the run.
    reason = unreadable_reason(fname)
    if reason is None:
        try:
            reader = open_reader([fname])
        except OSError:
            reason = "cannot be opened"
    if reason is not None:
        print(f"WARNING: skipping {fname}: {reason}", flush=True)
        acc["diag_lines"].append(f"[SKIPPED] {fname}: {reason}")
        for k in _ARRAY_KEYS:
            acc[k] = np.empty(0)
        acc["n_events"] = 0
        acc["seconds"] = time.time() - t_start
        return acc
    i = 0
    for event in reader.get("events"):
        if i == 0:
            # Dropped-hit-collection autodetect, per file (see the no-hits
            # mode notes). The env var only forces the mode; the hit-count
            # source is always measured.
            detected, NHITS_SOURCE = _detect_mode(event)
            NO_HITS = detected if _env_no_hits is None else _env_no_hits not in ("0", "false", "False", "")
            acc.update(no_hits=NO_HITS, detected_no_hits=detected, nhits_source=NHITS_SOURCE)

        rz_stau_r = []
        rz_stau_z = []
        rz_bib_r = []
        rz_bib_z = []

        xy_stau_x = []
        xy_stau_y = []
        xy_bib_x = []
        xy_bib_y = []

        hits_in_sitracks_stau_track = set()

        rz_stau_sitracks = []

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
        # stau (single-stau events). Needs the hit links, so unused in no-hits mode.
        _seed_mu_hits = keyed_hit_counts(seedTracks, signal_keys)
        # Per stau (MCP index): seed hits from that stau. Coned + BIB reco
        # outputs drop the sim hits, so there may be no per-hit truth at all:
        # then no seed-stage efficiency (hit_truth False).
        stau_keys = stau_hit_keys(hitRelations) if hitRelations is not None else {}
        hit_truth = bool(signal_keys)
        _moving = signal_particles(mcParticles)
        _selected = select_staus(mcParticles, _moving)
        acc["n_staus_moving"] += len(_moving)
        acc["n_staus_selected"] += len(_selected)
        selected_staus = []   # (TLorentzVector, per-seed hit counts from this stau, in acceptance)
        mass_cands = []       # (pT, beta, mass) of the stau-matched SiTracks passing the mass cuts
        # Geometric truth match (GEO_TOL_*): each selected stau's straight line
        # from its vertex, and per seed the line its hits all lie on (or -1).
        geo_lines = []
        for _p in _selected:
            _v, _mom = mcParticles[_p].getVertex(), mcParticles[_p].getMomentum()
            _pp = sqrt(_mom[0] ** 2 + _mom[1] ** 2 + _mom[2] ** 2)
            geo_lines.append((_v[0], _v[1], _v[2], _mom[0] / _pp, _mom[1] / _pp, _mom[2] / _pp))
        geo_seed_line, geo_seed_non = seed_line_match(seedTracks, geo_lines) if not NO_HITS else (None, None)

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

        # Truth links, kept only above PURITY_MIN; the best link weight per
        # particle (any purity) for the purity plots.
        link_purity_si = {}
        link_purity_all = {}
        for siTrackLink in trackMCParticlesLinks:
            mcp     = siTrackLink.getTo()
            siTrack = siTrackLink.getFrom()
            w       = siTrackLink.getWeight()

            mcp_id = mcp.getObjectID().index
            link_purity_si[mcp_id] = max(w, link_purity_si.get(mcp_id, 0.0))
            if w > PURITY_MIN:
                mcp_to_SiTracks[mcp_id].append((siTrack, w))

        for allTrackLink in allTracksMCParticlesLinks:
            mcp      = allTrackLink.getTo()
            allTrack = allTrackLink.getFrom()
            w        = allTrackLink.getWeight()

            mcp_id = mcp.getObjectID().index
            link_purity_all[mcp_id] = max(w, link_purity_all.get(mcp_id, 0.0))
            if w > PURITY_MIN:
                mcp_to_AllTracks[mcp_id].append((allTrack, w))

        mcpid = None
        mcp_tlv = None
        SiTrackpT = None
        stau_si_keys = set()   # obj_key()s of the SiTracks linked to a signal stau
        stau_all_idx = set()   # AllTracks positions of the tracks linked to a signal stau
        stau_si_nhits = {}     # signal-stau MCP index -> (truth pT, truth eta, NHits of its SiTrack)
        n_si_matched = 0
        for _j, _pos in enumerate(_selected):
            mcp = mcParticles[_pos]
            mcp_p = mcp.getMomentum()
            mcp_tlv = ROOT.TLorentzVector()
            mcp_tlv.SetPxPyPzE(mcp_p[0], mcp_p[1], mcp_p[2], mcp.getEnergy())
            mcpid = mcp.getObjectID().index
            n_si_matched += len(mcp_to_SiTracks.get(mcpid, []))
            in_acc = stau_in_acceptance(mcp_tlv.Eta(), mcp)
            _seed_stau = keyed_hit_counts(seedTracks, stau_keys.get(mcpid, set()))
            selected_staus.append((mcp_tlv, _seed_stau, in_acc))
            _end = mcp.getEndpoint()
            acc["stau_truth_beta"].append(mcp_tlv.Beta())
            acc["stau_truth_eta"].append(mcp_tlv.Eta())
            acc["stau_truth_pt"].append(mcp_tlv.Perp())
            acc["stau_decay_r"].append(float(np.hypot(_end[0], _end[1])))
            acc["stau_decay_z"].append(float(_end[2]))
            acc["stau_in_acc"].append(in_acc)
            acc["stau_has_stau_daughter"].append(has_stau_daughter(mcp))
            stau_si_keys.update(obj_key(t) for t, _ in mcp_to_SiTracks.get(mcpid, []))
            stau_all_idx.update(t.getObjectID().index for t, _ in mcp_to_AllTracks.get(mcpid, []))
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
            # Efficiencies: staus in the acceptance only (stau_in_acceptance).
            if in_acc:
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
                acc["stau_purity_si"].append(link_purity_si.get(mcpid, np.nan))
                acc["stau_purity_all"].append(link_purity_all.get(mcpid, np.nan))
                if not NO_HITS and hit_truth:
                    acc["seed_eff_eta"].append(mcp_tlv.Eta())
                    acc["seed_eff_pt"].append(mcp_tlv.Perp())
                    acc["seed_eff_matched"].append(bool((_seed_stau >= SEED_MATCH_MIN_HITS).any()))
                if not NO_HITS:
                    # most of a seed's 3 hits on this stau's line, over all seeds
                    _on = geo_seed_non[geo_seed_line == _j]
                    _best = int(_on.max()) if len(_on) else 0
                    acc["geo_seed_best"].append(_best)
                    acc["eff_geo_seed"].append(_best >= 3)
                    _g = (geo_stau_hits(geo_lines[_j], _xs, _ys, _zs, _cids) if _keys is not None
                          else (0, 0, 0, 0, np.empty(0)))
                    for _k, _n in zip(("geo_nhits_vxd", "geo_nstations_vxd", "geo_nhits_it", "geo_nhits_ot"), _g):
                        acc[_k].append(_n)
                    acc["geo_vxd_resid"].extend(_g[4].tolist())
            acc["stau_matched_sitracks_per_event"].append(len(mcp_to_SiTracks.get(mcpid, [])))

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
                    # WHY no link was made (no sim-relation for the hits, or the stau
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

            # --- AllTracks (stau-matched) chi2/Ndf collection (minimal — just for the chi2 overlay) ---
            for allTrack, _ in mcp_to_AllTracks.get(mcpid, []):
                _at_n = track_nhits(allTrack)
                if track_has_hits(allTrack) and allTrack.getNdf() > 0:
                    acc["track_chi2_reduced_alltracks"].append(allTrack.getChi2() / float(allTrack.getNdf()))
                _ts = ip_state(allTrack)
                if _ts is not None and _ts.omega == 0:
                    acc["n_ip_omega_zero_alltracks"] += 1
                if track_has_hits(allTrack) and _ts is not None and _ts.omega != 0:
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
            _tier = {key: False for key, *_ in TIERS}
            for siTrack, w in mcp_to_SiTracks.get(mcpid, []):
                hit_matched_counter = 0   # must reset per track; n_hits_si below is per-track
                ts = ip_state(siTrack)   # track state documentation: https://edm4hep.web.cern.ch/classedm4hep_1_1_track_state.html
                # Some AtIP states come back with omega = 0 (no pT): such a
                # track is left out of the IP-parameter lists below, but its
                # hits still go into the tiers and the beta fit.
                ip_ok = ts is not None and ts.omega != 0
                if ts is not None and ts.omega == 0:
                    acc["n_ip_omega_zero_sitracks"] += 1
                chi2_reduced_si = 0
                if(siTrack.getNdf() > 0):
                    chi2_reduced_si = siTrack.getChi2() / float(siTrack.getNdf())
                chi2_sitracks_list.append(chi2_reduced_si)
                if ip_ok:
                    omega = ts.omega
                    SiTrackpT = 0.2998 * BField / fabs(omega * 1000.)
                    theta_si = np.pi/2 - np.arctan(ts.tanLambda)
                    eta_si = -np.log(np.tan(theta_si/2))
                    phi_si = ts.phi
                    d0_si = ts.D0
                    z0_si = ts.Z0
                LC_pixel_nhit_si = 0
                LC_inner_nhit_si = 0
                LC_outer_nhit_si = 0
                vxd_tier_layers = 0.0   # VXD hits weighted by vxd_tier_weight (= layers)
                # Everything in this loop reads hit positions and cellIDs,
                # so it only runs when the hit collection was written out.
                for hit in ([] if NO_HITS else siTrack.getTrackerHits()):
                    position = hit.getPosition()
                    r = sqrt(position[0] ** 2 + position[1] ** 2)
                    hkey = obj_key(hit)
                    hits_in_sitracks_stau_track.add(hkey)
                    # decode system and layer from cellID: "system:5,side:-2,layer:6,module:11,sensor:8"
                    cell_id = int(hit.getCellID())
                    system = cell_id & 0x1F
                    layer_si = cell_layer(cell_id)
                    if 1 <= system <= 6:
                        acc["layer_hits_sitracks"][system][layer_si] += 1
                    if system == 1 or system == 2:
                        LC_pixel_nhit_si += 1
                        vxd_tier_layers += vxd_tier_weight(system, layer_si)
                    elif system == 3 or system == 4:
                        LC_inner_nhit_si += 1
                    elif system == 5 or system == 6:
                        LC_outer_nhit_si += 1
                    if hkey in signal_keys:
                        rz_stau_r.append(r)
                        rz_stau_z.append(position[2])
                        xy_stau_x.append(position[0])
                        xy_stau_y.append(position[1])
                        rz_stau_sitracks.append((r, position[2]))
                        hit_matched_counter += 1
                    else:
                        rz_bib_r.append(r)
                        rz_bib_z.append(position[2])
                        xy_bib_x.append(position[0])
                        xy_bib_y.append(position[1])
                n_hits_si = track_nhits(siTrack)
                _beta = None
                if not NO_HITS:
                    _beta = fit_beta(siTrack.getTrackerHits())
                    if _beta is not None:
                        acc["stau_vel_reco_beta"].append(_beta)
                        acc["stau_vel_truth_beta"].append(mcp_tlv.Beta())
                        acc["stau_vel_nhits"].append(n_hits_si)
                        acc["stau_vel_in_acc"].append(in_acc)
                if not NO_HITS and siTrack.getNdf() > 0 and chi2_reduced_si <= TIER_CHI2NDF_MAX:
                    for key, _, n_vb, n_ib, n_ob in TIERS:
                        if (vxd_tier_layers >= n_vb and LC_inner_nhit_si >= n_ib
                                and LC_outer_nhit_si >= n_ob):
                            _tier[key] = True
                            if in_acc:
                                acc[f"n_tier_tracks_{key}"] += 1   # per track, as Mira counts
                # Mass candidate (see MASS_*): Mira's track-level cuts
                if (ip_ok and _beta is not None and 0 < _beta < 1 and vxd_tier_layers >= 3
                        and LC_inner_nhit_si >= 2 and LC_outer_nhit_si >= 2 and abs(eta_si) < MASS_ETA_MAX
                        and siTrack.getNdf() > 0 and chi2_reduced_si < MASS_CHI2NDF_MAX
                        and SiTrackpT < MASS_PT_MAX):
                    _p = SiTrackpT * sqrt(1.0 + ts.tanLambda ** 2)
                    mass_cands.append((SiTrackpT, _beta, _p * sqrt(1.0 / _beta ** 2 - 1.0)))
                if track_has_hits(siTrack) and ip_ok:
                    p_magnitude = sqrt(mcp_p[0]**2 + mcp_p[1]**2 + mcp_p[2]**2)
                    charge = mcp.getCharge()
                    truthqoverp = float(charge) / p_magnitude
                    qoverp_si = (ts.omega * sin(theta_si) * 1000.) / (0.2998 * BField)
                    stau_si_nhits[mcpid] = (mcp_tlv.Perp(), mcp_tlv.Eta(), n_hits_si)
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
                    # Stau/BIB split and the subsystem breakdown both need
                    # the hits themselves; only the total survives without
                    # them, from the trackerHits relation size.
                    if NO_HITS:
                        if n_hits_si > 0:
                            acc["total_nhits_sitracks"].append(n_hits_si)
                    else:
                        acc["track_percent_stau_hits_sitracks"].append(float(hit_matched_counter) / n_hits_si)
                        acc["num_bib_hits_sitracks"].append(n_hits_si - hit_matched_counter)
                        acc["pixel_nhits_sitracks"].append(LC_pixel_nhit_si)
                        acc["inner_nhits_sitracks"].append(LC_inner_nhit_si)
                        acc["outer_nhits_sitracks"].append(LC_outer_nhit_si)
                        acc["total_nhits_sitracks"].append(LC_pixel_nhit_si + LC_inner_nhit_si + LC_outer_nhit_si)
            if in_acc and not NO_HITS:
                for key, *_ in TIERS:
                    acc[f"eff_tier_{key}"].append(_tier[key])

        # Leading / subleading mass candidate of the event (by pT), NaN if none.
        mass_cands.sort(key=lambda c: -c[0])
        for _name, _k in (("lead", 0), ("sublead", 1)):
            _c = mass_cands[_k] if len(mass_cands) > _k else (np.nan, np.nan, np.nan)
            for _q, _v in zip(("pt", "beta", "mass"), _c):
                acc[f"{_name}_{_q}"].append(_v)

        # Tracks linked to any stau (stau -> stau copies too) are not fakes,
        # as in LLPStudies/MarkLLPCode/analysis/analysis.py.
        for _p in _moving:
            _id = mcParticles[_p].getObjectID().index
            stau_si_keys.update(obj_key(t) for t, _ in mcp_to_SiTracks.get(_id, []))
            stau_all_idx.update(t.getObjectID().index for t, _ in mcp_to_AllTracks.get(_id, []))
        # stau -> stau (+ photon): how often only the daughter stau got the SiTrack.
        # If often, matching to either stau of the chain may be needed.
        for _p in _selected:
            _m = mcParticles[_p]
            _dau = [d.getObjectID().index for d in _m.getDaughters() if is_signal_pdg(d.getPDG())]
            if not _dau:
                continue
            acc["n_sel_stau_to_stau"] += 1
            if not mcp_to_SiTracks.get(_m.getObjectID().index) and any(mcp_to_SiTracks.get(d) for d in _dau):
                acc["n_sel_stau_to_stau_daughter_only"] += 1

        # Event hit maps, drawn later by plot_entry: hit positions only, so
        # nothing to draw in no-hits mode (hits_in_sitracks_stau_track stays
        # empty there anyway).
        want_display = (not NO_HITS and hit_truth and n_si_matched == 1
                        and len(hits_in_sitracks_stau_track) > 0 and len(rz_bib_r) > 0
                        and len(acc["event_displays"]) < MAX_EVENT_DISPLAYS)
        want_all_rz = not NO_HITS and n_si_matched >= 1 and len(acc["all_rz_displays"]) < 10
        if want_display or want_all_rz:
            # Stau hits the track did not pick up.
            _missed = np.zeros(0, dtype=bool)
            if _keys is not None and signal_keys:
                _missed = np.isin(_keys, np.fromiter(signal_keys, dtype=np.int64, count=len(signal_keys)))
                _missed &= ~np.isin(_keys, np.fromiter(hits_in_sitracks_stau_track, dtype=np.int64,
                                                       count=len(hits_in_sitracks_stau_track)))
            disp = dict(
                rz_stau_r=rz_stau_r, rz_stau_z=rz_stau_z, rz_bib_r=rz_bib_r, rz_bib_z=rz_bib_z,
                xy_stau_x=xy_stau_x, xy_stau_y=xy_stau_y, xy_bib_x=xy_bib_x, xy_bib_y=xy_bib_y,
                rz_stau_notintrack_r=_rs[_missed].tolist() if _missed.any() else [],
                rz_stau_notintrack_z=_zs[_missed].tolist() if _missed.any() else [],
                xy_stau_notintrack_x=_xs[_missed].tolist() if _missed.any() else [],
                xy_stau_notintrack_y=_ys[_missed].tolist() if _missed.any() else [],
                rz_stau_sitracks=rz_stau_sitracks, chi2_sitracks_list=chi2_sitracks_list,
                truth_pt=mcp_tlv.Perp() if mcp_tlv is not None else None, reco_pt=SiTrackpT,
            )
            if want_display:
                acc["event_displays"].append((i, disp))
            if want_all_rz:
                acc["all_rz_displays"].append((i, disp))

        # Every event, so entries can be paired event by event.
        acc["stau_si_nhits_per_event"].append(stau_si_nhits)

        # Fake SiTracks: with hits, but not truth-linked to a signal stau. pT and
        # eta at the IP state, as for the stau-matched SiTracks.
        _si_keys = np.fromiter((obj_key(t) for t in siTracks), dtype=np.int64, count=len(siTracks))
        _fake = _si["has_hits"] & ~np.isin(_si_keys, np.fromiter(stau_si_keys, dtype=np.int64,
                                                                 count=len(stau_si_keys)))
        acc["fake_mult_per_event"].append(int(_fake.sum()))
        # Pixel / inner / outer hits of fake SiTracks and fake AllTracks (AllTracks
        # not linked to a signal stau, picked by position, no per-track Python).
        # Needs the hit cellIDs, so not in no-hits mode.
        if not NO_HITS:
            _at_fake = _at["has_hits"].copy()
            _at_fake[list(stau_all_idx)] = False
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
        # quantities) for every seed, and for the stau's 3/3 seeds with its truth pT.
        if not NO_HITS:
            _d0s, _zos = seed_impact(seedTracks)
            raw["seed_d0_all"].append(_d0s)
            raw["seed_zorigin_all"].append(_zos)
            _mu = _seed_mu_hits >= SEED_MATCH_MIN_HITS
            raw["seed_d0_mu"].append(_d0s[_mu])
            raw["seed_zorigin_mu"].append(_zos[_mu])
            _truth_pt = np.full(len(_mu), np.nan)
            for _tlv, _cnt, _ in selected_staus:
                _truth_pt[_cnt >= SEED_MATCH_MIN_HITS] = _tlv.Perp()
            raw["seed_mu_truth_pt"].append(_truth_pt[_mu])
            _lay = seed_layers(seedTracks)
            raw["seed_layers_all"].append(_lay)
            raw["seed_layers_mu"].append(_lay[_mu])
        # The stau's seed: one with >= SEED_MATCH_MIN_HITS stau hits (of several,
        # the one with the most, then the closest to the truth direction); if
        # there is none, the closest seed within SEED_MATCH_MAX_DR.
        # Staus in the acceptance only, like the efficiency denominator.
        for mcp_tlv, _seed_stau, _in_acc in (selected_staus if len(pt_s) else []):
            if not _in_acc:
                continue
            dphi = (phi_s - mcp_tlv.Phi() + np.pi) % (2 * np.pi) - np.pi
            dr = np.hypot(eta_s - mcp_tlv.Eta(), dphi)
            dr[np.isnan(dr)] = np.inf
            n_stau_hits = _seed_stau[_ok]
            cand = np.nonzero(n_stau_hits >= SEED_MATCH_MIN_HITS)[0]
            j = None
            if len(cand):
                j = cand[np.lexsort((dr[cand], -n_stau_hits[cand]))[0]]
                acc["n_seed_hit_matched"] += 1
            elif dr.min() < SEED_MATCH_MAX_DR:
                j = int(np.argmin(dr))
                acc["n_seed_dr_matched"] += 1
            if j is not None:
                acc["seed_pt_best_per_stau"].append((mcp_tlv.Perp(), float(pt_s[j])))

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
    m["event_displays"] = m["event_displays"][:MAX_EVENT_DISPLAYS]
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
    """Stau / non-stau / missed-stau hits of one event; view is "rz" or "xy"."""
    def pts(name):
        return (d[f"rz_{name}_z"], d[f"rz_{name}_r"]) if view == "rz" else (d[f"xy_{name}_x"], d[f"xy_{name}_y"])

    if len(pts("stau")[0]) > 0:
        ax_non.scatter(*pts("stau"), s=30, color='steelblue', alpha=0.6, label="Stau-matched hits in track")
    if len(pts("bib")[0]) > 0:
        ax_non.scatter(*pts("bib"), s=30, color='tab:orange', alpha=0.6, label="Non-stau hits")
    if len(pts("stau_notintrack")[0]) > 0:
        ax_non.scatter(*pts("stau_notintrack"), s=40, marker='^', color='red', alpha=0.7, label="Stau hits (not in track)")


def _hit_map_title(view_name, d):
    title_non = f"SiTracks {view_name} Hit Map"
    if d["chi2_sitracks_list"]:
        title_non += f"\nχ²/Ndf = {np.mean(d['chi2_sitracks_list']):.2f}"
    if d["truth_pt"] is not None:
        title_non += f"\nTruth Stau $p_{{T}}$ = {d['truth_pt']:.1f} GeV"
    if d["reco_pt"] is not None:
        title_non += f"\nReconstructed Track $p_{{T}}$ = {d['reco_pt']:.1f} GeV"
    return title_non


def _sorted_fit_points(d):
    sorted_rz = sorted(d["rz_stau_sitracks"], key=lambda p: float(p[0])**2 + float(p[1])**2)
    z_fit = np.array([float(p[1]) for p in sorted_rz], dtype=float)
    r_fit = np.array([float(p[0]) for p in sorted_rz], dtype=float)
    return z_fit, r_fit


def draw_event_rz(d, i, plotdir, pdf):
    """r-z hit map of an event with exactly one stau-matched SiTrack."""
    fig, ax_non = plt.subplots(1, 1, figsize=(6, 5))
    _draw_detector_rz(ax_non)
    _scatter_event_hits(ax_non, d, "rz")
    if len(d["rz_stau_sitracks"]) > 0:
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
    """r-z hit map for the first 10 events with any stau-matched SiTrack."""
    fig, ax_non = plt.subplots(1, 1, figsize=(6, 5))
    _draw_detector_rz(ax_non)
    _scatter_event_hits(ax_non, d, "rz")
    if len(d["rz_stau_sitracks"]) > 0:
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
        with PdfPages(os.path.join(plotdir, "all_rz_hit_maps_stau_matched.pdf")) as all_rz_pdf:
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
    track_percent_stau_hits_sitracks = acc["track_percent_stau_hits_sitracks"]
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
    stau_matched_sitracks_per_event = acc["stau_matched_sitracks_per_event"]
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
    seed_pt_best_per_stau = acc["seed_pt_best_per_stau"]
    layer_nhits_per_event = acc["layer_nhits_per_event"]
    fake_mult_per_event = acc["fake_mult_per_event"]
    fake_pt = acc["fake_pt"]
    fake_eta = acc["fake_eta"]

    if len(stau_matched_sitracks_per_event) > 0:
        max_mult = max(stau_matched_sitracks_per_event) if stau_matched_sitracks_per_event else 5
        bins_mm = np.arange(-0.5, max_mult + 1.5, 1)
        plt.figure()
        plt.hist(stau_matched_sitracks_per_event, bins=bins_mm,
                 weights=unit_weights(stau_matched_sitracks_per_event, bins_mm),
                 histtype='step', linewidth=1.5, label="SiTracks")
        plt.xlabel("Number of SiTracks matched to the stau")
        plt.ylabel("Fraction of selected staus")
        plt.yscale('log')
        plt.legend()
        plt.title("Stau-Matched Track Multiplicity per Stau")
        plt.xticks(range(max_mult + 2))
        plt.gca().xaxis.set_minor_locator(NullLocator())
        plt.tight_layout()
        plt.savefig(plotdir + "/stau_matched_track_mult_per_event.pdf")
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
              "Fake SiTracks: SiTracks not matched to the stau):")
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

    # --- Fake SiTracks (with hits, not matched to the stau): rate vs pT threshold, pT and eta
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
        plt.title(r"Fake Track Rate vs $p_T$ Threshold (SiTracks not matched to the stau)")
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
        plt.title(r"Fake Track $p_T$ (SiTracks not matched to the stau)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plotdir + "/fake_pt.pdf")
        plt.close()

        plt.figure()
        plt.hist(fake_eta, bins=np.linspace(-3, 3, 61), weights=np.ones(len(fake_eta)) / len(fake_eta),
                 histtype='step', linewidth=1.5, label=fake_label)
        plt.xlabel(r"Track $\eta$")
        plt.ylabel("Fraction of fake tracks")
        plt.title(r"Fake Track $\eta$ (SiTracks not matched to the stau)")
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
            for key, color, name in [(all_key, 'gray', "all seeds"), (mu_key, 'steelblue', "stau seeds (3/3)")]:
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
                plt.colorbar(img, ax=ax, label="Stau seeds")
            ax.axhline(cut, color='red', linestyle='--', linewidth=1)
            ax.set_xscale('log')
            ax.set_xlabel(r"Truth Stau $p_T$ [GeV]")
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

    # Per stau in the acceptance (up to 2 per event), not per event as for the
    # one-muon samples: the counters only count staus in the acceptance.
    n_acc = len(eff_truth_eta)
    eff_sitracks  = matched_tracks_sitracks  / n_acc if n_acc > 0 else 0.0
    eff_alltracks = matched_tracks_alltracks / n_acc if n_acc > 0 else 0.0
    print(f"AllTracks (>=1 truth-matched, purity > {PURITY_MIN:g}) efficiency: {matched_tracks_alltracks}/{n_acc} "
          f"staus in acceptance ({100*eff_alltracks:.1f}%)")
    print(f"SiTracks (exactly 1 matched, purity > {PURITY_MIN:g}) efficiency:   {matched_tracks_sitracks}/{n_acc} "
          f"staus in acceptance ({100*eff_sitracks:.1f}%)")

    # --- Track purity: the TrackTruthAlg link weight of each stau in the acceptance ---
    pur = {"SiTracks": np.asarray(acc["stau_purity_si"], dtype=float),
           "AllTracks": np.asarray(acc["stau_purity_all"], dtype=float)}
    if len(pur["SiTracks"]):
        for name, p in pur.items():
            linked = np.isfinite(p)
            print(f"[PURITY] {name}: {int(linked.sum())}/{len(p)} staus in acceptance have a link (any purity), "
                  f"{int((p > PURITY_MIN).sum())} above {PURITY_MIN:g}; purity median "
                  f"{np.median(p[linked]) if linked.any() else float('nan'):.3f}, "
                  f"< 0.5: {int((p[linked] < 0.5).sum())}")
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
        pb = np.linspace(0, 1, 41)
        for (name, p), color in zip(pur.items(), ("steelblue", "purple")):
            linked = np.isfinite(p)
            axs[0].hist(p[linked], bins=pb, histtype='step', linewidth=1.5, color=color,
                        label=f"{name} ({int(linked.sum())} staus with a link)")
            cuts = np.linspace(0, 1, 101)
            axs[1].plot(cuts, [np.mean(p > c) if len(p) else np.nan for c in cuts], color=color,
                        label=f"{name}: {100 * np.mean(p > PURITY_MIN):.1f}% at > {PURITY_MIN:g}")
        for ax in axs:
            ax.axvline(PURITY_MIN, color='gray', linestyle='--', linewidth=0.8)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(fontsize=8)
        axs[0].set_xlabel("Track purity (stau hits / track hits, TrackTruthAlg link weight)")
        axs[0].set_ylabel("Staus in acceptance")
        axs[0].set_yscale('log')
        axs[1].set_xlabel("Minimum purity")
        axs[1].set_ylabel("Efficiency (staus in acceptance)")
        axs[1].set_ylim(0, 1.05)
        fig.tight_layout()
        plt.savefig(plotdir + "/stau_track_purity.pdf")
        plt.close(fig)

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
        plt.title("Matched Stau Tracks Chi2 Reduced")
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
        plt.xlabel("Truth Stau $p_T$ [GeV]")
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
        plt.xlabel("Truth Stau $p_T$ [GeV]")
        plt.ylabel("Chi2 Reduced")
        plt.title("$p_T$ vs Chi2 Reduced (SiTracks)")
        plt.tight_layout()
        plt.savefig(plotdir + "/pt_vs_chi2_sitracks_hist2d.pdf")
        plt.close()

    # The panels below all need the per-hit stau/BIB classification, which only
    # exists when the hit relations were written out.
    if len(track_chi2_reduced_sitracks) > 0 and len(track_percent_stau_hits_sitracks) > 0:
        xbins = np.linspace(0, 2500, 26)
        ybins = np.linspace(0, 1, 21)
        plt.figure()
        img = hist2d_lognorm(track_matched_pt_sitracks, track_percent_stau_hits_sitracks, [xbins, ybins])
        if img is not None:
            plt.colorbar(img, label="Counts")
        plt.xlabel("Truth Stau $p_T$ [GeV]")
        plt.ylabel("Fraction of Stau Hits")
        plt.ylim(0, 1)
        plt.title("$p_T$ vs Fraction of Stau Hits (SiTracks)")
        plt.tight_layout()
        plt.savefig(plotdir + "/pt_vs_stauhitfraction_sitracks_hist2d.pdf")
        plt.close()

        xbins = np.linspace(0, 1, 21)
        ybins = np.logspace(-2, 4, 60)
        plt.figure()
        img = hist2d_lognorm(track_percent_stau_hits_sitracks, track_chi2_reduced_sitracks, [xbins, ybins])
        if img is not None:
            plt.colorbar(img, label="Counts")
        plt.yscale("log")
        plt.xlabel("Fraction of Stau Hits")
        plt.ylabel("Chi2 Reduced")
        plt.xlim(0, 1)
        plt.title("Fraction of Stau Hits vs Chi2 Reduced (SiTracks)")
        plt.tight_layout()
        plt.savefig(plotdir + "/stauhitfraction_vs_chi2_sitracks_hist2d.pdf")
        plt.close()

        xbins = np.linspace(0, 10, 11)
        ybins = np.logspace(-2, 4, 60)
        plt.figure()
        img = hist2d_lognorm(num_bib_hits_sitracks, track_chi2_reduced_sitracks, [xbins, ybins])
        if img is not None:
            plt.colorbar(img, label="Counts")
        plt.yscale("log")
        plt.xlabel("# BIB Hits in Stau Matched Track")
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
        img = hist2d_lognorm(track_percent_stau_hits_sitracks, track_pt_resolution_sitracks, [xbins, ybins], weights=w_pt_si)
        if img is not None:
            plt.colorbar(img, label="Fraction of tracks")
        plt.xlabel("Fraction of Stau Hits")
        plt.ylabel(r"$p_T$ Resolution $(p_T^{\mathrm{reco}} - p_T^{\mathrm{truth}}) / p_T^{\mathrm{truth}}$")
        plt.xlim(0, 1)
        plt.title(r"$p_T$ Resolution vs Fraction of Stau Hits (SiTracks)")
        plt.tight_layout()
        plt.savefig(plotdir + "/pt_resolution_vs_stauhitfraction_sitracks_hist2d.pdf")
        plt.close()

        min_dqp_si = max(min(track_delta_qoverp_sq_sitracks), 1e-12)
        xbins = np.linspace(0, 1, 21)
        ybins = np.logspace(np.log10(min_dqp_si), np.log10(max(track_delta_qoverp_sq_sitracks) * 1.1), 81)
        w_si = np.ones(len(track_delta_qoverp_sq_sitracks)) / len(track_delta_qoverp_sq_sitracks)
        plt.figure()
        img = hist2d_lognorm(track_percent_stau_hits_sitracks, track_delta_qoverp_sq_sitracks, [xbins, ybins], weights=w_si)
        if img is not None:
            plt.colorbar(img, label="Fraction of tracks")
        plt.xlabel("Fraction of Stau Hits")
        plt.ylabel(r"$\sigma(q/p)^2 = (q/p_{\mathrm{reco}} - q/p_{\mathrm{truth}})^2$ [GeV$^{-2}$]")
        plt.xlim(0, 1)
        plt.yscale("log")
        plt.title(r"$\sigma(q/p)^2$ vs Fraction of Stau Hits (SiTracks)")
        plt.tight_layout()
        plt.savefig(plotdir + "/delta_qoverp_sq_vs_stauhitfraction_sitracks_hist2d.pdf")
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
                             ("seed_layers_mu", "stau seeds, 3/3 stau hits", "darkorange")]:
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
        print(f"Staus with a matched seed: {len(seed_pt_best_per_stau)}/{len(eff_truth_pt)} "
              f"({acc['n_seed_hit_matched']} by >= {SEED_MATCH_MIN_HITS} of 3 stau hits, "
              f"{acc['n_seed_dr_matched']} by closest seed within Delta R < {SEED_MATCH_MAX_DR:g})")

    if len(seed_pt_best_per_stau) > 0:
        truth_pts_s = [x[0] for x in seed_pt_best_per_stau]
        seed_pts_s  = [x[1] for x in seed_pt_best_per_stau]
        resolutions_s = [(s - t) / t for t, s in seed_pt_best_per_stau]

        pt_max_s = max(max(truth_pts_s), max(seed_pts_s)) * 1.2
        xbins_s = np.linspace(0, pt_max_s, 51)
        ybins_s = np.linspace(0, pt_max_s, 51)
        plt.figure()
        img = hist2d_lognorm(truth_pts_s, seed_pts_s, [xbins_s, ybins_s])
        if img is not None:
            plt.colorbar(img, label="Counts")
        plt.plot([0, pt_max_s], [0, pt_max_s], 'r--', linewidth=1, label="truth = seed")
        plt.xlabel(r"Truth Stau $p_T$ [GeV]")
        plt.ylabel(r"Best-matched Seed $p_T$ [GeV]")
        plt.legend()
        plt.title(rf"Truth vs Seed $p_T$ ($\geq${SEED_MATCH_MIN_HITS}/3 stau hits, else closest $\Delta R < {SEED_MATCH_MAX_DR:g}$)")
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
        plt.title(rf"Seed $p_T$ Resolution ($\geq${SEED_MATCH_MIN_HITS}/3 stau hits, else closest $\Delta R < {SEED_MATCH_MAX_DR:g}$)")
        plt.tight_layout()
        plt.savefig(plotdir + "/seed_pt_resolution.pdf")
        plt.close()

    # === 1D histogram plots (stau-matched tracks) ===

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
        plt.title("Hit Multiplicity (SiTracks, Stau Matched)")
        plt.tight_layout()
        plt.savefig(plotdir + "/hit_mult_sitracks.pdf")
        plt.close()

    # --- Hit multiplicity: stau-matched vs BIB overlaid (SiTracks) ---
    if len(total_nhits_sitracks) > 0 and len(num_bib_hits_sitracks) > 0:
        stau_hits_si = [nhit - nbib for nhit, nbib in zip(total_nhits_sitracks, num_bib_hits_sitracks)]
        plt.figure()
        plt.hist(stau_hits_si,          bins=bins_nhit, color='steelblue', alpha=0.6, edgecolor='black',
                 label=f"Stau-matched hits  Mean: {np.mean(stau_hits_si):.2f}, Median: {np.median(stau_hits_si):.1f}")
        plt.hist(num_bib_hits_sitracks, bins=bins_nhit, color='salmon',    alpha=0.6, edgecolor='black',
                 label=f"Non-stau hits  Mean: {np.mean(num_bib_hits_sitracks):.2f}, Median: {np.median(num_bib_hits_sitracks):.1f}")
        plt.yscale('log')
        plt.xlim(0, 20)
        plt.xlabel("N hits")
        plt.ylabel("Tracks")
        plt.legend(fontsize=9)
        plt.title("Hit Multiplicity Overlay (SiTracks, Stau Matched)")
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
        plt.title(r"Track $p_T$ (Stau Matched)")
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
        plt.title(r"Track $\eta$ (Stau Matched)")
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
        plt.title(r"Track $\phi$ (Stau Matched)")
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
        plt.title(r"Track $d_0$ (Stau Matched)")
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
        plt.title(r"Track $z_0$ (Stau Matched)")
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
        plt.xlabel("# non-stau (BIB) hits per track")
        plt.ylabel("Fraction of tracks")
        plt.legend(fontsize=9)
        plt.title("# BIB Hits per Track (Stau Matched)")
        plt.tight_layout()
        plt.savefig(plotdir + "/num_bib_hits_overlay.pdf")
        plt.close()

    # --- Stau-hit fraction per track (SiTracks) ---
    if len(track_percent_stau_hits_sitracks) > 0:
        bins_mhf = np.linspace(0, 1, 21)
        plt.figure()
        w = np.ones(len(track_percent_stau_hits_sitracks)) / len(track_percent_stau_hits_sitracks)
        plt.hist(track_percent_stau_hits_sitracks, bins=bins_mhf, weights=w, histtype='step', linewidth=1.7,
                 color='salmon', label=f"SiTracks  mean={np.mean(track_percent_stau_hits_sitracks):.3f}")
        plt.yscale('log')
        plt.xlabel("Stau-matched hit fraction")
        plt.ylabel("Fraction of tracks")
        plt.legend(fontsize=9)
        plt.title("Stau-Matched Hit Fraction per Track")
        plt.tight_layout()
        plt.savefig(plotdir + "/stauhitfrac_overlay.pdf")
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

    # --- Truth stau kinematics: pT, eta, phi, theta ------------------------
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
        fig.suptitle(f"Truth kinematics, staus in acceptance  ({i} events)", fontsize=13)
        fig.tight_layout()
        plt.savefig(plotdir + "/truth_kinematics.pdf")
        plt.close(fig)
        print(f"[TRUTH] staus in the acceptance (efficiency denominator): {len(eff_truth_pt)} in {i} events; "
              f"pT [{min(eff_truth_pt):.2f}, {max(eff_truth_pt):.2f}] GeV, "
              f"theta [{min(eff_truth_theta_deg):.2f}, {max(eff_truth_theta_deg):.2f}] deg")
    else:
        print(f"[TRUTH] WARNING: no staus in the acceptance in {i} events: no efficiency plots.")

    # --- Staus: selection, acceptance, beta -----------------------------------
    if acc["stau_truth_beta"]:
        beta_t = np.asarray(acc["stau_truth_beta"])
        eta_t = np.asarray(acc["stau_truth_eta"])
        dec_r, dec_z = np.asarray(acc["stau_decay_r"]), np.asarray(acc["stau_decay_z"])
        in_acc = np.asarray(acc["stau_in_acc"], dtype=bool)
        n_sel = len(beta_t)
        print(f"[STAU] {acc['n_staus_moving']} moving staus, {n_sel} selected (at the origin, <= "
              f"{STAUS_PER_EVENT} / event: {n_sel / max(i, 1):.2f} / event); in acceptance (|eta| < "
              f"{ACC_ETA_MAX:g}, decay outside r {TRACKER_R:g} / |z| {TRACKER_Z:g} mm): {in_acc.sum()} "
              f"({100 * in_acc.mean():.1f}%); |eta| cut alone {100 * np.mean(np.abs(eta_t) < ACC_ETA_MAX):.1f}%")
        print(f"[STAU] stau-matched tracks with an AtIP state of omega = 0 (left out of the pT / "
              f"residual plots): SiTracks {acc['n_ip_omega_zero_sitracks']}, AllTracks {acc['n_ip_omega_zero_alltracks']}")
        print(f"[STAU] selected with a stau daughter (stau -> stau): {acc['n_sel_stau_to_stau']} "
              f"({100 * acc['n_sel_stau_to_stau'] / n_sel:.1f}%); of those, SiTrack on the daughter only: "
              f"{acc['n_sel_stau_to_stau_daughter_only']}")

        fig, axs = plt.subplots(2, 2, figsize=(11, 8))
        ax = axs[0][0]
        bb = np.linspace(0.4, 1.0, 61)
        ax.hist(beta_t, bins=bb, histtype='step', linewidth=1.5, color='steelblue', label=f"selected ({n_sel})")
        ax.hist(beta_t[in_acc], bins=bb, histtype='step', linewidth=1.5, color='darkorange',
                label=f"in acceptance ({in_acc.sum()})")
        ax.set_xlabel(r"Truth stau $\beta$")
        ax.set_ylabel("Staus")
        ax.set_title(f"mean={beta_t.mean():.3f}, std={beta_t.std():.3f}", fontsize=10)
        ax.legend(fontsize=8)
        ax = axs[0][1]
        ax.hist(eta_t, bins=np.linspace(-3, 3, 61), histtype='step', linewidth=1.5, color='steelblue',
                label="selected")
        ax.hist(eta_t[in_acc], bins=np.linspace(-3, 3, 61), histtype='step', linewidth=1.5, color='darkorange',
                label="in acceptance")
        for x in (-ACC_ETA_MAX, ACC_ETA_MAX):
            ax.axvline(x, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xlabel(r"Truth stau $\eta$")
        ax.set_ylabel("Staus")
        ax.legend(fontsize=8)
        ax = axs[1][0]
        rb, zb = np.linspace(0, 20000, 81), np.linspace(-20000, 20000, 81)
        img = hist2d_lognorm(np.clip(dec_z, zb[0], zb[-1]), np.clip(dec_r, rb[0], rb[-1]), [zb, rb])
        plt.sca(ax)
        if img is not None:
            plt.colorbar(img, ax=ax, label="Staus")
        ax.add_patch(patches.Rectangle((-TRACKER_Z, 0), 2 * TRACKER_Z, TRACKER_R, fill=False, color='red',
                                       linewidth=1.2, label="tracker"))
        ax.set_xlabel("Decay z [mm] (edge bins: overflow)")
        ax.set_ylabel("Decay r [mm] (edge bins: overflow)")
        ax.legend(fontsize=8, loc='upper right')
        ax = axs[1][1]
        ax.hist(np.clip(dec_r, rb[0], rb[-1]), bins=rb, histtype='step', linewidth=1.5, color='steelblue')
        ax.axvline(TRACKER_R, color='red', linestyle='--', linewidth=0.8, label=f"r = {TRACKER_R:g} mm")
        ax.set_xlabel("Decay r [mm] (last bin: overflow)")
        ax.set_ylabel("Staus")
        ax.set_title(f"decay inside the tracker: {100 * np.mean(~np.array([r > TRACKER_R or abs(z) > TRACKER_Z for r, z in zip(dec_r, dec_z)])):.1f}%",
                     fontsize=10)
        ax.legend(fontsize=8)
        fig.suptitle(f"Selected staus ({i} events)", fontsize=13)
        fig.tight_layout()
        plt.savefig(plotdir + "/stau_selection.pdf")
        plt.close(fig)

    # Velocity from the matched SiTracks' raw hit times (fit_beta)
    if acc["stau_vel_reco_beta"]:
        b_r = np.asarray(acc["stau_vel_reco_beta"])
        b_t = np.asarray(acc["stau_vel_truth_beta"])
        v_acc = np.asarray(acc["stau_vel_in_acc"], dtype=bool)
        print(f"[STAU] beta fit on {len(b_r)} stau SiTracks ({v_acc.sum()} in acceptance): "
              f"reco mean {b_r.mean():.3f} median {np.median(b_r):.3f}, truth mean {b_t.mean():.3f}")
        fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
        ax = axs[0]
        bb = np.linspace(0.3, 1.3, 101)
        ax.hist(np.clip(b_r, bb[0], bb[-1]), bins=bb, histtype='step', linewidth=1.5, color='steelblue',
                label=f"reco, all ({len(b_r)})")
        ax.hist(np.clip(b_r[v_acc], bb[0], bb[-1]), bins=bb, histtype='step', linewidth=1.5,
                color='darkorange', label=f"reco, in acceptance ({v_acc.sum()})")
        ax.hist(b_t, bins=bb, histtype='step', linewidth=1.5, color='black', linestyle='--', label="truth")
        ax.axvline(1.0, color='gray', linestyle=':', linewidth=0.8)
        ax.set_xlabel(r"$\beta$ (edge bins: overflow)")
        ax.set_ylabel("Stau SiTracks")
        ax.set_yscale('log')
        ax.set_title("Fitted velocity: position vs. raw hit time", fontsize=10)
        ax.legend(fontsize=8)
        ax = axs[1]
        db = np.linspace(-0.2, 0.2, 81)
        for sel, col, name in [(np.ones(len(b_r), dtype=bool), 'steelblue', "all"), (v_acc, 'darkorange', "in acceptance")]:
            d = b_r[sel] - b_t[sel]
            if len(d):
                ax.hist(np.clip(d, db[0], db[-1]), bins=db, histtype='step', linewidth=1.5, color=col,
                        label=f"{name}: median {np.median(d):.3g}, std {np.std(d):.3g}")
        ax.set_xlabel(r"$\beta^{\mathrm{reco}} - \beta^{\mathrm{truth}}$ (edge bins: overflow)")
        ax.set_ylabel("Stau SiTracks")
        ax.set_yscale('log')
        ax.legend(fontsize=8)
        ax = axs[2]
        nh = np.asarray(acc["stau_vel_nhits"])
        plt.sca(ax)
        img = hist2d_lognorm(nh, np.clip(b_r - b_t, db[0], db[-1]),
                             [np.arange(BETA_FIT_MIN_HITS - 0.5, max(nh.max(), BETA_FIT_MIN_HITS) + 1.5), db])
        if img is not None:
            plt.colorbar(img, ax=ax, label="Stau SiTracks")
        ax.set_xlabel("Hits on the SiTrack")
        ax.set_ylabel(r"$\beta^{\mathrm{reco}} - \beta^{\mathrm{truth}}$")
        fig.tight_layout()
        plt.savefig(plotdir + "/stau_beta_fit.pdf")
        plt.close(fig)

    # Reconstructed mass of the leading / subleading stau track per event (MASS_*)
    if acc["lead_mass"]:
        n_ev = len(acc["lead_mass"])
        fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
        for ax, (q, bins, xlabel) in zip(axs, STAU_MASS_PLOTS):
            for name, color in (("lead", 'steelblue'), ("sublead", 'darkorange')):
                v = np.asarray(acc[f"{name}_{q}"], dtype=float)
                v = v[np.isfinite(v)]
                ax.hist(np.clip(v, bins[0], bins[-1]), bins=bins, weights=np.full(len(v), 1. / n_ev),
                        histtype='step', linewidth=1.5, color=color,
                        label=f"{'leading' if name == 'lead' else 'subleading'}: {100 * len(v) / n_ev:.1f}% of events"
                              + (f", median {np.median(v):.4g}" if len(v) else ""))
            ax.set_xlabel(xlabel + " (edge bins: overflow)")
            ax.set_ylabel("Fraction of events per bin")
            ax.legend(fontsize=8)
        fig.suptitle(f"Leading / subleading stau track ({n_ev} events; VB+IB+OB tier, |eta| < {MASS_ETA_MAX:g}, "
                     f"chi2/ndf < {MASS_CHI2NDF_MAX:g}, 0 < beta < 1)", fontsize=11)
        fig.tight_layout()
        plt.savefig(plotdir + "/stau_mass_lead_sublead.pdf")
        plt.close(fig)
        for name in ("lead", "sublead"):
            v = np.asarray(acc[f"{name}_mass"], dtype=float)
            v = v[np.isfinite(v)]
            print(f"[MASS] {name}ing track: {len(v)}/{n_ev} events ({100 * len(v) / n_ev:.1f}%)"
                  + (f", mass median {np.median(v):.0f} GeV, 16-84% [{np.percentile(v, 16):.0f}, "
                     f"{np.percentile(v, 84):.0f}] GeV" if len(v) else ""))

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

    # --- Truth stau eta distribution ---
    if len(eff_truth_eta) > 0:
        plt.figure()
        plt.hist(eff_truth_eta, bins=50, range=(-3, 3), color='steelblue', edgecolor='black',
                 label=f"Mean: {np.mean(eff_truth_eta):.2f}\nMedian: {np.median(eff_truth_eta):.2f}")
        plt.xlabel(r"Truth Stau $\eta$")
        plt.ylabel("Staus")
        plt.legend(fontsize=9)
        plt.title("Truth Stau $\\eta$ Distribution")
        plt.tight_layout()
        plt.savefig(plotdir + "/truth_stau_eta.pdf")
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
        eta_bins = ETA_EFF_BINS
        c_eta, eff_si_eta, err_si_eta = compute_efficiency(eff_truth_eta, eff_matched_si, eta_bins)
        c_eta_a, eff_alltracks_eta, err_alltracks_eta = compute_efficiency(eff_truth_eta, eff_matched_alltracks, eta_bins)
        plt.figure()
        plt.errorbar(c_eta_a, eff_alltracks_eta, yerr=err_alltracks_eta, fmt='D-', color='purple',
                     label=f"AllTracks ({100*int_alltracks:.1f}%)",
                     capsize=3)
        plt.errorbar(c_eta, eff_si_eta, yerr=err_si_eta, fmt='o-', color='steelblue',
                     label=f"SiTracks ({100*int_si:.1f}%)",
                     capsize=3)
        plt.axvline(-0.6, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        plt.axvline( 0.6, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        plt.xlabel(r"Truth Stau $\eta$")
        plt.ylabel("Tracking Efficiency")
        plt.xlim(ETA_EFF_BINS[0], ETA_EFF_BINS[-1])
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
                     label=f"AllTracks ({100*int_alltracks:.1f}%)",
                     capsize=3)
        plt.errorbar(c_th, eff_si_th, yerr=err_si_th, fmt='o-', color='steelblue',
                     label=f"SiTracks ({100*int_si:.1f}%)",
                     capsize=3)
        plt.axvline(TH_BRL, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        plt.axvline(np.pi - TH_BRL, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        plt.text(np.pi / 2, 0.03, r"Barrel ($|\eta|<0.6$)", fontsize=7, color='gray', ha='center')
        plt.xlabel(r"Truth Stau $\theta$ [rad]")
        plt.ylabel("Tracking Efficiency")
        plt.xlim(THETA_BINS[0], THETA_BINS[-1])
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
            plt.title(f"Tracking Efficiency vs {xlabel.replace('Truth Stau ', '').split(' [')[0]}")
            plt.grid(True, linestyle=":", alpha=0.7)
            plt.tight_layout()
            plt.savefig(plotdir + f"/efficiency_vs_{tag}.pdf")
            plt.close()

            v = np.asarray(acc[key], dtype=float)
            plt.figure()
            plt.hist(x, bins=bins, color='steelblue', edgecolor='black', alpha=0.7,
                     label=f"N={len(v)}, range [{np.min(v):.3g}, {np.max(v):.3g}] mm")
            plt.xlabel(xlabel + " (edge bins: overflow)")
            plt.ylabel("Staus")
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
             r"Truth Stau $d_0$ [mm]", r"Truth Stau $z_0$ [mm]", "d0_vs_z0"),
            (rxy, vz, DISPLACEMENT_EFF[2][1], np.linspace(-300, 300, 61),
             r"Truth Stau production vertex $r_{xy}$ [mm]", r"Truth Stau production vertex $z$ [mm]", "vrxy_vs_vz"),
        ]:
            plt.figure(figsize=(7, 5.5))
            img = hist2d_lognorm(np.clip(xv, xb[0], xb[-1]), np.clip(yv, yb[0], yb[-1]), [xb, yb])
            if img is not None:
                plt.colorbar(img, label="Staus")
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
             r"Truth Stau $d_0$ [mm]", r"Truth Stau $z_0$ [mm]", "d0_vs_z0"),
            (rxy, vz, np.linspace(0, 300, 21), np.linspace(-300, 300, 41),
             r"Production vertex $r_{xy}$ [mm]", r"Production vertex $z$ [mm]", "vrxy_vs_vz"),
        ]:
            xs, ys = np.clip(xv, xb[0], xb[-1]), np.clip(yv, yb[0], yb[-1])
            tot, _, _ = np.histogram2d(xs, ys, bins=[xb, yb])
            for flags, name in [(eff_matched_alltracks, "alltracks"), (eff_matched_si, "sitracks")]:
                hit, _, _ = np.histogram2d(xs, ys, bins=[xb, yb], weights=np.asarray(flags, dtype=float))
                with np.errstate(divide="ignore", invalid="ignore"):
                    eff = np.where(tot >= EFF2D_MIN_STAUS, hit / tot, np.nan)
                plt.figure(figsize=(7, 5.5))
                m = plt.pcolormesh(xb, yb, np.ma.masked_invalid(eff).T, vmin=0, vmax=1, cmap="viridis")
                plt.colorbar(m, label="Tracking Efficiency")
                plt.xlabel(xl + " (edge bins: overflow)")
                plt.ylabel(yl + " (edge bins: overflow)")
                plt.title(f"Efficiency, {'AllTracks' if name == 'alltracks' else 'SiTracks'} "
                          f"(blank: < {EFF2D_MIN_STAUS} staus)")
                plt.tight_layout()
                plt.savefig(plotdir + f"/efficiency_{tag}_{name}.pdf")
                plt.close()

        # Efficiency vs pT
        pt_bins = STAU_PT_BINS
        c_pt, eff_si_pt, err_si_pt = compute_efficiency(eff_truth_pt, eff_matched_si, pt_bins)
        c_pt_a, eff_alltracks_pt, err_alltracks_pt = compute_efficiency(eff_truth_pt, eff_matched_alltracks, pt_bins)
        plt.figure()
        plt.errorbar(c_pt_a, eff_alltracks_pt, yerr=err_alltracks_pt, fmt='D-', color='purple',
                     label=f"AllTracks ({100*int_alltracks:.1f}%)",
                     capsize=3)
        plt.errorbar(c_pt, eff_si_pt, yerr=err_si_pt, fmt='o-', color='steelblue',
                     label=f"SiTracks ({100*int_si:.1f}%)",
                     capsize=3)
        plt.xlabel(r"Truth Stau $p_T$ [GeV]")
        plt.ylabel("Tracking Efficiency")
        plt.xlim(pt_bins[0], pt_bins[-1])
        eff_ticks()
        plt.legend(fontsize=8, loc='lower center')
        plt.title(r"Tracking Efficiency vs Truth $p_T$")
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.tight_layout()
        plt.savefig(plotdir + "/efficiency_vs_pt.pdf")
        plt.close()

        # Efficiency by stage: the stau has a seed (all 3 hits its own), an
        # AllTracks track, a SiTracks track
        for xkey, skey, bins, xlabel, logx, tag in [
            ("eff_truth_eta", "seed_eff_eta", ETA_EFF_BINS, r"Truth Stau $\eta$", False, "eta"),
            ("eff_truth_pt", "seed_eff_pt", STAU_PT_BINS, r"Truth Stau $p_T$ [GeV]", False, "pt"),
        ]:
            plt.figure()
            if acc["seed_eff_matched"]:
                xs = np.clip(acc[skey], bins[0], None) if logx else acc[skey]
                c, e, err = compute_efficiency(xs, acc["seed_eff_matched"], bins)
                plt.errorbar(c, e, yerr=err, fmt='s-', color='darkorange', capsize=3,
                             label=f"Seed, 3/3 stau hits ({100*np.mean(acc['seed_eff_matched']):.1f}%)")
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

    # --- Tiered efficiency (TIERS): chi2/ndf <= 3 and VB / VB+IB / VB+IB+OB hits ---
    if acc["eff_tier_vb"] and len(acc["eff_tier_vb"]) == len(eff_truth_eta):
        n_acc = len(eff_truth_eta)
        for key, name, *_ in TIERS:
            n_trk = acc[f"n_tier_tracks_{key}"]
            print(f"[TIER] {name}: {100 * np.mean(acc[f'eff_tier_{key}']):.1f}% of {n_acc} staus; "
                  f"counting tracks, as Mira does: {n_trk} / {n_acc} = {100 * n_trk / n_acc:.1f}%")
        for xkey, bins, xlabel, tag in [("eff_truth_eta", ETA_EFF_BINS, r"Truth Stau $\eta$", "eta"),
                                        ("eff_truth_pt", STAU_PT_BINS, r"Truth Stau $p_T$ [GeV]", "pt")]:
            plt.figure()
            c, e, err = compute_efficiency(acc[xkey], eff_matched_si, bins)
            plt.errorbar(c, e, yerr=err, fmt='o-', color='gray', capsize=3,
                         label=f"SiTracks, no cuts ({100 * np.mean(eff_matched_si):.1f}%)")
            for (key, name, *_), color, fmt in zip(TIERS, ("steelblue", "darkorange", "green"), ("s-", "D-", "^-")):
                flags = acc[f"eff_tier_{key}"]
                c, e, err = compute_efficiency(acc[xkey], flags, bins)
                plt.errorbar(c, e, yerr=err, fmt=fmt, color=color, capsize=3,
                             label=f"{name} ({100 * np.mean(flags):.1f}%)")
            plt.xlabel(xlabel)
            plt.ylabel("Tracking Efficiency")
            plt.xlim(bins[0], bins[-1])
            eff_ticks(eta_x=(tag == "eta"))
            plt.legend(fontsize=8, loc='lower center')
            plt.title(rf"Efficiency, SiTracks with $\chi^2$/ndf $\leq$ {TIER_CHI2NDF_MAX:g} (VXD doublet hits count 1/2)")
            plt.grid(True, linestyle=":", alpha=0.7)
            plt.tight_layout()
            plt.savefig(plotdir + f"/efficiency_tiers_vs_{tag}.pdf")
            plt.close()

    # --- Seeding diagnostics from the geometric truth match (GEO_TOL_*) ---
    # Where a stau in the acceptance is lost: no stau hits on enough VXD
    # stations to seed, no seed with its hits on the stau's line, or a stau
    # seed that the CKF did not turn into an AllTracks track. A seed can hold a
    # BIB hit (the triplet windows are sized for 1.5 GeV), and the CKF may still
    # find the stau from it, so seeds with 2 of 3 hits on the line are shown too.
    if acc["eff_geo_seed"] and len(acc["eff_geo_seed"]) == len(eff_truth_eta):
        seed_best = np.asarray(acc["geo_seed_best"])
        seed_ok = seed_best >= 3
        seed_ok2 = seed_best >= 2
        stations = np.asarray(acc["geo_nstations_vxd"])
        seedable = stations >= GEO_MIN_VXD_STATIONS
        found_all = np.asarray(eff_matched_alltracks, dtype=bool)
        found_si = np.asarray(eff_matched_si, dtype=bool)
        n = len(seed_ok)
        pc = lambda m: f"{int(m.sum())} ({100 * m.mean():.1f}%)" if len(m) else "0"
        lost = ~found_all
        print(f"[SEED-GEO] {n} staus in acceptance; a hit is the stau's within {1000 * GEO_TOL_VXD:g} um (VXD) / "
              f"{GEO_TOL_OTHER:g} mm (IT, OT) of its truth line")
        print(f"[SEED-GEO]   stau hits on >= {GEO_MIN_VXD_STATIONS} VXD stations (seedable): {pc(seedable)}")
        print(f"[SEED-GEO]   stau seed, all 3 hits on its line:            {pc(seed_ok)}")
        print(f"[SEED-GEO]   stau seed, >= 2 of 3 hits on its line:        {pc(seed_ok2)}")
        print(f"[SEED-GEO]   AllTracks: {pc(found_all)}, SiTracks: {pc(found_si)}")
        for tag, ok in (("3/3", seed_ok), (">=2/3", seed_ok2)):
            print(f"[SEED-GEO]   {int(lost.sum())} without an AllTracks track ({tag} seed): not seedable "
                  f"{int((lost & ~seedable).sum())}, seedable but no stau seed {int((lost & seedable & ~ok).sum())}, "
                  f"stau seed but no track {int((lost & ok).sum())}; found with no such seed: {int((found_all & ~ok).sum())}")
        print(f"[SEED-GEO]   best seed hits on the line (0-3), found staus: "
              f"{np.bincount(seed_best[found_all], minlength=4).tolist()}, lost: {np.bincount(seed_best[lost], minlength=4).tolist()}")
        for name, key in (("VXD hits", "geo_nhits_vxd"), ("VXD stations", "geo_nstations_vxd"),
                          ("IT hits", "geo_nhits_it"), ("OT hits", "geo_nhits_ot")):
            v = np.asarray(acc[key])
            print(f"[SEED-GEO]   stau {name:<12s} mean: found {v[found_all].mean() if found_all.any() else float('nan'):.2f}, "
                  f"lost {v[lost].mean() if lost.any() else float('nan'):.2f}")

        for xkey, bins, xlabel, tag in [("eff_truth_eta", ETA_EFF_BINS, r"Truth Stau $\eta$", "eta"),
                                        ("eff_truth_pt", STAU_PT_BINS, r"Truth Stau $p_T$ [GeV]", "pt")]:
            plt.figure()
            for flags, fmt, color, name in [
                    (seedable, 'v-', 'gray', f"Stau hits on $\\geq${GEO_MIN_VXD_STATIONS} VXD stations"),
                    (seed_ok2, '^-', 'goldenrod', "Stau seed ($\\geq$2 of 3 hits on its line)"),
                    (seed_ok, 's-', 'darkorange', "Stau seed (3 hits on its line)"),
                    (found_all, 'D-', 'purple', "AllTracks"),
                    (found_si, 'o-', 'steelblue', "SiTracks")]:
                c, e, err = compute_efficiency(acc[xkey], flags, bins)
                plt.errorbar(c, e, yerr=err, fmt=fmt, color=color, capsize=3,
                             label=f"{name} ({100 * np.mean(flags):.1f}%)")
            plt.xlabel(xlabel)
            plt.ylabel("Efficiency")
            plt.xlim(bins[0], bins[-1])
            eff_ticks(eta_x=(tag == "eta"))
            plt.legend(fontsize=8, loc='lower center')
            plt.title(f"Stau seeding (geometric match: {1000 * GEO_TOL_VXD:g} um VXD, {GEO_TOL_OTHER:g} mm IT/OT)",
                      fontsize=10)
            plt.grid(True, linestyle=":", alpha=0.7)
            plt.tight_layout()
            plt.savefig(plotdir + f"/efficiency_seeding_geo_vs_{tag}.pdf")
            plt.close()

        fig, axs = plt.subplots(2, 2, figsize=(11, 8))
        for ax, (name, key, top) in zip(axs.ravel(), [("VXD stations", "geo_nstations_vxd", 8),
                                                     ("VXD hits", "geo_nhits_vxd", 12),
                                                     ("IT hits", "geo_nhits_it", 8),
                                                     ("OT hits", "geo_nhits_ot", 8)]):
            v = np.asarray(acc[key])
            b = np.arange(-0.5, top + 1.5)
            for sel, color, label in [(found_all, 'steelblue', "found (AllTracks)"), (lost, 'salmon', "lost")]:
                if sel.any():
                    ax.hist(np.clip(v[sel], 0, top), bins=b, weights=np.full(sel.sum(), 1 / sel.sum()),
                            histtype='step', linewidth=1.5, color=color, label=f"{label} ({int(sel.sum())})")
            ax.set_xlabel(f"Stau {name} on its truth line (last bin: overflow)")
            ax.set_ylabel("Fraction of staus")
            ax.legend(fontsize=8)
        fig.suptitle("Stau reco hits on the truth line, found vs lost staus (in acceptance)", fontsize=12)
        fig.tight_layout()
        plt.savefig(plotdir + "/geo_stau_hits_found_vs_lost.pdf")
        plt.close(fig)

        # Tolerance check: distance of VXD hits from the stau lines. The stau's
        # own hits should form a peak well inside GEO_TOL_VXD; BIB hits rise
        # roughly linearly with distance (the area grows with it).
        resid = np.asarray(acc["geo_vxd_resid"]) * 1000.   # um
        if len(resid):
            plt.figure()
            rb = np.linspace(0, 1000 * GEO_RESID_RANGE, 101)
            plt.hist(resid, bins=rb, histtype='step', linewidth=1.5, color='steelblue',
                     label=f"{len(resid)} VXD hits, {100 * np.mean(resid < 1000 * GEO_TOL_VXD):.1f}% within the cut")
            plt.axvline(1000 * GEO_TOL_VXD, color='red', linestyle='--', linewidth=0.8,
                        label=f"GEO_TOL_VXD = {1000 * GEO_TOL_VXD:g} um")
            plt.yscale('log')
            plt.xlabel(r"Distance of a VXD hit from the stau truth line [$\mu$m]")
            plt.ylabel("VXD hits")
            plt.title("Geometric match tolerance check (staus in acceptance)", fontsize=10)
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(plotdir + "/geo_vxd_residuals.pdf")
            plt.close()

    # --- Average hits per stau-matched track vs. layer number ---
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
            ax.set_ylabel("Avg hits / stau track")
            ax.set_title(sys_name)
            ax.legend()
            ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
        fig.suptitle("Average Hits per Stau Track vs. Layer (Stau Matched)", fontsize=13)
        fig.tight_layout()
        plt.savefig(plotdir + "/avg_hits_per_layer.pdf")
        plt.close(fig)

    # --- pT / d0 / z0 residuals and pulls (stau matched), one page per collection ---
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
        fig.suptitle(f"Residuals and Pulls ({title_tag}, Stau Matched)", fontsize=13)
        fig.tight_layout()
        plt.savefig(plotdir + f"/residuals_pulls_{file_suffix}.pdf")
        plt.close(fig)

    # Per-stau results for compareStauWithMira.py (plain pickle, see save_acc).
    save_acc({k: acc[k] for k in STAU_OVERLAY_KEYS}, os.path.join(plotdir, "stau_summary.pkl"))

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
        # stau vs BIB hit counts
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
        "track_percent_stau_hits_sitracks":list(track_percent_stau_hits_sitracks),
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
        "stau_matched_sitracks_per_event": list(stau_matched_sitracks_per_event),
        "stau_si_nhits_per_event":         list(acc["stau_si_nhits_per_event"]),
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
        "seed_pt_best_per_stau":   list(seed_pt_best_per_stau),
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
        # stau-specific results for plot_overlays
        "stau": {k: list(acc[k]) for k in STAU_OVERLAY_KEYS},
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

    # ── 1. Stau-matched track multiplicity per event (SiTracks) ──────────
    if _has("stau_matched_sitracks_per_event"):
        all_counts_flat = [v for f in file_names
                           for v in all_files_hit_data[f]["stau_matched_sitracks_per_event"]]
        max_mult = max(all_counts_flat) if all_counts_flat else 5
        bins_mm = np.arange(-0.5, max_mult + 1.5, 1)
        plt.figure()
        for idx, fname in enumerate(file_names):
            arr = all_files_hit_data[fname]["stau_matched_sitracks_per_event"]
            if arr:
                plt.hist(arr, bins=bins_mm, weights=unit_weights(arr, bins_mm), histtype='step', linewidth=1.5,
                         **_style(idx), label=labels[idx])
        plt.xlabel("Number of SiTracks matched to the stau")
        plt.ylabel("Fraction of selected staus")
        plt.yscale('log')
        plt.xticks(range(max_mult + 2))
        plt.gca().xaxis.set_minor_locator(NullLocator())
        plt.title("Stau-Matched Track Multiplicity per Stau (SiTracks)")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(OVERLAY_DIR, "stau_matched_track_mult_per_event_overlay.pdf"))
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
        plt.title(f"Chi2 Reduced ({title_tag}, Stau Matched)")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(OVERLAY_DIR, f"chi2_reduced_{fname_suffix}_overlay.pdf"))
        plt.close()

    # ── 4. Hit multiplicity: pixel / inner / outer / total (2×2 panels) ──────
    # Stau-matched tracks, then fake (not stau-linked) tracks on a log scale.
    for suffix, pixel_key, inner_key, outer_key, total_key, title_tag, ylabel, logy in [
        ("sitracks", "pixel_nhits_sitracks",  "inner_nhits_sitracks", "outer_nhits_sitracks", "total_nhits_sitracks", "SiTracks",
         "Fraction of stau matched tracks", False),
        ("alltracks", "pixel_nhits_alltracks", "inner_nhits_alltracks", "outer_nhits_alltracks", "total_nhits_alltracks", "AllTracks",
         "Fraction of stau matched tracks", False),
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

    # ── 5. Hit multiplicity: stau-matched vs BIB (2 panels) ──────────────────
    for suffix, total_key, bib_key, title_tag in [
        ("sitracks", "total_nhits_sitracks",  "num_bib_hits_sitracks", "SiTracks"),
    ]:
        # Needs the stau/BIB split, so files written without hit relations
        # contribute nothing here.
        if not _has(total_key) or not _has(bib_key):
            continue
        bins_nhit = np.arange(-0.5, 20.5, 1)
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        ax_stau, ax_bib = axs
        for idx, fname in enumerate(file_names):
            total = all_files_hit_data[fname][total_key]
            bib   = all_files_hit_data[fname][bib_key]
            if not total or not bib:
                continue
            stau_hits = [n - b for n, b in zip(total, bib)]
            w_m = np.ones(len(stau_hits)) / len(stau_hits)
            w_b = np.ones(len(bib)) / len(bib)
            ax_stau.hist(stau_hits, bins=bins_nhit, weights=w_m, histtype='step',
                         linewidth=1.5, **_style(idx), label=labels[idx])
            ax_bib.hist(bib, bins=bins_nhit, weights=w_b, histtype='step',
                        linewidth=1.5, **_style(idx), label=labels[idx])
        for ax, title in [
            (ax_stau, f"Stau-matched hits ({title_tag})"),
            (ax_bib,  f"Non-stau (BIB) hits ({title_tag})"),
        ]:
            ax.set_yscale('log')
            ax.set_xlim(0, 20)
            ax.set_xlabel("N hits")
            ax.set_ylabel("Fraction of tracks")
            ax.set_title(title)
            ax.legend(fontsize=7)
        fig.tight_layout()
        plt.savefig(os.path.join(OVERLAY_DIR, f"hit_mult_stau_bib_{suffix}_overlay.pdf"))
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
        plt.title(r"Track $p_T$ (SiTracks, Stau Matched)")
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
        plt.title(r"Track $\eta$ (SiTracks, Stau Matched)")
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
        plt.title(r"Track $\phi$ (SiTracks, Stau Matched)")
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
            plt.title(r"Track $d_0$ (SiTracks, Stau Matched)")
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
            plt.title(r"Track $z_0$ (SiTracks, Stau Matched)")
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(os.path.join(OVERLAY_DIR, "track_z0_overlay.pdf"))
            plt.close()

    # ── 11. Truth stau eta ───────────────────────────────────────────────────
    if _has("eff_truth_eta"):
        plt.figure()
        for idx, fname in enumerate(file_names):
            arr = all_files_hit_data[fname]["eff_truth_eta"]
            if not arr:
                continue
            plt.hist(arr, bins=25, range=(-3, 3), histtype='step', linewidth=1.5,
                     **_style(idx), label=labels[idx])
        plt.xlabel(r"Truth Stau $\eta$")
        plt.ylabel("Staus")
        plt.title("Truth Stau $\\eta$ Distribution")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(OVERLAY_DIR, "truth_stau_eta_overlay.pdf"))
        plt.close()

    # ── 12. Tracking efficiency vs eta (SiTracks) ────────────────────────────
    if _has("eff_truth_eta"):
        eta_bins = ETA_EFF_BINS
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
        plt.xlabel(r"Truth Stau $\eta$")
        plt.ylabel("Tracking Efficiency")
        plt.xlim(ETA_EFF_BINS[0], ETA_EFF_BINS[-1])
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
            plt.xlabel(r"Truth Stau $\eta$")
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
        plt.title(f"Tracking Efficiency vs {xlabel.replace('Truth Stau ', '').split(' [')[0]} (SiTracks)")
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
        plt.ylabel("Fraction of staus")
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
        plt.xlabel(r"Truth Stau $\theta$ [rad]")
        plt.ylabel("Tracking Efficiency")
        plt.xlim(THETA_BINS[0], THETA_BINS[-1])
        eff_ticks()
        plt.title(r"Tracking Efficiency vs $\theta$ (SiTracks)")
        plt.legend(fontsize=7, loc='lower center')
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(OVERLAY_DIR, "efficiency_vs_theta_overlay.pdf"))
        plt.close()

    # ── 13. Tracking efficiency vs pT (SiTracks) ─────────────────────────────
    if _has("eff_truth_pt"):
        pt_bins = STAU_PT_BINS
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
        plt.xlabel(r"Truth Stau $p_T$ [GeV]")
        plt.ylabel("Tracking Efficiency")
        plt.xlim(pt_bins[0], pt_bins[-1])
        eff_ticks()
        plt.title(r"Tracking Efficiency vs $p_T$ (SiTracks)")
        plt.legend(fontsize=7, loc='lower center')
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(OVERLAY_DIR, "efficiency_vs_pt_overlay.pdf"))
        plt.close()

    # ── 13c. Seed-stage and AllTracks-stage efficiency vs eta and pT ─────────
    for flag_key, eta_key, pt_key, stage, stem in [
        ("seed_eff_matched", "seed_eff_eta", "seed_eff_pt", "Seed, 3/3 stau hits", "seed"),
        ("eff_matched_alltracks", "eff_truth_eta", "eff_truth_pt", "AllTracks", "alltracks"),
    ]:
        if not _has(flag_key):
            continue
        for xkey, bins, xlabel, logx, tag in [
            (eta_key, ETA_EFF_BINS, r"Truth Stau $\eta$", False, "eta"),
            (pt_key, STAU_PT_BINS, r"Truth Stau $p_T$ [GeV]", False, "pt"),
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
            plt.title(f"Efficiency vs {xlabel.replace('Truth Stau ', '').replace(' [GeV]', '')} ({stage})")
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
            ax.set_ylabel("Avg hits / stau track")
            ax.set_title(sys_name)
            ax.legend(fontsize=6)
            ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
        fig.suptitle(f"Avg Hits per Stau Track vs. Layer ({title_tag})", fontsize=13)
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
    for key, name in [("seed_layers_all", "all seeds"), ("seed_layers_mu", "stau seeds, 3/3 stau hits")]:
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

    # ── 17. Seed pT resolution (best-matched seed vs truth stau) ─────────────
    if _has("seed_pt_best_per_stau"):
        bins_res_s = np.linspace(-2, 2, 41)
        plt.figure()
        for idx, fname in enumerate(file_names):
            pairs = all_files_hit_data[fname]["seed_pt_best_per_stau"]
            if not pairs:
                continue
            res = [(s - t) / t for t, s in pairs]
            w = np.ones(len(res)) / len(res)
            plt.hist(res, bins=bins_res_s, weights=w, histtype='step', linewidth=1.5,
                     **_style(idx),
                     label=f"{labels[idx]}  μ={np.mean(res):.3f} σ={np.std(res):.3f}")
        plt.xlabel(r"$(p_T^{\mathrm{seed}} - p_T^{\mathrm{truth}}) / p_T^{\mathrm{truth}}$")
        plt.ylabel("Fraction of seeds")
        plt.title(rf"Seed $p_T$ Resolution ($\geq${SEED_MATCH_MIN_HITS}/3 stau hits, else closest $\Delta R < {SEED_MATCH_MAX_DR:g}$)")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(OVERLAY_DIR, "seed_pt_resolution_overlay.pdf"))
        plt.close()

    # ── 18. pT / d0 / z0 residuals and pulls (stau matched), per collection ──
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
            plt.title(f"{title} ({title_tag}, Stau Matched)")
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(os.path.join(OVERLAY_DIR, f"{file_stem}_{file_suffix}_overlay.pdf"))
            plt.close()

    # ── 19. Stau-matched hit fraction — one plot per selection stage ─────────
    for key, fname_suffix, title_tag in [
        ("track_percent_stau_hits_sitracks", "sitracks",    "SiTracks"),
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
        plt.xlabel("Stau-matched hit fraction")
        plt.ylabel("Fraction of tracks")
        plt.yscale('log')
        plt.title(f"Stau-Matched Hit Fraction ({title_tag})")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(OVERLAY_DIR, f"stauhitfrac_{fname_suffix}_overlay.pdf"))
        plt.close()

    # ── 22. Extended / nominal timing: occupancy and per-layer hit ratios ───
    tag_to_entry = {PLOT_TAGS[f]: f for f in file_names}
    for idx, fname in enumerate(file_names):
        if is_timing_variant[idx]:
            plot_timing_ratios(base_tags[idx],
                               all_files_hit_data[tag_to_entry[base_tags[idx]]],
                               all_files_hit_data[fname])

    # ── 23. Fake SiTracks (not matched to the stau): rate vs pT threshold, pT and eta ──
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
        plt.title(r"Fake Track Rate vs $p_T$ Threshold (SiTracks not matched to the stau)")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(OVERLAY_DIR, "fake_rate_vs_pt_threshold_overlay.pdf"))
        plt.close()

        for key, bins, xlabel, title, outname, logx, logy in [
            ("fake_pt", np.logspace(-1, 4, 51), r"Track $p_T$ [GeV]",
             r"Fake Track $p_T$ (SiTracks not matched to the stau)", "fake_pt_overlay.pdf", True, True),
            ("fake_eta", np.linspace(-3, 3, 31), r"Track $\eta$",
             r"Fake Track $\eta$ (SiTracks not matched to the stau)", "fake_eta_overlay.pdf", False, False),
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
        plt.title(("Stau seeds (3/3 stau hits)" if key.endswith("_mu") else "All seeds")
                  + ", red: current cut")
        plt.legend(fontsize=6)
        plt.tight_layout()
        plt.savefig(os.path.join(OVERLAY_DIR, f"seed_{tag}_overlay.pdf"))
        plt.close()

    # ── Stau-specific overlays (plot_entry's "stau" results) ─────────────
    st = {f: all_files_hit_data[f].get("stau", {}) for f in file_names}

    def _st(f, key):
        return np.asarray(st[f].get(key, []), dtype=float)

    def _eff_overlay(ax, flags_of, title, xkey="eff_truth_eta", bins=ETA_EFF_BINS, xlabel=r"Truth Stau $\eta$"):
        """Efficiency of every input on one axis; flags_of(f) -> per-stau flags."""
        for idx, f in enumerate(file_names):
            x, fl = _st(f, xkey), flags_of(f)
            if not len(fl) or len(fl) != len(x):
                continue
            c, e, err = compute_efficiency(x, fl.astype(bool), bins)
            ax.errorbar(c, e, yerr=err, fmt='o', markersize=3, capsize=2, **_style(idx),
                        label=f"{labels[idx]} ({100 * np.mean(fl):.1f}%)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Efficiency")
        ax.set_xlim(bins[0], bins[-1])
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.legend(fontsize=6, loc='lower center')

    def _save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(OVERLAY_DIR, name))
        plt.close(fig)

    # Track efficiency and the tiers vs eta
    if any(len(_st(f, "eff_tier_vb")) for f in file_names):
        fig, axs = plt.subplots(1, 4, figsize=(22, 5))
        _eff_overlay(axs[0], lambda f: _st(f, "eff_matched_si"), f"SiTracks, purity > {PURITY_MIN:g}")
        for ax, (key, name, *_) in zip(axs[1:], TIERS):
            _eff_overlay(ax, lambda f, k=key: _st(f, f"eff_tier_{k}"),
                         f"{name}, chi2/ndf <= {TIER_CHI2NDF_MAX:g}")
        _save(fig, "stau_efficiency_tiers_vs_eta_overlay.pdf")

    # Geometric seeding (GEO_TOL_*) vs eta
    if any(len(_st(f, "geo_seed_best")) for f in file_names):
        fig, axs = plt.subplots(1, 4, figsize=(22, 5))
        _eff_overlay(axs[0], lambda f: _st(f, "geo_nstations_vxd") >= GEO_MIN_VXD_STATIONS,
                     f"Stau hits on >= {GEO_MIN_VXD_STATIONS} VXD stations")
        _eff_overlay(axs[1], lambda f: _st(f, "geo_seed_best") >= 2, "Stau seed, >= 2 of 3 hits on its line")
        _eff_overlay(axs[2], lambda f: _st(f, "geo_seed_best") >= 3, "Stau seed, 3 of 3 hits on its line")
        _eff_overlay(axs[3], lambda f: _st(f, "eff_matched_alltracks"), f"AllTracks, purity > {PURITY_MIN:g}")
        _save(fig, "stau_efficiency_seeding_geo_vs_eta_overlay.pdf")

    # Track purity (TrackTruthAlg link weight)
    if any(len(_st(f, "stau_purity_si")) for f in file_names):
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
        pb = np.linspace(0, 1, 41)
        cuts = np.linspace(0, 1, 101)
        for idx, f in enumerate(file_names):
            p = _st(f, "stau_purity_si")
            if not len(p):
                continue
            linked = np.isfinite(p)
            axs[0].hist(p[linked], bins=pb, weights=np.full(linked.sum(), 1. / max(linked.sum(), 1)),
                        histtype='step', linewidth=1.5, **_style(idx), label=labels[idx])
            axs[1].plot(cuts, [np.mean(p > c) for c in cuts], **_style(idx),
                        label=f"{labels[idx]}: {100 * np.mean(p > PURITY_MIN):.1f}% at > {PURITY_MIN:g}")
        for ax in axs:
            ax.axvline(PURITY_MIN, color='gray', linestyle='--', linewidth=0.8)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(fontsize=6)
        axs[0].set_yscale('log')
        axs[0].set_xlabel("SiTrack purity (TrackTruthAlg link weight)")
        axs[0].set_ylabel("Fraction of linked staus")
        axs[1].set_xlabel("Minimum purity")
        axs[1].set_ylabel("SiTracks efficiency (staus in acceptance)")
        axs[1].set_ylim(0, 1.05)
        _save(fig, "stau_track_purity_overlay.pdf")

    # Fitted velocity
    if any(len(_st(f, "stau_vel_reco_beta")) for f in file_names):
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
        bb, db = np.linspace(0.3, 1.3, 101), np.linspace(-0.2, 0.2, 81)
        for idx, f in enumerate(file_names):
            br, bt = _st(f, "stau_vel_reco_beta"), _st(f, "stau_vel_truth_beta")
            sel = _st(f, "stau_vel_in_acc").astype(bool)
            if not sel.any():
                continue
            d = br[sel] - bt[sel]
            axs[0].hist(np.clip(br[sel], bb[0], bb[-1]), bins=bb, weights=np.full(sel.sum(), 1. / sel.sum()),
                        histtype='step', linewidth=1.5, **_style(idx), label=f"{labels[idx]} ({int(sel.sum())})")
            axs[1].hist(np.clip(d, db[0], db[-1]), bins=db, weights=np.full(len(d), 1. / len(d)),
                        histtype='step', linewidth=1.5, **_style(idx),
                        label=f"{labels[idx]}: median {np.median(d):.3g}, std {np.std(d):.3g}")
        axs[0].set_xlabel(r"Reconstructed $\beta$ (edge bins: overflow)")
        axs[0].set_title("Fitted velocity: position vs. raw hit time (staus in acceptance)", fontsize=10)
        axs[1].set_xlabel(r"$\beta^{\mathrm{reco}} - \beta^{\mathrm{truth}}$ (edge bins: overflow)")
        for ax in axs:
            ax.set_yscale('log')
            ax.set_ylabel("Fraction of stau SiTracks")
            ax.legend(fontsize=6)
        _save(fig, "stau_beta_fit_overlay.pdf")

    # Leading / subleading stau track: mass, beta, pT (fraction of events)
    if any(len(_st(f, "lead_mass")) for f in file_names):
        fig, axs = plt.subplots(2, 3, figsize=(17, 9))
        for row, name in enumerate(("lead", "sublead")):
            for ax, (q, bins, xlabel) in zip(axs[row], STAU_MASS_PLOTS):
                for idx, f in enumerate(file_names):
                    v = _st(f, f"{name}_{q}")
                    n_ev = len(v)
                    v = v[np.isfinite(v)]
                    if not n_ev:
                        continue
                    ax.hist(np.clip(v, bins[0], bins[-1]), bins=bins, weights=np.full(len(v), 1. / n_ev),
                            histtype='step', linewidth=1.5, **_style(idx),
                            label=f"{labels[idx]}: {100 * len(v) / n_ev:.1f}% of events"
                                  + (f", median {np.median(v):.4g}" if len(v) else ""))
                ax.set_xlabel(f"{'Leading' if name == 'lead' else 'Subleading'} track: {xlabel} (edge bins: overflow)")
                ax.set_ylabel("Fraction of events per bin")
                ax.legend(fontsize=6)
        fig.suptitle(f"Leading / subleading stau track (VB+IB+OB tier, |eta| < {MASS_ETA_MAX:g}, "
                     f"chi2/ndf < {MASS_CHI2NDF_MAX:g}, 0 < beta < 1)", fontsize=11)
        _save(fig, "stau_mass_lead_sublead_overlay.pdf")

    # Geometric-match tolerance check
    if any(len(_st(f, "geo_vxd_resid")) for f in file_names):
        fig = plt.figure()
        rb = np.linspace(0, 1000 * GEO_RESID_RANGE, 101)
        for idx, f in enumerate(file_names):
            r = _st(f, "geo_vxd_resid") * 1000.
            if len(r):
                plt.hist(r, bins=rb, weights=np.full(len(r), 1. / len(r)), histtype='step', linewidth=1.5,
                         **_style(idx), label=f"{labels[idx]}: {100 * np.mean(r < 1000 * GEO_TOL_VXD):.1f}% within the cut")
        plt.axvline(1000 * GEO_TOL_VXD, color='red', linestyle='--', linewidth=0.8)
        plt.yscale('log')
        plt.xlabel(r"Distance of a VXD hit from the stau truth line [$\mu$m]")
        plt.ylabel("Fraction of VXD hits")
        plt.legend(fontsize=6)
        _save(fig, "geo_vxd_residuals_overlay.pdf")


def twoway_nhits_compare(all_files_hit_data):
    """NHits of each stau's matched SiTrack, two-way entry vs the one-way
    file_names[0], stau by stau: a printout, and the mean change vs truth eta
    (every stau matched in both, unchanged ones included).

    Events are paired by index, so every entry must be reco of the same events
    in the same order; a stau whose truth pT differs is counted as misaligned
    and skipped.
    """
    labels = list(OVERLAY_LABELS) if OVERLAY_LABELS is not None else [PLOT_TAGS[f] for f in file_names]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    eta_bins = np.linspace(-2.5, 2.5, 26)
    centers = 0.5 * (eta_bins[:-1] + eta_bins[1:])
    ref = all_files_hit_data[file_names[0]]["stau_si_nhits_per_event"]
    print(f"[TWOWAY] Stau-matched SiTrack NHits, two-way vs {labels[0]} (same event, same stau):")
    plt.figure()
    for idx in range(1, len(file_names)):
        other = all_files_hit_data[file_names[idx]]["stau_si_nhits_per_event"]
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
            print("    no stau has a matched SiTrack in both")
        else:
            print(f"    {n} staus matched in both; NHits incremented for {len(up)} ({100. * len(up) / n:.1f}%), "
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
    plt.xlabel(r"Truth Stau $\eta$")
    plt.ylabel(f"Mean NHits change vs {labels[0]}")
    plt.xlim(-2.5, 2.5)
    plt.title(r"Hits Added by Two-Way CKF vs $\eta$ (SiTracks, Stau Matched)")
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
              f"(subsystem split, stau/BIB split, per-layer, event displays, occupancy).")
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

    n_total = len(tasks)

    def collect(entry, k, res):
        """res None: a stage-1 result file that could not be loaded, left out."""
        nonlocal n_done
        n_done += 1
        results[entry][k] = res
        pending[entry] -= 1
        if res is not None:
            print(f"[{n_done}/{n_total}] {PLOT_TAGS[entry]}: {os.path.basename(res['fname'])}, "
                  f"{res['n_events']} events in {res['seconds']:.0f} s "
                  f"({time.time() - t_start:.0f} s elapsed)", flush=True)
        # An entry is plotted as soon as its last file is in, while the
        # workers carry on with the remaining files.
        if pending[entry] == 0:
            all_files_hit_data[entry] = finish_entry(entry, [r for r in results.pop(entry) if r is not None])

    # Stage-1 results (ACC_GLOB, from condor/stauacc/) are loaded, not processed.
    for entry, k, task in [t for t in tasks if t[2][0].endswith(".pkl")]:
        try:
            res = load_acc(task[0])
        except Exception as err:   # e.g. an upload that was cut off
            print(f"WARNING: skipping {task[0]}: cannot be loaded ({err})", flush=True)
            res = None
        collect(entry, k, res)
    tasks = [t for t in tasks if not t[2][0].endswith(".pkl")]
    n_workers = max(1, min(N_WORKERS, len(tasks)))

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
            sys.exit(f"A worker process died: out of memory with N_WORKERS={n_workers} (check dmesg; "
                     f"rerun with a lower N_WORKERS), or a crash such as a segfault on a broken input "
                     f"file (check the log above for a stack trace).")
        finally:
            pool.shutdown(cancel_futures=True)

    plot_overlays({entry: all_files_hit_data[entry] for entry in file_names})
    if TWOWAY_NHITS_COMPARE and len(file_names) > 1:
        twoway_nhits_compare(all_files_hit_data)


if __name__ == "__main__":
    main()
