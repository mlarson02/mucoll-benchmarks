#!/usr/bin/env python3
"""Stage 1 of the track analysis: one reco file in, one ntuple file out.

    python3 trackNtupler.py --input <sample>_reco<N>.edm4hep.root \\
                            --output <sample>_ntuple<N>.root [--coned 0|1]

Runs the event loop of edm4hepTrackAnalyzer.py on a single file and writes two
trees, which trackPlotter.C reads for a whole directory with a TChain:
  events   one entry per event: counts, and arrays for the signal muons, the
           muon-matched SiTracks / AllTracks, every AllTracks track, every
           seed, the hits near z = 0 and the event-display hits
  summary  one entry per file: r-z occupancy, cellID counts and hit sample,
           and the settings the plots quote
condor/ntuple/ runs one job per reco file.
"""
import argparse
import os
import time
import zlib
from collections import defaultdict
from math import sqrt, fabs, sin, cos, atan2, pi

import numpy as np
import ROOT
from podio.reading import get_reader
import edm4hep  # noqa: F401  (loads the EDM4hep dictionaries)

ROOT.gROOT.SetBatch(True)
ROOT.gInterpreter.LoadFile("podio/LinkNavigator.h")
ROOT.gInterpreter.LoadFile("podio/LinkCollection.h")

# ---- settings --------------------------------------------------------------
BField = 5   # T
# |PDG| of the signal particle (both charge signs); SIGNAL_PDG env var overrides.
SIGNAL_PDG = abs(int(os.environ.get("SIGNAL_PDG", 13)))
EFF_DEBUG = False   # per-muon debug printout; slow on BIB (walks every hit link in Python)
GENERATE_ID_DIAG = True
ID_DIAG_TARGET_HITS_PER_SYSTEM = 5000   # cellID scatter: random hits kept per subdetector
ZSEAM_COLLECT_HALF = 25.0      # keep merged hits with |z| < this [mm]
ZSEAM_MAX_HITS = 2000000       # stop collecting them past this many (the plotter caps per input)
# The muon's seed: >= SEED_MATCH_MIN_HITS of its 3 hits from the muon; if none,
# the seed closest in (eta, phi) within SEED_MATCH_MAX_DR.
SEED_MATCH_MIN_HITS = 3
SEED_MATCH_MAX_DR = 0.05
OCC_Z_EDGES = np.linspace(-2300, 2300, 461)   # occupancy map, 10 mm bins
OCC_R_EDGES = np.linspace(0, 1600, 161)

# [DIAG] printout: per-subdetector hit collections ("Coned" appended when coned).
DIAG_COLLECTIONS = ("VXDBarrelHits", "VXDEndcapHits", "ITBarrelHits",
                    "ITEndcapHits", "OTBarrelHits", "OTEndcapHits")

# No-hits mode (hit collections dropped from the reco output), detected per
# file from its first event; NO_HITS env var ("1"/"0") forces it.
NO_HITS_REQUIRED_COLLECTIONS = ("MergedTrackerHits", "MergedTrackerHitsRelations")
_env_no_hits = os.environ.get("NO_HITS")
NO_HITS = False
NHITS_SOURCE = "trackerHits"


def is_signal_pdg(pdg):
    """True when `pdg` is the signal species, either charge sign."""
    return abs(int(pdg)) == SIGNAL_PDG


def cell_layer(cell_id):
    """Layer field of a tracker cellID ("system:5,side:-2,layer:6,module:11,sensor:8"),
    bits 7-12. Works on ints and numpy arrays."""
    return (cell_id >> 7) & 0x3F


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


# --- C++ helpers for the whole-collection loops -----------------------------
# A getter called from Python costs ~1 us, and a 100% BIB event has millions of
# hits and hit links, so the loops over entire collections run in C++ and hand
# back numpy arrays. Templated on the collection type; _cpp() instantiates them
# for the C++ class of the collection passed in.
_CPP_HELPERS_OK = ROOT.gInterpreter.Declare(r"""
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include "podio/ObjectID.h"

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

// Per track: hit count (0 without reading the relation when countHits is
// false), chi2, ndf, number of track states, and the parameters of state
// stateIdx (NaN when the track has no such state).
template <class TrackColl>
void fillTracks(const TrackColl& tracks, int stateIdx, bool countHits,
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
    if (stateIdx >= 0 && states.size() > static_cast<std::size_t>(stateIdx)) {
      const auto& ts = states[stateIdx];
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


def track_arrays(tracks, state_idx):
    """Per-track arrays from ana::fillTracks. `nhits` and `has_hits` follow
    track_nhits() / track_has_hits(); state_idx = -1 reads no track state."""
    n = len(tracks)
    nhits, ndf, nstates = (np.zeros(n, dtype=np.int64) for _ in range(3))
    chi2, omega, tan_lambda, phi, d0, z0 = (np.empty(n) for _ in range(6))
    if n:
        _cpp("fillTracks", tracks)(tracks, state_idx, not NO_HITS or NHITS_SOURCE == "trackerHits",
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


def coll_size(event, name):
    """Size of collection `name` in `event`, -1 if it is not there."""
    try:
        coll = event.get(name)
    except KeyError:
        return -1
    return -1 if coll is None else len(coll)


# ---- output trees ----------------------------------------------------------
# No 64-bit integer leaves: PyROOT (6.38) will not bind a numpy int64 buffer
# ('l') to Long64_t* ('q'). Large counts go in doubles (exact to 2^53).
_LEAF_CODE = {np.dtype(np.int32): "I", np.dtype(np.float32): "F",
              np.dtype(np.float64): "D", np.dtype(np.bool_): "O"}


class TreeWriter:
    """A TTree whose branches read from numpy buffers: scalars, fixed-size
    arrays, and variable-length arrays that share a count branch. The arrays
    of a count branch are handed over per entry with set_group()."""

    def __init__(self, name, title, directory):
        directory.cd()
        self.tree = ROOT.TTree(name, title)
        self.tree.SetDirectory(directory)
        self.buf = {}
        self.groups = {}   # count branch -> [(array branch, dtype)]

    def _branch(self, name, buf, leaf):
        self.buf[name] = buf
        self.tree.Branch(name, buf, leaf)

    def scalar(self, name, dtype):
        b = np.zeros(1, dtype=dtype)
        self._branch(name, b, f"{name}/{_LEAF_CODE[b.dtype]}")

    def fixed(self, name, n, dtype):
        b = np.zeros(n, dtype=dtype)
        self._branch(name, b, f"{name}[{n}]/{_LEAF_CODE[b.dtype]}")

    def arrays(self, count, columns):
        self.scalar(count, np.int32)
        self.groups[count] = [(name, np.dtype(dt)) for name, dt in columns]
        for name, dt in self.groups[count]:
            self._branch(name, np.zeros(1, dtype=dt), f"{name}[{count}]/{_LEAF_CODE[dt]}")

    def set(self, name, value):
        self.buf[name][0] = value

    def set_group(self, count, **columns):
        n = None
        for name, dt in self.groups[count]:
            a = np.ascontiguousarray(columns[name], dtype=dt)
            if n is None:
                n = len(a)
            if len(a) != n:
                raise ValueError(f"{name} has {len(a)} entries, {count} = {n}")
            if n == 0:
                a = np.zeros(1, dtype=dt)   # never hand ROOT an empty buffer
            self.buf[name] = a
            self.tree.SetBranchAddress(name, a)
        self.buf[count][0] = n

    def fill(self):
        self.tree.Fill()


MU_COLS = ("mu_eta", "mu_pt", "mu_phi", "mu_theta_deg",
           "mu_matched_si", "mu_matched_all", "mu_n_si", "mu_n_all")
SI_FLOAT_COLS = ("si_chi2red", "si_eta", "si_truth_pt", "si_reco_pt", "si_pt_res", "si_phi",
                 "si_d0", "si_z0", "si_d0_res", "si_z0_res", "si_dqop2", "si_mufrac")
SI_INT_COLS = ("si_nbib", "si_npix", "si_ninner", "si_nouter", "si_ntotal")
MAT_INT_COLS = ("mat_npix", "mat_ninner", "mat_nouter", "mat_ntotal")
RAW_COLS = ("raw_nhits", "raw_chi2red", "raw_pt", "raw_d0", "raw_z0")
DH_COLS = ("dh_kind", "dh_x", "dh_y", "dh_z", "dh_r")


def ntuple_file(in_path, out_path, use_coned):
    """Event loop over one reco file; writes the events and summary trees."""
    global NO_HITS, NHITS_SOURCE
    t_start = time.time()
    rng = np.random.default_rng(zlib.crc32(os.path.basename(in_path).encode()))   # reproducible ID-diag sample
    acc = dict(id_diag_counts=defaultdict(int), id_diag_seen_per_sys=defaultdict(int), id_diag_sample={})
    occupancy = np.zeros((len(OCC_Z_EDGES) - 1, len(OCC_R_EDGES) - 1), dtype=np.int64)
    zseam_n = 0

    reader = get_reader(in_path)
    out = ROOT.TFile(out_path, "RECREATE")
    f8, f4, i4, b1 = np.float64, np.float32, np.int32, np.bool_
    ev = TreeWriter("events", "track analysis ntuple, one entry per event", out)
    for name in ("no_hits", "detected_no_hits", "nhits_unavailable", "use_coned", "has_muon"):
        ev.scalar(name, b1)
    for name in ("file_event", "n_alltracks", "n_deduped", "n_sitracks", "n_seeds",
                 "muon_n_si_links", "seed_match", "disp_kind"):
        ev.scalar(name, i4)
    for name in ("seed_truth_pt", "seed_match_pt", "disp_chi2_mean", "disp_truth_pt", "disp_reco_pt"):
        ev.scalar(name, f8)
    ev.fixed("n_diag", len(DIAG_COLLECTIONS) + 1, i4)
    ev.arrays("n_lay", [("lay_code", i4), ("lay_nhits", i4)])
    ev.arrays("n_mu", [(c, f8) for c in MU_COLS[:4]] + [(c, i4) for c in MU_COLS[4:]])
    ev.arrays("n_si", [(c, f8) for c in SI_FLOAT_COLS] + [(c, i4) for c in SI_INT_COLS])
    ev.arrays("n_sihit", [("sihit_code", i4)])
    ev.arrays("n_mat", [("mat_chi2red", f8)] + [(c, i4) for c in MAT_INT_COLS])
    ev.arrays("n_raw", [("raw_nhits", i4)] + [(c, f4) for c in RAW_COLS[1:]])
    ev.arrays("n_sd", [("sd_pt", f4), ("sd_eta", f4)])
    ev.arrays("n_zs", [("zs_z", f4), ("zs_r", f4), ("zs_sys", i4)])
    ev.arrays("n_dh", [("dh_kind", i4)] + [(c, f8) for c in DH_COLS[1:]])

    detected = False
    i = 0
    for event in reader.get("events"):
        if i == 0:
            # Dropped-hit-collection autodetect (see edm4hepTrackAnalyzer.py's
            # no-hits notes). The env var only forces the mode; the hit-count
            # source is always measured.
            detected, NHITS_SOURCE = _detect_mode(event)
            NO_HITS = detected if _env_no_hits is None else _env_no_hits not in ("0", "false", "False", "")

        rz_muon_r = []
        rz_muon_z = []
        rz_bib_r = []
        rz_bib_z = []
        xy_muon_x = []
        xy_muon_y = []
        xy_bib_x = []
        xy_bib_y = []
        hits_in_nonrefit_muon_track = set()
        chi2_nonrefit_list = []

        hitRelations = None if NO_HITS else event.get("MergedTrackerHitsRelations")
        hitCollection = None if NO_HITS else event.get("MergedTrackerHits")

        # Hit-collection sizes for the [DIAG] printout: -1 = missing,
        # -2 = not looked at (no-hits mode).
        n_diag = [-2] * (len(DIAG_COLLECTIONS) + 1)
        if not NO_HITS:
            suffix = "Coned" if use_coned else ""
            n_diag = [coll_size(event, name + suffix) for name in DIAG_COLLECTIONS]
            n_diag.append(len(hitCollection))

        # Position, cellID and obj_key of every merged tracker hit, read in C++
        # (ana::fillHits). Everything that looks at all hits uses these arrays.
        _keys = _cids = None
        lay_code = lay_nhits = ()
        zs = ((), (), ())
        if hitCollection is not None and len(hitCollection) > 0:
            _xs, _ys, _zs, _cids, _keys = hit_arrays(hitCollection)
            _rs = np.hypot(_xs, _ys)

            # Hits per (system, layer), as system * 64 + layer.
            _sl = np.bincount((_cids & 0x1F) * 64 + cell_layer(_cids))
            lay_code = np.nonzero(_sl)[0]
            lay_nhits = _sl[lay_code]

            # Tracker r-z occupancy: bin every merged tracker hit into the file's histogram.
            _h2d, _, _ = np.histogram2d(_zs, _rs, bins=[OCC_Z_EDGES, OCC_R_EDGES])
            occupancy += _h2d.astype(np.int64)

            # z = 0 seam diagnostic: every hit near z = 0, unsampled, with its subdetector.
            if zseam_n < ZSEAM_MAX_HITS:
                _near = np.abs(_zs) < ZSEAM_COLLECT_HALF
                zs = (_zs[_near], _rs[_near], _cids[_near] & 0x1F)
                zseam_n += int(_near.sum())

            # ID verification: decode cellID for every merged tracker hit and accumulate
            if GENERATE_ID_DIAG:
                _id_diag_add(acc, rng, _rs, _zs, _cids)

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

        _at = track_arrays(allTracks, 1)
        n_alltracks = int(_at["has_hits"].sum())
        n_deduped = int(track_arrays(dedupedTracks, -1)["has_hits"].sum())
        n_sitracks = int(track_arrays(siTracks, -1)["has_hits"].sum())

        # Every AllTracks track with hits, no MC matching; NaN where the
        # quantity is undefined (ndf <= 0, no state at the IP, omega = 0).
        _sel = _at["has_hits"]
        _ndf = _at["ndf"][_sel]
        _state = _at["nstates"][_sel] > 1
        _om = _at["omega"][_sel]
        _pt_ok = _state & (_om != 0)
        raw = dict(
            raw_nhits=_at["nhits"][_sel],
            raw_chi2red=np.where(_ndf > 0, _at["chi2"][_sel] / np.maximum(_ndf, 1), np.nan),
            raw_pt=np.where(_pt_ok, 0.2998 * BField / np.abs(np.where(_pt_ok, _om, 1.0) * 1000.), np.nan),
            raw_d0=np.where(_state, _at["d0"][_sel], np.nan),
            raw_z0=np.where(_state, _at["z0"][_sel], np.nan),
        )

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

        mu = defaultdict(list)    # per signal muon
        si = defaultdict(list)    # per muon-matched SiTrack with hits
        mat = defaultdict(list)   # per muon-matched AllTracks track
        sihit_code = []           # system * 64 + layer of every hit on a muon-matched SiTrack
        mcpid = None
        mcp_tlv = None
        NonRefittedpT = None
        for _pos in signal_particles(mcParticles):
            mcp = mcParticles[_pos]
            mcp_p = mcp.getMomentum()
            mcp_tlv = ROOT.TLorentzVector()
            mcp_tlv.SetPxPyPzE(mcp_p[0], mcp_p[1], mcp_p[2], mcp.getEnergy())
            mcpid = mcp.getObjectID().index
            # Compute truth helix d0/z0 only if this MCP has at least one
            # matched track in any selection stage — otherwise it's a wasted
            # call (the residual containers won't be appended either way).
            d0_truth = z0_truth = None
            if mcp_to_SiTracks.get(mcpid):
                d0_truth, z0_truth = truth_d0_z0_helix(mcp, Bz_T=BField)
            hits_in_nonrefit_muon_track = set()
            rz_muon_r = []
            rz_muon_z = []
            rz_bib_r = []
            rz_bib_z = []
            xy_muon_x = []
            xy_muon_y = []
            xy_bib_x = []
            xy_bib_y = []
            mu["mu_eta"].append(mcp_tlv.Eta())
            mu["mu_pt"].append(mcp_tlv.Perp())
            mu["mu_phi"].append(mcp_tlv.Phi())
            mu["mu_theta_deg"].append(np.degrees(mcp_tlv.Theta()))
            mu["mu_matched_si"].append(any(track_has_hits(t) for t, _ in mcp_to_SiTracks.get(mcpid, [])))
            mu["mu_matched_all"].append(any(track_has_hits(t) for t, _ in mcp_to_AllTracks.get(mcpid, [])))
            mu["mu_n_si"].append(len(mcp_to_SiTracks.get(mcpid, [])))
            mu["mu_n_all"].append(len(mcp_to_AllTracks.get(mcpid, [])))

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
                print(f"[eff-debug] {os.path.basename(in_path)} evt={i} mcpid={mcpid} "
                      f"nMuMCPs={len(signal_particles(mcParticles))} "
                      f"nSeeds={len(seedTracks)} nAllTracks={len(allTracks)} nLinksMu(All)={n_links_mu_all} "
                      f"nLinksMu(Si)={n_links_mu_si} nHits={n_hits} nHitRel={n_hit_rels} [{_sys_str}]")

                # The composition dump needs the hit -> SimTrackerHit map, so it
                # is only possible when the hit relations were written out.
                if n_links_mu_all == 0 and not NO_HITS:
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

            # --- AllTracks (muon-matched): chi2/Ndf and Pixel/Inner/Outer/Total hits.
            # The subsystem split needs a cellID per hit, so in no-hits mode only
            # the total is available. -1 = not filled.
            for allTrack, _ in mcp_to_AllTracks.get(mcpid, []):
                _at_n = track_nhits(allTrack)
                mat["mat_chi2red"].append(allTrack.getChi2() / float(allTrack.getNdf())
                                          if track_has_hits(allTrack) and allTrack.getNdf() > 0 else np.nan)
                _at_pixel = _at_inner = _at_outer = _at_total = -1
                if NO_HITS:
                    if _at_n > 0:
                        _at_total = _at_n
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
                    _at_total = _at_pixel + _at_inner + _at_outer
                for c, v in zip(MAT_INT_COLS, (_at_pixel, _at_inner, _at_outer, _at_total)):
                    mat[c].append(v)

            # --- Non-refitted tracks ---
            for siTrack, w in mcp_to_SiTracks.get(mcpid, []):
                hit_matched_counter = 0   # must reset per track; n_hits_nr below is per-track
                ts = siTrack.getTrackStates()[1]   # at IP state after back propagation
                omega = ts.omega
                NonRefittedpT = 0.2998 * BField / fabs(omega * 1000.)
                chi2_reduced_nr = 0
                if(siTrack.getNdf() > 0):
                    chi2_reduced_nr = siTrack.getChi2() / float(siTrack.getNdf())
                chi2_nonrefit_list.append(chi2_reduced_nr)
                theta_nr = np.pi/2 - np.arctan(ts.tanLambda)
                eta_nr = -np.log(np.tan(theta_nr/2))
                phi_nr = ts.phi
                d0_nr = ts.D0
                z0_nr = ts.Z0
                LC_pixel_nhit_nr = 0
                LC_inner_nhit_nr = 0
                LC_outer_nhit_nr = 0
                # Everything in this loop reads hit positions and cellIDs,
                # so it only runs when the hit collection was written out.
                for hit in ([] if NO_HITS else siTrack.getTrackerHits()):
                    position = hit.getPosition()
                    r = sqrt(position[0] ** 2 + position[1] ** 2)
                    hkey = obj_key(hit)
                    hits_in_nonrefit_muon_track.add(hkey)
                    # decode system and layer from cellID: "system:5,side:-2,layer:6,module:11,sensor:8"
                    cell_id = int(hit.getCellID())
                    system = cell_id & 0x1F
                    layer_nr = cell_layer(cell_id)
                    if 1 <= system <= 6:
                        sihit_code.append(system * 64 + layer_nr)
                    if system == 1 or system == 2:
                        LC_pixel_nhit_nr += 1
                    elif system == 3 or system == 4:
                        LC_inner_nhit_nr += 1
                    elif system == 5 or system == 6:
                        LC_outer_nhit_nr += 1
                    if hkey in signal_keys:
                        rz_muon_r.append(r)
                        rz_muon_z.append(position[2])
                        xy_muon_x.append(position[0])
                        xy_muon_y.append(position[1])
                        hit_matched_counter += 1
                    else:
                        rz_bib_r.append(r)
                        rz_bib_z.append(position[2])
                        xy_bib_x.append(position[0])
                        xy_bib_y.append(position[1])
                n_hits_nr = track_nhits(siTrack)
                if track_has_hits(siTrack):
                    p_magnitude = sqrt(mcp_p[0]**2 + mcp_p[1]**2 + mcp_p[2]**2)
                    charge = mcp.getCharge()
                    truthqoverp = float(charge) / p_magnitude
                    qoverp_nr = (ts.omega * sin(theta_nr) * 1000.) / (0.2998 * BField)
                    # Muon/BIB split and the subsystem breakdown both need the
                    # hits themselves; only the total survives without them.
                    values = dict(
                        si_chi2red=chi2_reduced_nr, si_eta=eta_nr, si_truth_pt=mcp_tlv.Perp(),
                        si_reco_pt=NonRefittedpT,
                        si_pt_res=(NonRefittedpT - mcp_tlv.Perp()) / mcp_tlv.Perp(),
                        si_phi=phi_nr, si_d0=d0_nr, si_z0=z0_nr,
                        si_d0_res=d0_nr - d0_truth if d0_truth is not None else np.nan,
                        si_z0_res=z0_nr - z0_truth if z0_truth is not None else np.nan,
                        si_dqop2=(qoverp_nr - truthqoverp)**2,
                        si_mufrac=np.nan if NO_HITS else float(hit_matched_counter) / n_hits_nr,
                        si_nbib=-1 if NO_HITS else n_hits_nr - hit_matched_counter,
                        si_npix=-1 if NO_HITS else LC_pixel_nhit_nr,
                        si_ninner=-1 if NO_HITS else LC_inner_nhit_nr,
                        si_nouter=-1 if NO_HITS else LC_outer_nhit_nr,
                        si_ntotal=n_hits_nr if NO_HITS else LC_pixel_nhit_nr + LC_inner_nhit_nr + LC_outer_nhit_nr,
                    )
                    for c, v in values.items():
                        si[c].append(v)

        # Event hit maps (drawn by the plotter): bit 1 = the per-event r-z/x-y
        # maps (exactly one matched SiTrack, with a non-muon hit on it), bit 2 =
        # a candidate for the "first 10 events" maps (any matched SiTrack).
        n_si_matched = len(mcp_to_SiTracks.get(mcpid, [])) if mcpid is not None else 0
        disp_kind = 0
        if not NO_HITS and n_si_matched == 1 and len(hits_in_nonrefit_muon_track) > 0 and len(rz_bib_r) > 0:
            disp_kind |= 1
        if not NO_HITS and n_si_matched >= 1:
            disp_kind |= 2
        dh = {c: [] for c in DH_COLS}   # kind 0 = muon hit on the track, 1 = other hit on it, 2 = missed muon hit
        if disp_kind:
            # Muon hits the track did not pick up.
            _missed = np.zeros(0, dtype=bool)
            if _keys is not None and signal_keys:
                _missed = np.isin(_keys, np.fromiter(signal_keys, dtype=np.int64, count=len(signal_keys)))
                _missed &= ~np.isin(_keys, np.fromiter(hits_in_nonrefit_muon_track, dtype=np.int64,
                                                       count=len(hits_in_nonrefit_muon_track)))
            points = [(0, xy_muon_x, xy_muon_y, rz_muon_z, rz_muon_r),
                      (1, xy_bib_x, xy_bib_y, rz_bib_z, rz_bib_r)]
            if _missed.any():
                points.append((2, _xs[_missed], _ys[_missed], _zs[_missed], _rs[_missed]))
            for kind, px, py, pz, pr in points:
                dh["dh_kind"].extend([kind] * len(px))
                dh["dh_x"].extend(px)
                dh["dh_y"].extend(py)
                dh["dh_z"].extend(pz)
                dh["dh_r"].extend(pr)

        # Seed track pT and eta of every seed
        _sd = track_arrays(seedTracks, 0)
        _ok = (_sd["nstates"] > 0) & (_sd["omega"] != 0)
        theta_s = np.pi/2 - np.arctan(_sd["tanLambda"][_ok])
        eta_s = -np.log(np.tan(theta_s / 2))
        phi_s = _sd["phi"][_ok]
        pt_s = 0.2998 * BField / np.abs(_sd["omega"][_ok] * 1000.)
        # The muon's seed: one with >= SEED_MATCH_MIN_HITS muon hits (of several,
        # the one with the most, then the closest to the truth direction); if
        # there is none, the closest seed within SEED_MATCH_MAX_DR.
        seed_match, seed_pt = 0, np.nan
        if mcp_tlv is not None and len(pt_s):
            dphi = (phi_s - mcp_tlv.Phi() + np.pi) % (2 * np.pi) - np.pi
            dr = np.hypot(eta_s - mcp_tlv.Eta(), dphi)
            dr[np.isnan(dr)] = np.inf
            n_muon_hits = keyed_hit_counts(seedTracks, signal_keys)[_ok]
            cand = np.nonzero(n_muon_hits >= SEED_MATCH_MIN_HITS)[0]
            if len(cand):
                seed_match, seed_pt = 1, float(pt_s[cand[np.lexsort((dr[cand], -n_muon_hits[cand]))[0]]])
            elif dr.min() < SEED_MATCH_MAX_DR:
                seed_match, seed_pt = 2, float(pt_s[int(np.argmin(dr))])

        ev.set("file_event", i)
        ev.set("no_hits", NO_HITS)
        ev.set("detected_no_hits", detected)
        ev.set("nhits_unavailable", NHITS_SOURCE == "unavailable")
        ev.set("use_coned", use_coned)
        ev.set("has_muon", mcpid is not None)
        ev.set("n_alltracks", n_alltracks)
        ev.set("n_deduped", n_deduped)
        ev.set("n_sitracks", n_sitracks)
        ev.set("n_seeds", len(pt_s))
        ev.set("muon_n_si_links", n_si_matched)
        ev.set("seed_match", seed_match)
        ev.set("seed_truth_pt", mcp_tlv.Perp() if seed_match else np.nan)
        ev.set("seed_match_pt", seed_pt)
        ev.set("disp_kind", disp_kind)
        ev.set("disp_chi2_mean", np.mean(chi2_nonrefit_list) if chi2_nonrefit_list else np.nan)
        ev.set("disp_truth_pt", mcp_tlv.Perp() if mcp_tlv is not None else np.nan)
        ev.set("disp_reco_pt", NonRefittedpT if NonRefittedpT is not None else np.nan)
        ev.buf["n_diag"][:] = n_diag
        ev.set_group("n_lay", lay_code=lay_code, lay_nhits=lay_nhits)
        ev.set_group("n_mu", **{c: mu[c] for c in MU_COLS})
        ev.set_group("n_si", **{c: si[c] for c in SI_FLOAT_COLS + SI_INT_COLS})
        ev.set_group("n_sihit", sihit_code=sihit_code)
        ev.set_group("n_mat", **{c: mat[c] for c in ("mat_chi2red",) + MAT_INT_COLS})
        ev.set_group("n_raw", **raw)
        ev.set_group("n_sd", sd_pt=pt_s, sd_eta=eta_s)
        ev.set_group("n_zs", zs_z=zs[0], zs_r=zs[1], zs_sys=zs[2])
        ev.set_group("n_dh", **dh)
        ev.fill()

        i += 1
        # Free this event and its per-hit arrays before the next event is read.
        del event
        _xs = _ys = _zs = _rs = _cids = _keys = _at = _sd = None

    # ---- per-file summary ---------------------------------------------------
    sm = TreeWriter("summary", "track analysis ntuple, one entry per file", out)
    for name in ("n_events", "signal_pdg", "seed_min_hits", "zseam_max_hits"):
        sm.scalar(name, i4)
    for name in ("bfield", "seed_max_dr", "zseam_collect_half"):
        sm.scalar(name, f8)
    sm.fixed("occ", occupancy.size, i4)   # [z bin][r bin], OCC_Z_EDGES x OCC_R_EDGES
    sm.arrays("n_id", [("id_code", i4), ("id_count", f8)])
    sm.arrays("n_samp", [("samp_sys", i4), ("samp_side", i4), ("samp_layer", i4),
                         ("samp_r", f4), ("samp_z", f4), ("samp_u", f8)])
    sm.set("n_events", i)
    sm.set("signal_pdg", SIGNAL_PDG)
    sm.set("seed_min_hits", SEED_MATCH_MIN_HITS)
    sm.set("zseam_max_hits", ZSEAM_MAX_HITS)
    sm.set("bfield", BField)
    sm.set("seed_max_dr", SEED_MATCH_MAX_DR)
    sm.set("zseam_collect_half", ZSEAM_COLLECT_HALF)
    sm.buf["occ"][:] = occupancy.ravel()
    # cellID (system, side, layer) counts, as system | side << 5 | layer << 7.
    keys = sorted(acc["id_diag_counts"])
    sm.set_group("n_id", id_code=[s | (side << 5) | (layer << 7) for s, side, layer in keys],
                 id_count=[acc["id_diag_counts"][k] for k in keys])
    # Each system's random hit sample (u is the sampling key, see _bottom_k).
    samp = {c: [] for c in ("samp_sys", "samp_side", "samp_layer", "samp_r", "samp_z", "samp_u")}
    for s, (u, r, z, side, layer) in sorted(acc["id_diag_sample"].items()):
        samp["samp_sys"].extend([s] * len(u))
        samp["samp_side"].extend(side)
        samp["samp_layer"].extend(layer)
        samp["samp_r"].extend(r)
        samp["samp_z"].extend(z)
        samp["samp_u"].extend(u)
    sm.set_group("n_samp", **samp)
    sm.fill()

    out.cd()
    out.Write("", ROOT.TObject.kOverwrite)
    out.Close()
    print(f">>> {in_path}: {i} events -> {out_path} in {time.time() - t_start:.0f} s "
          f"(no_hits={NO_HITS}, coned={use_coned})")


def main():
    p = argparse.ArgumentParser(description="Write the track-analysis ntuple of one reco file.")
    p.add_argument("--input", required=True, help="reco .edm4hep.root file")
    p.add_argument("--output", required=True, help="ntuple .root file to write")
    p.add_argument("--coned", type=int, choices=(0, 1), default=None,
                   help="1 if the reco used the coned hit collections (default: 'coned' in the input path)")
    args = p.parse_args()
    use_coned = bool(args.coned) if args.coned is not None else "coned" in args.input.lower()
    ntuple_file(args.input, args.output, use_coned)


if __name__ == "__main__":
    main()
