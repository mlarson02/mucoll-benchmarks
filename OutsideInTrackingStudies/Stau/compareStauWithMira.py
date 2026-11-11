#!/usr/bin/env python3
"""SiTracks efficiency vs eta: ours (edm4hepStauTrackAnalyzer.py's
stau_summary.pkl) overlaid with Mira's MuColl_v1 reco (LCIO), with the same stau
selection and acceptance as the analyzer.

  extract  (needs pyLCIO): read Mira's .slcio reco files, save per-stau results
      python3 compareStauWithMira.py extract \\
          --input-dir /ospool/.../reder_timing/loose4/seeding_10GeV/bib/4000_30 \\
          --output mira_4000_30_loose_bib.pkl [--workers 8] [--max-files N]

  plot     (needs matplotlib)
      python3 compareStauWithMira.py plot \\
          --ours "Ours, 4D beta=0.6, MAIA, full BIB=<plot dir>/stau_summary.pkl" \\
          --mira "Mira, MuColl_v1 (3.57 T), loose, 10% BIB=mira_4000_30_loose_bib.pkl"

A stau is found when its truth-linked SiTrack (MCParticle_SiTracks; one link per
particle, weight = the track's hit fraction from it) has purity > PURITY_MIN and
passes the track selection: our FilterTracksAlg cuts (SITRACK_*), which Mira's
SiTracks never had, and the strictest tier's per-subdetector hits (TIER_*). Ours
already passed the filter in reco; its tier flag also has chi2/ndf <= 3. Mira's
efficiency is shown with and without the purity cut.
"""
import argparse
import os
import pickle
import sys
from math import asinh, hypot, sqrt

# ---- same selection as edm4hepStauTrackAnalyzer.py ---------------------------
STAU_PDGS = (1000015, 2000015)
ORIGIN_TOL = 1e-3          # mm: staus produced at the origin
STAUS_PER_EVENT = 2        # longest decay length first
ACC_ETA_MAX = 0.8
TRACKER_R = 1500.          # mm: in the acceptance when decaying outside r / |z|
TRACKER_Z = 2200.
PURITY_MIN = 0.5
# Workers are replaced after this many files: pyLCIO keeps memory from every
# event read (~1.5M hits each), which reached 7 GB per worker otherwise.
FILES_PER_WORKER = 10

# Track selection, applied to Mira's tracks at plot time:
# our SiTracks filter (Tracking/CKF_tracking.py track_filter_cfg, FilterTracksAlg)
SITRACK_MIN_HITS = 9       # NHitsTotal = 8, kept when nHits > 8
SITRACK_MAX_D0 = 10.       # mm
SITRACK_MAX_Z0 = 10.       # mm
SITRACK_MIN_PT = 1.0       # GeV
SITRACK_MAX_HOLES = 2
# and the VB+IB+OB tier: VXD layers (a doublet hit counts 1/2; in MuColl_v1 every
# VXD layer is a doublet), IT hits, OT hits
TIER_MIN_VXD, TIER_MIN_IT, TIER_MIN_OT = 3, 2, 2
MUCOLL_V1_BZ = 3.57        # T, for Mira's pT
TRACK_KEYS = ("n_hits", "vxd_layers", "n_it", "n_ot", "d0", "z0", "trk_pt", "n_holes")
PART_VERSION = 2           # per-file results of an older extract are read again


# ---- extract -----------------------------------------------------------------
def extract_file(path):
    """eta, pT, in-acceptance and SiTrack link purity (NaN: no link) of the
    selected staus in every event of one .slcio file."""
    from pyLCIO import IOIMPL, UTIL
    out = {k: [] for k in ("eta", "pt", "in_acc", "purity") + TRACK_KEYS}
    n_ev = 0
    reader = IOIMPL.LCFactory.getInstance().createLCReader()
    reader.open(path)
    for event in reader:
        n_ev += 1
        names = list(event.getCollectionNames())
        if "MCParticle" not in names:
            continue
        nav = (UTIL.LCRelationNavigator(event.getCollection("MCParticle_SiTracks"))
               if "MCParticle_SiTracks" in names else None)
        cands = []
        for p in event.getCollection("MCParticle"):
            if abs(p.getPDG()) not in STAU_PDGS:
                continue
            v, e = p.getVertex(), p.getEndpoint()
            length = sqrt(sum((e[k] - v[k]) ** 2 for k in range(3)))
            if length == 0 or any(abs(v[k]) >= ORIGIN_TOL for k in range(3)):
                continue
            cands.append((length, p))
        cands.sort(key=lambda c: -c[0])
        for _, p in cands[:STAUS_PER_EVENT]:
            m, e = p.getMomentum(), p.getEndpoint()
            pt = hypot(m[0], m[1])
            eta = asinh(m[2] / pt) if pt > 0 else float("nan")
            links = (list(zip(nav.getRelatedToObjects(p), nav.getRelatedToWeights(p)))
                     if nav is not None else [])
            out["eta"].append(eta)
            out["pt"].append(pt)
            out["in_acc"].append(abs(eta) < ACC_ETA_MAX
                                 and (hypot(e[0], e[1]) > TRACKER_R or abs(e[2]) > TRACKER_Z))
            trk, w = max(links, key=lambda tw: tw[1]) if links else (None, float("nan"))
            out["purity"].append(float(w))
            for k, v in zip(TRACK_KEYS, _track_quantities(trk)):
                out[k].append(v)
    reader.close()
    return path, n_ev, out


def _track_quantities(trk):
    """TRACK_KEYS of an LCIO track, all NaN without one."""
    if trk is None:
        return (float("nan"),) * len(TRACK_KEYS)
    hits = trk.getTrackerHits()
    vxd = n_it = n_ot = 0.0
    for h in hits:
        system = int(h.getCellID0()) & 0x1F
        if system in (1, 2):
            vxd += 0.5          # MuColl_v1: every VXD layer is a doublet
        elif system in (3, 4):
            n_it += 1
        elif system in (5, 6):
            n_ot += 1
    omega = trk.getOmega()
    pt = abs(0.3 * MUCOLL_V1_BZ / omega / 1000.) if omega != 0 else float("nan")
    return (float(len(hits)), vxd, n_it, n_ot, float(trk.getD0()), float(trk.getZ0()), pt,
            float(trk.getNholes()))


def _safe_extract(path):
    try:
        return extract_file(path)
    except Exception as err:   # e.g. a truncated file
        return path, str(err), None


def extract(a):
    """Each file's result is saved under <output>.parts/ as it finishes, so a
    rerun skips the files already done; then all parts are merged."""
    import glob
    import multiprocessing
    files = sorted(glob.glob(os.path.join(a.input_dir, "*.slcio")))
    if a.max_files:
        files = files[:a.max_files]
    parts = a.output + ".parts"
    os.makedirs(parts, exist_ok=True)
    part_of = {f: os.path.join(parts, os.path.basename(f) + ".pkl") for f in files}
    def _part_ok(f):
        try:
            with open(part_of[f], "rb") as fh:
                return len(pickle.load(fh)) == 3   # (n, out, PART_VERSION)
        except Exception:
            return False
    todo = [f for f in files if not _part_ok(f)]
    print(f"{len(files)} files in {a.input_dir}: {len(files) - len(todo)} already done, {len(todo)} to read "
          f"with {a.workers} workers", flush=True)
    n_bad = 0
    with multiprocessing.get_context("spawn").Pool(a.workers, maxtasksperchild=FILES_PER_WORKER) as pool:
        for i, (path, n, out) in enumerate(pool.imap_unordered(_safe_extract, todo), 1):
            if out is None:
                n_bad += 1
                print(f"WARNING: skipping {path}: {n}", flush=True)
                continue
            with open(part_of[path] + ".tmp", "wb") as f:
                pickle.dump((n, out, PART_VERSION), f, protocol=4)
            os.replace(part_of[path] + ".tmp", part_of[path])
            if i % 50 == 0 or i == len(todo):
                print(f"[{i}/{len(todo)}] files read this run", flush=True)
    res = {k: [] for k in ("eta", "pt", "in_acc", "purity") + TRACK_KEYS}
    n_ev = n_done = 0
    for f in files:
        if not os.path.exists(part_of[f]):
            continue
        with open(part_of[f], "rb") as fh:
            part = pickle.load(fh)
        if len(part) != 3:
            continue
        n, out, _ = part
        n_ev += n
        n_done += 1
        for k in res:
            res[k].extend(out[k])
    res.update(n_events=n_ev, n_files=n_done, input_dir=a.input_dir)
    with open(a.output, "wb") as f:
        pickle.dump(res, f, protocol=4)
    print(f">>> {a.output}: {n_done}/{len(files)} files, {n_ev} events, {len(res['eta'])} selected staus, "
          f"{sum(res['in_acc'])} in acceptance ({n_bad} files skipped this run)")


# ---- plot --------------------------------------------------------------------
def _decode(o):
    """Inverse of the analyzer's _acc_encode (numpy arrays stored as bytes)."""
    import numpy as np
    if isinstance(o, dict):
        if len(o) == 1 and "__nd__" in o:
            dtype, shape, buf = o["__nd__"]
            return np.frombuffer(buf, dtype=np.dtype(dtype)).reshape(shape).copy()
        return {k: _decode(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(_decode(v) for v in o)
    return o


def _label_path(s):
    """'label=path' (the label may contain '='), or a bare path."""
    label, sep, path = s.rpartition("=")
    return (label, path) if sep else (os.path.basename(os.path.dirname(os.path.abspath(s))), s)


def plot(a):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = []   # (label, eta of the staus in the acceptance, found flags, line style)
    for s in a.ours:
        label, path = _label_path(s)
        with open(path, "rb") as f:
            d = _decode(pickle.load(f))
        # SiTracks passed the filter in reco; add the VB+IB+OB tier.
        found = np.asarray(d["eff_matched_si"], bool) & np.asarray(d["eff_tier_ob"], bool)
        series.append((f"{label} (purity > {PURITY_MIN:g})", np.asarray(d["eff_truth_eta"], float), found, "-"))
    for s in a.mira:
        label, path = _label_path(s)
        with open(path, "rb") as f:
            d = pickle.load(f)
        if not all(k in d for k in TRACK_KEYS):
            sys.exit(f"{path} has no track quantities: rerun extract (it re-reads every file)")
        acc = np.asarray(d["in_acc"], bool)
        q = {k: np.asarray(d[k], float)[acc] for k in ("eta", "purity") + TRACK_KEYS}
        with np.errstate(invalid="ignore"):
            selected = ((q["n_hits"] >= SITRACK_MIN_HITS) & (np.abs(q["d0"]) <= SITRACK_MAX_D0)
                        & (np.abs(q["z0"]) <= SITRACK_MAX_Z0) & (q["trk_pt"] >= SITRACK_MIN_PT)
                        & (q["n_holes"] <= SITRACK_MAX_HOLES) & (q["vxd_layers"] >= TIER_MIN_VXD)
                        & (q["n_it"] >= TIER_MIN_IT) & (q["n_ot"] >= TIER_MIN_OT))
            series.append((f"{label} (purity > {PURITY_MIN:g})", q["eta"], selected & (q["purity"] > PURITY_MIN), "-"))
        series.append((f"{label} (no purity cut)", q["eta"], selected & np.isfinite(q["purity"]), ":"))
        linked = np.isfinite(q["purity"])
        print(f"{label}: {d['n_events']} events, {len(d['eta'])} selected staus, {int(acc.sum())} in acceptance, "
              f"{int(linked.sum())} with a linked track, {int((linked & selected).sum())} of them passing the track cuts")
    print(f"Track cuts: >= {SITRACK_MIN_HITS} hits, |d0| and |z0| <= {SITRACK_MAX_D0:g} mm, pT >= {SITRACK_MIN_PT:g} GeV, "
          f"<= {SITRACK_MAX_HOLES} holes, >= {TIER_MIN_VXD} VXD layers, >= {TIER_MIN_IT} IT and >= {TIER_MIN_OT} OT hits")

    bins = np.linspace(-ACC_ETA_MAX, ACC_ETA_MAX, 17)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_of = {}
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for label, eta, found, ls in series:
        c = color_of.setdefault(label.rsplit(" (", 1)[0], colors[len(color_of) % len(colors)])
        tot, _ = np.histogram(eta, bins=bins)
        hit, _ = np.histogram(eta, bins=bins, weights=found.astype(float))
        with np.errstate(divide="ignore", invalid="ignore"):
            e = np.where(tot > 0, hit / tot, np.nan)
            err = np.where(tot > 0, np.sqrt(e * (1 - e) / np.maximum(tot, 1)), np.nan)
        ax.errorbar(0.5 * (bins[1:] + bins[:-1]), e, yerr=err, fmt="o", markersize=4, capsize=2, color=c,
                    linestyle=ls, label=f"{label}: {100 * found.mean():.1f}% of {len(eta)}")
        print(f"  {label:70s} {100 * found.mean():5.1f}% of {len(eta)} staus in acceptance")
    ax.set_xlabel(r"Truth stau $\eta$")
    ax.set_ylabel("Tracking efficiency")
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=7, loc="lower center")
    ax.set_title(f"Staus in acceptance (|eta| < {ACC_ETA_MAX:g}, decay outside r {TRACKER_R:g} / |z| {TRACKER_Z:g} mm)\n"
                 f"track: >= {SITRACK_MIN_HITS} hits, |d0|, |z0| <= {SITRACK_MAX_D0:g} mm, pT >= {SITRACK_MIN_PT:g} GeV, "
                 f"<= {SITRACK_MAX_HOLES} holes, >= {TIER_MIN_VXD} VXD layers + {TIER_MIN_IT} IT + {TIER_MIN_OT} OT hits",
                 fontsize=9)
    fig.tight_layout()
    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, "stau_eff_vs_eta_vs_mira.pdf")
    fig.savefig(out)
    print(f"plot -> {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract", help="read Mira's .slcio files (needs pyLCIO)")
    e.add_argument("--input-dir", required=True)
    e.add_argument("--output", required=True)
    e.add_argument("--workers", type=int, default=4)
    e.add_argument("--max-files", type=int, default=0, help="0 = all")
    q = sub.add_parser("plot", help="overlay our stau_summary.pkl with extracted Mira results")
    q.add_argument("--ours", action="append", default=[], help="'label=<plot dir>/stau_summary.pkl', repeatable")
    q.add_argument("--mira", action="append", default=[], help="'label=<extract output>', repeatable")
    q.add_argument("--outdir", default="stau_vs_mira")
    a = p.parse_args()
    if a.cmd == "extract":
        extract(a)
    elif not a.ours and not a.mira:
        sys.exit("plot: give at least one --ours or --mira input")
    else:
        plot(a)


if __name__ == "__main__":
    main()
