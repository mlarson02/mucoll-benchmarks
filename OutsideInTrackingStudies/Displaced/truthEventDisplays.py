#!/usr/bin/env python3
"""Per-event x-y and r-z displays of the truth muon: helix from its production
vertex, its sim hits, the digitised hits linked to them (<Sys>HitsRelations),
and the hits on AllTracks / SiTracks. One PDF per event.

Duplicate displays: the muon's AllTracks track (its AllTracksRelations link)
and every other AllTracks track sharing >= MIN_SHARED hits with it, each with
its hits and the helix of its AtIP state. Only for muons with >= MIN_DUPS such
duplicates.

  python truthEventDisplays.py <reco.edm4hep.root> [nEvents=50] [nDuplicateDisplays=10]

Writes <reco basename>_truth_displays/event_<i>.pdf for the first nEvents
events and <reco basename>_duplicate_displays/event_<i>.pdf for the first
nDuplicateDisplays muons that qualify.
"""
import os
import sys
from math import asinh, atan2, ceil, cos, hypot, pi, sin

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from podio.root_io import Reader

BFIELD = 5.0  # T
AT_IP = 1     # edm4hep::TrackState::AtIP
SIM_COLLS = ["VertexBarrelCollection", "VertexEndcapCollection",
             "InnerTrackerBarrelCollection", "InnerTrackerEndcapCollection",
             "OuterTrackerBarrelCollection", "OuterTrackerEndcapCollection"]
DIGI_SYSTEMS = ["VXDBarrel", "VXDEndcap", "ITBarrel", "ITEndcap", "OTBarrel", "OTEndcap"]

# A duplicate shares >= MIN_SHARED hits with the muon's AllTracksRelations track
# (TrackTruthAlg links only the best track per particle, so the links alone
# never give two). Duplicate displays need >= MIN_DUPS duplicates.
MIN_SHARED = 3
MIN_DUPS = 2
TRACK_COLORS = ["tab:red", "tab:green", "tab:purple", "tab:brown", "tab:pink",
                "tab:olive", "tab:cyan", "tab:blue", "tab:orange", "tab:gray"]

# Layer outlines [mm], as in edm4hepTrackAnalyzer_displaced.py
vb_r, vb_l = [30, 51, 74, 102], 130
ve_z = [80, 120, 200, 280]
ib_r, ib_l = [127, 340, 554], [963.2, 963.2, 1384.6]
ie_z = [524, 808, 1093, 1377, 1661, 1946, 2190]
ob_r, ob_l = [819, 1153, 1486], 2528.4
oe_z = [1310, 1617, 1883, 2190]
R_MAX, Z_MAX = 1600., 2400.


def helix_xyz(x0, y0, z0, pt, q, phi0, cot, s):
    """Points at transverse arc lengths s along the helix through (x0, y0, z0)
    with momentum direction phi0 there. Bz > 0."""
    R = pt / (0.299792458 * BFIELD) * 1000.0
    phi = phi0 - q * s / R
    return (x0 - (R / q) * (np.sin(phi) - sin(phi0)),
            y0 + (R / q) * (np.cos(phi) - cos(phi0)),
            z0 + s * cot)


def arc_length(x0, y0, pt, q, phi0, x, y):
    """Transverse arc length along that helix from (x0, y0) to the point at the
    azimuth of (x, y) around the circle centre, within half a turn either way."""
    R = pt / (0.299792458 * BFIELD) * 1000.0
    xc, yc = x0 + (R / q) * sin(phi0), y0 - (R / q) * cos(phi0)
    phi = np.arctan2(-q * (x - xc), q * (y - yc))
    return -q * R * ((phi - phi0 + pi) % (2 * pi) - pi)


def perigee(x0, y0, z0, pt, q, phi0, cot):
    """(d0, z0) of that helix w.r.t. the z axis, ACTS signs: PCA = d0 (-sin phi, cos phi)."""
    R = pt / (0.299792458 * BFIELD) * 1000.0
    xc, yc = x0 + (R / q) * sin(phi0), y0 - (R / q) * cos(phi0)
    f = 1 - R / hypot(xc, yc)
    xp, yp = f * xc, f * yc
    phi = atan2(-q * (xp - xc), q * (yp - yc))
    return -xp * sin(phi) + yp * cos(phi), z0 + arc_length(x0, y0, pt, q, phi0, xp, yp) * cot


def helix(mcp, ds=2.0):
    """Truth helix (x, y, z) from the production vertex until it leaves
    r < R_MAX, |z| < Z_MAX or completes half a turn. Bz > 0."""
    v, p = mcp.getVertex(), mcp.getMomentum()
    pt = hypot(p.x, p.y)
    R = pt / (0.299792458 * BFIELD) * 1000.0
    s = np.arange(0, min(pi * R, 2e4), ds)
    x, y, z = helix_xyz(v.x, v.y, v.z, pt, mcp.getCharge(), atan2(p.y, p.x), p.z / pt, s)
    out = (np.hypot(x, y) > R_MAX) | (np.abs(z) > Z_MAX)
    n = int(np.argmax(out)) if out.any() else len(s)
    return x[:n], y[:n], z[:n]


def fitted_helix(ts, hits, ds=2.0):
    """Helix of an AtIP state from its PCA, over the arc covering the PCA and
    the hits; NaN outside r < R_MAX, |z| < Z_MAX."""
    q = 1.0 if ts.omega > 0 else -1.0
    pt = 0.299792458 * BFIELD / abs(ts.omega * 1000.)
    ref = ts.referencePoint
    x0, y0, z0 = ref.x - ts.D0 * sin(ts.phi), ref.y + ts.D0 * cos(ts.phi), ref.z + ts.Z0
    hx, hy, _ = xyz(hits)
    s_hits = arc_length(x0, y0, pt, q, ts.phi, hx, hy)
    s = np.arange(min(0., s_hits.min()) - 20, max(0., s_hits.max()) + 20, ds)
    x, y, z = helix_xyz(x0, y0, z0, pt, q, ts.phi, ts.tanLambda, s)
    out = (np.hypot(x, y) > R_MAX) | (np.abs(z) > Z_MAX)
    x[out] = y[out] = z[out] = np.nan
    return x, y, z


def draw_detector(ax_xy, ax_rz):
    for radii, col in ((vb_r, "#348ec2"), (ib_r, "#34b18f"), (ob_r, "#dd7e33")):
        for r in radii:
            ax_xy.add_patch(patches.Circle((0, 0), r, fill=False, color=col, lw=0.8))
    for r in vb_r:
        ax_rz.plot([-vb_l / 2, vb_l / 2], [r, r], color="#348ec2", lw=0.8)
    for z in ve_z:
        ax_rz.plot([z, z], [0, 115], color="#348ec2", lw=0.8, alpha=0.5)
        ax_rz.plot([-z, -z], [0, 115], color="#348ec2", lw=0.8, alpha=0.5)
    for r, l in zip(ib_r, ib_l):
        ax_rz.plot([-l / 2, l / 2], [r, r], color="#34b18f", lw=0.8)
    for z in ie_z:
        ax_rz.plot([z, z], [100, 570], color="#34b18f", lw=0.8, alpha=0.5)
        ax_rz.plot([-z, -z], [100, 570], color="#34b18f", lw=0.8, alpha=0.5)
    for r in ob_r:
        ax_rz.plot([-ob_l / 2, ob_l / 2], [r, r], color="#dd7e33", lw=0.8)
    for z in oe_z:
        ax_rz.plot([z, z], [600, 1430], color="#dd7e33", lw=0.8, alpha=0.5)
        ax_rz.plot([-z, -z], [600, 1430], color="#dd7e33", lw=0.8, alpha=0.5)


def draw_truth(ax_xy, ax_rz, mu):
    """Truth helix and production vertex."""
    v = mu.getVertex()
    hx, hy, hz = helix(mu)
    ax_xy.plot(hx, hy, color="k", lw=1, label="truth helix")
    ax_rz.plot(hz, np.hypot(hx, hy), color="k", lw=1)
    for ax, (a, b) in ((ax_xy, (v.x, v.y)), (ax_rz, (v.z, hypot(v.x, v.y)))):
        ax.scatter([a], [b], marker="*", s=150, color="tab:purple", zorder=7)
    ax_xy.scatter([], [], marker="*", s=150, color="tab:purple", label="production vertex")


def truth_title(i, mu):
    v, p = mu.getVertex(), mu.getMomentum()
    pt, rxy = hypot(p.x, p.y), hypot(v.x, v.y)
    d0, z0 = perigee(v.x, v.y, v.z, pt, mu.getCharge(), atan2(p.y, p.x), p.z / pt)
    return (f"Event {i}: $p_T$ = {pt:.1f} GeV, η = {asinh(p.z / pt):+.2f}, "
            f"vertex $r_{{xy}}$ = {rxy:.0f} mm, z = {v.z:.0f} mm, t = {mu.getTime():.2f} ns\n"
            f"truth $d_0$ = {d0:.1f} mm, $z_0$ = {z0:.1f} mm")


def label_axes(ax_xy, ax_rz):
    ax_xy.set_xlabel("x [mm]"); ax_xy.set_ylabel("y [mm]")
    ax_rz.set_xlabel("z [mm]"); ax_rz.set_ylabel("r [mm]")
    for ax in (ax_xy, ax_rz):
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)


def zoom(ax_xy, ax_rz, x, y, z):
    """Axis limits around the points, with a margin."""
    r = np.hypot(x, y)
    pad = lambda a: 0.08 * np.ptp(a) + 20
    pxy = max(pad(x), pad(y))
    ax_xy.set_xlim(x.min() - pxy, x.max() + pxy); ax_xy.set_ylim(y.min() - pxy, y.max() + pxy)
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_rz.set_xlim(z.min() - pad(z), z.max() + pad(z))
    ax_rz.set_ylim(max(0., r.min() - pad(r)), r.max() + pad(r))


def xyz(objs):
    pts = np.array([(o.getPosition().x, o.getPosition().y, o.getPosition().z) for o in objs]).reshape(-1, 3)
    return pts[:, 0], pts[:, 1], pts[:, 2]


def scatter(ax_xy, ax_rz, objs, **style):
    if not objs:
        return
    x, y, z = xyz(objs)
    ax_xy.scatter(x, y, **style)
    style.pop("label", None)
    ax_rz.scatter(z, np.hypot(x, y), **style)


def obj_key(obj):
    """(collectionID, index) as one int, as in edm4hepTrackAnalyzer_displaced.py."""
    oid = obj.getObjectID()
    return (int(oid.collectionID) << 31) | int(oid.index)


def ip_state(track):
    return next((s for s in track.getTrackStates() if s.location == AT_IP), None)


def track_hits(frame, names, coll):
    if coll not in names:
        return [], 0
    tracks = frame.get(coll)
    return [h for t in tracks for h in t.getTrackerHits()], len(tracks)


def truth_display(i, frame, names, mu, digi, outdir):
    """The muon's sim hits, digitised hits and the hits on AllTracks / SiTracks."""
    mu_idx = mu.getObjectID().index
    sim = [h for c in SIM_COLLS if c in names for h in frame.get(c)
           if h.getParticle().getObjectID().index == mu_idx]
    all_hits, n_all = track_hits(frame, names, "AllTracks")
    si_hits, n_si = track_hits(frame, names, "SiTracks")

    fig, (ax_xy, ax_rz) = plt.subplots(1, 2, figsize=(14, 6.2))
    draw_detector(ax_xy, ax_rz)
    draw_truth(ax_xy, ax_rz, mu)
    scatter(ax_xy, ax_rz, sim, s=18, color="tab:blue", zorder=3, label=f"muon sim hits ({len(sim)})")
    scatter(ax_xy, ax_rz, digi, s=70, facecolors="none", edgecolors="tab:orange", lw=1.3, zorder=4,
            label=f"digitised muon hits ({len(digi)})")
    scatter(ax_xy, ax_rz, all_hits, s=40, marker="x", color="tab:green", zorder=5,
            label=f"AllTracks hits ({n_all} tracks)")
    scatter(ax_xy, ax_rz, si_hits, s=60, marker="+", color="tab:red", zorder=6,
            label=f"SiTracks hits ({n_si} tracks)")

    ax_xy.set_xlim(-R_MAX, R_MAX); ax_xy.set_ylim(-R_MAX, R_MAX); ax_xy.set_aspect("equal")
    ax_rz.set_xlim(-Z_MAX, Z_MAX); ax_rz.set_ylim(0, R_MAX)
    label_axes(ax_xy, ax_rz)
    ax_xy.legend(loc="upper left", fontsize=8)
    fig.suptitle(truth_title(i, mu))
    fig.tight_layout()
    fig.savefig(f"{outdir}/event_{i}.pdf")
    plt.close(fig)
    print(f"event {i}: muon sim hits {len(sim)}, digitised {len(digi)}, AllTracks {n_all}, SiTracks {n_si}")


def duplicate_display(i, frame, names, mu, digi, outdir):
    """The muon's AllTracksRelations track and its duplicates (AllTracks tracks
    sharing >= MIN_SHARED hits with it): hits as rings, shrinking per track so
    shared hits show as concentric rings, and dashed AtIP-state helices.
    Returns whether it was drawn (>= MIN_DUPS duplicates)."""
    if "AllTracks" not in names or "AllTracksRelations" not in names:
        return False
    tracks = frame.get("AllTracks")
    keys = [{obj_key(h) for h in t.getTrackerHits()} for t in tracks]
    pos = {obj_key(t): k for k, t in enumerate(tracks)}
    mu_idx = mu.getObjectID().index
    linked = [pos[obj_key(l.getFrom())] for l in frame.get("AllTracksRelations")
              if l.getTo().getObjectID().index == mu_idx]
    if not linked:
        return False
    m = linked[0]
    shared = {k: len(keys[k] & keys[m]) for k in range(len(tracks)) if k != m}
    dups = sorted((k for k, n in shared.items() if n >= MIN_SHARED), key=lambda k: (-shared[k], k))
    if len(dups) < MIN_DUPS:
        return False
    kept = {c: {obj_key(t) for t in frame.get(c)} for c in ("DedupedTracks", "SiTracks") if c in names}
    mu_keys = {obj_key(h) for h in digi}

    leg_h = 0.2 * ceil((len(dups) + 1) / 2) + 0.3
    fig, (ax_xy, ax_rz) = plt.subplots(1, 2, figsize=(14, 6.2 + leg_h))
    draw_detector(ax_xy, ax_rz)
    draw_truth(ax_xy, ax_rz, mu)
    scatter(ax_xy, ax_rz, digi, s=14, color="0.55", zorder=3, label=f"digitised muon hits ({len(digi)})")

    handles, drawn = [], list(digi)
    for n, k in enumerate([m] + dups):
        t, c = tracks[k], TRACK_COLORS[n % len(TRACK_COLORS)]
        hits = list(t.getTrackerHits())
        drawn += hits
        scatter(ax_xy, ax_rz, hits, s=max(170 - 25 * n, 30), facecolors="none", edgecolors=c,
                lw=1.2, zorder=4 + n)
        label = f"{'matched' if k == m else 'duplicate'} #{k}: {len(hits)} hits ({len(keys[k] & mu_keys)} muon"
        label += ")" if k == m else f", {shared[k]} shared)"
        label += f", {t.getNholes()} holes"
        ts = ip_state(t)
        if ts is not None and ts.omega != 0:
            hx, hy, hz = fitted_helix(ts, hits)
            ax_xy.plot(hx, hy, color=c, lw=0.8, ls="--")
            ax_rz.plot(hz, np.hypot(hx, hy), color=c, lw=0.8, ls="--")
            label += (f", $p_T$ {0.299792458 * BFIELD / abs(ts.omega * 1000.):.2f} GeV, "
                      f"$d_0$ {ts.D0:+.1f}, $z_0$ {ts.Z0:+.1f} mm")
        if t.getNdf() > 0:
            label += f", χ²/ndf {t.getChi2() / t.getNdf():.1f}"
        in_colls = [cname for cname, ks in kept.items() if obj_key(t) in ks]
        if in_colls:
            label += f"  [in {', '.join(in_colls)}]"
        handles.append(Line2D([], [], color=c, lw=0.8, ls="--", marker="o", ms=7, mfc="none", label=label))

    v = mu.getVertex()
    x, y, z = xyz(drawn)
    zoom(ax_xy, ax_rz, np.append(x, v.x), np.append(y, v.y), np.append(z, v.z))
    label_axes(ax_xy, ax_rz)
    ax_xy.legend(loc="upper left", fontsize=8)
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7.5, frameon=False)
    fig.suptitle(truth_title(i, mu) + f"\nAllTracks: {len(tracks)} tracks; matched #{m} + "
                 f"{len(dups)} duplicates sharing ≥ {MIN_SHARED} hits with it")
    fig.tight_layout(rect=(0, leg_h / (6.2 + leg_h), 1, 1))
    fig.savefig(f"{outdir}/event_{i}.pdf")
    plt.close(fig)
    print(f"event {i}: AllTracks #{m} + {len(dups)} duplicates, shared hits {[shared[k] for k in dups]}")
    return True


def main():
    infile = sys.argv[1]
    n_events = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    n_dup_max = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    base = os.path.splitext(os.path.splitext(infile)[0])[0]
    truth_dir, dup_dir = base + "_truth_displays", base + "_duplicate_displays"
    for d in (truth_dir, dup_dir):
        os.makedirs(d, exist_ok=True)

    n_dup = 0
    for i, frame in enumerate(Reader(infile).get("events")):
        if i >= n_events and n_dup >= n_dup_max:
            break
        names = set(frame.getAvailableCollections())
        mus = [m for m in frame.get("MCParticles") if m.getGeneratorStatus() == 1 and abs(m.getPDG()) == 13]
        if not mus:
            continue
        mu = mus[0]
        mu_idx = mu.getObjectID().index

        digi = []
        for s in DIGI_SYSTEMS:
            if s + "HitsRelations" in names:
                digi += [l.getFrom() for l in frame.get(s + "HitsRelations")
                         if l.getTo().getParticle().getObjectID().index == mu_idx]
        if i < n_events:
            truth_display(i, frame, names, mu, digi, truth_dir)
        if n_dup < n_dup_max and duplicate_display(i, frame, names, mu, digi, dup_dir):
            n_dup += 1
    print(f"Wrote {truth_dir}/ and {n_dup} duplicate displays to {dup_dir}/")


if __name__ == "__main__":
    main()
