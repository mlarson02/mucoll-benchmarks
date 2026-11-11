"""Standalone z = 0 module-seam r-z scatter, split out of edm4hepTrackAnalyzer.py.

Reads only MergedTrackerHits and writes, per input file, into <file>_plots/:
  zseam_rz_zoom.pdf       all merged hits within +-ZSEAM_ZOOM_HALF of z = 0
  zseam_rz_zoom_VXD.pdf   same window, VXD barrel only (r zoomed to the VXD)
  zseam_rz_zoom_IT.pdf    same window, IT barrel only
  zseam_rz_zoom_OT.pdf    same window, OT barrel only
The per-barrel pages are coloured by layer instead of by subsystem.

Usage: python zseamRZPlotter.py [file.edm4hep.root ...]
"""
from podio.reading import get_reader
import edm4hep  # noqa: F401  (loads the EDM dictionaries)
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

system_to_relname = {
    1: "VXDBarrel", 2: "VXDEndcap",
    3: "ITBarrel",  4: "ITEndcap",
    5: "OTBarrel",  6: "OTEndcap",
}

# (system id, short tag used in the output filename, plot title)
BARRELS = [
    (1, "VXD", "VXD Barrel (odd nmodules -> module centred on z=0)"),
    (3, "IT",  "IT Barrel (even nz -> module EDGE on z=0)"),
    (5, "OT",  "OT Barrel (even nz -> module EDGE on z=0)"),
]

ZSEAM_ZOOM_HALF = 1.0        # x-range of the zoomed scatter [mm]
ZSEAM_MAX_HITS  = 20000000   # safety cap so a BIB sample cannot blow up memory

file_names = ["/scratch/larsonma/tutorial2024/ACTSGaudi/TemplateWorkspace/mucoll-benchmarks_OLD/k4MuCPlayground/10_2500GeV_muon_10Ev_theta90_100pcBIB_reco_MAIACONFIG_VXD_ONLY.edm4hep.root"]
if len(sys.argv) > 1:
    file_names = sys.argv[1:]


def collect_near_z0(infile):
    """Return (z, r, system, layer) arrays for every merged hit with |z| <= ZSEAM_ZOOM_HALF."""
    zs, rs, cids = [], [], []
    n_events = 0
    for event in get_reader(infile).get("events"):
        if n_events % 10 == 0:
            print(f"  Processing event {n_events}.")
        n_events += 1
        hits = event.get("MergedTrackerHits")
        if hits is None or len(zs) >= ZSEAM_MAX_HITS:
            continue
        # Single pass: only hits inside the window pay for the cellID/radius.
        for h in hits:
            p = h.getPosition()
            if abs(p[2]) <= ZSEAM_ZOOM_HALF:
                zs.append(p[2])
                rs.append(np.hypot(p[0], p[1]))
                cids.append(int(h.getCellID()))
    cids = np.array(cids, dtype=np.int64)
    return (np.array(zs), np.array(rs), cids & 0x1F, (cids >> 7) & 0x1FF, n_events)


def finish_axes(ax, title):
    ax.axvline(0.0, color='red', linestyle='--', linewidth=1.2, alpha=0.8)
    ax.set_xlim(-ZSEAM_ZOOM_HALF, ZSEAM_ZOOM_HALF)
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("r [mm]")
    ax.set_title(title)
    ax.grid(True, linestyle=':', alpha=0.5)


def no_hits(ax):
    ax.text(0.5, 0.5, f"no hits with |z| < {ZSEAM_ZOOM_HALF:g} mm",
            ha='center', va='center', transform=ax.transAxes, fontsize=12)


for infile in file_names:
    print("Processing file:", infile)
    plotdir = os.path.basename(infile).replace(".edm4hep.root", "_plots")
    os.makedirs(plotdir, exist_ok=True)

    _zz, _zr, _zs_sys, _zs_lay, n_events = collect_near_z0(infile)
    print(f"[ZSEAM] {len(_zz)} hits within |z| < {ZSEAM_ZOOM_HALF:g} mm over {n_events} events")

    # All subsystems together, coloured by subsystem.
    fig, ax = plt.subplots(figsize=(9, 6))
    if len(_zz):
        for _sid in sorted(np.unique(_zs_sys)):
            _m = _zs_sys == _sid
            ax.scatter(_zz[_m], _zr[_m], s=6, alpha=0.6,
                       label=f"sys={_sid} ({system_to_relname.get(_sid, '?')}), {_m.sum()} hits")
        ax.legend(fontsize=7, markerscale=3, loc='best')
    else:
        no_hits(ax)
    finish_axes(ax, f"Merged tracker hits within $\\pm${ZSEAM_ZOOM_HALF:g} mm of z = 0  "
                    f"({len(_zz)} hits, {n_events} events)")
    fig.tight_layout()
    plt.savefig(plotdir + "/zseam_rz_zoom.pdf")
    plt.close(fig)

    # One page per barrel, r-range zoomed to that barrel, coloured by layer.
    for _sid, _tag, _title in BARRELS:
        _m_sys = _zs_sys == _sid
        print(f"[ZSEAM]   sys={_sid} ({system_to_relname[_sid]:10s}): {int(_m_sys.sum()):8d} hits")
        fig, ax = plt.subplots(figsize=(9, 6))
        if _m_sys.any():
            for _lay in sorted(np.unique(_zs_lay[_m_sys])):
                _m = _m_sys & (_zs_lay == _lay)
                ax.scatter(_zz[_m], _zr[_m], s=6, alpha=0.6,
                           label=f"layer {_lay}: <r> = {_zr[_m].mean():.1f} mm, {_m.sum()} hits")
            ax.legend(fontsize=7, markerscale=3, loc='best')
            _rlo, _rhi = _zr[_m_sys].min(), _zr[_m_sys].max()
            _pad = max(0.05 * (_rhi - _rlo), 2.0)
            ax.set_ylim(_rlo - _pad, _rhi + _pad)
        else:
            no_hits(ax)
        finish_axes(ax, f"{_title}\n{int(_m_sys.sum())} hits within $\\pm${ZSEAM_ZOOM_HALF:g} mm "
                        f"of z = 0, {n_events} events")
        fig.tight_layout()
        plt.savefig(f"{plotdir}/zseam_rz_zoom_{_tag}.pdf")
        plt.close(fig)

    print(f"[ZSEAM] wrote plots to {plotdir}/")
