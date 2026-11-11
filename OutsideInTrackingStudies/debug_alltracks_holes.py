"""Tabulate AllTracks nHits x nHoles and hit-layer patterns for one event.

Usage: python debug_alltracks_holes.py <reco.edm4hep.root> [event] [collection]
"""
import sys
from collections import Counter
from podio.root_io import Reader

SYS = {1: "VXDB", 2: "VXDE", 3: "ITB", 4: "ITE", 5: "OTB", 6: "OTE"}

fname = sys.argv[1]
ievt = int(sys.argv[2]) if len(sys.argv) > 2 else 0
coll = sys.argv[3] if len(sys.argv) > 3 else "AllTracks"

evt = Reader(fname).get("events")[ievt]
tracks = evt.get(coll)

hits_holes = Counter()
patterns = Counter()
for t in tracks:
    ids = [h.getCellID() for h in t.getTrackerHits()]
    hits_holes[(len(ids), t.getNholes())] += 1
    patterns[" ".join(f"{SYS.get(c & 0x1F, '?')}{(c >> 7) & 0x3F}" for c in ids)] += 1

n = len(tracks)
print(f"{fname} event {ievt}: {n} {coll}")
print("\nnHits nHoles   tracks  fraction")
for (nh, nho), k in sorted(hits_holes.items()):
    print(f"{nh:5d} {nho:6d} {k:8d}  {k / n:8.3%}")
print("\nTop 30 hit patterns (state order as written):")
for p, k in patterns.most_common(30):
    print(f"{k:8d}  {k / n:8.3%}  {p}")
