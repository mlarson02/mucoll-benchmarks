# Seeding sensor selections and eta coverage (MAIA_v0)

Why the outside-in seeding options in
`configs/MAIAConfig/MAIAConfig/Tracking/CKF_tracking.py` are what they are.
`SeedingSensorsCellIDs` restricts which hits can become seed space points; an
ACTS seed needs **3 space points**, so a selection only covers the eta range
where a track from the IP crosses at least three of the selected surfaces.

## Geometry used

Layer radii, disk z and their radial extents, as laid out in MAIA_v0 (the same
numbers the r-z sketches in `edm4hepTrackAnalyzer.py` draw), cross checked
against the ACTS volume envelopes in
`install/share/k4ActsTracking/data/MAIA_v0.json`:

| subdetector | system | surfaces | extent |
|---|---|---|---|
| VXD barrel  | 1 | r = 30, 51, 74, 102 | \|z\| < 65 |
| VXD endcap  | 2 | z = 80, 120, 200, 280 | r up to 125 |
| IT barrel   | 3 | r = 127, 340, 554 | \|z\| < 482, 482, 692 |
| IT endcap   | 4 | z = 524, 808, 1093, 1377, 1661, 1946, 2190 | r ~ 100-570 |
| OT barrel   | 5 | r = 819, 1153, 1486 | \|z\| < 1264 |
| OT endcap   | 6 | z = 1310, 1617, 1883, 2190 | r ~ 600-1430 |

ACTS envelopes agree on the volumes: OT central r 600-1550 / \|z\| < 1300, OT
endcap r 570-1550 / z 1300-2210, IT endcap r 120-600 / z 750-2210.

## Surfaces crossed vs eta

Counting surfaces a straight track from the IP crosses:

| eta | OT barrel | OT disks | IT barrel | IT disks |
|---|---|---|---|---|
| 0.0 - 0.77 | 3 | 0 | 3 | 0 |
| 0.80 | 2 | 0-1 | 3 | 0 |
| 0.95 | 1 | 1-2 | 3 | 0 |
| 1.2 | 1 | 3 | 1 | 2 |
| 1.7 | 0 | 3 | 1 | 4 |
| 2.0 | 0 | 1 | 1 | 6 |
| 2.2 - 2.7 | 0 | 0 | 0 | 7 |
| > 2.75 | 0 | 0 | 0 | 0 |

## What this means for each selection

- **`["system:5"]` (OT barrel only)** covers **\|eta\| < 0.77** and nothing
  beyond it. The outermost OT layer (r = 1486, half-length 1264) is crossed
  only below that, so the third space point disappears. This is why the
  OT-seeded studies so far have used a theta 88-92 deg (eta ~ 0) sample: the
  configuration cannot be evaluated anywhere else.

- **`["system:5", "system:6"]` (OT barrel + disks)** carries it to
  **\|eta\| ~ 2.0**, where the disks stop being crossed (at z = 2190 the inner
  edge r ~ 600 corresponds to eta 1.99).

- **`["system:3", "system:4", "system:5", "system:6"]` (IT + OT, barrels and
  disks)** covers **the full tracker acceptance, \|eta\| ~ 2.7**, which is
  where the last IT disk runs out. The VXD is deliberately excluded so the
  vertex hits are what the CKF has to find rather than what it starts from.

## The barrel/disk transition near |eta| ~ 0.8

One caveat worth knowing before choosing the OT-only option. Whether the OT
alone keeps 3 space points through the barrel/disk transition depends on how
far out the OT disks actually reach, and the two sources disagree:

- r_max = 1430 (as drawn in `edm4hepTrackAnalyzer.py`): dips to **2** space
  points at eta ~ 0.80 and ~ 0.95 - i.e. a hole in the seeding efficiency.
- r_max = 1550 (ACTS volume envelope in `MAIA_v0.json`): **3** throughout, no
  hole.

The envelope usually carries margin over the real sensor extent, so the drawn
value may well be the honest one. **This has not been verified against data.**
The IT barrel covers that eta band regardless (3 layers out to \|eta\| ~ 0.9),
which is why it is included in the full-coverage selection - it removes the
question rather than answering it.

To settle it, look at `tracker_occupancy_rz.pdf` from `edm4hepTrackAnalyzer.py`
on a file that still has hits: it bins real hit positions, so the outer edge of
the OT disk band can be read straight off it.

## Caveats

- Straight-line tracks from the IP. A low-pT track curves and crosses fewer
  outer surfaces, so these are upper limits on coverage; at the
  `SeedFinding_MinPt = 1500` MeV currently configured the difference is small
  for the OT, larger for the IT.
- Assumes every crossed sensitive surface yields a hit (no inefficiency, no
  gaps between modules, nothing lost to the z = 0 module seam).
- `SeedFinding_DeltaRMax = 400` also has to be satisfied between consecutive
  space points in a seed. The OT barrel spacing (334, 333 mm) is inside it, and
  the disk-to-disk spacing at fixed eta is smaller still, so it does not bite
  for any of these selections - but it would if a selection ever mixed VXD with
  OT.
